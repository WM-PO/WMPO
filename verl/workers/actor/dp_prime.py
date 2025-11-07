# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process PRM
"""
import itertools
from typing import Iterable

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.models.registry import check_model_support_rmpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPRIME']
PRIME_LOSS = {
    'ce': core_algos.compute_ce_dpo_loss_rm
}

class DataParallelPRIME(BasePPOActor):

    def __init__(
        self,
        config,
        reward_module: nn.Module,
        reference_module: nn.Module,
        reward_optimizer: torch.optim.Optimizer = None,
        prime_loss_fn='ce',
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.reward_module = reward_module
        self.reference_module = reference_module
        self.reward_optimizer = reward_optimizer
        self.prime_loss_fn = PRIME_LOSS[prime_loss_fn]
        self.use_remove_padding = self.config.prime_model.get('use_remove_padding', False)
        print(f'PRM use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.use_dynamic_bsz}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor
        See PPO paper for details. https://arxiv.org/abs/1707.06347
        """
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'acc', 'old_log_probs']
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.mini_batch_size,
                                  epochs=1)

    def _optimizer_step(self):
        assert self.config.prime_model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.prime_model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.prime_model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        raise NotImplementedError

    def _forward_micro_batch(self, module, micro_batch, prompt_length, no_grad=False):
        response_length = micro_batch['responses'].size(-1)
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context, torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
            else:
                output = module(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        use_cache=False)
                logits = output.logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs


    def compute_implicit_reward(self, micro_batch, log_probs, ref_log_probs):
        response_length = micro_batch['responses'].shape[-1]
        max_positions = micro_batch['attention_mask'][:, -response_length:].sum(-1)

        ref_log_probs.to(log_probs.dtype)
        q = log_probs[:, -response_length:] - ref_log_probs[:, -response_length:]  # this is actually diff of q

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            step_ends = []
            if self.config.prime_granularity == 'token':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append(list(range(max_positions[i])))
            elif self.config.prime_granularity == 'whole':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append([max_positions[i] - 1])
            else:
                raise NotImplementedError

            token_level_score = torch.zeros_like(micro_batch['input_ids'][:, -response_length:]).to(torch.float32)
            # the strategy of translating q to reward function:
            for i, step_end in enumerate(step_ends):
                for j in range(len(step_end)):
                    step_range = [min(step_end[j - 1] + 1, response_length - 1) if j > 0 else 0,
                                  min(response_length - 1, step_end[j])]
                    token_level_score[i, step_range[1]] = q[i, step_range[0]:step_range[1] + 1].sum()

        return token_level_score, q

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        beta = self.config.prime_model.get('beta_train', 0.05)
        n_samples = data.meta_info['n_samples']
        prompt_length = data.batch['prompts'].shape[-1]
        acc = data.batch['acc']
        attention_mask = data.batch['attention_mask']
        eos_mask = attention_mask[:, prompt_length:]

        assert self.config.mini_batch_size % self.config.micro_batch_size == 0
        self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size

        batch = data.batch
        dataloader = batch.split(self.config.mini_batch_size)

        # dataloader = self._make_minibatch_iterator(data=data)

        metrics = {}
        token_level_scores = []
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            # batch = data.batch#.cuda()
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size if "ppo_micro_batch_size" in self.config else self.config.micro_batch_size)
                # micro_batches = data.batch.split(self.config.micro_batch_size)

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                batch_attention_mask = data['attention_mask']
                batch_eos_mask = batch_attention_mask[:, prompt_length:]
                batch_acc = data['acc']

                # log_prob = torch.cat([self._forward_micro_batch(module=self.reward_module, micro_batch=data[i:i + 1], prompt_length=prompt_length) for i in range(len(data))])
                entropy, log_prob = self._forward_micro_batch(module=self.reward_module, micro_batch=data, prompt_length=prompt_length)
                if self.reference_module is not None:
                    #ref_log_prob = torch.cat([self._forward_micro_batch(module=self.reference_module, micro_batch=data[i:i + 1], prompt_length=prompt_length, no_grad=True) for i in range(len(data))])
                    _, ref_log_prob = self._forward_micro_batch(module=self.reference_module, micro_batch=data, prompt_length=prompt_length)
                else:
                    ref_log_prob = data['old_log_probs']

                token_level_score, q = self.compute_implicit_reward(data, log_prob, ref_log_prob)
                token_level_scores.append(token_level_score)
                prime_loss = self.prime_loss_fn(q, batch_acc, eos_mask=batch_eos_mask, beta=beta)
                loss = prime_loss / self.gradient_accumulation
                loss.backward()
                
                # logits = torch.cat([result_tuple[0] for result_tuple in result_tuples], dim=0)
                # token_level_score = torch.cat([output_tuple[0] for output_tuple in output_tuples])
                # token_level_scores.append(token_level_score)
                # original_token_level_score = torch.cat([output_tuple[1] for output_tuple in output_tuples])
            
            grad_norm = self._optimizer_step()
            data = {
                'reward_model/prm_loss':prime_loss.detach().item(),
                'reward_model/grad_norm': grad_norm.detach().item(),
                }
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()
        torch.cuda.empty_cache()
        

        token_level_scores = torch.cat(token_level_scores, 0).cpu()
        try:
            dpo_acc_before = core_algos.compute_dpo_accuracy(token_level_scores, acc, eos_mask=eos_mask,
                                                         n_samples=n_samples)
        except:
            dpo_acc_before = torch.tensor(0.5)
        data = {
            'reward_model/dpo_acc_before': dpo_acc_before.detach().item(),
        }
        append_to_dict(metrics, data)

        if self.config.prime_model.update == "before":
            token_level_scores = []
            dataloader = self._make_minibatch_iterator(data=data)
            for batch_idx, data in enumerate(dataloader):
                micro_batches = data.batch.split(self.config.micro_batch_size)
                for data in micro_batches:
                    data = data.cuda()
                    batch_attention_mask = data['attention_mask']
                    batch_eos_mask = batch_attention_mask[:, prompt_length::]
                    batch_acc = data['acc']

                    log_prob = torch.cat([self._forward_micro_batch(module=self.reward_module, micro_batch=data[i:i + 1], prompt_length=prompt_length) for i in range(len(data))])
                    if self.reference_module is not None:
                        ref_log_prob = torch.cat([self._forward_micro_batch(module=self.reference_module, micro_batch=data[i:i + 1], prompt_length=prompt_length, no_grad=True) for i in range(len(data))])
                    else:
                        ref_log_prob = data['old_log_probs']

                    token_level_score, q = self.compute_implicit_reward(data, log_prob, ref_log_prob, prompt_length)
                    token_level_scores.append(token_level_score)

            token_level_scores = torch.cat(token_level_scores, 0).cpu()
            dpo_acc_after = core_algos.compute_dpo_accuracy(token_level_scores, acc, eos_mask=eos_mask,
                                                            n_samples=n_samples)
            data = {
                'reward_model/dpo_acc_after': dpo_acc_after.detach().item(),
            }
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()

        if self.config.prime_norm == 'batch_norm':  # this method will still consider the relative value of rewards. The key is to control the absolute value of RETURN from being too high. so the normalization is done by controlling the maximum of reverse cumulative sum
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]),dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores/(reverse_cumsum.abs().max()+1e-6)
        else:
            # no normalization, the reward will be normalized by beta_train
            token_level_scores = token_level_scores * beta
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return token_level_scores, metrics
