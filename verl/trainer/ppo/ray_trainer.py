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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Callable, Type, Tuple, Union
import uuid
from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer

from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rob_dataset import BufferedDataLoader

from tqdm import tqdm

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl', action_token_len=7, action_chunks_len=8):
    responses = data.batch['responses']
    
    traj_length = responses.size(1) * action_chunks_len  
    action_length = action_token_len  # next fix
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    #attention_mask = data.batch['attention_mask']
    finish_step = data.batch['finish_step'] * action_length
    
    steps = torch.arange(traj_length*action_length, device=data.batch['responses'].device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, config):

    responses = data.batch['responses']
    response_length = responses.size(1) *  responses.size(2)
    # attention_mask = data.batch['attention_mask']
    finish_step = data.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
    steps = torch.arange(response_length, device=data.batch['responses'].device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    token_level_rewards = data.batch['token_level_rewards'] if 'token_level_rewards' in list(data.batch.keys()) else data.batch['token_level_scores']

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'rloo':
        # prompt_ids = data.batch['prompts']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(-1)
        advantages, returns = core_algos.compute_rloo_returns(data=data,
                                                eos_mask=response_mask,n_samples=config.data.n_samples, config=config)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(1) *  responses.size(2)
        finish_step = data.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
        steps = torch.arange(response_length, device=data.batch['responses'].device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
        response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch,config):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']
    #add
    finish_step = batch.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
    steps = torch.arange(batch.batch['responses'].size(1)*batch.batch['responses'].size(2), device=advantages.device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(batch.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
    #
    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask.bool()]).detach().item(),
        'critic/advantages/min': torch.min(advantages[response_mask.bool()]).detach().item(),
        # returns
        'critic/returns/mean': masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max': torch.max(returns[response_mask.bool()]).detach().item(),
        'critic/returns/min': torch.min(returns[response_mask.bool()]).detach().item(),
        # response length
  
    }
    return metrics

class RayTrainer(object):
   
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping and config.algorithm.kl_ctrl.kl_coef > 0
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):   # next fix
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rob_dataset import StateDataset, collate_fn
        # self.train_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
        #                                     num_trials_per_task=self.config.data.num_trials_per_task,
        #                                     task_id = task_id,
        #                                     train_val ="train",
        #                                     pad_to=pad_to if task_id is not None else 0
        #                                     )
        # self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
        #                                    batch_size=int(self.config.data.train_batch_size*self.config.data.oversample_factor),
        #                                    shuffle=True,
        #                                    drop_last=True,
        #                                    collate_fn=collate_fn))
        if self.config.data.task_name == "aloha":
            from verl.utils.dataset.rob_dataset import AlohaDataset
            self.train_dataset = AlohaDataset()
            self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
                                            batch_size=int(self.config.data.train_batch_size*self.config.data.oversample_factor),
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate_fn))

            self.rollout_dataset = AlohaDataset()
            self.rollout_dataloader = DataLoader(dataset=self.rollout_dataset,
                                            batch_size=self.config.data.rollout_batch_size,
                                            collate_fn=collate_fn)
        else:
            state_path = f'./verl/utils/dataset/{self.config.data.task_name}_d0_states.pkl'
            self.train_dataset = StateDataset(state_path)
            self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
                                            batch_size=int(self.config.data.train_batch_size*self.config.data.oversample_factor),
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate_fn))

            self.val_dataset = StateDataset(state_path)
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                            batch_size=self.config.data.val_batch_size,
                                            collate_fn=collate_fn)
            
            self.rollout_dataset = StateDataset(state_path)
            self.rollout_dataloader = DataLoader(dataset=self.rollout_dataset,
                                            batch_size=self.config.data.rollout_batch_size,
                                            collate_fn=collate_fn)
            # assert len(self.train_dataloader) >= 1
            assert len(self.val_dataloader) >= 1
            print(f'Size of train dataloader: {len(self.train_dataloader)}')
            print(f'Size of val dataloader: {len(self.val_dataloader)}')
        

    def _save_rollouts(self, global_steps=0, rollout_epoch = 1, use_wm = False):
        # batch_size = self.config.data.rollout_batch_size
        reward_tensor_lst = []
    
        for epoch_idx in range(rollout_epoch):
            for test_data in self.rollout_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                test_batch.meta_info = {
                    'do_sample': True, 
                    'recompute_log_prob': False,
                    'rollout_base_dir': self.config.actor_rollout_ref.rollout_base_dir,
                    'save_to_hdfs': True,
                    'global_steps': global_steps,
                    'return_rollouts': True, 
                    'save_val': False,
                    'use_wm': use_wm
                }
                # breakpoint()
                gen_test_batch = self.actor_rollout_wg.generate_sequences(test_batch)
                # breakpoint()
                test_batch = test_batch.union(gen_test_batch)
                # mask = ~test_batch.batch['drop'].squeeze()
                # test_batch = test_batch.slice(mask)
                verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(test_batch)
                reward_tensor=torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)
                reward_tensor_lst.append(reward_tensor)
            print(f"Epoch {epoch_idx}")
        metric_dict = {}
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        metric_dict[f'rollout_reward/{self.config.data.task_name}'] = reward_tensor.mean().item()
        return metric_dict


    def _validate(self, global_steps=0):
        reward_tensor_lst = []
        data_source_lst = []
        metric_dict = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = DataProto.from_single_dict({'dumpy': torch.zeros(self.config.data.val_batch_size)})
            # test_batch = test_batch.select(
            #         batch_keys=['task_id', 'trial_id', 'drop'],
            #         non_tensor_batch_keys={"task_suite_name"},
            #         meta_info_keys={}
            #     )
            test_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'return_rollouts': False, 
                'validate': True,
                # 'save_to_hdfs': True, # TODO
                # 'return_rollouts': True, # TODO
                # 'rollout_base_dir': self.config.actor_rollout_ref.rollout_base_dir, #TODO
                "global_steps":global_steps,
            }
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_batch)
            print("test_output_gen_batch", test_output_gen_batch)
            print('validation generation end')
            test_batch = test_batch.union(test_output_gen_batch)
            # mask = ~test_batch.batch['drop'].squeeze()
            # test_batch = test_batch.slice(mask)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(test_batch)
            reward_tensor=torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)

            for k, v in reward_metrics.items():
                metric_dict['test_reward/' + k] = v
                
            for k, v in format_metrics.items():
                metric_dict['format_acc/' + k] = v
                
            for k, v in reward_format_metrics.items():
                metric_dict['acc_wformat/' + k] = v
            reward_tensor_lst.append(reward_tensor)
            #data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            data_source_lst.append( [self.config.data.task_name] * reward_tensor.shape[0])
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        metric_dict[f'test_score/all'] = reward_tensor.mean().item()
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['rloo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['grpo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['reinforce_plus_plus']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def fit(self):
        """
        The training loop of VLA-RL.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          local_dir=self.config.trainer.default_local_dir,
                          wandb_mode=self.config.trainer.wandb_mode,
                          debug = self.config.trainer.get('debug', False),
                          config=OmegaConf.to_container(self.config, resolve=True))

        global_steps = 0
        dp_size = self.actor_rollout_wg.world_size // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples

        # perform validation before training
        # breakpoint()
        if self.config.trainer.get('rollout_before_train', False):
            self._save_rollouts(global_steps = global_steps, rollout_epoch=self.config.trainer.get('sim_rollout_epoch', 1000), use_wm=True)
            return
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
            print("val_metrics", val_metrics)
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)
            if self.config.trainer.get('val_only', False):
                return
        update_wm = self.config.actor_rollout_ref.wm.get('update_wm', False)
        skip_first_rollout = False
        skip_first_wm_update = False
        skip_first_rm_update = False
        self.train_dataloader.start_new_epoch()
        for epoch in range(self.config.trainer.total_epochs):
            if True:
                valid_batch = []
                buffer_batch = []

                metrics = defaultdict(list)
                metrics['timing/gen'] = 0
                metrics['timing/verify'] = 0
                metrics['timing/acc&trunc_filter'] = 0
                metrics['timing/filter_format_error'] = 0
                metrics['timing/compute_all_entropy'] = 0

                while len(valid_batch) < batch_size * n_samples:
                    

                    # generate a batch
                    with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:
                        newbatch = self.train_dataloader.get_next_batch()
                        # newbatch = DataProto.from_single_dict({'dumpy': torch.zeros(self.config.data.train_batch_size)})
                        newbatch = DataProto.from_single_dict(newbatch)
                        gen_batch = newbatch.select(batch_keys=['states', 'state_id'], meta_info_keys={})
 
                        newbatch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                                                             dtype=object)

                        batch_lst = sum([[newbatch[i:i + 1] for _ in range(n_samples)] for i in range(len(newbatch))],
                                        [])

                        gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'n_samples': n_samples,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'use_wm': self.config.actor_rollout_ref.wm.enable,
                            # 'rollout_base_dir': self.config.actor_rollout_ref.rollout_base_dir,
                            # 'save_to_hdfs': True,
                            # 'return_rollouts': True
                        }
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
                        roll_batch = DataProto.concat(batch_lst)
                        #roll_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                        roll_batch = roll_batch.union(gen_batch_output)

                    metrics['timing/gen'] += timer.last
                    
                    # do accuracy filtering and score logging
                    with Timer(name='verify', text="{name}: {seconds:.1f} seconds") as timer:
                        scores_tensor, reward_metrics, format_metrics, reward_format_metrics = self.reward_fn.verify(roll_batch)
                        for k, v in reward_metrics.items():
                            metrics['train_verify_score/' + k].append(v)
                            
                        for k, v in format_metrics.items():
                            metrics['format_score/' + k].append(v)
                            
                        for k, v in reward_format_metrics.items():
                            metrics['train_verify_score_wo_format/' + k].append(v)    
                            
                    metrics['timing/verify'] += timer.last
                    
                    # do accuracy filtering and score logging
                    with Timer(name='acc&trunc_filter', text="{name}: {seconds:.1f} seconds") as timer:
                        if self.config.data.filter_accuracy or self.config.data.filter_truncated:
                            print(f"before filtering: {len(roll_batch)}")
                            filtered_roll_batch = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
                            print(f"after filtering: {len(filtered_roll_batch)}")
                        else:
                            filtered_roll_batch = roll_batch
                    metrics['timing/acc&trunc_filter'] += timer.last

                    
                    if self.config.data.filter_warmup:
                        raise ValueError
                        roll_batch_to_add = filtered_roll_batch if len(filtered_roll_batch) > 0 else roll_batch
                    else:
                        roll_batch_to_add = filtered_roll_batch
                    
                    if len(valid_batch) == 0:
                        valid_batch = roll_batch_to_add
                    else:
                        valid_batch = DataProto.concat([valid_batch, roll_batch_to_add])
                    print(
                        f"collected {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                if len(valid_batch) < batch_size * n_samples:
                    assert False
                    break
                elif len(valid_batch) > batch_size * n_samples:
                    valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)

                for k, v in reward_metrics.items():
                    metrics['train_verify_score/' + k] = np.mean(metrics['train_verify_score/' + k])
                    
                for k, v in format_metrics.items():
                    metrics['format_score/' + k] = np.mean(metrics['format_score/' + k])
                    
                for k, v in reward_format_metrics.items():
                    metrics['train_verify_score_wo_format/' + k] = np.mean(metrics['train_verify_score_wo_format/' + k])
                batch = valid_batch
                if self.use_reference_policy:
                    # compute reference log_prob
                    with Timer(name='ref', text="{name}: {seconds:.1f} seconds") as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                    metrics['timing/ref'] = timer.last
                
                with Timer(name='reward', text="{name}: {seconds:.1f} seconds") as timer:
                    if self.use_rm:
                        print("Not implement yet")
                        raise ValueError
                        # batch.meta_info['n_samples'] = n_samples
                        # reward_model_tensor= self.rm_wg.compute_rm_score(batch)
                        # if 'metrics' in reward_model_tensor.meta_info:
                        #     reward_model_metrics = reduce_metrics(reward_model_tensor.meta_info.pop('metrics'))
                        #     metrics.update(reward_model_metrics)
                        # batch = batch.union(reward_model_tensor)

                metrics['timing/reward_model'] = timer.last
                with Timer(name='adv', text="{name}: {seconds:.1f} seconds") as timer:
                    # directly reuse previously computed rewards; but with reward shaping
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor_dict['all']
                    for k, v in reward_metrics.items():
                        metrics['train_reward/' + k] = v
                    # decomposed rewards:
                    for k,v in reward_tensor_dict.items():
                        batch.batch[k]=v

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                         kl_ctrl=self.kl_ctrl,
                                                         kl_penalty=self.config.algorithm.kl_penalty,
                                                         action_token_len=self.config.actor_rollout_ref.model.action_token_len, 
                                                         action_chunks_len=self.config.actor_rollout_ref.model.action_chunks_len,)
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              self.config.algorithm.gamma,
                                              self.config.algorithm.lam,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              config = self.config)
                metrics['timing/adv'] = timer.last

                # critic is disabled

                # implement critic warmup
                if self.config.trainer.critic_warmup <= global_steps:
                    # update actor
                    with Timer(name='update_actor', text="{name}: {seconds:.1f} seconds") as timer:
                        batch.meta_info['is_filtered'] = True
                        batch.meta_info['train_mode'] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        entropy_output = self.actor_rollout_wg.compute_entropy(data=batch)
                    metrics['timing/update_actor'] = timer.last
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    entropy_output_metrics = reduce_metrics(entropy_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)
                    metrics.update(entropy_output_metrics)
                # validate
                if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                    with Timer(name='testing', text="{name}: {seconds:.1f} seconds") as timer:
                        val_metrics: dict = self._validate(global_steps=global_steps+1)
                        val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                    metrics['timing/testing'] = timer.last
                    metrics.update(val_metrics)
                    logger.log(data=val_metrics, step=global_steps)

                # collect metrics
                with Timer(name='logging1', text="{name}: {seconds:.1f} seconds") as timer:
                    data_metrics = compute_data_metrics(batch=batch, config = self.config)
                with Timer(name='logging2', text="{name}: {seconds:.1f} seconds") as timer:
                    metrics.update(data_metrics)
                with Timer(name='logging3', text="{name}: {seconds:.1f} seconds") as timer:
                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=global_steps)

                if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                    actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                    f'global_step_{global_steps}')
                    actor_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                        # self.config.trainer.default_hdfs_dir, 'actor')
                    self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                    if self.use_critic:
                        critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                         f'global_step_{global_steps}')
                        critic_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                            # self.config.trainer.default_hdfs_dir, 'critic')
                        self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
                    if self.use_rm:
                        prm_local_path = os.path.join(self.config.trainer.default_local_dir, 'prm',
                                                         f'global_step_{global_steps}')
                        prm_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                            # self.config.trainer.default_hdfs_dir, 'critic')
                        self.rm_wg.save_checkpoint(prm_local_path, prm_remote_path)

                global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate(global_steps=global_steps)
            pprint(f'Final validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)

    def filter_format(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        if self.config.data.filter_format:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Format distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= 1)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled

        # Combine both masks
        combined_mask = acc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered format batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        
        return filtered_batch

    def filter(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        # First do accuracy filtering if enabled
        if self.config.data.filter_accuracy:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Accuracy distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (
                        acc_tensor <= self.config.data.accuracy_upper_bound)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch['responses']
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -responses.size(1):]

            # Calculate response lengths
            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(-1, n_samples)  # (num_prompts, n_samples)

            # Get max possible length from config
            max_len = self.config.data.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            print("Truncation distribution:", 
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}")
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        return filtered_batch

    def add_to_buffer(self, batch, batch_size, n_samples):
        buffer_length = len(batch) // n_samples - batch_size
        # buffer_batch = batch.slice(range(batch_size * n_samples, (buffer_length + batch_size) * n_samples, n_samples))
        # # notice that we only add prompts to buffer, and slicing strategy should be exactly consistent to what is in ray_trainer.py
        # buffer_batch = buffer_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        # buffer_batch.slice_batch(start=0, length=self.config.data.max_prompt_length, dim=1)
        buffer_mask = torch.ones(buffer_length + batch_size, dtype=torch.bool)
        buffer_mask[batch_size:] = False
        buffer_mask = buffer_mask.repeat_interleave(n_samples)
        batch = batch.slice(buffer_mask)
        # self.train_dataloader.add_to_buffer(buffer_batch)
        return batch