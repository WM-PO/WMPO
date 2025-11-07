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
The main entry point to run the PPO algorithm
"""

import os
import logging
import warnings
import ray
import datetime
from zoneinfo import ZoneInfo
import imageio
import numpy as np
import uuid
import h5py
import shutil
import tempfile
import time
import torch
import random
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
from webdataset import ShardWriter, TarWriter
from omegaconf import DictConfig, open_dict
from transformers import AutoModelForCausalLM
from transformers import (
    VideoMAEConfig,
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
)

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, load_fsdp_grad, offload_fsdp_grad, init_fn, get_init_weight_context_manager, get_fsdp_wrap_policy_vla
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
import verl.utils.hdfs_io as hdfs_io
from verl.utils import hf_tokenizer
from ..trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from codetiming import Timer


from verl.utils.openvla_utils import update_auto_map , check_model_logic_mismatch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import json


from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import read_config
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import CheckpointIO, model_sharding, record_model_param_shape
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.lr_scheduler import LinearWarmupLR
from opensora.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema
from tensordict import TensorDict

from opensora.datasets.success_classifier_vit_v1_3 import MixedSuccessWindowDataset, SuccessWindowDataset
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
import gc

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

def save_to_hdfs_fun(output, prompts):
    # 过滤掉 drop 的样本
    # mask = ~output.batch['drop'].squeeze(1)
    # output = output.slice(mask)
    state_ids = prompts.batch['state_id'].cpu()
    attention_mask = output.batch['attention_mask']
    input_ids = output.batch['input_ids']
    pixel_values = output.batch['pixel_values']
    responses = output.batch['responses']
    video = output.batch['video']
    action = output.batch['action']
    # task_id = output.batch['task_id']
    # trial_id = output.batch['trial_id']
    complete = output.batch['complete']
    finish_step = output.batch['finish_step']

    rollout_base_dir = prompts.meta_info.get('rollout_base_dir', './debug')
    global_steps = prompts.meta_info.get('global_steps', 0)
    val = prompts.meta_info.get('save_val', False)
    rank = dist.get_rank()

    os.makedirs(rollout_base_dir, exist_ok=True)
    start = time.time()
    total = len(video)

    # 统一划分
    if val:
        val_indices = set()
        train_indices = set()
        # for i in range(total):
        #     if random.random() < 0.01:
        #         val_indices.add(i)
        #     else:
        #         train_indices.add(i)
        val_index = random.randint(0, total - 1)  # 随机选择一个索引作为验证集
        val_indices.add(val_index)
        train_indices = set(range(total)) - val_indices  # 其余作为训练集
    else:
        train_indices = set(range(total))

    def build_sample(i):
        return {
            "__key__": f"{i:09d}",
            "video.npy": video[i].numpy().astype(np.uint8),
            "action.npy": action[i].numpy().astype(np.float32),
            "input_ids.npy": input_ids[i].numpy().astype(np.int64),
            "attention_mask.npy": attention_mask[i].numpy().astype(np.int64),
            "responses.npy": responses[i].numpy().astype(np.int64),
            "pixel_values.npy": pixel_values[i].numpy().astype(np.float32),
            "state_id.npy": state_ids[i].numpy().astype(np.uint8),
            "meta.json": json.dumps({
                # "task_id": task_id[i][0].item(),
                # "trial_id": trial_id[i][0].item(),
                "complete": complete[i].item(),
                "finish_step": finish_step[i].item(),
                "unique_id": str(uuid.uuid4())
            }).encode("utf-8")
        }

    # 写验证集
    if val:
        val_dir = os.path.join(rollout_base_dir, 'val', f'global_steps={global_steps}', f'rank_{rank}')
        os.makedirs(val_dir, exist_ok=True)

        for i in val_indices:
            sample = build_sample(i)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            tar_path = os.path.join(val_dir, f"{timestamp}.tar")
            with TarWriter(tar_path) as sink:
                sink.write(sample)
            print(f"[rank {rank}] ✅ 写入 val tar: {tar_path}")

    # 写训练集（多个 sample 合并打包）
    train_dir = os.path.join(rollout_base_dir, 'train', f'global_steps={global_steps}', f'rank_{rank}')
    os.makedirs(train_dir, exist_ok=True)


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    shard_pattern = os.path.join(train_dir, f"shard_{timestamp}_%05d.tar")

    # with ShardWriter(shard_pattern, maxcount=train_tar_max_samples) as sink:
    #     for i in sorted(train_indices):
    #         sample = build_sample(i)
    #         sink.write(sample)
    with ShardWriter(shard_pattern, maxsize=4 << 30) as sink:  # 4GB = 4 * 2^30 bytes
        for i in sorted(train_indices):
            sample = build_sample(i)
            sink.write(sample)

            
    print(f"[rank {rank}] ✅ 写入 train tar: {train_dir}")
    tar_files = [f for f in os.listdir(train_dir) if f.endswith(".tar")]
    for f in sorted(tar_files):
        print(f" - {f}")
    if val:
        print(f"[rank {rank}] ✅ 保存完成，train: {len(train_indices)}，val: {len(val_indices)}，耗时 {time.time() - start:.2f} 秒")
    else:
        print(f"[rank {rank}] ✅ 保存完成，train: {len(train_indices)}，耗时 {time.time() - start:.2f} 秒")




def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class RobWMActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._is_lora = self.config.model.get('lora_rank', 0) > 0
        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        #add oft
         
        if self.config.model.vla == "openvla-oft":
            from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
            
        elif self.config.model.vla == "openvla":
            from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
        
        #add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, local_files_only=True, model = self.config.model.vla)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, local_files_only=True)
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # if self.rank == 0:
        #     print(f'Model config after override: {actor_model_config}')

        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.config.model.vla == "openvla-oft":
                start_time = time.time()
                actor_module = AutoModelForVision2Seq.from_pretrained(
                                                        pretrained_model_name_or_path=local_path,
                                                        torch_dtype=torch_dtype,
                                                        #attn_implementation="flash_attention_2",
                                                        config=actor_model_config,              
                                                        trust_remote_code=True,
                                                        local_files_only=True
                                                        # low_cpu_mem_usage=True,         # 💡 激活 mmap + 懒加载（推荐）   
                                                    )
                print("init actor model load time: ", time.time() - start_time) # 41s
                print("load model successfully")
                #oft add
                actor_module.vision_backbone.set_num_images_in_input(self.config.actor.num_images_in_input)
                
                dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")
                if os.path.isfile(dataset_statistics_path):
                    with open(dataset_statistics_path, "r") as f:
                        norm_stats = json.load(f)
                    actor_module.norm_stats = norm_stats
                else:
                    print(
                        "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                        "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                        "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                    )
            elif self.config.model.vla == "openvla":
                actor_module = AutoModelForVision2Seq.from_pretrained(
                                                    pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation="flash_attention_2",
                                                    config=actor_model_config,              
                                                    trust_remote_code=True,    
                                                )
           
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
            # lora add
            if self._is_lora:
                print("Applying LoRA to actor module")
                
                lora_config = {
                    #'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    "lora_dropout": 0 ,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'init_lora_weights': "gaussian"
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))  
                actor_module.print_trainable_parameters()
            # lora end
                
                
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None
        
        #oft add
        auto_wrap_policy = get_fsdp_wrap_policy_vla(module=actor_module, config=fsdp_config.get('wrap_policy', None), is_lora=self.config.model.get('lora_rank', 0) > 0)
        #oft add end
        

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        
        # TODO: add transformer policy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module_fsdp = FSDP(
                actor_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import RobWMHFRollout
            from verl.workers.hybrid_engine import BaseShardingManager
            rollout = RobWMHFRollout(module=self.actor_module_fsdp, world_model_mapping = self.world_model_mapping, config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            raise ValueError
            # from verl.workers.rollout.vllm_rollout import vLLMRollout
            # from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            # log_gpu_memory_usage('Before building vllm rollout', logger=None)
            # rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
            #                       config=self.config.rollout,
            #                       tokenizer=self.tokenizer,
            #                       model_hf_config=self.actor_model_config)
            # log_gpu_memory_usage('After building vllm rollout', logger=None)
            # if torch.distributed.get_world_size() == 1:
            #     self.config.rollout.load_format = 'dummy_hf'
            # sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
            #                                            inference_engine=rollout.inference_engine,
            #                                            model_config=self.actor_model_config,
            #                                            full_params='hf' in self.config.rollout.load_format)
            # log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    def _build_world_model(self):
        update_wm = self.config.wm.get('update_wm', False)
        if not update_wm:
            device = torch.cuda.current_device()
            dtype = torch.bfloat16
            cfg = read_config(self.config.wm.inference_config_path)
            vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
            image_size = cfg.get("image_size") or get_image_size(cfg.resolution, cfg.aspect_ratio)
            num_frames = get_num_frames(cfg.num_frames)
            latent_size = vae.get_latent_size((num_frames, *image_size))
            model = build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae.out_channels,
            ).to(device, dtype).eval()
            # print(f"init world model load from hdfs {time.time() - start_time:.2f} seconds")
            print(f"World model config loaded from: {self.config.wm.inference_config_path}")
            print(f"Checkpoint loaded from: {cfg.model.from_pretrained}")
            scheduler = build_module(cfg.scheduler, SCHEDULERS)
            model_args = prepare_multi_resolution_info(cfg.get("multi_resolution"), 1, image_size, num_frames, cfg.fps, torch.cuda.current_device(), torch.bfloat16)

            feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=self.config.wm.rm_img_size)
            rw_cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=8, num_labels=2)
            rm_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=rw_cfg).to(device)
            state_dict = torch.load(self.config.wm.reward_model_path, map_location="cpu")
            rm_model.load_state_dict(state_dict, strict=True)

            world_model_mapping = {
                "model": model,
                "scheduler": scheduler,
                "vae": vae,
                "model_args": model_args,
                "rm_model": rm_model,
                "feature_extractor": feature_extractor,
                "rm_threshold": self.config.wm.rm_threshold
            }
            return world_model_mapping
        cfg = read_config(self.config.wm.train_config_path)
        record_time = cfg.get("record_time", False)
        record_time = True
        # == device and dtype ==
        assert torch.cuda.is_available(), "Training currently requires at least one GPU."
        cfg_dtype = cfg.get("dtype", "bf16")
        assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
        dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
        checkpoint_io = CheckpointIO()
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        set_seed(cfg.get("seed", 1024))
        PinMemoryCache.force_dtype = dtype
        pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", [])
        PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels
        coordinator = DistCoordinator()
        device = get_current_device()

        # == init exp_dir ==
        # exp_name, exp_dir = define_experiment_workspace(cfg)
        # coordinator.block_all()
        # if coordinator.is_master():
        #     os.makedirs(exp_dir, exist_ok=True)
        #     save_training_config(cfg.to_dict(), exp_dir)
        coordinator.block_all()

        # == init logger, tensorboard & wandb ==
        # wm_logger = create_logger("./") # 
        # wm_logger.info("Experiment directory created at %s", exp_dir)
        # wm_logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
        # if coordinator.is_master():
        #     tb_writer = create_tensorboard_writer(exp_dir)
        #     if cfg.get("wandb", False):
        #         wandb.init(project="Open-Sora", name=exp_name, entity="yourname", config=cfg.to_dict(), dir="./outputs/wandb")

        # == init ColossalAI booster ==
        plugin = create_colossalai_plugin(
            plugin=cfg.get("plugin", "zero2"),
            dtype=cfg_dtype,
            grad_clip=cfg.get("grad_clip", 0),
            sp_size=cfg.get("sp_size", 1),
            reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
        )
        booster = Booster(plugin=plugin)
        torch.set_num_threads(1)

        # == build vae ==
        vae = build_module(cfg.get("vae", None), MODELS)
        image_size = cfg.dataset.image_size
        num_frames = cfg.dataset.Ta + cfg.dataset.To
        if vae is not None:
            vae = vae.to(device, dtype).eval()
        if vae is not None:
            input_size = (num_frames, *image_size)
            latent_size = vae.get_latent_size(input_size)
            vae_out_channels = vae.out_channels
        else:
            latent_size = (None, None, None)
            vae_out_channels = cfg.get("vae_out_channels", 4)

        # == build diffusion model ==
        start_time = time.time()
        model = (
            build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae_out_channels,
                # caption_channels=text_encoder_output_dim,
                # model_max_length=text_encoder_model_max_length,
                enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
            )
            .to(device, dtype)
            .train()
        )
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        model_numel, model_numel_trainable = get_model_numel(model)
        # wm_logger.info(
        #     "[Diffusion] Trainable model params: %s, Total model params: %s",
        #     format_numel_str(model_numel_trainable),
        #     format_numel_str(model_numel),
        # )

        # == build ema for diffusion model ==
        ema = deepcopy(model).cpu().to(torch.float32)
        requires_grad(ema, False)
        ema_shape_dict = record_model_param_shape(ema)
        ema.eval()

        # == setup loss function, build scheduler ==
        scheduler = build_module(cfg.scheduler, SCHEDULERS)
        model_args = prepare_multi_resolution_info(cfg.get("multi_resolution"), 1, image_size, num_frames, cfg.fps, device, dtype)


        # == setup optimizer ==
        optimizer = HybridAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            adamw_mode=True,
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0),
            eps=cfg.get("adam_eps", 1e-8),
        )

        warmup_steps = cfg.get("warmup_steps", None)

        if warmup_steps is None:
            lr_scheduler = None
        else:
            lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

        # == additional preparation ==
        if cfg.get("grad_checkpoint", False):
            set_grad_checkpoint(model)
        if cfg.get("mask_ratios", None) is not None:
            mask_generator = MaskGenerator(cfg.mask_ratios)

        # =======================================================
        # 4. distributed training preparation with colossalai
        # =======================================================
        # wm_logger.info("Preparing for distributed training...")
        # == boosting ==
        # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
        torch.set_default_dtype(dtype)
        model, optimizer, _, _, lr_scheduler = booster.boost(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            # dataloader=dataloader,
        )
        torch.set_default_dtype(torch.float)
        # wm_logger.info("Boosting model for distributed training")

        # == global variables ==
        cfg_epochs = cfg.get("epochs", 1000)
        start_epoch = start_step = log_step = acc_step = 0
        running_loss = 0.0
        # wm_logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)
        # == resume ==
        if cfg.get("load", None) is not None:
            # wm_logger.info("Loading checkpoint")
            ret = checkpoint_io.load(
                booster,
                cfg.load,
                model=model,
                ema=ema,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                sampler=None ,
            )
            print(f"loading checkpoint from {cfg.load}")
            # if not cfg.get("start_from_scratch", False):
            #     start_epoch, start_step = ret
            # wm_logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

        model_sharding(ema, device=device)
        # =======================================================
        # 5. training loop
        # =======================================================
        dist.barrier()
        timers = {}
        timer_keys = [
            "move_data",
            "encode",
            "mask",
            "diffusion",
            "backward",
            "update_ema",
            "reduce_loss",
            "optim",
            "ckpt",
        ]
        for key in timer_keys:
            if record_time:
                timers[key] = Timer(key, coordinator=coordinator)
            else:
                timers[key] = nullcontext()

        start_time = time.time()
        feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=self.config.wm.rm_img_size)
        rw_cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=8, num_labels=2)
        rm_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=rw_cfg).to(device)
        state_dict = torch.load(self.config.wm.reward_model_path, map_location="cpu")
        rm_model.load_state_dict(state_dict, strict=True)
        rm_model = nn.parallel.DistributedDataParallel(rm_model, device_ids=[torch.cuda.current_device()])
        rm_criterion = nn.CrossEntropyLoss()
        rm_optim = torch.optim.AdamW(rm_model.parameters(), lr=self.config.wm.rm_lr, weight_decay=1e-4)
        print(f"Reward model loaded in {time.time() - start_time:.2f} seconds")
        world_model_mapping = {
            "cfg": cfg,
            "rm_model": rm_model,
            "rm_criterion": rm_criterion,
            "rm_optim": rm_optim,
            "rm_threshold": self.config.wm.rm_threshold, 
            "feature_extractor": feature_extractor,
            "model": model,
            "ema": ema,
            "optimizer": optimizer,
            'scheduler': scheduler, 
            'model_args': model_args,
            "lr_scheduler": lr_scheduler,
            # "dataloader": dataloader,
            # "sampler": sampler,
            "vae": vae,
            "latent_size": latent_size,
            "config": cfg,
            "booster": booster,
            "device": device,
            "dtype": dtype,
            "coordinator": coordinator,
            # "logger": wm_logger,
            # "tb_writer": tb_writer if coordinator.is_master() else None,
            "mask_generator": mask_generator if cfg.get("mask_ratios", None) is not None else None,
            "checkpoint_io": checkpoint_io,
            # "exp_dir": exp_dir,
            # "exp_name": exp_name,
            "start_epoch": start_epoch,
            "start_step": start_step,
            "timers": timers,
            "record_time": record_time,
        }
        self.world_model_mapping = world_model_mapping
        return world_model_mapping

    def _build_terminal_model_dataloader(self, global_steps=0):
        rollout_base_dir = self.config.rollout_base_dir + "/train/**/*.tar"
        rollout_shards = sorted(glob(rollout_base_dir, recursive=True))
        print(f"[Terminal Dataloader] 发现 {len(rollout_shards)} 个 rollout_shards")
        expert_shards = sorted(glob(self.config.rollout_expert_dir, recursive=True))
        print(f"[Terminal Dataloader] 发现 {len(expert_shards)} 个 expert_shards")
        weights = [0.7, 0.3]
        dataset = MixedSuccessWindowDataset(
            rollout_shards, expert_shards, weights, 8, 8, self.config.wm.rm_img_size, mode="train"
        )
        dataloader = DataLoader(
            dataset,
            self.config.wm.rm_batch_size,
            num_workers=self.config.wm.rm_num_workers,
            pin_memory=True,
        )

        val_rollout_base_dir = self.config.rollout_base_dir + f"/val/global_steps={global_steps}/**/*.tar"
        val_rollout_shards = sorted(glob(val_rollout_base_dir, recursive=True))
        print(f"[Terminal Dataloader] 发现 {len(val_rollout_shards)} 个 rollout_shards")
        val_dataset = SuccessWindowDataset(
            val_rollout_shards, 8, 8, self.config.wm.rm_img_size, mode="val"
        )
        val_dataloader = DataLoader(
            val_dataset,
            self.config.wm.rm_val_batch_size,
            num_workers=self.config.wm.rm_num_workers,
            pin_memory=True,
        )
        return dataloader, val_dataloader
    
    @torch.no_grad()
    def _evaluate_terminal_model(self, dataloader):
        """各 rank 本地推断后 all_gather 收集到 rank0 计算指标（支持多阈值）"""
        coordinator = self.world_model_mapping['coordinator']
        model = self.world_model_mapping['rm_model']
        rm_threshold = self.world_model_mapping['rm_threshold']
        device = self.world_model_mapping['device']

        world_size = torch.distributed.get_world_size()
        model.eval()
        logits_local, trues_local, metas_local = [], [], []
        
        # for vids, ys, meta in tqdm(dataloader, desc="Processing batches", leave=False, disable=not coordinator.is_master()):
        for vids, ys, meta in dataloader:
            vids = vids.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            logits = model(pixel_values=vids).logits
            logits_local.extend(logits.squeeze(1).cpu().tolist())  # 假设是二分类且 logits 是 (B, 1)
            trues_local.extend(ys.cpu().tolist())
            # videos_local.extend(video)
            for i in range(len(ys)):
                metas_local.append({k: v[i] for k, v in meta.items()})
        # 使用 all_gather_object 收集不同长度 list
        logits_gather, trues_gather = [None] * world_size, [None] * world_size
        metas_gather = [None] * world_size
        dist.all_gather_object(logits_gather, logits_local)
        dist.all_gather_object(trues_gather, trues_local)
        dist.all_gather_object(metas_gather, metas_local)
        all_metas = []
        for local_meta in metas_gather:
            all_metas.extend(local_meta)
        torch.cuda.empty_cache()
        model.train()
        if coordinator.is_master():
            logits = [logit for sub in logits_gather for logit in sub]
            trues = [t for sub in trues_gather for t in sub]

            thresholds = np.linspace(0.3, 1.0, 20)
            all_metrics = {}

            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            best_f1 = 0
            best_thresh = thresholds[0]
            for thresh in thresholds:
                preds = [1 if p[1] >= thresh else 0 for p in probs]
                TP = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
                TN = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
                FP = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
                FN = sum((p == 0 and t == 1) for p, t in zip(preds, trues))
                pred_pos = sum(preds)
                pred_neg = len(preds) - pred_pos
                true_pos = sum(trues)
                true_neg = len(trues) - true_pos
                acc = accuracy_score(trues, preds)
                prec = precision_score(trues, preds, zero_division=0)
                rec = recall_score(trues, preds, zero_division=0)
                f1 = f1_score(trues, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

                metrics = OrderedDict([
                    ("acc",        acc),
                    ("precision",  prec),
                    ("recall",     rec),
                    ("f1",         f1),
                    ("TP",         TP),
                    ("TN",         TN),
                    ("FP",         FP),
                    ("FN",         FN),
                    ("pred_pos",   pred_pos),
                    ("pred_neg",   pred_neg),
                    ("true_pos",   true_pos),
                    ("true_neg",   true_neg),
                ])
                all_metrics[f"thresh_{thresh:.2f}"] = metrics
            metrics = all_metrics[f"thresh_{rm_threshold:.2f}"]
            f1 = metrics['f1']
            # preds = [1 if p[1] >= best_thresh else 0 for p in probs]

            # FP_idxs =  [i for i, (p, t) in enumerate(zip(preds, trues)) if (p == 1 and t == 0) ]
            # FN_idxs =  [i for i, (p, t) in enumerate(zip(preds, trues)) if (p == 0 and t == 1) ]
            # key = f'thresh_{best_thresh:.2f}'
            # f1 = all_metrics[key]['f1']
            # prec = all_metrics[key]['precision']
            # recall = all_metrics[key]['recall']
            # acc = all_metrics[key]['acc']
            # FP_dir = f'./debug/thre_{best_thresh:.2f}_f1_{f1:.2f}_acc_{acc:.2f}_prec_{prec:.2f}_recall_{recall:.2f}/FP'
            # FN_dir = f'./debug/thre_{best_thresh:.2f}_f1_{f1:.2f}_acc_{acc:.2f}_prec_{prec:.2f}_recall_{recall:.2f}/FN'
            # os.makedirs(FP_dir, exist_ok=True)
            # os.makedirs(FN_dir, exist_ok=True)
            # for i in FP_idxs:
            #     meta = all_metas[i]
            #     with h5py.File(meta['fpath'], 'r') as f:
            #         # 查看文件中所有的主键（相当于最外层的组/数据集名）
            #         base_name = os.path.basename(meta['fpath'])
            #         task_id = base_name.split("_")[1]
            #         trial_id = base_name.split("_")[3]
            #         start = meta['video_start'].item()
            #         end = meta['video_end'].item()
            #         video = f['video'][start:end]
            #         suffix = 'succ' if meta['complete'] else 'fail'
            #         video_path = os.path.join(FP_dir, f"task_id_{task_id}_trial_id_{trial_id}_{start}_{end}_{suffix}.mp4")
            #         imageio.mimwrite(video_path, video, fps=1)
            # for i in FN_idxs:
            #     meta = all_metas[i]
            #     with h5py.File(meta['fpath'], 'r') as f:
            #         # 查看文件中所有的主键（相当于最外层的组/数据集名）
            #         base_name = os.path.basename(meta['fpath'])
            #         task_id = base_name.split("_")[1]
            #         trial_id = base_name.split("_")[3]
            #         start = meta['video_start'].item()
            #         end = meta['video_end'].item()
            #         video = f['video'][start:end]
            #         suffix = 'succ' if meta['complete'] else 'fail'
            #         video_path = os.path.join(FN_dir, f"task_id_{task_id}_trial_id_{trial_id}_{start}_{end}_{suffix}.mp4")
            #         imageio.mimwrite(video_path, video, fps=1)

            return f1
        else:
            return None


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _update_terminal_model(self, global_steps):
        coordinator = self.world_model_mapping['coordinator']
        model = self.world_model_mapping['rm_model']
        criterion = self.world_model_mapping['rm_criterion']
        optim = self.world_model_mapping['rm_optim']
        device = self.world_model_mapping['device']

        model.train()
        dataloader, val_dataloader = self._build_terminal_model_dataloader(global_steps)

        f1 = self._evaluate_terminal_model(val_dataloader) #  0.7692
        if coordinator.is_master():
            print(f"before training f1: {f1:.4f}")

        dataloader = iter(dataloader)
        running_loss = 0

        # with tqdm(
        #     enumerate(dataloader),
        #     desc=f"Training Terminal Model.",
        #     disable=not coordinator.is_master(),
        #     total=self.config.wm.rm_training_steps_per_epoch,
        # ) as pbar:
        #     for rm_training_step, batch in pbar:
        for rm_training_step, batch in enumerate(dataloader):
            if rm_training_step >= self.config.wm.rm_training_steps_per_epoch:
                break
            if True:
                video, label, _ = batch
                vids, ys = video.to(device), label.to(device)
                logits = model(pixel_values=vids).logits
                loss = criterion(logits, ys)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                loss = loss.detach().item()
                running_loss += loss
                # pbar.set_postfix({"loss": loss, "step": rm_training_step})
                if rm_training_step != 0 and rm_training_step % 100 == 0 and coordinator.is_master():
                    running_loss = running_loss/100
                    if coordinator.is_master():
                        print(f"loss: {running_loss}, step: {rm_training_step}")
                    running_loss = 0

                if rm_training_step != 0 and rm_training_step % 500 == 0:
                    f1 = self._evaluate_terminal_model(val_dataloader)
                    if coordinator.is_master():
                        print(f"step: {rm_training_step} f1: {f1:.4f}") # 0.06103515625
        
        f1 = self._evaluate_terminal_model(val_dataloader)
        if coordinator.is_master():
            print(f"After training {rm_training_step} step, f1: {f1:.4f}") # 0.06103515625
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache() 

    def _build_world_model_dataloader(self, global_steps=0):
        cfg = self.world_model_mapping['cfg']
        cfg = deepcopy(cfg)
        cfg.dataset['shards_pattern'] = self.config.rollout_base_dir + f"/train/global_steps={global_steps}/**/*.tar"
        # old_data_dirs = [self.config.rollout_expert_dir+'/**/*.tar']
        # for i in range(global_steps-1):
        #     if os.path.exists(self.config.rollout_base_dir + f"/train/global_steps={i}"):
        #         old_data_dirs.append(self.config.rollout_base_dir + f"/train/global_steps={i}/**/*.tar")
        # cfg.dataset['shards_pattern2'] = old_data_dirs
        # cfg.dataset['weights'] = [0.5, 0.5]
        # ======================================================
        # 2. build dataset and dataloader
        # ======================================================
        # wm_logger.info("Building dataset...")
        # == build dataset ==
        # print(f"shards_pattern1: {cfg.dataset['shards_pattern1']}")
        # print(f"shards_pattern2: {cfg.dataset['shards_pattern2']}")
        train_dataset = build_module(cfg.dataset, DATASETS)
        # utils.pytorch_worker_info(group=group)
        # == build dataloader ==
        cache_pin_memory = cfg.get("cache_pin_memory", False)
        train_dataloader_args = dict(
            dataset=train_dataset,
            batch_size=cfg.get("batch_size", None),
            num_workers=cfg.get("num_workers", 4),
            seed=cfg.get("seed", 1024),
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            process_group=get_data_parallel_group(),
            prefetch_factor=cfg.get("prefetch_factor", None),
            cache_pin_memory=cache_pin_memory,
        )
        train_dataloader, _ = prepare_dataloader(
            bucket_config=cfg.get("bucket_config", None),
            num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
            **train_dataloader_args,
        )

        # cfg.val_dataset['shards_pattern'] = self.config.rollout_base_dir + f"/val/global_steps={global_steps}/**/*.tar"
        # val_dataset = build_module(cfg.val_dataset, DATASETS)
        # utils.pytorch_worker_info(group=group)
        # == build dataloader ==
        # cache_pin_memory = cfg.get("cache_pin_memory", False)
        # val_dataloader_args = dict(
        #     dataset=val_dataset,
        #     batch_size=cfg.get("val_batch_size", None),
        #     num_workers=0,
        #     seed=cfg.get("seed", 1024),
        #     shuffle=True,
        #     drop_last=True,
        #     pin_memory=True,
        #     process_group=get_data_parallel_group(),
        #     prefetch_factor=cfg.get("prefetch_factor", None),
        #     cache_pin_memory=cache_pin_memory,
        # )
        # val_dataloader, _ = prepare_dataloader(
        #     bucket_config=cfg.get("bucket_config", None),
        #     num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        #     val = True,
        #     **val_dataloader_args,
        # )
        
        # return train_dataloader, val_dataloader
        return train_dataloader, None

    @torch.no_grad()
    def _evaluate_world_model(self, val_dataloader):
        device = self.world_model_mapping['device']
        dtype = self.world_model_mapping['dtype']
        mask_generator = self.world_model_mapping['mask_generator']
        model_args = self.world_model_mapping['model_args']
        scheduler = self.world_model_mapping['scheduler']
        vae = self.world_model_mapping['vae']
        ema = self.world_model_mapping['ema']
        model = self.world_model_mapping['model']
        cfg = self.world_model_mapping['cfg']
        coordinator = self.world_model_mapping['coordinator']
        latent_size = self.world_model_mapping['latent_size']
        Ta = cfg.dataset.Ta
        To = cfg.dataset.To

        model.eval()

        latent_l2_loss_sum = torch.zeros(1, device=device, dtype=dtype)
        num_batches = torch.zeros(1, device=device, dtype=dtype)

        dataloader_iter = iter(val_dataloader)
        # with tqdm(
        #     enumerate(dataloader_iter),
        #     desc=f"Evaluate World Model",
        #     disable=not coordinator.is_master()
        # ) as pbar:
        for step, batch in enumerate(dataloader_iter):
            pinned_video = batch.pop("video")
            
            x = pinned_video.to(device, dtype, non_blocking=True)  # [B, C, T, H, W]
            y = batch.pop("action").to(device, dtype, non_blocking=True)  # [B, T, action_dim]

            # rank = dist.get_rank() if dist.is_initialized() else 0
            # os.makedirs("./debug/val", exist_ok=True)
            # log_file = open(f"./debug/val/debug_rank_{rank}.log", "a")
            # for i in range(x.shape[0]):
            #     info = {
            #         "unique_id":     batch["meta"]["unique_id"][i],
            #         "episode_name":  batch["meta"]["episode_name"][i],
            #         "video_start":   int(batch["meta"]["video_start"][i]),
            #         "video_end":     int(batch["meta"]["video_end"][i]),
            #         "action_start":  int(batch["meta"]["action_start"][i]),
            #         "action_end":    int(batch["meta"]["action_end"][i]),
            #     }
            #     log_file.write(f"[RANK {rank}] : {info}\n")
            # continue
            with torch.no_grad():
                x = vae.encode(x)  # [B, C, T, H/P, W/P]

            current_batch_size = x.shape[0]

            image_history_tensor = x[:, :, :To]

            z = torch.randn(current_batch_size, vae.out_channels, Ta, *latent_size[1:], device=device, dtype=dtype)
            
            # z_combined 的形状: (B, C, T_history + chunk, H, W)
            z_combined = torch.concat([image_history_tensor, z], dim=2)
            
            mask = mask_generator.get_masks(x)
            with torch.no_grad():
                samples = scheduler.sample(model, z=z_combined, y=y, device=device, additional_args=model_args, progress=False, mask=mask)
            
            # pred_latents 的形状: (B, C_latent, chunk, H_latent, W_latent)
            pred_latents = samples[:, :, -Ta:].to(dtype)
            true_latents = x[:, :, -Ta:]

            # 计算 latent_l2_loss
            batch_loss = torch.mean(torch.square(pred_latents - true_latents))
            latent_l2_loss_sum += batch_loss
            num_batches += 1
            if step % 1 == 0 and coordinator.is_master():
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{current_time}] step: {step}, loss: {latent_l2_loss_sum.item() / num_batches.item()}")

            # if step % 10 == 0:
            #     break
                # pbar.set_postfix({"loss": latent_l2_loss_sum.item() / num_batches, "step": step})
        dist.all_reduce(latent_l2_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

        final_loss = latent_l2_loss_sum / num_batches
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        model.train()
        return final_loss.item()
        

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _update_world_model(self, global_steps):
        timers = self.world_model_mapping['timers']
        device = self.world_model_mapping['device']
        dtype = self.world_model_mapping['dtype']
        record_time = self.world_model_mapping['record_time']
        cfg = self.world_model_mapping['cfg']
        mask_generator = self.world_model_mapping['mask_generator']
        model_args = self.world_model_mapping['model_args']
        vae = self.world_model_mapping['vae']
        model = self.world_model_mapping['model']
        optimizer = self.world_model_mapping['optimizer']
        scheduler = self.world_model_mapping['scheduler']
        booster = self.world_model_mapping['booster']
        lr_scheduler = self.world_model_mapping['lr_scheduler']
        ema = self.world_model_mapping['ema']
        coordinator = self.world_model_mapping['coordinator']
        model.train()
        timer_list = []

        # train_dataloader, val_dataloader = self._build_world_model_dataloader(global_steps)
        train_dataloader, _ = self._build_world_model_dataloader(global_steps)
        val_dataloader = train_dataloader
        start_time = time.time()
        latent_l2_before_train = self._evaluate_world_model(val_dataloader)
        if coordinator.is_master():
            end_time = time.time()
            print(f"evaluate_world_model time: {end_time - start_time}")
            print(f"latent_l2_before_train: {latent_l2_before_train}")
            # breakpoint()
        # return latent_l2_before_train
        running_loss = 0 
        dataloader_iter = iter(train_dataloader)
        # with tqdm(
        #     enumerate(dataloader_iter),
        #     desc=f"Training World Model",
        #     disable=not coordinator.is_master(),
        #     total=self.config.wm.wm_training_steps_per_epoch,
        # ) as pbar:
        #     for wm_training_step, batch in pbar:
        for wm_training_step, batch in enumerate(dataloader_iter):
            # print(f"wm_training_step: {wm_training_step}")
            if wm_training_step >= self.config.wm.wm_training_steps_per_epoch:
                break
            with timers["move_data"] as move_data_t:
                pinned_video = batch.pop("video")
                x = pinned_video.to(device, dtype, non_blocking=True)  # [B, C, T, H, W]
                y = batch.pop("action")
            if record_time:
                timer_list.append(move_data_t)
            # os.makedirs("./debug/train", exist_ok=True)
            # rank = dist.get_rank() if dist.is_initialized() else 0
            # log_file = open(f"./debug/train/debug_rank_{rank}.log", "a")
            # for i in range(x.shape[0]):
            #     info = {
            #         "unique_id":     batch["meta"]["unique_id"][i],
            #         "episode_name":  batch["meta"]["episode_name"][i],
            #         "video_start":   int(batch["meta"]["video_start"][i]),
            #         "video_end":     int(batch["meta"]["video_end"][i]),
            #         "action_start":  int(batch["meta"]["action_start"][i]),
            #         "action_end":    int(batch["meta"]["action_end"][i]),
            #     }
            #     log_file.write(f"[RANK {rank}] : {info}\n")
            # continue
            # == visual and text encoding ==
            with timers["encode"] as encode_t:
                with torch.no_grad():
                    # Prepare visual inputs
                    if cfg.get("load_video_features", False):
                        x = x.to(device, dtype)
                    else:
                        x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    model_args = {"y": y.to(device, dtype)}
            if record_time:
                timer_list.append(encode_t)

            # == mask ==
            with timers["mask"] as mask_t:
                mask = None
                if cfg.get("mask_ratios", None) is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask
            if record_time:
                timer_list.append(mask_t)

            # == video meta info ==
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_args[k] = v.to(device, dtype)

            # == diffusion loss computation ==
            with timers["diffusion"] as loss_t:
                loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
            if record_time:
                timer_list.append(loss_t)

            # == backward & update ==
            with timers["backward"] as backward_t:
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
            if record_time:
                timer_list.append(backward_t)

            with timers["optim"] as optim_t:
                optimizer.step()
                optimizer.zero_grad()

                # update learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()
            if record_time:
                timer_list.append(optim_t)

            # == update EMA ==
            with timers["update_ema"] as ema_t:
                update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
            if record_time:
                timer_list.append(ema_t)

            # == update log info ==
            with timers["reduce_loss"] as reduce_loss_t:
                all_reduce_mean(loss)
                running_loss += loss.item()

            # pbar.set_postfix({"loss": loss.item(), "step": wm_training_step})
            if wm_training_step != 0 and wm_training_step % 100 == 0 and coordinator.is_master():
                running_loss = running_loss/100
                print(f"loss: {running_loss:.4f}, step: {wm_training_step}")
                running_loss = 0

            if wm_training_step != 0 and wm_training_step % 500 == 0:
                latent_l2 = self._evaluate_world_model(val_dataloader)
                if coordinator.is_master():
                    print(f"step: {wm_training_step} latent_l2: {latent_l2}")
        
        latent_l2_after_train = self._evaluate_world_model(val_dataloader)
        if coordinator.is_master():
            print(f"latent_l2_after_train: {latent_l2_after_train}") 

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.wm.enable:
            start_time = time.time()
            self.world_model_mapping = self._build_world_model()
            print(f"init world model time: {time.time() - start_time}")


        start_time = time.time()
        from verl.workers.actor import RobDataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=True) #self.config.model.get('trust_remote_code', True)

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        print(f"init actor time: {time.time() - start_time}")

        if self._is_rollout:
            start_time = time.time()
            self.rollout, self.sharding_manager = self._build_rollout()
            print(f"init rollout time: {time.time() - start_time}") # 很快

        if self._is_ref:
            start_time = time.time()
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=True)[0] #self.config.model.get('trust_remote_code', False)
                                                                   
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            print(f"init ref time: {time.time() - start_time}")
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        #data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        #data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts, use_wm=False):
        
        prompts = prompts.to('cuda')
        save_to_hdfs = prompts.meta_info.get('save_to_hdfs', False)
        use_wm = prompts.meta_info.get('use_wm', False)
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        
        #tmp_sample = prompts.meta_info.get('n_samples', -1)
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print(f"gen seq will start, and the num samples are: {tmp_sample}")
    
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)    
            prompts = self.sharding_manager.preprocess_data(prompts)
            if use_wm:
                output = self.rollout.generate_wm_sequences(prompts=prompts)
            else:
                output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()
        gc.collect()
        # with Timer(name=f'gen seq end ,  old log will begin', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print("gen seq end ,  old log will begin")
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs
        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        if save_to_hdfs:
            save_to_hdfs_fun(output, prompts)
            try:
                output.pop(batch_keys=['video', 'action'])
            except:
                print("No video and action in the output")
        # TODO remove this line 
        # output.pop(batch_keys=['task_id', 'trial_id', 'drop'])
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from peft import PeftModel
        import transformers
        
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        #lora add
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            if dist.get_rank() == 0:
                os.makedirs(local_path, exist_ok=True)

            lora_save_path = os.path.join(local_path, "lora_adapter")

            if isinstance(self.actor_module_fsdp, FSDP):
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False, offload_to_cpu=True):
                    if dist.get_rank() == 0:
                        from typing import OrderedDict
                        lora_params = OrderedDict()
                        model = self.actor_module_fsdp._fsdp_wrapped_module.base_model.model
                        for name, param in model.named_parameters():
                            if ".lora_" in name:
                                name = "base_model.model." + name.replace("._fsdp_wrapped_module.", ".")
                                lora_params[name] = param
                        self.actor_module_fsdp.save_pretrained(
                            lora_save_path,
                            state_dict=lora_params,
                            safe_serialization=True
                        )
            else:
                self.actor_module.save_pretrained(lora_save_path, safe_serialization=True)

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")
            
            # save total model
            base_vla = AutoModelForVision2Seq.from_pretrained(
                self.config.model.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="cpu"
            )
            merged_vla = PeftModel.from_pretrained(base_vla, lora_save_path)
            merged_vla = merged_vla.merge_and_unload()

            if dist.get_rank() == 0:
                merged_vla.save_pretrained(local_path)
                print(f"Saved merged model at: {local_path}")

            # Wait for merged model to be saved
            dist.barrier()    
                
        
        # TODO: support DCP and save sharded checkpoints
        else:
            import torch.distributed
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.actor.actor_module.state_dict()
            if self.rank == 0:
                print(f'Saving actor checkpoint to {local_path}')
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_world_model_mapping(self, actor_local_path):
        exp_dir = os.path.join(actor_local_path, "world_model_mapping")
        global_step = self.config.wm.wm_training_steps_per_epoch
        checkpoint_io = CheckpointIO()
        booster = self.world_model_mapping['booster']
        model = self.world_model_mapping['model']
        ema = self.world_model_mapping['ema']
        optimizer = self.world_model_mapping['optimizer']
        lr_scheduler = self.world_model_mapping['lr_scheduler']
        cfg = self.world_model_mapping['cfg']
        coordinator = self.world_model_mapping['coordinator']
        ema_shape_dict = record_model_param_shape(ema)

        # if coordinator.is_master():
        #     breakpoint()

        save_dir = checkpoint_io.save(
            booster,
            exp_dir,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            step=0,
            global_step=global_step,
            batch_size=cfg.get("batch_size", None),
            ema_shape_dict=ema_shape_dict,
            async_io=False,
        )

        if coordinator.is_master():
            print(f"✅ 已保存 world model checkpoint 到: {exp_dir}")
            rm_state = {
                "rm_model_state_dict": self.world_model_mapping["rm_model"].state_dict(),
                "rm_optim_state_dict": self.world_model_mapping["rm_optim"].state_dict(),
            }
            save_path = os.path.join(exp_dir, "rm_state.pt")
            torch.save(rm_state, save_path)
            print(f"✅ 已保存 terminal model 到: {save_path}")

        torch.distributed.barrier()


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_world_model_mapping(self, actor_local_path):
        exp_dir = os.path.join(actor_local_path, "world_model_mapping")
        exp_dir = os.path.join(exp_dir, os.listdir(exp_dir)[0])
        checkpoint_io = CheckpointIO()
        booster = self.world_model_mapping['booster']
        model = self.world_model_mapping['model']
        ema = self.world_model_mapping['ema']
        optimizer = self.world_model_mapping['optimizer']
        lr_scheduler = self.world_model_mapping['lr_scheduler']
        cfg = self.world_model_mapping['cfg']
        coordinator = self.world_model_mapping['coordinator']
        ret = checkpoint_io.load(
            booster,
            exp_dir,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None,
        )
        if coordinator.is_master():
            print(f"✅ 已加载 world model checkpoint 从: {exp_dir}")
        load_path = os.path.join(os.path.dirname(exp_dir), "rm_state.pt")
        assert os.path.isfile(load_path), f"❌ 文件不存在: {load_path}"
        wm_state = torch.load(load_path, map_location="cpu")
        self.world_model_mapping["rm_model"].load_state_dict(wm_state["rm_model_state_dict"])
        self.world_model_mapping["rm_optim"].load_state_dict(wm_state["rm_optim_state_dict"])
        if coordinator.is_master():
            print(f"✅ 已加载 Terminal model 从: {load_path}")




class RobActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._is_lora = self.config.model.get('lora_rank', 0) > 0
        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        #add oft
         
        if self.config.model.vla == "openvla-oft":
            from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
            
        elif self.config.model.vla == "openvla":
            from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
            from verl.utils.vla_utils.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from verl.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
        
        #add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, local_files_only=True, model = self.config.model.vla)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, local_files_only=True)
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # if self.rank == 0:
        #     print(f'Model config after override: {actor_model_config}')

        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.config.model.vla == "openvla-oft":
                actor_module = AutoModelForVision2Seq.from_pretrained(
                                                        pretrained_model_name_or_path=local_path,
                                                        torch_dtype=torch_dtype,
                                                        #attn_implementation="flash_attention_2",
                                                        config=actor_model_config,              
                                                        trust_remote_code=True,
                                                        local_files_only=True
                                                    )
                print("load model successfully")
                #oft add
                actor_module.vision_backbone.set_num_images_in_input(self.config.actor.num_images_in_input)
                
                dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")
                if os.path.isfile(dataset_statistics_path):
                    with open(dataset_statistics_path, "r") as f:
                        norm_stats = json.load(f)
                    actor_module.norm_stats = norm_stats
                else:
                    print(
                        "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                        "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                        "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                    )
            elif self.config.model.vla == "openvla":
                actor_module = AutoModelForVision2Seq.from_pretrained(
                                                    pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation="flash_attention_2",
                                                    config=actor_model_config,              
                                                    trust_remote_code=True,
                                                )
           
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
            # lora add
            if self._is_lora:
                print("Applying LoRA to actor module")
                
                lora_config = {
                    #'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    "lora_dropout": 0 ,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'init_lora_weights': "gaussian"
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))  
                actor_module.print_trainable_parameters()
            # lora end
                
                
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None
        
        #oft add
        auto_wrap_policy = get_fsdp_wrap_policy_vla(module=actor_module, config=fsdp_config.get('wrap_policy', None), is_lora=self.config.model.get('lora_rank', 0) > 0)
        #oft add end
        

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        
        # TODO: add transformer policy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module_fsdp = FSDP(
                actor_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import RobHFRollout
            from verl.workers.hybrid_engine import BaseShardingManager
            rollout = RobHFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            raise ValueError
            # from verl.workers.rollout.vllm_rollout import vLLMRollout
            # from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            # log_gpu_memory_usage('Before building vllm rollout', logger=None)
            # rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
            #                       config=self.config.rollout,
            #                       tokenizer=self.tokenizer,
            #                       model_hf_config=self.actor_model_config)
            # log_gpu_memory_usage('After building vllm rollout', logger=None)
            # if torch.distributed.get_world_size() == 1:
            #     self.config.rollout.load_format = 'dummy_hf'
            # sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
            #                                            inference_engine=rollout.inference_engine,
            #                                            model_config=self.actor_model_config,
            #                                            full_params='hf' in self.config.rollout.load_format)
            # log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import RobDataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=True) #self.config.model.get('trust_remote_code', True)

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)
        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=True)[0] #self.config.model.get('trust_remote_code', False)
                                                                   
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # breakpoint()
        #data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        #data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        # breakpoint()
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)
        save_to_hdfs = prompts.meta_info.get('save_to_hdfs', False)
        

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        
        #tmp_sample = prompts.meta_info.get('n_samples', -1)
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print(f"gen seq will start, and the num samples are: {tmp_sample}")
        
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)    
            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger)
            # breakpoint()
            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        # with Timer(name=f'gen seq end ,  old log will begin', text="{name}: {seconds:.1f} seconds") as timer:    
        #     print("gen seq end ,  old log will begin")
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)

        if save_to_hdfs:
            save_to_hdfs_fun(output, prompts)
            output.pop(batch_keys=['video', 'action'])
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from peft import PeftModel
        import transformers
        
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        #lora add
        if self._is_lora and isinstance(self.actor_module, PeftModel):
            if dist.get_rank() == 0:
                os.makedirs(local_path, exist_ok=True)

            lora_save_path = os.path.join(local_path, "lora_adapter")

            if isinstance(self.actor_module_fsdp, FSDP):
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False, offload_to_cpu=True):
                    if dist.get_rank() == 0:
                        from typing import OrderedDict
                        lora_params = OrderedDict()
                        model = self.actor_module_fsdp._fsdp_wrapped_module.base_model.model
                        for name, param in model.named_parameters():
                            if ".lora_" in name:
                                name = "base_model.model." + name.replace("._fsdp_wrapped_module.", ".")
                                lora_params[name] = param
                        self.actor_module_fsdp.save_pretrained(
                            lora_save_path,
                            state_dict=lora_params,
                            safe_serialization=True
                        )
            else:
                self.actor_module.save_pretrained(lora_save_path, safe_serialization=True)

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")
            
            # save total model
            base_vla = AutoModelForVision2Seq.from_pretrained(
                self.config.model.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="cpu"
            )
            merged_vla = PeftModel.from_pretrained(base_vla, lora_save_path)
            merged_vla = merged_vla.merge_and_unload()

            if dist.get_rank() == 0:
                merged_vla.save_pretrained(local_path)
                print(f"Saved merged model at: {local_path}")

            # Wait for merged model to be saved
            dist.barrier()    
                
        
        # TODO: support DCP and save sharded checkpoints
        else:
            import torch.distributed
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.actor.actor_module.state_dict()
            if self.rank == 0:
                print(f'Saving actor checkpoint to {local_path}')
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        if self.config.model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # if self.rank == 0:
        #     print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            actor_module = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code, local_files_only=True)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.hybrid_engine import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.hybrid_engine import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                       inference_engine=rollout.inference_engine,
                                                       model_config=self.actor_model_config,
                                                       full_params='hf' in self.config.rollout.load_format)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            # breakpoint()
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size()
        self.config.ppo_micro_batch_size //= torch.distributed.get_world_size()

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                 torch_dtype=torch_dtype,
                                                                 config=critic_model_config,
                                                                 attn_implementation='flash_attention_2',
                                                                 trust_remote_code=trust_remote_code)
            critic_module.lm_head = nn.Sequential(nn.Linear(critic_model_config.hidden_size, 1, dtype=torch_dtype),
                                                  LambdaLayer(fn=squeeze))

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable()
        if self.rank == 0:
            print_model_size(critic_module)

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ppo_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())
        metrics = self.critic.update_critic(data=data)

        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics['critic/lr(1e-4)'] = lr * 1e4

        output = DataProto(batch=None, meta_info={'metrics': metrics})
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                               torch_dtype=torch.bfloat16,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload))

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.reward_module(input_ids=micro_batch['input_ids'],
                                        attention_mask=micro_batch['attention_mask'],
                                        position_ids=micro_batch['position_ids'])
            rm_score = output.logits  # (batch_size,)
            rm_score = rm_score.squeeze(-1)
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()
        micro_batches = rm_data.batch.split(self.config.micro_batch_size)
        output = []
        for micro_batch in micro_batches:
            rm_score = self._forward_micro_batch(micro_batch)
            output.append(rm_score)
        scores = torch.cat(output, dim=0)  # (batch_size)
        token_level_scores = self._expand_to_token_level(data, scores)
        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

class PRIMERewardModelWorker(Worker):
    """
    PRIME reward model.
    Can update itself whenever compute_rm_score is called.
    """
    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        world_size = torch.distributed.get_world_size()
        self.config.mini_batch_size //= world_size
        self.config.micro_batch_size //= world_size
        # build device mesh
        
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._is_offload_param = self.config.prime_model.fsdp_config.get('param_offload', False)
        self._is_offload_grad = self.config.prime_model.fsdp_config.get('grad_offload', False)
        self._is_offload_optimizer = self.config.prime_model.fsdp_config.get('optimizer_offload', False)

    def _build_model_optimizer(self, config, enable_gradient_checkpointing=False):
        # the following line is necessary
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.prime_model.path)

        if self.config.prime_model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.prime_model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.prime_model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.prime_model.get('trust_remote_code', False))

        trust_remote_code = config.prime_model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        if config.prime_model.use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            reward_module = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            reward_module.to(torch.float32)
            if enable_gradient_checkpointing:
                reward_module.gradient_checkpointing_enable()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)
        if config.prime_model.get('enable_gradient_checkpointing', False):
            reward_module.gradient_checkpointing_enable()

        if config.prime_model.get("ref_type", 'freeze') == 'freeze':
            reference_module = AutoLigerKernelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(config.prime_model.ref_path),
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code)
            reference_module.to(torch.bfloat16)
            for param in reference_module.parameters():
                param.requires_grad = False
        else:
            reference_module = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.prime_model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reference_module, config=self.config.prime_model.fsdp_config)
        if reference_module is not None:
            reference_module = FSDP(
                reference_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
                device_mesh=self.device_mesh,
                sync_module_states=True)

        self.update_dpo_type = self.config.prime_model.get('update', 'none')
        if self.update_dpo_type in ['before', 'after']:

            from torch import optim
            self.reward_optimizer = optim.AdamW(reward_module.parameters(),
                                                lr=config.prime_model.optim.lr,
                                                betas=config.prime_model.optim.get('betas', (0.9, 0.999)),
                                                weight_decay=config.prime_model.optim.get('weight_decay', 1e-2))

            total_steps = config.prime_model.optim.get('total_training_steps', 0)
            num_warmup_steps_ratio = config.prime_model.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            self.reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.reward_optimizer,
                                                                         num_warmup_steps=num_warmup_steps)

            # fsdp offload configurations
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=reward_module, offload_grad=self._is_offload_grad)
            if reference_module is not None:
                offload_fsdp_param_and_grad(module=reference_module, offload_grad=self._is_offload_grad)

        return reward_module, reference_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPRIME
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.prime_model.get('external_lib', None))
        self.reward_module, self.reference_module = self._build_model_optimizer(config=self.config, enable_gradient_checkpointing=self.config.prime_model.get('enable_gradient_checkpointing', False))
        self.prm = DataParallelPRIME(config=self.config,
                                    reward_module=self.reward_module,
                                    reference_module=self.reference_module,
                                    reward_optimizer=self.reward_optimizer,
                                    prime_loss_fn=self.config.prime_model.get('loss_type', 'ce'))
        torch.cuda.empty_cache()

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        n_samples=data.meta_info['n_samples']
        beta=self.config.prime_model.get('beta_train', 0.05)
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_data=data

        if self.update_dpo_type!='none':
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.reward_module,device_id=torch.cuda.current_device(),load_grad=self._is_offload_grad)
            if self.reference_module is not None:
                load_fsdp_param_and_grad(module=self.reference_module,device_id=torch.cuda.current_device(),load_grad=self._is_offload_grad)
        
        token_level_scores, metrics = self.prm.update_policy(rm_data)

        output=DataProto.from_dict(tensors = {'rm_scores': token_level_scores}, meta_info = {'metrics': metrics})

        if self.update_dpo_type != 'none':
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)
            self.reward_lr_scheduler.step()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.reward_module, offload_grad=self._is_offload_grad)
            if self.reference_module is not None:
                offload_fsdp_param_and_grad(module=self.reference_module, offload_grad=self._is_offload_grad)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.reward_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.reward_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.reward_module.state_dict()
        if self.rank == 0:
            print(f'Saving reward checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.reward_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            if hdfs_path is not None:
                print(f'Uploading reward checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.reward_module, offload_grad=self._is_offload_grad)

# @hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
# def main(config):

#     print()
    
# if __name__ == '__main__':
#     main()