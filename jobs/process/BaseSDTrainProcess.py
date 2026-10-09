import copy
import glob
import inspect
import json
import math
import random
import shutil
from collections import OrderedDict, deque
import os
import re
import traceback
from typing import Dict, Union, List, Optional, TYPE_CHECKING

import numpy as np
import yaml
from diffusers import T2IAdapter, ControlNetModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from safetensors.torch import save_file, load_file
# from lycoris.config import PRESET
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
import torch
import torch.backends.cuda
from huggingface_hub import HfApi, Repository, interpreter_login
from huggingface_hub.utils import HfFolder
from toolkit.memory_management import MemoryManager, release_offload_buffers, log_memory_breakdown, release_cached_memory

from toolkit.basic import value_map
from toolkit.cache_paths import get_cache_root_candidates, get_cache_dir_candidates
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.data_loader import (
    apply_dataloader_resume_state,
    get_dataloader_from_datasets,
    get_dataloader_datasets,
    get_dataloader_resume_state,
    mark_dataloader_batch_processed,
    start_next_dataloader_epoch,
    trigger_dataloader_setup_epoch,
)
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.dataloader_mixins import (
    cache_file_item_latents_to_disk,
    compact_latent_shards,
    flush_buffered_file_item_latents_to_disk,
    get_buffered_file_item_latent_count,
    clear_buffered_file_item_latents_to_disk,
)
from toolkit.ema import ExponentialMovingAverage
from toolkit.embedding import Embedding
from toolkit.image_utils import show_tensors, show_latents, reduce_contrast
from toolkit.ip_adapter import IPAdapter
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lorm import convert_diffusers_unet_to_lorm, count_parameters, print_lorm_extract_details, \
    lorm_ignore_if_contains, lorm_parameter_threshold, LORM_TARGET_REPLACE_MODULE
from toolkit.lycoris_special import LycorisSpecialNetwork
from toolkit.models.decorator import Decorator
from toolkit.network_mixins import Network
from toolkit.optimizer import get_optimizer
from toolkit.paths import CONFIG_ROOT
from toolkit.progress_bar import ToolkitProgressBar
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit.saving import save_t2i_from_diffusers, load_t2i_model, save_ip_adapter_from_diffusers, \
    load_ip_adapter_model, load_custom_adapter_model

from toolkit.scheduler import get_lr_scheduler
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_base_model_info_to_meta, \
    parse_metadata_from_safetensors
from toolkit.train_tools import get_torch_dtype, LearnableSNRGamma, apply_learnable_snr_gos, apply_snr_weight
import gc

from tqdm import tqdm

from toolkit.config_modules import SaveConfig, LoggingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig, \
    GenerateImageConfig, EmbeddingConfig, DatasetConfig, preprocess_dataset_raw_config, AdapterConfig, GuidanceConfig, validate_configs, \
    DecoratorConfig
from toolkit.logging_aitk import create_logger
from diffusers import FluxTransformer2DModel
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.print import print_acc
import time
from toolkit.checkpoint_writer import (BackgroundCheckpointWriter, CheckpointJob, UnsupportedState,
                                       remove_stale_temps, snapshot_to_cpu)


class SamplingInterrupted(Exception):
    """Raised between sample images to stop sampling early (the job is stopping)."""
from accelerate import Accelerator
import transformers
import diffusers
import hashlib

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from toolkit.util.blended_blur_noise import get_blended_blur_noise
from toolkit.util.get_model import get_model_class

def flush():
    torch.cuda.empty_cache()
    gc.collect()


class BaseSDTrainProcess(BaseTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, custom_pipeline=None):
        super().__init__(process_id, job, config)
        self.accelerator: Accelerator = get_accelerator()
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        self.sd: StableDiffusion
        self.embedding: Union[Embedding, None] = None

        self.custom_pipeline = custom_pipeline
        self.step_num = 0
        self.start_step = 0
        self.epoch_num = 0
        self.last_save_step = 0
        # start at 1 so we can do a sample at the start
        self.grad_accumulation_step = 1
        # if true, then we do not do an optimizer step. We are accumulating gradients
        self.is_grad_accumulation_step = False
        self.device = str(self.accelerator.device)
        self.device_torch = self.accelerator.device
        network_config = self.get_conf('network', None)
        if network_config is not None:
            self.network_config = NetworkConfig(**network_config)
        else:
            self.network_config = None
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        model_config = self.get_conf('model', {})
        self.modules_being_trained: List[torch.nn.Module] = []

        # update modelconfig dtype to match train
        model_config['dtype'] = self.train_config.dtype
        self.model_config = ModelConfig(**model_config)

        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        first_sample_config = self.get_conf('first_sample', None)
        if first_sample_config is not None:
            self.has_first_sample_requested = True
            self.first_sample_config = SampleConfig(**first_sample_config)
        else:
            self.has_first_sample_requested = False
            self.first_sample_config = self.sample_config
        self.logging_config = LoggingConfig(**self.get_conf('logging', {}))
        self.logger = create_logger(self.logging_config, config, self.save_root)
        self.cleanup_caches_on_startup = self.get_conf('cleanup_caches_on_startup', False)
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler = None
        self.data_loader: Union[DataLoader, None] = None
        self.data_loader_reg: Union[DataLoader, None] = None
        self.trigger_word = self.get_conf('trigger_word', None)

        self.guidance_config: Union[GuidanceConfig, None] = None
        guidance_config_raw = self.get_conf('guidance', None)
        if guidance_config_raw is not None:
            self.guidance_config = GuidanceConfig(**guidance_config_raw)

        # store is all are cached. Allows us to not load vae if we don't need to
        self.is_latents_cached = True
        raw_datasets = self.get_conf('datasets', None)
        if raw_datasets is not None and len(raw_datasets) > 0:
            raw_datasets = preprocess_dataset_raw_config(raw_datasets)
        self.datasets = None
        self.datasets_reg = None
        self.dataset_configs: List[DatasetConfig] = []
        self.params = []
        
        # add dataset text embedding cache to their config
        if self.train_config.cache_text_embeddings:
            for raw_dataset in raw_datasets:
                raw_dataset['cache_text_embeddings'] = True
        
        if raw_datasets is not None and len(raw_datasets) > 0:
            for raw_dataset in raw_datasets:
                dataset = DatasetConfig(**raw_dataset)
                # handle trigger word per dataset
                if dataset.trigger_word is None and self.trigger_word is not None:
                    dataset.trigger_word = self.trigger_word
                is_caching = dataset.cache_latents or dataset.cache_latents_to_disk
                if not is_caching:
                    self.is_latents_cached = False
                if dataset.is_reg:
                    if self.datasets_reg is None:
                        self.datasets_reg = []
                    self.datasets_reg.append(dataset)
                else:
                    if self.datasets is None:
                        self.datasets = []
                    self.datasets.append(dataset)
                self.dataset_configs.append(dataset)
        
        self.is_caching_text_embeddings = any(
            dataset.cache_text_embeddings for dataset in self.dataset_configs
        )
        self.has_lazy_latent_caching = any(
            getattr(dataset, 'lazy_cache_latents', False) and getattr(dataset, 'cache_latents_to_disk', False)
            for dataset in self.dataset_configs
        )
        self.has_lazy_text_embedding_caching = any(
            getattr(dataset, 'lazy_cache_text_embeddings', False) for dataset in self.dataset_configs
        )
        self.unload_text_encoder_for_training = (
            self.train_config.unload_text_encoder or self.is_caching_text_embeddings
        )
        self.can_replace_text_encoder_after_caching = (
            self.is_caching_text_embeddings and not self.has_lazy_text_embedding_caching
        )
        self._adaptive_lazy_flush_history = deque(maxlen=256)
        self._adaptive_lazy_flush_min_history = 24
        self._adaptive_lazy_flush_cooldown_steps = 32
        self._adaptive_lazy_flush_min_pending_latents = 128
        self._last_adaptive_lazy_flush_step = -10**9

        self.embed_config = None
        embedding_raw = self.get_conf('embedding', None)
        if embedding_raw is not None:
            self.embed_config = EmbeddingConfig(**embedding_raw)
        
        self.decorator_config: DecoratorConfig = None
        decorator_raw = self.get_conf('decorator', None)
        if decorator_raw is not None:
            if not self.model_config.is_flux:
                raise ValueError("Decorators are only supported for Flux models currently")
            self.decorator_config = DecoratorConfig(**decorator_raw)

        # t2i adapter
        self.adapter_config = None
        adapter_raw = self.get_conf('adapter', None)
        if adapter_raw is not None:
            self.adapter_config = AdapterConfig(**adapter_raw)
            # sdxl adapters end in _xl. Only full_adapter_xl for now
            if self.model_config.is_xl and not self.adapter_config.adapter_type.endswith('_xl'):
                self.adapter_config.adapter_type += '_xl'

        # to hold network if there is one
        self.network: Union[Network, None] = None
        self.adapter: Union[T2IAdapter, IPAdapter, ClipVisionAdapter, ReferenceAdapter, CustomAdapter, ControlNetModel, None] = None
        self.embedding: Union[Embedding, None] = None
        self.decorator: Union[Decorator, None] = None

        is_training_adapter = self.adapter_config is not None and self.adapter_config.train

        self.do_lorm = self.get_conf('do_lorm', False)
        self.lorm_extract_mode = self.get_conf('lorm_extract_mode', 'ratio')
        self.lorm_extract_mode_param = self.get_conf('lorm_extract_mode_param', 0.25)
        # 'ratio', 0.25)

        # get the device state preset based on what we are training
        self.train_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.unload_text_encoder_for_training,
            require_grads=False  # we ensure them later
        )
        
        self.get_params_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.unload_text_encoder_for_training,
            require_grads=True  # We check for grads when getting params
        )

        # fine_tuning here is for training actual SD network, not LoRA, embeddings, etc. it is (Dreambooth, etc)
        self.is_fine_tuning = True
        if self.network_config is not None or is_training_adapter or self.embed_config is not None or self.decorator_config is not None:
            self.is_fine_tuning = False

        self.named_lora = False
        if self.embed_config is not None or is_training_adapter:
            self.named_lora = True
        self.snr_gos: Union[LearnableSNRGamma, None] = None
        self.ema: ExponentialMovingAverage = None
        
        validate_configs(self.train_config, self.model_config, self.save_config, self.dataset_configs)
        
        do_profiler = self.get_conf('torch_profiler', False)
        self.torch_profiler = None if not do_profiler else torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        
        self.current_boundary_index = 0
        self.steps_this_boundary = 0
        self.num_consecutive_oom = 0
        # OOM diagnostics (see _describe_oom / _record_oom_skip)
        self._oom_skipped_steps = None  # seeded from skipped_oom*.txt so the tally survives resumes
        self._oom_skip_total = None
        self._oom_rescued_steps = None  # OOMs the one-time retry turned into a normal step
        self._oom_rescued_total = None
        self._oom_last_step = None
        self._vram_outside_pytorch_at_start = None
        self._vram_baseline_provisional = False
        # per-step bookkeeping, reset before every step and updated by hook_train_loop
        self._current_microbatch_index = None
        self._oom_phase = None
        self._oom_lora_updated = False
        self._optimizer_update_probe_limit = 8
        self._consecutive_no_optimizer_updates = 0
        self._consecutive_no_optimizer_grads = 0
        self._consecutive_nonfinite_grad_steps = 0
        self._did_log_optimizer_update = False
        # Optimizer-update health check: only count steps where the LR is actually
        # > 0 (so warmup-from-0 and zero-LR ramps don't false-abort), require many
        # consecutive bad steps before erroring, and allow a full opt-out.
        self._optimizer_update_check_enabled = not bool(
            getattr(self.train_config, 'disable_optimizer_update_check', False)
        )
        self._optimizer_update_check_max_consecutive = max(
            1, int(getattr(self.train_config, 'optimizer_update_check_max_consecutive', 25))
        )

    def post_process_generate_image_config_list(self, generate_image_config_list: List[GenerateImageConfig]):
        # override in subclass
        return generate_image_config_list

    def sample(self, step=None, is_first=False):
        if not self.accelerator.is_main_process:
            return
        flush()
        sample_folder = os.path.join(self.save_root, 'samples')
        gen_img_config_list = []

        sample_config = self.first_sample_config if is_first else self.sample_config
        start_seed = sample_config.seed
        current_seed = start_seed

        test_image_paths = []
        if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
            test_image_path_list = self.adapter_config.test_img_path
            # divide up images so they are evenly distributed across prompts
            for i in range(len(sample_config.prompts)):
                test_image_paths.append(test_image_path_list[i % len(test_image_path_list)])

        for i in range(len(sample_config.prompts)):
            if sample_config.walk_seed:
                current_seed = start_seed + i

            step_num = ''
            if step is not None:
                # zero-pad 9 digits
                step_num = f"_{str(step).zfill(9)}"

            filename = f"[time]_{step_num}_[count].{self.sample_config.ext}"

            output_path = os.path.join(sample_folder, filename)

            prompt = sample_config.prompts[i]

            # add embedding if there is one
            # note: diffusers will automatically expand the trigger to the number of added tokens
            # ie test123 will become test123 test123_1 test123_2 etc. Do not add this yourself here
            if self.embedding is not None:
                prompt = self.embedding.inject_embedding_to_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
                prompt = self.adapter.inject_trigger_into_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.trigger_word is not None:
                prompt = self.sd.inject_trigger_into_prompt(
                    prompt, self.trigger_word, add_if_not_present=False
                )

            extra_args = {}
            if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
                extra_args['adapter_image_path'] = test_image_paths[i]
            
            sample_item = sample_config.samples[i]
            if sample_item.seed is not None:
                current_seed = sample_item.seed

            gen_img_config_list.append(GenerateImageConfig(
                prompt=prompt,  # it will autoparse the prompt
                width=sample_item.width,
                height=sample_item.height,
                negative_prompt=sample_item.neg,
                seed=current_seed,
                guidance_scale=sample_item.guidance_scale,
                guidance_rescale=sample_config.guidance_rescale,
                num_inference_steps=sample_item.sample_steps,
                network_multiplier=sample_item.network_multiplier,
                output_path=output_path,
                output_ext=sample_config.ext,
                adapter_conditioning_scale=sample_config.adapter_conditioning_scale,
                refiner_start_at=sample_config.refiner_start_at,
                extra_values=sample_config.extra_values,
                logger=self.logger,
                num_frames=sample_item.num_frames,
                fps=sample_item.fps,
                ctrl_img=sample_item.ctrl_img,
                ctrl_idx=sample_item.ctrl_idx,
                ctrl_img_1=sample_item.ctrl_img_1,
                ctrl_img_2=sample_item.ctrl_img_2,
                ctrl_img_3=sample_item.ctrl_img_3,
                do_cfg_norm=sample_config.do_cfg_norm,
                **extra_args
            ))

        # post process
        gen_img_config_list = self.post_process_generate_image_config_list(gen_img_config_list)

        # if we have an ema, set it to validation mode
        if self.ema is not None:
            self.ema.eval()

        # let adapter know we are sampling
        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            self.adapter.is_sampling = True
        
        # send to be generated
        rng_before_sampling = self._capture_rng_resume_state()
        try:
            self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)
        except SamplingInterrupted:
            print_acc("\nSampling stopped early: the job is stopping")
            self._restore_rng_resume_state(rng_before_sampling)
            try:
                self.sd.restore_device_state()  # generate_images does this only when it finishes
            except Exception as e:
                print_acc(f"Could not restore the device state after sampling: {e}")
        finally:
            if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
                self.adapter.is_sampling = False

            if self.ema is not None:
                self.ema.train()

    def update_training_metadata(self):
        o_dict = OrderedDict({
            "training_info": self.get_training_info()
        })
        o_dict['ss_base_model_version'] = self.sd.get_base_model_version()

        # o_dict = add_base_model_info_to_meta(
        #     o_dict,
        #     is_v2=self.model_config.is_v2,
        #     is_xl=self.model_config.is_xl,
        # )
        o_dict['ss_output_name'] = self.job.name

        if self.trigger_word is not None:
            # just so auto1111 will pick it up
            o_dict['ss_tag_frequency'] = {
                f"1_{self.trigger_word}": {
                    f"{self.trigger_word}": 1
                }
            }

        self.add_meta(o_dict)

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num,
            'epoch': self.epoch_num,
        })
        return info

    def clean_up_saves(self, remove_stop_saves=False):
        if not self.accelerator.is_main_process:
            return
        # remove old saves
        # get latest saved step
        latest_item = None
        if os.path.exists(self.save_root):
            # pattern is {job_name}_{zero_filled_step} for both files and directories
            pattern = f"{self.job.name}_*"
            items = glob.glob(os.path.join(self.save_root, pattern))
            # Separate files and directories
            safetensors_files = [f for f in items if f.endswith('.safetensors')]
            pt_files = [f for f in items if f.endswith('.pt')]
            directories = [d for d in items if os.path.isdir(d) and not d.endswith('.safetensors')]
            embed_files = []
            # do embedding files
            if self.embed_config is not None:
                embed_pattern = f"{self.embed_config.trigger}_*"
                embed_items = glob.glob(os.path.join(self.save_root, embed_pattern))
                # will end in safetensors or pt
                embed_files = [f for f in embed_items if f.endswith('.safetensors') or f.endswith('.pt')]

            # check for critic files
            critic_pattern = f"CRITIC_{self.job.name}_*"
            critic_items = glob.glob(os.path.join(self.save_root, critic_pattern))

            # A save made for a stop lands between the save_every steps. It is the newest
            # checkpoint until the run goes on and saves again; it must not push regular
            # checkpoints out of max_step_saves_to_keep, so it is set aside here and removed
            # once a later save exists.
            safetensors_files, stop_saves = self._split_stop_saves(safetensors_files)

            # Sort the lists by creation time if they are not empty
            if safetensors_files:
                safetensors_files.sort(key=os.path.getctime)
            if pt_files:
                pt_files.sort(key=os.path.getctime)
            if directories:
                directories.sort(key=os.path.getctime)
            if embed_files:
                embed_files.sort(key=os.path.getctime)
            if critic_items:
                critic_items.sort(key=os.path.getctime)

            # Combine and sort the lists
            combined_items = safetensors_files + directories + pt_files
            combined_items.sort(key=os.path.getctime)
            
            num_saves_to_keep = self.save_config.max_step_saves_to_keep
            
            if hasattr(self.sd, 'max_step_saves_to_keep_multiplier'):
                num_saves_to_keep *= self.sd.max_step_saves_to_keep_multiplier

            # Use slicing with a check to avoid 'NoneType' error
            safetensors_to_remove = safetensors_files[
                                    :-num_saves_to_keep] if safetensors_files else []
            pt_files_to_remove = pt_files[:-num_saves_to_keep] if pt_files else []
            directories_to_remove = directories[:-num_saves_to_keep] if directories else []
            embeddings_to_remove = embed_files[:-num_saves_to_keep] if embed_files else []
            critic_to_remove = critic_items[:-num_saves_to_keep] if critic_items else []

            items_to_remove = safetensors_to_remove + pt_files_to_remove + directories_to_remove + embeddings_to_remove + critic_to_remove
            newest_step = max((s for s in map(self._save_step_of, safetensors_files + stop_saves) if s is not None), default=-1)
            items_to_remove += [f for f in stop_saves
                                if remove_stop_saves or self._save_step_of(f) != newest_step]

            # remove all but the latest max_step_saves_to_keep
            # items_to_remove = combined_items[:-num_saves_to_keep]

            # remove duplicates
            items_to_remove = list(dict.fromkeys(items_to_remove))

            for item in items_to_remove:
                print_acc(f"Removing old save: {item}")
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                # see if a yaml file with same name exists
                yaml_file = os.path.splitext(item)[0] + ".yaml"
                if os.path.exists(yaml_file):
                    os.remove(yaml_file)
            if combined_items:
                latest_item = combined_items[-1]
        return latest_item

    def post_save_hook(self, save_path):
        # override in subclass
        pass

    @staticmethod
    def _save_step_of(path):
        match = re.search(r"_(\d{9})\.safetensors$", os.path.basename(path))
        return int(match.group(1)) if match else None

    def _checkpoint_exists_for_step(self, step):
        lora_name = self.job.name + ('_LoRA' if self.named_lora else '')
        return os.path.exists(os.path.join(self.save_root, f"{lora_name}_{str(step).zfill(9)}.safetensors"))

    @staticmethod
    def _is_stop_save(path):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                return (f.metadata() or {}).get("aitk_saved_before_stop") == "true"
        except Exception:
            return False

    def _split_stop_saves(self, files):
        regular, stop_saves = [], []
        for f in files:
            (stop_saves if self._is_stop_save(f) else regular).append(f)
        return regular, stop_saves

    # ---- checkpoints written in the background ----------------------------------------------
    # A save used to stop training for about a minute (hashing, then writing ~0.9 GB of LoRA and
    # ~0.95 GB of optimizer state to the job folder's USB drive). Now the training thread takes
    # private CPU copies and toolkit/checkpoint_writer.py writes them while training continues.

    def hook_background_save_finished(self, result):
        """Called on the training thread once a background-written checkpoint is on disk."""
        pass

    def hook_waiting_for_background_save(self, step):
        """Called before the process waits, on its way out, for a checkpoint still being written."""
        pass

    def save_before_stop_requested(self) -> bool:
        """True when a stop was requested and the step being trained should be saved first."""
        return False

    def background_save_possible(self) -> bool:
        """Whether save() writes this run's checkpoints in the background (cheap; any thread)."""
        if not getattr(self.save_config, "async_save", True) or os.environ.get("AI_TOOLKIT_ASYNC_SAVE", "1") == "0":
            return False
        if self.is_fine_tuning or self.train_config.merge_network_on_save:
            return False
        if getattr(self, "network", None) is None and getattr(getattr(self, "sd", None), "network", None) is None:
            return False
        # these write more files, or change live state while saving: keep them synchronous
        if self.embedding is not None or self.decorator is not None:
            return False
        if self.adapter is not None and getattr(self.adapter_config, "train", False):
            return False
        if hasattr(getattr(self, "sd", None), "save_lora"):
            return False  # writes its own files (wan22: one per noise level) with save_file
        return True

    def _can_save_in_background(self, network) -> bool:
        if not self.background_save_possible():
            return False
        base_model = network.base_model_ref() if getattr(network, "base_model_ref", None) is not None else None
        return not (base_model is not None and hasattr(base_model, "save_lora"))

    def _save_in_background(self, step, network, file_path, save_meta):
        started = time.monotonic()
        # everything the checkpoint holds is copied here, on the training thread, before the next
        # step can change it; the writer only ever sees these private CPU copies
        prev_multiplier = network.multiplier
        network.multiplier = 1.0
        try:
            tensors = network.get_state_dict(dtype=get_torch_dtype(self.save_config.dtype))
        finally:
            network.multiplier = prev_multiplier
        companions = []
        notes = []  # parts that could not be copied: the checkpoint is then reported as incomplete
        committed_early = []
        if self.optimizer is not None:
            optimizer_path = os.path.join(self.save_root, 'optimizer.pt')
            try:
                try:
                    state_dict = unwrap_model(self.optimizer).state_dict()
                except Exception as e:
                    state_dict = self.optimizer.state_dict()
                try:
                    companions.append((optimizer_path, snapshot_to_cpu(state_dict), "optimizer"))
                except UnsupportedState as e:
                    # an optimizer with state objects of its own: save it here, as before
                    print_acc(f"Saving the optimizer on the training thread ({e})")
                    torch.save(state_dict, optimizer_path)
                    print_acc(f"Saved optimizer to {optimizer_path}")
                    committed_early.append('optimizer.pt')
                state_dict = None
            except Exception as e:
                print_acc(e)
                print_acc("Could not save optimizer")
                notes.append(f"Could not save optimizer: {e}")
        if step is not None:
            try:
                companions.append((self._get_dataloader_resume_state_path(),
                                   self._build_dataloader_resume_state(step), "dataloader resume state"))
            except Exception as e:
                print_acc(f"Could not save dataloader resume state: {e}")
                notes.append(f"Could not save dataloader resume state: {e}")
        if self.snr_gos:
            json_data = {
                'offset_1': self.snr_gos.offset_1.item(),
                'offset_2': self.snr_gos.offset_2.item(),
                'scale': self.snr_gos.scale.item(),
                'gamma': self.snr_gos.gamma.item(),
            }
            with open(os.path.join(self.save_root, 'learnable_snr.json'), 'w') as f:
                json.dump(json_data, f, indent=4)
        if self.ema is not None:
            self.ema.train()  # back to the live weights before training continues

        writer = getattr(self, "_checkpoint_writer", None)
        if writer is None:
            writer = self._checkpoint_writer = BackgroundCheckpointWriter()
        metadata = OrderedDict(save_meta or {})
        if getattr(self, "_saving_before_stop", False):
            # clean_up_saves keeps it only until the run saves again (see _split_stop_saves)
            metadata["aitk_saved_before_stop"] = "true"
        writer.submit(CheckpointJob(step=step, lora_path=file_path, lora_tensors=tensors,
                                    lora_metadata=metadata, companions=companions, notes=notes,
                                    committed_early=committed_early))
        self.last_save_in_background = True
        tensors = companions = None
        print_acc(f"Copied the checkpoint in {time.monotonic() - started:.1f}s; "
                  f"writing {os.path.basename(file_path)} in the background")
        if step is None:
            # the final save: nothing after it (push to hub, clean-up, "completed") may run first
            self._wait_for_background_save()
        flush()

    def _report_background_save(self, result, raise_errors=True):
        for note in result.notes:
            print_acc(note)
        if result.error is not None:
            if result.committed:
                consequence = (f"{', '.join(result.committed)} already hold step {result.step}, "
                               f"but the newest LoRA is still the previous one")
            else:
                consequence = "the previous checkpoint is untouched"
            message = f"Saving the checkpoint for step {result.step} failed ({result.error!r}); {consequence}"
            if raise_errors:
                raise RuntimeError(message) from result.error
            print_acc(message)
            return
        print_acc(f"Saved checkpoint to {result.lora_path} (written in the background in {result.seconds:.0f}s)")
        try:
            # only now: never remove an old save before the new one is on disk; the end-of-training
            # save (step None) also supersedes any stop-save
            self.clean_up_saves(remove_stop_saves=result.step is None)
        except Exception as e:
            print_acc(f"Could not remove old saves: {e}")
        self.post_save_hook(result.lora_path)
        self.hook_background_save_finished(result)

    def _wait_for_background_save(self, timeout=None, raise_errors=True):
        writer = getattr(self, "_checkpoint_writer", None)
        if writer is None:
            return
        result = writer.wait(timeout)
        if result is not None:
            self._report_background_save(result, raise_errors=raise_errors)

    def _poll_background_save(self):
        """Once per step: report a checkpoint that has finished writing, then remove old saves."""
        writer = getattr(self, "_checkpoint_writer", None)
        result = writer.poll() if writer is not None else None
        if result is None:
            return
        pause = self.progress_bar is not None and not self.progress_bar.paused
        if pause:
            self.progress_bar.pause()
        try:
            self._report_background_save(result)
        finally:
            if pause:
                self.progress_bar.unpause()

    def _wait_for_background_save_on_exit(self):
        writer = getattr(self, "_checkpoint_writer", None)
        if writer is None:
            return
        if writer.busy:
            step = writer.pending_step
            print_acc(f"Waiting for the checkpoint of step {step} to finish writing before exiting...")
            try:
                self.hook_waiting_for_background_save(step)
            except Exception:
                pass
        try:
            timeout = float(os.environ.get("AI_TOOLKIT_SAVE_EXIT_WAIT_S", "1200"))
        except ValueError:
            timeout = 1200.0
        self._wait_for_background_save(timeout=timeout, raise_errors=False)
        if writer.busy:
            print_acc("Gave up waiting for the checkpoint being written; the previous one is intact")

    def cleanup_runtime_caches(self):
        try:
            flush_buffered_file_item_latents_to_disk(force=True)
        finally:
            clear_buffered_file_item_latents_to_disk()

        sd = getattr(self, 'sd', None)
        if sd is not None:
            if hasattr(sd, 'remove_after_sample_image_hook'):
                sd.remove_after_sample_image_hook(self.sample_step_hook)
            if hasattr(self, 'status_update_hook_func') and hasattr(sd, 'remove_status_update_hook'):
                sd.remove_status_update_hook(self.status_update_hook_func)
            if hasattr(sd, 'clear_sample_prompts_cache'):
                sd.clear_sample_prompts_cache()
            elif hasattr(sd, 'sample_prompts_cache'):
                sd.sample_prompts_cache = None

        prompt_cache = getattr(self, 'prompt_cache', None)
        if prompt_cache is not None and hasattr(prompt_cache, 'clear'):
            prompt_cache.clear()
        if hasattr(self, 'prompt_cache'):
            self.prompt_cache = None

        for attr_name, empty_value in (
            ('prompt_pairs', []),
            ('anchor_pairs', []),
            ('dataset_prompts', []),
            ('prompt_txt_list', None),
        ):
            value = getattr(self, attr_name, None)
            if isinstance(value, list):
                value.clear()
                setattr(self, attr_name, empty_value.copy() if isinstance(empty_value, list) else empty_value)
            elif hasattr(self, attr_name):
                setattr(self, attr_name, empty_value)

        if hasattr(self, 'handle_timing_print_hook') and hasattr(self.timer, 'remove_after_print_hook'):
            self.timer.remove_after_print_hook(self.handle_timing_print_hook)

    def _get_dataloader_resume_state_path(self):
        return os.path.join(self.save_root, 'dataloader_state.pt')

    def _capture_rng_resume_state(self):
        state = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state['cuda_random_state_all'] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    def _restore_rng_resume_state(self, state):
        if not state:
            return
        python_random_state = state.get('python_random_state')
        if python_random_state is not None:
            random.setstate(python_random_state)
        numpy_random_state = state.get('numpy_random_state')
        if numpy_random_state is not None:
            np.random.set_state(numpy_random_state)
        torch_random_state = state.get('torch_random_state')
        if torch_random_state is not None:
            torch.random.set_rng_state(torch_random_state)
        cuda_random_state_all = state.get('cuda_random_state_all')
        if cuda_random_state_all is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_random_state_all)
            except Exception:
                pass

    def _save_dataloader_resume_state(self, checkpoint_step: int):
        if not self.accelerator.is_main_process:
            return
        torch.save(self._build_dataloader_resume_state(checkpoint_step), self._get_dataloader_resume_state_path())

    def _build_dataloader_resume_state(self, checkpoint_step: int):
        # fresh copies of the sampler position and RNG states: taken at the save point, they
        # stay valid however long the write takes
        return {
            'version': 1,
            'checkpoint_step': checkpoint_step,
            'next_step': checkpoint_step + 1,
            'process_epoch_num': self.epoch_num,
            'train_loader': get_dataloader_resume_state(getattr(self, 'data_loader', None)),
            'reg_loader': get_dataloader_resume_state(getattr(self, 'data_loader_reg', None)),
            'rng_state': self._capture_rng_resume_state(),
        }

    def _load_saved_dataloader_resume_state(self):
        path = self._get_dataloader_resume_state_path()
        if not os.path.exists(path):
            return None
        try:
            state = torch.load(path, weights_only=False)
        except Exception as e:
            print_acc(f"Failed to load dataloader resume state from {path}")
            print_acc(e)
            return None
        if not isinstance(state, dict):
            return None
        return state

    def _apply_loaded_dataloader_resume_state(self, state):
        if not state:
            return False

        checkpoint_step = state.get('checkpoint_step')
        if checkpoint_step is not None and checkpoint_step != self.start_step:
            return False

        restored = False
        train_state = state.get('train_loader')
        if train_state and apply_dataloader_resume_state(getattr(self, 'data_loader', None), train_state):
            restored = True

        reg_state = state.get('reg_loader')
        if reg_state and apply_dataloader_resume_state(getattr(self, 'data_loader_reg', None), reg_state):
            restored = True

        if not restored:
            return False

        next_step = state.get('next_step')
        if next_step is not None:
            self.step_num = int(next_step)
            self.start_step = self.step_num
            self._resumed_at_new_step = True
        process_epoch_num = state.get('process_epoch_num')
        if process_epoch_num is not None:
            self.epoch_num = int(process_epoch_num)
        self._restore_rng_resume_state(state.get('rng_state'))
        print_acc(f"Restored dataloader resume state from {self._get_dataloader_resume_state_path()}")
        return True

    def _free_memory_on_startup(self):
        self.cleanup_runtime_caches()

        for attr_name in ('data_loader', 'data_loader_reg', 'progress_bar'):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        flush()

        if self.device_torch.type == 'cuda' and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats(self.device_torch)
            except Exception:
                pass

        print_acc("Freed startup memory state")

    def _iter_dataset_cache_roots(self):
        seen = set()
        for dataset in self.dataset_configs:
            for candidate in (
                getattr(dataset, 'dataset_path', None),
                getattr(dataset, 'folder_path', None),
                getattr(dataset, 'clip_image_path', None),
            ):
                if not candidate:
                    continue
                if not os.path.exists(candidate):
                    continue
                for root in get_cache_root_candidates(
                        candidate,
                        override_root=getattr(dataset, 'cache_dir_override', None),
                ):
                    if not root:
                        continue
                    normalized_root = os.path.normcase(os.path.normpath(root))
                    if normalized_root in seen:
                        continue
                    seen.add(normalized_root)
                    yield root

    def _cleanup_all_caches_on_startup(self):
        if not self.cleanup_caches_on_startup:
            return

        cache_dir_names = {
            '_latent_cache',
            '_t_e_cache',
            '_clip_vision_cache',
        }
        removed_dirs = 0
        removed_files = 0

        # NOTE: .aitk_size.json (the image-dimension cache) is deliberately NOT
        # deleted here. Its per-file key is "size:mtime", so it is SELF-INVALIDATING:
        # an edited or replaced image misses on its own and is re-read, while
        # unchanged images stay valid forever. Wiping it can therefore never fix a
        # correctness problem -- it only forces a full re-scan of every image on the
        # next run (116k images here). The latent / text-embed / clip caches below
        # ARE wiped, because those legitimately need clearing when switching models.
        # To force a dimension re-scan anyway, delete the file by hand.

        for root in self._iter_dataset_cache_roots():
            for current_root, dirnames, _ in os.walk(root, topdown=True):
                cache_dirs = [d for d in dirnames if d in cache_dir_names]
                for cache_dir_name in cache_dirs:
                    cache_dir_path = os.path.join(current_root, cache_dir_name)
                    try:
                        shutil.rmtree(cache_dir_path)
                        removed_dirs += 1
                    except Exception as e:
                        print_acc(f"Failed to remove cache directory {cache_dir_path}: {e}")
                dirnames[:] = [d for d in dirnames if d not in cache_dir_names]

        print_acc(
            f"Startup cache cleanup removed {removed_dirs} cache directories and {removed_files} cache files"
        )

    def _compact_latent_shards_on_startup(self):
        if getattr(self.train_config, 'disable_latent_cache_compaction', False):
            print_acc("Latent shard compaction disabled (train.disable_latent_cache_compaction=true); skipping.")
            return
        seen_dirs = set()
        compacted_dirs = 0
        total_moved_items = 0
        total_removed_shards = 0
        total_created_shards = 0

        for dataset in self.dataset_configs:
            if not getattr(dataset, 'cache_latents_to_disk', False):
                continue

            shard_size = max(1, int(getattr(dataset, 'latent_cache_shard_size', 1)))
            if shard_size <= 1:
                continue

            source_path = getattr(dataset, 'folder_path', None) or getattr(dataset, 'dataset_path', None)
            if not source_path:
                continue

            for latent_dir in get_cache_dir_candidates(
                    source_path,
                    '_latent_cache',
                    override_root=getattr(dataset, 'cache_dir_override', None),
                    min_free_gb=getattr(dataset, 'cache_dir_override_min_free_gb', None),
            ):
                normalized_dir = os.path.normcase(os.path.normpath(latent_dir))
                if normalized_dir in seen_dirs:
                    continue
                seen_dirs.add(normalized_dir)

                try:
                    result = compact_latent_shards(latent_dir, shard_size)
                except Exception as e:
                    print_acc(
                        f"Warning: latent shard compaction failed for {latent_dir}; "
                        f"skipping it (cache still usable): {e}"
                    )
                    continue
                if not result.get("compacted"):
                    continue

                compacted_dirs += 1
                total_moved_items += result.get("moved_items", 0)
                total_removed_shards += result.get("removed_shards", 0)
                total_created_shards += result.get("created_shards", 0)
                print_acc(
                    f"Compacted latent shards in {latent_dir}: "
                    f"moved {result.get('moved_items', 0)} items, "
                    f"replaced {result.get('removed_shards', 0)} underfilled shards "
                    f"with {result.get('created_shards', 0)} grouped shards"
                )

        if compacted_dirs > 0:
            print_acc(
                f"Latent shard compaction complete across {compacted_dirs} cache directories: "
                f"moved {total_moved_items} items, removed {total_removed_shards} underfilled shards, "
                f"created {total_created_shards} grouped shards"
            )

    @staticmethod
    def _pick_auto_low_vram(free_gib: float, total_gib: float, layer_offloading: bool):
        free_ratio = free_gib / max(total_gib, 1.0)

        if layer_offloading:
            if free_gib >= 14 and free_ratio >= 0.82:
                return False
            if free_gib >= 12 and free_ratio >= 0.9:
                return False
            return True

        if free_gib >= 18 or (free_gib >= 14 and free_ratio >= 0.9):
            return False
        return True

    def _resolve_auto_low_vram(self, model_config: ModelConfig):
        if not getattr(model_config, "auto_low_vram", False):
            return

        if self.device_torch.type != 'cuda' or not torch.cuda.is_available():
            print_acc(
                "Auto low_vram requested, but CUDA VRAM is unavailable. Keeping configured low_vram setting."
            )
            return

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_torch)
        except Exception as e:
            print_acc(
                f"Auto low_vram requested, but VRAM query failed: {e}. Keeping configured low_vram setting."
            )
            return

        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        resolved_low_vram = self._pick_auto_low_vram(
            free_gib,
            total_gib,
            getattr(model_config, "layer_offloading", False),
        )

        model_config.low_vram = resolved_low_vram
        self.model_config.low_vram = resolved_low_vram

        print_acc(
            "Auto low_vram resolved from "
            f"{free_gib:.1f} / {total_gib:.1f} GiB free VRAM: "
            f"{'enabled' if resolved_low_vram else 'disabled'}"
        )

    @staticmethod
    def _pick_auto_layer_offloading_percents(free_gib: float, total_gib: float):
        free_ratio = free_gib / max(total_gib, 1.0)

        if free_gib >= 24 or (free_gib >= 18 and free_ratio >= 0.8):
            return 0.05, 0.0
        if free_gib >= 18 or (free_gib >= 14 and free_ratio >= 0.65):
            return 0.15, 0.0
        if free_gib >= 14 or (free_gib >= 12 and free_ratio >= 0.55):
            return 0.35, 0.0
        if free_gib >= 12:
            return 0.5, 0.1
        if free_gib >= 10:
            return 0.65, 0.25
        if free_gib >= 8:
            return 0.8, 0.5
        if free_gib >= 6:
            return 0.9, 0.75
        return 1.0, 1.0

    def _resolve_auto_layer_offloading_percents(self, model_config: ModelConfig):
        if not model_config.layer_offloading or not model_config.auto_layer_offloading_percent:
            return

        if self.device_torch.type != 'cuda' or not torch.cuda.is_available():
            print_acc(
                "Auto layer offloading % requested, but CUDA VRAM is unavailable. Keeping configured percentages."
            )
            return

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_torch)
        except Exception as e:
            print_acc(
                f"Auto layer offloading % requested, but VRAM query failed: {e}. Keeping configured percentages."
            )
            return

        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        transformer_percent, text_encoder_percent = self._pick_auto_layer_offloading_percents(free_gib, total_gib)

        prev_transformer = model_config.layer_offloading_transformer_percent
        prev_text_encoder = model_config.layer_offloading_text_encoder_percent

        model_config.layer_offloading_transformer_percent = transformer_percent
        model_config.layer_offloading_text_encoder_percent = text_encoder_percent
        self.model_config.layer_offloading_transformer_percent = transformer_percent
        self.model_config.layer_offloading_text_encoder_percent = text_encoder_percent

        print_acc(
            "Auto layer offloading resolved from "
            f"{free_gib:.1f} / {total_gib:.1f} GiB free VRAM: "
            f"transformer={transformer_percent * 100:.0f}%, "
            f"text_encoder={text_encoder_percent * 100:.0f}%"
        )

        if transformer_percent < prev_transformer or text_encoder_percent < prev_text_encoder:
            print_acc(
                "  WARNING: auto REDUCED your configured layer offloading "
                f"(transformer {prev_transformer * 100:.0f}% -> {transformer_percent * 100:.0f}%, "
                f"text_encoder {prev_text_encoder * 100:.0f}% -> {text_encoder_percent * 100:.0f}%)."
                "  This heuristic reads FREE VRAM AT STARTUP, which on an idle GPU is close to total"
                " capacity; it does NOT know how large the model is. If the checkpoint does not fit in"
                " VRAM this will overcommit, and Windows silently spills to system RAM: GPU pinned at"
                " 100% but low wattage, steps 10-100x slower, or a hang at step 0."
                "  Set auto_layer_offloading_percent: false to honor your configured percentages."
            )
    
    def done_hook(self):
        pass
    
    def end_step_hook(self):
        pass

    def _get_network_state_stats(self, network: Optional[Network]):
        if network is None:
            return {
                "total_tensors": 0,
                "nonzero_tensors": 0,
                "total_abs_sum": 0.0,
            }

        total_tensors = 0
        nonzero_tensors = 0
        total_abs_sum = 0.0
        # lora_up (lora_B) starts at zero: if it is still all zero after training, the LoRA does
        # nothing (lora_A alone keeps its random init, so the plain count above cannot tell)
        up_tensors = 0
        nonzero_up_tensors = 0

        with torch.no_grad():
            for key, value in network.state_dict().items():
                if not isinstance(value, torch.Tensor):
                    continue
                total_tensors += 1
                abs_sum = float(value.detach().float().abs().sum().item())
                total_abs_sum += abs_sum
                if abs_sum > 0.0:
                    nonzero_tensors += 1
                if "lora_up" in key or "lora_B" in key:
                    up_tensors += 1
                    if abs_sum > 0.0:
                        nonzero_up_tensors += 1

        return {
            "total_tensors": total_tensors,
            "nonzero_tensors": nonzero_tensors,
            "total_abs_sum": total_abs_sum,
            "up_tensors": up_tensors,
            "nonzero_up_tensors": nonzero_up_tensors,
        }

    def _log_network_state_stats(self, label: str, network: Optional[Network]):
        stats = self._get_network_state_stats(network)
        print_acc(
            f"{label}: "
            f"{stats['nonzero_tensors']}/{stats['total_tensors']} nonzero tensors, "
            f"abs_sum={stats['total_abs_sum']:.4f}"
        )
        return stats

    def _get_network_for_save(self) -> Optional[Network]:
        candidates: List[Network] = []

        if self.network is not None:
            candidates.append(self.network)

        if getattr(self.sd, "network", None) is not None:
            prepared_network = unwrap_model(self.sd.network)
            if prepared_network is not None and all(
                prepared_network is not candidate for candidate in candidates
            ):
                candidates.append(prepared_network)

        if len(candidates) == 0:
            return None

        ranked_candidates = []
        for candidate in candidates:
            stats = self._get_network_state_stats(candidate)
            ranked_candidates.append((candidate, stats))

        chosen_network, chosen_stats = max(
            ranked_candidates,
            key=lambda entry: (
                entry[1]["nonzero_tensors"],
                entry[1]["total_abs_sum"],
            ),
        )

        if self.network is not chosen_network:
            print_acc(
                "Using prepared network instance for save "
                f"({chosen_stats['nonzero_tensors']}/{chosen_stats['total_tensors']} nonzero tensors)"
            )
            self.network = chosen_network

        up_note = ""
        if chosen_stats.get("up_tensors"):
            up_note = f", lora_up nonzero {chosen_stats['nonzero_up_tensors']}/{chosen_stats['up_tensors']}"
        print_acc(
            "LoRA save stats: "
            f"{chosen_stats['nonzero_tensors']}/{chosen_stats['total_tensors']} nonzero tensors, "
            f"abs_sum={chosen_stats['total_abs_sum']:.4f}{up_note}"
        )

        if chosen_stats["nonzero_tensors"] == 0 and chosen_stats["total_tensors"] > 0:
            print_acc(
                "WARNING: all save-side LoRA tensors are zero before checkpoint write."
            )
        elif (chosen_stats.get("up_tensors") and chosen_stats["nonzero_up_tensors"] == 0
              and self.step_num > self.start_step):
            print_acc(
                "WARNING: every lora_up (lora_B) tensor is still exactly zero after training, so this "
                "LoRA has no effect: no optimizer step changed it. Check the log for skipped steps "
                "(NaN/inf or zero gradients)."
            )

        return chosen_network

    def _get_live_network(self) -> Optional[Network]:
        if getattr(self.sd, "network", None) is not None:
            live_network = unwrap_model(self.sd.network)
            if live_network is not None:
                return live_network
        return self.network

    def _get_live_trainable_network_params(self) -> List[torch.nn.Parameter]:
        live_network = self._get_live_network()
        if live_network is None:
            return []

        params: List[torch.nn.Parameter] = []
        seen_ids = set()
        for module in live_network.get_all_modules():
            for param in self._iter_module_parameters_for_probe(module):
                if not isinstance(param, torch.nn.Parameter):
                    continue
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen_ids:
                    continue
                seen_ids.add(param_id)
                params.append(param)
        return params

    def _iter_module_parameters_for_probe(self, module):
        yield from self._iter_module_parameters_for_probe_safe(module, set())

    def _iter_module_parameters_for_probe_safe(self, module, seen_modules):
        module_id = id(module)
        if module_id in seen_modules:
            return
        seen_modules.add(module_id)

        raw_parameters = getattr(module, "_parameters", None)
        if isinstance(raw_parameters, dict):
            for param in raw_parameters.values():
                if isinstance(param, torch.nn.Parameter):
                    yield param

        raw_modules = getattr(module, "_modules", None)
        if not isinstance(raw_modules, dict):
            return

        for child_name, child_module in raw_modules.items():
            if child_name == "org_module":
                continue
            if child_module is None:
                continue
            yield from self._iter_module_parameters_for_probe_safe(child_module, seen_modules)

    def _validate_optimizer_tracks_live_network(self):
        if self.optimizer is None:
            return

        live_params = self._get_live_trainable_network_params()
        if len(live_params) == 0:
            return

        optimizer_param_ids = set()
        for group in self.optimizer.param_groups:
            for param in group.get("params", []):
                if isinstance(param, torch.nn.Parameter):
                    optimizer_param_ids.add(id(param))

        tracked_params = [
            param for param in live_params if id(param) in optimizer_param_ids
        ]

        print_acc(
            "Optimizer coverage: "
            f"{len(tracked_params)}/{len(live_params)} live trainable LoRA params tracked"
        )

        if len(tracked_params) == 0:
            raise RuntimeError(
                "Optimizer is not tracking any live trainable LoRA parameters."
            )

        if len(tracked_params) != len(live_params):
            print_acc(
                "WARNING: optimizer is not tracking every live trainable LoRA parameter."
            )

    def _get_current_optimizer_lr(self):
        # Highest LR across param groups. The LR scheduler writes the scheduled LR
        # into param_groups['lr'], so warmup-from-0 / zero-LR steps read as 0.0 here.
        if self.optimizer is None:
            return 0.0
        lrs = []
        for group in self.optimizer.param_groups:
            lr = group.get("lr", None)
            if lr is None:
                continue
            try:
                lrs.append(float(lr))
            except (TypeError, ValueError):
                continue
        if len(lrs) == 0:
            return 0.0
        return max(lrs)

    def capture_optimizer_update_probe(self):
        live_params = self._get_live_trainable_network_params()
        if len(live_params) == 0:
            return None

        nonzero_grad_params = 0
        nonfinite_grad_params = 0
        total_grad_abs_sum = 0.0
        sampled_params = []

        for param in live_params:
            grad = param.grad
            if grad is None:
                continue

            grad_abs_sum = float(grad.detach().float().abs().sum().item())
            # Every comparison with NaN is False, so `<= 0.0` used to wave NaN
            # gradients through as "nonzero". A step full of NaN grads then looked
            # like an optimizer that silently refused to update, while it was in
            # fact writing NaN into the LoRA. Count them separately.
            if not math.isfinite(grad_abs_sum):
                nonfinite_grad_params += 1
                continue
            if grad_abs_sum <= 0.0:
                continue

            nonzero_grad_params += 1
            total_grad_abs_sum += grad_abs_sum

            if len(sampled_params) < self._optimizer_update_probe_limit:
                sampled_params.append((param, param.detach().clone()))

        return {
            "live_param_count": len(live_params),
            "nonzero_grad_params": nonzero_grad_params,
            "nonfinite_grad_params": nonfinite_grad_params,
            "total_grad_abs_sum": total_grad_abs_sum,
            "sampled_params": sampled_params,
        }

    def should_skip_nonfinite_optimizer_step(self, probe) -> bool:
        # One backward pass with NaN/inf gradients is enough to turn every LoRA
        # weight it touches into NaN, permanently: every later forward is NaN, and
        # the next checkpoint save writes the poisoned weights to disk. Skip that
        # optimizer step instead (the same thing a GradScaler does) and keep the
        # weights as they were. A run that keeps producing them is broken, so stop
        # it rather than burn hours of "loss is nan".
        count = probe.get("nonfinite_grad_params", 0) if probe else 0
        if count == 0:
            self._consecutive_nonfinite_grad_steps = 0
            return False
        self._consecutive_nonfinite_grad_steps += 1
        limit = self._optimizer_update_check_max_consecutive
        print_acc(
            f"WARNING: skipped optimizer step -- {count}/{probe['live_param_count']} "
            f"trainable LoRA params have NaN/inf gradients; the LoRA weights were left "
            f"untouched ({self._consecutive_nonfinite_grad_steps}/{limit} consecutive)."
        )
        if self._consecutive_nonfinite_grad_steps >= limit:
            raise RuntimeError(
                f"NaN/inf gradients on {self._consecutive_nonfinite_grad_steps} consecutive "
                "optimizer steps; stopping before a checkpoint is written. The LoRA "
                "weights were never updated with them."
            )
        return True

    def finalize_optimizer_update_probe(self, probe):
        if probe is None:
            return

        nonzero_grad_params = probe["nonzero_grad_params"]
        if nonzero_grad_params == 0:
            self._consecutive_no_optimizer_grads += 1
            if self._consecutive_no_optimizer_grads in (1, 5):
                print_acc(
                    "WARNING: optimizer step had no nonzero LoRA gradients "
                    f"({self._consecutive_no_optimizer_grads} consecutive step(s))."
                )
            # a whole run of steps without a single nonzero gradient trains nothing; stop instead
            # of producing a LoRA that silently does nothing
            if (self._optimizer_update_check_enabled and self._get_current_optimizer_lr() > 0.0
                    and self._consecutive_no_optimizer_grads >= self._optimizer_update_check_max_consecutive):
                raise RuntimeError(
                    f"No nonzero LoRA gradients for {self._consecutive_no_optimizer_grads} consecutive "
                    "steps: the LoRA is not being trained. If this is expected, set "
                    "train.disable_optimizer_update_check: true."
                )
            return

        self._consecutive_no_optimizer_grads = 0

        changed_params = 0
        nonfinite_params = 0
        total_delta_abs_sum = 0.0
        for param, before_value in probe["sampled_params"]:
            delta_abs_sum = float(
                (param.detach().float() - before_value.detach().float()).abs().sum().item()
            )
            # `> 0.0` is False for NaN, so weights that just became NaN used to be
            # reported as "did not change".
            if not math.isfinite(delta_abs_sum):
                nonfinite_params += 1
                continue
            total_delta_abs_sum += delta_abs_sum
            if delta_abs_sum > 0.0:
                changed_params += 1

        if nonfinite_params > 0:
            raise RuntimeError(
                f"optimizer.step() produced NaN/inf in {nonfinite_params}/"
                f"{len(probe['sampled_params'])} sampled LoRA weights from finite "
                "gradients. The in-memory LoRA is corrupt; stopping before it is saved."
            )

        if changed_params == 0:
            # Don't penalize warmup / zero-LR steps (and adaptive optimizers whose
            # effective step can start ~0): those legitimately produce no weight
            # delta. Only a sustained run of real (LR>0) steps with no movement is
            # a genuine "optimizer not wired to params" failure.
            current_lr = self._get_current_optimizer_lr()
            if not self._optimizer_update_check_enabled or current_lr <= 0.0:
                return
            self._consecutive_no_optimizer_updates += 1
            print_acc(
                "WARNING: optimizer.step() did not change any sampled LoRA weights "
                f"despite {nonzero_grad_params} params having nonzero gradients "
                f"(lr={current_lr:.2e}, {self._consecutive_no_optimizer_updates}/"
                f"{self._optimizer_update_check_max_consecutive} consecutive)."
            )
            if self._consecutive_no_optimizer_updates >= self._optimizer_update_check_max_consecutive:
                raise RuntimeError(
                    "Optimizer is not updating sampled LoRA weights after "
                    f"{self._consecutive_no_optimizer_updates} consecutive steps. "
                    "If this is a false positive (e.g. Prodigy/DAdapt or a very long "
                    "warmup), set train.disable_optimizer_update_check: true."
                )
            return

        if not self._did_log_optimizer_update:
            print_acc(
                "Optimizer update confirmed: "
                f"{changed_params}/{len(probe['sampled_params'])} sampled LoRA params changed, "
                f"grad_abs_sum={probe['total_grad_abs_sum']:.4f}, "
                f"delta_abs_sum={total_delta_abs_sum:.4f}"
            )
            self._did_log_optimizer_update = True

        self._consecutive_no_optimizer_updates = 0

    def save(self, step=None):
        if not self.accelerator.is_main_process:
            return
        # one checkpoint at a time: finish (and report) the one still being written, if any
        self._wait_for_background_save()
        self.last_save_in_background = False
        log_memory_breakdown(getattr(self, "sd", None), getattr(self, "network", None),
                             label=f"at step {self.step_num}", printer=print_acc)
        flush_buffered_file_item_latents_to_disk(force=True)
        flush()
        if self.ema is not None:
            # always save params as ema
            self.ema.eval()

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            self.last_save_step = step
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        self.update_training_metadata()
        filename = f'{self.job.name}{step_num}.safetensors'
        file_path = os.path.join(self.save_root, filename)

        save_meta = copy.deepcopy(self.meta)
        # get extra meta
        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            additional_save_meta = self.adapter.get_additional_save_metadata()
            if additional_save_meta is not None:
                for key, value in additional_save_meta.items():
                    save_meta[key] = value

        # prepare meta
        save_meta = get_meta_for_safetensors(save_meta, self.job.name)
        if not self.is_fine_tuning and not self.train_config.merge_network_on_save:
            network_to_save = self._get_network_for_save()
            if network_to_save is not None:
                lora_name = self.job.name
                if self.named_lora:
                    # add _lora to name
                    lora_name += '_LoRA'

                filename = f'{lora_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                if self._can_save_in_background(network_to_save):
                    self._save_in_background(step, network_to_save, file_path, save_meta)
                    return
                prev_multiplier = network_to_save.multiplier
                network_to_save.multiplier = 1.0

                # if we are doing embedding training as well, add that
                embedding_dict = self.embedding.state_dict() if self.embedding else None
                network_to_save.save_weights(
                    file_path,
                    dtype=get_torch_dtype(self.save_config.dtype),
                    metadata=save_meta,
                    extra_state_dict=embedding_dict
                )
                network_to_save.multiplier = prev_multiplier
                # if we have an embedding as well, pair it with the network

            # even if added to lora, still save the trigger version
            if self.embedding is not None:
                emb_filename = f'{self.embed_config.trigger}{step_num}.safetensors'
                emb_file_path = os.path.join(self.save_root, emb_filename)
                # for combo, above will get it
                # set current step
                self.embedding.step = self.step_num
                # change filename to pt if that is set
                if self.embed_config.save_format == "pt":
                    # replace extension
                    emb_file_path = os.path.splitext(emb_file_path)[0] + ".pt"
                self.embedding.save(emb_file_path)
            
            if self.decorator is not None:
                dec_filename = f'{self.job.name}{step_num}.safetensors'
                dec_file_path = os.path.join(self.save_root, dec_filename)
                decorator_state_dict = self.decorator.state_dict()
                for key, value in decorator_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        decorator_state_dict[key] = value.clone().to('cpu', dtype=get_torch_dtype(self.save_config.dtype))
                save_file(
                    decorator_state_dict,
                    dec_file_path,
                    metadata=save_meta,
                )

            if self.adapter is not None and self.adapter_config.train:
                adapter_name = self.job.name
                if self.network_config is not None or self.embedding is not None:
                    # add _lora to name
                    if self.adapter_config.type == 't2i':
                        adapter_name += '_t2i'
                    elif self.adapter_config.type == 'control_net':
                        adapter_name += '_cn'
                    elif self.adapter_config.type == 'clip':
                        adapter_name += '_clip'
                    elif self.adapter_config.type.startswith('ip'):
                        adapter_name += '_ip'
                    else:
                        adapter_name += '_adapter'

                filename = f'{adapter_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                # save adapter
                state_dict = self.adapter.state_dict()
                if self.adapter_config.type == 't2i':
                    save_t2i_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype)
                    )
                elif self.adapter_config.type == 'control_net':
                    # save in diffusers format
                    name_or_path = file_path.replace('.safetensors', '')
                    # move it to the new dtype and cpu
                    orig_device = self.adapter.device
                    orig_dtype = self.adapter.dtype
                    self.adapter = self.adapter.to(torch.device('cpu'), dtype=get_torch_dtype(self.save_config.dtype))
                    self.adapter.save_pretrained(
                        name_or_path,
                        dtype=get_torch_dtype(self.save_config.dtype),
                        safe_serialization=True
                    )
                    meta_path = os.path.join(name_or_path, 'aitk_meta.yaml')
                    with open(meta_path, 'w') as f:
                        yaml.dump(self.meta, f)
                    # move it back
                    self.adapter = self.adapter.to(orig_device, dtype=orig_dtype)
                else:
                    direct_save = False
                    if self.adapter_config.train_only_image_encoder:
                        direct_save = True
                    elif isinstance(self.adapter, CustomAdapter):
                        direct_save = self.adapter.do_direct_save
                    save_ip_adapter_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype),
                        direct_save=direct_save
                    )
        else:
            if self.network is not None and self.train_config.merge_network_on_save:
                # merge the network weights into a full model and save that
                if not self.network.can_merge_in:
                    raise ValueError("Network cannot merge in weights. Cannot save full model.")
                
                print_acc("Merging network weights into full model for saving...")
                
                self.network.merge_in(merge_weight=1.0)
                # reset weights to zero
                self.network.reset_weights()
                self.network.is_merged_in = False
                
                print_acc("Done merging network weights.")
                
            if self.save_config.save_format == "diffusers":
                # saving as a folder path
                file_path = file_path.replace('.safetensors', '')
                # convert it back to normal object
                save_meta = parse_metadata_from_safetensors(save_meta)

            if self.sd.refiner_unet and self.train_config.train_refiner:
                # save refiner
                refiner_name = self.job.name + '_refiner'
                filename = f'{refiner_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                self.sd.save_refiner(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )
            if self.train_config.train_unet or self.train_config.train_text_encoder:
                self.sd.save(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )

        # save learnable params as json if we have thim
        if self.snr_gos:
            json_data = {
                'offset_1': self.snr_gos.offset_1.item(),
                'offset_2': self.snr_gos.offset_2.item(),
                'scale': self.snr_gos.scale.item(),
                'gamma': self.snr_gos.gamma.item(),
            }
            path_to_save = file_path = os.path.join(self.save_root, 'learnable_snr.json')
            with open(path_to_save, 'w') as f:
                json.dump(json_data, f, indent=4)
        
        print_acc(f"Saved checkpoint to {file_path}")

        # save optimizer
        if self.optimizer is not None:
            try:
                filename = f'optimizer.pt'
                file_path = os.path.join(self.save_root, filename)
                try:
                    state_dict = unwrap_model(self.optimizer).state_dict()
                except Exception as e:
                    state_dict = self.optimizer.state_dict()
                torch.save(state_dict, file_path)
                print_acc(f"Saved optimizer to {file_path}")
            except Exception as e:
                print_acc(e)
                print_acc("Could not save optimizer")

        if step is not None:
            try:
                self._save_dataloader_resume_state(step)
            except Exception as e:
                print_acc(f"Could not save dataloader resume state: {e}")

        self.clean_up_saves(remove_stop_saves=step is None)
        self.post_save_hook(file_path)

        if self.ema is not None:
            self.ema.train()
        flush()

    # Called before the model is loaded
    def hook_before_model_load(self):
        # override in subclass
        pass

    def hook_after_model_load(self):
        # override in subclass
        pass

    def hook_add_extra_train_params(self, params):
        # override in subclass
        return params

    # ---- OOM diagnostics --------------------------------------------------------
    # An OOM skip used to leave one line in the log ("skipping batch 1/3") and
    # nothing else: not which stage ran out, not how memory was split, not which
    # images were dropped. These record exactly that, as plain values only -- the
    # exception object is never kept, because its traceback pins every tensor of
    # the failed step (the same leak that broke the latent pre-pass retry).
    #
    # Outcomes written to skipped_oom.txt:
    #   skipped              OOM, no retry possible (e.g. the retry itself is the 2nd try)
    #   skipped after retry  OOM, cleaned up, retried once, OOM again -> images dropped
    #   rescued by retry     OOM, cleaned up, retried once, the step trained normally
    #   LoRA already updated OOM after optimizer.step(); not retried (would update twice)
    #   run stopped          the 4th failed step in a row; training aborts
    #   retried              the first attempt of a retried step (its own rows, never counted)

    SKIPPED_OUTCOMES = ("skipped", "skipped after retry")
    RESCUED_OUTCOME = "rescued by retry"
    # Retrying an OOM'd step is only safe when the trainer's hook_train_loop sets
    # self._oom_lora_updated before optimizer.step() -- otherwise an OOM after the update
    # would be retried and the same batch applied twice. Opt-in per trainer.
    supports_oom_retry = False

    @staticmethod
    def _oom_gib(match):
        if match is None:
            return None
        scale = {"bytes": 1 / 1024 ** 3, "KiB": 1 / 1024 ** 2, "MiB": 1 / 1024, "GiB": 1.0, "TiB": 1024.0}
        return float(match.group(1)) * scale[match.group(2)]

    def _vram_outside_pytorch_gib(self):
        # Memory on this GPU not held by this process's PyTorch allocator: the CUDA
        # context plus every other program using the card (desktop, browser, ...).
        try:
            if not torch.cuda.is_available() or self.device_torch.type != "cuda":
                return None
            free, total = torch.cuda.mem_get_info(self.device_torch)
            return (total - free - torch.cuda.memory_reserved(self.device_torch)) / 1024 ** 3
        except Exception:
            return None

    def _vram_total_gib(self):
        try:
            if not torch.cuda.is_available() or self.device_torch.type != "cuda":
                return None
            return torch.cuda.mem_get_info(self.device_torch)[1] / 1024 ** 3
        except Exception:
            return None

    def _describe_oom(self, e: BaseException) -> dict:
        # Call INSIDE the except block: it reads the traceback, the memory state at the
        # moment of failure and this attempt's step bookkeeping, and returns only plain
        # values -- so a later retry cannot overwrite what this attempt saw.
        info = {"message": "", "stage": "unknown", "where": "", "tried_gib": None,
                "allocated_gib": None, "reserved_unused_gib": None, "outside_pytorch_gib": None,
                "device_total_gib": None,
                "failed_microbatch": getattr(self, "_current_microbatch_index", None),
                "phase": getattr(self, "_oom_phase", None),
                "lora_updated": bool(getattr(self, "_oom_lora_updated", False))}
        try:
            text = str(e).strip()
            msg = text.splitlines()[0] if text else type(e).__name__
            info["message"] = msg[:300]
            unit = r"([\d.]+) (bytes|KiB|MiB|GiB|TiB)"
            info["tried_gib"] = self._oom_gib(re.search(r"Tried to allocate " + unit, msg))
            info["allocated_gib"] = self._oom_gib(re.search(unit + r" is allocated by PyTorch", msg))
            info["reserved_unused_gib"] = self._oom_gib(
                re.search(unit + r" is reserved by PyTorch but unallocated", msg))

            frames = traceback.extract_tb(e.__traceback__)  # FrameSummary: no frame refs
            names = {f.name for f in frames}
            files = [f.filename.replace("\\", "/") for f in frames]
            bases = [fl.rsplit("/", 1)[-1].lower() for fl in files]
            if names & {"backward", "_engine_run_backward"}:
                stage = "backward pass"
                if any("torch/utils/checkpoint" in fl for fl in files):
                    stage += " (gradient-checkpoint recompute)"
            elif "clip_grad_norm_" in names or "clip_grad.py" in bases:
                stage = "gradient clipping"
            elif "sync_grad_transfers" in names:
                stage = "gradient sync (offloaded grads)"
            elif any("lr_scheduler" in fl for fl in files):
                stage = "LR scheduler step"
            elif any(("optim" in fl or "bitsandbytes" in fl) and f.name == "step"
                     for fl, f in zip(files, frames)):
                stage = "optimizer step"
            elif "update" in names and any(b.startswith("ema") for b in bases):
                stage = "EMA update"
            elif "encode_images" in names:
                stage = "latent encode (VAE)"
            elif names & {"encode_prompt", "get_prompt_embeds", "encode_prompt_embeds_for_file_items",
                          "encode_krea_prompt", "encode_krea_prompts_batched"}:
                stage = "text encode"
            elif names & {"predict_noise", "get_noise_prediction", "predict_velocity"}:
                stage = "forward pass"
            else:
                stage = "unknown"
            if names & {"_stage_forward_weight", "_stage_backward_weight", "_dequantize_staged"}:
                stage += ", while staging an offloaded weight"
            if stage.startswith("unknown") and info["phase"] and info["phase"] != "micro-batch":
                stage = f"unknown, {info['phase']}"
            info["stage"] = stage

            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            repo_norm = repo.replace("\\", "/")
            ours = [f for f, fl in zip(frames, files) if fl.startswith(repo_norm)]
            where = []
            if ours:
                f = ours[-1]
                where.append(f"{os.path.relpath(f.filename, repo)}:{f.lineno} {f.name}")
            if frames and (not ours or frames[-1] is not ours[-1]):
                f = frames[-1]
                where.append(f"in {os.path.basename(f.filename)}:{f.lineno} {f.name}")
            info["where"] = " | ".join(where)
        except Exception:
            pass
        info["outside_pytorch_gib"] = self._vram_outside_pytorch_gib()
        info["device_total_gib"] = self._vram_total_gib()
        return info

    def _caption_token_count(self, caption):
        if not caption:
            return None
        tok = getattr(self.sd, "tokenizer", None)
        if isinstance(tok, (list, tuple)):
            tok = tok[0] if len(tok) else None
        if tok is None:
            return None
        try:
            return len(tok(caption, add_special_tokens=False)["input_ids"])
        except Exception:
            return None

    def _oom_log_path(self):
        return os.path.join(self.save_root, "skipped_oom.txt")

    def _read_logged_oom_steps(self):
        # -> (skipped_steps, rescued_steps) across every skipped_oom*.txt in the job
        # folder (older layouts are rotated to skipped_oom.N.txt), each parsed by its own
        # header. Files from before the outcome column count as skipped unless marked
        # "[run stopped]". Malformed lines (e.g. half-written after a crash) are skipped.
        skipped, rescued = set(), set()
        try:
            names = [n for n in os.listdir(self.save_root)
                     if n.startswith("skipped_oom") and n.endswith(".txt")]
        except Exception:
            return skipped, rescued
        for name in names:
            try:
                with open(os.path.join(self.save_root, name), "r", encoding="utf-8", errors="replace") as f:
                    header = f.readline().rstrip("\n").split("\t")
                    step_col = header.index("step") if "step" in header else 1
                    out_col = header.index("outcome") if "outcome" in header else None
                    stage_col = header.index("stage") if "stage" in header else None
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) <= step_col:
                            continue
                        value = parts[step_col].strip().strip("\x00")
                        if not value.isdigit():
                            continue
                        if out_col is not None:
                            outcome = parts[out_col].strip() if len(parts) > out_col else ""
                            if outcome in self.SKIPPED_OUTCOMES:
                                skipped.add(int(value))
                            elif outcome == self.RESCUED_OUTCOME:
                                rescued.add(int(value))
                        else:
                            stage = parts[stage_col] if stage_col is not None and len(parts) > stage_col else ""
                            if not stage.endswith("[run stopped]"):
                                skipped.add(int(value))
            except Exception:
                continue
        return skipped, rescued

    def _seed_oom_tally(self):
        skipped, rescued = self._read_logged_oom_steps()
        resume_step = int(getattr(self, "step_num", 0) or 0)
        if resume_step > 0:
            # steps at or after the resume point are about to be trained again
            skipped = {st for st in skipped if st < resume_step}
            rescued = {st for st in rescued if st < resume_step}
        self._oom_skipped_steps, self._oom_rescued_steps = skipped, rescued
        self._oom_skip_total, self._oom_rescued_total = len(skipped), len(rescued)
        self._oom_last_step = max(skipped) if skipped else None

    def _record_oom_skip(self, info: dict, batch_list, outcome: str = "skipped", first_info=None) -> None:
        # info: the attempt to report (for "rescued by retry": the attempt that OOM'd);
        # first_info: the first attempt when the retry also failed.
        stopping = outcome == "run stopped"
        try:
            if self._oom_skipped_steps is None or self._oom_rescued_steps is None:
                self._seed_oom_tally()
            if outcome in self.SKIPPED_OUTCOMES:
                self._oom_skipped_steps.add(int(self.step_num))
                self._oom_last_step = int(self.step_num)
            elif outcome == self.RESCUED_OUTCOME:
                self._oom_rescued_steps.add(int(self.step_num))
            self._oom_skip_total = len(self._oom_skipped_steps)
            self._oom_rescued_total = len(self._oom_rescued_steps)

            failed = info.get("failed_microbatch", self._current_microbatch_index)
            phase = info.get("phase", self._oom_phase)
            lora_updated = bool(info.get("lora_updated", self._oom_lora_updated))
            batches = [b for b in (batch_list or []) if b is not None]
            rows = []
            for mb_index, batch in enumerate(batch_list or []):
                if batch is None:
                    continue
                for fi in getattr(batch, "file_items", None) or []:
                    w, h = getattr(fi, "crop_width", None), getattr(fi, "crop_height", None)
                    caption = getattr(fi, "caption", None) or ""
                    rows.append({
                        "microbatch": mb_index,
                        "bucket": f"{w}x{h}" if w and h else "",
                        "caption_tokens": self._caption_token_count(caption),
                        "caption_chars": len(caption),
                        "image": getattr(fi, "path", ""),
                    })

            stage = info.get("stage") or "unknown"

            def gib(x):
                return "?" if x is None else f"{x:.2f} GiB"

            def describe_failed(mb):
                chosen = [r for r in rows if r["microbatch"] == mb]
                return ", ".join(
                    f"{r['bucket'] or '?'}, caption "
                    f"{r['caption_tokens'] if r['caption_tokens'] is not None else '?'} tokens"
                    for r in chosen) or "?"

            start = self._vram_outside_pytorch_at_start
            provisional = bool(self._vram_baseline_provisional)
            tried, unused = info.get("tried_gib"), info.get("reserved_unused_gib")
            outside = info.get("outside_pytorch_gib")
            likely = self._oom_verdict(info)

            if stopping:
                headline = (f"OOM at step {self.step_num}: {self.num_consecutive_oom} failed steps in a row "
                            f"-- stopping the run")
            elif outcome == self.RESCUED_OUTCOME:
                headline = (f"OOM at step {self.step_num} RESCUED by the retry -- memory was freed and the "
                            f"step then trained normally")
            elif outcome == "LoRA already updated" or lora_updated:
                headline = (f"OOM at step {self.step_num} during or after the LoRA update -- not retried "
                            f"(that could apply this batch twice); the LR scheduler was not stepped")
            elif outcome == "skipped after retry":
                headline = (f"OOM: skipped step {self.step_num} -- the retry failed too; {len(rows)} "
                            f"image(s) dropped, no LoRA update")
            else:
                headline = (f"OOM: skipped step {self.step_num} -- {len(rows)} image(s) dropped, "
                            f"no LoRA update")

            print_acc("")
            print_acc("#" * 72)
            print_acc(f"# {headline}")
            print_acc(f"#   OOM skips so far: {self._oom_skip_total}   rescued by retry: {self._oom_rescued_total}"
                      f"   (failed steps in a row: {self.num_consecutive_oom}; the run stops at 4 in a row)")
            if first_info:
                print_acc(f"#   first attempt: {first_info.get('stage') or 'unknown'}"
                          + (f", micro-batch {first_info['failed_microbatch'] + 1}"
                             if first_info.get("failed_microbatch") is not None else "")
                          + f", tried {gib(first_info.get('tried_gib'))}"
                          + f" with {gib(first_info.get('reserved_unused_gib'))} reserved but unused")
                print_acc("#   retry (details below):")
            print_acc(f"#   stage: {stage}")
            if info.get("where"):
                print_acc(f"#   where: {info['where']}")
            if failed is not None:
                print_acc(f"#   failed micro-batch: {failed + 1} of {len(batches)} ({describe_failed(failed)})")
            elif batches:
                print_acc(f"#   failed micro-batch: none -- {phase or 'outside the micro-batch loop'}; "
                          f"no single image is to blame")
            print_acc(f"#   memory: tried to allocate {gib(tried)} | PyTorch using {gib(info.get('allocated_gib'))}"
                      f" + {gib(unused)} reserved but unused")
            print_acc(f"#           outside PyTorch (CUDA context + other programs): {gib(outside)}"
                      f" ({'before first step' if provisional else 'at start of training'}: {gib(start)})")
            print_acc(f"#   likely: {likely}")
            print_acc(f"#   details: {self._oom_log_path()}")
            print_acc("#" * 72)
            print_acc("")

            if self.accelerator.is_main_process:
                try:
                    if first_info:
                        self._append_oom_rows(
                            first_info, rows, first_info.get("stage") or "unknown",
                            first_info.get("failed_microbatch"), bool(first_info.get("lora_updated")),
                            self._oom_verdict(first_info), start, "retried")
                    self._append_oom_rows(info, rows, stage, failed, lora_updated, likely, start, outcome)
                except Exception as err:
                    print_acc(f"(could not write {self._oom_log_path()}: {err})")
        except Exception as err:
            # diagnostics must never be the thing that stops a training run
            print_acc(f"(could not record OOM details: {err})")
        finally:
            if not stopping:
                try:
                    self.hook_oom_tally_changed()
                except Exception:
                    pass

    def _oom_verdict(self, info: dict) -> str:
        # "likely": only claim a cause the numbers support
        tried, unused = info.get("tried_gib"), info.get("reserved_unused_gib")
        outside, start = info.get("outside_pytorch_gib"), self._vram_outside_pytorch_at_start
        total = info.get("device_total_gib")
        provisional = bool(self._vram_baseline_provisional)
        big_outside = outside is not None and outside > max(2.5, 0.15 * total if total else 0.0)
        growth_limit = 1.5 if provisional else 0.5
        if tried is not None and unused is not None and unused >= tried:
            likely = ("fragmentation: PyTorch had enough memory reserved, but not as one "
                      "usable block (split up, or cached for the offloader's transfer stream)")
        elif outside is not None and start is not None and outside - start > growth_limit:
            likely = (f"other programs / the driver took {outside - start:.2f} GiB more than "
                      + ("before the first step" if provisional else "at the start of training"))
        elif big_outside:
            likely = (f"{outside:.2f} GiB of VRAM is in use outside PyTorch -- another GPU "
                      "program (ComfyUI, a browser, a game) may be holding it")
        elif tried is None or unused is None:
            likely = "undetermined: the OOM message could not be parsed"
        elif outside is None or start is None:
            likely = "undetermined: no VRAM baseline was available"
        else:
            likely = "a genuine peak: this step needed more memory than the card has"
        if big_outside and not likely.startswith(f"{outside:.2f} GiB"):
            likely += f" (note: {outside:.2f} GiB is in use outside PyTorch)"
        return likely

    def _append_oom_rows(self, info, rows, stage, failed, lora_updated, likely, start, outcome):
        import datetime as _dt
        columns = ["time", "step", "outcome", "stage", "where", "failed_microbatch", "microbatch", "bucket",
                   "caption_tokens", "caption_chars", "lora_updated", "tried_gib", "pytorch_allocated_gib",
                   "reserved_unused_gib", "outside_pytorch_gib", "outside_pytorch_at_start_gib",
                   "likely", "message", "image"]
        header = "\t".join(columns)
        path = self._oom_log_path()
        if os.path.exists(path):
            # a file written with a different column layout is rotated aside, never appended to
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    existing = f.readline().rstrip("\n")
            except Exception:
                existing = header
            if existing != header:
                n = 1
                while os.path.exists(os.path.join(self.save_root, f"skipped_oom.{n}.txt")):
                    n += 1
                os.replace(path, os.path.join(self.save_root, f"skipped_oom.{n}.txt"))
        is_new = not os.path.exists(path)

        def cell(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")

        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(header + "\n")
            for r in rows or [{"microbatch": None, "bucket": "", "caption_tokens": None,
                               "caption_chars": None, "image": ""}]:
                record = {
                    "time": now, "step": self.step_num, "outcome": outcome, "stage": stage,
                    "where": info.get("where"),
                    "failed_microbatch": "yes" if failed is not None and r["microbatch"] == failed else "",
                    "microbatch": None if r["microbatch"] is None else r["microbatch"] + 1,
                    "bucket": r["bucket"], "caption_tokens": r["caption_tokens"],
                    "caption_chars": r["caption_chars"], "lora_updated": "yes" if lora_updated else "",
                    "tried_gib": info.get("tried_gib"), "pytorch_allocated_gib": info.get("allocated_gib"),
                    "reserved_unused_gib": info.get("reserved_unused_gib"),
                    "outside_pytorch_gib": info.get("outside_pytorch_gib"),
                    "outside_pytorch_at_start_gib": start, "likely": likely,
                    "message": info.get("message"), "image": r["image"],
                }
                f.write("\t".join(cell(record[c]) for c in columns) + "\n")

    def hook_oom_tally_changed(self):
        pass

    def hook_before_train_step(self, batch_list):
        """Called with this step's batches right before they are trained on."""
        pass

    def hook_before_forward_pass(self, microbatch_index, batch):
        """Called right before each micro-batch's forward pass (once per batch, GA times per step)."""
        pass

    def hook_after_train_step(self):
        """Called once the step's batches are done: trained, or skipped after an OOM."""
        pass

    def hook_before_train_loop(self):
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()

    def cache_progress_hook(
            self,
            cache_kind: str,
            processed: int,
            total: int,
            elapsed_seconds: float,
            dataset_path: str,
    ):
        pass
        
    def sample_step_hook(self, img_num, total_imgs):
        pass

    def _is_windows_dataloader_pickle_error(self, error: Exception) -> bool:
        if os.name != 'nt':
            return False
        import pickle
        if isinstance(error, (pickle.PickleError, EOFError, BrokenPipeError)):
            return True
        message = str(error).lower()
        return any(
            needle in message
            for needle in (
                "can't get local object",
                "cannot pickle",
                "picklingerror",
                "ran out of input",
                "broken pipe",
            )
        )

    def _clone_dataloader_with_num_workers(self, dataloader: DataLoader, num_workers: int) -> DataLoader:
        # Preserve ordering on the fallback clone. The real loaders use
        # ResumableShuffleSampler (a custom Sampler, NOT a RandomSampler), so the old
        # `isinstance(..., RandomSampler)` check silently produced shuffle=False →
        # sequential iteration. Reuse the live sampler whenever it isn't sequential.
        sampler = getattr(dataloader, 'sampler', None)
        reuse_sampler = (
            sampler
            if isinstance(sampler, Sampler) and not isinstance(sampler, SequentialSampler)
            else None
        )
        data_loader_kwargs = {
            'dataset': dataloader.dataset,
            'batch_size': dataloader.batch_size,
            'drop_last': dataloader.drop_last,
            'collate_fn': dataloader.collate_fn,
            'num_workers': num_workers,
            'pin_memory': getattr(dataloader, 'pin_memory', False),
        }
        if reuse_sampler is not None:
            data_loader_kwargs['sampler'] = reuse_sampler
        else:
            data_loader_kwargs['shuffle'] = isinstance(sampler, RandomSampler)

        pin_memory_device = getattr(dataloader, 'pin_memory_device', '')
        if pin_memory_device:
            data_loader_kwargs['pin_memory_device'] = pin_memory_device

        timeout = getattr(dataloader, 'timeout', 0)
        if timeout:
            data_loader_kwargs['timeout'] = timeout

        if num_workers > 0:
            prefetch_factor = getattr(dataloader, 'prefetch_factor', None)
            if prefetch_factor is not None:
                data_loader_kwargs['prefetch_factor'] = prefetch_factor
            persistent_workers = getattr(dataloader, 'persistent_workers', False)
            if persistent_workers:
                data_loader_kwargs['persistent_workers'] = persistent_workers

        return DataLoader(**data_loader_kwargs)

    def _shutdown_failed_dataloader_iterator(self, dataloader: Optional[DataLoader]):
        if dataloader is None:
            return

        iterator = getattr(dataloader, "_iterator", None)
        if iterator is None:
            return

        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception:
                pass

        try:
            dataloader._iterator = None
        except Exception:
            pass

    def _rebuild_dataloader_with_num_workers(self, dataloader: DataLoader, num_workers: int) -> DataLoader:
        dataloader_datasets = get_dataloader_datasets(dataloader)
        if len(dataloader_datasets) == 0:
            return self._clone_dataloader_with_num_workers(dataloader, num_workers)

        dataset_configs = []
        for dataset in dataloader_datasets:
            dataset_config = copy.deepcopy(getattr(dataset, "dataset_config", None))
            if dataset_config is None:
                return self._clone_dataloader_with_num_workers(dataloader, num_workers)
            dataset_config.num_workers = num_workers
            if num_workers == 0:
                dataset_config.persistent_workers = False
                dataset_config.enable_windows_num_workers = False
            dataset_configs.append(dataset_config)

        batch_size = dataloader.batch_size
        if batch_size is None:
            batch_size = getattr(dataloader_datasets[0], "batch_size", 1)

        return get_dataloader_from_datasets(
            dataset_configs,
            batch_size=batch_size,
            sd=self.sd,
            cache_progress_hook=self.cache_progress_hook,
            global_bucket_batching=getattr(
                self.train_config, 'global_bucket_batching', False
            ),
        )

    def _prepare_dataloader_with_windows_fallback(self, dataloader: Optional[DataLoader], label: str):
        if dataloader is None:
            return None, None

        try:
            return dataloader, iter(dataloader)
        except Exception as error:
            if not self._is_windows_dataloader_pickle_error(error):
                raise
            if getattr(dataloader, 'num_workers', 0) == 0:
                raise

            print_acc("")
            print_acc("############################################################")
            print_acc(
                f"# Windows dataloader workers failed for {label}; retrying with num_workers=0 #"
            )
            print_acc("############################################################")
            print_acc("")

            self._shutdown_failed_dataloader_iterator(dataloader)
            fallback_dataloader = self._rebuild_dataloader_with_num_workers(dataloader, 0)
            return fallback_dataloader, iter(fallback_dataloader)

    def _resolve_epoch_based_steps(self):
        # When train_by_epoch is set, convert the user's epoch count into a concrete
        # step count now that the dataset is loaded. len(self.data_loader) is the
        # exact batches-per-epoch (honors batch_size, num_repeats, bucketing,
        # must_include_caption, video, etc.). Each training step consumes
        # gradient_accumulation batches, so:
        #   steps = ceil(epochs * batches_per_epoch / gradient_accumulation)
        if not getattr(self.train_config, 'train_by_epoch', False):
            return
        epochs = max(1, int(getattr(self.train_config, 'epochs', 0) or 0))
        if self.data_loader is None:
            print_acc("Train by epoch is on but no training dataset is loaded; keeping steps as-is.")
            return
        try:
            batches_per_epoch = len(self.data_loader)
        except TypeError:
            print_acc("Train by epoch: dataloader has no length; keeping steps as-is.")
            return
        if batches_per_epoch <= 0:
            print_acc("Train by epoch: training dataset is empty; keeping steps as-is.")
            return
        grad_accum = max(1, int(self.train_config.gradient_accumulation or 1))
        # ceil division on positive ints, no float/import needed
        resolved_steps = max(1, -(-(epochs * batches_per_epoch) // grad_accum))
        print_acc(
            f"Train by epoch: {epochs} epoch(s) x {batches_per_epoch} batches/epoch "
            f"/ {grad_accum} grad-accum = {resolved_steps} steps "
            f"(was {self.train_config.steps})."
        )
        self.train_config.steps = resolved_steps

    def _get_pretrained_lora_base_step(self):
        # The finished step of the EXTERNAL pretrained LoRA we're continuing from,
        # read from its metadata (fixed across runs) so "train on top" is
        # deterministic and survives a crash/resume. Returns None when not applicable
        # (no pretrained lora, resume_training_state off, explicit start_step, etc.).
        if not getattr(self.train_config, 'train_on_top_of_loaded_lora', True):
            return None
        if self.train_config.start_step is not None:
            return None
        if not self._should_resume_training_state_from_pretrained_lora():
            return None
        path = self._get_pretrained_lora_path()
        if path is None or not os.path.exists(path):
            return None
        try:
            if os.path.isdir(path):
                meta_path = os.path.join(path, 'aitk_meta.yaml')
                if not os.path.exists(meta_path):
                    return None
                with open(meta_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
            else:
                meta = load_metadata_from_safetensors(path)
        except Exception:
            return None
        if meta and 'training_info' in meta and 'step' in meta['training_info']:
            try:
                return int(meta['training_info']['step'])
            except (TypeError, ValueError):
                return None
        return None

    def _apply_train_on_top_of_loaded_steps(self):
        # When continuing from a loaded pretrained LoRA, the configured/resolved steps
        # is the number of steps to train ON TOP of that LoRA's finished steps, not an
        # absolute target. Extend the target deterministically as (pretrained_step +
        # steps): a fresh run trains the full amount AND a crash/resume continues to
        # the same target. Same-job resume with no pretrained LoRA is unaffected.
        base_step = self._get_pretrained_lora_base_step()
        if base_step is None or base_step <= 0:
            return
        amount = int(self.train_config.steps)
        new_total = base_step + amount
        print_acc(
            f"Train-on-top of loaded LoRA: target = pretrained step {base_step} + "
            f"{amount} step(s) = {new_total} (was {self.train_config.steps})."
        )
        self.train_config.steps = new_total

    def prepare_accelerator(self):
        # set some config
        self.accelerator.even_batches=False
        
        # # prepare all the models stuff for accelerator (hopefully we dont miss any)
        self.sd.vae = self.accelerator.prepare(self.sd.vae)
        if self.sd.unet is not None:
            self.sd.unet = self.accelerator.prepare(self.sd.unet)
            # todo always tdo it?
            self.modules_being_trained.append(self.sd.unet)
        if self.sd.text_encoder is not None and self.train_config.train_text_encoder:
            if isinstance(self.sd.text_encoder, list):
                self.sd.text_encoder = [self.accelerator.prepare(model) for model in self.sd.text_encoder]
                self.modules_being_trained.extend(self.sd.text_encoder)
            else:
                self.sd.text_encoder = self.accelerator.prepare(self.sd.text_encoder)
                self.modules_being_trained.append(self.sd.text_encoder)
        if self.sd.refiner_unet is not None and self.train_config.train_refiner:
            self.sd.refiner_unet = self.accelerator.prepare(self.sd.refiner_unet)
            self.modules_being_trained.append(self.sd.refiner_unet)
        # todo, do we need to do the network or will "unet" get it?
        if self.sd.network is not None:
            self.sd.network = self.accelerator.prepare(self.sd.network)
            self.modules_being_trained.append(self.sd.network)
            # Keep the local reference aligned with the prepared module so
            # saves and runtime checks operate on the same weights being trained.
            self.network = unwrap_model(self.sd.network)
        if self.adapter is not None and self.adapter_config.train:
            # todo adapters may not be a module. need to check
            self.adapter = self.accelerator.prepare(self.adapter)
            self.modules_being_trained.append(self.adapter)
        
        # prepare other things
        self.optimizer = self.accelerator.prepare(self.optimizer)
        # Use the live optimizer param groups for later grad clipping and checks.
        self.params = self.optimizer.param_groups
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        # self.data_loader = self.accelerator.prepare(self.data_loader)
        # if self.data_loader_reg is not None:
        #     self.data_loader_reg = self.accelerator.prepare(self.data_loader_reg)
        self._validate_optimizer_tracks_live_network()
            

    def ensure_params_requires_grad(self, force=False):
        if self.train_config.do_paramiter_swapping and not force:
            # the optimizer will handle this if we are not forcing
            return
        for group in self.params:
            for param in group['params']:
                if isinstance(param, torch.nn.Parameter):  # Ensure it's a proper parameter
                    param.requires_grad_(True)

    def setup_ema(self):
        if self.train_config.ema_config.use_ema:
            # our params are in groups. We need them as a single iterable
            params = []
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    params.append(param)
            self.ema = ExponentialMovingAverage(
                params,
                decay=self.train_config.ema_config.ema_decay,
                use_feedback=self.train_config.ema_config.use_feedback,
                param_multiplier=self.train_config.ema_config.param_multiplier,
            )

    def before_dataset_load(self):
        pass

    def get_params(self):
        # you can extend this in subclass to get params
        # otherwise params will be gathered through normal means
        return None

    def hook_train_loop(self, batch):
        # return loss
        return 0.0
    
    def hook_after_sd_init_before_load(self):
        pass

    def get_latest_save_path(self, name=None, post=''):
        if name == None:
            name = self.job.name
        # get latest saved step
        latest_path = None
        if os.path.exists(self.save_root):
            # Define patterns for both files and directories
            patterns = [
                f"{name}*{post}.safetensors",
                f"{name}*{post}.pt",
                f"{name}*{post}"
            ]
            # Search for both files and directories
            paths = []
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(self.save_root, pattern)))

            # Filter out non-existent paths and sort by creation time
            if paths:
                paths = [p for p in paths if os.path.exists(p)]
                # remove false positives
                if '_LoRA' not in name:
                    paths = [p for p in paths if '_LoRA' not in p]
                if '_refiner' not in name:
                    paths = [p for p in paths if '_refiner' not in p]
                if '_t2i' not in name:
                    paths = [p for p in paths if '_t2i' not in p]
                if '_cn' not in name:
                    paths = [p for p in paths if '_cn' not in p]

                if len(paths) > 0:
                    latest_path = max(paths, key=os.path.getctime)
        
        if latest_path is None and self.network_config is not None and self.network_config.pretrained_lora_path is not None:
            # set pretrained lora path as load path if we do not have a checkpoint to resume from
            if os.path.exists(self.network_config.pretrained_lora_path):
                latest_path = self.network_config.pretrained_lora_path
                print_acc(f"Using pretrained lora path from config: {latest_path}")
            else:
                # no pretrained lora found
                print_acc(f"Pretrained lora path from config does not exist: {self.network_config.pretrained_lora_path}")

        return latest_path

    def _get_pretrained_lora_path(self) -> Optional[str]:
        if self.network_config is None:
            return None
        pretrained_lora_path = getattr(self.network_config, 'pretrained_lora_path', None)
        if pretrained_lora_path is None:
            return None
        pretrained_lora_path = str(pretrained_lora_path).strip()
        if pretrained_lora_path == '':
            return None
        return pretrained_lora_path

    def _is_pretrained_lora_path(self, path: Optional[str]) -> bool:
        pretrained_lora_path = self._get_pretrained_lora_path()
        if path is None or pretrained_lora_path is None:
            return False

        try:
            normalized_path = os.path.normcase(os.path.abspath(path))
            normalized_pretrained = os.path.normcase(os.path.abspath(pretrained_lora_path))
        except Exception:
            normalized_path = os.path.normcase(path)
            normalized_pretrained = os.path.normcase(pretrained_lora_path)

        return normalized_path == normalized_pretrained

    def _should_resume_training_state_from_pretrained_lora(self) -> bool:
        return bool(
            self.network_config is not None
            and getattr(self.network_config, 'resume_training_state', False)
            and self._get_pretrained_lora_path() is not None
        )

    def _get_imported_optimizer_state_path(self) -> Optional[str]:
        if not self._should_resume_training_state_from_pretrained_lora():
            return None

        pretrained_lora_path = self._get_pretrained_lora_path()
        if pretrained_lora_path is None or not os.path.exists(pretrained_lora_path):
            return None

        state_root = pretrained_lora_path if os.path.isdir(pretrained_lora_path) else os.path.dirname(pretrained_lora_path)
        if state_root == '':
            return None

        optimizer_state_path = os.path.join(state_root, 'optimizer.pt')
        if os.path.exists(optimizer_state_path):
            return optimizer_state_path
        return None

    def load_training_state_from_metadata(self, path):
        if not self.accelerator.is_main_process:
            return
        if (
                path is not None
                and self._is_pretrained_lora_path(path)
                and not self._should_resume_training_state_from_pretrained_lora()
        ):
            # dont load metadata from pretrained lora
            return
        meta = None
        # if path is folder, then it is diffusers
        if os.path.isdir(path):
            meta_path = os.path.join(path, 'aitk_meta.yaml')
            # load it
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
        else:
            meta = load_metadata_from_safetensors(path)
        # if 'training_info' in Orderdict keys
        if meta is not None and 'training_info' in meta and 'step' in meta['training_info'] and self.train_config.start_step is None:
            self.step_num = meta['training_info']['step']
            if 'epoch' in meta['training_info']:
                self.epoch_num = meta['training_info']['epoch']
            self.start_step = self.step_num
            print_acc(f"Found step {self.step_num} in metadata, starting from there")

    def load_weights(self, path):
        if self.network is not None:
            extra_weights = self.network.load_weights(path)
            self.load_training_state_from_metadata(path)
            self._log_network_state_stats(
                f"Loaded LoRA state stats from {path}",
                self._get_live_network(),
            )
            return extra_weights
        else:
            print_acc("load_weights not implemented for non-network models")
            return None

    def apply_snr(self, seperated_loss, timesteps):
        if self.train_config.learnable_snr_gos:
            # add snr_gamma
            seperated_loss = apply_learnable_snr_gos(seperated_loss, timesteps, self.snr_gos)
        elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001:
            # add snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma, fixed=True)
        elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
            # add min_snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        return seperated_loss

    def load_lorm(self):
        latest_save_path = self.get_latest_save_path()
        if latest_save_path is not None:
            # hacky way to reload weights for now
            # todo, do this
            state_dict = load_file(latest_save_path, device=self.device)
            self.sd.unet.load_state_dict(state_dict)

            meta = load_metadata_from_safetensors(latest_save_path)
            # if 'training_info' in Orderdict keys
            if 'training_info' in meta and 'step' in meta['training_info']:
                self.step_num = meta['training_info']['step']
                if 'epoch' in meta['training_info']:
                    self.epoch_num = meta['training_info']['epoch']
                self.start_step = self.step_num
                print_acc(f"Found step {self.step_num} in metadata, starting from there")

    # def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
    #     self.sd.noise_scheduler.set_timesteps(1000, device=self.device_torch)
    #     sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device_torch, dtype=dtype)
    #     schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device_torch, )
    #     timesteps = timesteps.to(self.device_torch, )
    #
    #     # step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    #     step_indices = [t for t in timesteps]
    #
    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    def load_additional_training_modules(self, params):
        # override in subclass
        return params

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def get_optimal_noise(self, latents, dtype=torch.float32):
        batch_num = latents.shape[0]
        chunks = torch.chunk(latents, batch_num, dim=0)
        noise_chunks = []
        for chunk in chunks:
            noise_samples = [torch.randn_like(chunk, device=chunk.device, dtype=dtype) for _ in range(self.train_config.optimal_noise_pairing_samples)]
            # find the one most similar to the chunk
            lowest_loss = 999999999999
            best_noise = None
            for noise in noise_samples:
                loss = torch.nn.functional.mse_loss(chunk, noise)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_noise = noise
            noise_chunks.append(best_noise)
        noise = torch.cat(noise_chunks, dim=0)
        return noise
    
    def get_consistent_noise(self, latents, batch: 'DataLoaderBatchDTO', dtype=torch.float32):
        batch_num = latents.shape[0]
        chunks = torch.chunk(latents, batch_num, dim=0)
        noise_chunks = []
        for idx, chunk in enumerate(chunks):
            # get seed from path
            file_item = batch.file_items[idx]
            img_path = file_item.path
            # add augmentors
            if file_item.flip_x:
                img_path += '_fx'
            if file_item.flip_y:
                img_path += '_fy'
            seed = int(hashlib.md5(img_path.encode()).hexdigest(), 16) & 0xffffffff
            generator = torch.Generator("cpu").manual_seed(seed)
            noise_chunk = torch.randn(chunk.shape, generator=generator).to(chunk.device, dtype=dtype)
            noise_chunks.append(noise_chunk)
        noise = torch.cat(noise_chunks, dim=0).to(dtype=dtype)
        return noise
            

    def get_noise(
        self, 
        latents, 
        batch_size, 
        dtype=torch.float32, 
        batch: 'DataLoaderBatchDTO' = None,
        timestep=None,
    ):
        if self.train_config.optimal_noise_pairing_samples > 1:
            noise = self.get_optimal_noise(latents, dtype=dtype)
        elif self.train_config.force_consistent_noise:
            if batch is None:
                raise ValueError("Batch must be provided for consistent noise")
            noise = self.get_consistent_noise(latents, batch, dtype=dtype)
        else:
            if hasattr(self.sd, 'get_latent_noise_from_latents'):
                noise = self.sd.get_latent_noise_from_latents(
                    latents,
                    noise_offset=self.train_config.noise_offset
                ).to(self.device_torch, dtype=dtype)
            else:
                # get noise
                noise = self.sd.get_latent_noise(
                    height=latents.shape[2],
                    width=latents.shape[3],
                    num_channels=latents.shape[1],
                    batch_size=batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)
        
        if self.train_config.blended_blur_noise:
            noise = get_blended_blur_noise(
                latents, noise, timestep
            )

        return noise

    def _should_lazy_cache_latents(self, batch: 'DataLoaderBatchDTO') -> bool:
        dataset_config = batch.dataset_config
        return (
            dataset_config is not None
            and getattr(dataset_config, "lazy_cache_latents", False)
            and getattr(dataset_config, "cache_latents_to_disk", False)
        )

    @staticmethod
    def _extract_first_frames_from_imgs(imgs: torch.Tensor) -> torch.Tensor:
        if len(imgs.shape) == 4:
            return imgs
        if len(imgs.shape) == 5:
            return imgs[:, 0]
        raise ValueError(f"Unknown frame shape {imgs.shape}")

    @staticmethod
    def _aux_latent_batch_attr_name(state_key: str) -> str:
        return {
            "first_frame_latent": "first_frame_latents",
            "audio_latent": "audio_latents",
            "first_frame_condition": "first_frame_conditions",
        }[state_key]

    def _build_batch_aux_latent_tensors(
            self,
            batch: 'DataLoaderBatchDTO',
            imgs: torch.Tensor,
            latents: torch.Tensor,
            file_item_indices: List[int],
    ):
        first_frame_latents = None
        first_frame_conditions = None
        if batch.dataset_config.num_frames > 1 and batch.dataset_config.do_i2v:
            need_first_frame_latents = self.sd.uses_i2v_first_frame_latents()
            need_first_frame_conditions = self.sd.uses_i2v_first_frame_conditions()
            first_frames = self._extract_first_frames_from_imgs(imgs)
            if need_first_frame_latents:
                first_frame_latents = self.sd.encode_images(first_frames).detach()
            if need_first_frame_conditions:
                first_frame_conditions = self.sd.build_i2v_first_frame_conditions(
                    latents=latents,
                    first_frames=first_frames,
                    first_frame_latents=first_frame_latents,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                if first_frame_conditions is not None:
                    first_frame_conditions = first_frame_conditions.detach()

        audio_latents = None
        if batch.audio_data is not None:
            audio_batch = [batch.audio_data[idx] for idx in file_item_indices]
            if all([audio_item is not None for audio_item in audio_batch]):
                try:
                    audio_latents = self.sd.encode_audio(audio_batch).detach()
                except NotImplementedError:
                    audio_latents = None

        return first_frame_latents, audio_latents, first_frame_conditions

    def _merge_batch_aux_latent_tensor(
            self,
            batch: 'DataLoaderBatchDTO',
            state_key: str,
            missing_tensor: Optional[torch.Tensor],
            dtype: torch.dtype,
    ):
        attr_name = self._aux_latent_batch_attr_name(state_key)
        cached_tensor = getattr(batch, attr_name, None)

        if batch.latent_cache_mask is None:
            merged_tensor = cached_tensor if cached_tensor is not None else missing_tensor
            if merged_tensor is not None:
                setattr(
                    batch,
                    attr_name,
                    merged_tensor.to(self.device_torch, dtype=dtype),
                )
            return

        if len(batch.cached_latent_indices) > 0 and cached_tensor is None:
            setattr(batch, attr_name, None)
            return
        if len(batch.missing_latent_indices) > 0 and missing_tensor is None:
            setattr(batch, attr_name, None)
            return

        reference_tensor = cached_tensor if cached_tensor is not None else missing_tensor
        if reference_tensor is None:
            setattr(batch, attr_name, None)
            return

        merged_tensor = torch.empty(
            (len(batch.file_items),) + tuple(reference_tensor.shape[1:]),
            device=self.device_torch,
            dtype=dtype,
        )
        if cached_tensor is not None:
            merged_tensor[batch.cached_latent_indices] = cached_tensor.to(
                self.device_torch,
                dtype=dtype,
            )
        if missing_tensor is not None:
            merged_tensor[batch.missing_latent_indices] = missing_tensor.to(
                self.device_torch,
                dtype=dtype,
            )
        setattr(batch, attr_name, merged_tensor)

    def _cache_missing_batch_latents(
            self,
            batch: 'DataLoaderBatchDTO',
            latents: torch.Tensor,
            imgs: torch.Tensor,
            file_item_indices: List[int],
    ):
        if not self._should_lazy_cache_latents(batch):
            return None, None, None
        if len(file_item_indices) == 0:
            return None, None, None

        first_frame_latents, audio_latents, first_frame_conditions = (
            self._build_batch_aux_latent_tensors(
                batch,
                imgs,
                latents,
                file_item_indices,
            )
        )

        file_item_state_dict_pairs = []
        for latent_idx, file_item_idx in enumerate(file_item_indices):
            file_item = batch.file_items[file_item_idx]
            file_item.is_caching_to_disk = True
            file_item.is_caching_to_memory = False
            file_item.latent_load_device = self.sd.device

            state_dict = OrderedDict([
                ("latent", latents[latent_idx].detach().to("cpu")),
            ])
            if first_frame_latents is not None:
                state_dict["first_frame_latent"] = first_frame_latents[latent_idx].detach().to("cpu")
            if audio_latents is not None:
                state_dict["audio_latent"] = audio_latents[latent_idx].detach().to("cpu")
            if first_frame_conditions is not None:
                state_dict["first_frame_condition"] = (
                    first_frame_conditions[latent_idx].detach().to("cpu")
                )

            file_item_state_dict_pairs.append((file_item, state_dict))

        cache_file_item_latents_to_disk(
            file_item_state_dict_pairs,
            batch.dataset_config.latent_cache_shard_size,
            buffered=True,
        )
        return first_frame_latents, audio_latents, first_frame_conditions

    def _encode_images_for_lazy_cache(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs is None:
            raise ValueError("imgs cannot be None for lazy latent encoding")
        if imgs.shape[0] <= 1:
            return self.sd.encode_images(imgs)

        use_chunked_encoding = any([
            getattr(self.model_config, "low_vram", False),
            getattr(self.model_config, "layer_offloading", False),
        ])
        if not use_chunked_encoding:
            return self.sd.encode_images(imgs)

        if not getattr(self, "_logged_lazy_encode_chunking", False):
            print_acc(
                "Low-VRAM lazy latent encoding detected; encoding cache misses "
                "one image at a time to avoid VAE/offload stalls."
            )
            self._logged_lazy_encode_chunking = True

        latent_chunks = []
        for img_chunk in torch.chunk(imgs, imgs.shape[0], dim=0):
            latent_chunks.append(self.sd.encode_images(img_chunk))
        return torch.cat(latent_chunks, dim=0)

    def _maybe_force_flush_lazy_latent_buffers(self, train_loop_elapsed: Optional[float]):
        if not self.has_lazy_latent_caching or train_loop_elapsed is None:
            return

        pending_latent_count = get_buffered_file_item_latent_count()
        if pending_latent_count < self._adaptive_lazy_flush_min_pending_latents:
            self._adaptive_lazy_flush_history.append(train_loop_elapsed)
            return

        prior_history = list(self._adaptive_lazy_flush_history)
        should_flush = False
        sigma_threshold = None
        trigger_threshold = None
        baseline_center = None
        baseline_std = None

        if (
            len(prior_history) >= self._adaptive_lazy_flush_min_history
            and (self.step_num - self._last_adaptive_lazy_flush_step) >= self._adaptive_lazy_flush_cooldown_steps
        ):
            baseline_history = prior_history
            exclude_recent = len(prior_history) // 4
            if len(prior_history) > exclude_recent:
                baseline_history = prior_history[:-exclude_recent]

            baseline_history = [value for value in baseline_history if value > 0]
            if len(baseline_history) >= self._adaptive_lazy_flush_min_history:
                # Use a robust baseline anchored to the faster half of the older history,
                # so gradual slowdowns do not poison the detector.
                sorted_history = sorted(baseline_history)
                faster_half = sorted_history[:max(1, len(sorted_history) // 2)]
                baseline_center = float(np.median(faster_half))
                deviations = np.abs(np.array(faster_half, dtype=np.float64) - baseline_center)
                mad = float(np.median(deviations))
                # Convert MAD to an approximate std-equivalent for a normal distribution.
                baseline_std = mad * 1.4826
            else:
                baseline_center = float(np.median(prior_history))
                baseline_std = float(np.std(prior_history))

            sigma_threshold = baseline_center + (3.0 * baseline_std)
            min_absolute_threshold = max(20.0, baseline_center * 1.75)
            trigger_threshold = max(sigma_threshold, min_absolute_threshold)
            should_flush = train_loop_elapsed > trigger_threshold

        if should_flush:
            print_acc("")
            print_acc("############################################################")
            print_acc(
                "# Adaptive lazy latent flush triggered: "
                f"{train_loop_elapsed:.2f}s step exceeded 3σ threshold "
                f"({sigma_threshold:.2f}s; baseline={baseline_center:.2f}s std≈{baseline_std:.2f}s) #"
            )
            print_acc("############################################################")
            print_acc("")
            flush_buffered_file_item_latents_to_disk(force=True)
            flush()
            self._adaptive_lazy_flush_history.clear()
            self._last_adaptive_lazy_flush_step = self.step_num
            return

        self._adaptive_lazy_flush_history.append(train_loop_elapsed)

    def process_general_training_batch(self, batch: 'DataLoaderBatchDTO'):
        with torch.no_grad():
            with self.timer('prepare_prompt'):
                prompts = batch.get_caption_list()
                is_reg_list = batch.get_is_reg_list()

                is_any_reg = any([is_reg for is_reg in is_reg_list])

                do_double = self.train_config.short_and_long_captions and not is_any_reg

                if self.train_config.short_and_long_captions and do_double:
                    # dont do this with regs. No point

                    # double batch and add short captions to the end
                    prompts = prompts + batch.get_caption_short_list()
                    is_reg_list = is_reg_list + is_reg_list
                if self.model_config.refiner_name_or_path is not None and self.train_config.train_unet:
                    prompts = prompts + prompts
                    is_reg_list = is_reg_list + is_reg_list

                conditioned_prompts = []

                for prompt, is_reg in zip(prompts, is_reg_list):

                    # make sure the embedding is in the prompts
                    if self.embedding is not None:
                        prompt = self.embedding.inject_embedding_to_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                        prompt = self.adapter.inject_trigger_into_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    # make sure trigger is in the prompts if not a regularization run
                    if self.trigger_word is not None:
                        prompt = self.sd.inject_trigger_into_prompt(
                            prompt,
                            trigger=self.trigger_word,
                            add_if_not_present=not is_reg,
                        )

                    if not is_reg and self.train_config.prompt_saturation_chance > 0.0:
                        # do random prompt saturation by expanding the prompt to hit at least 77 tokens
                        if random.random() < self.train_config.prompt_saturation_chance:
                            est_num_tokens = len(prompt.split(' '))
                            if est_num_tokens < 77:
                                num_repeats = int(77 / est_num_tokens) + 1
                                prompt = ', '.join([prompt] * num_repeats)


                    conditioned_prompts.append(prompt)

            with self.timer('prepare_latents'):
                dtype = get_torch_dtype(self.train_config.dtype)
                imgs = None
                is_reg = any(batch.get_is_reg_list())
                use_lazy_latent_cache = self._should_lazy_cache_latents(batch)
                if batch.tensor is not None:
                    imgs = batch.tensor
                    imgs = imgs.to(self.device_torch, dtype=dtype)
                    # dont adjust for regs.
                    if self.train_config.img_multiplier is not None and not is_reg and not use_lazy_latent_cache:
                        # do it ad contrast
                        imgs = reduce_contrast(imgs, self.train_config.img_multiplier)
                if batch.latents is not None:
                    cached_latents = batch.latents.to(self.device_torch, dtype=dtype)
                    if batch.latent_cache_mask is not None:
                        if imgs is None:
                            raise ValueError("Mixed latent-cache batches require image tensors for uncached items")
                        missing_imgs = imgs[batch.missing_latent_indices]
                        missing_latents = self._encode_images_for_lazy_cache(missing_imgs)
                        (
                            missing_first_frame_latents,
                            missing_audio_latents,
                            missing_first_frame_conditions,
                        ) = self._cache_missing_batch_latents(
                            batch,
                            missing_latents,
                            missing_imgs,
                            batch.missing_latent_indices,
                        )
                        latents = torch.empty(
                            (len(batch.file_items),) + tuple(cached_latents.shape[1:]),
                            device=self.device_torch,
                            dtype=dtype,
                        )
                        latents[batch.cached_latent_indices] = cached_latents
                        latents[batch.missing_latent_indices] = missing_latents.to(
                            self.device_torch,
                            dtype=dtype,
                        )
                        self._merge_batch_aux_latent_tensor(
                            batch,
                            "first_frame_latent",
                            missing_first_frame_latents,
                            dtype,
                        )
                        self._merge_batch_aux_latent_tensor(
                            batch,
                            "audio_latent",
                            missing_audio_latents,
                            dtype,
                        )
                        self._merge_batch_aux_latent_tensor(
                            batch,
                            "first_frame_condition",
                            missing_first_frame_conditions,
                            dtype,
                        )
                    else:
                        latents = cached_latents
                    batch.latents = latents
                else:
                    if use_lazy_latent_cache:
                        latents = self._encode_images_for_lazy_cache(imgs)
                        (
                            first_frame_latents,
                            audio_latents,
                            first_frame_conditions,
                        ) = self._cache_missing_batch_latents(
                            batch,
                            latents,
                            imgs,
                            list(range(len(batch.file_items))),
                        )
                        if first_frame_latents is not None:
                            batch.first_frame_latents = first_frame_latents.to(
                                self.device_torch,
                                dtype=dtype,
                            )
                        if audio_latents is not None:
                            batch.audio_latents = audio_latents.to(
                                self.device_torch,
                                dtype=dtype,
                            )
                        if first_frame_conditions is not None:
                            batch.first_frame_conditions = first_frame_conditions.to(
                                self.device_torch,
                                dtype=dtype,
                            )
                    else:
                        # normalize to
                        if self.train_config.standardize_images:
                            if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                                target_mean_list = [0.0002, -0.1034, -0.1879]
                                target_std_list = [0.5436, 0.5116, 0.5033]
                            else:
                                target_mean_list = [-0.0739, -0.1597, -0.2380]
                                target_std_list = [0.5623, 0.5295, 0.5347]
                            # Mean: tensor([-0.0739, -0.1597, -0.2380])
                            # Standard Deviation: tensor([0.5623, 0.5295, 0.5347])
                            imgs_channel_mean = imgs.mean(dim=(2, 3), keepdim=True)
                            imgs_channel_std = imgs.std(dim=(2, 3), keepdim=True)
                            imgs = (imgs - imgs_channel_mean) / imgs_channel_std
                            target_mean = torch.tensor(target_mean_list, device=self.device_torch, dtype=dtype)
                            target_std = torch.tensor(target_std_list, device=self.device_torch, dtype=dtype)
                            # expand them to match dim
                            target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                            target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                            imgs = imgs * target_std + target_mean
                            batch.tensor = imgs

                            # show_tensors(imgs, 'imgs')

                        latents = self.sd.encode_images(imgs)
                    batch.latents = latents

                if self.train_config.standardize_latents:
                    if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                        target_mean_list = [-0.1075, 0.0231, -0.0135, 0.2164]
                        target_std_list = [0.8979, 0.7505, 0.9150, 0.7451]
                    else:
                        target_mean_list = [0.2949, -0.3188, 0.0807, 0.1929]
                        target_std_list = [0.8560, 0.9629, 0.7778, 0.6719]

                    latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True)
                    latents_channel_std = latents.std(dim=(2, 3), keepdim=True)
                    latents = (latents - latents_channel_mean) / latents_channel_std
                    target_mean = torch.tensor(target_mean_list, device=self.device_torch, dtype=dtype)
                    target_std = torch.tensor(target_std_list, device=self.device_torch, dtype=dtype)
                    # expand them to match dim
                    target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    latents = latents * target_std + target_mean
                    batch.latents = latents

                    # show_latents(latents, self.sd.vae, 'latents')


                if batch.unconditional_tensor is not None and batch.unconditional_latents is None:
                    unconditional_imgs = batch.unconditional_tensor
                    unconditional_imgs = unconditional_imgs.to(self.device_torch, dtype=dtype)
                    unconditional_latents = self.sd.encode_images(unconditional_imgs)
                    batch.unconditional_latents = unconditional_latents * self.train_config.latent_multiplier

                unaugmented_latents = None
                if self.train_config.loss_target == 'differential_noise':
                    # we determine noise from the differential of the latents
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor)

            with self.timer('prepare_scheduler'):
                
                batch_size = len(batch.file_items)
                min_noise_steps = self.train_config.min_denoising_steps
                max_noise_steps = self.train_config.max_denoising_steps
                if self.model_config.refiner_name_or_path is not None:
                    # if we are not training the unet, then we are only doing refiner and do not need to double up
                    if self.train_config.train_unet:
                        max_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = True
                    else:
                        min_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = False

                num_train_timesteps = self.train_config.num_train_timesteps

                if self.train_config.noise_scheduler in ['custom_lcm']:
                    # we store this value on our custom one
                    self.sd.noise_scheduler.set_timesteps(
                        self.sd.noise_scheduler.train_timesteps, device=self.device_torch
                    )
                elif self.train_config.noise_scheduler in ['lcm']:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch, original_inference_steps=num_train_timesteps
                    )
                elif self.train_config.noise_scheduler == 'flowmatch':
                    linear_timesteps = any([
                        self.train_config.linear_timesteps,
                        self.train_config.linear_timesteps2,
                        self.train_config.timestep_type == 'linear',
                        self.train_config.timestep_type == 'one_step',
                    ])
                    
                    timestep_type = 'linear' if linear_timesteps else None
                    if timestep_type is None:
                        timestep_type = self.train_config.timestep_type
                    
                    if self.train_config.timestep_type == 'next_sample':
                        # simulate a sample
                        num_train_timesteps = self.train_config.next_sample_timesteps
                        timestep_type = 'shift'
                    
                    patch_size = 1
                    if self.sd.is_flux or 'flex' in self.sd.arch:
                        # flux is a patch size of 1, but latents are divided by 2, so we need to double it
                        patch_size = 2
                    elif hasattr(self.sd.unet.config, 'patch_size'):
                        patch_size = self.sd.unet.config.patch_size
                    
                    self.sd.noise_scheduler.set_train_timesteps(
                        num_train_timesteps,
                        device=self.device_torch,
                        timestep_type=timestep_type,
                        latents=latents,
                        patch_size=patch_size,
                    )
                else:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch
                    )
            if self.sd.is_multistage:
                with self.timer('adjust_multistage_timesteps'):
                    # get our current sample range
                    boundaries = [1] + self.sd.multistage_boundaries
                    boundary_max, boundary_min = boundaries[self.current_boundary_index], boundaries[self.current_boundary_index + 1]
                    asc_timesteps = torch.flip(self.sd.noise_scheduler.timesteps, dims=[0])
                    lo = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_max * 1000, device=asc_timesteps.device), right=False)
                    hi = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_min * 1000, device=asc_timesteps.device), right=True)
                    first_idx = (lo - 1).item() if hi > lo else 0
                    last_idx  = (hi - 1).item() if hi > lo else 999
                    min_noise_steps = first_idx
                    max_noise_steps = last_idx

            # clip min max indicies
            min_noise_steps = max(min_noise_steps, 0)
            max_noise_steps = min(max_noise_steps, num_train_timesteps - 1)
            
                    
            with self.timer('prepare_timesteps_indices'):

                content_or_style = self.train_config.content_or_style
                if is_reg:
                    content_or_style = self.train_config.content_or_style_reg

                # if self.train_config.timestep_sampling == 'style' or self.train_config.timestep_sampling == 'content':
                if self.train_config.timestep_type == 'next_sample':
                    timestep_indices = torch.randint(
                            0,
                            num_train_timesteps - 2, # -1 for 0 idx, -1 so we can step
                            (batch_size,),
                            device=self.device_torch
                        )
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type == 'one_step':
                    timestep_indices = torch.zeros((batch_size,), device=self.device_torch, dtype=torch.long)
                elif content_or_style in ['style', 'content']:
                    # this is from diffusers training code
                    # Cubic sampling for favoring later or earlier timesteps
                    # For more details about why cubic sampling is used for content / structure,
                    # refer to section 3.4 of https://arxiv.org/abs/2302.08453

                    # for content / structure, it is best to favor earlier timesteps
                    # for style, it is best to favor later timesteps

                    orig_timesteps = torch.rand((batch_size,), device=latents.device)

                    if content_or_style == 'content':
                        timestep_indices = orig_timesteps ** 3 * self.train_config.num_train_timesteps
                    elif content_or_style == 'style':
                        timestep_indices = (1 - orig_timesteps ** 3) * self.train_config.num_train_timesteps

                    timestep_indices = value_map(
                        timestep_indices,
                        0,
                        self.train_config.num_train_timesteps - 1,
                        min_noise_steps,
                        max_noise_steps
                    )
                    timestep_indices = timestep_indices.long().clamp(
                        min_noise_steps,
                        max_noise_steps
                    )
                    
                elif content_or_style == 'balanced':
                    if min_noise_steps == max_noise_steps:
                        timestep_indices = torch.ones((batch_size,), device=self.device_torch) * min_noise_steps
                    else:
                        # todo, some schedulers use indices, otheres use timesteps. Not sure what to do here
                        min_idx = min_noise_steps + 1
                        max_idx = max_noise_steps - 1
                        if self.train_config.noise_scheduler == 'flowmatch':
                            # flowmatch uses indices, so we need to use indices
                            min_idx = min_noise_steps
                            max_idx = max_noise_steps
                        timestep_indices = torch.randint(
                            min_idx,
                            max_idx,
                            (batch_size,),
                            device=self.device_torch
                        )
                    timestep_indices = timestep_indices.long()
                else:
                    raise ValueError(f"Unknown content_or_style {content_or_style}")
            with self.timer('convert_timestep_indices_to_timesteps'):
                # convert the timestep_indices to a timestep
                timesteps = self.sd.noise_scheduler.timesteps[timestep_indices.long()]
                
            with self.timer('prepare_noise'):
                # get noise
                noise = self.get_noise(latents, batch_size, dtype=dtype, batch=batch, timestep=timesteps)

                # add dynamic noise offset. Dynamic noise is offsetting the noise to the same channelwise mean as the latents
                # this will negate any noise offsets
                if self.train_config.dynamic_noise_offset and not is_reg:
                    latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
                    # subtract channel mean to that we compensate for the mean of the latents on the noise offset per channel
                    noise = noise + latents_channel_mean

                if self.train_config.loss_target == 'differential_noise':
                    differential = latents - unaugmented_latents
                    # add noise to differential
                    # noise = noise + differential
                    noise = noise + (differential * 0.5)
                    # noise = value_map(differential, 0, torch.abs(differential).max(), 0, torch.abs(noise).max())
                    latents = unaugmented_latents

                noise_multiplier = self.train_config.noise_multiplier
                
                s = (noise.shape[0], noise.shape[1], 1, 1)
                if len(noise.shape) == 5:
                    # if we have a 5d tensor, then we need to do it on a per batch item, per channel basis, per frame
                    s = (noise.shape[0], noise.shape[1], noise.shape[2], 1, 1)
                
                noise = noise * noise_multiplier
                
                if self.train_config.do_signal_correction_noise:
                    batch_noise = latents.clone().to(noise.device, dtype=noise.dtype)
                    scn_scale = torch.randn(
                        batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                        device=batch_noise.device, 
                        dtype=batch_noise.dtype
                    ) * self.train_config.signal_correction_noise_scale
                    batch_noise = batch_noise * scn_scale
                    noise = noise + batch_noise 
                
                if self.train_config.do_batch_noise_correction:
                    if latents.shape[0] == 1:
                        # if we only have a batch size of 1, then we cant do batch noise correction, so we skip it
                        print_acc("Skipping batch noise correction because batch size is 1, increase batch size and num_repeats to use this feature")
                    else:
                        # shuffle tensors ensuring that no tensor is in the same position as before
                        batch_noise = latents.clone().roll(shifts=torch.randint(1, latents.shape[0], (1,)).item(), dims=0).to(noise.device, dtype=noise.dtype)
                        batch_noise_scale = torch.randn(
                            batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                            device=batch_noise.device,
                            dtype=batch_noise.dtype
                        ) * self.train_config.batch_noise_correction_scale
                        batch_noise = batch_noise * batch_noise_scale
                        noise = noise + batch_noise
                
                if self.train_config.random_noise_shift > 0.0:
                    # get random noise -1 to 1
                    noise_shift = torch.randn(
                        batch_size, latents.shape[1], 1, 1,
                        device=noise.device,
                        dtype=noise.dtype
                    ) * self.train_config.random_noise_shift
                    # add to noise
                    noise += noise_shift
                
                if self.train_config.random_noise_multiplier > 0.0:
                    sigma = self.train_config.random_noise_multiplier
                    noise_multiplier = torch.exp(torch.randn(s, device=noise.device, dtype=noise.dtype) * sigma)
                
            with self.timer('make_noisy_latents'):

                latent_multiplier = self.train_config.latent_multiplier

                # handle adaptive scaling mased on std
                if self.train_config.adaptive_scaling_factor:
                    std = latents.std(dim=(2, 3), keepdim=True)
                    normalizer = 1 / (std + 1e-6)
                    latent_multiplier = normalizer

                latents = latents * latent_multiplier
                
                if self.train_config.do_blank_stabilization:
                    # zero out latents with blank prompts
                    blank_latent = torch.zeros_like(latents)
                    for i, prompt in enumerate(conditioned_prompts):
                        if prompt.strip() == '':
                            latents[i] = blank_latent[i]
                
                batch.latents = latents

                # normalize latents to a mean of 0 and an std of 1
                # mean_zero_latents = latents - latents.mean()
                # latents = mean_zero_latents / mean_zero_latents.std()

                if batch.unconditional_latents is not None:
                    batch.unconditional_latents = batch.unconditional_latents * self.train_config.latent_multiplier


                noisy_latents = self.sd.add_noise(latents, noise, timesteps)

                # determine scaled noise
                # todo do we need to scale this or does it always predict full intensity
                # noise = noisy_latents - latents

                # https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170C17-L1171C77
                if self.train_config.loss_target == 'source' or self.train_config.loss_target == 'unaugmented':
                    sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # add it to the batch
                    batch.sigmas = sigmas
                    # todo is this for sdxl? find out where this came from originally
                    # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

            def double_up_tensor(tensor: torch.Tensor):
                if tensor is None:
                    return None
                return torch.cat([tensor, tensor], dim=0)

            if do_double:
                if self.model_config.refiner_name_or_path:
                    # apply refiner double up
                    refiner_timesteps = torch.randint(
                        max_noise_steps,
                        self.train_config.max_denoising_steps,
                        (batch_size,),
                        device=self.device_torch
                    )
                    refiner_timesteps = refiner_timesteps.long()
                    # add our new timesteps on to end
                    timesteps = torch.cat([timesteps, refiner_timesteps], dim=0)

                    refiner_noisy_latents = self.sd.noise_scheduler.add_noise(latents, noise, refiner_timesteps)
                    noisy_latents = torch.cat([noisy_latents, refiner_noisy_latents], dim=0)

                else:
                    # just double it
                    noisy_latents = double_up_tensor(noisy_latents)
                    timesteps = double_up_tensor(timesteps)

                noise = double_up_tensor(noise)
                # prompts are already updated above
                imgs = double_up_tensor(imgs)
                batch.mask_tensor = double_up_tensor(batch.mask_tensor)
                batch.control_tensor = double_up_tensor(batch.control_tensor)

            noisy_latent_multiplier = self.train_config.noisy_latent_multiplier

            if noisy_latent_multiplier != 1.0:
                noisy_latents = noisy_latents * noisy_latent_multiplier

            # remove grads for these
            noisy_latents.requires_grad = False
            noisy_latents = noisy_latents.detach()
            noise.requires_grad = False
            noise = noise.detach()

        return noisy_latents, noise, timesteps, conditioned_prompts, imgs

    def setup_adapter(self):
        # t2i adapter
        is_t2i = self.adapter_config.type == 't2i'
        is_control_net = self.adapter_config.type == 'control_net'
        if self.adapter_config.type == 't2i':
            suffix = 't2i'
        elif self.adapter_config.type == 'control_net':
            suffix = 'cn'
        elif self.adapter_config.type == 'clip':
            suffix = 'clip'
        elif self.adapter_config.type == 'reference':
            suffix = 'ref'
        elif self.adapter_config.type.startswith('ip'):
            suffix = 'ip'
        else:
            suffix = 'adapter'
        adapter_name = self.name
        if self.network_config is not None:
            adapter_name = f"{adapter_name}_{suffix}"
        latest_save_path = self.get_latest_save_path(adapter_name)
        
        if latest_save_path is not None and not self.adapter_config.train:
            # the save path is for something else since we are not training
            latest_save_path = self.adapter_config.name_or_path

        dtype = get_torch_dtype(self.train_config.dtype)
        existing_adapter = getattr(self.sd, 'adapter', None)
        if existing_adapter is not None and hasattr(existing_adapter, 'detach'):
            existing_adapter.detach()
        if is_t2i:
            # if we do not have a last save path and we have a name_or_path,
            # load from that
            if latest_save_path is None and self.adapter_config.name_or_path is not None:
                self.adapter = T2IAdapter.from_pretrained(
                    self.adapter_config.name_or_path,
                    torch_dtype=get_torch_dtype(self.train_config.dtype),
                    varient="fp16",
                    # use_safetensors=True,
                )
            else:
                self.adapter = T2IAdapter(
                    in_channels=self.adapter_config.in_channels,
                    channels=self.adapter_config.channels,
                    num_res_blocks=self.adapter_config.num_res_blocks,
                    downscale_factor=self.adapter_config.downscale_factor,
                    adapter_type=self.adapter_config.adapter_type,
                )
        elif is_control_net:
            if self.adapter_config.name_or_path is None:
                raise ValueError("ControlNet requires a name_or_path to load from currently")
            load_from_path = self.adapter_config.name_or_path
            if latest_save_path is not None:
                load_from_path = latest_save_path
            self.adapter = ControlNetModel.from_pretrained(
                load_from_path,
                torch_dtype=get_torch_dtype(self.train_config.dtype),
            )
        elif self.adapter_config.type == 'clip':
            self.adapter = ClipVisionAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type == 'reference':
            self.adapter = ReferenceAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type.startswith('ip'):
            self.adapter = IPAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
            if self.train_config.gradient_checkpointing:
                self.adapter.enable_gradient_checkpointing()
        else:
            self.adapter = CustomAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
                train_config=self.train_config,
            )
        self.adapter.to(self.device_torch, dtype=dtype)
        if latest_save_path is not None and not is_control_net:
            # load adapter from path
            print_acc(f"Loading adapter from {latest_save_path}")
            if is_t2i:
                loaded_state_dict = load_t2i_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
            elif self.adapter_config.type.startswith('ip'):
                # ip adapter
                loaded_state_dict = load_ip_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype,
                    direct_load=self.adapter_config.train_only_image_encoder
                )
                self.adapter.load_state_dict(loaded_state_dict)
            else:
                # custom adapter
                loaded_state_dict = load_custom_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
        if latest_save_path is not None and self.adapter_config.train:
            self.load_training_state_from_metadata(latest_save_path)
        # set trainable params
        self.sd.adapter = self.adapter

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        self._free_memory_on_startup()
        self._cleanup_all_caches_on_startup()
        self._compact_latent_shards_on_startup()
        # run base process run
        BaseTrainProcess.run(self)
        params = []

        ### HOOK ###
        self.hook_before_model_load()
        model_config_to_load = copy.deepcopy(self.model_config)

        if self.is_fine_tuning or self.train_config.merge_network_on_save:
            # get the latest checkpoint
            # check to see if we have a latest save
            latest_save_path = self.get_latest_save_path()

            if latest_save_path is not None:
                print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                model_config_to_load.name_or_path = latest_save_path
                self.load_training_state_from_metadata(latest_save_path)

        ModelClass = get_model_class(self.model_config)
        # if the model class has get_train_scheduler static method
        if hasattr(ModelClass, 'get_train_scheduler'):
            sampler = ModelClass.get_train_scheduler()
        else:
            # get the noise scheduler
            arch = 'sd'
            if self.model_config.is_pixart:
                arch = 'pixart'
            if self.model_config.is_flux:
                arch = 'flux'
            if self.model_config.is_lumina2:
                arch = 'lumina2'
            sampler = get_sampler(
                self.train_config.noise_scheduler,
                {
                    "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
                },
                arch=arch,
            )

        if self.train_config.train_refiner and self.model_config.refiner_name_or_path is not None and self.network_config is None:
            previous_refiner_save = self.get_latest_save_path(self.job.name + '_refiner')
            if previous_refiner_save is not None:
                model_config_to_load.refiner_name_or_path = previous_refiner_save
                self.load_training_state_from_metadata(previous_refiner_save)

        self._resolve_auto_low_vram(model_config_to_load)
        self._resolve_auto_layer_offloading_percents(model_config_to_load)

        self.sd = ModelClass(
            # todo handle single gpu and multi gpu here
            # device=self.device,
            device=self.accelerator.device,
            model_config=model_config_to_load,
            dtype=self.train_config.dtype,
            custom_pipeline=self.custom_pipeline,
            noise_scheduler=sampler,
        )
        
        self.hook_after_sd_init_before_load()
        # run base sd process run
        self.sd.load_model()
        
        # compile the model if needed
        if self.model_config.compile:
            try:
                torch.compile(self.sd.unet, dynamic=True, fullgraph=True, mode='max-autotune')
            except Exception as e:
                print_acc(f"Failed to compile model: {e}")
                print_acc("Continuing without compilation")

        self.sd.add_after_sample_image_hook(self.sample_step_hook)

        dtype = get_torch_dtype(self.train_config.dtype)

        # model is loaded from BaseSDProcess
        unet = self.sd.unet
        vae = self.sd.vae
        tokenizer = self.sd.tokenizer
        text_encoder = self.sd.text_encoder
        noise_scheduler = self.sd.noise_scheduler

        if self.train_config.xformers:
            vae.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    # if it has it
                    if hasattr(te, 'enable_xformers_memory_efficient_attention'):
                        te.enable_xformers_memory_efficient_attention()
        
        if self.train_config.attention_backend != 'native':
            if hasattr(vae, 'set_attention_backend'):
                vae.set_attention_backend(self.train_config.attention_backend)
            if hasattr(unet, 'set_attention_backend'):
                unet.set_attention_backend(self.train_config.attention_backend)
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'set_attention_backend'):
                        te.set_attention_backend(self.train_config.attention_backend)
            else:
                if hasattr(text_encoder, 'set_attention_backend'):
                    text_encoder.set_attention_backend(self.train_config.attention_backend)
        if self.train_config.sdp:
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # # check if we have sage and is flux
        # if self.sd.is_flux:
        #     # try_to_activate_sage_attn()
        #     try:
        #         from sageattention import sageattn
        #         from toolkit.models.flux_sage_attn import FluxSageAttnProcessor2_0
        #         model: FluxTransformer2DModel = self.sd.unet
        #         # enable sage attention on each block
        #         for block in model.transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
        #         for block in model.single_transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
                    
        #     except ImportError:
        #         print_acc("sage attention is not installed. Using SDP instead")

        if self.train_config.gradient_checkpointing:
            # if has method enable_gradient_checkpointing
            if hasattr(unet, 'enable_gradient_checkpointing'):
                unet.enable_gradient_checkpointing()
            elif hasattr(unet, 'gradient_checkpointing'):
                unet.gradient_checkpointing = True
            else:
                print("Gradient checkpointing not supported on this model")
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'enable_gradient_checkpointing'):
                        te.enable_gradient_checkpointing()
                    if hasattr(te, "gradient_checkpointing_enable"):
                        te.gradient_checkpointing_enable()
            else:
                if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                    text_encoder.enable_gradient_checkpointing()
                if hasattr(text_encoder, "gradient_checkpointing_enable"):
                    text_encoder.gradient_checkpointing_enable()

        if self.sd.refiner_unet is not None:
            self.sd.refiner_unet.to(self.device_torch, dtype=dtype)
            self.sd.refiner_unet.requires_grad_(False)
            self.sd.refiner_unet.eval()
            if self.train_config.xformers:
                self.sd.refiner_unet.enable_xformers_memory_efficient_attention()
            if self.train_config.gradient_checkpointing:
                self.sd.refiner_unet.enable_gradient_checkpointing()

        if isinstance(text_encoder, list):
            for te in text_encoder:
                te.requires_grad_(False)
                te.eval()
        else:
            text_encoder.requires_grad_(False)
            text_encoder.eval()
        unet.to(self.device_torch, dtype=dtype)
        unet.requires_grad_(False)
        unet.eval()
        vae = vae.to(torch.device('cpu'), dtype=dtype)
        vae.requires_grad_(False)
        vae.eval()
        if self.train_config.learnable_snr_gos:
            self.snr_gos = LearnableSNRGamma(
                self.sd.noise_scheduler, device=self.device_torch
            )
            # check to see if previous settings exist
            path_to_load = os.path.join(self.save_root, 'learnable_snr.json')
            if os.path.exists(path_to_load):
                with open(path_to_load, 'r') as f:
                    json_data = json.load(f)
                    if 'offset' in json_data:
                        # legacy
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset'], device=self.device_torch)
                    else:
                        self.snr_gos.offset_1.data = torch.tensor(json_data['offset_1'], device=self.device_torch)
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset_2'], device=self.device_torch)
                    self.snr_gos.scale.data = torch.tensor(json_data['scale'], device=self.device_torch)
                    self.snr_gos.gamma.data = torch.tensor(json_data['gamma'], device=self.device_torch)

        self.hook_after_model_load()
        flush()
        if not self.is_fine_tuning:
            if self.network_config is not None:
                # TODO should we completely switch to LycorisSpecialNetwork?
                network_kwargs = self.network_config.network_kwargs
                is_lycoris = False
                is_lorm = self.network_config.type.lower() == 'lorm'
                # default to LoCON if there are any conv layers or if it is named
                NetworkClass = LoRASpecialNetwork
                if self.network_config.type.lower() == 'locon' or self.network_config.type.lower() == 'lycoris':
                    NetworkClass = LycorisSpecialNetwork
                    is_lycoris = True

                if is_lorm:
                    network_kwargs['ignore_if_contains'] = lorm_ignore_if_contains
                    network_kwargs['parameter_threshold'] = lorm_parameter_threshold
                    network_kwargs['target_lin_modules'] = LORM_TARGET_REPLACE_MODULE

                # if is_lycoris:
                #     preset = PRESET['full']
                # NetworkClass.apply_preset(preset)
                
                if hasattr(self.sd, 'target_lora_modules'):
                    network_kwargs['target_lin_modules'] = self.sd.target_lora_modules

                self.network = NetworkClass(
                    text_encoder=text_encoder,
                    unet=self.sd.get_model_to_train(),
                    lora_dim=self.network_config.linear,
                    multiplier=1.0,
                    alpha=self.network_config.linear_alpha,
                    train_unet=self.train_config.train_unet,
                    train_text_encoder=self.train_config.train_text_encoder,
                    conv_lora_dim=self.network_config.conv,
                    conv_alpha=self.network_config.conv_alpha,
                    is_sdxl=self.model_config.is_xl or self.model_config.is_ssd,
                    is_v2=self.model_config.is_v2,
                    is_v3=self.model_config.is_v3,
                    is_pixart=self.model_config.is_pixart,
                    is_auraflow=self.model_config.is_auraflow,
                    is_flux=self.model_config.is_flux,
                    is_lumina2=self.model_config.is_lumina2,
                    is_ssd=self.model_config.is_ssd,
                    is_vega=self.model_config.is_vega,
                    dropout=self.network_config.dropout,
                    use_text_encoder_1=self.model_config.use_text_encoder_1,
                    use_text_encoder_2=self.model_config.use_text_encoder_2,
                    use_bias=is_lorm,
                    is_lorm=is_lorm,
                    network_config=self.network_config,
                    network_type=self.network_config.type,
                    transformer_only=self.network_config.transformer_only,
                    is_transformer=self.sd.is_transformer,
                    base_model=self.sd,
                    **network_kwargs
                )


                # todo switch everything to proper mixed precision like this
                self.network.force_to(self.device_torch, dtype=torch.float32)
                # give network to sd so it can use it
                self.sd.network = self.network
                self.network._update_torch_multiplier()

                self.network.apply_to(
                    text_encoder,
                    unet,
                    self.train_config.train_text_encoder,
                    self.train_config.train_unet
                )

                # we cannot merge in if quantized
                if self.model_config.quantize or self.model_config.layer_offloading:
                    # todo find a way around this
                    self.network.can_merge_in = False

                if is_lorm:
                    self.network.is_lorm = True
                    # make sure it is on the right device
                    self.sd.unet.to(self.sd.device, dtype=dtype)
                    original_unet_param_count = count_parameters(self.sd.unet)
                    self.network.setup_lorm()
                    new_unet_param_count = original_unet_param_count - self.network.calculate_lorem_parameter_reduction()

                    print_lorm_extract_details(
                        start_num_params=original_unet_param_count,
                        end_num_params=new_unet_param_count,
                        num_replaced=len(self.network.get_all_modules()),
                    )

                self.network.prepare_grad_etc(text_encoder, unet)
                flush()

                # LyCORIS doesnt have default_lr
                config = {
                    'text_encoder_lr': self.train_config.lr,
                    'unet_lr': self.train_config.lr,
                }
                sig = inspect.signature(self.network.prepare_optimizer_params)
                if 'default_lr' in sig.parameters:
                    config['default_lr'] = self.train_config.lr
                if 'learning_rate' in sig.parameters:
                    config['learning_rate'] = self.train_config.lr
                params_net = self.network.prepare_optimizer_params(
                    **config
                )

                params += params_net

                if self.train_config.gradient_checkpointing:
                    self.network.enable_gradient_checkpointing()

                lora_name = self.name
                # need to adapt name so they are not mixed up
                if self.named_lora:
                    lora_name = f"{lora_name}_LoRA"

                latest_save_path = self.get_latest_save_path(lora_name)
                extra_weights = None
                if latest_save_path is not None and not self.train_config.merge_network_on_save:
                    print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                    print_acc(f"Loading from {latest_save_path}")
                    extra_weights = self.load_weights(latest_save_path)
                    self.network.multiplier = 1.0
                
                if self.network_config.layer_offloading:
                    MemoryManager.attach(
                        self.network,
                        self.device_torch
                    )

            if self.embed_config is not None:
                # we are doing embedding training as well
                self.embedding = Embedding(
                    sd=self.sd,
                    embed_config=self.embed_config
                )
                latest_save_path = self.get_latest_save_path(self.embed_config.trigger)
                # load last saved weights
                if latest_save_path is not None:
                    self.embedding.load_embedding_from_file(latest_save_path, self.device_torch)
                    if self.embedding.step > 1:
                        self.step_num = self.embedding.step
                        self.start_step = self.step_num

                # self.step_num = self.embedding.step
                # self.start_step = self.step_num
                params.append({
                    'params': list(self.embedding.get_trainable_params()),
                    'lr': self.train_config.embedding_lr
                })

                flush()
            
            if self.decorator_config is not None:
                self.decorator = Decorator(
                    num_tokens=self.decorator_config.num_tokens,
                    token_size=4096 # t5xxl hidden size for flux
                )
                latest_save_path = self.get_latest_save_path()
                # load last saved weights
                if latest_save_path is not None:
                    state_dict = load_file(latest_save_path)
                    self.decorator.load_state_dict(state_dict)
                    self.load_training_state_from_metadata(latest_save_path)
                    
                params.append({
                    'params': list(self.decorator.parameters()),
                    'lr': self.train_config.lr
                })
                
                # give it to the sd network
                self.sd.decorator = self.decorator
                self.decorator.to(self.device_torch, dtype=torch.float32)
                self.decorator.train()

                flush()

            if self.adapter_config is not None:
                self.setup_adapter()
                if self.adapter_config.train:

                    if isinstance(self.adapter, IPAdapter):
                        # we have custom LR groups for IPAdapter
                        adapter_param_groups = self.adapter.get_parameter_groups(self.train_config.adapter_lr)
                        for group in adapter_param_groups:
                            params.append(group)
                    else:
                        # set trainable params
                        params.append({
                            'params': list(self.adapter.parameters()),
                            'lr': self.train_config.adapter_lr
                        })

                if self.train_config.gradient_checkpointing:
                    self.adapter.enable_gradient_checkpointing()
                flush()

            params = self.load_additional_training_modules(params)

        else:  # no network, embedding or adapter
            # set the device state preset before getting params
            self.sd.set_device_state(self.get_params_device_state_preset)

            # params = self.get_params()
            if len(params) == 0:
                # will only return savable weights and ones with grad
                params = self.sd.prepare_optimizer_params(
                    unet=self.train_config.train_unet,
                    text_encoder=self.train_config.train_text_encoder,
                    text_encoder_lr=self.train_config.lr,
                    unet_lr=self.train_config.lr,
                    default_lr=self.train_config.lr,
                    refiner=self.train_config.train_refiner and self.sd.refiner_unet is not None,
                    refiner_lr=self.train_config.refiner_lr,
                )
            # we may be using it for prompt injections
            if self.adapter_config is not None and self.adapter is None:
                self.setup_adapter()
        flush()
        ### HOOK ###
        params = self.hook_add_extra_train_params(params)
        self.params = params
        # self.params = []

        # for param in params:
        #     if isinstance(param, dict):
        #         self.params += param['params']
        #     else:
        #         self.params.append(param)

        if self.train_config.start_step is not None:
            self.step_num = self.train_config.start_step
            self.start_step = self.step_num

        optimizer_type = self.train_config.optimizer.lower()
        
        # esure params require grad
        self.ensure_params_requires_grad(force=True)
        optimizer = get_optimizer(self.params, optimizer_type, learning_rate=self.train_config.lr,
                                  optimizer_params=self.train_config.optimizer_params)
        self.optimizer = optimizer
        
        # set it to do paramiter swapping
        if self.train_config.do_paramiter_swapping:
            # only works for adafactor, but it should have thrown an error prior to this otherwise
            self.optimizer.enable_paramiter_swapping(self.train_config.paramiter_swapping_factor)

        optimizer_state_file_path = os.path.join(self.save_root, 'optimizer.pt')
        if not os.path.exists(optimizer_state_file_path):
            imported_optimizer_state_path = self._get_imported_optimizer_state_path()
            if imported_optimizer_state_path is not None:
                optimizer_state_file_path = imported_optimizer_state_path

        if os.path.exists(optimizer_state_file_path):
            # try to load
            # previous param groups
            # previous_params = copy.deepcopy(optimizer.param_groups)
            previous_lrs = []
            for group in optimizer.param_groups:
                previous_lrs.append(group['lr'])

            load_optimizer = True
            if self.network is not None:
                if self.network.did_change_weights:
                    # do not load optimizer if the network changed, it will result in
                    # a double state that will oom.
                    load_optimizer = False

            if load_optimizer:
                try:
                    print_acc(f"Loading optimizer state from {optimizer_state_file_path}")
                    optimizer_state_dict = torch.load(optimizer_state_file_path, weights_only=True)
                    optimizer.load_state_dict(optimizer_state_dict)
                    del optimizer_state_dict
                    flush()
                except Exception as e:
                    print_acc(f"Failed to load optimizer state from {optimizer_state_file_path}")
                    print_acc(e)

            # update the optimizer LR from the params
            print_acc(f"Updating optimizer LR from params")
            if len(previous_lrs) > 0:
                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = previous_lrs[i]
                    group['initial_lr'] = previous_lrs[i]

            # Update the learning rates if they changed
            # optimizer.param_groups = previous_params

        ### HOOk ###
        self.before_dataset_load()
        # load datasets if passed in the root process
        if self.datasets is not None:
            self.data_loader = get_dataloader_from_datasets(
                self.datasets,
                self.train_config.batch_size,
                self.sd,
                cache_progress_hook=self.cache_progress_hook,
                global_bucket_batching=getattr(
                    self.train_config, 'global_bucket_batching', False
                ),
            )
        if self.datasets_reg is not None:
            self.data_loader_reg = get_dataloader_from_datasets(
                self.datasets_reg,
                self.train_config.batch_size,
                self.sd,
                cache_progress_hook=self.cache_progress_hook,
                global_bucket_batching=getattr(
                    self.train_config, 'global_bucket_batching', False
                ),
            )

        # Resolve epochs -> steps now that the dataset is loaded (exact, using the
        # real batches-per-epoch), THEN build the LR scheduler so it gets the
        # resolved total_iters. The progress bar and training loop below also read
        # the resolved self.train_config.steps.
        self._resolve_epoch_based_steps()
        # If continuing from a loaded pretrained LoRA, train the configured amount ON
        # TOP of its finished steps (extends the target). Must run after the epoch
        # resolution and before the scheduler / progress bar / loop read steps.
        self._apply_train_on_top_of_loaded_steps()

        lr_scheduler_params = self.train_config.lr_scheduler_params
        # make sure it had bare minimum
        if 'max_iterations' not in lr_scheduler_params:
            lr_scheduler_params['total_iters'] = self.train_config.steps

        lr_scheduler = get_lr_scheduler(
            self.train_config.lr_scheduler,
            optimizer,
            **lr_scheduler_params
        )
        self.lr_scheduler = lr_scheduler

        saved_dataloader_resume_state = self._load_saved_dataloader_resume_state()
        if saved_dataloader_resume_state is not None:
            self._apply_loaded_dataloader_resume_state(saved_dataloader_resume_state)

        flush()
        self.last_save_step = self.step_num
        ### HOOK ###
        self.hook_before_train_loop()

        if self.has_first_sample_requested and self.step_num <= 1 and not self.train_config.disable_sampling:
            print_acc("Generating first sample from first sample config")
            self.sample(0, is_first=True)

        skip_baseline_sample = self.train_config.skip_first_sample or self.train_config.disable_sampling
        if (
            not skip_baseline_sample
            and self.step_num <= 1
            and not self.train_config.force_first_sample
            and os.getenv('IS_AI_TOOLKIT_UI') == '1'
        ):
            skip_baseline_sample = True
            print_acc("Skipping baseline sample at startup for UI job. Enable Force First Sample to restore it.")

        # sample first
        if skip_baseline_sample:
            print_acc("Skipping first sample due to config setting")
        elif self.step_num <= 1 or self.train_config.force_first_sample:
            print_acc("Generating baseline samples before training")
            self.sample(self.step_num)
        
        if self.accelerator.is_local_main_process:
            self.progress_bar = ToolkitProgressBar(
                total=self.train_config.steps,
                desc=self.job.name,
                leave=True,
                initial=self.step_num,
                iterable=range(0, self.train_config.steps),
            )
            self.progress_bar.pause()
        else:
            self.progress_bar = None

        if self.data_loader is not None:
            self.data_loader, dataloader_iterator = self._prepare_dataloader_with_windows_fallback(
                self.data_loader,
                "train dataset",
            )
            dataloader = self.data_loader
        else:
            dataloader = None
            dataloader_iterator = None

        if self.data_loader_reg is not None:
            self.data_loader_reg, dataloader_iterator_reg = self._prepare_dataloader_with_windows_fallback(
                self.data_loader_reg,
                "regularization dataset",
            )
            dataloader_reg = self.data_loader_reg
        else:
            dataloader_reg = None
            dataloader_iterator_reg = None

        # zero any gradients
        optimizer.zero_grad(set_to_none=True)

        self.lr_scheduler.step(self.step_num)

        self.sd.set_device_state(self.train_device_state_preset)
        flush()
        # self.step_num = 0

        # print_acc(f"Compiling Model")
        # torch.compile(self.sd.unet, dynamic=True)

        # make sure all params require grad
        self.ensure_params_requires_grad(force=True)


        ###################################################################
        # TRAIN LOOP
        ###################################################################


        start_step_num = self.step_num
        # Loading/quantizing left cached GPU blocks and pinned staging buffers of
        # loading-time sizes behind; hand them back so training starts consolidated
        # (and so the VRAM baseline below measures the training state, not loading).
        release_cached_memory(printer=print_acc)
        # OOM diagnostics: seed the skip tally from earlier sessions (minus steps this
        # resume will re-run) and take a provisional "VRAM outside PyTorch" baseline, so
        # an OOM on the very first steps still has something to compare against.
        self._seed_oom_tally()
        self._vram_outside_pytorch_at_start = self._vram_outside_pytorch_gib()
        self._vram_baseline_provisional = self._vram_outside_pytorch_at_start is not None
        if self._oom_skip_total or self._oom_rescued_total:
            self.hook_oom_tally_changed()
        # where the process's RAM goes; repeated at every checkpoint save, so growth
        # during a run shows up as a rising "not attributed" line
        log_memory_breakdown(self.sd, getattr(self, "network", None),
                             label="at start of training", printer=print_acc)
        did_first_flush = False
        flush_next = False
        if self.accelerator.is_main_process:
            remove_stale_temps(self.save_root, printer=print_acc)
        self._in_train_loop = True
        for step in range(start_step_num, self.train_config.steps):
            if self.train_config.do_paramiter_swapping:
                self.optimizer.optimizer.swap_paramiters()
            self.timer.start('train_loop')
            if flush_next:
                flush()
                flush_next = False
            if self.train_config.do_random_cfg:
                self.train_config.do_cfg = True
                self.train_config.cfg_scale = value_map(random.random(), 0, 1, 1.0, self.train_config.max_cfg_scale)
            self.step_num = step
            # default to true so various things can turn it off
            self.is_grad_accumulation_step = True
            if self.train_config.free_u:
                self.sd.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
            if self.progress_bar is not None:
                self.progress_bar.unpause()
            with torch.no_grad():
                # if is even step and we have a reg dataset, use that
                # todo improve this logic to send one of each through if we can buckets and batch size might be an issue
                is_reg_step = False
                is_save_step = self.save_config.save_every and self.step_num % self.save_config.save_every == 0
                saved_this_step = False
                self.step_was_saved = False  # read by stop handling once the step has ended
                is_sample_step = self.sample_config.sample_every and self.step_num % self.sample_config.sample_every == 0
                if self.train_config.disable_sampling:
                    is_sample_step = False

                batch_list = []

                for b in range(self.train_config.gradient_accumulation):
                    # keep track to alternate on an accumulation step for reg   
                    batch_step = step
                    # don't do a reg step on sample or save steps as we dont want to normalize on those
                    if batch_step % 2 == 0 and dataloader_reg is not None and not is_save_step and not is_sample_step:
                        try:
                            with self.timer('get_batch:reg'):
                                batch = next(dataloader_iterator_reg)
                            mark_dataloader_batch_processed(dataloader_reg)
                        except StopIteration:
                            with self.timer('reset_batch:reg'):
                                # hit the end of an epoch, reset
                                if self.progress_bar is not None:
                                    self.progress_bar.pause()
                                start_next_dataloader_epoch(dataloader_reg)
                                dataloader_iterator_reg = iter(dataloader_reg)
                                trigger_dataloader_setup_epoch(dataloader_reg)

                            with self.timer('get_batch:reg'):
                                batch = next(dataloader_iterator_reg)
                            mark_dataloader_batch_processed(dataloader_reg)
                            if self.progress_bar is not None:
                                self.progress_bar.unpause()
                        is_reg_step = True
                    elif dataloader is not None:
                        try:
                            with self.timer('get_batch'):
                                batch = next(dataloader_iterator)
                            mark_dataloader_batch_processed(dataloader)
                        except StopIteration:
                            with self.timer('reset_batch'):
                                # hit the end of an epoch, reset
                                if self.progress_bar is not None:
                                    self.progress_bar.pause()
                                start_next_dataloader_epoch(dataloader)
                                dataloader_iterator = iter(dataloader)
                                trigger_dataloader_setup_epoch(dataloader)
                                self.epoch_num += 1
                                if self.train_config.gradient_accumulation_steps == -1:
                                    # if we are accumulating for an entire epoch, trigger a step
                                    self.is_grad_accumulation_step = False
                                    self.grad_accumulation_step = 0
                            with self.timer('get_batch'):
                                batch = next(dataloader_iterator)
                            mark_dataloader_batch_processed(dataloader)
                            if self.progress_bar is not None:
                                self.progress_bar.unpause()
                    else:
                        batch = None
                    batch_list.append(batch)
                    batch_step += 1

                # setup accumulation
                if self.train_config.gradient_accumulation_steps == -1:
                    # epoch is handling the accumulation, dont touch it
                    pass
                else:
                    # determine if we are accumulating or not
                    # since optimizer step happens in the loop, we trigger it a step early
                    # since we cannot reprocess it before them
                    optimizer_step_at = self.train_config.gradient_accumulation_steps
                    is_optimizer_step = self.grad_accumulation_step >= optimizer_step_at
                    self.is_grad_accumulation_step = not is_optimizer_step
                    if is_optimizer_step:
                        self.grad_accumulation_step = 0

            # flush()
            ### HOOK ###
            if self.torch_profiler is not None:
                self.torch_profiler.start()
            self.hook_before_train_step(batch_list)
            loss_dict = None
            oom_info = None
            first_oom_info = None
            # A step writes results back into its batches (scaled latents, doubled
            # tensors, sigmas, ...) and advances the multistage counters. Snapshot both
            # so a retry after an OOM starts from exactly what the first attempt saw --
            # re-running on already-processed batches would scale latents twice.
            batch_snapshots = [(b, dict(b.__dict__)) for b in batch_list
                               if b is not None and hasattr(b, "__dict__")]
            boundary_snapshot = (self.current_boundary_index, self.steps_this_boundary)
            for attempt in range(2):  # the step, then at most one retry after cleaning up
                did_oom = False
                oom_info = None
                # per-attempt OOM bookkeeping; hook_train_loop updates these as it goes
                self._current_microbatch_index = None
                self._oom_phase = None
                self._oom_lora_updated = False
                try:
                    with self.accelerator.accumulate(self.modules_being_trained):
                        loss_dict = self.hook_train_loop(batch_list)
                except torch.cuda.OutOfMemoryError as e:
                    did_oom = True
                    oom_info = self._describe_oom(e)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        did_oom = True
                        oom_info = self._describe_oom(e)
                    else:
                        raise  # not an OOM; surface real errors
                if not did_oom:
                    break
                # one retry only; only for trainers that track the update flag; and never
                # re-run a step whose LoRA update already started -- that would apply the
                # same batch twice
                will_retry = (attempt == 0 and getattr(self, "supports_oom_retry", False)
                              and not (oom_info or {}).get("lora_updated"))
                if will_retry:
                    # restore BEFORE the clean-up, so the GPU tensors the failed attempt
                    # stored into the batches are released by the flush below
                    for b, snap in batch_snapshots:
                        b.__dict__.clear()
                        b.__dict__.update(snap)
                    self.current_boundary_index, self.steps_this_boundary = boundary_snapshot
                optimizer.zero_grad(set_to_none=True)
                # The layer offloader's staged weights live in module-global ring
                # buffers, so gc + empty_cache() below cannot touch them: without
                # this the retry starts at the same VRAM watermark that just OOM'd.
                # Synchronizes internally before dropping the references.
                release_offload_buffers()
                if will_retry:
                    # flush() empties the cache BEFORE collecting garbage; collect first so
                    # everything the failed attempt left is actually returned to the driver
                    import gc as _gc
                    _gc.collect()
                flush()
                torch.cuda.ipc_collect()
                if not will_retry:
                    break
                # Most OOMs here are fragmentation (plenty reserved, no single usable
                # block); the clean-up above consolidates free memory, so give the same
                # images one more chance instead of dropping them for the whole phase.
                first_oom_info = oom_info or {}
                print_acc(f"OOM at step {self.step_num} ({first_oom_info.get('stage') or 'unknown'}); "
                          f"memory freed, retrying the step once")
            batch_snapshots = None
            if did_oom:
                self.num_consecutive_oom += 1
                aborting = self.num_consecutive_oom > 3
                if aborting:
                    outcome = "run stopped"
                elif (oom_info or {}).get("lora_updated"):
                    outcome = "LoRA already updated"
                elif first_oom_info is not None:
                    outcome = "skipped after retry"
                else:
                    outcome = "skipped"
                # say exactly what was lost and why, then keep going (or stop)
                self._record_oom_skip(oom_info or {}, batch_list, outcome=outcome, first_info=first_oom_info)
                if aborting:
                    raise RuntimeError("OOM on 4 training steps in a row, aborting training (details in skipped_oom.txt)")
            else:
                self.num_consecutive_oom = 0
                if first_oom_info is not None:
                    self._record_oom_skip(first_oom_info, batch_list, outcome="rescued by retry")
                if self._vram_outside_pytorch_at_start is None or self._vram_baseline_provisional:
                    # replace the provisional baseline with one taken after a full step, so
                    # lazily loaded CUDA kernels are not mistaken for other programs
                    self._vram_outside_pytorch_at_start = self._vram_outside_pytorch_gib()
                    self._vram_baseline_provisional = False
            self.hook_after_train_step()
            if self.torch_profiler is not None:
                torch.cuda.synchronize()  # Make sure all CUDA ops are done
                self.torch_profiler.stop()
                
                print("\n==== Profile Results ====")
                print(self.torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
            self.timer.stop('train_loop')
            train_loop_timings = self.timer.timers.get('train_loop')
            train_loop_elapsed = train_loop_timings[-1] if train_loop_timings is not None and len(train_loop_timings) > 0 else None
            if did_oom or first_oom_info is not None:
                # a retried or failed step's wall time is not a real step time
                train_loop_elapsed = None
            progress_bar_was_paused = self.progress_bar.paused if self.progress_bar is not None else True
            if self.progress_bar is not None and not progress_bar_was_paused:
                self.progress_bar.pause()
            try:
                self._maybe_force_flush_lazy_latent_buffers(train_loop_elapsed)
            finally:
                if self.progress_bar is not None and not progress_bar_was_paused:
                    self.progress_bar.unpause()
            if not did_first_flush:
                flush()
                did_first_flush = True
            # flush()
            # setup the networks to gradient checkpointing and everything works
            if self.adapter is not None and isinstance(self.adapter, ReferenceAdapter):
                self.adapter.clear_memory()

            with torch.no_grad():
                # torch.cuda.empty_cache()
                # if optimizer has get_lrs method, then use it
                learning_rate = 0.0
                if not did_oom and loss_dict is not None:
                    if hasattr(optimizer, 'get_avg_learning_rate'):
                        learning_rate = optimizer.get_avg_learning_rate()
                    elif hasattr(optimizer, 'get_learning_rates'):
                        learning_rate = optimizer.get_learning_rates()[0]
                    elif self.train_config.optimizer.lower().startswith('dadaptation') or \
                            self.train_config.optimizer.lower().startswith('prodigy'):
                        learning_rate = (
                                optimizer.param_groups[0]["d"] *
                                optimizer.param_groups[0]["lr"]
                        )
                    else:
                        learning_rate = optimizer.param_groups[0]['lr']

                    prog_bar_string = f"lr: {learning_rate:.1e}"
                    for key, value in loss_dict.items():
                        prog_bar_string += f" {key}: {value:.3e}"

                    if self.progress_bar is not None:
                        self.progress_bar.set_postfix_str(prog_bar_string)

                # Clean up every accumulated batch DTO, not just the last one.
                with self.timer('batch_cleanup'):
                    for batch_item in batch_list:
                        if isinstance(batch_item, DataLoaderBatchDTO):
                            batch_item.cleanup()

                # don't do on first step (unless it is a new step a resume continues with, e.g. right
                # after a stop-save at save_every*k-1: skipping it would skip a regular save)
                if self.step_num != self.start_step or getattr(self, "_resumed_at_new_step", False):
                    if is_sample_step or is_save_step:
                        self.accelerator.wait_for_everyone()
                        
                    if is_save_step:
                        self.accelerator
                        # print above the progress bar
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        print_acc(f"\nSaving at step {self.step_num}")
                        self.save(self.step_num)
                        saved_this_step = True
                        self.step_was_saved = True
                        self.ensure_params_requires_grad()
                        # clear any grads
                        optimizer.zero_grad(set_to_none=True)
                        flush()
                        flush_next = True
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()
                            
                    if is_sample_step and not self.save_before_stop_requested():
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        flush()
                        # print above the progress bar
                        if self.train_config.free_u:
                            self.sd.pipeline.disable_freeu()
                        self.sample(self.step_num)
                        if self.unload_text_encoder_for_training:
                            # make sure the text encoder is unloaded
                            self.sd.text_encoder_to('cpu')
                        flush()

                        self.ensure_params_requires_grad()
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()

                    if self.logging_config.log_every and self.step_num % self.logging_config.log_every == 0:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        with self.timer('log_to_tensorboard'):
                            # log to tensorboard
                            if self.accelerator.is_main_process:
                                if self.writer is not None:
                                    if loss_dict is not None:
                                        for key, value in loss_dict.items():
                                            self.writer.add_scalar(f"{key}", value, self.step_num)
                                        self.writer.add_scalar(f"lr", learning_rate, self.step_num)
                                if self.progress_bar is not None:
                                    self.progress_bar.unpause()
                        
                        if self.accelerator.is_main_process:
                            # log to logger
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            if loss_dict is not None:
                                for key, value in loss_dict.items():
                                    self.logger.log({
                                        f'loss/{key}': value,
                                    })
                    elif self.logging_config.log_every is None:
                        if self.accelerator.is_main_process:
                            # log every step
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            for key, value in loss_dict.items():
                                self.logger.log({
                                    f'loss/{key}': value,
                                })


                    if self.performance_log_every > 0 and self.step_num % self.performance_log_every == 0:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        # print the timers and clear them
                        self.timer.print()
                        self.timer.reset()
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()
                
                # commit log
                if self.accelerator.is_main_process:
                    with self.timer('commit_logger'):
                        self.logger.commit(step=self.step_num)

                # sets progress bar to match out step
                if self.progress_bar is not None:
                    self.progress_bar.update(step - self.progress_bar.n)

                #############################
                # End of step
                #############################

                self._poll_background_save()
                # a stop was requested: save this finished step first (it then stops in end_step_hook)
                if self.save_before_stop_requested():
                    self._stop_save_step = self.step_num  # the checkpoint this stop leaves behind
                    self._stop_save_reused = False
                    if saved_this_step:
                        pass  # the regular save of this step covers it
                    elif self._checkpoint_exists_for_step(self.step_num):
                        # a resume without the data-loader state re-trains the step it resumed
                        # from; that checkpoint is on disk already and must not be overwritten
                        print_acc(f"\nStep {self.step_num} is already saved; stopping")
                        self._stop_save_reused = True
                        self.step_was_saved = True
                    else:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        print_acc(f"\nSaving at step {self.step_num} before stopping")
                        self._saving_before_stop = True
                        try:
                            self.save(self.step_num)
                        finally:
                            self._saving_before_stop = False
                        self.step_was_saved = True
                        self.ensure_params_requires_grad()
                        optimizer.zero_grad(set_to_none=True)
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()

                # update various steps
                self.step_num = step + 1
                self.grad_accumulation_step += 1
                self.end_step_hook()


        ###################################################################
        ##  END TRAIN LOOP
        ###################################################################
        self.accelerator.wait_for_everyone()
        if self.progress_bar is not None:
            self.progress_bar.close()
        if self.train_config.free_u:
            self.sd.pipeline.disable_freeu()
        if not self.train_config.disable_sampling:
            self.sample(self.step_num)
            self.logger.commit(step=self.step_num)
        print_acc("")
        if self.accelerator.is_main_process:
            self.save()
            self.logger.finish()
        # stopping is saving no longer: the final checkpoint is on disk
        self._in_train_loop = False
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            # push to hub
            if self.save_config.push_to_hub:
                if("HF_TOKEN" not in os.environ):
                    interpreter_login(new_session=False, write_permission=True)
                self.push_to_hub(
                    repo_id=self.save_config.hf_repo_id,
                    private=self.save_config.hf_private
                )
        self.cleanup_runtime_caches()
        del (
            self.sd,
            unet,
            noise_scheduler,
            optimizer,
            self.network,
            tokenizer,
            text_encoder,
        )

        flush()
        self.done_hook()

    def on_error(self, e: Exception):
        try:
            # a checkpoint still being written (e.g. the one saved for a stop) finishes first
            self._wait_for_background_save_on_exit()
        except BaseException as wait_error:
            print_acc(f"Error while waiting for the checkpoint being written: {wait_error!r}")
        self.cleanup_runtime_caches()

    def push_to_hub(
    self,
    repo_id: str,
    private: bool = False,
    ):  
        if not self.accelerator.is_main_process:
            return
        readme_content = self._generate_readme(repo_id)
        readme_path = os.path.join(self.save_root, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        api = HfApi()

        api.create_repo(
            repo_id,
            private=private,
            exist_ok=True
        )

        api.upload_folder(
            repo_id=repo_id,
            folder_path=self.save_root,
            ignore_patterns=["*.yaml", "*.pt", ".current_step.json*"],
            repo_type="model",
        )


    def _generate_readme(self, repo_id: str) -> str:
        """Generates the content of the README.md file."""

        # Gather model info
        base_model = self.model_config.name_or_path
        instance_prompt = self.trigger_word if hasattr(self, "trigger_word") else None
        if base_model == "black-forest-labs/FLUX.1-schnell":
            license = "apache-2.0"
        elif base_model == "black-forest-labs/FLUX.1-dev":
            license = "other"
            license_name = "flux-1-dev-non-commercial-license"
            license_link = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"
        else:
            license = "creativeml-openrail-m"
        tags = [
            "text-to-image",
        ]
        if self.model_config.is_xl:
            tags.append("stable-diffusion-xl")
        if self.model_config.is_flux:
            tags.append("flux")
        if self.model_config.is_lumina2:
            tags.append("lumina2")
        if self.model_config.is_v3:
            tags.append("sd3")
        if self.network_config:
            tags.extend(
                [
                    "lora",
                    "diffusers",
                    "template:sd-lora",
                    "ai-toolkit",
                ]
            )

        # Generate the widget section
        widgets = []
        sample_image_paths = []
        samples_dir = os.path.join(self.save_root, "samples")
        if os.path.isdir(samples_dir):
            for filename in os.listdir(samples_dir):
                #The filenames are structured as 1724085406830__00000500_0.jpg
                #So here we capture the 2nd part (steps) and 3rd (index the matches the prompt)
                match = re.search(r"__(\d+)_(\d+)\.jpg$", filename)
                if match:
                    steps, index = int(match.group(1)), int(match.group(2))
                    #Here we only care about uploading the latest samples, the match with the # of steps
                    if steps == self.train_config.steps:
                        sample_image_paths.append((index, f"samples/{filename}"))

            # Sort by numeric index
            sample_image_paths.sort(key=lambda x: x[0])

            # Create widgets matching prompt with the index 
            for i, prompt in enumerate(self.sample_config.prompts):
                if i < len(sample_image_paths):
                    # Associate prompts with sample image paths based on the extracted index
                    _, image_path = sample_image_paths[i]
                    widgets.append(
                        {
                            "text": prompt,
                            "output": {
                                "url": image_path
                            },
                        }
                    )
        dtype = "torch.bfloat16" if self.model_config.is_flux else "torch.float16"
        # Construct the README content
        readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
license: {license}
{'license_name: ' + license_name if license == "other" else ""}
{'license_link: ' + license_link if license == "other" else ""}
---

# {self.job.name}
Model trained with [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit)
<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, etc.

Weights for this model are available in Safetensors format.

[Download](/{repo_id}/tree/main) them in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('{base_model}', torch_dtype={dtype}).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='{self.job.name}.safetensors')
image = pipeline('{instance_prompt if not widgets else self.sample_config.prompts[0]}').images[0]
image.save("my_image.png")
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

"""
        return readme_content
