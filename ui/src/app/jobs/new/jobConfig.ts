import { JobConfig, DatasetConfig, SliderConfig, AdapterConfig } from '@/types';

export const defaultDatasetConfig: DatasetConfig = {
  folder_path: '/path/to/images/folder',
  mask_path: null,
  mask_min_value: 0.1,
  default_caption: '',
  caption_ext: 'txt',
  must_include_caption: false,
  caption_dropout_rate: 0.05,
  cache_latents: false,
  cache_latents_to_disk: false,
  lazy_cache_latents: false,
  caches_only: false,
  lazy_cache_text_embeddings: true,
  text_embedding_cache_shard_size: 64,
  cache_dir_override: null,
  cache_dir_override_min_free_gb: 10,
  clip_image_path: null,
  clip_image_from_same_folder: false,
  clip_image_from_same_folder_balanced: false,
  num_workers: 2,
  prefetch_factor: 2,
  persistent_workers: true,
  pin_memory: true,
  enable_windows_num_workers: false,
  is_reg: false,
  network_weight: 1,
  resolution: [512, 768, 1024],
  controls: [],
  shrink_video_to_frames: true,
  num_frames: 1,
  flip_x: false,
  flip_y: false,
  num_repeats: 1,
};

export const defaultAdapterConfig: AdapterConfig = {
  type: 'ip',
  train: true,
  image_encoder_path: 'openai/clip-vit-large-patch14',
  image_encoder_arch: 'clip',
  train_image_encoder: false,
  train_only_image_encoder: false,
  clip_layer: 'last_hidden_state',
  test_img_path: [],
};

export const defaultNetworkConfig = {
  type: 'lora',
  linear: 32,
  linear_alpha: 32,
  conv: 16,
  conv_alpha: 16,
  lokr_full_rank: true,
  lokr_factor: -1,
  pretrained_lora_path: undefined,
  resume_training_state: false,
  network_kwargs: {
    ignore_if_contains: [],
  },
};

export const defaultSliderConfig: SliderConfig = {
  guidance_strength: 3.0,
  anchor_strength: 1.0,
  positive_prompt: 'person who is happy',
  negative_prompt: 'person who is sad',
  target_class: 'person',
  anchor_class: '',
};

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        trigger_word: null,
        performance_log_every: 10,
        cleanup_caches_on_startup: false,
        ui_step_update_interval: 1.0,
        ui_stop_poll_interval: 1.0,
        network: {
          ...defaultNetworkConfig,
          network_kwargs: {
            ...defaultNetworkConfig.network_kwargs,
            ignore_if_contains: [...defaultNetworkConfig.network_kwargs.ignore_if_contains],
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          save_format: 'diffusers',
          push_to_hub: false,
        },
        datasets: [defaultDatasetConfig],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: true,
          steps: 3000,
          gradient_accumulation: 1,
          train_by_epoch: false,
          epochs: 10,
          global_bucket_batching: false,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'adamw8bit',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: {
            weight_decay: 1e-4,
          },
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          adapter_lr: 0.0001,
          ema_config: {
            use_ema: false,
            ema_decay: 0.99,
          },
          skip_first_sample: true,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1.0,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
          loss_type: 'mse',
        },
        logging: {
          log_every: 10,
          use_ui_logger: true,
        },
        model: {
          name_or_path: 'ostris/Flex.1-alpha',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          arch: 'flex1',
          low_vram: false,
          auto_low_vram: false,
          auto_layer_offloading_percent: false,
          model_kwargs: {},
        },
        sample: {
          sampler: 'flowmatch',
          sample_every: 250,
          width: 1024,
          height: 1024,
          samples: [
            {
              prompt: 'woman with red hair, playing chess at the park, bomb going off in the background',
            },
            {
              prompt: 'a woman holding a coffee cup, in a beanie, sitting at a cafe',
            },
            {
              prompt: 'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
            },
            {
              prompt:
                'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
            },
            {
              prompt: 'a bear building a log cabin in the snow covered mountains',
            },
            {
              prompt: 'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
            },
            {
              prompt: 'hipster man with a beard, building a chair, in a wood shop',
            },
            {
              prompt:
                'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
            },
            {
              prompt: "a man holding a sign that says, 'this is a sign'",
            },
            {
              prompt:
                'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle',
            },
          ],
          neg: '',
          seed: 42,
          walk_seed: true,
          guidance_scale: 4,
          sample_steps: 25,
          num_frames: 1,
          fps: 1,
        },
      },
    ],
  },
  meta: {
    name: '[name]',
    version: '1.0',
  },
};

export const migrateJobConfig = (jobConfig: JobConfig): JobConfig => {
  const processConfig = jobConfig?.config?.process?.[0] as any;
  const defaultProcessConfig = defaultJobConfig.config.process[0];

  // Bail out safely on a partial/half-typed config (the Advanced YAML editor calls
  // this on every keystroke) instead of throwing on a missing process/model/etc.
  if (!processConfig) {
    return jobConfig;
  }

  // upgrade prompt strings to samples
  if (
    jobConfig?.config?.process &&
    processConfig?.sample &&
    Array.isArray(processConfig.sample.prompts) &&
    processConfig.sample.prompts.length > 0
  ) {
    let newSamples = [];
    for (const prompt of processConfig.sample.prompts) {
      newSamples.push({
        prompt: prompt,
      });
    }
    processConfig.sample.samples = newSamples;
    delete processConfig.sample.prompts;
  }

  // upgrade job from ui_trainer to diffusion_trainer
  if (jobConfig?.config?.process && processConfig?.type === 'ui_trainer') {
    processConfig.type = 'diffusion_trainer';
  }

  if (processConfig.model && 'auto_memory' in processConfig.model) {
    processConfig.model.layer_offloading = (processConfig.model.auto_memory || false) as boolean;
    delete processConfig.model.auto_memory;
  }

  if (processConfig.model && !('auto_layer_offloading_percent' in processConfig.model)) {
    processConfig.model.auto_layer_offloading_percent = defaultProcessConfig.model.auto_layer_offloading_percent;
  }

  if (processConfig.model && !('auto_low_vram' in processConfig.model)) {
    processConfig.model.auto_low_vram = defaultProcessConfig.model.auto_low_vram;
  }

  if (!('logging' in processConfig)) {
    processConfig.logging = {
      log_every: 10,
      use_ui_logger: true,
    };
  }

  if (!('performance_log_every' in processConfig)) {
    processConfig.performance_log_every = defaultProcessConfig.performance_log_every;
  }

  if (!processConfig.network && !processConfig.adapter) {
    processConfig.network = {
      ...defaultNetworkConfig,
      network_kwargs: {
        ...defaultNetworkConfig.network_kwargs,
        ignore_if_contains: [...defaultNetworkConfig.network_kwargs.ignore_if_contains],
      },
    };
  }

  if (processConfig.network && !('pretrained_lora_path' in processConfig.network)) {
    processConfig.network.pretrained_lora_path = defaultNetworkConfig.pretrained_lora_path;
  }

  if (processConfig.network && !('resume_training_state' in processConfig.network)) {
    processConfig.network.resume_training_state = defaultNetworkConfig.resume_training_state;
  }

  if ('adapter' in processConfig && processConfig.adapter) {
    const adapter = processConfig.adapter;
    if (!('type' in adapter)) {
      adapter.type = defaultAdapterConfig.type;
    }
    if (!('train' in adapter)) {
      adapter.train = defaultAdapterConfig.train;
    }
    if (!('image_encoder_path' in adapter)) {
      adapter.image_encoder_path = defaultAdapterConfig.image_encoder_path;
    }
    if (!('image_encoder_arch' in adapter)) {
      adapter.image_encoder_arch = defaultAdapterConfig.image_encoder_arch;
    }
    if (!('train_image_encoder' in adapter)) {
      adapter.train_image_encoder = defaultAdapterConfig.train_image_encoder;
    }
    if (!('train_only_image_encoder' in adapter)) {
      adapter.train_only_image_encoder = defaultAdapterConfig.train_only_image_encoder;
    }
    if (!('clip_layer' in adapter)) {
      adapter.clip_layer = defaultAdapterConfig.clip_layer;
    }
    if (!('test_img_path' in adapter) || adapter.test_img_path == null) {
      adapter.test_img_path = [...(defaultAdapterConfig.test_img_path || [])];
    } else if (typeof adapter.test_img_path === 'string') {
      adapter.test_img_path = adapter.test_img_path
        .split(',')
        .map((part: string) => part.trim())
        .filter((part: string) => part.length > 0);
    }
  }

  if (!('cleanup_caches_on_startup' in processConfig)) {
    processConfig.cleanup_caches_on_startup = defaultProcessConfig.cleanup_caches_on_startup;
  }

  if (!('ui_step_update_interval' in processConfig)) {
    processConfig.ui_step_update_interval = defaultProcessConfig.ui_step_update_interval;
  }

  if (!('ui_stop_poll_interval' in processConfig)) {
    processConfig.ui_stop_poll_interval = defaultProcessConfig.ui_stop_poll_interval;
  }

  for (const dataset of Array.isArray(processConfig.datasets) ? processConfig.datasets : []) {
    if (!('cache_latents' in dataset)) {
      dataset.cache_latents = defaultDatasetConfig.cache_latents;
    }
    if (!('must_include_caption' in dataset)) {
      dataset.must_include_caption = defaultDatasetConfig.must_include_caption;
    }
    if (!('num_workers' in dataset)) {
      dataset.num_workers = defaultDatasetConfig.num_workers;
    }
    if (!('lazy_cache_latents' in dataset)) {
      dataset.lazy_cache_latents = defaultDatasetConfig.lazy_cache_latents;
    }
    if (!('caches_only' in dataset)) {
      dataset.caches_only = defaultDatasetConfig.caches_only;
    }
    if (!('lazy_cache_text_embeddings' in dataset)) {
      dataset.lazy_cache_text_embeddings = defaultDatasetConfig.lazy_cache_text_embeddings;
    }
    if (!('text_embedding_cache_shard_size' in dataset)) {
      dataset.text_embedding_cache_shard_size = defaultDatasetConfig.text_embedding_cache_shard_size;
    }
    if (!('cache_dir_override' in dataset)) {
      dataset.cache_dir_override = defaultDatasetConfig.cache_dir_override;
    }
    if (!('cache_dir_override_min_free_gb' in dataset)) {
      dataset.cache_dir_override_min_free_gb = defaultDatasetConfig.cache_dir_override_min_free_gb;
    }
    if (!('clip_image_path' in dataset)) {
      dataset.clip_image_path = defaultDatasetConfig.clip_image_path;
    }
    if (!('clip_image_from_same_folder' in dataset)) {
      dataset.clip_image_from_same_folder = defaultDatasetConfig.clip_image_from_same_folder;
    }
    if (!('clip_image_from_same_folder_balanced' in dataset)) {
      dataset.clip_image_from_same_folder_balanced = defaultDatasetConfig.clip_image_from_same_folder_balanced;
    }
    if (!('prefetch_factor' in dataset)) {
      dataset.prefetch_factor = defaultDatasetConfig.prefetch_factor;
    }
    if (!('persistent_workers' in dataset)) {
      dataset.persistent_workers = defaultDatasetConfig.persistent_workers;
    }
    if (!('pin_memory' in dataset)) {
      dataset.pin_memory = defaultDatasetConfig.pin_memory;
    }
    if (!('enable_windows_num_workers' in dataset)) {
      dataset.enable_windows_num_workers = defaultDatasetConfig.enable_windows_num_workers;
    }
    if (dataset.lazy_cache_latents) {
      dataset.cache_latents_to_disk = true;
      dataset.cache_latents = false;
    }
  }

  if (processConfig.train && !('skip_first_sample' in processConfig.train)) {
    processConfig.train.skip_first_sample = defaultProcessConfig.train.skip_first_sample;
  }

  if (processConfig.train && !('adapter_lr' in processConfig.train)) {
    processConfig.train.adapter_lr = defaultProcessConfig.train.adapter_lr;
  }

  if (processConfig.train && !('train_by_epoch' in processConfig.train)) {
    processConfig.train.train_by_epoch = defaultProcessConfig.train.train_by_epoch;
  }

  if (processConfig.train && !('global_bucket_batching' in processConfig.train)) {
    processConfig.train.global_bucket_batching = defaultProcessConfig.train.global_bucket_batching;
  }

  if (processConfig.train && !('epochs' in processConfig.train)) {
    processConfig.train.epochs = defaultProcessConfig.train.epochs;
  }

  return jobConfig;
};
