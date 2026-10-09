import { GroupedSelectOption, SelectOption, JobConfig } from '@/types';
import { defaultSliderConfig } from './jobConfig';

type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections =
  | 'model.quantize'
  | 'train.timestep_type'
  | 'network.conv'
  | 'trigger_word'
  | 'train.diff_output_preservation'
  | 'train.blank_prompt_preservation'
  | 'train.unload_text_encoder'
  | 'slider';

type AdditionalSections =
  | 'datasets.control_path'
  | 'datasets.multi_control_paths'
  | 'datasets.do_i2v'
  | 'datasets.do_audio'
  | 'datasets.audio_normalize'
  | 'datasets.audio_preserve_pitch'
  | 'sample.ctrl_img'
  | 'sample.multi_ctrl_imgs'
  | 'train.audio_loss_multiplier'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path';

type ModelGroup = 'image' | 'instruction' | 'video' | 'experimental';

export interface ModelArch {
  name: string;
  label: string;
  group: ModelGroup;
  controls?: Control[];
  isVideoModel?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
  accuracyRecoveryAdapters?: { [key: string]: string };
}

const defaultNameOrPath = '';
// fallback rank restored when a model that overrides network.linear is deselected
const defaultLinearRank = 32;

export const modelArchs: ModelArch[] = [
  {
    name: 'flux',
    label: 'FLUX.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.1-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'flux_kontext',
    label: 'FLUX.1-Kontext-dev',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.1-Kontext-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
  },
  {
    name: 'flex1',
    label: 'Flex.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Flex.1-alpha', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'flex2',
    label: 'Flex.2',
    group: 'image',
    controls: ['depth', 'line', 'pose', 'inpaint'],
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Flex.2-preview', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.model_kwargs': [
        {
          invert_inpaint_mask_chance: 0.2,
          inpaint_dropout: 0.5,
          control_dropout: 0.5,
          inpaint_random_chance: 0.2,
          do_random_inpainting: true,
          random_blur_mask: true,
          random_dialate_mask: true,
        },
        {},
      ],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'chroma',
    label: 'Chroma',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['lodestones/Chroma1-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'zeta_chroma',
    label: 'Zeta Chroma',
    group: 'experimental',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['Tongyi-MAI/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'wan21:1b',
    label: 'Wan 2.1 (1.3B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-1.3B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram'],
  },
  {
    name: 'wan21_i2v:14b480p',
    label: 'Wan 2.1 I2V (14B-480P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram'],
  },
  {
    name: 'wan21_i2v:14b',
    label: 'Wan 2.1 I2V (14B-720P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-I2V-14B-720P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram'],
  },
  {
    name: 'wan21:14b',
    label: 'Wan 2.1 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-14B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram'],
  },
  {
    name: 'wan22_14b:t2v',
    label: 'Wan 2.2 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'model.multistage', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      // '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_14b_i2v',
    label: 'Wan 2.2 I2V (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.low_vram',
      'model.multistage',
      'model.layer_offloading',
    ],
    accuracyRecoveryAdapters: {
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_5b',
    label: 'Wan 2.2 TI2V (5B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.2-TI2V-5B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [true, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.do_i2v'],
  },
  {
    name: 'lumina2',
    label: 'Lumina2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Alpha-VLLM/Lumina-Image-2.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'qwen_image',
    label: 'Qwen-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image:2512',
    label: 'Qwen-Image-2512',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-2512', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    // Training an ARA now, the other one will not work
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors',
    },
  },
  {
    name: 'qwen_image_edit',
    label: 'Qwen-Image-Edit',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus',
    label: 'Qwen-Image-Edit-2509',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit-2509', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus:2511',
    label: 'Qwen-Image-Edit-2511',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit-2511', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors',
    },
  },
  {
    name: 'hidream',
    label: 'HiDream',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['HiDream-ai/HiDream-I1-Full', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0002, 0.0001],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/hidream_i1_full_torchao_uint3.safetensors',
    },
  },
  {
    name: 'hidream_e1',
    label: 'HiDream E1',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['HiDream-ai/HiDream-E1-1', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0001, 0.0001],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram'],
  },
  {
    name: 'sdxl',
    label: 'SDXL',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['stabilityai/stable-diffusion-xl-base-1.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'sd15',
    label: 'SD 1.5',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['stable-diffusion-v1-5/stable-diffusion-v1-5', defaultNameOrPath],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.width': [512, 1024],
      'config.process[0].sample.height': [512, 1024],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'omnigen2',
    label: 'OmniGen2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['OmniGen2/OmniGen2', defaultNameOrPath],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
  },
  {
    name: 'flux2',
    label: 'FLUX.2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'zimage:turbo',
    label: 'Z-Image Turbo (w/ Training Adapter)',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Tongyi-MAI/Z-Image-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.assistant_lora_path': [
        'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [8, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'zimage',
    label: 'Z-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Tongyi-MAI/Z-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.sample_steps': [30, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'zimage:deturbo',
    label: 'Z-Image De-Turbo (De-Distilled)',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Z-Image-De-Turbo', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['Tongyi-MAI/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.guidance_scale': [3, 4],
      'config.process[0].sample.sample_steps': [25, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'ltx2',
    label: 'LTX-2',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Lightricks/LTX-2', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.layer_offloading', 'model.low_vram', 'datasets.do_audio', 'datasets.audio_normalize', 'datasets.audio_preserve_pitch', 'datasets.do_i2v', 'train.audio_loss_multiplier'],
  },
  {
    name: 'ltx2.3',
    label: 'LTX-2.3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.layer_offloading', 'model.low_vram', 'datasets.do_audio', 'datasets.audio_normalize', 'datasets.audio_preserve_pitch', 'datasets.do_i2v', 'train.audio_loss_multiplier'],
  },
  {
    name: 'flux2_klein_4b',
    label: 'FLUX.2-klein-base-4B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-klein-base-4B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'flux2_klein_9b',
    label: 'FLUX.2-klein-base-9B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-klein-base-9B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  // Krea 2 (SingleStreamDiT + Qwen-Image VAE + Qwen3-VL text encoder).
  // NOTE vs upstream: upstream's entries carry a `gateUrl` field linking to the gated
  // HF repo, and the edit variants list a 'model.model_kwargs.kv_cache' section.
  // Neither exists in this fork's ModelArch/AdditionalSections types, so both are
  // omitted. The repos are still GATED -- accept the license on huggingface.co and be
  // logged in (huggingface-cli login) or the weights will 401 on first load. kv_cache
  // is still enabled via model_kwargs below; only its UI toggle is absent.
  {
    name: 'krea2',
    label: 'Krea 2 (raw)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'krea2:turbo',
    label: 'Krea 2 Turbo (w/ Training Adapter)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [9, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'krea2:o_edit',
    label: 'Krea 2 (raw) [Edit Training]',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'krea2:o_edit_turbo',
    label: 'Krea 2 Turbo (w/ Training Adapter) [Edit Training]',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [8, 25],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.assistant_lora_path',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'qwen_image_2',
    label: 'Qwen-Image-2.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/Qwen-Image-2.1', defaultNameOrPath],
      // DiT: torchao float8 from the bf16 file, the fork's LoRA + layer-offloading path (it
      // cannot train LoRAs on the pre-quantized convrot8 DiT). Text encoder: the pre-quantized
      // convrot8 file (9.35 GB instead of 17.5 GB); it is never trained.
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['convrot8', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].sample.guidance_scale': [3, 4],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'minimax_h3',
    label: 'MiniMax-H3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/MiniMax-H3', defaultNameOrPath],
      // the Comfy-Org weights are pre-quantized (int8 convrot DiT, nvfp4 TE); these
      // qtypes match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['nvfp4', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['adaln_proj'], []],
      'config.process[0].sample.num_frames': [107, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [28, 25],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].num_frames': [39, undefined],
      'config.process[0].model.assistant_lora_path': [
        'ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
        undefined,
      ],
    },
    disableSections: ['network.conv'],
    // NOTE vs upstream: 'datasets.auto_frame_count' is omitted -- this fork has no
    // such UI section and no DatasetConfig field for it, and nothing here calls the
    // model's get_frame_count_snapper() hook. Set datasets num_frames to the VAE's
    // 17n+5 grid (e.g. 39 = 17*2+5) by hand instead.
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'model.assistant_lora_path',
    ],
  },
].sort((a, b) => {
  // Sort by label, case-insensitive
  return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedModelOptions: GroupedSelectOption[] = modelArchs.reduce((acc, arch) => {
  const group = acc.find(g => g.label === arch.group);
  if (group) {
    group.options.push({ value: arch.name, label: arch.label });
  } else {
    acc.push({
      label: arch.group,
      options: [{ value: arch.name, label: arch.label }],
    });
  }
  return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
  { value: '', label: '- NONE -' },
  { value: 'qfloat8', label: 'qfloat8 (default)' },
  { value: 'float8', label: 'float8' },
  // ostris custom backends (toolkit/util/ostris_quant.py + convrot/nvfp4 quantizers).
  // All verified to resolve via get_qtype() in this tree.
  { value: 'convrot8', label: '8bit convrot' },
  { value: 'convrot4', label: '4bit convrot (nvfp4)' },
  { value: 'nvfp4', label: 'nvfp4 (4bit weight only)' },
  { value: 'convrotint7', label: '7bit convrot' },
  { value: 'convrotint6', label: '6bit convrot' },
  { value: 'convrotint5', label: '5bit convrot' },
  { value: 'convrotint4', label: '4bit convrot' },
  { value: 'convrotint3', label: '3bit convrot' },
  { value: 'convrotint2', label: '2bit convrot' },
  { value: 'convrotbitnet', label: '1.58bit convrot (bitnet)' },
  { value: 'uint7', label: '7 bit' },
  { value: 'uint6', label: '6 bit' },
  { value: 'uint5', label: '5 bit' },
  { value: 'uint4', label: '4 bit' },
  { value: 'uint3', label: '3 bit' },
  { value: 'uint2', label: '2 bit' },
];

export const defaultQtype = 'qfloat8';

interface JobTypeOption extends SelectOption {
  disableSections?: DisableableSections[];
  processSections?: string[];
  onActivate?: (config: JobConfig) => JobConfig;
  onDeactivate?: (config: JobConfig) => JobConfig;
}

export const jobTypeOptions: JobTypeOption[] = [
  {
    value: 'diffusion_trainer',
    label: 'LoRA Trainer',
    disableSections: ['slider'],
  },
  {
    value: 'concept_slider',
    label: 'Concept Slider',
    disableSections: ['trigger_word', 'train.diff_output_preservation'],
    onActivate: (config: JobConfig) => {
      // add default slider config
      config.config.process[0].slider = { ...defaultSliderConfig };
      return config;
    },
    onDeactivate: (config: JobConfig) => {
      // remove slider config
      delete config.config.process[0].slider;
      return config;
    },
  },
];
