import torch
import torch.nn.functional as F


def _normalize_wan_latents(latents, vae, device, dtype):
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(device, dtype)
    return (latents - latents_mean) * latents_std


def _build_first_frame_mask(
    batch_size,
    num_latent_frames,
    latent_height,
    latent_width,
    vae_scale_factor_temporal,
    device,
    dtype,
):
    num_frames = (num_latent_frames - 1) * 4 + 1

    mask_lat_size = torch.ones(
        batch_size,
        1,
        num_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )
    mask_lat_size[:, :, list(range(1, num_frames))] = 0

    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(
        first_frame_mask,
        dim=2,
        repeats=vae_scale_factor_temporal,
    )

    mask_lat_size = torch.concat(
        [first_frame_mask, mask_lat_size[:, :, 1:, :]],
        dim=2,
    )
    mask_lat_size = mask_lat_size.view(
        batch_size,
        -1,
        vae_scale_factor_temporal,
        latent_height,
        latent_width,
    )
    mask_lat_size = mask_lat_size.transpose(1, 2)
    return mask_lat_size.to(device, dtype)


def build_first_frame_condition(
    latent_condition,
    latent_model_input,
    vae,
):
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)

    batch_size = latent_condition.shape[0]
    num_latent_frames = latent_model_input.shape[2]
    latent_height = latent_condition.shape[3]
    latent_width = latent_condition.shape[4]
    mask_lat_size = _build_first_frame_mask(
        batch_size,
        num_latent_frames,
        latent_height,
        latent_width,
        vae_scale_factor_temporal,
        device,
        dtype,
    )
    return torch.concat([mask_lat_size, latent_condition.to(device, dtype)], dim=1)


def apply_first_frame_condition(
    latent_model_input,
    first_frame_condition,
):
    return torch.cat(
        [latent_model_input, first_frame_condition.to(latent_model_input.device, latent_model_input.dtype)],
        dim=1,
    )


def encode_first_frame_condition(
    latent_model_input,
    first_frame,
    vae,
):
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    num_latent_frames = latent_model_input.shape[2]
    num_frames = (num_latent_frames - 1) * 4 + 1

    if len(first_frame.shape) == 3:
        first_frame = first_frame.unsqueeze(0)

    if first_frame.shape[0] != latent_model_input.shape[0]:
        first_frame = first_frame.expand(latent_model_input.shape[0], -1, -1, -1)

    vae_scale_factor = vae.config.scale_factor_spatial
    first_frame = F.interpolate(
        first_frame,
        size=(
            latent_model_input.shape[3] * vae_scale_factor,
            latent_model_input.shape[4] * vae_scale_factor,
        ),
        mode="bilinear",
        align_corners=False,
    )
    first_frame = first_frame.unsqueeze(2)

    zero_frame = torch.zeros_like(first_frame)
    video_condition = torch.cat(
        [
            first_frame,
            *[zero_frame for _ in range(num_frames - 1)],
        ],
        dim=2,
    )

    latent_condition = vae.encode(
        video_condition.to(device, dtype)
    ).latent_dist.sample()
    latent_condition = _normalize_wan_latents(latent_condition.to(device, dtype), vae, device, dtype)
    return build_first_frame_condition(latent_condition, latent_model_input, vae)


def add_first_frame_conditioning(
    latent_model_input,
    first_frame,
    vae
):
    """
    Adds first frame conditioning to a video diffusion model input.

    Args:
        latent_model_input: Original latent input (bs, channels, num_frames, height, width)
        first_frame: Tensor of first frame to condition on (bs, channels, height, width)
        vae: VAE model for encoding the conditioning

    Returns:
        conditioned_latent: The complete conditioned latent input (bs, 36, num_frames, height, width)
    """
    first_frame_condition = encode_first_frame_condition(
        latent_model_input=latent_model_input,
        first_frame=first_frame,
        vae=vae,
    )
    return apply_first_frame_condition(latent_model_input, first_frame_condition)


def add_first_frame_conditioning_v22_from_encoded_latent(
    latent_model_input,
    encoded_first_frame,
    last_encoded_frame=None,
):
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    bs, _, T, H, W = latent_model_input.shape

    if encoded_first_frame.ndim == 4:
        encoded_first_frame = encoded_first_frame.unsqueeze(0)
    if encoded_first_frame.shape[0] != bs:
        encoded_first_frame = encoded_first_frame.expand(bs, -1, -1, -1, -1)

    latent = latent_model_input.clone()
    encoded_first_frame = encoded_first_frame.to(device, dtype)
    latent[:, :, :encoded_first_frame.shape[2]] = encoded_first_frame

    mask = torch.ones(bs, 1, T, H, W, device=device, dtype=dtype)
    mask[:, :, :encoded_first_frame.shape[2]] = 0.0

    if last_encoded_frame is not None:
        if last_encoded_frame.ndim == 4:
            last_encoded_frame = last_encoded_frame.unsqueeze(0)
        if last_encoded_frame.shape[0] != bs:
            last_encoded_frame = last_encoded_frame.expand(bs, -1, -1, -1, -1)
        last_encoded_frame = last_encoded_frame.to(device, dtype)
        latent[:, :, -last_encoded_frame.shape[2]:] = last_encoded_frame
        mask[:, :, -last_encoded_frame.shape[2]:] = 0.0
        mask = mask.clamp(0.0, 1.0)

    return latent, mask


def add_first_frame_conditioning_v22(
    latent_model_input,
    first_frame,
    vae,
    last_frame=None
):
    """
    Overwrites first few time steps in latent_model_input with VAE-encoded first_frame,
    and returns the modified latent + binary mask (0=conditioned, 1=noise).

    Args:
        latent_model_input: torch.Tensor of shape (bs, 48, T, H, W)
        first_frame: torch.Tensor of shape (bs, 3, H*scale, W*scale)
        vae: VAE model with .encode() and .config.latents_mean/std

    Returns:
        latent: (bs, 48, T, H, W) - modified input latent
        mask: (bs, 1, T, H, W) - binary mask
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    bs, _, T, H, W = latent_model_input.shape
    scale = vae.config.scale_factor_spatial
    target_h = H * scale
    target_w = W * scale

    # Ensure shape
    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if first_frame.shape[0] != bs:
        first_frame = first_frame.expand(bs, -1, -1, -1)

    # Resize and encode
    first_frame_up = F.interpolate(first_frame, size=(target_h, target_w), mode="bilinear", align_corners=False)
    first_frame_up = first_frame_up.unsqueeze(2)  # (bs, 3, 1, H, W)
    encoded = vae.encode(first_frame_up).latent_dist.sample().to(dtype).to(device)
    encoded = _normalize_wan_latents(encoded, vae, device, dtype)

    if last_frame is not None:
        # If last_frame is provided, encode it similarly
        last_frame_up = F.interpolate(last_frame, size=(target_h, target_w), mode="bilinear", align_corners=False)
        last_frame_up = last_frame_up.unsqueeze(2)
        last_encoded = vae.encode(last_frame_up).latent_dist.sample().to(dtype).to(device)
        last_encoded = _normalize_wan_latents(last_encoded, vae, device, dtype)
    else:
        last_encoded = None

    return add_first_frame_conditioning_v22_from_encoded_latent(
        latent_model_input=latent_model_input,
        encoded_first_frame=encoded,
        last_encoded_frame=last_encoded,
    )
