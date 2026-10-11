import os
from typing import TYPE_CHECKING, List, Union
import cv2
import torch

from PIL import Image
from PIL.ImageOps import exif_transpose

from toolkit import image_utils
from toolkit.basic import get_quick_signature_string


# EXIF orientation values 5-8 mean the stored image is rotated 90/270 degrees,
# so the DISPLAYED width/height are swapped relative to the stored ones.
_EXIF_ORIENTATION_TAG = 0x0112
_EXIF_SWAPPED_ORIENTATIONS = (5, 6, 7, 8)


# Reading the EXIF orientation is normally cheap (it lives in the header), but for
# a few formats PIL must scan PAST the pixel data to find it, forcing a full decode.
# Those are listed here and SKIPPED; everything else -- including any format added
# in future -- gets its EXIF read, so the default is correctness, not speed.
#
# Measured on a real 116k-image PNG dataset:
#     image_utils.get_image_size (header parse)   0.037 ms/img
#     Image.open().size                           3.9   ms/img
#     Image.open().size + getexif()              58.9   ms/img   <- the old default
# i.e. ~2.3 hours instead of seconds, to look for a tag that 0 of 60 sampled PNGs
# actually set. PNG orientation is legal (eXIf chunk) but vanishingly rare.
#
# If a future format turns out to be slow here too, add its extension below; the
# only cost of NOT listing one is time, never correctness.
_EXIF_DECODE_REQUIRED_FORMATS = (".png",)


def get_image_size_exif_aware(path):
    """(width, height) as displayed, without decoding the pixels.

    The old default path did `exif_transpose(Image.open(path))` purely to read
    .size -- exif_transpose applies the rotation, which fully decodes the image.

    Here the dimensions come from a pure header parse, and the EXIF orientation
    is applied on top for every format except those where reading it would cost
    a decode. Orientations 5-8 mean the stored image is rotated a quarter turn,
    so the displayed dimensions are swapped -- reproducing exif_transpose's
    .size exactly, which is the only thing this path ever used it for.

    Strictly more correct than the pre-existing `fast_image_size` shortcut,
    which skipped EXIF for EVERY format and so returned sideways dimensions for
    rotated JPEGs.
    """
    try:
        w, h = image_utils.get_image_size(path)
    except Exception:
        with Image.open(path) as img:
            w, h = img.size

    if os.path.splitext(path)[1].lower() not in _EXIF_DECODE_REQUIRED_FORMATS:
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                orientation = exif.get(_EXIF_ORIENTATION_TAG) if exif else None
            if orientation in _EXIF_SWAPPED_ORIENTATIONS:
                return h, w
        except Exception:
            pass
    return w, h
from toolkit.dataloader_mixins import (
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    LatentCachingFileItemDTOMixin,
    hydrate_file_items_from_cached_latents,
    hydrate_file_items_from_cached_text_embeddings,
    ControlFileItemDTOMixin,
    ArgBreakMixin,
    PoiFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
)
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig

printed_messages = []


def print_once(msg):
    global printed_messages
    if msg not in printed_messages:
        print(msg)
        printed_messages.append(msg)


class FileItemDTO(
    LatentCachingFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    ControlFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    PoiFileItemDTOMixin,
    ArgBreakMixin,
):
    def __init__(self, *args, **kwargs):
        self.path = kwargs.get("path", "")
        self.dataset_config: "DatasetConfig" = kwargs.get("dataset_config", None)
        self.is_video = self.dataset_config.num_frames > 1
        size_database = kwargs.get("size_database", {})
        dataset_root = kwargs.get("dataset_root", None)
        self.encode_control_in_text_embeddings = kwargs.get(
            "encode_control_in_text_embeddings", False
        )
        self.te_padding_side = kwargs.get("te_padding_side", "right")
        self.latent_space_version = kwargs.get("latent_space_version", "sd1")
        self.text_embedding_space_version = kwargs.get("text_embedding_space_version", "sd1")
        if dataset_root is not None:
            # remove dataset root from path
            file_key = self.path.replace(dataset_root, "")
        else:
            file_key = os.path.basename(self.path)

        signature_map = kwargs.get("signature_map", None)
        if signature_map is not None and self.path in signature_map:
            file_signature = signature_map[self.path]
        elif self.dataset_config.caches_only and not os.path.exists(self.path):
            # missing source file; resolved from the size database + disk caches
            file_signature = None
        else:
            file_signature = get_quick_signature_string(self.path)

        # ingredients needed to resolve disk caches when the source file and/or
        # caption sidecar no longer exist (caches_only mode). Plain values only;
        # FileItemDTOs get deepcopied per fetch, so never hold the db reference.
        self.cached_src_stat = None
        self.cached_caption_text = None

        db_entry = size_database.get(file_key)
        if not isinstance(db_entry, (list, tuple)) or len(db_entry) < 3:
            db_entry = None

        if file_signature is None:
            # source file is gone
            if not self.dataset_config.caches_only:
                raise Exception(f"Error: Could not get file signature for {self.path}")
            if db_entry is None:
                raise Exception(
                    f"Error: {self.path} is missing and has no size-database entry; "
                    f"cannot train it from caches"
                )
            # trust the size database from the last run the file existed
            w, h = db_entry[0], db_entry[1]
            if len(db_entry) >= 5 and db_entry[3] is not None and db_entry[4] is not None:
                self.cached_src_stat = (int(db_entry[3]), int(db_entry[4]))
            if len(db_entry) >= 6 and db_entry[5] is not None:
                self.cached_caption_text = db_entry[5]
            self.width: int = w
            self.height: int = h
            self.dataloader_transforms = kwargs.get("dataloader_transforms", None)
            super().__init__(*args, **kwargs)
            self._post_size_init(**kwargs)
            return

        use_db_entry = db_entry is not None and db_entry[2] == file_signature

        if use_db_entry:
            w, h = db_entry[0], db_entry[1]
        elif self.is_video:
            # Open the video file
            video = cv2.VideoCapture(self.path)

            # Check if video opened successfully
            if not video.isOpened():
                raise Exception(f"Error: Could not open video file {self.path}")

            # Get width and height
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w, h = width, height

            # Release the video capture object immediately
            video.release()
        else:
            if self.dataset_config.fast_image_size:
                # original method is significantly faster, but some images are read sideways. Not sure why. Do slow method by default.
                try:
                    w, h = image_utils.get_image_size(self.path)
                except image_utils.UnknownImageFormat:
                    print_once(
                        f"Warning: Some images in the dataset cannot be fast read. "
                        + f"This process is faster for png, jpeg"
                    )
                    img = exif_transpose(Image.open(self.path))
                    w, h = img.size
            else:
                # header-only + EXIF orientation. Identical dimensions to the old
                # exif_transpose(Image.open(...)) path, without the full decode.
                try:
                    w, h = get_image_size_exif_aware(self.path)
                except Exception:
                    # anything PIL cannot parse lazily falls back to the old path
                    img = exif_transpose(Image.open(self.path))
                    w, h = img.size

        # persist the ingredients of the latent/text-embedding cache keys next to
        # the dimensions so a later caches_only run can resolve the caches after
        # the source images / caption files are deleted.
        src_size = None
        src_mtime_ns = None
        stat_map = kwargs.get("stat_map", None)
        if stat_map is not None and self.path in stat_map:
            src_size, src_mtime_ns = stat_map[self.path]
        else:
            try:
                _stat = os.stat(self.path)
                src_size, src_mtime_ns = _stat.st_size, _stat.st_mtime_ns
            except OSError:
                pass
        if src_size is not None:
            self.cached_src_stat = (src_size, src_mtime_ns)

        caption_text = None
        caption_signature = None
        self._caption_from_cache = False
        if self.dataset_config.caches_only or self.dataset_config.cache_text_embeddings:
            caption_path = os.path.splitext(self.path)[0] + self.dataset_config.caption_ext
            # The startup directory scan already holds every sidecar's size and mtime
            # (ns). If the caption is byte-for-byte the file we stored last time, reuse
            # the stored text instead of opening it: reading 65k small caption files one
            # by one was the whole of a 5+ minute "Preprocessing image dimensions" step
            # on a slow disk, every single job start.
            if stat_map is not None and caption_path in stat_map:
                _c_size, _c_mtime_ns = stat_map[caption_path]
                caption_signature = f"{_c_size}:{_c_mtime_ns}"
            if (caption_signature is not None and db_entry is not None and len(db_entry) >= 7
                    and db_entry[6] == caption_signature and db_entry[5] is not None):
                caption_text = db_entry[5]
                self._caption_from_cache = True
            else:
                try:
                    if os.path.exists(caption_path):
                        with open(caption_path, "r", encoding="utf-8") as f:
                            caption_text = f.read()
                except OSError:
                    pass
                if caption_text is None:
                    caption_signature = None
                    if db_entry is not None and len(db_entry) >= 6:
                        # keep the previously persisted caption if the sidecar is gone
                        caption_text = db_entry[5]
        elif db_entry is not None and len(db_entry) >= 6:
            caption_text = db_entry[5]
            if len(db_entry) >= 7:
                caption_signature = db_entry[6]
        self.cached_caption_text = caption_text

        size_database[file_key] = (w, h, file_signature, src_size, src_mtime_ns, caption_text, caption_signature)
        self.width: int = w
        self.height: int = h
        self.dataloader_transforms = kwargs.get("dataloader_transforms", None)
        super().__init__(*args, **kwargs)
        self._post_size_init(**kwargs)

    def _post_size_init(self, **kwargs):
        # self.caption_path: str = kwargs.get('caption_path', None)
        self.raw_caption: str = kwargs.get("raw_caption", None)
        # we scale first, then crop
        self.scale_to_width: int = kwargs.get(
            "scale_to_width", int(self.width * self.dataset_config.scale)
        )
        self.scale_to_height: int = kwargs.get(
            "scale_to_height", int(self.height * self.dataset_config.scale)
        )
        # crop values are from scaled size
        self.crop_x: int = kwargs.get("crop_x", 0)
        self.crop_y: int = kwargs.get("crop_y", 0)
        self.crop_width: int = kwargs.get("crop_width", self.scale_to_width)
        self.crop_height: int = kwargs.get("crop_height", self.scale_to_height)
        self.flip_x: bool = kwargs.get("flip_x", False)
        self.flip_y: bool = kwargs.get("flip_x", False)
        self.augments: List[str] = self.dataset_config.augments
        self.loss_multiplier: float = self.dataset_config.loss_multiplier

        self.network_weight: float = self.dataset_config.network_weight
        self.is_reg = self.dataset_config.is_reg
        self.prior_reg = self.dataset_config.prior_reg
        self.tensor: Union[torch.Tensor, None] = None
        self.audio_data = None
        self.audio_tensor = None

    def cleanup(self):
        self.tensor = None
        self.audio_data = None
        self.audio_tensor = None
        self.cleanup_latent()
        self.cleanup_text_embedding()
        self.cleanup_control()
        self.cleanup_inpaint()
        self.cleanup_clip_image()
        self.cleanup_mask()
        self.cleanup_unconditional()


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        try:
            self.file_items: List["FileItemDTO"] = kwargs.get("file_items", None)
            self.tensor: Union[torch.Tensor, None] = None
            self.latents: Union[torch.Tensor, None] = None
            self.control_tensor: Union[torch.Tensor, None] = None
            self.control_tensor_list: Union[List[List[torch.Tensor]], None] = None
            self.clip_image_tensor: Union[torch.Tensor, None] = None
            self.mask_tensor: Union[torch.Tensor, None] = None
            self.unaugmented_tensor: Union[torch.Tensor, None] = None
            self.unconditional_tensor: Union[torch.Tensor, None] = None
            self.unconditional_latents: Union[torch.Tensor, None] = None
            self.clip_image_embeds: Union[List[dict], None] = None
            self.clip_image_embeds_unconditional: Union[List[dict], None] = None
            self.sigmas: Union[torch.Tensor, None] = (
                None  # can be added elseware and passed along training code
            )
            self.extra_values: Union[torch.Tensor, None] = (
                torch.tensor([x.extra_values for x in self.file_items])
                if len(self.file_items[0].extra_values) > 0
                else None
            )
            self.audio_data: Union[List, None] = (
                [x.audio_data for x in self.file_items]
                if self.file_items[0].audio_data is not None
                else None
            )
            self.audio_tensor: Union[torch.Tensor, None] = None
            self.first_frame_latents: Union[torch.Tensor, None] = None
            self.first_frame_conditions: Union[torch.Tensor, None] = None
            self.audio_latents: Union[torch.Tensor, None] = None

            # just for holding noise and preds during training
            self.audio_target: Union[torch.Tensor, None] = None
            self.audio_pred: Union[torch.Tensor, None] = None
            # Per-step audio bookkeeping for joint audio+video models (minimax_h3 reads and
            # writes these in get_noise_prediction). Ported from upstream ostris/ai-toolkit
            # as of the commit the minimax_h3 extension was copied from.
            # the noise drawn for the audio stream on the primary (grad enabled)
            # prediction. Secondary passes (cfg / guidance loss / prior preds)
            # reuse it so their noisy audio matches the stored audio_target.
            self.audio_noise: Union[torch.Tensor, None] = None
            # audio predictions from the non primary passes. Kept separate so
            # they cannot stomp the primary pred we backprop through.
            self.audio_pred_uncond: Union[torch.Tensor, None] = None
            self.audio_pred_prior: Union[torch.Tensor, None] = None
            self.audio_pred_preservation: Union[torch.Tensor, None] = None
            # which of the above the current secondary pass writes to. None (the
            # default) means no secondary pass is in flight: any grad-enabled
            # prediction is a primary one and writes audio_pred (and the
            # noisy/sigma bookkeeping) directly. Nothing in this fork's trainer sets
            # it yet, so predictions made under no_grad (prior / unconditional
            # passes) are simply not stored -- see set_secondary_audio_pred.
            self.audio_pred_slot: Union[str, None] = None
            # noisy audio rows and audio sigma of the primary pass, used to
            # rebuild the clean audio estimate for perceptual losses
            self.audio_noisy: Union[torch.Tensor, None] = None
            self.audio_sigma: Union[torch.Tensor, None] = None
            # frames per item: the dataset's num_frames (1 for image datasets). Upstream
            # keeps this per file; in this fork a dataset is all video or all image.
            self.num_frames: int = self.file_items[0].dataset_config.num_frames
            self.latent_cache_mask: Union[torch.Tensor, None] = None
            self.cached_latent_indices: List[int] = []
            self.missing_latent_indices: List[int] = []
            self.text_embedding_cache_mask: Union[torch.Tensor, None] = None
            self.cached_text_embedding_indices: List[int] = []
            self.missing_text_embedding_indices: List[int] = []

            cached_mask = [x.is_latent_cached for x in self.file_items]
            has_cached_latents = any(cached_mask)
            has_uncached_latents = any([not value for value in cached_mask])
            is_latents_cached = has_cached_latents
            if has_cached_latents and has_uncached_latents:
                self.latent_cache_mask = torch.tensor(cached_mask, dtype=torch.bool)
                self.cached_latent_indices = [
                    idx for idx, is_cached in enumerate(cached_mask) if is_cached
                ]
                self.missing_latent_indices = [
                    idx for idx, is_cached in enumerate(cached_mask) if not is_cached
                ]

            # if we have encoded latents, we concatenate them
            self.latents: Union[torch.Tensor, None] = None
            cached_file_items = []
            if is_latents_cached:
                cached_file_items = (
                    [self.file_items[idx] for idx in self.cached_latent_indices]
                    if self.latent_cache_mask is not None
                    else self.file_items
                )
                hydrate_file_items_from_cached_latents(cached_file_items)
                missing_latent = next(
                    (x.path for x in cached_file_items if x._encoded_latent is None),
                    None,
                )
                if missing_latent is not None:
                    raise Exception(f"Could not load cached latent for {missing_latent}")

                self.latents = torch.cat(
                    [x._encoded_latent.unsqueeze(0) for x in cached_file_items]
                )
                if (
                    len(cached_file_items) > 0
                    and all(
                        x._cached_first_frame_latent is not None
                        for x in cached_file_items
                    )
                ):
                    self.first_frame_latents = torch.cat(
                        [
                            x._cached_first_frame_latent.unsqueeze(0)
                            for x in cached_file_items
                        ]
                    )
                if (
                    len(cached_file_items) > 0
                    and all(
                        x._cached_first_frame_condition is not None
                        for x in cached_file_items
                    )
                ):
                    self.first_frame_conditions = torch.cat(
                        [
                            x._cached_first_frame_condition.unsqueeze(0)
                            for x in cached_file_items
                        ]
                    )
                if any([x._cached_audio_latent is not None for x in cached_file_items]):
                    self.audio_latents = torch.cat(
                        [
                            x._cached_audio_latent.unsqueeze(0)
                            if x._cached_audio_latent is not None
                            else torch.zeros_like(
                                cached_file_items[0]._cached_audio_latent
                            ).unsqueeze(0)
                            for x in cached_file_items
                        ]
                    )

            latent_space_version = getattr(self.file_items[0], "latent_space_version", "")
            needs_i2v_tensor_fallback = (
                self.file_items[0].dataset_config.num_frames > 1
                and self.file_items[0].dataset_config.do_i2v
                and (
                    (
                        latent_space_version == "wan22_14b_i2v"
                        and self.first_frame_conditions is None
                    )
                    or (
                        latent_space_version in {"wan22_5b", "ltx2", "ltx2.3"}
                        and self.first_frame_latents is None
                    )
                )
            )

            if (
                not is_latents_cached
                or self.latent_cache_mask is not None
                or needs_i2v_tensor_fallback
            ):
                if self.latent_cache_mask is not None:
                    for file_item in self.file_items:
                        if file_item.tensor is None:
                            file_item.load_and_process_image(
                                file_item.dataloader_transforms,
                                prefer_cached_latent=False,
                            )
                elif needs_i2v_tensor_fallback:
                    for file_item in self.file_items:
                        if file_item.tensor is None:
                            file_item.load_and_process_image(
                                file_item.dataloader_transforms,
                                prefer_cached_latent=False,
                            )
                self.tensor = torch.cat(
                    [x.tensor.unsqueeze(0) for x in self.file_items]
                )

            self.prompt_embeds: Union[PromptEmbeds, None] = None
            cached_text_embedding_mask = [x.is_text_embedding_cached for x in self.file_items]
            has_cached_text_embeddings = any(cached_text_embedding_mask)
            has_uncached_text_embeddings = any([not value for value in cached_text_embedding_mask])
            if has_cached_text_embeddings and has_uncached_text_embeddings:
                self.text_embedding_cache_mask = torch.tensor(cached_text_embedding_mask, dtype=torch.bool)
                self.cached_text_embedding_indices = [
                    idx for idx, is_cached in enumerate(cached_text_embedding_mask) if is_cached
                ]
                self.missing_text_embedding_indices = [
                    idx for idx, is_cached in enumerate(cached_text_embedding_mask) if not is_cached
                ]
            elif has_cached_text_embeddings:
                self.cached_text_embedding_indices = list(range(len(self.file_items)))

            if has_cached_text_embeddings:
                cached_file_items = (
                    [self.file_items[idx] for idx in self.cached_text_embedding_indices]
                    if self.text_embedding_cache_mask is not None
                    else self.file_items
                )
                hydrate_file_items_from_cached_text_embeddings(cached_file_items)
                missing_prompt_embed = next(
                    (x.path for x in cached_file_items if x.prompt_embeds is None),
                    None,
                )
                if missing_prompt_embed is not None:
                    raise Exception(f"Could not load cached text embedding for {missing_prompt_embed}")

                padding_side = self.file_items[0].te_padding_side
                self.prompt_embeds = concat_prompt_embeds(
                    [x.prompt_embeds for x in cached_file_items],
                    padding_side=padding_side,
                )

            # if self.file_items[0].control_tensor is not None:
            # if any have a control tensor, we concatenate them
            if any([x.control_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_control_tensor = None
                for x in self.file_items:
                    if x.control_tensor is not None:
                        base_control_tensor = x.control_tensor
                        break
                control_tensors = []
                for x in self.file_items:
                    if x.control_tensor is None:
                        control_tensors.append(torch.zeros_like(base_control_tensor))
                    else:
                        control_tensors.append(x.control_tensor)
                self.control_tensor = torch.cat(
                    [x.unsqueeze(0) for x in control_tensors]
                )

            # handle control tensor list
            if any([x.control_tensor_list is not None for x in self.file_items]):
                self.control_tensor_list = []
                for x in self.file_items:
                    if x.control_tensor_list is not None:
                        self.control_tensor_list.append(x.control_tensor_list)
                    else:
                        raise Exception(
                            f"Could not find control tensors for all file items, missing for {x.path}"
                        )

            self.inpaint_tensor: Union[torch.Tensor, None] = None
            if any([x.inpaint_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_inpaint_tensor = None
                for x in self.file_items:
                    if x.inpaint_tensor is not None:
                        base_inpaint_tensor = x.inpaint_tensor
                        break
                inpaint_tensors = []
                for x in self.file_items:
                    if x.inpaint_tensor is None:
                        inpaint_tensors.append(torch.zeros_like(base_inpaint_tensor))
                    else:
                        inpaint_tensors.append(x.inpaint_tensor)
                self.inpaint_tensor = torch.cat(
                    [x.unsqueeze(0) for x in inpaint_tensors]
                )

            self.loss_multiplier_list: List[float] = [
                x.loss_multiplier for x in self.file_items
            ]

            if any([x.clip_image_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_clip_image_tensor = None
                for x in self.file_items:
                    if x.clip_image_tensor is not None:
                        base_clip_image_tensor = x.clip_image_tensor
                        break
                clip_image_tensors = []
                for x in self.file_items:
                    if x.clip_image_tensor is None:
                        clip_image_tensors.append(
                            torch.zeros_like(base_clip_image_tensor)
                        )
                    else:
                        clip_image_tensors.append(x.clip_image_tensor)
                self.clip_image_tensor = torch.cat(
                    [x.unsqueeze(0) for x in clip_image_tensors]
                )

            if any([x.mask_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_mask_tensor = None
                for x in self.file_items:
                    if x.mask_tensor is not None:
                        base_mask_tensor = x.mask_tensor
                        break
                mask_tensors = []
                for x in self.file_items:
                    if x.mask_tensor is None:
                        mask_tensors.append(torch.zeros_like(base_mask_tensor))
                    else:
                        mask_tensors.append(x.mask_tensor)
                self.mask_tensor = torch.cat([x.unsqueeze(0) for x in mask_tensors])

            # add unaugmented tensors for ones with augments
            if any([x.unaugmented_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unaugmented_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unaugmented_tensor = x.unaugmented_tensor
                        break
                unaugmented_tensor = []
                for x in self.file_items:
                    if x.unaugmented_tensor is None:
                        unaugmented_tensor.append(
                            torch.zeros_like(base_unaugmented_tensor)
                        )
                    else:
                        unaugmented_tensor.append(x.unaugmented_tensor)
                self.unaugmented_tensor = torch.cat(
                    [x.unsqueeze(0) for x in unaugmented_tensor]
                )

            # add unconditional tensors
            if any([x.unconditional_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unconditional_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unconditional_tensor = x.unconditional_tensor
                        break
                unconditional_tensor = []
                for x in self.file_items:
                    if x.unconditional_tensor is None:
                        unconditional_tensor.append(
                            torch.zeros_like(base_unconditional_tensor)
                        )
                    else:
                        unconditional_tensor.append(x.unconditional_tensor)
                self.unconditional_tensor = torch.cat(
                    [x.unsqueeze(0) for x in unconditional_tensor]
                )

            if any([x.clip_image_embeds is not None for x in self.file_items]):
                self.clip_image_embeds = []
                for x in self.file_items:
                    if x.clip_image_embeds is not None:
                        self.clip_image_embeds.append(x.clip_image_embeds)
                    else:
                        raise Exception("clip_image_embeds is None for some file items")

            if any(
                [x.clip_image_embeds_unconditional is not None for x in self.file_items]
            ):
                self.clip_image_embeds_unconditional = []
                for x in self.file_items:
                    if x.clip_image_embeds_unconditional is not None:
                        self.clip_image_embeds_unconditional.append(
                            x.clip_image_embeds_unconditional
                        )
                    else:
                        raise Exception(
                            "clip_image_embeds_unconditional is None for some file items"
                        )

            if self.prompt_embeds is None and any([x.prompt_embeds is not None for x in self.file_items]):
                # find one to use as a base
                base_prompt_embeds = None
                for x in self.file_items:
                    if x.prompt_embeds is not None:
                        base_prompt_embeds = x.prompt_embeds
                        break
                prompt_embeds_list = []
                for x in self.file_items:
                    if x.prompt_embeds is None:
                        y = base_prompt_embeds
                    else:
                        y = x.prompt_embeds
                    if x.text_embedding_space_version.startswith("zimage"):
                        # z image needs to be a list if it is not already
                        if not isinstance(y.text_embeds, list):
                            y.text_embeds = [y.text_embeds]
                    prompt_embeds_list.append(y)
                padding_side = self.file_items[0].te_padding_side
                
                self.prompt_embeds = concat_prompt_embeds(prompt_embeds_list, padding_side=padding_side)

            if any([x.audio_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_audio_tensor = None
                for x in self.file_items:
                    if x.audio_tensor is not None:
                        base_audio_tensor = x.audio_tensor
                        break
                audio_tensors = []
                for x in self.file_items:
                    if x.audio_tensor is None:
                        audio_tensors.append(torch.zeros_like(base_audio_tensor))
                    else:
                        audio_tensors.append(x.audio_tensor)
                self.audio_tensor = torch.cat([x.unsqueeze(0) for x in audio_tensors])

        except Exception as e:
            print(e)
            raise e

    def get_is_reg_list(self):
        return [x.is_reg for x in self.file_items]

    def get_network_weight_list(self):
        return [x.network_weight for x in self.file_items]

    def get_caption_list(
        self, trigger=None, to_replace_list=None, add_if_not_present=True
    ):
        return [x.caption for x in self.file_items]

    def get_caption_short_list(
        self, trigger=None, to_replace_list=None, add_if_not_present=True
    ):
        return [x.caption_short for x in self.file_items]

    def pin_memory(self):
        def _pin(value):
            if value is None:
                return None
            if hasattr(value, "pin_memory"):
                return value.pin_memory()
            if isinstance(value, list):
                return [_pin(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_pin(item) for item in value)
            if isinstance(value, dict):
                return {key: _pin(item) for key, item in value.items()}
            return value

        self.tensor = _pin(self.tensor)
        self.latents = _pin(self.latents)
        self.control_tensor = _pin(self.control_tensor)
        self.control_tensor_list = _pin(self.control_tensor_list)
        self.clip_image_tensor = _pin(self.clip_image_tensor)
        self.mask_tensor = _pin(self.mask_tensor)
        self.unaugmented_tensor = _pin(self.unaugmented_tensor)
        self.unconditional_tensor = _pin(self.unconditional_tensor)
        self.unconditional_latents = _pin(self.unconditional_latents)
        self.sigmas = _pin(self.sigmas)
        self.extra_values = _pin(self.extra_values)
        self.audio_tensor = _pin(self.audio_tensor)
        self.first_frame_latents = _pin(self.first_frame_latents)
        self.first_frame_conditions = _pin(self.first_frame_conditions)
        self.audio_latents = _pin(self.audio_latents)
        self.audio_target = _pin(self.audio_target)
        self.audio_pred = _pin(self.audio_pred)
        self.latent_cache_mask = _pin(self.latent_cache_mask)
        self.text_embedding_cache_mask = _pin(self.text_embedding_cache_mask)
        self.inpaint_tensor = _pin(self.inpaint_tensor)
        self.clip_image_embeds = _pin(self.clip_image_embeds)
        self.clip_image_embeds_unconditional = _pin(self.clip_image_embeds_unconditional)
        self.prompt_embeds = _pin(self.prompt_embeds)
        return self

    def set_secondary_audio_pred(self, pred):
        """Route an audio prediction from a non primary pass (prior,
        unconditional/guidance, preservation) to its own slot so it cannot
        stomp the primary prediction the loss backprops through. Passes that
        did not declare a slot (e.g. a trainer's extra no_grad prediction)
        are simply not stored."""
        if self.audio_pred_slot is not None:
            setattr(self, self.audio_pred_slot, pred)

    def cleanup(self):
        del self.latents
        del self.tensor
        del self.control_tensor
        del self.audio_tensor
        del self.audio_data
        del self.audio_target
        del self.audio_pred
        del self.audio_noise
        del self.audio_pred_uncond
        del self.audio_pred_prior
        del self.audio_pred_preservation
        del self.audio_noisy
        del self.audio_sigma
        del self.first_frame_latents
        del self.first_frame_conditions
        del self.audio_latents
        for file_item in self.file_items:
            file_item.cleanup()

    @property
    def dataset_config(self) -> "DatasetConfig":
        if len(self.file_items) > 0:
            return self.file_items[0].dataset_config
        else:
            return None
