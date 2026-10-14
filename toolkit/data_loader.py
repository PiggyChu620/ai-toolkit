import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import random
import traceback
from functools import lru_cache
from typing import Callable, List, Optional, TYPE_CHECKING, Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from tqdm import tqdm
import albumentations as A

from toolkit import image_utils
from toolkit.buckets import get_bucket_for_image_size, BucketResolution
from toolkit.cache_paths import get_cache_file_candidates, has_sufficient_free_space
from toolkit.config_modules import DatasetConfig, check_control_from_same_folder, preprocess_dataset_raw_config
from toolkit.dataloader_mixins import CaptionMixin, BucketsMixin, LatentCachingMixin, Augments, CLIPCachingMixin, ControlCachingMixin, TextEmbeddingCachingMixin
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator

import platform

def is_native_windows():
    return platform.system() == "Windows" and platform.release() != "2"

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    

image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']


def _normalize_caption_ext(caption_ext: Optional[str]) -> Optional[str]:
    if caption_ext is None:
        return None
    caption_ext = str(caption_ext).strip()
    if caption_ext == "":
        return None
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    return caption_ext


def _caption_path_for_file(file_path: str, caption_ext: Optional[str]) -> Optional[str]:
    caption_ext = _normalize_caption_ext(caption_ext)
    if caption_ext is None:
        return None
    return os.path.splitext(file_path)[0] + caption_ext


def _has_sidecar_caption(file_path: str, caption_ext: Optional[str]) -> bool:
    caption_path = _caption_path_for_file(file_path, caption_ext)
    return caption_path is not None and os.path.isfile(caption_path)


def _find_media_files_recursive(folder_path: str, extensions: List[str]) -> List[str]:
    extension_tuple = tuple(ext.lower() for ext in extensions)
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.lower().endswith(extension_tuple):
                file_list.append(os.path.join(root, file))
    return file_list


class _TreeScan:
    """Everything the dataset needs from one walk of the folder tree."""
    __slots__ = ("media_files", "dir_file_names", "signature_map", "stat_map",
                 "file_count", "dir_count", "seconds")


def _scan_dataset_tree(folder_path: str, extensions: List[str], max_workers: Optional[int] = None) -> _TreeScan:
    """One pass over the dataset tree instead of three.

    Startup used to list every folder three times: os.walk to find the media files,
    again to check which have caption sidecars, and again for size + mtime (the
    image-size and caption caches). Each folder is now listed ONCE -- in parallel,
    since an NVMe dataset drive serves concurrent directory reads well -- and all
    three results come from that listing.

    media_files is in EXACTLY the order _find_media_files_recursive (os.walk with
    sorted dirs and files) produces: the resumable sampler stores dataset positions,
    so a different order would resume onto different images. Same rules too: a
    folder that cannot be listed contributes nothing, and symlinked folders are not
    descended into (os.path.islink, as os.walk does).
    """
    started = time.monotonic()
    extension_tuple = tuple(ext.lower() for ext in extensions)
    if max_workers is None:
        try:
            max_workers = max(1, int(os.environ.get("AI_TOOLKIT_SCAN_THREADS", "8")))
        except ValueError:
            max_workers = 8

    def list_dir(directory):
        files, dirs, names, sigs, stats = [], [], set(), {}, {}
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        is_dir = False
                    (dirs if is_dir else files).append(entry.name)
                    try:
                        if entry.is_file():
                            names.add(os.path.normcase(entry.name))
                            st = entry.stat()  # served from the directory listing on Windows
                            sigs[entry.path] = f"{st.st_size}:{int(st.st_mtime)}"
                            stats[entry.path] = (st.st_size, st.st_mtime_ns)
                    except OSError:
                        continue
        except OSError:
            return directory, None
        files.sort()
        dirs.sort()
        descend = [os.path.join(directory, d) for d in dirs
                   if not os.path.islink(os.path.join(directory, d))]
        return directory, (files, descend, names, sigs, stats)

    listings = {}
    level = [folder_path]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while level:
            next_level = []
            for directory, listing in executor.map(list_dir, level):
                listings[directory] = listing
                if listing is not None:
                    next_level.extend(listing[1])
            level = next_level

    scan = _TreeScan()
    scan.media_files, scan.dir_file_names, scan.signature_map, scan.stat_map = [], {}, {}, {}
    scan.file_count, scan.dir_count = 0, 0
    stack = [folder_path]
    while stack:  # depth-first, pre-order, sorted: os.walk's order
        directory = stack.pop()
        listing = listings.get(directory)
        if listing is None:
            continue
        files, descend, names, sigs, stats = listing
        scan.dir_count += 1
        scan.file_count += len(sigs)
        scan.media_files.extend(os.path.join(directory, f) for f in files
                                if f.lower().endswith(extension_tuple))
        scan.dir_file_names[directory] = names
        scan.signature_map.update(sigs)
        scan.stat_map.update(stats)
        stack.extend(reversed(descend))
    scan.seconds = time.monotonic() - started
    return scan


def _filter_files_with_sidecar_captions(
        file_list: List[str],
        caption_ext: Optional[str],
        dir_file_names: Optional[dict] = None,
) -> tuple[List[str], int]:
    caption_ext = _normalize_caption_ext(caption_ext)
    if caption_ext is None:
        return [], len(file_list)
    # Cache one directory scan per unique folder instead of an os.path.isfile() stat
    # per image. On large datasets / slow or AV-scanned filesystems this turns tens
    # of thousands of stat() syscalls into a handful of directory listings. A caller
    # that already listed the tree (_scan_dataset_tree) passes those listings in.
    dir_file_names = dict(dir_file_names) if dir_file_names else {}
    filtered_file_list = []
    for file_path in file_list:
        directory = os.path.dirname(file_path)
        names = dir_file_names.get(directory)
        if names is None:
            names = set()
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        try:
                            if entry.is_file():
                                names.add(os.path.normcase(entry.name))
                        except OSError:
                            continue
            except OSError:
                pass
            dir_file_names[directory] = names
        caption_name = os.path.splitext(os.path.basename(file_path))[0] + caption_ext
        if os.path.normcase(caption_name) in names:
            filtered_file_list.append(file_path)
    return filtered_file_list, len(file_list) - len(filtered_file_list)


# Drop images whose area-equivalent resolution sqrt(w*h) is below this fraction of a
# dataset's selected resolution. Empirically-tuned floor below which low detail starts
# to hurt the LoRA. Area-based because sqrt(w*h) IS the bucketer's real_resolution and
# the image's latent-token count, independent of aspect ratio (a 512x2048 panorama and
# a 1024x1024 square both have sqrt=1024 and the same token count).
LOW_RES_MIN_FRACTION = 0.625


def _seeded_order(seed_text: str, paths) -> List[str]:
    """`paths` in an order that looks random and is fixed by `seed_text` and each file name
    alone. No path's place depends on which other paths are there: adding or removing one
    leaves the rest in the same order relative to each other.

    FORK: for dataset_config.control_from_same_folder (the file names are of one folder)."""
    base = hashlib.sha1(seed_text.encode('utf-8'))

    def rank(path):
        digest = base.copy()
        digest.update(os.path.basename(path).encode('utf-8'))
        return digest.digest()

    return sorted(paths, key=rank)


class DatasetBelowResolutionError(ValueError):
    """Raised when every image in a dataset is below its selected resolution (so the
    dataset is empty after the always-on low-res filter). get_dataloader_from_datasets
    catches this and skips just that dataset/resolution instead of crashing the run."""
    pass


def _filter_below_min_resolution(file_list, scale, min_resolution):
    """Drop items whose area-equivalent resolution int(sqrt(w*h)) (after scale) is
    below min_resolution. Area-based, so aspect-independent (matches the bucketer's
    get_resolution). Returns (kept_list, dropped_count). No-op if min_resolution falsy."""
    if not min_resolution:
        return file_list, 0
    min_resolution = int(min_resolution)
    kept = []
    dropped = 0
    for file_item in file_list:
        scaled_w = int(file_item.width * scale)
        scaled_h = int(file_item.height * scale)
        if int((scaled_w * scaled_h) ** 0.5) < min_resolution:
            dropped += 1
            continue
        kept.append(file_item)
    return kept, dropped


class ResumableShuffleSampler(Sampler[int]):
    def __init__(self, data_source, state: Optional[Dict[str, Any]] = None):
        self.data_source = data_source
        self._rng = random.Random()
        self._order: Optional[List[int]] = None
        self._processed = 0
        self._epoch = 0
        if state is not None:
            self.load_state(state)
        else:
            self.start_new_epoch()

    def __iter__(self):
        if self._order is None or len(self._order) != len(self.data_source):
            self.start_new_epoch()
        start_index = min(self._processed, len(self._order))
        for item_index in self._order[start_index:]:
            yield item_index

    def __len__(self):
        if self._order is None or len(self._order) != len(self.data_source):
            return len(self.data_source)
        return max(0, len(self._order) - self._processed)

    def mark_processed(self, count: int = 1):
        if self._order is None:
            return
        self._processed = min(len(self._order), self._processed + max(0, count))

    def start_new_epoch(self):
        order = list(range(len(self.data_source)))
        self._rng.shuffle(order)
        self._order = order
        self._processed = 0
        self._epoch += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "epoch": self._epoch,
            "order": list(self._order) if self._order is not None else None,
            "processed": self._processed,
            "rng_state": self._rng.getstate(),
            "data_length": len(self.data_source),
        }

    def load_state(self, state: Dict[str, Any]):
        order = state.get("order")
        if order is None:
            self.start_new_epoch()
            return
        if state.get("data_length") != len(self.data_source):
            raise ValueError("Saved sampler length does not match current dataset length")
        self._rng.setstate(state["rng_state"])
        self._order = list(order)
        self._processed = min(int(state.get("processed", 0)), len(self._order))
        self._epoch = int(state.get("epoch", 0))


class RescaleTransform:
    """Transform to rescale images to the range [-1, 1]."""

    def __call__(self, image):
        return image * 2 - 1


class NormalizeSDXLTransform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([ 0.0002, -0.1034, -0.1879])
    Standard Deviation: tensor([0.5436, 0.5116, 0.5033])
    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[0.0002, -0.1034, -0.1879],
            std=[0.5436, 0.5116, 0.5033],
        )(image)


class NormalizeSD15Transform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([-0.1600, -0.2450, -0.3227])
    Standard Deviation: tensor([0.5319, 0.4997, 0.5139])

    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[-0.1600, -0.2450, -0.3227],
            std=[0.5319, 0.4997, 0.5139],
        )(image)


def dto_collation(batch: List['FileItemDTO']):
    return DataLoaderBatchDTO(file_items=batch)



class ImageDataset(Dataset, CaptionMixin):
    def __init__(self, config):
        self.config = config
        self.name = self.get_config('name', 'dataset')
        self.path = self.get_config('path', required=True)
        self.scale = self.get_config('scale', 1)
        self.random_scale = self.get_config('random_scale', False)
        self.include_prompt = self.get_config('include_prompt', False)
        self.default_prompt = self.get_config('default_prompt', '')
        if self.include_prompt:
            self.caption_type = self.get_config('caption_ext', 'txt')
        else:
            self.caption_type = None
        self.must_include_caption = self.get_config('must_include_caption', False)
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else self.get_config('random_crop', False)

        self.resolution = self.get_config('resolution', 256)
        self.file_list = _find_media_files_recursive(self.path, image_extensions)
        if self.must_include_caption:
            self.file_list, skipped_caption_count = _filter_files_with_sidecar_captions(
                self.file_list,
                self.get_config('caption_ext', 'txt'),
            )
            print_acc(f"  -  Must Include Caption enabled; skipped {skipped_caption_count} files without sidecar captions")

        # this might take a while
        print_acc(f"  -  Preprocessing image dimensions")
        new_file_list = []
        bad_count = 0
        for file in tqdm(self.file_list):
            try:
                w, h = image_utils.get_image_size(file)
            except image_utils.UnknownImageFormat:
                img = exif_transpose(Image.open(file))
                w, h = img.size
            # img = Image.open(file)
            if int(min([w, h]) * self.scale) >= self.resolution:
                new_file_list.append(file)
            else:
                bad_count += 1

        self.file_list = new_file_list

        print_acc(f"  -  Found {len(self.file_list)} images")
        print_acc(f"  -  Found {bad_count} images that are too small")
        assert len(self.file_list) > 0, f"no images found in {self.path}"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RescaleTransform(),
        ])

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        try:
            img = exif_transpose(Image.open(img_path)).convert('RGB')
        except Exception as e:
            print_acc(f"Error opening image: {img_path}")
            print_acc(e)
            # make a noise image if we can't open it
            img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))

        # Downscale the source image first
        img = img.resize((int(img.size[0] * self.scale), int(img.size[1] * self.scale)), Image.BICUBIC)
        min_img_size = min(img.size)

        if self.random_crop:
            if self.random_scale and min_img_size > self.resolution:
                if min_img_size < self.resolution:
                    print_acc(
                        f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.resolution}, image file={img_path}")
                    scale_size = self.resolution
                else:
                    scale_size = random.randint(self.resolution, int(min_img_size))
                scaler = scale_size / min_img_size
                scale_width = int((img.width + 5) * scaler)
                scale_height = int((img.height + 5) * scaler)
                img = img.resize((scale_width, scale_height), Image.BICUBIC)
            img = transforms.RandomCrop(self.resolution)(img)
        else:
            img = transforms.CenterCrop(min_img_size)(img)
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        img = self.transform(img)

        if self.include_prompt:
            prompt = self.get_caption_item(index)
            return img, prompt
        else:
            return img





class AugmentedImageDataset(ImageDataset):
    def __init__(self, config):
        super().__init__(config)
        self.augmentations = self.get_config('augmentations', [])
        self.augmentations = [Augments(**aug) for aug in self.augmentations]

        augmentation_list = []
        for aug in self.augmentations:
            # make sure method name is valid
            assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
            # get the method
            method = getattr(A, aug.method_name)
            # add the method to the list
            augmentation_list.append(method(**aug.params))

        self.aug_transform = A.Compose(augmentation_list)
        self.original_transform = self.transform
        # replace transform so we get raw pil image
        self.transform = transforms.Compose([])

    def __getitem__(self, index):
        # get the original image
        # image is a PIL image, convert to bgr
        pil_image = super().__getitem__(index)
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        augmented = self.aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # return both # return image as 0 - 1 tensor
        return transforms.ToTensor()(pil_image), transforms.ToTensor()(augmented)


class PairedImageDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.size = self.get_config('size', 512)
        self.path = self.get_config('path', None)
        self.pos_folder = self.get_config('pos_folder', None)
        self.neg_folder = self.get_config('neg_folder', None)

        self.default_prompt = self.get_config('default_prompt', '')
        self.network_weight = self.get_config('network_weight', 1.0)
        self.pos_weight = self.get_config('pos_weight', self.network_weight)
        self.neg_weight = self.get_config('neg_weight', self.network_weight)

        supported_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPEG', '.JPG', '.PNG', '.WEBP')
        self.must_include_caption = self.get_config('must_include_caption', False)
        caption_ext = self.get_config('caption_ext', 'txt')

        if self.pos_folder is not None and self.neg_folder is not None:
            # find matching files
            self.pos_file_list = _find_media_files_recursive(self.pos_folder, image_extensions)
            self.neg_file_list = _find_media_files_recursive(self.neg_folder, image_extensions)
            if self.must_include_caption:
                self.pos_file_list, skipped_pos_caption_count = _filter_files_with_sidecar_captions(
                    self.pos_file_list,
                    caption_ext,
                )
                self.neg_file_list, skipped_neg_caption_count = _filter_files_with_sidecar_captions(
                    self.neg_file_list,
                    caption_ext,
                )
                print_acc(
                    f"  -  Must Include Caption enabled; skipped "
                    f"{skipped_pos_caption_count + skipped_neg_caption_count} paired files without sidecar captions"
                )

            matched_files = []
            for pos_file in self.pos_file_list:
                pos_file_no_ext = os.path.splitext(pos_file)[0]
                for neg_file in self.neg_file_list:
                    neg_file_no_ext = os.path.splitext(neg_file)[0]
                    if os.path.basename(pos_file_no_ext) == os.path.basename(neg_file_no_ext):
                        matched_files.append((neg_file, pos_file))
                        break

            # remove duplicates
            matched_files = [t for t in (set(tuple(i) for i in matched_files))]

            self.file_list = matched_files
            print_acc(f"  -  Found {len(self.file_list)} matching pairs")
        else:
            self.file_list = _find_media_files_recursive(self.path, image_extensions)
            if self.must_include_caption:
                self.file_list, skipped_caption_count = _filter_files_with_sidecar_captions(
                    self.file_list,
                    caption_ext,
                )
                print_acc(
                    f"  -  Must Include Caption enabled; skipped {skipped_caption_count} files without sidecar captions"
                )
            print_acc(f"  -  Found {len(self.file_list)} images")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RescaleTransform(),
        ])

    def get_all_prompts(self):
        prompts = []
        for index in range(len(self.file_list)):
            prompts.append(self.get_prompt_item(index))

        # remove duplicates
        prompts = list(set(prompts))
        return prompts

    def __len__(self):
        return len(self.file_list)

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def get_prompt_item(self, index):
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path_or_tuple[0])[0]
            prompt_path = path_no_ext + '.txt'
            if not os.path.exists(prompt_path):
                path_no_ext = os.path.splitext(img_path_or_tuple[1])[0]
                prompt_path = path_no_ext + '.txt'
        else:
            img_path = img_path_or_tuple
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + '.txt'

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # remove any newlines
                prompt = prompt.replace('\n', ', ')
                # remove new lines for all operating systems
                prompt = prompt.replace('\r', ', ')
                prompt_split = prompt.split(',')
                # remove empty strings
                prompt_split = [p.strip() for p in prompt_split if p.strip()]
                # join back together
                prompt = ', '.join(prompt_split)
        else:
            prompt = self.default_prompt
        return prompt

    def __getitem__(self, index):
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            # load both images
            img_path = img_path_or_tuple[0]
            img1 = exif_transpose(Image.open(img_path)).convert('RGB')
            img_path = img_path_or_tuple[1]
            img2 = exif_transpose(Image.open(img_path)).convert('RGB')

            # always use # 2 (pos)
            bucket_resolution = get_bucket_for_image_size(
                width=img2.width,
                height=img2.height,
                resolution=self.size,
                # divisibility=self.
            )

            # images will be same base dimension, but may be trimmed. We need to shrink and then central crop
            if bucket_resolution['width'] > bucket_resolution['height']:
                img1_scale_to_height = bucket_resolution["height"]
                img1_scale_to_width = int(img1.width * (bucket_resolution["height"] / img1.height))
                img2_scale_to_height = bucket_resolution["height"]
                img2_scale_to_width = int(img2.width * (bucket_resolution["height"] / img2.height))
            else:
                img1_scale_to_width = bucket_resolution["width"]
                img1_scale_to_height = int(img1.height * (bucket_resolution["width"] / img1.width))
                img2_scale_to_width = bucket_resolution["width"]
                img2_scale_to_height = int(img2.height * (bucket_resolution["width"] / img2.width))

            img1_crop_height = bucket_resolution["height"]
            img1_crop_width = bucket_resolution["width"]
            img2_crop_height = bucket_resolution["height"]
            img2_crop_width = bucket_resolution["width"]

            # scale then center crop images
            img1 = img1.resize((img1_scale_to_width, img1_scale_to_height), Image.BICUBIC)
            img1 = transforms.CenterCrop((img1_crop_height, img1_crop_width))(img1)
            img2 = img2.resize((img2_scale_to_width, img2_scale_to_height), Image.BICUBIC)
            img2 = transforms.CenterCrop((img2_crop_height, img2_crop_width))(img2)

            # combine them side by side
            img = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
            img.paste(img1, (0, 0))
            img.paste(img2, (img1.width, 0))
        else:
            img_path = img_path_or_tuple
            img = exif_transpose(Image.open(img_path)).convert('RGB')
            height = self.size
            # determine width to keep aspect ratio
            width = int(img.size[0] * height / img.size[1])

            # Downscale the source image first
            img = img.resize((width, height), Image.BICUBIC)

        prompt = self.get_prompt_item(index)
        img = self.transform(img)

        return img, prompt, (self.neg_weight, self.pos_weight)


class AiToolkitDataset(LatentCachingMixin, ControlCachingMixin, CLIPCachingMixin, TextEmbeddingCachingMixin, BucketsMixin, CaptionMixin, Dataset):

    def __init__(
            self,
            dataset_config: 'DatasetConfig',
            batch_size=1,
            sd: 'StableDiffusion' = None,
            cache_progress_hook: Optional[Callable[[str, int, int, float, str], None]] = None,
    ):
        self.dataset_config = dataset_config
        # update bucket divisibility
        self.dataset_config.bucket_tolerance = sd.get_bucket_divisibility()
        self.is_video = dataset_config.num_frames > 1
        super().__init__()
        folder_path = dataset_config.folder_path
        self.dataset_path = dataset_config.dataset_path
        if self.dataset_path is None:
            self.dataset_path = folder_path

        self.is_caching_latents = dataset_config.cache_latents or dataset_config.cache_latents_to_disk
        self.is_caching_latents_to_memory = dataset_config.cache_latents
        self.is_caching_latents_to_disk = dataset_config.cache_latents_to_disk
        self.is_caching_clip_vision_to_disk = dataset_config.cache_clip_vision_to_disk
        self.is_generating_controls = len(dataset_config.controls) > 0
        self.epoch_num = 0

        self.sd = sd
        self.sd_device = self.sd.device if self.sd is not None else "cpu"

        if self.sd is None and self.is_caching_latents:
            raise ValueError(f"sd is required for caching latents")

        self.caption_type = dataset_config.caption_ext
        self.default_caption = dataset_config.default_caption
        self.random_scale = dataset_config.random_scale
        self.scale = dataset_config.scale
        self.batch_size = batch_size
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else dataset_config.random_crop
        self.resolution = dataset_config.resolution
        self.caption_dict = None
        self.file_list: List['FileItemDTO'] = []
        self._cache_progress_hooks: List[Callable[[str, int, int, float, str], None]] = []
        if cache_progress_hook is not None:
            self.add_cache_progress_hook(cache_progress_hook)

        # check if dataset_path is a folder or json
        tree_scan = None
        if os.path.isdir(self.dataset_path):
            extensions = image_extensions
            if self.is_video:
                # only look for videos
                extensions = video_extensions
            # one listing of the tree feeds the file list, the caption check and the
            # size/caption caches below (it used to be three full scans)
            tree_scan = _scan_dataset_tree(self.dataset_path, extensions)
            file_list = tree_scan.media_files
            print_acc(f"  -  Scanned {tree_scan.file_count} files in {tree_scan.dir_count} folders "
                      f"in {tree_scan.seconds:.1f}s")
        else:
            # assume json
            with open(self.dataset_path, 'r') as f:
                self.caption_dict = json.load(f)
                # keys are file paths
                file_list = sorted(self.caption_dict.keys())
                
        # remove items in the _controls_ folder
        file_list = [x for x in file_list if not os.path.basename(os.path.dirname(x)) == "_controls"]

        if self.dataset_config.must_include_caption:
            if self.dataset_config.caches_only:
                print_acc(
                    "  -  Train from Caches Only enabled; skipping Must Include Caption check"
                )
            else:
                file_list, skipped_caption_count = _filter_files_with_sidecar_captions(
                    file_list,
                    self.dataset_config.caption_ext,
                    dir_file_names=tree_scan.dir_file_names if tree_scan is not None else None,
                )
                print_acc(
                    f"  -  Must Include Caption enabled; skipped {skipped_caption_count} files without sidecar captions"
                )

        if self.dataset_config.num_repeats > 1:
            # repeat the list
            file_list = file_list * self.dataset_config.num_repeats

        if self.dataset_config.standardize_images:
            if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                NormalizeMethod = NormalizeSDXLTransform
            else:
                NormalizeMethod = NormalizeSD15Transform

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
                NormalizeMethod(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
            ])

        # this might take a while
        print_acc(f"Dataset: {self.dataset_path}")
        if self.is_video:
            print_acc(f"  -  Preprocessing video dimensions")
        else:
            print_acc(f"  -  Preprocessing image dimensions")
        dataset_folder = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            dataset_folder = os.path.dirname(dataset_folder)

        dataset_size_file_candidates = get_cache_file_candidates(
            dataset_folder,
            '.aitk_size.json',
            override_root=self.dataset_config.cache_dir_override,
            min_free_gb=self.dataset_config.cache_dir_override_min_free_gb,
        )
        dataset_size_file = dataset_size_file_candidates[0]
        if self.dataset_config.cache_dir_override:
            if has_sufficient_free_space(
                    self.dataset_config.cache_dir_override,
                    self.dataset_config.cache_dir_override_min_free_gb,
            ):
                print_acc(
                    f"  -  Using cache override root {self.dataset_config.cache_dir_override}"
                )
            else:
                print_acc(
                    "  -  Cache override root is below the free-space threshold; "
                    "using dataset-local caches"
                )
        dataloader_version = "0.1.2"
        existing_size_file = next(
            (path for path in dataset_size_file_candidates if os.path.exists(path)),
            None,
        )
        if existing_size_file is not None:
            try:
                with open(existing_size_file, 'r') as f:
                    self.size_database = json.load(f)
                
                if "__version__" not in self.size_database or self.size_database["__version__"] != dataloader_version:
                    print_acc("Upgrading size database to new version")
                    # old version, delete and recreate
                    self.size_database = {}
            except Exception as e:
                print_acc(f"Error loading size database: {existing_size_file}")
                print_acc(e)
                self.size_database = {}
        else:
            self.size_database = {}
        
        # Cache portability: file keys carry the separator of whatever OS wrote them
        # (Windows writes \\sub\\img.jpg, Linux writes /sub/img.jpg). Normalize to THIS
        # OS's separator so a cache built on one machine resolves on another.
        # Without this a Windows-built cache is a 100%% miss on Linux, twice over:
        # posixpath.basename("/ds\\sub\\img.jpg") -> "ds\\sub\\img.jpg" (which feeds
        # the latent cache key) and posixpath.dirname(...) -> "/" (so it looks for
        # _latent_cache one directory too high).
        if any(('\\' in k or '/' in k) for k in self.size_database.keys()):
            self.size_database = {
                (k if k == "__version__" else k.replace('\\', os.sep).replace('/', os.sep)): v
                for k, v in self.size_database.items()
            }
        self.size_database["__version__"] = dataloader_version

        # Train from Caches Only: recover files recorded in the size database that
        # no longer exist on disk. Their dimensions (and cache-key ingredients) come
        # from the database, and their latents/text embeddings from the disk caches,
        # so the source images and caption files are not needed.
        if self.dataset_config.caches_only and os.path.isdir(self.dataset_path):
            _existing_paths = set(file_list)
            _recovered = []
            for _fk, _entry in self.size_database.items():
                if _fk == "__version__":
                    continue
                if not isinstance(_entry, (list, tuple)) or len(_entry) < 3:
                    continue
                if _fk.startswith(os.sep) or _fk.startswith('/') or _fk.startswith('\\'):
                    _candidate = dataset_folder + _fk
                else:
                    _candidate = os.path.join(dataset_folder, _fk)
                if _candidate in _existing_paths:
                    continue
                _ext = os.path.splitext(_candidate)[1].lower()
                _expected_exts = video_extensions if self.is_video else image_extensions
                if _ext.lstrip('.') not in [e.lstrip('.').lower() for e in _expected_exts]:
                    continue
                if os.path.basename(os.path.dirname(_candidate)) == "_controls":
                    continue
                _recovered.append(_candidate)
            if len(_recovered) > 0:
                print_acc(
                    f"  -  Train from Caches Only: recovered {len(_recovered)} missing "
                    f"files from the size database"
                )
                if self.dataset_config.num_repeats > 1:
                    _recovered = _recovered * self.dataset_config.num_repeats
                file_list = file_list + sorted(_recovered)

        # set latent space version
        latent_space_version = "sd1"
        if self.sd is not None and self.sd.model_config.latent_space_version is not None:
            latent_space_version = self.sd.model_config.latent_space_version
        elif self.sd.is_xl:
            latent_space_version = 'sdxl'
        elif self.sd.is_v3:
            latent_space_version = 'sd3'
        elif self.sd.is_auraflow:
            latent_space_version = 'sdxl'
        elif self.sd.is_flux:
            latent_space_version = 'flux1'
        elif self.sd.model_config.is_pixart_sigma:
            latent_space_version = 'sdxl'
        else:
            latent_space_version = self.sd.model_config.arch if self.sd is not None else "sd1"
        
        # Pre-scan each unique directory ONCE (os.scandir returns size+mtime without a
        # separate os.stat per file, especially on Windows) so the per-image cache
        # signature check is a dict lookup instead of ~N stat() syscalls — a big win on
        # large datasets / slow disks. Falls back to a per-file stat if a path is missed.
        signature_map = {}
        stat_map = {}
        _scanned_dirs = set()
        if tree_scan is not None:
            # already collected by the single tree scan above
            signature_map, stat_map = tree_scan.signature_map, tree_scan.stat_map
            _scanned_dirs = set(tree_scan.dir_file_names)
        for file in file_list:
            d = os.path.dirname(file)
            if d in _scanned_dirs:
                continue
            _scanned_dirs.add(d)
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        try:
                            if entry.is_file():
                                st = entry.stat()
                                signature_map[entry.path] = f"{st.st_size}:{int(st.st_mtime)}"
                                stat_map[entry.path] = (st.st_size, st.st_mtime_ns)
                        except OSError:
                            continue
            except OSError:
                continue

        def _save_size_database():
            # Atomic write (temp + os.replace) so a crash/BSOD mid-write can't corrupt
            # the cache, and so partial progress is persisted for the next run.
            try:
                os.makedirs(os.path.dirname(dataset_size_file), exist_ok=True)
                _tmp_path = dataset_size_file + '.tmp'
                with open(_tmp_path, 'w') as f:
                    json.dump(self.size_database, f)
                os.replace(_tmp_path, dataset_size_file)
            except Exception as e:
                print_acc(f"Warning: could not save image-size cache: {e}")

        bad_count = 0
        _prep_started = time.monotonic()
        _captions_cached = 0
        for _idx, file in enumerate(tqdm(file_list)):
            try:
                file_item = FileItemDTO(
                    sd=self.sd,
                    path=file,
                    dataset_config=dataset_config,
                    dataloader_transforms=self.transform,
                    size_database=self.size_database,
                    signature_map=signature_map,
                    stat_map=stat_map,
                    dataset_root=dataset_folder,
                    encode_control_in_text_embeddings=self.sd.encode_control_in_text_embeddings if self.sd else False,
                    text_embedding_space_version=(
                        self.sd.get_text_embedding_space_version()
                        if hasattr(self.sd, "get_text_embedding_space_version")
                        else (self.sd.model_config.arch if self.sd else "sd1")
                    ),
                    te_padding_side=self.sd.te_padding_side if self.sd else "right",
                    latent_space_version=latent_space_version,
                )
                self.file_list.append(file_item)
                _captions_cached += int(getattr(file_item, "_caption_from_cache", False))
            except Exception as e:
                print_acc(traceback.format_exc())
                if self.is_video:
                    print_acc(f"Error processing video: {file}")
                else:
                    print_acc(f"Error processing image: {file}")
                print_acc(e)
                bad_count += 1
            # incrementally persist the cache so a crash mid-build isn't lost
            if (_idx + 1) % 5000 == 0:
                _save_size_database()

        # final save of the size database
        _save_size_database()
        
        _noun_found = 'videos' if self.is_video else 'images'
        _prep_seconds = time.monotonic() - _prep_started
        _caption_note = ""
        if self.dataset_config.caches_only or self.dataset_config.cache_text_embeddings:
            _caption_note = (f"; captions: {_captions_cached} unchanged (from cache), "
                             f"{len(self.file_list) - _captions_cached} read from disk")
        print_acc(f"  -  Found {len(self.file_list)} {_noun_found} in {_prep_seconds:.1f}s{_caption_note}")
        if self.dataset_config.caches_only:
            assert len(self.file_list) > 0, (
                f"no {_noun_found} or cached entries found in {self.dataset_path}. "
                f"Train from Caches Only needs the dataset's size database "
                f"(.aitk_size.json) and disk caches from a previous run."
            )
        else:
            assert len(self.file_list) > 0, f"no {_noun_found} found in {self.dataset_path}"

        # FORK: control_from_same_folder wants to know whether any folder held two images
        # before the filter below
        self._folder_pair_before_low_res_filter = False
        if self.dataset_config.control_from_same_folder:
            folders_seen = set()
            for path in {file_item.path for file_item in self.file_list}:
                folder = os.path.dirname(path)
                if folder in folders_seen:
                    self._folder_pair_before_low_res_filter = True
                    break
                folders_seen.add(folder)

        # Always-on "skip low-res images" filter. An image's effective training
        # resolution is its area-equivalent sqrt(w*h) (= the bucketer's real_resolution
        # and its latent-token count), so we filter on AREA, not min/max side. Drop
        # images whose sqrt(w*h) (after scale) is below LOW_RES_MIN_FRACTION of THIS
        # dataset's selected resolution. (resolution is a single int per dataset;
        # multi-resolution is split upstream, so each split checks against its own res.)
        _res = self.resolution
        if isinstance(_res, (list, tuple)):
            _res = min(_res) if len(_res) > 0 else 0
        _res = int(_res)
        _floor = int(_res * LOW_RES_MIN_FRACTION)
        if _floor > 0:
            self.file_list, _dropped_small = _filter_below_min_resolution(
                self.file_list, self.scale, _floor,
            )
            if _dropped_small > 0:
                _noun = 'videos' if self.is_video else 'images'
                print_acc(
                    f"  -  Skipped {_dropped_small} low-res {_noun} "
                    f"(sqrt(w*h) < {_floor} = {LOW_RES_MIN_FRACTION:g} x res {_res}); "
                    f"{len(self.file_list)} remain."
                )
            if len(self.file_list) == 0:
                raise DatasetBelowResolutionError(
                    f"no {'videos' if self.is_video else 'images'} in {self.dataset_path} are large "
                    f"enough (need sqrt(w*h) >= {_floor} = {LOW_RES_MIN_FRACTION:g} x the selected "
                    f"resolution {_res})"
                )

        # handle x axis flips
        if self.dataset_config.flip_x:
            print_acc("  -  adding x axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the x axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_x = True
                self.file_list.append(new_file_item)

        # handle y axis flips
        if self.dataset_config.flip_y:
            print_acc("  -  adding y axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the y axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_y = True
                self.file_list.append(new_file_item)

        if self.dataset_config.flip_x or self.dataset_config.flip_y:
            if self.is_video:
                print_acc(f"  -  Found {len(self.file_list)} videos after adding flips")
            else:
                print_acc(f"  -  Found {len(self.file_list)} images after adding flips")

        # FORK: before setup_epoch(), which builds the buckets and the caches from file_list
        self.assign_same_folder_control_images()

        self.setup_epoch()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Windows dataloader workers use spawn + pickle. Keep worker copies free of
        # the live StableDiffusion object and main-process hooks.
        state["sd"] = None
        state["_cache_progress_hooks"] = []
        return state

    def add_cache_progress_hook(self, hook: Callable[[str, int, int, float, str], None]):
        if hook is None:
            return
        if hook not in self._cache_progress_hooks:
            self._cache_progress_hooks.append(hook)

    def remove_cache_progress_hook(self, hook: Callable[[str, int, int, float, str], None]):
        self._cache_progress_hooks = [
            existing_hook for existing_hook in self._cache_progress_hooks if existing_hook != hook
        ]

    def report_cache_progress(
            self,
            cache_kind: str,
            processed: int,
            total: int,
            elapsed_seconds: float,
    ):
        if len(self._cache_progress_hooks) == 0:
            return

        for hook in list(self._cache_progress_hooks):
            try:
                hook(cache_kind, processed, total, elapsed_seconds, self.dataset_path)
            except Exception:
                continue

    def _drop_uncached_missing_items(self):
        # Train from Caches Only: items whose source file is gone AND whose cached
        # latent cannot be resolved can never be encoded; drop them instead of
        # crashing when the encoder tries to open the missing image.
        kept = []
        dropped = 0
        for file_item in self.file_list:
            if not os.path.exists(file_item.path) and file_item.get_latent_storage_entry() is None:
                dropped += 1
                continue
            kept.append(file_item)
        if dropped > 0:
            print_acc(
                f"  -  Train from Caches Only: dropped {dropped} items with no cached "
                f"latent and no source file; {len(kept)} remain"
            )
            self.file_list = kept
            if len(self.file_list) == 0:
                raise Exception(
                    f"Train from Caches Only: no cached latents could be resolved for "
                    f"{self.dataset_path}"
                )
            if self.dataset_config.buckets:
                # batch indices reference file_list positions; rebuild them
                self.setup_buckets(quiet=True)

    def assign_same_folder_control_images(self):
        """FORK: dataset_config.control_from_same_folder.

        Every target's reference (control) images are other images of the target's own
        folder. With one folder per character, the references then show the same character
        in different pictures, so a reference-conditioned model has to carry the identity
        over rather than copy the target. "Other images" are the ones this dataset trains
        on: an image that was skipped as too small for this resolution is not a reference
        either.

        When they are chosen:
          * text embeddings cached, on a model that encodes the references into them: once
            per target, from a seed, so the cache holds the matching embedding and stays
            valid from run to run. A target that is in the dataset more than once (through
            num_repeats, the flips or targets_per_folder) gets a choice for each of its
            copies, pass by pass. Every copy ranks the other images of the folder in an order
            of its own (see _seeded_order) and takes the first ones that no earlier copy of
            the same target has taken; when the images left would not do for a copy, a new
            round starts with all of them. So within a round no image serves the same target
            twice, and a round leaves out fewer images than the copy that ends it takes
            (none, with a count of one). A copy's choice changes when one of its images goes
            away, when a new image ranks ahead of them, when an earlier copy of its round
            changes in a way that takes or frees one of them, or when the rounds end
            elsewhere (as they do in a folder that gains or loses an image): editing a
            folder redoes part of its cache, more of it the more copies a target has. The
            first copy of a target is not touched by the others. (A target that takes every
            other image of its folder, because there are no more than its count, takes a new
            one too.)
          * otherwise: again every time a target is loaded.
        How many comes from control_from_same_folder_count. A target with no other image of
        its folder in this dataset has nothing to be paired with and is left out.

        Two settings keep large folders from outweighing small ones (one or the other):
          * control_from_same_folder_max_targets: only that many images of a folder, chosen
            from the seed, are trained as targets, the same ones in every pass. The others
            are dropped from the dataset but stay available as references. A folder with
            fewer images gives what it has. 'min' takes the number from the dataset: as many
            as the smallest folder that has a pair.
          * control_from_same_folder_targets_per_folder: every folder gives that many
            targets per pass, a pass being one copy of the dataset as num_repeats and the
            flips make them. The images of a folder are put in an order by the seed and
            handed out in turn, going round as often as it takes. So a folder with more
            images than a pass takes gives other ones in the next pass, and a folder with
            fewer gives its images several times, each copy with a choice of references of
            its own (above). 'median', 'mean' and 'min' take the number from the sizes of
            the folders that have a pair. Two limits:
              - a target comes up only as often per pass as the other images of its folder
                can serve it without one of them serving twice, at count_min a time: a
                folder of n images gives at most n x max(1, (n - 1) // count_min) per pass.
                With a fixed count, no pairing of a target with its references then comes
                up more often than there are passes;
              - a number larger than the largest folder is brought down to its size: every
                folder would only repeat its images within a pass, which is what
                num_repeats is for.
        """
        dataset_config = self.dataset_config
        if not dataset_config.control_from_same_folder:
            return
        # the trainer has run this for every dataset before building the first; again here
        # for a dataset that is built on its own
        check_control_from_same_folder(self.sd.model_config if self.sd is not None else None, dataset_config)
        if not getattr(self.sd, 'use_raw_control_images', False):
            raise ValueError("control_from_same_folder: the model does not take reference images at their own size")

        seed = dataset_config.control_from_same_folder_seed
        dataset_root = self.dataset_path if os.path.isdir(self.dataset_path) else os.path.dirname(self.dataset_path)

        def seed_name(path):
            # the same on every machine and drive, so that the seed strings are too
            try:
                return os.path.relpath(path, dataset_root).replace(os.sep, '/')
            except ValueError:  # another drive on Windows, which only a json file list can reach
                return path.replace(os.sep, '/')

        folder_paths: Dict[str, set] = {}
        for file_item in self.file_list:
            folder_paths.setdefault(os.path.dirname(file_item.path), set()).add(file_item.path)
        # one sorted tuple per folder, shared by its file items
        folder_pools = {folder: tuple(sorted(paths)) for folder, paths in folder_paths.items()}
        # the folders that have a pair, in an order that is the same on every machine
        paired = sorted((folder for folder, pool in folder_pools.items() if len(pool) >= 2), key=seed_name)

        def folder_sizes(which):
            """'min', 'median' or 'mean' of the sizes of the folders that have a pair, as a
            whole number (a half rounds up)."""
            sizes = sorted(len(folder_pools[folder]) for folder in paired)
            if len(sizes) == 0:
                return None
            if which == 'min':
                return sizes[0]
            if which == 'median':
                middle = len(sizes) // 2
                if len(sizes) % 2 == 1:
                    return sizes[middle]
                return (sizes[middle - 1] + sizes[middle] + 1) // 2
            return (2 * sum(sizes) + len(sizes)) // (2 * len(sizes))

        # the images of each folder that are trained as targets: all, or the first so many ...
        max_targets = dataset_config.control_from_same_folder_max_targets
        smallest_text = None
        if max_targets == 'min':
            # as many as the smallest folder that has a pair
            max_targets = folder_sizes('min')
            smallest = sorted(seed_name(folder) for folder in paired if len(folder_pools[folder]) == max_targets)
            if len(smallest) > 0:
                smallest_text = smallest[0] + (f" and {len(smallest) - 1} more" if len(smallest) > 1 else "")
        folder_targets: Dict[str, set] = {}
        if max_targets is not None:
            for folder, pool in folder_pools.items():
                if len(pool) > max_targets:
                    order = _seeded_order(f"{seed}|targets|{seed_name(folder)}|", pool)
                    folder_targets[folder] = set(order[:max_targets])

        count_min, count_max = dataset_config.control_from_same_folder_count

        # ... or so many per folder and pass, going round the folder's images
        per_folder = dataset_config.control_from_same_folder_targets_per_folder
        passes = 0
        per_pass: Dict[str, int] = {}  # what each folder gives per pass
        in_turn: List['FileItemDTO'] = []  # its targets, pass by pass
        if per_folder is not None and len(paired) > 0:
            sizes = sorted(len(folder_pools[folder]) for folder in paired)
            if isinstance(per_folder, str):
                how, per_folder = f"{per_folder} -> ", folder_sizes(per_folder)
                note = ""
            elif per_folder > sizes[-1]:
                # every folder would only repeat its images within a pass, which num_repeats
                # does. (It also keeps a number of another kind, typed here by mistake, within
                # what the largest folder could give.)
                how, per_folder = f"{per_folder} -> ", sizes[-1]
                note = ", the size of the largest folder (num_repeats gives more passes)"
            else:
                how = note = ""
            # the copies that num_repeats and the flips made of every image, in the order they
            # were made: the p-th copy of each image is the one of pass p. (An image that has
            # fewer copies than the others, because a read failed once, lends another of its
            # copies to the passes it misses, which may be flipped differently.)
            copies: Dict[str, list] = {}
            for file_item in self.file_list:
                copies.setdefault(file_item.path, []).append(file_item)
            passes = max(len(copies[path]) for folder in paired for path in folder_pools[folder])
            for folder in paired:
                # a target only as often per pass as the other images of its folder can serve it
                # without one of them serving twice (they go count_min at a time, at least)
                size = len(folder_pools[folder])
                per_pass[folder] = min(per_folder, size * max(1, (size - 1) // count_min))
            # (said before the copies are made)
            print_acc(
                f"  -  control_from_same_folder_targets_per_folder: {how}{per_folder} per folder and "
                f"pass{note} (the folders hold {sizes[0]} to {sizes[-1]} images)"
            )
            flips = dataset_config.flip_x or dataset_config.flip_y
            fewer = sum(1 for folder in paired if per_pass[folder] < per_folder)
            total = sum(per_pass.values()) * passes
            if fewer == 0:
                each = f"{per_folder * passes} target(s) from each of the {len(paired)} folder(s), {total} in all"
            else:
                each = (f"{per_folder * passes} target(s) from each of {len(paired) - fewer} folder(s) and "
                        f"fewer from {fewer}, {total} in all")
            print_acc(f"  -  {passes} pass(es) (num_repeats{', times the flips' if flips else ''}): {each}")
            used = set()  # id() of the file items that stay
            added: List['FileItemDTO'] = []
            for folder in paired:
                order = _seeded_order(f"{seed}|targets|{seed_name(folder)}|", folder_pools[folder])
                for slot in range(per_pass[folder] * passes):
                    path_copies = copies[order[slot % len(order)]]
                    source = path_copies[(slot // per_pass[folder]) % len(path_copies)]
                    if id(source) not in used:
                        used.add(id(source))
                    else:
                        # this pass has the image already (its folder holds fewer than it
                        # gives): one more of it. What a file item holds besides plain values
                        # at this point (its config, its transforms) is only read, and every
                        # fetch works on a deep copy of the item, so a shallow copy will do.
                        source = copy.copy(source)
                        added.append(source)
                    in_turn.append(source)
            # the ones that are there already keep their order; the folders without a pair
            # stay for the loop below to count them
            self.file_list = [
                file_item for file_item in self.file_list
                if id(file_item) in used or len(folder_pools[os.path.dirname(file_item.path)]) < 2
            ] + added

        # per target: how many copies of it have their references, and the images those copies
        # have taken in the round that is under way
        reference_rounds: Dict[str, tuple] = {}

        def fixed_references(file_item):
            """The references of one more copy of this target."""
            path = file_item.path
            others = [other for other in file_item.same_folder_control_pool if other != path]
            copy_index, taken = reference_rounds.get(path, (0, frozenset()))
            # every copy ranks the folder's other images in an order of its own ...
            seed_text = f"{seed}|references|{seed_name(path)}|{copy_index}|"
            draw = int.from_bytes(hashlib.sha1(f"{seed_text}count".encode('utf-8')).digest()[:8], 'big')
            count = min(count_min + draw % (count_max - count_min + 1), len(others))
            if count > len(others) - len(taken):
                # too few left that no earlier copy has had: a new round, with all of them
                taken = frozenset()
            # ... and takes the first ones that no earlier copy has taken in this round
            references = [other for other in _seeded_order(seed_text, others) if other not in taken][:count]
            reference_rounds[path] = (copy_index + 1, taken.union(references))
            return references

        # cached embeddings contain the references, so the choice has to hold still
        fixed = bool(
            dataset_config.cache_text_embeddings
            and getattr(self.sd, 'encode_control_in_text_embeddings', False)
        )
        kept: List['FileItemDTO'] = []
        target_paths, alone_paths, reference_only_paths = set(), set(), set()
        folder_targets_made: Dict[str, int] = {}
        for file_item in self.file_list:
            folder = os.path.dirname(file_item.path)
            pool = folder_pools[folder]
            if len(pool) < 2:
                alone_paths.add(file_item.path)
                continue
            if folder in folder_targets and file_item.path not in folder_targets[folder]:
                reference_only_paths.add(file_item.path)
                continue
            file_item.same_folder_control_pool = pool
            file_item.has_control_image = True
            if not fixed:
                file_item.same_folder_control_redraw = True
                file_item.control_path = None
            folder_targets_made[folder] = folder_targets_made.get(folder, 0) + 1
            target_paths.add(file_item.path)
            kept.append(file_item)
        self.file_list = kept
        if passes > 0:
            # the images that no pass got round to
            reference_only_paths = {path for folder in paired for path in folder_pools[folder]} - target_paths
        # (how many targets of each folder have the references that another copy of the same
        # image has: for the report)
        pairings, folder_repeats = set(), {}
        if fixed:
            # The copies of a target take their references pass by pass, so that a round's
            # choices fall to the copies of one pass (with the flips: of one flip state) as far
            # as they go. Without targets_per_folder the file list is in that order already.
            for file_item in (in_turn if passes > 0 else kept):
                file_item.control_path = fixed_references(file_item)
                pairing = (
                    file_item.path, getattr(file_item, 'flip_x', False), getattr(file_item, 'flip_y', False),
                    frozenset(file_item.control_path),
                )
                if pairing in pairings:
                    folder = os.path.dirname(file_item.path)
                    folder_repeats[folder] = folder_repeats.get(folder, 0) + 1
                else:
                    pairings.add(pairing)

        count_text = f"{count_min}" if count_min == count_max else f"{count_min}-{count_max}"
        folders_used = len(paired)
        if fixed:
            when = "chosen once per target for this run (text embeddings are cached)"
        else:
            when = "chosen again every time a target is loaded"
        print_acc(
            f"  -  References from the same folder: {len(target_paths)} target image(s) in "
            f"{folders_used} folder(s), {count_text} per target, {when}"
        )
        if smallest_text is not None:
            print_acc(
                f"  -  control_from_same_folder_max_targets: min -> {max_targets} per folder, "
                f"the size of the smallest folder ({smallest_text})"
            )
        if passes > 0:
            def described(folder):
                return f"{seed_name(folder)}, {len(folder_pools[folder])} images"

            repeating = [folder for folder in paired if per_pass[folder] > len(folder_pools[folder])]
            if len(repeating) > 0:
                most = min(repeating, key=lambda folder: (
                    -per_pass[folder] / len(folder_pools[folder]), seed_name(folder)))
                print_acc(
                    f"  -  {len(repeating)} folder(s) give more targets per pass than they hold images, so an "
                    f"image of theirs is a target more than once per pass (the most in {described(most)}: "
                    f"about {per_pass[most] / len(folder_pools[most]):.1f} times each)"
                )
            limited = [folder for folder in paired if per_pass[folder] < per_folder]
            if len(limited) > 0:
                smallest = min(limited, key=lambda folder: (len(folder_pools[folder]), seed_name(folder)))
                print_acc(
                    f"  -  {len(limited)} folder(s) hold too few images to give {per_folder} per pass: a target "
                    f"comes up only as often per pass as the other images of its folder can serve it, "
                    f"{count_min} at a time, without one of them serving twice (the smallest: "
                    f"{described(smallest)}, gives {per_pass[smallest]} per pass)"
                )
            if len(reference_only_paths) > 0:
                more = sum(1 for folder in paired if len(folder_pools[folder]) > per_pass[folder] * passes)
                print_acc(
                    f"  -  {len(reference_only_paths)} more image(s) serve as references only "
                    f"({more} folder(s) hold more than the {per_folder * passes} that are used)"
                )
        elif len(reference_only_paths) > 0:
            print_acc(
                f"  -  {len(reference_only_paths)} more image(s) serve as references only "
                f"(control_from_same_folder_max_targets: {max_targets} per folder)"
            )
        if len(folder_repeats) > 0:
            # by chance now and then; a folder where it is most of them holds too few images
            # for the number of times its targets come up
            most = min(folder_repeats, key=lambda folder: (
                -folder_repeats[folder] / folder_targets_made[folder], -folder_repeats[folder], seed_name(folder)))
            flipped = " flipped the same way" if dataset_config.flip_x or dataset_config.flip_y else ""
            print_acc(
                f"  -  {sum(folder_repeats.values())} of the {len(self.file_list)} targets have the same "
                f"references as another copy of the same image{flipped} (most in {seed_name(most)}, "
                f"{len(folder_pools[most])} images: {folder_repeats[most]} of its {folder_targets_made[most]})"
            )
        if len(alone_paths) > 0:
            print_acc(
                f"  -  Left out {len(alone_paths)} image(s) with no other image of their folder "
                f"in this dataset to pair with"
            )
        max_pixels = dataset_config.control_from_same_folder_max_pixels
        if max_pixels is not None:
            match_target_res = bool(self.sd.model_config.model_kwargs.get('match_target_res', False))
            model_limit = getattr(self.sd, '_ref_target_pixels', lambda target_pixels: None)(None)
            further = ""
            if not match_target_res and model_limit is not None and model_limit < max_pixels:
                further = f"; the model then reduces them to its own limit of {model_limit}"
            print_acc(
                f"  -  References with more than {max_pixels} pixels are scaled down to that "
                f"when they are loaded{further}"
            )
        if len(self.file_list) == 0:
            if self._folder_pair_before_low_res_filter:
                # the low-res filter took the pairs apart; the other resolutions of a
                # multi-resolution dataset can still have them
                raise DatasetBelowResolutionError(
                    f"control_from_same_folder: no folder under {self.dataset_path} has two images "
                    f"that are large enough for resolution {self.resolution}"
                )
            raise ValueError(
                f"control_from_same_folder: no folder under {self.dataset_path} holds two or more "
                f"usable images. Point folder_path at the parent of the per-character folders."
            )

    def setup_epoch(self):
        if self.epoch_num == 0:
            # initial setup
            # do not call for now
            if self.dataset_config.buckets:
                # setup buckets
                self.setup_buckets()
            if self.dataset_config.clip_image_from_same_folder:
                self.prepare_same_folder_clip_image_pairs()
            if self.dataset_config.caches_only and self.is_caching_latents_to_disk:
                # must run after setup_buckets (cache keys depend on bucket dims)
                self._drop_uncached_missing_items()
            if self.is_caching_latents:
                if self.dataset_config.lazy_cache_latents:
                    print_acc("Lazy latent caching enabled; latents will be cached on first use")
                else:
                    self.cache_latents_all_latents()
            if self.is_caching_clip_vision_to_disk:
                self.cache_clip_vision_to_disk()
            if self.is_caching_text_embeddings:
                if self.dataset_config.lazy_cache_text_embeddings:
                    print_acc("Lazy text-embedding caching enabled; embeddings will be cached on first use")
                else:
                    self.cache_text_embeddings()
            if self.is_generating_controls:
                # always do this last
                self.setup_controls()
        else:
            if self.dataset_config.poi is not None:
                # handle cropping to a specific point of interest
                # setup buckets every epoch
                self.setup_buckets(quiet=True)
            if self.dataset_config.clip_image_from_same_folder:
                self.prepare_same_folder_clip_image_pairs()
        self.epoch_num += 1

    def _build_resume_file_signature(self):
        digest = hashlib.md5()
        for file_item in self.file_list:
            digest.update(file_item.path.encode("utf-8"))
            digest.update(b"|")
            digest.update(b"1" if getattr(file_item, "flip_x", False) else b"0")
            digest.update(b"|")
            digest.update(b"1" if getattr(file_item, "flip_y", False) else b"0")
            digest.update(b"\n")
        return digest.hexdigest()

    def get_resume_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "dataset_path": self.dataset_path,
            "epoch_num": self.epoch_num,
            "file_count": len(self.file_list),
            "file_signature": self._build_resume_file_signature(),
        }
        if self.dataset_config.buckets:
            state["batch_indices"] = [list(batch) for batch in getattr(self, "batch_indices", [])]
        if self.dataset_config.clip_image_from_same_folder:
            state["clip_image_epoch_paths"] = [
                getattr(file_item, "clip_image_epoch_path", None)
                for file_item in self.file_list
            ]
        return state

    def apply_resume_state(self, state: Optional[Dict[str, Any]]) -> bool:
        if not state:
            return False
        if state.get("file_count") != len(self.file_list):
            return False
        if state.get("file_signature") != self._build_resume_file_signature():
            return False

        if self.dataset_config.buckets and state.get("batch_indices") is not None:
            self.batch_indices = [list(batch) for batch in state["batch_indices"]]

        if self.dataset_config.clip_image_from_same_folder:
            clip_image_epoch_paths = state.get("clip_image_epoch_paths")
            if isinstance(clip_image_epoch_paths, list) and len(clip_image_epoch_paths) == len(self.file_list):
                for file_item, clip_image_path in zip(self.file_list, clip_image_epoch_paths):
                    file_item.clip_image_epoch_path = clip_image_path

        self.epoch_num = int(state.get("epoch_num", self.epoch_num))
        return True

    def __len__(self):
        if self.dataset_config.buckets:
            return len(self.batch_indices)
        return len(self.file_list)

    def _get_single_item(self, index) -> 'FileItemDTO':
        file_item: 'FileItemDTO' = copy.deepcopy(self.file_list[index])
        if self.dataset_config.lazy_cache_latents and self.is_caching_latents_to_disk:
            file_item.is_caching_to_disk = True
            file_item.is_caching_to_memory = False
            file_item.latent_load_device = self.sd_device
            if file_item.get_latent_storage_entry() is not None:
                file_item.is_latent_cached = True
        file_item.load_caption(self.caption_dict)
        if self.dataset_config.lazy_cache_text_embeddings and self.is_caching_text_embeddings:
            if file_item.get_text_embedding_storage_entry() is not None:
                file_item.is_text_embedding_cached = True
        file_item.load_and_process_image(self.transform)
        return file_item

    def __getitem__(self, item):
        if self.dataset_config.buckets:
            # for buckets we collate ourselves for now
            # todo allow a scheduler to dynamically make buckets
            # we collate ourselves
            if len(self.batch_indices) - 1 < item:
                # tried everything to solve this. No way to reset length when redoing things. Pick another index
                item = random.randint(0, len(self.batch_indices) - 1)
            idx_list = self.batch_indices[item]
            return [self._get_single_item(idx) for idx in idx_list]
        else:
            # Dataloader is batching
            return self._get_single_item(item)


# ---------------------------------------------------------------------------
# Global (cross-dataset) bucket batching -- opt-in via train.global_bucket_batching
# ---------------------------------------------------------------------------
# Implements the upstream TODO that used to sit below ConcatDataset(datasets):
#   "todo build scheduler that can get buckets from all datasets that match"
# Batches are cut from ONE global pool keyed by a compatibility tuple, so items
# from different datasets can share a batch whenever their tensors can stack.
GLOBAL_BUCKET_BATCHING_VERSION = 1
# Fixed seed: composition must be reproducible across runs and across a LOST
# resume state, so that a crash costs only the batch ORDER, never the contents.
# Uses its own Random instance so enabling the flag does not perturb the global
# RNG stream.
GLOBAL_BUCKET_BATCHING_SEED = 0x41544B47


def _global_bucket_control_arity(file_item: 'FileItemDTO') -> int:
    # control_path is resolved PER FILE ITEM (dataloader_mixins.py ControlFileItemDTOMixin
    # globs for a matching filename; non-matching files get None) and mutated again by
    # add_control_path_to_file_item during setup_controls. Never read it off dataset_config.
    if getattr(file_item, 'same_folder_control_pool', None) is not None:
        # FORK: control_from_same_folder. These always carry their references as a list, any
        # number of them per item, so the count does not have to match within a batch. 0 keeps
        # them apart from the one-image control_path items (1), whose single stacked tensor
        # cannot share a batch with a list.
        return 0
    control_path = getattr(file_item, 'control_path', None)
    if control_path is None:
        return 0
    if isinstance(control_path, list):
        return len(control_path)
    return 1


def global_bucket_compat_key(file_item: 'FileItemDTO') -> tuple:
    """Two FileItemDTOs may share one DataLoaderBatchDTO iff this key is equal.

    Tier A entries are hard tensor-shape constraints enforced by the torch.cat /
    torch.stack calls in DataLoaderBatchDTO.__init__.
    Tier B entries are semantic: the shapes match, but the trainer reads these
    batch-wide off file_items[0] (or collapses them with any()), so mixing them
    would train the wrong thing without raising.

    Must be evaluated AFTER every dataset's setup_epoch() has completed, because
    setup_controls() mutates control_path / has_mask_image / has_inpaint_image in
    place. get_dataloader_from_datasets() satisfies that: AiToolkitDataset.__init__
    calls setup_epoch() as its last statement.
    """
    dc = file_item.dataset_config
    return (
        # ---- Tier A: hard tensor-shape constraints ----
        int(file_item.crop_width),
        int(file_item.crop_height),
        int(dc.num_frames),                 # image [C,H,W] vs video [F,C,H,W]
        int(dc.fps),                        # audio length = num_frames/fps * sample_rate
        len(getattr(file_item, 'extra_values', None) or []),
        bool(file_item.has_control_image),
        _global_bucket_control_arity(file_item),
        bool(getattr(file_item, 'full_size_control_images', False)),
        bool(getattr(file_item, 'use_raw_control_images', False)),
        bool(file_item.has_inpaint_image),
        bool(file_item.has_mask_image),
        bool(file_item.has_clip_image),
        bool(file_item.has_augmentations),
        bool(file_item.has_unconditional),
        bool(dc.do_audio),
        str(getattr(file_item, 'latent_space_version', '')),
        str(getattr(file_item, 'text_embedding_space_version', '')),
        str(getattr(file_item, 'te_padding_side', '')),
        # ---- Tier B: same shape, wrong training ----
        bool(dc.standardize_images),        # selects a different self.transform
        bool(dc.do_i2v),
        bool(dc.is_reg),                    # is_reg = any(batch.get_is_reg_list())
        bool(dc.prior_reg),                 # any([... prior_reg ...])
        str(dc.guidance_type),              # read off file_items[0]
        bool(dc.cache_latents),
        bool(dc.cache_latents_to_disk),     # mixing cached+uncached forces permanent
        bool(dc.lazy_cache_latents),        #   pixel decode via latent_cache_mask
        int(dc.latent_cache_shard_size),
        bool(dc.cache_text_embeddings),
        bool(dc.lazy_cache_text_embeddings),
        int(dc.text_embedding_cache_shard_size),
        float(dc.mask_min_value),
        bool(dc.invert_mask),
    )


class GlobalBucketDataset(Dataset):
    """Owns the AiToolkitDatasets in place of ConcatDataset and cuts batches from one
    global compatibility-keyed pool, so a batch may hold items from several datasets.

    One index == one batch, the same contract AiToolkitDataset.__getitem__ has when
    buckets are on, so DataLoader(batch_size=None) + dto_collation are unchanged.

    Composition is built ONCE in __init__ and never rebuilt. That is deliberate:
      * persistent_workers=True pickles the dataset to workers exactly once, so a
        main-process rebuild would be invisible to them and the saved resume state
        would describe batches the workers never delivered;
      * it keeps len() invariant, avoiding the start_next_dataloader_epoch() ->
        iter() -> trigger_dataloader_setup_epoch() ordering hazard;
      * it matches today's behaviour, since setup_buckets() already early-returns
        after epoch 0 and per-epoch variation comes only from ResumableShuffleSampler.

    No __getstate__ is needed: this object holds only the sub-datasets (each with its
    own __getstate__ nulling sd and the progress hooks), ints and lists of ints.
    """

    def __init__(self, datasets: List['AiToolkitDataset'], batch_size: int = 1):
        # NAME MATTERS: get_dataloader_datasets() finds sub-datasets via `.datasets`.
        self.datasets: List['AiToolkitDataset'] = datasets
        self.batch_size = max(1, int(batch_size))
        self.len = None  # trigger_dataloader_setup_epoch assigns this; __len__ ignores it
        self.batches: List[List[List[int]]] = []
        self.pooled_dataset_count = 0
        self.build_global_batches()

    @staticmethod
    def _is_poolable(dataset: 'AiToolkitDataset') -> bool:
        # Datasets that are not poolable keep their own per-dataset batch_indices,
        # appended verbatim as passthrough batches. Their composition is already
        # frozen after epoch 0 today, so this is byte-equivalent for them.
        dc = dataset.dataset_config
        if getattr(dc, 'do_audio', False):
            # audio_tensor length depends on the SOURCE file's sample rate, which is
            # unknowable at bucket time. Do not widen that exposure across folders.
            return False
        if dataset._use_balanced_same_folder_clip_sampling():
            # global batching would discard _build_balanced_folder_order's round-robin
            return False
        return True

    def build_global_batches(self):
        rng = random.Random(GLOBAL_BUCKET_BATCHING_SEED)
        groups: Dict[tuple, List[List[int]]] = {}
        batches: List[List[List[int]]] = []
        pooled = 0
        for ds_idx, dataset in enumerate(self.datasets):
            if not self._is_poolable(dataset):
                for batch in getattr(dataset, 'batch_indices', []):
                    batches.append([[ds_idx, int(i)] for i in batch])
                continue
            pooled += 1
            for item_idx in dataset.active_file_indices:
                key = global_bucket_compat_key(dataset.file_list[item_idx])
                groups.setdefault(key, []).append([ds_idx, int(item_idx)])
        for key in sorted(groups.keys(), key=repr):  # deterministic group order
            members = groups[key]
            rng.shuffle(members)
            for start in range(0, len(members), self.batch_size):
                batches.append(members[start:start + self.batch_size])
        rng.shuffle(batches)
        self.batches = batches
        self.pooled_dataset_count = pooled

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        if len(self.batches) - 1 < item:
            # parity with AiToolkitDataset.__getitem__'s out-of-range band-aid
            item = random.randint(0, len(self.batches) - 1)
        # Delegate: _get_single_item applies THAT dataset's transform, caption_dict,
        # sd_device and lazy-cache flags. A pooled FileItemDTO cannot supply those itself.
        return [
            self.datasets[ds_idx]._get_single_item(item_idx)
            for ds_idx, item_idx in self.batches[item]
        ]

    def setup_epoch(self):
        # trigger_dataloader_setup_epoch hits the `setup_epoch` branch FIRST and never
        # reaches the `.datasets` branch, so fan out here or per-epoch poi re-bucketing
        # and clip pairing would silently stop running.
        for dataset in self.datasets:
            dataset.setup_epoch()
            dataset.len = None
        # Deliberately does NOT rebuild self.batches -- see the class docstring.

    def build_signature(self) -> str:
        digest = hashlib.md5()
        digest.update(str(GLOBAL_BUCKET_BATCHING_VERSION).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(self.batch_size).encode("utf-8"))
        for dataset in self.datasets:
            digest.update(b"|")
            digest.update(dataset._build_resume_file_signature().encode("utf-8"))
        return digest.hexdigest()

    def get_resume_state(self) -> Dict[str, Any]:
        return {
            "version": GLOBAL_BUCKET_BATCHING_VERSION,
            "batch_size": self.batch_size,
            "signature": self.build_signature(),
            "batches": [[list(pair) for pair in batch] for batch in self.batches],
        }

    def apply_resume_state(self, state: Optional[Dict[str, Any]]) -> bool:
        # FAIL CLOSED. Applying a saved sampler permutation of batch INDICES to a
        # differently-composed batch list raises nothing and trains the wrong items.
        if not state:
            return False
        if state.get("version") != GLOBAL_BUCKET_BATCHING_VERSION:
            return False
        if state.get("batch_size") != self.batch_size:
            return False
        if state.get("signature") != self.build_signature():
            return False
        saved = state.get("batches")
        if not isinstance(saved, list) or len(saved) != len(self.batches):
            return False
        rebuilt: List[List[List[int]]] = []
        for batch in saved:
            if not isinstance(batch, list) or len(batch) == 0:
                return False
            pairs: List[List[int]] = []
            for pair in batch:
                try:
                    ds_idx, item_idx = int(pair[0]), int(pair[1])
                except (TypeError, ValueError, IndexError):
                    return False
                if ds_idx < 0 or ds_idx >= len(self.datasets):
                    return False
                if item_idx < 0 or item_idx >= len(self.datasets[ds_idx].file_list):
                    return False
                pairs.append([ds_idx, item_idx])
            rebuilt.append(pairs)
        self.batches = rebuilt
        return True


def get_dataloader_from_datasets(
        dataset_options,
        batch_size=1,
        sd: 'StableDiffusion' = None,
        cache_progress_hook: Optional[Callable[[str, int, int, float, str], None]] = None,
        global_bucket_batching: bool = False,
) -> DataLoader:
    if dataset_options is None or len(dataset_options) == 0:
        return None

    datasets = []
    has_buckets = False
    is_caching_latents = False

    dataset_config_list = []
    # preprocess them all
    for dataset_option in dataset_options:
        if isinstance(dataset_option, DatasetConfig):
            dataset_config_list.append(dataset_option)
        else:
            # preprocess raw data
            split_configs = preprocess_dataset_raw_config([dataset_option])
            for x in split_configs:
                dataset_config_list.append(DatasetConfig(**x))

    for config in dataset_config_list:

        if config.type == 'image':
            try:
                dataset = AiToolkitDataset(
                    config,
                    batch_size=batch_size,
                    sd=sd,
                    cache_progress_hook=cache_progress_hook,
                )
            except DatasetBelowResolutionError as e:
                # All images in this dataset are below its selected resolution; skip
                # just this dataset/resolution. With multi-resolution (split per-res),
                # the same images may still train at a lower selected resolution.
                print_acc(f"  -  Skipping dataset at resolution {config.resolution}: {e}")
                continue
            datasets.append(dataset)
            if config.buckets:
                has_buckets = True
            if config.cache_latents or config.cache_latents_to_disk:
                is_caching_latents = True
        else:
            raise ValueError(f"invalid dataset type: {config.type}")

    if len(datasets) == 0:
        raise ValueError(
            "No usable datasets remain — every dataset/resolution had all images below its selected "
            "resolution. Add lower resolutions to your datasets or use larger images."
        )

    # Cross-dataset bucket batching (opt-in). Implements the long-standing TODO:
    # "build scheduler that can get buckets from all datasets that match".
    # Reg images are NOT handled here -- they go through data_loader_reg and are
    # alternated by the batch_step % 2 logic in BaseSDTrainProcess.
    use_global_buckets = False
    if global_bucket_batching:
        if not has_buckets:
            print_acc("Global bucket batching: bucketing is off; using per-dataset batches.")
        elif len(datasets) < 2:
            print_acc("Global bucket batching: only one dataset; using per-dataset batches.")
        elif any(d.dataset_config.poi is not None for d in datasets):
            print_acc(
                "Global bucket batching: point-of-interest cropping re-buckets and changes "
                "crop dimensions every epoch, which a fixed global batch map cannot follow; "
                "using per-dataset batches."
            )
        else:
            use_global_buckets = True

    if use_global_buckets:
        per_dataset_batches = sum(len(getattr(d, 'batch_indices', [])) for d in datasets)
        concatenated_dataset = GlobalBucketDataset(datasets, batch_size=batch_size)
        print_acc(
            f"Global bucket batching: {len(concatenated_dataset)} batches/epoch "
            f"(was {per_dataset_batches} with per-dataset batching); "
            f"{concatenated_dataset.pooled_dataset_count}/{len(datasets)} datasets pooled. "
            f"Note: with Train by Epoch on, fewer batches/epoch means fewer resolved steps."
        )
    else:
        concatenated_dataset = ConcatDataset(datasets)
    sampler = ResumableShuffleSampler(concatenated_dataset)

    # check if is caching latents

    dataloader_kwargs = {}
    has_windows_lazy_disk_latent_cache = any(
        config.lazy_cache_latents and config.cache_latents_to_disk
        for config in dataset_config_list
    )
    requested_num_workers = dataset_config_list[0].num_workers
    if is_native_windows() and not dataset_config_list[0].enable_windows_num_workers:
        requested_num_workers = 0

    if is_native_windows() and has_windows_lazy_disk_latent_cache:
        print_acc(
            "Windows lazy disk latent caching detected; using SQLite-backed "
            "index lookups with immediate SQLite pending latent storage for worker visibility."
        )
        dataloader_kwargs['num_workers'] = dataset_config_list[0].num_workers
        if dataloader_kwargs['num_workers'] > 0:
            dataloader_kwargs['prefetch_factor'] = dataset_config_list[0].prefetch_factor
            persistent_workers = dataset_config_list[0].persistent_workers
            if persistent_workers is None:
                persistent_workers = True
            dataloader_kwargs['persistent_workers'] = persistent_workers
    elif is_native_windows() and not dataset_config_list[0].enable_windows_num_workers:
        dataloader_kwargs['num_workers'] = 0
    else:
        dataloader_kwargs['num_workers'] = dataset_config_list[0].num_workers
        if dataloader_kwargs['num_workers'] > 0:
            dataloader_kwargs['prefetch_factor'] = dataset_config_list[0].prefetch_factor
            persistent_workers = dataset_config_list[0].persistent_workers
            if persistent_workers is None:
                persistent_workers = True
            dataloader_kwargs['persistent_workers'] = persistent_workers

    pin_memory = dataset_config_list[0].pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if (
            pin_memory
            and is_native_windows()
            and has_windows_lazy_disk_latent_cache
            and dataloader_kwargs.get('num_workers', 0) > 0
    ):
        print_acc(
            "Disabling pin_memory on native Windows for lazy disk latent caching with worker processes "
            "to avoid recurrent DataLoader access-violation crashes."
        )
        pin_memory = False
    dataloader_kwargs['pin_memory'] = pin_memory

    if has_buckets:
        # make sure they all have buckets
        for dataset in datasets:
            assert dataset.dataset_config.buckets, f"buckets not found on dataset {dataset.dataset_config.folder_path}, you either need all buckets or none"

        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=None,  # we batch in the datasets for now
            drop_last=False,
            sampler=sampler,
            collate_fn=dto_collation,  # Use the custom collate function
            **dataloader_kwargs
        )
    else:
        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=dto_collation,
            **dataloader_kwargs
        )
    return data_loader


def trigger_dataloader_setup_epoch(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    dataloader.len = None
    if isinstance(dataloader.dataset, list):
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    if hasattr(sub_dataset, 'setup_epoch'):
                        sub_dataset.setup_epoch()
                        sub_dataset.len = None
            elif hasattr(dataset, 'setup_epoch'):
                dataset.setup_epoch()
                dataset.len = None
    elif hasattr(dataloader.dataset, 'setup_epoch'):
        dataloader.dataset.setup_epoch()
        dataloader.dataset.len = None
    elif hasattr(dataloader.dataset, 'datasets'):
        dataloader.dataset.len = None
        for sub_dataset in dataloader.dataset.datasets:
            if hasattr(sub_dataset, 'setup_epoch'):
                sub_dataset.setup_epoch()
                sub_dataset.len = None

def get_dataloader_datasets(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    if isinstance(dataloader.dataset, list):
        datasets = []
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    datasets.append(sub_dataset)
            else:
                datasets.append(dataset)
        return datasets
    elif hasattr(dataloader.dataset, 'datasets'):
        return dataloader.dataset.datasets
    else:
        return [dataloader.dataset]


def _get_resumable_sampler(dataloader: Optional[DataLoader]) -> Optional[ResumableShuffleSampler]:
    if dataloader is None:
        return None
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, ResumableShuffleSampler):
        return sampler
    return None


def mark_dataloader_batch_processed(dataloader: Optional[DataLoader], count: int = 1):
    sampler = _get_resumable_sampler(dataloader)
    if sampler is not None:
        sampler.mark_processed(count)


def start_next_dataloader_epoch(dataloader: Optional[DataLoader]):
    sampler = _get_resumable_sampler(dataloader)
    if sampler is not None:
        sampler.start_new_epoch()


def get_dataloader_resume_state(dataloader: Optional[DataLoader]) -> Optional[Dict[str, Any]]:
    if dataloader is None:
        return None
    sampler = _get_resumable_sampler(dataloader)
    datasets = get_dataloader_datasets(dataloader)
    resume_state = {
        "sampler": sampler.get_state() if sampler is not None else None,
        "datasets": [
            dataset.get_resume_state()
            for dataset in datasets
            if hasattr(dataset, "get_resume_state")
        ],
    }
    root_dataset = getattr(dataloader, "dataset", None)
    if isinstance(root_dataset, GlobalBucketDataset):
        # Batch composition is owned by the wrapper, not by any single dataset.
        # Without it the saved sampler permutation is meaningless.
        resume_state["global_buckets"] = root_dataset.get_resume_state()
    return resume_state


def apply_dataloader_resume_state(dataloader: Optional[DataLoader], state: Optional[Dict[str, Any]]) -> bool:
    if dataloader is None or not state:
        return False

    try:
        datasets = get_dataloader_datasets(dataloader)
        dataset_states = state.get("datasets", [])
        if len(dataset_states) != len(datasets):
            return False

        for dataset, dataset_state in zip(datasets, dataset_states):
            if hasattr(dataset, "apply_resume_state"):
                if not dataset.apply_resume_state(dataset_state):
                    return False

        # Restore global batch composition BEFORE the sampler: load_state validates
        # data_length against len(data_source), which is len(wrapper).
        root_dataset = getattr(dataloader, "dataset", None)
        if isinstance(root_dataset, GlobalBucketDataset):
            if not root_dataset.apply_resume_state(state.get("global_buckets")):
                # Includes the flag-was-off-when-saved case (no "global_buckets" key).
                return False
        elif state.get("global_buckets") is not None:
            # Saved with global bucket batching ON, now OFF. The sampler order refers
            # to batches that no longer exist; a half-restore would silently train the
            # wrong items. Start the epoch fresh instead.
            return False

        sampler_state = state.get("sampler")
        sampler = _get_resumable_sampler(dataloader)
        if sampler is not None and sampler_state is not None:
            sampler.load_state(sampler_state)
    except Exception:
        return False

    return True
