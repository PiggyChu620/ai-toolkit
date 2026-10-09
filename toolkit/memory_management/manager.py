import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager, OstrisLinearLayerMemoryManager
import random

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
    # custom-quantized layers (convrot8, nvfp4, ...: toolkit/util/ostris_quant.py). Without this
    # they were neither offloaded nor parked, e.g. a convrot8 text encoder stayed whole on the GPU
    "OstrisLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
        keep_unmanaged_on_device: bool = False,
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.keep_unmanaged_on_device: bool = keep_unmanaged_on_device
        self.unmanaged_modules: list[torch.nn.Module] = []
        self.unmanaged_modules_ready_on_process_device: bool = False

    @staticmethod
    def _extract_to_spec(*args, **kwargs):
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        non_blocking = kwargs.get("non_blocking", False)
        copy = kwargs.get("copy", False)
        memory_format = kwargs.get("memory_format", None)

        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                device = arg.device
                dtype = arg.dtype

        return device, dtype, non_blocking, copy, memory_format

    @classmethod
    def _fallback_move_module(cls, module: torch.nn.Module, *args, **kwargs):
        device, dtype, non_blocking, copy, memory_format = cls._extract_to_spec(
            *args, **kwargs
        )

        def move_tensor(t: torch.Tensor):
            move_kwargs = {}
            if device is not None:
                move_kwargs["device"] = device
            if dtype is not None and (t.is_floating_point() or t.is_complex()):
                move_kwargs["dtype"] = dtype
            if non_blocking:
                move_kwargs["non_blocking"] = non_blocking
            if copy:
                move_kwargs["copy"] = copy
            if memory_format is not None:
                move_kwargs["memory_format"] = memory_format
            if not move_kwargs:
                return t
            return t.to(**move_kwargs)

        module._apply(move_tensor)
        return module

    @staticmethod
    def _is_same_device(a, b) -> bool:
        """torch.device('cuda') != torch.device('cuda:0'), but they name one device.

        Comparing them raw makes the "already staged on the compute device" fast path
        miss whenever the caller and the manager spell the device differently, which
        silently degrades every .to(compute_device) into a full re-walk.
        """
        if a is None or b is None:
            return False
        a = torch.device(a)
        b = torch.device(b)
        if a.type != b.type:
            return False
        if a.index is None or b.index is None:
            return True
        return a.index == b.index

    @staticmethod
    def _tensor_matches_to_spec(
        tensor: torch.Tensor,
        device,
        dtype,
    ) -> bool:
        if device is not None and tensor.device != torch.device(device):
            return False
        if (
            dtype is not None
            and (tensor.is_floating_point() or tensor.is_complex())
            and tensor.dtype != dtype
        ):
            return False
        return True

    @classmethod
    def _module_matches_to_spec(
        cls,
        module: torch.nn.Module,
        device,
        dtype,
    ) -> bool:
        tensors = list(module.parameters(recurse=False))
        tensors.extend(list(module.buffers(recurse=False)))
        if len(tensors) == 0:
            return True
        return all(cls._tensor_matches_to_spec(tensor, device, dtype) for tensor in tensors)

    def memory_managed_to(self, *args, **kwargs):
        target_device, target_dtype, _, _, _ = self._extract_to_spec(*args, **kwargs)
        leaving_process_device = target_device is not None and not self._is_same_device(
            target_device, self.process_device
        )
        if (
            self.keep_unmanaged_on_device
            and self.unmanaged_modules_ready_on_process_device
            and target_device is not None
            and target_dtype is None
            and not leaving_process_device
        ):
            return self.module

        # keep_unmanaged_on_device used to VETO the move whenever the target was not the
        # compute device (should_move_unmanaged = False), which turned every .to("cpu")
        # into a silent no-op: text_encoder_to("cpu") moved nothing, so Qwen3-VL's
        # unquantized ~151936x2560 embedding table (~0.78 GB bf16) stayed resident in
        # VRAM for the whole run while the log claimed the encoder was parked on CPU.
        # The flag exists to skip redundant re-moves *onto* the compute device, which the
        # fast path above still does; parking off it must be honoured for real.
        should_move_unmanaged = True
        if leaving_process_device:
            # Load-bearing: the block below only ever sets this latch True. Without the
            # reset, the next .to(compute_device) would take the early return above and
            # leave the unmanaged modules stranded on CPU -> device-mismatch crash.
            self.unmanaged_modules_ready_on_process_device = False

        # first move all the unmanaged modules
        if should_move_unmanaged:
            for module in self.unmanaged_modules:
                if isinstance(module, torch.nn.Parameter):
                    if self._tensor_matches_to_spec(module.data, target_device, target_dtype):
                        continue
                    # Parameter cannot move this way
                    module.data = module.data.to(*args, **kwargs)
                else:
                    if self._module_matches_to_spec(module, target_device, target_dtype):
                        continue
                    try:
                        module.to(*args, **kwargs)
                    except TypeError as e:
                        try:
                            self._fallback_move_module(module, *args, **kwargs)
                        except Exception:
                            raise e
            if (
                self.keep_unmanaged_on_device
                and target_device is not None
                and target_dtype is None
                and self._is_same_device(target_device, self.process_device)
            ):
                self.unmanaged_modules_ready_on_process_device = True
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls, 
        module: torch.nn.Module, 
        device: torch.device, 
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = [],
        keep_unmanaged_on_device: bool = False,
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(
            module,
            device,
            keep_unmanaged_on_device=keep_unmanaged_on_device,
        )

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)
            
        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear; OstrisLinear bounces its quantized buffers instead of a
                        # dequantized weight (its .weight is a property that dequantizes)
                        if getattr(child_module, "is_ostris_quantized", False):
                            OstrisLinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        else:
                            LinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara, 
                                    device,
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara, 
                                    device,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue

        if keep_unmanaged_on_device:
            for unmanaged_module in module._memory_manager.unmanaged_modules:
                if isinstance(unmanaged_module, torch.nn.Parameter):
                    unmanaged_module.data = unmanaged_module.data.to(device=device)
                else:
                    cls._fallback_move_module(unmanaged_module, device=device)
            module._memory_manager.unmanaged_modules_ready_on_process_device = True
