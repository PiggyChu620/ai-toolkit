import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def to(self, *args, **kwargs):
        return self


def _release_text_encoder(module):
    """Free a text encoder's weights even while something else (e.g. a prompt encoder) still holds
    the module: its tensors become meta. If a tensor type cannot, the module is only dropped (never
    copied to the CPU, which would spike RAM), and is freed once nothing references it."""
    if module is None or isinstance(module, FakeTextEncoder):
        return
    try:
        torch.nn.Module.to(module, "meta")
    except Exception:
        pass


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                pipe.text_encoder.to('cpu')
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
            # text encoders the pipeline does not expose (e.g. Qwen-Image 2.1, whose prompt
            # encoder holds its own): model.text_encoder is the source of truth, so free those too
            # instead of leaving several GB on the GPU for the whole run
            if len(text_encoder_list) < len(model.text_encoder):
                seen = set()
                for te in model.text_encoder[len(text_encoder_list):]:
                    if id(te) not in seen:
                        seen.add(id(te))
                        _release_text_encoder(te)
                    text_encoder_list.append(FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype))
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)

    flush()
