from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from kvcache.config import GPT2Config
from kvcache.model import GPT

_PREFIX = "transformer."
_MASK_BUFFER_SUFFIXES = (".attn.bias", ".attn.masked_bias")
_CONV1D_WEIGHT_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def download_gpt2_checkpoint(repo_id: str = "gpt2", revision: str = "main") -> Path:
    """Download model.safetensors into the standard Hugging Face cache and return its path."""
    return Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision))


def convert_hf_gpt2_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a Hugging Face GPT-2 state dict onto the parameter names of `GPT`.

    HF stores the attention and MLP projections as Conv1D, with weights laid out
    (in, out); nn.Linear expects (out, in), so those four weights are transposed.
    """
    converted: dict[str, Tensor] = {}
    for name, tensor in sd.items():
        name = name.removeprefix(_PREFIX)
        if name.endswith(_MASK_BUFFER_SUFFIXES) or name == "lm_head.weight":
            continue
        if name.endswith(_CONV1D_WEIGHT_SUFFIXES) and tensor.dim() == 2:
            tensor = tensor.t()
        converted[name] = tensor
    return converted


def load_gpt2_weights(model: GPT, sd_converted: dict[str, Tensor]) -> None:
    """Copy converted weights into model, failing on any missing, unexpected or misshapen key.

    A parameter left at its random initialization raises no error on its own, it only
    degrades the output, so every parameter must be matched exactly.
    """
    params = dict(model.named_parameters())
    missing = sorted(params.keys() - sd_converted.keys())
    unexpected = sorted(sd_converted.keys() - params.keys())
    if missing or unexpected:
        raise ValueError(
            f"state dict does not match the model: missing keys {missing}, "
            f"unexpected keys {unexpected}"
        )

    for name, param in params.items():
        tensor = sd_converted[name]
        if tensor.shape != param.shape:
            raise ValueError(
                f"shape mismatch for {name}: expected {tuple(param.shape)}, "
                f"found {tuple(tensor.shape)}"
            )

    with torch.no_grad():
        for name, param in params.items():
            param.copy_(sd_converted[name].to(param.dtype))


def load_pretrained_gpt2(
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    revision: str = "main",
) -> GPT:
    """Build GPT-2 small with its pretrained weights, in eval mode."""
    sd = load_file(download_gpt2_checkpoint(revision=revision))
    with torch.device(device):
        model = GPT(GPT2Config()).to(dtype)
    load_gpt2_weights(model, convert_hf_gpt2_state_dict(sd))
    return model.eval()
