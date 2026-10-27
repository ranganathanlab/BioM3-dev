"""
I/O module for Stage 1 PenCL

"""

import torch
import torch.nn as nn

import biom3.Stage1.model as mod
from biom3.backend.device import setup_logger
from biom3.core.io import load_state_dict_unwrap_pl

logger = setup_logger(__name__)


def load_pencl_weights(model: nn.Module, weights_path: str, device=None) -> None:
    """Load PenCL weights into ``model``, raising if any parameter stays unloaded.

    The format is read from the file, not its name: a Lightning checkpoint is
    unwrapped and its ``model.`` prefix stripped whatever the extension. The
    load itself is non-strict, but a bare non-strict load also hides the case
    where no key matches and the model keeps its initial weights, so the result
    is checked: every parameter must come from the file. Buffers the file lacks
    and keys the model does not use are reported, not fatal.
    """
    state_dict = load_state_dict_unwrap_pl(weights_path, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)

    # A tied parameter has several names; any one of them loads it.
    names_by_param = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        names_by_param.setdefault(id(param), []).append(name)
    unloaded_params = [
        names[0] for names in names_by_param.values()
        if all(name in missing for name in names)
    ]
    unloaded_buffers = [
        name for name, _ in model.named_buffers() if name in missing
    ]
    n_params = len(names_by_param)

    if unloaded_params:
        raise RuntimeError(
            f"PenCL weights at {weights_path} did not populate "
            f"{len(unloaded_params)}/{n_params} parameters "
            f"(e.g. {unloaded_params[:5]}). Embeddings from this model would "
            f"come from untrained weights. File keys look like "
            f"{sorted(state_dict)[:3]}."
        )
    if unloaded_buffers:
        logger.warning(
            "PenCL: %d buffer(s) not present in %s: %s",
            len(unloaded_buffers), weights_path, unloaded_buffers[:5],
        )
    if unexpected:
        logger.warning(
            "PenCL: %d key(s) in %s are not used by the model: %s",
            len(unexpected), weights_path, unexpected[:5],
        )
    logger.info(
        "PenCL: loaded %d/%d parameters from %s",
        n_params, n_params, weights_path,
    )


def prepare_model_pfam_PenCL(
        config_args, 
        state_dict_fpath=None, 
        strict=True,
        device=None,
        eval=False,
) -> nn.Module:
    """TODO: implement"""
    # # TODO: Need to test loading using strict=False. Possible that weights 
    # # are not being properly loaded.
    # model = mod.pfam_PEN_CL(args=config_args)
    # if state_dict_fpath:
    #     model.load_state_dict(
    #         torch.load(state_dict_fpath, map_location=device), 
    #         strict=strict,
    #     )
    # if device:
    #     model.to(device)
    # if eval:
    #     model.eval()
    # print("Model loaded successfully with weights!")
    # return model

    raise NotImplementedError()