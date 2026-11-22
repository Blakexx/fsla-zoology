"""FSLA sequence mixer for zoology: a thin adapter over `fla.layers.FSLA`, as `gla.py` adapts fla's GLA.

It adds zoology's mixer convention and nothing else: the model builds ``mixer(d_model=..., layer_idx=..., **kwargs)``
from a serializable kwargs dict and calls ``mixer(x)`` expecting a tensor. ``levels`` and ``decay`` are fsla's own
objects as JSON (``fla.layers.fsla.encode``), and every other argument keeps fsla's name; everything is validated and
refused by fsla, never here.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from fla.layers.fsla import FSLA


class FSLAMixer(nn.Module):
    """zoology's mixer over fla's FSLA layer: ``d_model`` is the layer's ``hidden_size`` and ``n_heads`` its
    ``num_heads``, zoology's names for them; ``levels``, ``d_v``, ``decay``, ``bias`` and ``gain_bias_init`` are fsla's."""

    def __init__(self, d_model: int, layer_idx: Optional[int] = None, *, n_heads: int, levels: list, d_v: int = 64,
                 decay: Optional[dict] = None, bias: bool = False, gain_bias_init: Optional[float] = None):
        super().__init__()
        self.layer = FSLA(hidden_size=d_model, num_heads=n_heads, levels=levels, d_v=d_v, decay=decay, bias=bias,
                          gain_bias_init=gain_bias_init, layer_idx=layer_idx)

    def forward(self, x):
        return self.layer(x)[0]

    def state_size(self, sequence_length: Optional[int] = None, **kwargs) -> int:
        return self.layer.state_size(sequence_length)
