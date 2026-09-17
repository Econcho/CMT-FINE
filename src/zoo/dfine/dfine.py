"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
        "cmt",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        cmt: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.cmt = cmt

    def forward(self, x, targets=None):
        images = x
        features = self.backbone(images)
        features = self.encoder(features)
        cmt_state = (
            self.cmt.prepare(images, features[0])
            if self.cmt is not None and self.cmt.active
            else None
        )
        x = self.decoder(features, targets, cmt=self.cmt, cmt_state=cmt_state)

        if self.training and cmt_state is not None:
            x["cmt_evidence_logits"] = cmt_state["evidence_logits"]
            x["cmt_base_stride"] = self.cmt.pyramid.base_stride

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
