"""Controlled standalone ViT baseline sharing the unchanged SCV ViT encoder.

Engineering control CTRL-VIT-001, not SCV authors' standalone reference code.
"""
import torch
from torch import nn
from rareids.scv_model import ViTEncoder

class ControlledViTOnly(nn.Module):
    def __init__(self, dropout=0.2, classes=10):
        super().__init__()
        self.vit=ViTEncoder(dim=128,heads=8,layers=4,mlp_dim=256,
                            dropout=dropout,patch=8)
        # Same 128->128->10 nonlinear output head pattern as old SCV reconstruction.
        # Original SCV used bidirectional crossattention/fusion; this control does not.
        self.classifier=nn.Sequential(nn.Linear(128,128),nn.GELU(),
                                      nn.Dropout(dropout),nn.Linear(128,classes))
    def forward(self,x):
        tokens=self.vit(x)
        return self.classifier(tokens.mean(dim=1))
