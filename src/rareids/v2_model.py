"""CTRL-CNNVIT-V2-S42: minimal CNN activation repair under frozen comparison protocol.

Single architectural change relative to independent reconstructed SCVIDS:
last CNN ReLU becomes LeakyReLU(slope=0.1), preserving negative preactivations.
No change to ViT, CNN conv filters, attention, classifier, optimizer or windows.
This is an engineering control, NOT a newly proposed algorithm and NOT author code.
"""
import torch
from torch import nn
from rareids.scv_model import SCVIDS

class SCVIDSLeakyFinalCNN(SCVIDS):
    def __init__(self,n_classes=10,dropout=0.2,patch=8,negative_slope=0.1):
        super().__init__(n_classes=n_classes,dropout=dropout,patch=patch)
        if not isinstance(self.cnn.layers[11],nn.ReLU):
            raise RuntimeError("Original paper-aligned CNN structure changed; re-audit before repair")
        self.cnn.layers[11]=nn.LeakyReLU(negative_slope=negative_slope,inplace=False)

    def forward_branches(self,x,zero_cnn=False,zero_vit=False):
        """Diagnostic inference branch suppression; same fusion as normal forward."""
        v=self.vit(x)
        c=self.cnn(x)
        if zero_cnn: c=torch.zeros_like(c)
        if zero_vit: v=torch.zeros_like(v)
        vc,_=self.vit_to_cnn(v,c,c,need_weights=False)
        cv,_=self.cnn_to_vit(c,v,v,need_weights=False)
        z=self.fusion(torch.cat([vc.mean(1),cv.mean(1)],dim=1))
        return self.classifier(z),c,v
