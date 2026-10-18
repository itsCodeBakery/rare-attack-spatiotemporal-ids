"""Documented SCV-IDS paper-aligned CNN-ViT with bidirectional crossattention.

Reference: Chen et al., IEEE Internet of Things Journal (2026), Section III,
Table II. No claim of author-code equivalence: QKV projections, classifier widths,
and float-to-24bit quantization are reconstruction choices.
"""
import torch
from torch import nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self,dim=128,heads=8,layers=4,mlp_dim=256,dropout=0.2,patch=8):
        super().__init__()
        if 16%patch: raise ValueError("Patch size must divide 16")
        self.patch=nn.Conv2d(3,dim,kernel_size=patch,stride=patch)
        n=(16//patch)**2
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=nn.Parameter(torch.zeros(1,n+1,dim))
        block=nn.TransformerEncoderLayer(
            d_model=dim,nhead=heads,dim_feedforward=mlp_dim,
            dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=layers,
                                            enable_nested_tensor=False)
        self.norm=nn.LayerNorm(dim)
        nn.init.normal_(self.pos,std=0.02)
        nn.init.normal_(self.cls,std=0.02)

    def forward(self,image):
        x=self.patch(image).flatten(2).transpose(1,2)
        cls=self.cls.expand(len(x),-1,-1)
        x=torch.cat([cls,x],dim=1)+self.pos
        x=self.norm(self.encoder(x))
        return x[:,1:,:] # paper excludes CLS token for fusion

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU()
        )
    def forward(self,image):
        x=self.layers(image)
        if x.shape[1:]!=(128,4,4): raise AssertionError(str(x.shape))
        return x.flatten(2).transpose(1,2) # B,16,128

class SCVIDS(nn.Module):
    def __init__(self,n_classes=10,dropout=0.2,patch=8):
        super().__init__()
        self.vit=ViTEncoder(patch=patch,dropout=dropout)
        self.cnn=CNNEncoder()
        self.vit_to_cnn=nn.MultiheadAttention(128,8,dropout=dropout,batch_first=True)
        self.cnn_to_vit=nn.MultiheadAttention(128,8,dropout=dropout,batch_first=True)
        self.fusion=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),
                                   nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Sequential(nn.Linear(128,128),nn.GELU(),
                                      nn.Dropout(dropout),nn.Linear(128,n_classes))

    def forward(self,image,return_features=False):
        v=self.vit(image)
        c=self.cnn(image)
        # ViT queries local CNN; CNN queries global ViT
        vc,_=self.vit_to_cnn(v,c,c,need_weights=False)
        cv,_=self.cnn_to_vit(c,v,v,need_weights=False)
        fused=self.fusion(torch.cat([vc.mean(dim=1),cv.mean(dim=1)],dim=1))
        logits=self.classifier(fused)
        return (logits,{"cnn":c.mean(1),"vit":v.mean(1),"fused":fused}) if return_features else logits

class MulticlassFocalLoss(nn.Module):
    def __init__(self,gamma=2.0):
        super().__init__()
        self.gamma=float(gamma)
    def forward(self,logits,y):
        logp=F.log_softmax(logits,dim=1).gather(1,y[:,None]).squeeze(1)
        p=logp.exp()
        return (-((1-p)**self.gamma)*logp).mean()
