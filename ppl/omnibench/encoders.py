"""Modality-specific learned branches and frozen visual/speech encoders."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class TinyImageCNN(nn.Module):
    def __init__(self, channels, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels,32,5,2,2),nn.GELU(),
            nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,dim,3,2,1),nn.GELU())
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        return self.norm(self.net(x).mean((-2,-1)))

class TemporalCNN(nn.Module):
    def __init__(self, channels, dim=128, max_history=32):
        super().__init__()
        self.position=nn.Parameter(torch.randn(1,max_history,channels)*0.01)
        self.net=nn.Sequential(nn.Conv1d(channels,64,3,padding=1),nn.GELU(),
            nn.Conv1d(64,dim,3,padding=1),nn.GELU())
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        # Windows end at the observation timestamp; no future samples present.
        if x.shape[1]>self.position.shape[1]:
            raise ValueError("History exceeds configured temporal embeddings")
        z=self.net((x+self.position[:,:x.shape[1]]).transpose(1,2))
        return self.norm(z[:,:,-1]+z.mean(-1))

class MeshGIN(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.input=nn.Linear(3,64)
        self.eps=nn.Parameter(torch.zeros(2))
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,64)) for _ in range(2)])
        self.output=nn.Sequential(nn.Linear(128,dim),nn.LayerNorm(dim))
    def forward(self,vertices,adjacency):
        # GIN aggregation is a sum over actual mesh connectivity, no placeholder graph.
        h=F.gelu(self.input(vertices))
        for i,layer in enumerate(self.layers):
            h=F.gelu(layer((1+self.eps[i])*h+torch.bmm(adjacency,h)))
        return self.output(torch.cat((h.mean(1),h.amax(1)),-1))

class PointNet(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,128),nn.GELU())
        self.output=nn.Sequential(nn.Linear(259,dim),nn.LayerNorm(dim))
    def forward(self,x):
        center=x.mean(1)
        z=self.point(x-center[:,None,:])
        return self.output(torch.cat((z.amax(1),z.mean(1),center),-1))

class PretrainedVision(nn.Module):
    def __init__(self,path,dim=128,max_history=32):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone=resnet18(weights=None)
        self.backbone.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        self.backbone.fc=nn.Identity()
        self.backbone.requires_grad_(False)
        self.temporal=TemporalCNN(512,dim,max_history)
        self.multiview_head=nn.Sequential(nn.Linear(1024,dim),nn.GELU(),nn.LayerNorm(dim))
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def train(self,mode=True):
        super().train(mode); self.backbone.eval(); return self
    def image_features(self,x):
        # Dataset supplies RGB in [0,1], not ImageNet-normalized values.
        with torch.no_grad():
            return self.backbone((x-self.mean)/self.std)
    def forward(self,rgb,multiview):
        b,l,c,h,w=rgb.shape
        all_images=torch.cat((rgb.reshape(b*l,c,h,w),multiview.reshape(-1,c,h,w)),0)
        z=self.image_features(all_images)
        return self.fuse_features(z[:b*l].reshape(b,l,512),z[b*l:].reshape(b,multiview.shape[1],512))
    def fuse_features(self,video_features,view):
        video=self.temporal(video_features)
        multiview=self.multiview_head(torch.cat((view.mean(1),view.amax(1)),-1))
        return video,multiview

class PretrainedSpeech(nn.Module):
    def __init__(self,path):
        super().__init__()
        from transformers import WhisperModel
        whisper=WhisperModel.from_pretrained(path,local_files_only=True,attn_implementation="eager")
        self.encoder=whisper.encoder
        self.encoder.requires_grad_(False)
        self.dim=self.encoder.config.d_model
        self.norm=nn.LayerNorm(self.dim,elementwise_affine=False)
    def train(self,mode=True):
        super().train(mode); self.encoder.eval(); return self
    def forward(self,x):
        # Whisper fixed context: right padding contains acoustic silence. Preserve
        # pooling mask so instruction duration, not padding length, defines feature.
        n=x.shape[-1]; required=self.encoder.config.max_source_positions*2
        if n>required: raise ValueError("Speech exceeds Whisper context")
        x=F.pad(x,(0,required-n),value=-1.5)
        with torch.no_grad(): z=self.encoder(x,return_dict=True).last_hidden_state
        valid=max(1,(n+1)//2)
        return self.norm(z[:,:valid].mean(1))
