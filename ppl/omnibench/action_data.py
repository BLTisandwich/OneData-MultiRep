"""Episode-disjoint causal action windows and the released EE/gripper loss."""
import torch
import torch.nn.functional as F
from .data import EEDataset,move
from .settings import CONFIG
from .runtime import autocast


class AlignedFrames(EEDataset):
    def __init__(self,records,split,augment=False):
        super().__init__(split);self.augment=augment
    def __getitem__(self,index):
        row=super().__getitem__(index)
        if self.augment:
            x=row['sensors']['points'];row['sensors']['points']=x[:,torch.randperm(x.shape[1])]
        return row


def losses(model,batch,device):
    with autocast(device):
        output=model(move(batch,device),batch['text'],batch['branches'],task='ee')
        target=batch['target'].to(device);scale=CONFIG['network']['position_loss_scale_m']
        xyz=F.smooth_l1_loss(output['action'][:,:3]/scale,target[:,:3]/scale)
        grip=.25*F.binary_cross_entropy_with_logits(output['gripper_logits'],target[:,3])
    mae=(output['action'][:,:3]-target[:,:3]).abs().mean()*1000
    accuracy=((output['action'][:,3]>.5)==(target[:,3]>.5)).float().mean()
    return xyz+grip,xyz,grip,mae,accuracy
