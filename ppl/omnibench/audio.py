"""Causal log-mel observation encoding."""
import numpy as np
import torch
import torch.nn.functional as F

def logmel(waveform,sr=16000,n_mels=64,frames=32):
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
    filters=mel_filter_bank(num_frequency_bins=257,num_mel_filters=n_mels,min_frequency=0,max_frequency=sr/2,sampling_rate=sr,norm='slaney',mel_scale='slaney')
    data=np.asarray(waveform,dtype=np.float32)
    if len(data)<512:data=np.pad(data,(0,512-len(data)))
    spec=spectrogram(data,window_function(512,'hann'),frame_length=512,hop_length=160,power=2.,mel_filters=filters,log_mel='log10')
    out=torch.from_numpy(spec.copy()).float()[None,None]
    return F.interpolate(out,size=(n_mels,frames),mode='bilinear',align_corners=False)[0]
