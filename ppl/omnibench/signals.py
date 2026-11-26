"""Proxy observations synthesized only from interactions that already occurred."""
import numpy as np


def observed_tactile(held,contact,events,now,task):
    y,x=np.mgrid[:16,:16];pad=np.exp(-((x-7.5)**2+(y-7.5)**2)/8)
    proxy=5. if held else contact
    if task in ['T5','T6'] and any(e['name'] in ['button_activated','tap_detected'] and abs(e['time_s']-now)<.15 for e in events):proxy=max(proxy,3.)
    return np.broadcast_to(pad[None]*proxy,(2,16,16)).copy().astype(np.float32)


def observed_audio(events,now,seed,sr=16000):
    end=int(now*sr);start=max(0,end-int(.25*sr));wave=np.zeros(end-start,np.float32)
    for index,event in enumerate(events):
        event_start=int(event['time_s']*sr)
        if event_start>=end:continue
        n=int(.18*sr);t=np.arange(n)/sr
        noise=np.random.default_rng(400+seed+index).standard_normal(n)
        signal=.17*(np.sin(2*np.pi*(390+110*index)*t)+.1*noise)*np.exp(-23*t)
        lo=max(start,event_start);hi=min(end,event_start+n)
        if hi>lo:wave[lo-start:hi-start]+=signal[lo-event_start:hi-event_start]
    # Same int16 quantization as the source WAV before feature extraction.
    return (np.int16(np.clip(wave,-1,1)*32767).astype(np.float32)/32768)
