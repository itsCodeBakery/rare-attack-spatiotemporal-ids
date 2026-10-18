"""SCV-IDS paper-aligned preparation. Train-only fitting; no test-set access.
SCV-IDS Section III: index categorical encoding, MinMax, five-row windows,
24-bit RGB values placed in a 16x16 image, class of the newest row.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MAX24=(1<<24)-1
UNSUPPORTED={"id","label","attack_cat","attack_category"}

@dataclass
class FittedPreprocessor:
    columns:list
    categorical:list
    vocabs:dict
    min_values:list
    max_values:list

    def as_dict(self):
        return {"columns":self.columns,"categorical":self.categorical,
                "vocabs":self.vocabs,"min_values":self.min_values,
                "max_values":self.max_values,
                "unknown_index_policy":"len(training vocabulary)",
                "scaling":"MinMax fit using INTERNAL training examples only",
                "out_of_range_policy":"clip to [0,1] prior to 24-bit quantization"}

def fit_preprocessor(frame,columns,train_indices):
    subset=frame.iloc[train_indices]
    categoricals=[c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    vocabs={}
    for c in categoricals:
        values=subset[c].astype(str).str.strip()
        vocabs[c]={v:i for i,v in enumerate(sorted(values.unique()))}
    encoded=_encode(subset,columns,categoricals,vocabs)
    scaler=MinMaxScaler()
    scaler.fit(encoded)
    result=FittedPreprocessor(list(columns),categoricals,vocabs,
                              scaler.data_min_.astype(float).tolist(),
                              scaler.data_max_.astype(float).tolist())
    return result

def _encode(frame,columns,categoricals,vocabs):
    arrays=[]
    for c in columns:
        if c in categoricals:
            values=frame[c].astype(str).str.strip()
            codes=values.map(vocabs[c]).fillna(len(vocabs[c]))
            arr=codes.to_numpy(dtype=np.float64)
        else:
            arr=pd.to_numeric(frame[c],errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("Invalid/nonfinite feature: "+str(c))
        arrays.append(arr)
    return np.stack(arrays,axis=1)

def transform_preprocessor(prep,frame):
    arr=_encode(frame,prep.columns,prep.categorical,prep.vocabs)
    minimum=np.asarray(prep.min_values,dtype=np.float64)
    maximum=np.asarray(prep.max_values,dtype=np.float64)
    span=maximum-minimum
    span[span==0]=1.0
    scaled=(arr-minimum)/span
    outside=int(np.count_nonzero((scaled<0)|(scaled>1)))
    return np.clip(scaled,0,1).astype(np.float32),outside

def windows_to_rgb(features,ends,window=5):
    """Return batch RGB channels-first uint8 images, 16x16.

    Quantization rounding is a documented reconstruction choice because the
    paper does not specify precise normalized-float -> 24-bit integer rounding.
    """
    features=np.asarray(features,dtype=np.float32)
    ends=np.asarray(ends,dtype=np.int64)
    if features.ndim!=2 or features.shape[1]!=42: raise ValueError("Expected 42 features")
    if window*42>256: raise ValueError("Too many features for 16x16 pixels")
    if len(ends)==0: return np.empty((0,3,16,16),dtype=np.uint8)
    if ends.min()<window-1 or ends.max()>=len(features):
        raise ValueError("Windows exceed permitted split boundaries")
    offsets=np.arange(window-1,-1,-1)
    rows=ends[:,None]-offsets[None,:]
    x=features[rows].reshape(len(ends),window*42)
    if np.any(x<0) or np.any(x>1) or not np.isfinite(x).all():
        raise ValueError("24-bit serialization requires finite 0..1 values")
    integer=np.rint(x.astype(np.float64)*MAX24).astype(np.uint32)
    packed=np.zeros((len(ends),256,3),dtype=np.uint8)
    packed[:,:window*42,0]=(integer>>16).astype(np.uint8)
    packed[:,:window*42,1]=(integer>>8).astype(np.uint8)
    packed[:,:window*42,2]=integer.astype(np.uint8)
    return packed.reshape(-1,16,16,3).transpose(0,3,1,2).copy()

def make_windows(features,labels,window=5,chunk=4096,progress=None):
    """Serialize windows separately INSIDE an already split partition."""
    features=np.asarray(features,dtype=np.float32)
    labels=np.asarray(labels,dtype=np.int64)
    if len(features)!=len(labels) or len(features)<window:
        raise ValueError("Not enough partition rows for window generation")
    count=len(features)-window+1
    images=np.empty((count,3,16,16),dtype=np.uint8)
    targets=np.empty(count,dtype=np.int64)
    it=range(0,count,chunk)
    if progress is not None: it=progress(it,total=(count+chunk-1)//chunk,
                                         desc="Serialize RGB windows",unit="chunks")
    for start in it:
        stop=min(start+chunk,count)
        ends=np.arange(start+window-1,stop+window-1)
        images[start:stop]=windows_to_rgb(features,ends,window)
        targets[start:stop]=labels[ends]
    return images,targets
