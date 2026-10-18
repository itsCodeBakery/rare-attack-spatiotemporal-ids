"""Paper-specific invariants; synthetic data only."""
import numpy as np
import pandas as pd
import pytest
from rareids.scv_preprocessing import fit_preprocessor,transform_preprocessor,make_windows,windows_to_rgb,MAX24

def _frame(n=20):
    f=pd.DataFrame({f"f{i}":np.linspace(0,1,n) for i in range(41)})
    f["proto"]=["tcp" if i%2 else "udp" for i in range(n)]
    return f

def test_training_only_scaling_and_shape():
    f=_frame()
    columns=list(f.columns)
    prep=fit_preprocessor(f,columns,np.arange(15))
    transformed,out=transform_preprocessor(prep,f)
    assert len(prep.vocabs["proto"])==2
    assert transformed.shape==(20,42)
    images,y=make_windows(transformed,np.arange(20)%10,window=5)
    assert images.shape==(16,3,16,16)
    assert y.tolist()[0]==4
    assert y[-1]==19%10
    assert np.all(images[:,:,13:,:]==0)==False # row feature values occupy first 210 pixels
    assert np.all(images.reshape(-1,3,256)[:,:,210:]==0)

def test_rgb_precision():
    x=np.zeros((5,42),dtype=np.float32)
    x[-1,0]=1.0
    result=windows_to_rgb(x,np.array([4]))
    flat=result.transpose(0,2,3,1).reshape(1,256,3)
    assert flat[0,168].tolist()==[255,255,255]
    assert flat[0,0].tolist()==[0,0,0]
    assert int(MAX24)==16777215

def test_split_bounds():
    with pytest.raises(ValueError):
        windows_to_rgb(np.zeros((5,42),np.float32),np.array([3]))
    with pytest.raises(ValueError):
        make_windows(np.zeros((4,42),np.float32),np.zeros(4,dtype=int))

def test_model_shapes_and_focal_loss():
    torch=pytest.importorskip("torch")
    from rareids.scv_model import SCVIDS,MulticlassFocalLoss
    model=SCVIDS(n_classes=10).eval()
    with torch.no_grad():
        logits,features=model(torch.rand(2,3,16,16),return_features=True)
    assert logits.shape==(2,10)
    assert features["cnn"].shape==(2,128)
    assert features["vit"].shape==(2,128)
    assert features["fused"].shape==(2,128)
    assert torch.isfinite(MulticlassFocalLoss()(logits,torch.tensor([0,1])))
