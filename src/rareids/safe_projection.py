"""Deterministic, variance-safe PCA coordinates for diagnostic scatter plots.

No model training, no t-SNE or stochastic embedding optimization.
Degenerate feature maps become zero-valued coordinates and are explicitly marked.
"""
import numpy as np

def robust_pca_2d(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1:
        raise ValueError("Expected at least two rows of numeric feature embeddings")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite embeddings: refuse to plot or conceal missing values")
    average=x.mean(axis=0,keepdims=True)
    centered=x-average
    std=np.sqrt(np.mean(centered**2,axis=0))
    varying=std>1e-10
    if not np.any(varying):
        return np.zeros((len(x),2),dtype=np.float64),{
            "projection":"PCA","status":"zero_variance",
            "informative_feature_dimensions":0,
            "total_feature_dimensions":int(x.shape[1]),
            "explained_variance_ratio_first_two":[0.0,0.0],
            "note":"All selected embeddings are constant; do not interpret separation."}
    z=centered[:,varying]/std[varying]
    z=z-z.mean(axis=0,keepdims=True)
    cov=(z.T@z)/(len(z)-1)
    cov=(cov+cov.T)*0.5
    eigenvalues,vectors=np.linalg.eigh(cov)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=np.maximum(eigenvalues[order],0.0)
    bases=vectors[:,order[:min(2,len(order))]].copy()
    coords=z@bases
    if coords.shape[1]==1:
        coords=np.column_stack((coords,np.zeros(len(z))))
    # Stable orientation across the same NumPy data; global sign has no meaning.
    for col in range(2):
        largest=np.argmax(np.abs(coords[:,col]))
        if coords[largest,col]<0: coords[:,col]*=-1.0
    total=float(eigenvalues.sum())
    ratios=[float(v/total) if total>0 else 0.0 for v in eigenvalues[:2]]
    ratios += [0.0]*(2-len(ratios))
    return coords,{
        "projection":"PCA","status":"ok",
        "informative_feature_dimensions":int(varying.sum()),
        "total_feature_dimensions":int(x.shape[1]),
        "explained_variance_ratio_first_two":ratios,
        "note":"Exploratory PCA of saved embeddings; no physical time or class-separation inference."}

def project_saved_embeddings(path,output,classes):
    """Take only actual frozen embeddings; persist coordinates and diagnostics."""
    import json
    from pathlib import Path
    import pandas as pd
    path,output=Path(path),Path(output)
    with np.load(path,allow_pickle=False) as arrays:
        required={"sample_index","y","raw","cnn","vit","fused"}
        if not required.issubset(arrays.files):
            raise ValueError("Saved embedding NPZ lacks: "+str(required-set(arrays.files)))
        sample=arrays["sample_index"].astype(np.int64)
        targets=arrays["y"].astype(np.int64)
        if not len(sample) or len(targets)!=len(sample):
            raise ValueError("Saved samples and labels have inconsistent lengths")
        coords=[]
        details={}
        for key in ("raw","cnn","vit","fused"):
            features=arrays[key]
            if features.shape[0]!=len(sample):
                raise ValueError("Embedding rows do not match sampled labels: "+key)
            xy,diag=robust_pca_2d(features)
            details[key]=diag
            for i,index in enumerate(sample):
                y=int(targets[i])
                if y<0 or y>=len(classes): raise ValueError("Invalid saved class label")
                coords.append({"representation":key,"class":classes[y],
                              "sample_id":int(index),"projection":"PCA",
                              "projection_status":diag["status"],
                              "x":float(xy[i,0]),"y":float(xy[i,1])})
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coords).to_csv(output,index=False)
    diagnostics=output.parent.parent/"projection_diagnostics.json"
    diagnostics.write_text(json.dumps(details,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return details
