"""Pure numerical verification helpers for CTRL-TAB-001."""
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix

def macro_eval(predicted,validation_dataset):
    labels=validation_dataset.get_label().astype(np.int64)
    p=np.asarray(predicted)
    if p.ndim==1:
        p=p.reshape(len(labels),-1)
    if p.shape!=(len(labels),10):
        raise ValueError(f"Unexpected LightGBM ten-class probability tensor shape: {p.shape}")
    return "macro_f1", float(f1_score(labels,p.argmax(axis=1),
                                    labels=np.arange(10),average="macro",
                                    zero_division=0)), True

def class_weights(labels,power,max_weight=12):
    labels=np.asarray(labels,dtype=np.int64)
    counts=np.bincount(labels,minlength=10)
    if np.any(counts==0):
        raise ValueError("Every class must exist in the training fold")
    weights=(counts.max()/counts.astype(float))**power
    weights=np.clip(weights,1,max_weight)
    return weights

def score_predictions(y,probs,class_names):
    y=np.asarray(y).astype(int); probs=np.asarray(probs)
    if probs.shape!=(len(y),10) or len(class_names)!=10:
        raise ValueError("Mismatched fixed 10-class probability/label count")
    if not np.isfinite(probs).all():
        raise FloatingPointError("Nonfinite model probabilities")
    pred=probs.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=np.arange(10))
    report={"count":int(len(y)),"correct":int(np.trace(cm)),
            "accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",labels=np.arange(10),
                                    zero_division=0)),
            "DoS_to_Exploits":int(cm[class_names.index("DoS"),class_names.index("Exploits")]),
            "Backdoor_to_Exploits":int(cm[class_names.index("Backdoor"),class_names.index("Exploits")])}
    return report,cm,pred
