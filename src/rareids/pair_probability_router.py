"""Conditional probability reallocation for confusion-aware classification.
All probabilities are numeric. Routing never uses example labels.
"""
import numpy as np

def pair_adjust(probabilities, class_a, class_b, conditional_a,
                mode="selective", mass_threshold=.3, mass_full=.7,
                confidence_threshold=.1, confidence_full=.45):
    p=np.asarray(probabilities,dtype=np.float64)
    q=np.asarray(conditional_a,dtype=np.float64)
    if p.ndim!=2 or q.shape!=(len(p),) or class_a==class_b:
        raise ValueError("Invalid pair dimensions")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite probabilities")
    if np.min(p)<0 or np.min(q)<0 or np.max(q)>1:
        raise ValueError("Invalid probabilities")
    mass=p[:,class_a]+p[:,class_b]
    if mode=="selective":
        relevance=np.clip((mass-mass_threshold)/(mass_full-mass_threshold),0,1)
        confidence=np.clip((np.abs(q-.5)-confidence_threshold)/
                           (confidence_full-confidence_threshold),0,1)
        weight=relevance*confidence
    elif mode=="unrestricted":
        weight=np.ones(len(p),dtype=np.float64)
    elif mode=="unchanged":
        weight=np.zeros(len(p),dtype=np.float64)
    else:
        raise ValueError("Unknown routing mode")
    output=p.copy()
    output[:,class_a]=(1-weight)*p[:,class_a]+weight*mass*q
    output[:,class_b]=(1-weight)*p[:,class_b]+weight*mass*(1-q)
    np.testing.assert_allclose(output.sum(1),p.sum(1),rtol=0,atol=1e-10)
    return output,weight
