"""V2 architecture is identical to V0 except final CNN nonlinearity."""
import pytest

def test_v2_minimal_activation_change():
    torch=pytest.importorskip("torch")
    from rareids.scv_model import SCVIDS
    from rareids.v2_model import SCVIDSLeakyFinalCNN
    original=SCVIDS()
    corrected=SCVIDSLeakyFinalCNN()
    assert isinstance(original.cnn.layers[11],torch.nn.ReLU)
    assert isinstance(corrected.cnn.layers[11],torch.nn.LeakyReLU)
    assert corrected.cnn.layers[11].negative_slope==0.1
    assert set(dict(original.named_parameters()))==set(dict(corrected.named_parameters()))
    assert sum(x.numel() for x in original.parameters())==sum(x.numel() for x in corrected.parameters())
    corrected.eval()
    with torch.no_grad():
        logits=corrected(torch.rand(2,3,16,16))
        masked,cnn,vit=corrected.forward_branches(torch.rand(2,3,16,16),zero_cnn=True)
    assert logits.shape==(2,10)
    assert masked.shape==(2,10)
    assert cnn.shape==(2,16,128) and vit.shape==(2,4,128)
