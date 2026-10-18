"""Standalone ViT preserves 4-block 128-wide encoder, has no CNN branch."""
import pytest
def test_standalone_vit_forward():
    torch=pytest.importorskip("torch")
    from rareids.controlled_models import ControlledViTOnly
    model=ControlledViTOnly().eval()
    assert not hasattr(model,"cnn")
    assert len(model.vit.encoder.layers)==4
    assert model.vit.encoder.layers[0].self_attn.num_heads==8
    with torch.no_grad():
        result=model(torch.rand(2,3,16,16))
    assert result.shape==(2,10)
