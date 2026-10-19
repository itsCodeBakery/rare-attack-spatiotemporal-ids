"""HARDREP-001 source and research-protocol regression tests."""
import ast,json
from pathlib import Path

def test_new_hardrep_architecture_and_frozen_split():
 c=json.loads(Path("configs/hardrep001_gpu_seed42.json").read_text())
 assert c["official_test"]=="FORBIDDEN"
 assert c["training_examples"]==140328
 assert c["target_validation_examples"]==35005
 assert c["class_pair_ids"]==[[2,3],[1,3],[0,3]]
 assert c["seed"]==42
 src=Path("scripts/run_hardrep001_gpu.py").read_text()
 ast.parse(src)
 for keyword in ("contrastive(z,by","pair_margin(z,by","best_model_PRIVATE.pt","prepare(source,ref)","official_test_used","torch.cuda.is_available"):
  assert keyword in src
 nb=json.loads(Path("notebooks/16_hardrep001_gpu_contrastive_ids.ipynb").read_text())
 code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
 ast.parse(code)
 assert "restore_ctrl003_tabular.py" in code
 assert "torch.cuda.is_available()" in code
