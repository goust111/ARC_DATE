已按要求实现并写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/3.py)。文件包含 `main(grid: np.ndarray) -> np.ndarray`、`generate_input() -> np.ndarray`，`main()` 带了按要求分四部分的 docstring，生成器会随机化网格大小、对象数、方向、对称类别和形状细节，并在生成时校验标记唯一性与布局合法性。

已完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/3.py")
mod = SourceFileLoader("new_arc_task", str(p)).load_module()
sigs = set()
for i in range(100):
    inp = mod.generate_input()
    out = mod.main(inp)
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert out.ndim == 2 and max(out.shape) <= 30
    sigs.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())
print("variants", len(sigs))
assert len(sigs) >= 20
PY
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/3.py"
```

结果通过，`variants 100`；验证产生的 `__pycache__` 也已清理。