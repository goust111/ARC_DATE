代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/17/1.py)。实现包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，`main()` 按灰色缺口方框解析旋转方向，再用框内标记色重绘旋转后的蓝色图案；生成器会随机化网格尺寸、方框数量、布局、缺口方向、标记色/位置和非对称图案，并避免标记点落入任意旋转后的图案位置。

已按要求完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/17/1.py")
mod = SourceFileLoader("new_arc_task", str(p)).load_module()
sigs = set()
for i in range(100):
    inp = mod.generate_input()
    out = mod.main(inp)
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30, f"input grid exceeds 30x30: {inp.shape}"
    assert out.ndim == 2 and max(out.shape) <= 30, f"output grid exceeds 30x30: {out.shape}"
    sigs.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())
print("variants", len(sigs))
assert len(sigs) >= 20
PY
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/17/1.py"
```
结果是 `variants 100`，并且已清理验证产生的 `__pycache__`。