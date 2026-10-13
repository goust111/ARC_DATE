代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/1.py)。

实现包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，`main()` 带了按要求分四部分的 docstring。求解逻辑会识别“矩形边框减去一个单像素缺口”的框，统计内部种子颜色数量，并按内部首次出现顺序在缺口朝向的框外并排生长分支。生成器会随机化网格大小、框数量、方向、尺寸、颜色组合和种子计数，并在生成时用规格输出对 `main()` 做一致性校验。

已执行并通过：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/1.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/1.py"
```

结果里 `variants` 为 `100`，并且已经清理了验证产生的 `__pycache__`。