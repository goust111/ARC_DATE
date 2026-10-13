已将实现写入 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/8.py)。

`main()` 通过边缘 `Maroon` 标记判断删除的是水平条纹还是垂直条纹，再用“某行/列里重复出现的唯一非黑非 `Maroon` 颜色”恢复条纹，并在删除被标记条纹后用正交条纹颜色补回交叉点。`generate_input()` 随机化了网格尺寸、四种标记边、横竖条纹数量、颜色、被删除条纹子集和每个交叉点的遮挡方向，并强制至少存在真实的“删上层后恢复下层”情形，避免退化样例。

自检已执行并通过：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/8.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/8.py"
```

结果里 `variants` 为 `100`，并且已清理 `__pycache__`。