已将实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/7.py)。

`main()` 按几何规则做了通用求解：4 连通分量拆分、识别直线引导段、基于“同色 + 同轴 + 间隔 1”做唯一配对，然后按对象跨度平移复制并擦除引导线。`generate_input()` 会随机化网格尺寸、对象数、颜色、方向、复制次数和印章形状，并在生成后用 `main()` 回验，避免退化或歧义样例。

已执行并通过：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/7.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/7.py"
```

结果里 `variants = 100`，并且已清理验证产生的 `__pycache__`。