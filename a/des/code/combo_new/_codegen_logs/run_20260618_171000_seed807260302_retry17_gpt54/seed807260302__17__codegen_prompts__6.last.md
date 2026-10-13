代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/6.py)。`main()` 实现了按边缘光源顺序进行的镜面反射追踪，支持按击中镜面的位置决定转向，并在第二次撞到同一面镜子时终止；`generate_input()` 会随机混合“撞边结束”和“回撞首镜结束”两类光路模板，再随机翻转、旋转、尺寸、颜色和光源数量。

已按要求完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/6.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/6.py"
```

结果里 `variants 100`，`py_compile` 通过，`__pycache__` 也已经清理完。