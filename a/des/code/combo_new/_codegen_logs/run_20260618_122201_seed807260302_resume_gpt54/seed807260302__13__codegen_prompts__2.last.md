已按要求把代码写到 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/2.py)。`main()` 会按颜色配对两个 4-连通碎片，在局部坐标里搜索唯一能拼成实心矩形的相对滑动关系，再把合并后的矩形放到两个原始包围盒中心中点的四舍五入位置；`generate_input()` 会随机化网格尺寸、对象数、颜色、横/纵方向、矩形尺寸、切分轮廓和间距，并用 `main()` 反验生成样例。

已完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/2.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/2.py"
```

结果里 `variants = 100`，并且已经清理了验证产生的 `__pycache__`。