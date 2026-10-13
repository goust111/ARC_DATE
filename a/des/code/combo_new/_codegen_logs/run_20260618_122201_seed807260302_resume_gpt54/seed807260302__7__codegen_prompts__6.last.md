已将实现写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/6.py)。

`main()` 按结构规则解析顶部颜色条、识别不接触边界的黑色封闭内区，并通过相邻灰色连通块判断内部孔洞数后填色。`generate_input()` 会随机化网格尺寸、颜色条长度、区域数量、区域形状、孔洞数量以及孔洞大小，并用 `main()` 自校验生成结果。

已执行并通过：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/6.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/6.py"
```

自检结果 `variants 100`，并且已清理 `__pycache__`。