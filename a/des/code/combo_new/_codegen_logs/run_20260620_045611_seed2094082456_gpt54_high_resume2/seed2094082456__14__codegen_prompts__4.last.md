代码已写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/4.py)。

实现内容是自包含的：`main(grid)` 先识别单色空心矩形框组件，建立嵌套树，再按深度奇偶执行“奇数层填充、偶数层内缩”；`generate_input()` 随机生成 1 到 2 棵嵌套矩形树，覆盖不同网格尺寸、深度、兄弟数量、位置和颜色，并用重试避免越界、重叠和退化布局。

已按要求完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/4.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/4.py"
```

结果：`variants 100`，`py_compile` 通过，`__pycache__` 已清理。