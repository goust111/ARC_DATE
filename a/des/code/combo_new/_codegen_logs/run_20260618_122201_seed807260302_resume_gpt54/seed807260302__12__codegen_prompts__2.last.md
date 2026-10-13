代码已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/2.py)。实现是自包含的：`main()` 通过连通块识别被灰墙完整包围的矩形房间，按标记像素的最近墙方向在对应墙面中央开门并擦除标记；`generate_input()` 随机生成不同大小、不同分区、不同标记颜色与方向、以及窄门/宽门两种分支的样例，并用 `main()` 回验避免退化输入。

已按要求完成自检：
```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/2.py")
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
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/2.py"
```
结果通过，`variants` 为 `100`。验证产生的 `__pycache__` 也已清理。