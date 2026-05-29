# Variant of 2ba387bc
# Dimension: D16 — 变换——分类后的形态变化 (transform)
# Change: 方块在分类后保持原始形态不变直接放入输出... → 实心方块缩小为2x2放入右列

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis

a) 具体化与行为追踪
   - 输入仍然只有两类 4x4 物体：空心框架和实心方块，背景仍是全局最常见色。
   - 分类规则不变：若 4x4 物体边框单色且中心 2x2 全是背景，则它是框架；否则视作实心类。
   - 输出左列仍放 4x4 框架，每个框架占 4 行、4 列，按输入中出现的行坐标从上到下排序。
   - 变体只改右列：每个 4x4 实心块不再原样输出，而是缩成 2x2，再按同样排序堆叠。
   - 具体例子：
     输入里若有 2 个框架 F1/F2 和 3 个实心块 S1/S2/S3，原任务输出高度是
     4*max(2,3)=12，右列每个 S 都是 4x4，放在行 0、4、8。
     现在应改为：F1/F2 仍在左列行 0、4；S1/S2/S3 改成 2x2，放在右列行 0、2、4。
     所以新输出高度变成 max(4*2, 2*3)=8，两列不再共享同一个 4 行节拍。

b) 复杂度校准
   - 不改输入物体形状，避免把 D16 变成“输入也改变”的复合变体。
   - 实心块保持纯色即可，因为缩小后面积从 16 像素降到 4 像素，视觉差异已经足够明显。
   - 仍保留 1..4 个框架和 1..4 个实心块，保证样例中既有分类，又能看到右列压缩。

c) 对原 main() 的逐段隐式假设审查
   - `find_components(... monochromatic=False)`：依赖不同物体之间不发生 4 邻接连通。该前提仍成立，
     生成器继续用 padding=1 保证物体隔开。
   - `bbox_h(b) != 4 or bbox_w(b) != 4`：依赖输入物体还是 4x4。变体没有改输入，因此仍成立。
   - `if len(bcols) == 1 and (sub[1:3,1:3] == bg).all()`：依赖框架识别靠中心 2x2 是否背景。
     变体没改分类定义，仍成立。
   - `outH = 4 * max(...)`：旧代码把两类物体都假定为 4 行高；在变体中失效，必须改成
     `max(4*len(frames), 2*len(solids))`。
   - `outW = 8` 与 `paste(..., sub, ..., 4)`：旧代码把右列宽度写死为 4。变体中右列实际只有 2 列，
     应改成总宽 6，并把缩小后的 2x2 粘到列 4..5。

d) 放置合法性
   - 输入端仍在一个较大的背景网格上随机放 4x4 物体。
   - 每次放置使用 `random_free_location(... padding=1, padding_connectivity=4)`，避免物体重叠、
     越界，且防止两个 4x4 在 4 邻接意义下粘连成一个组件。
   - 生成后再重新跑一次分类统计，确认识别到的框架数和实心数与计划一致。

e) 视觉可观察性
   - 每个样本至少包含 1 个实心块，因此右列一定会出现 2x2 小块，而不是退化回原任务。
   - 左列仍是 4x4 框架，右列变成 2x2 实心块，两类输出在大小与堆叠节拍上都明显不同，
     人眼从 2-3 个样例就能看出“实心块被缩小”。

f) 可视化整洁度
   - 输出只保留两块区域：左侧 4 列框架区，右侧 2 列实心区，没有额外标记或重叠。
   - 右列以 2 行步长连续堆叠，会与左列形成刻意的不齐平效果，但不会互相遮挡。

g) 颜色变化可推断性
   - 本变体不引入颜色映射。颜色只继承自输入物体本身。
   - 因为不同物体继续使用不同颜色，解题者不会把“缩小”误解成“换色”。

h) 生成器如何避免退化
   - 生成器只产生至少 1 个实心块的样本，并在生成后调用新 solver 做一次自检。
   - 同时检查输出右列非背景像素数是否恰好等于 `4 * n_solids`，保证每个实心块确实变成 2x2。

i) 新 solver 的新增假设
   - 右列默认锚定在列 4..5，因此总输出宽度固定为 6。
   - 所有需要缩小的对象仍来自 4x4 输入组件；生成器继续强制这一点。
   - 对“实心类”的 2x2 缩小采用中心区域 `sub[1:3, 1:3]`。对本任务的纯色实心块来说，
     中心裁切与下采样等价，且实现最直接。
"""


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        bcols = set(map(int, border.flatten()))

        if len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _shrink_solid(sub: np.ndarray) -> np.ndarray:
    return np.array(sub[1:3, 1:3], dtype=int, copy=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type.

    Variant D16:
    - Frames stay 4x4 in the left column.
    - Solids are shrunk from 4x4 to 2x2 and stacked in the right column.
    - Left and right columns therefore use different row strides: 4 for frames, 2 for solids.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)

    outH = max(4 * int(len(frames)), 2 * int(len(solids)))
    outW = 6
    out = np.full((outH, outW), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, _shrink_solid(sub), int(i * 2), 4)

    return out


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(18, 30))
    W = int(np.random.randint(18, 30))
    g = np.full((H, W), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    try:
        used_cols: set[int] = set()

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        frames, solids = _collect_objects(g, bg)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None

        out = main(g)
        expected_shape = (max(4 * n_frames, 2 * n_solids), 6)
        if out.shape != expected_shape:
            return None
        if int((out[:, 4:6] != bg).sum()) != 4 * n_solids:
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
