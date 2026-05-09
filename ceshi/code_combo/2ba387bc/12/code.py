# Combo variant of 2ba387bc (index 12)
# combo_spec: D3/2+D10/1
# Name: 三种方块类型+三列输出
#
# 设计分析
# a) 具体化与行为追踪
#    - 原题输入是在背景上散布多个 4x4 方块，只有两类：
#      1) frame：外圈 1 像素边框有色，中心 2x2 是背景。
#      2) solid：16 个像素全是同一种非背景色。
#    - 原题输出把 frame 堆到左列、solid 堆到右列，每列宽 4，总宽 8；
#      同类内部按输入中的 (row, col) 排序后自上而下堆叠。
#    - 本融合版新增第 3 类 cross：仍然是 4x4，外圈同色，中心 2x2 为固定
#      对角图案 [[color, bg], [bg, color]]。这样三类在肉眼上分别是：
#      空心中心 / 全实心 / 斜对角半填充。
#    - 小例子追踪：
#      原逻辑看到一个 uniform border 的 4x4，如果中心全背景则进 frames，
#      否则一律落入 solids。于是新加的 cross 在原逻辑下会被误判为 solid。
#      融合后必须改成显式三分类：center 全 bg -> frame；整体单色 -> solid；
#      center 等于固定对角图案 -> cross。输出也改为 3 列宽 12，依次放
#      frame / solid / cross。
# b) 复杂度校准
#    - 第三类只增加一个固定模板，不引入旋转、镜像或多种中心花样，避免样例
#      太少时规则不可归纳。
#    - 每个生成样例都至少包含一份三种类型，保证“第三列存在”不是罕见角落。
# c) 原 main() 的隐式假设审查
#    - `bbox_h(b) != 4 or bbox_w(b) != 4`：仍成立；所有合法对象依旧固定为 4x4。
#    - `find_components(..., monochromatic=False)`：仍成立；三类对象都应作为单个
#      多色/带孔连通体被整体提取，生成器用间距保证它们不粘连。
#    - `if border uniform and center all bg -> frame else solid`：失效；cross 会被
#      误吞进 solid，因此改为 `_classify_square()` 的三分逻辑。
#    - `outW = 8` 与 `max(len(frames), len(solids))`：失效；现在必须是宽 12，且
#      高度取三类计数的最大值。
# d) 放置合法性
#    - 生成器继续通过 `random_free_location(..., border=1, padding=1,
#      padding_connectivity=4)` 放置 4x4 patch，保证不越界、不重叠、4 邻接下
#      不连成一片。
#    - 每个对象颜色唯一，避免同色相邻时组件或视觉判断混淆。
# e) 视觉可观察性
#    - 解题者只需观察 4x4 中心结构与输出列位置的对应关系：
#      中心空 -> 左列；全满 -> 中列；对角十字 -> 右列。
#    - 排序仍是输入里的纵向位置优先，和原题一致，因此新增变化只集中在“类型”
#      与“列数”两个可见维度。
# f) 可视化整洁度
#    - 输入仍是稀疏散布的独立 4x4 方块，没有额外标记、覆盖或共享区域。
#    - 输出是规整的三列堆叠，列宽固定一致，便于肉眼比较。
# g) 变化的可推断性
#    - 不引入颜色映射；每个对象输出保持原色。颜色只用于区分对象，不承载隐式
#      规则，因此人类不需要猜测数值级别的颜色变换。
# 9) 新 main() 的新增假设与 generator 保证
#    - 假设：cross 的中心方向固定为主对角线。生成器只生成这一种 cross 模板。
#    - 假设：合法对象只有 frame / solid / cross 三类；未识别 4x4 会被忽略。
#      生成器只构造这三类，因此不会丢对象。
#    - 假设：同类排序按 `(r0, c0)` 足够稳定。生成器随机散布但不会制造重叠，
#      所以该排序与视觉“从上到下、同行时从左到右”一致。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


Obj = tuple[int, int, np.ndarray]


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_cross(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = np.array([[int(col), int(bg)], [int(bg), int(col)]], dtype=int)
    return a


def _classify_square(sub: np.ndarray, bg: int) -> str | None:
    center = sub[1:3, 1:3]
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    whole_colors = set(map(int, sub.flatten()))

    if len(whole_colors) == 1:
        only = int(next(iter(whole_colors)))
        if only != int(bg):
            return "solid"
        return None

    if len(border_colors) != 1:
        return None

    col = int(next(iter(border_colors)))
    if col == int(bg):
        return None

    if bool((center == int(bg)).all()):
        return "frame"

    cross_center = np.array([[col, int(bg)], [int(bg), col]], dtype=int)
    if bool((center == cross_center).all()):
        return "cross"

    return None


def _collect_objects(grid: np.ndarray) -> tuple[int, list[Obj], list[Obj], list[Obj]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames: list[Obj] = []
    solids: list[Obj] = []
    crosses: list[Obj] = []

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        kind = _classify_square(sub, bg)
        item = (int(r0), int(c0), sub)
        if kind == "frame":
            frames.append(item)
        elif kind == "solid":
            solids.append(item)
        elif kind == "cross":
            crosses.append(item)

    frames.sort(key=lambda t: (t[0], t[1]))
    solids.sort(key=lambda t: (t[0], t[1]))
    crosses.sort(key=lambda t: (t[0], t[1]))
    return bg, frames, solids, crosses


def _build_output(frames: list[Obj], solids: list[Obj], crosses: list[Obj], bg: int) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids), len(crosses), 0)
    out = np.full((out_h, 12), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    for i, (_r0, _c0, sub) in enumerate(crosses):
        paste(out, sub, int(i * 4), 8)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into three columns by internal structure:
    frames left, solids middle, diagonal-cross squares right.
    """
    bg, frames, solids, crosses = _collect_objects(np.array(grid, dtype=int))
    return _build_output(frames, solids, crosses, bg)


def _legacy_binary_output(grid: np.ndarray) -> np.ndarray:
    """
    Reference for anti-degeneration checks:
    emulate the original two-column behavior where every non-frame becomes solid.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames: list[Obj] = []
    solids: list[Obj] = []

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))
        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (t[0], t[1]))
    solids.sort(key=lambda t: (t[0], t[1]))
    out_h = 4 * max(len(frames), len(solids), 0)
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    """
    Quantify whether the generated sample really exercises the fusion behavior.
    - `crosses`: number of third-type objects present.
    - `three_way_active`: whether all three categories appear.
    - `differs_from_legacy`: whether new output differs from original two-column rule.
    """
    bg, frames, solids, crosses = _collect_objects(grid)
    new_out = _build_output(frames, solids, crosses, bg)
    old_out = _legacy_binary_output(grid)
    return {
        "frames": len(frames),
        "solids": len(solids),
        "crosses": len(crosses),
        "three_way_active": len(frames) > 0 and len(solids) > 0 and len(crosses) > 0,
        "differs_from_legacy": new_out.shape != old_out.shape or not np.array_equal(new_out, old_out),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 33))
    w = int(np.random.randint(22, 33))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    n_crosses = int(np.random.randint(1, 4))

    need = n_frames + n_solids + n_crosses
    colors = [int(c) for c in np.random.permutation(np.arange(1, 10))[:need]]
    if len(colors) != need:
        return None

    frames: list[Obj] = []
    solids: list[Obj] = []
    crosses: list[Obj] = []

    try:
        for _ in range(n_frames):
            col = colors.pop()
            sub = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            frames.append((int(r0), int(c0), sub))

        for _ in range(n_solids):
            col = colors.pop()
            sub = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            solids.append((int(r0), int(c0), sub))

        for _ in range(n_crosses):
            col = colors.pop()
            sub = _make_cross(col, bg)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            crosses.append((int(r0), int(c0), sub))

        frames.sort(key=lambda t: (t[0], t[1]))
        solids.sort(key=lambda t: (t[0], t[1]))
        crosses.sort(key=lambda t: (t[0], t[1]))
        expected = _build_output(frames, solids, crosses, bg)
        metrics = fusion_metrics(g)
        if not bool(metrics["three_way_active"]) or not bool(metrics["differs_from_legacy"]):
            return None
        if np.array_equal(main(g), expected):
            return g
    except Exception:
        return None

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
