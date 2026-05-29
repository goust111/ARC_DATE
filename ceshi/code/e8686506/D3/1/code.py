# Variant of e8686506
# Dimension: D3 — 目标结构：拼合后的形状要求 (dst)
# Change: 拼合后形成一个紧凑的矩形复合体，所有碎片无间隙无重叠地拼成完整矩形... → 拼合后填入已有框架的洞中

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location


# Design analysis
# a) 具体化与行为追踪:
#    - 原题: 输入只有散落碎片，solver 先从总面积推出一个密实矩形，再把全部碎片塞进去。
#    - 变体: 输入里额外出现一个已经摆正的大框架。它是唯一最大的单色 8 连通对象、左右对称，
#      并且 bbox 内保留若干背景色洞位；框架外散落着若干单色碎片。
#    - 具体执行差异: 原逻辑会对碎片总面积做因子分解得到 HxW 矩形；新逻辑不会再猜矩形尺寸，
#      而是先定位框架 bbox，把 bbox 内的背景色 4 连通分量直接当作目标洞槽，再去框架外找同形状碎片。
# b) 复杂度校准:
#    - 如果只是给旧矩形外面加一圈边框，洞是简单矩形，变化不明显；因此生成器使用“外边框 +
#      内部左右对称折线路径”把内部切分成多个不规则小洞，肉眼可见这是在填洞。
#    - 参考任务 67e490f4 的基线是“先识别洞，再做形状匹配”。这里采用同级别机制，但不允许旋转/翻转，
#      保持与原题一致的朝向约束。
# c) 隐式假设审查:
#    - 原 main() 依赖 `total_area -> 唯一矩形因子对`，在本变体中已失效，必须删除。
#    - 原 main() 的 corner/bar/vertical-line 约束都服务于矩形边界；洞填充时这些锚点没有稳定语义，
#      若保留会把内部洞位误判成边界位置，因此移除。
#    - 原 main() 默认 1x1 只能落在外边界；现在洞可以在框架内部，singleton 允许出现在内部小洞中。
# d) 放置合法性:
#    - 生成器先整块阻塞框架 bbox，再把碎片散落到外部，最后只把框架色画回 bbox，因此输入中洞里不会
#      预先混入碎片。
#    - 每个洞组件面积都限制在 1-8；每个碎片正是从某个洞组件裁下来的，因此不会越界、重叠，也不会匹配到
#      错误尺寸。
# e) 视觉可观察性:
#    - 输入明显包含一个大框架和其内部空洞，输出则是同一框架内部被外部碎片补满。
#    - 新规则不依赖颜色 id 的不可见排序；颜色只是随机重映射后的可见外观。
# f) 可视化整洁度:
#    - 框架始终保持单色大对象，碎片全部散落在框架外，结构层次清楚。
#    - 输出中框架色与填充色不重叠，洞组件之间由框架分隔，不会出现视觉混线。
# g) 颜色变化可推断性:
#    - 颜色不编码隐藏规则；人类只需要看“哪个外部形状填哪个内部洞”。
#    - 总颜色数控制在 4-6（含框架色），既有重复支撑，又不会太杂。
# h) 新的 solver 假设与 generator 对应保证:
#    - 假设存在唯一最大单色 8 连通框架组件；生成器用大单色框架且面积远大于任一碎片保证。
#    - 假设框架 bbox 内只有框架色与背景色；生成器通过 bbox 阻塞保证。
#    - 假设每个洞槽都有同形状外部碎片；生成器直接从洞槽裁出碎片，保证精确对应。
# i) 量化的变体判据:
#    - 对输入取最大单色 8 连通组件 bbox，统计其内部背景色数量 `hole_area`。本变体要求 `hole_area > 0`
#      且这些背景色不接触 bbox 边界；原题不存在这种“框架内留洞”的结构。测试阶段会统计这一指标。


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _mirror_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    mirrored = tuple(sorted((int(r), int(w) - 1 - int(c)) for r, c in coords))
    return int(h), int(w), mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of a symmetric frame using shape-matched scattered fragments.

    Input structure:
    - bg = most common color.
    - One large monochrome frame already sits in final position; it is LR-symmetric and encloses bg holes.
    - Outside the frame bbox are scattered monochrome fragments; each fragment matches one hole component exactly.

    Rule:
    - Detect the unique largest non-bg 8-connected component as the frame and crop to its bbox.
    - Split the bg cells inside that bbox into 4-connected hole components, and extract the outside fragments as 4-connected sprites.
    - For each hole component, find the outside sprite with the same shape and fill the hole with that sprite's color.

    Key insight: the frame removes the old “find the rectangle” step; now the task is to read the hole geometry from the fixed scaffold, then use the outside fragments as shape-coded color labels for those hole slots.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    frame_candidates: list[tuple[int, int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            frame_candidates.append((len(pts), int(col), np.array(pts, dtype=int)))

    assert frame_candidates, "input must contain a non-background frame"
    max_area = max(area for area, _, _ in frame_candidates)
    frame_choices = [(col, pts) for area, col, pts in frame_candidates if area == max_area]
    assert len(frame_choices) == 1, "frame must be the unique largest monochrome component"
    frame_col, frame_pts = frame_choices[0]

    r0, c0, r1, c1 = bbox_coords(frame_pts)
    sub = crop_bbox(g, (r0, c0, r1, c1))
    frame_mask = sub == int(frame_col)
    hole_mask = sub == int(bg)

    assert np.all(frame_mask | hole_mask), "frame bbox may only contain frame color and background"
    assert hole_mask.any(), "frame must contain at least one hole cell"
    assert not hole_mask[0].any() and not hole_mask[-1].any()
    assert not hole_mask[:, 0].any() and not hole_mask[:, -1].any()
    assert np.array_equal(frame_mask, frame_mask[:, ::-1]), "frame mask must be LR-symmetric"
    assert np.array_equal(hole_mask, hole_mask[:, ::-1]), "hole mask must be LR-symmetric"

    holes: list[tuple[tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray]] = []
    for pts in components_mask(hole_mask, connectivity=4):
        arr = np.array(pts, dtype=int)
        holes.append((_piece_sig(arr), arr))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    pieces_by_key: dict[tuple[int, int, tuple[tuple[int, int], ...]], Counter] = {}
    for col in sorted(set(map(int, np.unique(outside))) - {bg}):
        for pts in components_mask(outside == col, connectivity=4):
            key = _piece_sig(np.array(pts, dtype=int))
            bucket = pieces_by_key.setdefault(key, Counter())
            bucket[int(col)] += 1

    assert pieces_by_key, "input must contain scattered fill pieces outside the frame"

    out = np.full(sub.shape, int(frame_col), dtype=int)
    filled_area = 0
    for key, pts in holes:
        cmap = pieces_by_key.get(key)
        assert cmap is not None, "every hole shape must have a matching outside piece"
        fill_col = cmap.most_common(1)[0][0]
        out[pts[:, 0], pts[:, 1]] = int(fill_col)
        filled_area += len(pts)

    assert filled_area == int(hole_mask.sum())
    assert np.array_equal(out, out[:, ::-1]), "filled result must remain LR-symmetric"
    return out.astype(int)


# --------------- generator (D3 hole-fill) ---------------

_OUT_H_CHOICES = [8]
_OUT_W = 7

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_FILL_COLOR_MIN = 3
_FILL_COLOR_MAX = 5

_MAX_PIECE_AREA = 8
_PIECE_COUNT_MIN = 4
_PIECE_COUNT_MAX = 9

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_FRAME_RETRIES = 120
_GEN_MAX_RETRIES = 400


def _zigzag_wall(H: int) -> tuple[set[tuple[int, int]], set[int]]:
    cells: set[tuple[int, int]] = set()
    visited_cols: set[int] = set()
    c = random.randint(1, 2)
    for r in range(1, H - 1):
        cells.add((r, c))
        visited_cols.add(c)
        if r == H - 2:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            nc = 3 - c
            cells.add((r, nc))
            visited_cols.add(nc)
            c = nc
        elif move == "diag":
            c = 3 - c
    return cells, visited_cols


def _make_frame_mask(H: int, W: int) -> np.ndarray | None:
    for _ in range(_FRAME_RETRIES):
        frame = np.zeros((H, W), dtype=bool)
        frame[0, :] = True
        frame[H - 1, :] = True
        frame[:, 0] = True
        frame[:, W - 1] = True

        wall_left, used_cols = _zigzag_wall(H)
        if len(used_cols) < 2:
            continue

        for r, c in wall_left:
            frame[r, c] = True
            frame[r, W - 1 - c] = True

        hole_mask = ~frame
        if hole_mask[0].any() or hole_mask[-1].any() or hole_mask[:, 0].any() or hole_mask[:, -1].any():
            continue
        if not np.array_equal(frame, frame[:, ::-1]):
            continue

        holes = components_mask(hole_mask, connectivity=4)
        if not (_PIECE_COUNT_MIN <= len(holes) <= _PIECE_COUNT_MAX):
            continue
        if any(len(hole) > _MAX_PIECE_AREA for hole in holes):
            continue
        if len(holes) < _FILL_COLOR_MIN:
            continue
        return frame
    return None


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, int]:
    frame_mask = _make_frame_mask(H, W)
    if frame_mask is None:
        return None, 0

    hole_mask = ~frame_mask
    holes = [np.array(h, dtype=int) for h in components_mask(hole_mask, connectivity=4)]
    if not (_PIECE_COUNT_MIN <= len(holes) <= _PIECE_COUNT_MAX):
        return None, 0
    if any(len(hole) > _MAX_PIECE_AREA for hole in holes):
        return None, 0

    shape_classes: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[np.ndarray]] = {}
    for hole in holes:
        sig = _piece_sig(hole)
        canon = min(sig, _mirror_sig(sig))
        shape_classes.setdefault(canon, []).append(hole)

    if len(shape_classes) < _FILL_COLOR_MIN:
        return None, 0

    n_fill_colors = random.randint(_FILL_COLOR_MIN, min(_FILL_COLOR_MAX, len(shape_classes)))
    palette = list(range(1, n_fill_colors + 1))
    class_keys = list(shape_classes.keys())
    random.shuffle(class_keys)

    class_colors: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for i, key in enumerate(class_keys):
        class_colors[key] = palette[i % len(palette)]

    out = np.zeros((H, W), dtype=int)
    frame_color = 0
    out[frame_mask] = frame_color
    for key, comps in shape_classes.items():
        col = int(class_colors[key])
        for pts in comps:
            out[pts[:, 0], pts[:, 1]] = col

    if np.any(hole_mask & (out == frame_color)):
        return None, 0
    if not np.array_equal(out, out[:, ::-1]):
        return None, 0
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, 0

    return out, frame_color


def _remap_colors(out: np.ndarray, frame_color: int) -> tuple[np.ndarray | None, int, int]:
    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, 0
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    return np.vectorize(lambda x: cmap[int(x)])(out), int(bg), int(cmap[frame_color])


def _extract_piece_patches(out: np.ndarray, bg: int, frame_col: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col == frame_col:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter_input(out: np.ndarray, bg: int, frame_col: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_piece_patches(out, bg, frame_col):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    region[:] = bg
    frame_mask = out == int(frame_col)
    region[frame_mask] = out[frame_mask]
    return canvas


def _non_frame_piece_count(out: np.ndarray, frame_col: int) -> int:
    return sum(
        len(components_mask(out == int(col), connectivity=4))
        for col in np.unique(out)
        if int(col) != int(frame_col)
    )


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_OUT_H_CHOICES)
    W = _OUT_W

    abstract_out, frame_abstract = _build_output(H, W)
    if abstract_out is None:
        return None

    out, bg, frame_col = _remap_colors(abstract_out, frame_abstract)
    if out is None:
        return None

    pc = _non_frame_piece_count(out, frame_col)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None

    try:
        inp = _scatter_input(out, bg, frame_col)
        got = main(inp)
    except Exception:
        return None

    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D3 input")
