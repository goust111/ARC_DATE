# Variant of e8686506
# Dimension: D13 — 背景与前景的关系 (meta)
# Change: 输入使用单一背景色，所有非背景像素都是有效碎片的一部分... → 双色背景（分区背景）
#
# Design analysis
# ----------------
# spec:
#   invariant_output:
#     - 输出仍然是一个严格左右镜像的 HxW 稠密矩形。
#     - HxW 仍由总碎片面积的“最接近正方形且唯一的竖向因子对”确定。
#     - 原题的框架碎片、角点/边条锚点、同色碎片不可 4 邻接等约束继续保留。
#   d13_change:
#     - 输入不再只有一个背景色，而是整张画布按竖直分界线切成左右两个背景区域。
#     - 左区和右区分别用两种不同背景色填充；碎片覆盖在各自区域之上。
#     - 非框架碎片按其在输出中的左右归属散布：左半来源的碎片进左背景区，右半来源的碎片进右背景区；
#       跨中心列的自对称碎片在两区之间分配。这样人眼能看到“背景分区”和“碎片归属”。
#   concrete_behavior_trace:
#     - 原题流程：`get_bg()` 取全局最多颜色，剩余颜色全部当碎片。
#     - 本变体里，如果仍这么做，那么较少的那种背景色会被误识别成一大块“碎片”，后续面积、框架检测都会崩。
#     - 新流程：先从保持纯背景的顶行/底行读出左右背景色与分界列，再只把“位于左区且不等于左背景色”、
#       “位于右区且不等于右背景色”的像素当作前景碎片。
#   complexity_calibration:
#     - 只引入一种稳定、可见的背景分区模式：竖直二分。避免出现多种分区几何导致 2-3 个样例无法归纳。
#     - 两种背景色都必须大面积可见，且两区都要放到碎片，防止退化成“只是换了一个角落颜色”。
#     - 输出规则不改，避免把 D13 和形状变换、颜色映射等别的维度混在一起。
#   implicit_assumptions_audit:
#     - 原 `bg = get_bg(g)` 依赖“背景唯一且最多”；在本变体中失效，必须改成分区背景检测。
#     - 原 `for col in unique(g) if col != bg` 依赖“除 bg 外全是碎片色”；现在要改成“排除两个区域背景色”。
#     - 原 8 连通框架搜索默认所有非 bg 颜色都是真碎片；现在若不先过滤右侧背景色，会把整块背景并入候选。
#     - 原 generator 默认整张 canvas 统一填一个 bg；现在必须改成先铺双色背景，再分别在左右子画布中找空位。
#   placement_legality:
#     - 外边框保留 1 像素纯背景带，因此顶行/底行一定完整暴露分区背景，solver 可稳定检测分界。
#     - 非框架碎片只在所属区域内散布，且仍使用 `random_free_location(..., padding=1)` 保证不重叠且保持间隔。
#     - 框架碎片组仍作为整体放置，bbox 位置合法且不越界；非框架碎片仍被禁止放进框架 bbox。
#   visual_observability:
#     - 人类会直接看到输入被两种背景色竖向切成两半，而且两边都各自散落了拼图块。
#     - 输出没有背景，只保留拼装后的矩形，因此规则可理解为“忽略两块背景，把两边碎片共同拼回去”。
#   cleanliness_and_color_inference:
#     - 背景色绝不复用为碎片色，避免“这个颜色到底是背景还是碎片”。
#     - 只有两种背景色，且它们形成连续大区域，视觉上整洁，不会与碎片层混杂成噪声。

from __future__ import annotations

import sys
import random
from collections import Counter

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _detect_partition_background(g: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    H, W = g.shape
    assert H >= 3 and W >= 3, "grid must be large enough for a visible 2-zone background"

    top = list(map(int, g[0]))
    bottom = list(map(int, g[H - 1]))
    left_bg = int(top[0])
    right_bg = int(top[W - 1])
    assert left_bg != right_bg, "dual-background variant requires two distinct background colors"

    changes = [c for c in range(1, W) if int(top[c - 1]) != int(top[c])]
    assert len(changes) == 1, "top border must reveal a single vertical background split"
    split = int(changes[0])
    assert 1 <= split <= W - 1, "background split must separate the canvas into two non-empty zones"
    assert all(int(v) == left_bg for v in top[:split]), "top-left zone must be uniformly background-colored"
    assert all(int(v) == right_bg for v in top[split:]), "top-right zone must be uniformly background-colored"
    assert all(int(v) == left_bg for v in bottom[:split]), "bottom-left zone must preserve the left background color"
    assert all(int(v) == right_bg for v in bottom[split:]), "bottom-right zone must preserve the right background color"

    bg_mask = np.zeros_like(g, dtype=bool)
    bg_mask[:, :split] = (g[:, :split] == left_bg)
    bg_mask[:, split:] = (g[:, split:] == right_bg)
    return split, left_bg, right_bg, bg_mask


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into one dense rectangle with exact left-right symmetry.

    Input structure:
    - The canvas background is split into two vertical regions, each with its own background color.
    - Colored fragments are scattered across both regions; 1-2 8-connected frame pieces still preserve the output bbox.
    - Optional anchors keep their old semantics: four singletons can mark corners, long bars can mark top/bottom edges, and a paired vertical line color can mark left/right edges.

    Rule:
    - Detect the two background zones first, then ignore only the pixels that match the proper background color of their zone.
    - Treat every remaining monochromatic 4-connected component as a puzzle piece and infer the output size from total foreground area.
    - Assemble all pieces into a portrait rectangle whose bitmap is exactly LR-mirrored and whose same-color pieces do not become 4-neighbors.

    Key insight:
    - The variant changes foreground extraction, not the assembly rule: the solver must separate two large background slabs from true fragments before running the original symmetry-and-frame logic.
    """
    g = np.asarray(grid, dtype=int)
    _split, _left_bg, _right_bg, bg_mask = _detect_partition_background(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    piece_colors = sorted(set(map(int, np.unique(g[~bg_mask]))))
    for col in piece_colors:
        comps = components_mask((g == col) & (~bg_mask), connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in piece_colors:
        for pts in components_mask((g == col) & (~bg_mask), connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (o5x, D13 variant) ---------------

_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_OUTER_BORDER = 1

_GEN_MAX_RETRIES = 1000


def _valid_dims():
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid: np.ndarray, H: int, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = fc_set | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)
    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)

    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, 0, set()

    bg_left, bg_right = random.sample(list(range(10)), 2)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c not in {bg_left, bg_right}]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return remapped, int(bg_left), int(bg_right), real_fcs


def _classify_patch_side(pts: np.ndarray, W: int) -> str:
    center = W // 2
    cols = [int(c) for _, c in pts]
    if max(cols) < center:
        return "left"
    if min(cols) > center:
        return "right"
    return "center"


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[dict]:
    patches = []
    W = out.shape[1]
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(
                {
                    "patch": patch,
                    "side": _classify_patch_side(pts, W),
                    "area": int((patch != bg).sum()),
                }
            )
    random.shuffle(patches)
    patches.sort(key=lambda item: (-item["area"], random.random()))
    return patches


def _place_patch_in_zone(canvas: np.ndarray, patch: np.ndarray, bg: int, c_start: int, c_end: int) -> None:
    sub = canvas[:, c_start:c_end]
    r, c = random_free_location(
        sub,
        patch,
        transparent=bg,
        border=_OUTER_BORDER,
        padding=_SCATTER_PADDING,
        padding_connectivity=8,
    )
    paste(canvas, patch, r, c_start + c, transparent=bg)


def _scatter(out: np.ndarray, bg_left: int, bg_right: int, fcs: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    split = cols // 2
    assert 2 <= split <= cols - 2, "canvas must have room for both background zones"

    canvas = np.full((rows, cols), bg_right, dtype=int)
    canvas[:, :split] = bg_left
    oh, ow = out.shape

    min_r = _OUTER_BORDER
    max_r = rows - oh - _OUTER_BORDER
    off_r = random.randint(min_r, max_r)

    # Keep the frame group centered around the zone split so both background slabs remain visible.
    off_c = split - (ow // 2) - 1
    if off_c < _OUTER_BORDER or off_c + ow > cols - _OUTER_BORDER:
        raise ValueError("frame bbox cannot straddle the background split on this canvas")

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    center_toggle = False
    for item in _extract_non_frame_patches(out, bg_left, fcs):
        patch = item["patch"]
        side = item["side"]
        if side == "left":
            _place_patch_in_zone(canvas, patch, bg_left, 0, split)
        elif side == "right":
            patch = np.where(patch == bg_left, bg_right, patch)
            _place_patch_in_zone(canvas, patch, bg_right, split, cols)
        else:
            if center_toggle:
                patch = np.where(patch == bg_left, bg_right, patch)
                _place_patch_in_zone(canvas, patch, bg_right, split, cols)
            else:
                _place_patch_in_zone(canvas, patch, bg_left, 0, split)
            center_toggle = not center_toggle

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    left_cols = max(0, min(split - off_c, ow))
    if left_cols > 0:
        bbox_region[:, :left_cols] = bg_left
    if left_cols < ow:
        bbox_region[:, left_cols:] = bg_right
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg_left, bg_right, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None
    try:
        inp = _scatter(out, bg_left, bg_right, fcs)
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
    raise RuntimeError("failed to generate a valid e8686506 D13 input")
