# Combo variant of e8686506 (index 23)
# combo_spec: D2/3+D8/3
# Name: 干扰像素+干扰碎片

# Design analysis
# yaml:
#   variant_goal:
#     keep:
#       - 输出仍是由真实碎片拼出的致密矩形，且必须严格左右镜像。
#       - 原题的框架锚点、面积分解、同色不同块不可 4 邻接等约束继续保留。
#     change:
#       - 输入同时混入两类噪声：
#         1) 位于框架 bbox 严格内侧的孤立干扰像素；
#         2) 位于框外、完全不参与拼装的 1-2 个干扰碎片。
#       - solver 必须先删单点噪声，再从剩余碎片里排除结构性噪声。
#   concrete_behavior:
#     visible_layout:
#       - 框架仍以 1-2 个 8 连通片段原位保留，因此目标矩形 bbox 肉眼可见。
#       - 真实非框架碎片散落在框外。
#       - 干扰像素只出现在框内背景位置，彼此不相邻，不覆盖框架。
#       - 干扰碎片也散落在框外，看起来和真实碎片同属“外部散件”区域。
#     small_trace:
#       - 例：真实输出为 8x5，目标面积 40。
#       - 原题：所有非背景块总面积也是 40，直接因式分解即可得到 8x5。
#       - 只有 D2 时：若框内多了 2 个孤立点，原总面积会变成 42；新 solver 先读框架 bbox=8x5，
#         再删框内单点，把面积拉回 40。
#       - 融合后：若再额外加入一个面积 3 的外部干扰块，则删单点后剩余面积仍是 43，
#         还必须继续做“删 1-2 个外部碎片”的子集筛选，才能回到 40 并完成拼装。
#   complexity_calibration:
#     why_two_stage_is_visible:
#       - 框内杂点与框外碎片分区明确，人类能看出有两种不同噪声角色。
#       - 干扰像素颜色复用真实填充色，避免退化成“忽略固定颜色”。
#       - 干扰碎片使用输出中不会出现的颜色，确保“有整块外部碎片被丢弃”在样例里足够显眼。
#     bounded_search:
#       - 候选框架仍只需枚举 1 或 2 个 8 连通分量组合。
#       - 外部干扰碎片最多 2 个，因此子集筛选只需测试很小的组合空间。
#       - 面积守恒先裁掉大部分子集，最后再用形状放置与对称性检查验证。
#   implicit_assumptions_audited:
#     old_assumptions_broken:
#       - 原 solver 假设 total_area 来自全部非背景像素；融合后被两类噪声同时破坏。
#       - 原 solver 直接按全部分量统计 corner/bar/vline 锚点；若不先删框内单点，会污染这些统计。
#       - 原 solver 假设每个 4 连通分量都参与拼装；融合后对外部干扰碎片不成立。
#     line_level_fix:
#       - 先枚举框架候选并读出 bbox，避免被噪声面积误导。
#       - 只把 bbox 严格内侧的 singleton 当作单点噪声丢弃，保留边界上的真实 1x1 片。
#       - 在剩余碎片上用面积方程筛子集，再用 req 约束 + 镜像拼装验证形状匹配。
#   placement_legality:
#     generator_guarantees:
#       - 真实框架 bbox 整块预留，真实碎片与干扰碎片都只散落在框外，互不重叠且带 8 邻域 padding。
#       - 干扰像素只写入框内的非框架背景格，因此不会覆盖框架，也不会与外部碎片相连。
#       - 干扰像素彼此以 Chebyshev 距离 > 1 分开，因此确实保持为孤立单点。
#   observability_cleanliness:
#     human_readability:
#       - 2-3 组样例就能看到：框里有散点要忽略，框外也有整块碎片要忽略。
#       - 规则不依赖颜色编号大小、扫描顺序等人类不可见因素。
#       - 框架、噪点、外部散件各在不同区域，画面不会叠在一起变脏。
#   post_change_solver_assumptions:
#     new_assumptions:
#       - 单点噪声只会是候选框架 bbox 严格内侧的 1x1 分量。
#       - 结构性噪声只会是框外的 1-2 个完整碎片。
#       - 正确子集在面积上恰好填满框架 bbox，并能通过原题的镜像拼装传播。
#     enforcement:
#       - generate_input() 明确按上述方式构造，因此 solver 与 generator 一致。

import random
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_UTILS = _ROOT.parents[4] / "solvers"
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """
    Build per-cell color requirements (req[r,c] == -1 means unconstrained).

    The original optional corner/bar/vertical markers are still available, but only
    after both classes of noise have been removed from the working piece set.
    """
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
                    bar_colors.append((r0, int(col)))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            req[0, 1 : W - 1] = bar_colors[0][1]
            req[H - 1, 1 : W - 1] = bar_colors[-1][1]

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((int(col), h0))
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
            board[r0 + dr, c0 + dc] = col

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
            assert places, "assembly dead end: a piece has no valid placement"
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


def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _frame_candidates(g: np.ndarray, bg: int) -> list[dict]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), np.array(pts, dtype=int)))

    candidates = []
    seen = set()
    for i, (c1, p1) in enumerate(comps8_all):
        groups = [[(c1, p1)]]
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            groups.append([(c1, p1), (c2, p2)])
        for parts in groups:
            union = np.concatenate([p for _, p in parts])
            r0, c0, r1, c1b = bbox_coords(union)
            H, W = r1 - r0 + 1, c1b - c0 + 1
            if H < W:
                continue
            if not _is_lr_symmetric(parts, H, W):
                continue
            cells = frozenset((int(r), int(c)) for _, pts in parts for r, c in pts)
            if cells in seen:
                continue
            seen.add(cells)
            candidates.append(
                {
                    "parts": parts,
                    "cells": cells,
                    "bbox": (int(r0), int(c0), int(r1), int(c1b)),
                    "H": int(H),
                    "W": int(W),
                }
            )
    candidates.sort(key=lambda cand: (-(cand["H"] * cand["W"]), len(cand["parts"])))
    return candidates


def _extract_piece_records(g: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    pid = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            pieces.append(
                {
                    "id": pid,
                    "col": int(col),
                    "pts": np.array(pts, dtype=int),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "area": int(len(coords)),
                }
            )
            pid += 1
    return pieces


def _group_types(selected: list[dict]) -> list[dict]:
    ctr = Counter((int(p["col"]), p["coords"]) for p in selected)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )
    return types


def _comps_by_color(selected: list[dict]) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for rec in selected:
        out.setdefault(int(rec["col"]), []).append(rec["pts"])
    return out


def _is_inner_noise(rec: dict, bbox: tuple[int, int, int, int]) -> bool:
    if int(rec["area"]) != 1:
        return False
    r0, c0, r1, c1 = bbox
    rr, cc = map(int, rec["pts"][0])
    return r0 < rr < r1 and c0 < cc < c1


def _candidate_subset_solve(
    filtered_pieces: list[dict],
    selected_ids: set[int],
    candidate: dict,
) -> np.ndarray:
    H, W = int(candidate["H"]), int(candidate["W"])
    selected = [p for p in filtered_pieces if p["id"] in selected_ids]
    comps = _comps_by_color(selected)
    req = _build_requirements(H, W, comps)

    all_frame_pts = np.concatenate([pts for _, pts in candidate["parts"]])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in candidate["parts"]:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if req[or_, oc] not in (-1, int(col)):
                raise AssertionError("frame cells conflict with inferred requirements")
            req[or_, oc] = int(col)

    color_piece_count = Counter(int(p["col"]) for p in selected)
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return _assemble(_group_types(selected), H, W, req, disallow_touch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true fragments into the symmetric rectangle after removing two layers of noise.

    Input structure:
    - The most common color is background; the other cells contain real monochrome fragments, a visible in-place frame outline, a few isolated dots inside that outline, and 1-2 extra outer fragments that do not belong to the answer.
    - The frame still appears as 1-2 left-right symmetric 8-connected parts whose union bbox equals the target rectangle, while the genuine non-frame pieces and the distractor fragments are scattered outside that box.

    Rule:
    - First enumerate frame candidates to read off the target bbox.
    - For each candidate, discard only the singleton components strictly inside the bbox; those are the pixel-noise layer.
    - Then keep the unique subset of the remaining pieces whose total area matches the frame-sized rectangle and whose shapes can be assembled into a dense left-right mirror with the original anchor constraints.

    Key insight:
    - The frame stabilizes the search order: it tells you where the rectangle is before area counting.
    - After inner singletons are stripped, the puzzle is still overfull, so one more frame-guided subset test is required; the correct subset is the one that both matches the target area and survives the original symmetry-driven placement propagation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    all_pieces = _extract_piece_records(g, bg)
    assert all_pieces, "grid must contain non-background pieces"

    for cand in _frame_candidates(g, bg):
        H, W = int(cand["H"]), int(cand["W"])
        target_area = H * W
        dims = [(h, w) for (h, w) in _factors(target_area) if h >= w]
        if not dims:
            continue
        best_gap = min(abs(h - w) for h, w in dims)
        best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best_dims) != 1 or best_dims[0] != (H, W):
            continue

        filtered_pieces = [rec for rec in all_pieces if not _is_inner_noise(rec, cand["bbox"])]
        frame_cells = cand["cells"]
        frame_ids = {
            int(rec["id"])
            for rec in filtered_pieces
            if all((int(r), int(c)) in frame_cells for r, c in rec["pts"])
        }
        if not frame_ids:
            continue

        frame_area = sum(rec["area"] for rec in filtered_pieces if rec["id"] in frame_ids)
        if frame_area > target_area:
            continue

        scatter = [rec for rec in filtered_pieces if rec["id"] not in frame_ids]
        required_scatter_area = target_area - frame_area
        extra_area = sum(rec["area"] for rec in scatter) - required_scatter_area
        if extra_area < 0:
            continue

        for drop_n in range(0, min(2, len(scatter)) + 1):
            for drop_group in combinations(scatter, drop_n):
                if sum(rec["area"] for rec in drop_group) != extra_area:
                    continue
                drop_ids = {int(rec["id"]) for rec in drop_group}
                selected_ids = frame_ids | {int(rec["id"]) for rec in scatter if int(rec["id"]) not in drop_ids}
                if sum(rec["area"] for rec in filtered_pieces if rec["id"] in selected_ids) != target_area:
                    continue
                try:
                    return _candidate_subset_solve(filtered_pieces, selected_ids, cand).astype(int)
                except Exception:
                    continue

    raise AssertionError("no valid two-stage filtered assembly found")


# --------------- generator ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_NOISE_MIN = 2
_NOISE_MAX = 3
_DISTRACTOR_MIN = 1
_DISTRACTOR_MAX = 2

# === CFG: Retry ===
_GEN_MAX_RETRIES = 900


def _valid_dims() -> list[tuple[int, int]]:
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

_DISTRACTOR_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]


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


def _fill_holes(
    grid: np.ndarray,
    H: int,
    W: int,
    frame1: int,
    frame2: int,
    split: int,
    fill_pool: list[int],
) -> None:
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

    half = W // 2
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


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return remapped, bg, real_fcs


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    arr = np.array(shape, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _make_distractors(out: np.ndarray, bg: int, count: int) -> list[np.ndarray]:
    used_colors = set(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg and c not in used_colors]
    if len(available) < count:
        raise ValueError("not enough unused colors for distractors")
    random.shuffle(available)
    distractors = []
    for i in range(count):
        shape = random.choice(_DISTRACTOR_SHAPES)
        distractors.append(_shape_to_patch(shape, available[i], bg))
    return distractors


def _add_noise(
    canvas: np.ndarray,
    out: np.ndarray,
    fcs: set[int],
    off_r: int,
    off_c: int,
) -> int:
    oh, ow = out.shape
    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)

    fill_colors = [int(col) for col in np.unique(out) if int(col) not in fcs]
    if not fill_colors:
        raise ValueError("need non-frame colors for singleton noise")

    candidates = [
        (off_r + r, off_c + c)
        for r in range(1, oh - 1)
        for c in range(1, ow - 1)
        if not frame_mask[r, c]
    ]
    random.shuffle(candidates)
    if len(candidates) < _NOISE_MIN:
        raise ValueError("not enough interior background cells for noise")

    target = random.randint(_NOISE_MIN, min(_NOISE_MAX, len(candidates)))
    chosen = []
    for rr, cc in candidates:
        if all(max(abs(rr - pr), abs(cc - pc)) > 1 for pr, pc in chosen):
            chosen.append((rr, cc))
            if len(chosen) == target:
                break
    if len(chosen) < target:
        raise ValueError("failed to place isolated singleton noise")

    for rr, cc in chosen:
        canvas[rr, cc] = random.choice(fill_colors)
    return len(chosen)


def _scatter(
    out: np.ndarray,
    bg: int,
    fcs: set[int],
    distractors: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    for patch in distractors:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    noise_count = _add_noise(canvas, out, fcs, off_r, off_c)
    return canvas, noise_count


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == int(col), connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None

    piece_count = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == int(col), connectivity=4)
    ):
        return None

    try:
        n_distractors = random.randint(_DISTRACTOR_MIN, _DISTRACTOR_MAX)
        distractors = _make_distractors(out, bg, n_distractors)
        inp, noise_count = _scatter(out, bg, fcs, distractors)
        got = main(inp)
    except Exception:
        return None

    if noise_count < _NOISE_MIN:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
