# Combo variant of e8686506 (index 49)
# combo_spec: D2/3+D9/0
# Name: 碎片紧密排列+干扰像素
#
# Design analysis
# yaml:
#   variant_goal:
#     keep:
#       - 输出仍是把真实碎片拼成一个严格左右镜像的致密矩形。
#       - 1-2 个框架片段仍作为稳定锚点，面积分解、镜像约束、同色不同块不可 4 连通接触都保留。
#     change:
#       - 非框架碎片不再随机散开，而是被压进一个紧凑托盘，piece bbox 之间只有 1 像素背景缝。
#       - 额外加入 1-3 个孤立干扰像素，专门落在这些狭窄缝隙里，颜色复用真实碎片颜色池。
#   concrete_behavior:
#     input_regions:
#       frame_box:
#         原题可见的 LR 对称框架仍按输出相对位置原样摆放，继续提供 HxW 锚点。
#       tight_tray:
#         其余真实碎片在框架外组成一个紧凑排布区，左右或上下不重叠，piece bbox 之间恰有 1 格背景带。
#       noise:
#         干扰像素写入托盘内部的背景缝隙格，四邻和八邻颜色都避开同色，保证它们是孤立单点而非真实块延伸。
#     visible_effect:
#       人眼会看到“一个清晰框架 + 一团挤得很近的小块 + 缝里夹着零散杂点”，边界判断明显比原题更难。
#   flow_trace:
#     original_step:
#       原 solver 把所有非背景 4 连通分量都当真块，并直接用总面积反推输出尺寸。
#     combo_step:
#       新 solver 先像 D2 那样从 8 连通框架候选反推 HxW，再把所有单点组件视为“可删候选”；
#       由于噪声只增加面积 1，新 solver 只需删去恰好 excess_area 个单点，并检查哪组删除能恢复
#       合法的锚点统计和传播拼装。
#     difference:
#       面积不再先验可信；真正可靠的是框架 bbox。紧密排布增加了单点附近的视觉混淆，但不改变真实块的
#       4 连通分量结构，所以“框架先行 + 单点子集过滤”能同时处理两种改动。
#   complexity_calibration:
#     why_tray_not_full_overlap:
#       如果把真实块完全堆叠或留太大间隔，要么非法要么退化回原题；1 像素缝能稳定制造边界压力，
#       但仍保留可分割的背景带。
#     why_noise_in_gaps:
#       只在托盘缝隙里加噪声，才能和 D9 的“紧密排列”形成协同，而不是简单并列两个独立现象。
#     why_subset_search_is_safe:
#       噪声固定为少量单点，故 excess_area 很小；即使真实任务里还有角点单块，枚举“删哪几个单点”
#       的组合规模仍可控。
#   implicit_assumptions_audited:
#     - 旧代码假设 total_area 来自全部非背景像素；融合后被噪声破坏，已改成框架候选驱动。
#     - 旧代码默认所有单点都是真实 piece；融合后不成立，改为只删除数量恰好等于 excess_area 的单点候选。
#     - 旧代码的角点/边条统计依赖按颜色计数组件；若删错单点，这些统计会失真，因此 solver 必须把
#       req 构造和拼装验证放在同一个候选检查链里。
#     - 旧代码 generator 依赖 random_free_location 的大间隔散布；融合后显式改成紧凑 shelf packing，
#       否则 D9 不会真正发生。
#   placement_legality:
#     - 托盘使用 piece bbox 做行式紧凑打包，横竖都留 1 格空带，因此真块之间不重叠、不越界，也不会 8 连到一起。
#     - 干扰像素只写在托盘内部背景格，且彼此 Chebyshev 距离 > 1，不会合并成更大噪声团。
#     - 噪声颜色避开八邻真实颜色，所以不会和任何真实同色块在 8 连通或 4 连通下粘连。
#   observability_cleanliness:
#     - 2-3 个样例就能看出：框架决定输出尺寸，托盘里挤着很多真块，而缝隙里的零星单点不参与拼装。
#     - 噪声数量保持很少，且始终附着在“1 像素缝”这个可见结构上，不依赖颜色编号等不可视规则。
#   post_change_solver_assumptions:
#     - 只有单点组件可能是噪声，且噪声总数等于 total_non_bg_area - frame_area。
#     - 至少存在一个 1-2 片组成的 LR 对称框架候选。
#     - 真实 piece 的 4 连通结构与原题相同；紧密排列只改变空间布局，不改变 piece 形状。
#     - generate_input() 逐条保证以上假设成立。

import random
from collections import Counter
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[np.ndarray]],
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
            req[0, 1 : W - 1] = bar_colors[0][1]
            req[H - 1, 1 : W - 1] = bar_colors[-1][1]

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
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in output height"
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


def _enumerate_frame_candidates(comps8_all: list[tuple[int, np.ndarray]]) -> list[dict]:
    candidates = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        h1, w1 = r1 - r0 + 1, c1b - c0 + 1
        if h1 >= w1 and _is_lr_symmetric([(c1, p1)], h1, w1):
            candidates.append({"parts": [(c1, p1)], "bbox": (r0, c0, r1, c1b), "H": h1, "W": w1})
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            h, w = ur1 - ur0 + 1, uc1 - uc0 + 1
            if h < w:
                continue
            parts = [(c1, p1), (c2, p2)]
            if _is_lr_symmetric(parts, h, w):
                candidates.append({"parts": parts, "bbox": (ur0, uc0, ur1, uc1), "H": h, "W": w})
    candidates.sort(key=lambda cand: (-(cand["H"] * cand["W"]), len(cand["parts"])))
    return candidates


def _try_kept_components(
    kept: list[tuple[int, np.ndarray]],
    candidate: dict,
) -> np.ndarray | None:
    H, W = int(candidate["H"]), int(candidate["W"])
    if not kept:
        return None

    pieces = []
    comps_by_color: dict[int, list[np.ndarray]] = {}
    color_piece_count: dict[int, int] = {}
    for col, pts in kept:
        comps_by_color.setdefault(int(col), []).append(pts)
        color_piece_count[int(col)] = color_piece_count.get(int(col), 0) + 1
        _, _, coords = _piece_sig(pts)
        pieces.append((int(col), coords))

    ctr = Counter(pieces)
    total_area = sum(len(coords) * cnt for (_, coords), cnt in ctr.items())
    if total_area != H * W:
        return None

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    if not dims:
        return None
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    if len(best_dims) != 1 or best_dims[0] != (H, W):
        return None

    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    req = _build_requirements(H, W, comps_by_color)
    all_frame_pts = np.concatenate([p for _, p in candidate["parts"]])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in candidate["parts"]:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if req[or_, oc] not in (-1, int(col)):
                return None
            req[or_, oc] = int(col)

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    try:
        return _assemble(types, H, W, req, disallow_touch)
    except Exception:
        return None


def _solve_from_frame_candidate(
    comps4_all: list[tuple[int, np.ndarray]],
    candidate: dict,
) -> np.ndarray | None:
    H, W = int(candidate["H"]), int(candidate["W"])
    total_area = sum(len(pts) for _, pts in comps4_all)
    excess_area = total_area - H * W
    if excess_area < 0:
        return None

    frame_cells = {
        (int(r), int(c))
        for _, pts in candidate["parts"]
        for r, c in pts
    }
    singleton_indices = []
    priority_indices = []
    r0, c0, r1, c1 = candidate["bbox"]
    for idx, (_col, pts) in enumerate(comps4_all):
        if len(pts) != 1:
            continue
        rr, cc = map(int, pts[0])
        if (rr, cc) in frame_cells:
            continue
        singleton_indices.append(idx)
        if r0 < rr < r1 and c0 < cc < c1:
            priority_indices.append(idx)

    if excess_area > len(singleton_indices):
        return None
    if excess_area == 0:
        return _try_kept_components(comps4_all, candidate)

    base_order = priority_indices + [idx for idx in singleton_indices if idx not in set(priority_indices)]
    for subset in combinations(base_order, excess_area):
        removed = set(subset)
        kept = [comp for i, comp in enumerate(comps4_all) if i not in removed]
        sol = _try_kept_components(kept, candidate)
        if sol is not None:
            return sol
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the real fragments into a dense LR-symmetric rectangle while ignoring gap clutter.

    Input structure:
    - The dominant color is background; non-background cells contain the true monochrome fragments plus a few isolated junk pixels.
    - One or two 8-connected frame fragments still sit in their output-relative arrangement and reveal the rectangle size.
    - The remaining real fragments are packed tightly elsewhere with only 1-pixel gaps, and some of those gaps contain singleton interference pixels.

    Rule:
    - Find the frame candidate first and use its bounding box as the output HxW anchor.
    - Treat only singleton components as removable noise candidates, and remove exactly as many as needed to make the remaining total area match that anchored rectangle.
    - Among those candidate removals, keep the one whose surviving pieces satisfy the anchor constraints and assemble into a perfectly left-right symmetric rectangle where distinct same-color pieces never 4-touch.

    Key insight:
    - Tight packing makes local boundaries visually noisy, so raw area and singleton counts cannot be trusted upfront.
    - The frame outline is still globally reliable; once it fixes HxW, the correct noise subset is exactly the one that restores a legal propagation-only assembly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps4_all: list[tuple[int, np.ndarray]] = []
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            comps4_all.append((int(col), pts))
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    for candidate in _enumerate_frame_candidates(comps8_all):
        sol = _solve_from_frame_candidate(comps4_all, candidate)
        if sol is not None:
            return sol.astype(int)

    raise AssertionError("no frame candidate yields a valid tight-pack assembly")


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

# === CFG: Canvas / tight tray ===
_CANVAS_MIN = 14
_CANVAS_MAX = 16
_NOISE_MIN = 1
_NOISE_MAX = 3
_TRAY_MAX_W = 8
_TIGHT_PAIR_MIN = 3

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


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
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
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


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
    patches.sort(key=lambda p: (-(p != bg).sum(), -p.shape[0], -p.shape[1]))
    return patches


def _count_tight_pairs(boxes: list[tuple[int, int, int, int]]) -> int:
    cnt = 0
    for i in range(len(boxes)):
        r0a, c0a, r1a, c1a = boxes[i]
        for j in range(i + 1, len(boxes)):
            r0b, c0b, r1b, c1b = boxes[j]
            col_overlap = max(0, min(c1a, c1b) - max(c0a, c0b) + 1)
            row_overlap = max(0, min(r1a, r1b) - max(r0a, r0b) + 1)
            if col_overlap > 0 and (r0b - r1a == 2 or r0a - r1b == 2):
                cnt += 1
            elif row_overlap > 0 and (c0b - c1a == 2 or c0a - c1b == 2):
                cnt += 1
    return cnt


def _gap_noise_candidates(tray: np.ndarray, bg: int) -> list[tuple[int, int]]:
    occ = tray != bg
    H, W = tray.shape
    out = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if occ[r, c]:
                continue
            north, south = bool(occ[r - 1, c]), bool(occ[r + 1, c])
            west, east = bool(occ[r, c - 1]), bool(occ[r, c + 1])
            if (west and east) or (north and south) or ((north or south) and (west or east)):
                out.append((r, c))
    random.shuffle(out)
    return out


def _build_tight_tray(patches: list[np.ndarray], bg: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int] | None:
    if not patches:
        return None

    max_patch_w = max(int(p.shape[1]) for p in patches)
    upper_w = min(_TRAY_MAX_W, sum(int(p.shape[1]) for p in patches) + len(patches) - 1)
    width_options = list(range(max_patch_w, upper_w + 1))
    random.shuffle(width_options)

    best = None
    for tray_w in width_options:
        placements = []
        x = 0
        y = 0
        row_h = 0
        used_w = 0
        ok = True
        for patch in patches:
            h, w = map(int, patch.shape)
            if x > 0 and x + w > tray_w:
                y += row_h + 1
                x = 0
                row_h = 0
            if w > tray_w:
                ok = False
                break
            placements.append((patch, y, x))
            used_w = max(used_w, x + w)
            row_h = max(row_h, h)
            x += w + 1
        if not ok:
            continue
        used_h = y + row_h
        tray = np.full((used_h, used_w), bg, dtype=int)
        boxes = []
        for patch, r, c in placements:
            h, w = map(int, patch.shape)
            paste(tray, patch, r, c, transparent=bg)
            boxes.append((r, c, r + h - 1, c + w - 1))
        tight_pairs = _count_tight_pairs(boxes)
        candidate_count = len(_gap_noise_candidates(tray, bg))
        score = (used_h * used_w, -tight_pairs, -candidate_count, abs(used_w - used_h))
        if candidate_count >= _NOISE_MIN and tight_pairs >= _TIGHT_PAIR_MIN:
            if best is None or score < best[0]:
                best = (score, tray, boxes, tight_pairs)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _add_gap_noise(
    canvas: np.ndarray,
    tray: np.ndarray,
    tray_r: int,
    tray_c: int,
    bg: int,
    palette: list[int],
) -> list[tuple[int, int]]:
    candidates = _gap_noise_candidates(tray, bg)
    if len(candidates) < _NOISE_MIN:
        raise ValueError("not enough tray gap cells for noise")

    target = random.randint(_NOISE_MIN, min(_NOISE_MAX, len(candidates)))
    chosen: list[tuple[int, int]] = []
    local_occ = tray != bg
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        neigh_colors = set()
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not local_occ[rr, cc]:
                    continue
                neigh_colors.add(int(tray[rr, cc]))
        opts = [col for col in palette if col not in neigh_colors]
        if not opts:
            continue
        chosen.append((r, c))
        canvas[tray_r + r, tray_c + c] = random.choice(opts)
        if len(chosen) == target:
            break
    if len(chosen) < _NOISE_MIN:
        raise ValueError("failed to place required noise")
    return [(tray_r + r, tray_c + c) for r, c in chosen]


def _scatter_tight(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, dict]:
    oh, ow = out.shape
    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    frame_patch = np.full((oh, ow), bg, dtype=int)
    frame_patch[frame_mask] = out[frame_mask]

    patches = _extract_non_frame_patches(out, bg, fcs)
    built = _build_tight_tray(patches, bg)
    if built is None:
        raise ValueError("failed to build tight tray")
    tray, boxes, tight_pairs = built
    th, tw = tray.shape

    min_rows = max(oh, th) + 2
    min_cols = ow + tw + 3
    if min_rows > _CANVAS_MAX or min_cols > _CANVAS_MAX:
        raise ValueError("layout does not fit canvas bounds")

    rows = random.randint(max(_CANVAS_MIN, min_rows), _CANVAS_MAX)
    cols = random.randint(max(_CANVAS_MIN, min_cols), _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    base_c = random.randint(1, cols - (ow + tw + 1) - 1)
    frame_left = random.choice([True, False])
    frame_c = base_c if frame_left else base_c + tw + 1
    tray_c = base_c + ow + 1 if frame_left else base_c
    frame_r = random.randint(1, rows - oh - 1)
    tray_r = random.randint(1, rows - th - 1)

    paste(canvas, frame_patch, frame_r, frame_c, transparent=bg)
    paste(canvas, tray, tray_r, tray_c, transparent=bg)

    piece_palette = sorted(int(col) for col in np.unique(out) if int(col) not in fcs)
    full_palette = sorted(int(col) for col in np.unique(out))
    noise_positions = _add_gap_noise(canvas, tray, tray_r, tray_c, bg, piece_palette or full_palette)

    metrics = {
        "noise_count": len(noise_positions),
        "tight_pairs": tight_pairs,
        "tray_bbox": (tray_r, tray_c, tray_r + th - 1, tray_c + tw - 1),
        "noise_positions": noise_positions,
    }
    return canvas, metrics


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None

    try:
        inp, metrics = _scatter_tight(out, bg, fcs)
        if metrics["noise_count"] < _NOISE_MIN or metrics["tight_pairs"] < _TIGHT_PAIR_MIN:
            return None
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
    raise RuntimeError("failed to generate a valid e8686506 combo input")
