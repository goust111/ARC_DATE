# Combo variant of e8686506 (index 9)
# combo_spec: D2/3+D13/2
# Name: 干扰像素+背景网格线
#
# Design analysis
# yaml:
#   variant_goal:
#     keep: 输出仍是把真实单色碎片拼成一个严格左右镜像的致密矩形；框架锚点、
#       最近平方因子定尺寸、同色不同块不可 4 连通接触等原始约束保持不变。
#     change: 输入同时加入两种前处理噪声：
#       1) 框内严格内侧的孤立干扰像素；
#       2) 覆盖整个输入背景的等距网格线。
#   concrete_behavior:
#     base_scene:
#       先按 D2/3 单变体生成一个“可识别框架 + 框外散落真实碎片 + 框内孤立噪声点”的基础画布。
#     grid_fusion:
#       再把该基础画布嵌入 1x1 单元格网格：每个原像素落在一个网格单元中心，单元之间由 1 像素宽网格线分隔。
#       因而最终输入上，真实碎片、背景和干扰点都出现在网格单元里，网格线颜色独立且等距重复。
#     visible_effect:
#       人眼会先看到大面积规则格栅，再看到框内几颗零散彩点，以及框外若干真正碎片。
#       干扰点位于网格单元中，视觉上容易和“普通网格交叉点附近的单点”混淆。
#   flow_trace:
#     example:
#       例如基础画布里一个 13x13 输入，框内有 2 个孤立彩点；加网格后变成 27x27，
#       偶数行列全是网格线色，原来那 2 个孤立点落在奇数行奇数列单元里。
#     original_step:
#       原 solver 直接把所有非背景 4 连通分量当真碎片，再用总面积推输出尺寸。
#     combo_step:
#       新 solver 先识别整幅图上的等距全行/全列网格线并压缩回基础画布；
#       再枚举 1-2 个 8 连通框架候选，对每个候选把 bbox 严格内侧的 1x1 分量当干扰点剔除；
#       最后仅用剩余真实碎片执行原有传播式拼装。
#     difference:
#       原逻辑的“直接统计非背景像素”在融合后会同时被网格线面积和干扰点面积污染，
#       所以必须改成“先去格栅，再由框架解释哪些单点是噪声”。
#   complexity_calibration:
#     why_grid_is_visible:
#       网格线覆盖整张输入，而不是局部装饰；即便只看 2-3 个样例，人也能稳定看到“先忽略格栅”的需要。
#     why_noise_is_not_trivial:
#       干扰点复用真实填充色，不能靠固定颜色删除；同时它们只放在框内背景单元里，
#       与原题允许的边界 1x1 真碎片形成可观察但不混乱的对照。
#     why_search_space_stays_controlled:
#       去网格后仍只需枚举 1-2 个 8 连通框架组合；噪声过滤只对 bbox 内部单点生效，
#       不会把求解空间扩成无界的“尝试删除任意像素”。
#   implicit_assumptions_audited:
#     - 旧代码假设原始输入的最常见颜色就是背景；融合后原始输入里更常见的是网格线色，
#       所以 main() 不能在去网格前调用 get_bg。
#     - 旧代码假设所有非背景像素都是有效碎片；融合后网格线和干扰点都违反该前提。
#     - 旧代码先用 total_area 定 H,W，再找框架；融合后 total_area 若先算会被两类噪声同时带偏。
#     - 旧代码把 singleton 计数直接当角点锚线索；融合后必须先删掉框内单点干扰，否则角点统计失真。
#   placement_legality:
#     基础画布阶段仍保证真实碎片互不重叠、框内噪声不覆盖框架且彼此不相邻。
#     网格扩张阶段只是把每个基础像素放入独立单元中心，网格线写在单元边界，不会引入新重叠或越界。
#   observability_cleanliness:
#     网格线统一为单一颜色且全局等距，框架仍然作为显著矩形轮廓保留；
#     真碎片在框外、干扰点在框内，两者空间分区清楚，视觉上不会糊成一团。
#   post_change_solver_assumptions:
#     - 网格线为 1 像素宽，且存在若干整行整列的同色网格线，行列间距一致。
#     - 去网格后得到的基础画布正是 D2/3 约束下的实例。
#     - 干扰像素只会是候选框架 bbox 严格内侧的孤立 1x1 分量。
#     generator_guarantee:
#       generate_input() 显式按以上三条构造，因此 solver 新假设都被生成器满足。

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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
            candidates.append(
                {"parts": [(c1, p1)], "bbox": (r0, c0, r1, c1b), "H": h1, "W": w1}
            )
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            h, w = ur1 - ur0 + 1, uc1 - uc0 + 1
            if h < w:
                continue
            parts = [(c1, p1), (c2, p2)]
            if _is_lr_symmetric(parts, h, w):
                candidates.append(
                    {"parts": parts, "bbox": (ur0, uc0, ur1, uc1), "H": h, "W": w}
                )
    candidates.sort(key=lambda cand: (-(cand["H"] * cand["W"]), len(cand["parts"])))
    return candidates


def _detect_grid_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    H, W = grid.shape
    candidates = []
    for col in sorted(map(int, np.unique(grid))):
        full_rows = [r for r in range(H) if np.all(grid[r, :] == col)]
        full_cols = [c for c in range(W) if np.all(grid[:, c] == col)]
        if len(full_rows) < 3 or len(full_cols) < 3:
            continue
        row_steps = [b - a for a, b in zip(full_rows, full_rows[1:])]
        col_steps = [b - a for a, b in zip(full_cols, full_cols[1:])]
        if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
            continue
        if row_steps[0] != col_steps[0] or row_steps[0] < 2:
            continue
        if full_rows[0] != 0 or full_cols[0] != 0:
            continue
        if full_rows[-1] != H - 1 or full_cols[-1] != W - 1:
            continue
        if any(r not in full_rows and np.all(grid[r, :] == col) for r in range(H)):
            continue
        if any(c not in full_cols and np.all(grid[:, c] == col) for c in range(W)):
            continue
        candidates.append((len(full_rows) + len(full_cols), col, full_rows, full_cols))
    assert len(candidates) == 1, "expected a unique regular grid-line color"
    _, grid_col, full_rows, full_cols = candidates[0]
    return int(grid_col), full_rows, full_cols


def _compress_grid_cells(grid: np.ndarray) -> np.ndarray:
    grid_col, full_rows, full_cols = _detect_grid_lines(grid)
    row_gaps = [b - a - 1 for a, b in zip(full_rows, full_rows[1:])]
    col_gaps = [b - a - 1 for a, b in zip(full_cols, full_cols[1:])]
    assert row_gaps and col_gaps, "grid must define at least one cell"
    assert len(set(row_gaps)) == 1 and row_gaps[0] >= 1, "row cell size must be uniform"
    assert len(set(col_gaps)) == 1 and col_gaps[0] >= 1, "col cell size must be uniform"

    out = np.zeros((len(full_rows) - 1, len(full_cols) - 1), dtype=int)
    for i in range(len(full_rows) - 1):
        for j in range(len(full_cols) - 1):
            block = grid[full_rows[i] + 1 : full_rows[i + 1], full_cols[j] + 1 : full_cols[j + 1]]
            vals = [int(v) for v in block.flatten() if int(v) != grid_col]
            assert vals, "every grid cell must contain a base-canvas color"
            out[i, j] = Counter(vals).most_common(1)[0][0]
    return out


def _solve_from_frame_candidate(
    comps4_all: list[tuple[int, np.ndarray]],
    candidate: dict,
    *,
    return_noise_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, int] | None:
    H, W = int(candidate["H"]), int(candidate["W"])
    r0, c0, r1, c1 = candidate["bbox"]

    comps_by_color: dict[int, list[np.ndarray]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}
    noise_count = 0

    for col, pts in comps4_all:
        if len(pts) == 1:
            rr, cc = map(int, pts[0])
            if r0 < rr < r1 and c0 < cc < c1:
                noise_count += 1
                continue
        comps_by_color.setdefault(int(col), []).append(pts)
        color_piece_count[int(col)] = color_piece_count.get(int(col), 0) + 1
        _, _, coords = _piece_sig(pts)
        pieces.append((int(col), coords))

    if not pieces:
        return None

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
        sol = _assemble(types, H, W, req, disallow_touch)
    except Exception:
        return None
    if return_noise_count:
        return sol, noise_count
    return sol


def _solve_base_noise_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps4_all: list[tuple[int, np.ndarray]] = []
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            comps4_all.append((int(col), pts))
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    candidates = _enumerate_frame_candidates(comps8_all)
    for candidate in candidates:
        sol = _solve_from_frame_candidate(comps4_all, candidate)
        if sol is not None:
            return sol.astype(int)

    raise AssertionError("no frame candidate yields a valid noise-filtered assembly")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Strip the background lattice and ignore stray dots before reassembling the symmetric rectangle.

    Input structure:
    - The visible canvas is overlaid with a regular grid of one special color; every meaningful pixel sits inside a grid cell rather than on the grid lines.
    - After removing that lattice, the underlying scene matches the noisy-fragment variant: 1-2 frame fragments stay aligned as an LR-symmetric outline, real non-frame pieces are scattered outside it, and a few isolated interference pixels may float strictly inside the frame box.

    Rule:
    - Detect the equally spaced full-row/full-column grid lines and compress each grid cell back to one base pixel, discarding the grid-line color entirely.
    - On the recovered base canvas, enumerate frame candidate(s), discard singleton pixels strictly inside the candidate frame bbox as interference, and treat every remaining component as a true puzzle fragment.
    - Assemble those true fragments into the dense portrait rectangle so the result is exactly left-right symmetric and distinct pieces of the same color never 4-touch.

    Key insight:
    - The grid is global, perfectly regular noise, while the frame is the stable semantic anchor inside the de-gridded scene; once both are recognized, the only unmatched single-cell residues are the deliberate distractors, and the original propagation-only packing logic becomes valid again.
    """
    base = _compress_grid_cells(np.asarray(grid, dtype=int))
    return _solve_base_noise_variant(base)


# --------------- generator (combo) ---------------

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
_NOISE_MIN = 1
_NOISE_MAX = 3

# === CFG: Grid overlay ===
_GRID_LINE_WIDTH = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 700


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

    _fill_holes(grid, W, frame1, frame2, split, fill_pool)
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
        frame_mask |= out == col

    fill_colors = [int(col) for col in np.unique(out) if int(col) not in fcs]
    if not fill_colors:
        raise ValueError

    candidates = [
        (off_r + r, off_c + c)
        for r in range(1, oh - 1)
        for c in range(1, ow - 1)
        if not frame_mask[r, c]
    ]
    random.shuffle(candidates)
    if len(candidates) < _NOISE_MIN:
        raise ValueError

    target = random.randint(_NOISE_MIN, min(_NOISE_MAX, len(candidates)))
    chosen: list[tuple[int, int]] = []
    for rr, cc in candidates:
        if all(max(abs(rr - pr), abs(cc - pc)) > 1 for pr, pc in chosen):
            chosen.append((rr, cc))
            if len(chosen) == target:
                break
    if len(chosen) < target:
        raise ValueError

    for rr, cc in chosen:
        canvas[rr, cc] = random.choice(fill_colors)
    return len(chosen)


def _scatter_base(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, int]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
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

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    noise_count = _add_noise(canvas, out, fcs, off_r, off_c)
    return canvas, noise_count


def _overlay_grid(base: np.ndarray) -> tuple[np.ndarray, int]:
    used = set(map(int, np.unique(base)))
    choices = [c for c in range(10) if c not in used]
    if not choices:
        raise ValueError("need a dedicated grid color")
    grid_col = random.choice(choices)

    H, W = base.shape
    expanded = np.full(
        (
            H * (_GRID_LINE_WIDTH + 1) + _GRID_LINE_WIDTH,
            W * (_GRID_LINE_WIDTH + 1) + _GRID_LINE_WIDTH,
        ),
        grid_col,
        dtype=int,
    )
    expanded[_GRID_LINE_WIDTH::(_GRID_LINE_WIDTH + 1), _GRID_LINE_WIDTH::(_GRID_LINE_WIDTH + 1)] = base
    return expanded, grid_col


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    compressed = _compress_grid_cells(inp)
    bg = get_bg(compressed)
    comps4_all: list[tuple[int, np.ndarray]] = []
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(compressed))) - {bg}):
        for pts in components_mask(compressed == col, connectivity=4):
            comps4_all.append((int(col), pts))
        for pts in components_mask(compressed == col, connectivity=8):
            comps8_all.append((int(col), pts))

    for candidate in _enumerate_frame_candidates(comps8_all):
        solved = _solve_from_frame_candidate(comps4_all, candidate, return_noise_count=True)
        if solved is not None:
            _, noise_count = solved
            grid_col, full_rows, full_cols = _detect_grid_lines(inp)
            return {
                "grid_color": int(grid_col),
                "grid_rows": len(full_rows),
                "grid_cols": len(full_cols),
                "noise_count": int(noise_count),
                "compressed_h": int(compressed.shape[0]),
                "compressed_w": int(compressed.shape[1]),
            }
    raise AssertionError("fusion metric extraction failed")


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
        base_inp, noise_count = _scatter_base(out, bg, fcs)
        if noise_count < _NOISE_MIN:
            return None
        inp, _grid_col = _overlay_grid(base_inp)
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
