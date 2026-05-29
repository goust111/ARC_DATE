# Variant of e8686506
# Dimension: D2 — 源对象：碎片的类型与结构 (src)
# Change: 散落的单色非背景像素碎片和精灵片段，每个碎片为单一颜色的连通区域... → 碎片含干扰像素

# Design analysis
# yaml:
#   variant_goal:
#     keep: 输出仍然是把真实碎片拼成一个严格左右镜像的致密矩形，框架锚点、面积分解、
#       同色不同块不可 4 连通接触等原始约束保持不变。
#     change: 输入里额外混入 1-3 个“孤立干扰像素”。这些像素不是任何真实碎片的一部分，
#       颜色来自真实填充碎片颜色池，因此不能靠固定颜色规则剔除。
#   concrete_behavior:
#     real_frame: 原题已经把 1-2 个 8 连通框架片段原样放在一个 HxW 识别框里。
#     real_pieces: 非框架碎片仍被散落在框外。
#     noise: 干扰像素只放在该框的严格内侧背景格里，彼此不相邻，也不覆盖框架。
#     visible_effect: 人眼会看到“可识别的矩形轮廓 + 框内散落的杂点 + 框外是真碎片”。
#   flow_trace:
#     original_step:
#       原 solver 先把所有非背景 4 连通分量都当作真实碎片，再用总面积推 HxW。
#     variant_step:
#       新 solver 先从 8 连通分量枚举可能的框架 bbox；对每个候选 bbox，把位于 bbox
#       严格内侧的孤立单点暂视为干扰并剔除，再检查剩余总面积是否恰好等于 bbox 面积，
#       并运行原有拼装传播过程验证。
#     difference:
#       “先求总面积”改为“先找框架，再用框架解释哪些单点是干扰”。这样噪声不会污染面积、
#       锚点统计或同色组件计数。
#   complexity_calibration:
#     why_not_color_rule: 干扰像素颜色复用真实碎片颜色，避免退化成“忽略某个固定颜色”。
#     why_singletons_inside_frame: 原任务允许真实 1x1 碎片，因此若随意在框外加单点会与真实块混淆；
#       把干扰固定在框内严格内侧，能形成稳定且可见的结构线索。
#     search_space: 候选框架只需枚举 1-2 个 8 连通分量组合，最多几十个组合，复杂度可控。
#   implicit_assumptions_audited:
#     - 旧代码假设 total_area 来自全部非背景像素；变体后不成立，已改为候选框架驱动。
#     - 旧代码把 comps_by_color 的 singleton 计数直接用于角点/边条推断；变体后若不先过滤，
#       噪声会破坏“恰好 4 个角点单点”等条件。
#     - 旧代码用 total_area 先确定 H,W，再找框架；变体后会被干扰面积带偏，已反转顺序。
#   placement_legality:
#     generator 仍先构造输出，再散布真实碎片；框架 bbox 整块预留，真实碎片保持在框外，
#     干扰像素只写回框内背景格，故不会越界、重叠，也不会与真实碎片连成同色组件。
#   observability_cleanliness:
#     框内杂点和框外真实碎片分区明确，2-3 个样例就能看出“框里的孤立点不参与拼装”。
#     干扰数量保持少量，避免画面杂乱，也不引入不可见排序或颜色 ID 规则。
#   post_change_solver_assumptions:
#     solver 新增假设：干扰只会是位于候选框架 bbox 严格内侧的 1x1 分量。
#     generate_input() 明确按此构造，因此 solver/generator 一致。

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
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
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
                {
                    "parts": [(c1, p1)],
                    "bbox": (r0, c0, r1, c1b),
                    "H": h1,
                    "W": w1,
                }
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
                    {
                        "parts": parts,
                        "bbox": (ur0, uc0, ur1, uc1),
                        "H": h,
                        "W": w,
                    }
                )
    candidates.sort(key=lambda cand: (-(cand["H"] * cand["W"]), len(cand["parts"])))
    return candidates


def _solve_from_frame_candidate(
    comps4_all: list[tuple[int, np.ndarray]],
    candidate: dict,
) -> np.ndarray | None:
    H, W = int(candidate["H"]), int(candidate["W"])
    r0, c0, r1, c1 = candidate["bbox"]

    comps_by_color: dict[int, list[np.ndarray]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col, pts in comps4_all:
        if len(pts) == 1:
            rr, cc = map(int, pts[0])
            if r0 < rr < r1 and c0 < cc < c1:
                continue
        comps_by_color.setdefault(int(col), []).append(pts)
        color_piece_count[int(col)] = color_piece_count.get(int(col), 0) + 1
        _, _, coords = _piece_sig(pts)
        pieces.append((int(col), coords))

    if not pieces:
        return None

    total_area = sum(len(coords) * cnt for (_, coords), cnt in Counter(pieces).items())
    if total_area != H * W:
        return None

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    if not dims:
        return None
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    if len(best_dims) != 1 or best_dims[0] != (H, W):
        return None

    ctr = Counter(pieces)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true fragments into a left-right symmetric rectangle while ignoring stray dots.

    Input structure:
    - The most common color is background; non-background cells include real monochrome fragments plus a few isolated interference pixels.
    - The 1-2 frame fragments still appear in-place as an LR-symmetric outline, while the real non-frame pieces stay scattered outside that outline.

    Rule:
    - First identify the frame candidate(s), which determine the output rectangle's bounding box.
    - Treat singleton pixels strictly inside that frame box as interference, discard them, and keep all remaining components as the actual jigsaw pieces.
    - Assemble those real pieces into the dense rectangle so it is exactly left-right symmetric and same-color distinct pieces never 4-touch.

    Key insight:
    - The outline frame is the stable anchor; once its bbox is known, any lone pixel floating inside that box is junk rather than a tile, so the original propagation-based assembly becomes valid again.
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

    candidates = _enumerate_frame_candidates(comps8_all)
    for candidate in candidates:
        sol = _solve_from_frame_candidate(comps4_all, candidate)
        if sol is not None:
            return sol.astype(int)

    raise AssertionError("no frame candidate yields a valid noise-filtered assembly")


# --------------- generator (o5x) ---------------

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
    bg: int,
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


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, int]:
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

    noise_count = _add_noise(canvas, out, bg, fcs, off_r, off_c)
    return canvas, noise_count


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


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
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None
    try:
        inp, noise_count = _scatter(out, bg, fcs)
        if noise_count < _NOISE_MIN:
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
    raise RuntimeError("failed to generate a valid e8686506 D2 input")
