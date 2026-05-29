# Variant of e8686506
# Dimension: D11 — 多步处理逻辑 (multi_step)
# Change: Step 1: 识别所有非背景碎片; Step 2: 通过边缘匹配将碎片拼合成矩形; Step 3: 紧凑裁剪输出... → 拼合后应用全局颜色变换

import importlib.util
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D11 variant
#
# a) Concretization and behavior trace
# - Keep the original visible input language: scattered monochrome fragments on a
#   mostly-uniform background, plus 1-2 frame fragments already positioned with
#   their output-relative geometry preserved.
# - The new behavior happens only after the old rectangle assembly is complete.
#   We assemble the same dense LR-symmetric rectangle as before, but then take
#   the geometric center pixel color and repaint the entire outer border
#   (top/bottom rows and left/right columns) to that center color.
# - Concrete trace on a 7x5 example:
#   original logic: scatter -> detect pieces -> assemble a 7x5 rectangle -> return it.
#   variant logic: scatter -> detect pieces -> assemble the same 7x5 rectangle ->
#   read cell (3,2) -> recolor the whole border to that color -> return the recolored grid.
#   The difference is therefore a genuine extra global step, not a different
#   assembly criterion.
#
# b) Complexity calibration
# - A free-form color permutation would be impossible to infer from 2-3 examples,
#   so the post-process is reduced to one stable rule with one stable source:
#   "border takes the center color".
# - To make this visible without disturbing the source-task assembly logic, the
#   generator now samples valid source-task instances and keeps only those where
#   the assembled rectangle already has a center-column clue: odd height, center
#   color absent from the pre-transform border, and a center-column run of at
#   least length 2 through the middle.
# - Heights are therefore restricted to odd values by filtering, so the source is
#   a single visible center pixel rather than an ambiguous middle band.
#
# c) Hidden-assumption audit against the original solver
# - Piece extraction by color/components is still valid because the input still
#   contains the pre-transform scattered pieces.
# - Output dimension inference from total piece area is still valid because the
#   post-process does not change area.
# - Frame detection and requirement locking still depend on the pre-transform
#   colors; this remains safe because recoloring happens strictly after assembly.
# - The original return `sol.astype(int)` is no longer valid: returning the raw
#   assembled board would silently collapse this variant back to the source task.
#   The new code must route every solved board through the border recolor step.
#
# d) Placement legality
# - Instead of synthesizing new piece layouts, the generator reuses source-task
#   inputs that are already known to scatter legally and assemble uniquely.
# - The only added legality constraint is the visibility filter on the assembled
#   rectangle: odd height, border color change guaranteed, and a non-trivial
#   center-column run so the source color is visually supported.
#
# e) Visual observability
# - A human sees the output border become exactly the same color as the internal
#   center stripe. This is a strong, repeated geometric cue across examples.
# - No invisible ordering by color id is used. The source color is selected only
#   by position: the unique center cell of the assembled rectangle.
#
# f) Visual cleanliness
# - Only one post-process layer is added, and it affects the border only.
#   Interior hole structure remains readable, while the new border color creates
#   a clean "frame takeover" effect instead of noisy local edits.
#
# g) Color-change inferability
# - The generator inherits the compact palette of the source task and only keeps
#   cases where the center color has a visible vertical presence.
# - Examples therefore repeatedly show the same structural dependency:
#   center-column color -> outer border.
#
# Post-change solver assumptions that the generator now guarantees
# - The assembled rectangle has odd height and odd width, so the center cell is
#   unambiguous.
# - The center color differs from the original border colors, so the extra step
#   changes the result on every accepted sample.
# - The assembled rectangle has a visible center-column run through the middle,
#   supporting human induction of the post-assembly recoloring rule.
# ---------------------------------------------------------------------------


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


def _apply_border_recolor(board: np.ndarray) -> np.ndarray:
    H, W = board.shape
    assert H % 2 == 1 and W % 2 == 1, "variant requires an odd-sized center cell"
    out = board.copy()
    center_col = int(board[H // 2, W // 2])
    out[0, :] = center_col
    out[H - 1, :] = center_col
    out[:, 0] = center_col
    out[:, W - 1] = center_col
    return out


def _assemble_base_rectangle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
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
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
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
    return _assemble(types, H, W, req, disallow_touch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered fragments into a symmetric rectangle, then repaint its outer border using the center color.

    Input structure:
    - The background is the most frequent color; every other connected component is a fragment of the hidden rectangle.
    - As in the source task, one or two frame fragments remain in output-relative alignment, while the remaining interior fragments are scattered elsewhere.

    Rule:
    - Recover the dense left-right symmetric rectangle by matching all non-background fragments into the unique closest-to-square portrait shape.
    - After the rectangle is fully assembled, read its center pixel color and replace the entire outer border with that color.

    Key insight:
    - The frame geometry and mirror constraints still determine the pre-transform rectangle uniquely; the new step is a global post-processing rule driven by the rectangle's geometric center, not by the original border colors.
    """
    base = _assemble_base_rectangle(grid)
    return _apply_border_recolor(base).astype(int)


# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10
_ODD_HEIGHT_ONLY = True

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 7
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_CENTER_STRIPE_LEN = 3

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 160


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX and (not _ODD_HEIGHT_ONLY or h % 2 == 1):
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


def _fill_holes(grid: np.ndarray, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
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


def _inject_center_stripe(grid: np.ndarray, seed_col: int) -> None:
    H, W = grid.shape
    mid_r = H // 2
    mid_c = W // 2
    rows = range(mid_r - (_CENTER_STRIPE_LEN // 2), mid_r + (_CENTER_STRIPE_LEN // 2) + 1)
    for r in rows:
        assert 0 < r < H - 1, "center stripe must stay off the border"
        grid[r, mid_c] = seed_col


def _build_base_output(H: int, W: int) -> tuple[np.ndarray | None, set[int], int]:
    total_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    base_colors = total_colors - 1
    palette = list(range(base_colors))
    random.shuffle(palette)

    use_two_frames = H >= _TWO_FRAME_MIN_H and total_colors >= _TWO_FRAME_MIN_COLORS and random.random() < _TWO_FRAME_PROB
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set(), -1

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)
    half = W // 2
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
        return None, set(), -1

    seed_col = base_colors
    _inject_center_stripe(grid, seed_col)

    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set, seed_col


def _make_outputs() -> tuple[np.ndarray | None, np.ndarray | None, int, set[int], int]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    base_out, abstract_fcs, abstract_seed = _build_base_output(H, W)
    if base_out is None:
        return None, None, 0, set(), -1

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(base_out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, None, 0, set(), -1
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_base = np.vectorize(lambda x: cmap[int(x)], otypes=[int])(base_out)
    real_fcs = {cmap[a] for a in abstract_fcs}
    real_seed = cmap[abstract_seed]
    real_final = _apply_border_recolor(real_base)
    return real_base, real_final, bg, real_fcs, real_seed


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


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for reserved frame box")
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
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask[0, :] = True
    mask[H - 1, :] = True
    mask[:, 0] = True
    mask[:, W - 1] = True
    return mask


_SOURCE_MOD = None
_SOURCE_PATH = "/home/lixinyu/snap/new/arc2_solver_select/e8686506.py"


def _load_source_module():
    global _SOURCE_MOD
    if _SOURCE_MOD is None:
        spec = importlib.util.spec_from_file_location("e8686506_source", _SOURCE_PATH)
        assert spec is not None and spec.loader is not None, "failed to load source task module"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SOURCE_MOD = mod
    return _SOURCE_MOD


def _center_column_run(base: np.ndarray) -> int:
    H, W = base.shape
    mid_r, mid_c = H // 2, W // 2
    col = int(base[mid_r, mid_c])
    run = 1
    r = mid_r - 1
    while r >= 0 and int(base[r, mid_c]) == col:
        run += 1
        r -= 1
    r = mid_r + 1
    while r < H and int(base[r, mid_c]) == col:
        run += 1
        r += 1
    return run


def _candidate_has_visible_transform(base: np.ndarray) -> bool:
    H, W = base.shape
    if H % 2 == 0 or W % 2 == 0:
        return False
    border = _border_mask(base.shape)
    center_color = int(base[H // 2, W // 2])
    if np.any(base[border] == center_color):
        return False
    if _center_column_run(base) < 2:
        return False
    transformed = _apply_border_recolor(base)
    return not np.array_equal(base, transformed)


def _gen_attempt() -> np.ndarray | None:
    src = _load_source_module()
    try:
        inp = src.generate_input()
    except Exception:
        return None
    try:
        base = _assemble_base_rectangle(inp)
    except Exception:
        return None
    if not _candidate_has_visible_transform(base):
        return None
    try:
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, _apply_border_recolor(base)):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D11 input")
