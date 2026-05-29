# Combo variant of e8686506 (index 50)
# combo_spec: D3/3+D17/0
# Name: 叠放到模板+原位轮廓痕迹

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
# - Source task:
#   1. Read all non-background fragments as puzzle pieces.
#   2. Infer the hidden dense LR-symmetric HxW rectangle.
#   3. Return only that cropped HxW rectangle.
# - D3/3 contribution:
#   1. A large hollow rectangular template is also visible in the input.
#   2. The solved HxW rectangle is not the whole output anymore; it must be centered inside the template.
# - D17/0 contribution:
#   1. Moved pieces do not simply disappear from their source cells.
#   2. Their original locations remain visible as fixed gray silhouettes in the output.
# - Combined behavior on a concrete 17x17-style canvas:
#   1. Input shows a 12x9 empty ring template at one place, a separate 8x5 frame-group patch elsewhere,
#      and the other pieces scattered around that canvas.
#   2. Solver removes the template from the piece inventory, reconstructs the same hidden 8x5 rectangle,
#      centers that rectangle into the 12x9 ring, and keeps the ring color.
#   3. Output keeps the full canvas size, shows the ring plus the centered solved rectangle, and also paints
#      every source fragment position (including the separated frame-group source patch) in gray.
# - Observable difference versus the source task:
#   the destination is determined by a visible template, and the old piece locations remain visible too.
#
# b) Complexity calibration
# - The template is always a single-color hollow rectangle with fixed margin 2 around the solved board.
#   This keeps centering deterministic instead of underconstrained.
# - Gray is reserved as the only trace color, so the new trajectory rule is visually consistent across samples.
# - The frame-group patch is still preserved as a recognizable LR-symmetric HxW anchor outside the template,
#   so the original assembly logic remains solvable without backtracking.
#
# c) Implicit assumption audit against the original main()
# - Original assumption: every non-bg component is a puzzle piece.
#   Broken by D3/3. Fix: detect the hollow template first and exclude it from piece extraction.
# - Original assumption: returning the solved HxW board is sufficient.
#   Broken by both dimensions. Fix: render a full-canvas answer containing template placement plus shadows.
# - Original assumption: frame detection can search the full non-bg grid.
#   Unsafe after adding the template. Fix: run frame detection on the template-stripped grid only.
# - New combo assumption: all movable pieces, including the source frame-group patch, lie outside the
#   template bbox so their gray traces do not overwrite the merged template result. The generator enforces this.
#
# d) Placement legality
# - Generator reserves disjoint rectangles for the template bbox and the source frame-group bbox.
# - All other patches are scattered only in remaining background area with Chebyshev padding 1.
# - The template interior stays empty in the input, so it is unambiguous and solver-detectable.
#
# e) Visual observability
# - Across examples, a human sees two separate cues:
#   one large empty ring showing the destination, and one gray-shadow rule showing the provenance.
# - No hidden color ordering is used. The only special color is gray, consistently used for traces.
#
# f) Visual cleanliness
# - Template/result live in one reserved region; source shadows live in other reserved regions.
# - This avoids clutter from stacking gray shadows on top of the solved placement.
#
# g) Inferability of the changed behavior
# - Template color is preserved rather than remapped, and gray always means "source location".
# - Because the source frame-group patch also turns into gray in the output, the user can infer that
#   all moved fragments leave traces, not only the freely scattered interior pieces.

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_TEMPLATE_MARGIN = 2
_SHADOW_COLOR = 5


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


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    candidates = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 9 or w < 9:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            if not np.all(patch[border_mask] == col):
                continue
            if np.any(patch[~border_mask] != bg):
                continue
            if len(pts) != int(border_mask.sum()):
                continue
            candidates.append((h * w, int(col), int(r0), int(c0), int(r1), int(c1)))
    assert len(candidates) == 1, "input must contain exactly one hollow rectangular template"
    _, col, r0, c0, r1, c1 = candidates[0]
    return r0, c0, r1, c1, col


def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

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
    return frame_parts


def _extract_piece_data(piece_grid: np.ndarray, bg: int) -> tuple[
    dict[int, list[list[tuple[int, int]]]],
    list[dict],
    dict[int, int],
    list[np.ndarray],
]:
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}
    shadow_components: list[np.ndarray] = []

    for col in sorted(set(map(int, np.unique(piece_grid)))):
        if col == bg:
            continue
        comps = components_mask(piece_grid == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            pts_arr = np.asarray(pts, dtype=int)
            shadow_components.append(pts_arr)
            _, _, coords = _piece_sig(pts_arr)
            pieces.append((col, coords))

    ctr = Counter(pieces)
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
    return comps_by_color, types, color_piece_count, shadow_components


def _infer_board(piece_grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    comps_by_color, types, color_piece_count, shadow_components = _extract_piece_data(piece_grid, bg)

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-template pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(piece_grid, bg, H, W)
    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    board = _assemble(types, H, W, req, disallow_touch)
    return board.astype(int), shadow_components


def _render_output(
    grid_shape: tuple[int, int],
    bg: int,
    template_patch: np.ndarray,
    tr0: int,
    tc0: int,
    board: np.ndarray,
    shadow_components: list[np.ndarray],
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    paste(out, template_patch, tr0, tc0, transparent=bg)

    TH, TW = template_patch.shape
    H, W = board.shape
    dr = TH - H
    dc = TW - W
    assert dr == 2 * _TEMPLATE_MARGIN and dc == 2 * _TEMPLATE_MARGIN, "template and board sizes must use fixed centering margin"
    r0 = tr0 + dr // 2
    c0 = tc0 + dc // 2
    out[r0 : r0 + H, c0 : c0 + W] = board

    for pts in shadow_components:
        for r, c in pts:
            out[int(r), int(c)] = _SHADOW_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered fragments, center the result into the visible template, and keep gray source shadows.

    Input structure:
    - The most common color is background; one large non-background hollow rectangle is a destination template.
    - Every other non-background component is a movable puzzle fragment from the original e8686506 task.
    - Among those fragments, 1-2 LR-symmetric frame parts still appear together as an anchored HxW source patch.

    Rule:
    - Detect the hollow template and remove it from the puzzle inventory.
    - Reconstruct the hidden dense LR-symmetric rectangle from the remaining fragments using the original
      symmetry, frame-lock, and same-color non-touch constraints.
    - Produce a full-canvas output: keep the template at its input location, center the solved rectangle inside it,
      and repaint every original fragment location in gray as a source trace.

    Key insight:
    - The template and the gray traces are independent output cues layered onto the same solution.
      First solve the old rectangle-assembly problem on the non-template pieces, then place that solved board
      into the template and separately mark where all movable pieces originally came from.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    tr0, tc0, tr1, tc1, _template_col = _find_template(g, bg)
    template_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    piece_grid = g.copy()
    piece_grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

    board, shadow_components = _infer_board(piece_grid, bg)
    return _render_output(g.shape, bg, template_patch, tr0, tc0, board, shadow_components)


# --------------- generator (o5x) ---------------

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

_CANVAS_EXTRA_MIN = 8
_CANVAS_EXTRA_MAX = 12
_SCATTER_PADDING = 1
_RESERVE_PADDING = 1

_GEN_MAX_RETRIES = 600


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
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
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


def _build_board(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
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
    half = W // 2
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


def _make_template_patch(board: np.ndarray, bg: int, template_col: int) -> np.ndarray:
    H, W = board.shape
    TH = H + 2 * _TEMPLATE_MARGIN
    TW = W + 2 * _TEMPLATE_MARGIN
    template = np.full((TH, TW), bg, dtype=int)
    template[0, :] = template_col
    template[-1, :] = template_col
    template[:, 0] = template_col
    template[:, -1] = template_col
    return template


def _make_output_assets() -> tuple[np.ndarray | None, np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    abstract_board, abstract_fcs = _build_board(H, W)
    if abstract_board is None:
        return None, None, 0, set()

    bg_choices = [c for c in range(10) if c != _SHADOW_COLOR]
    bg = random.choice(bg_choices)
    abstract = sorted(np.unique(abstract_board))
    available = [c for c in range(10) if c not in (bg, _SHADOW_COLOR)]
    random.shuffle(available)
    if len(available) < len(abstract) + 1:
        return None, None, 0, set()

    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    board = np.vectorize(lambda x: cmap[int(x)])(abstract_board).astype(int)
    real_fcs = {cmap[a] for a in abstract_fcs}
    used = set(map(int, np.unique(board))) | {bg}
    template_col = next(c for c in available[len(abstract) :] if c not in used)
    template = _make_template_patch(board, bg, template_col).astype(int)
    return board, template, bg, real_fcs


def _extract_non_frame_patches(board: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(board))):
        if col in fcs:
            continue
        for pts in components_mask(board == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _frame_patch(board: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    patch = np.full(board.shape, bg, dtype=int)
    mask = np.zeros(board.shape, dtype=bool)
    for col in fcs:
        mask |= board == col
    patch[mask] = board[mask]
    return patch


def _reserve_rect(canvas: np.ndarray, h: int, w: int, bg: int) -> tuple[int, int]:
    block = np.full((h, w), -1, dtype=int)
    r, c = random_free_location(
        canvas,
        block,
        transparent=bg,
        border=1,
        padding=_RESERVE_PADDING,
        padding_connectivity=8,
    )
    paste(canvas, block, r, c, transparent=bg)
    return r, c


def _shadow_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    out = np.full_like(patch, bg)
    out[patch != bg] = _SHADOW_COLOR
    return out


def _scatter(board: np.ndarray, template: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    bh, bw = board.shape
    th, tw = template.shape

    side_min = max(bh, th) + _CANVAS_EXTRA_MIN
    side_max = max(bh, th) + _CANVAS_EXTRA_MAX
    rows = random.randint(side_min, side_max)
    cols = random.randint(side_min, side_max)
    canvas = np.full((rows, cols), bg, dtype=int)

    tr, tc = _reserve_rect(canvas, th, tw, bg)
    fr, fc = _reserve_rect(canvas, bh, bw, bg)

    source_specs: list[tuple[np.ndarray, int, int]] = []
    frame_source = _frame_patch(board, bg, fcs)
    canvas[fr : fr + bh, fc : fc + bw] = bg
    paste(canvas, frame_source, fr, fc, transparent=bg)
    source_specs.append((frame_source.copy(), fr, fc))

    for patch in _extract_non_frame_patches(board, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            border=1,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
        source_specs.append((patch.copy(), r, c))

    canvas[tr : tr + th, tc : tc + tw] = bg
    paste(canvas, template, tr, tc, transparent=bg)

    expected = np.full_like(canvas, bg)
    paste(expected, template, tr, tc, transparent=bg)
    expected[tr + _TEMPLATE_MARGIN : tr + _TEMPLATE_MARGIN + bh, tc + _TEMPLATE_MARGIN : tc + _TEMPLATE_MARGIN + bw] = board

    shadow_cells = 0
    for patch, r, c in source_specs:
        shadow = _shadow_patch(patch, bg)
        shadow_cells += int((shadow != bg).sum())
        paste(expected, shadow, r, c, transparent=bg)

    metrics = {
        "shadow_cells": int(shadow_cells),
        "template_area": int((template != bg).sum()),
        "frame_shadow_cells": int((frame_source != bg).sum()),
    }
    return canvas, expected, metrics


def _piece_count(board: np.ndarray) -> int:
    return sum(len(components_mask(board == col, connectivity=4)) for col in np.unique(board))


def _gen_attempt() -> tuple[np.ndarray | None, dict[str, int]]:
    board, template, bg, fcs = _make_output_assets()
    if board is None or template is None:
        return None, {}
    if not np.array_equal(board, board[:, ::-1]):
        return None, {}
    if not (_N_COLORS_MIN <= len(np.unique(board)) <= _N_COLORS_MAX):
        return None, {}
    pc = _piece_count(board)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, {}
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(board)
        for pts in components_mask(board == col, connectivity=4)
    ):
        return None, {}

    try:
        inp, expected, metrics = _scatter(board, template, bg, fcs)
        got = main(inp)
    except Exception:
        return None, {}

    if metrics["shadow_cells"] <= 0 or metrics["frame_shadow_cells"] <= 0 or metrics["template_area"] <= 0:
        return None, {}
    if np.array_equal(got, expected):
        return inp, metrics
    return None, {}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, metrics = _gen_attempt()
        if grid is not None and metrics["shadow_cells"] > 0 and metrics["frame_shadow_cells"] > 0:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
