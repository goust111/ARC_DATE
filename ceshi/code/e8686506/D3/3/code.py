# Variant of e8686506
# Dimension: D3 — 目标结构：拼合后的形状要求 (dst)
# Change: 拼合后形成一个紧凑的矩形复合体，所有碎片无间隙无重叠地拼成完整矩形... → 碎片拼合后叠放到模板上

# Design analysis for the D3 variant
# ----------------------------------
# spec:
#   core_preserved:
#     - Pieces are still assembled into the same LR-symmetric dense rectangle as in the source task.
#     - Output rectangle dimensions still come from the unique closest-to-square portrait factor pair
#       of the scattered-piece area, and the zigzag frame pieces still act as the key assembly anchors.
#   dst_change:
#     - The input now also contains one large hollow rectangular template, drawn as a 1-cell-thick ring.
#     - The scattered pieces do NOT fill that template in the input; they stay elsewhere on the canvas.
#     - After the original rectangle is reconstructed, it must be centered inside the template and merged
#       with the template's non-background pixels. The output is therefore the template patch, not the
#       bare assembled rectangle.
#   concrete_behavior_trace:
#     - Original: pieces -> reconstruct 8x5 dense rectangle -> return that 8x5 rectangle.
#     - Variant: pieces + visible 12x9 hollow ring template -> reconstruct the same 8x5 rectangle ->
#       center it at offset (2,2) inside the 12x9 ring -> return the 12x9 merged patch.
#     - Observable difference: the destination shape is no longer just the dense rectangle; the final
#       rectangle is spatially aligned to another visible object from the input.
#   complexity_calibration:
#     - The template is always a single hollow rectangle, which is visually obvious and solver-detectable.
#     - Its size is deterministically tied to the hidden assembled rectangle: template dims are (H+4, W+4),
#       so centering is visible but not overly free-form.
#     - This avoids degenerate cases where the template has no effect or where many different alignments
#       would be plausible.
#   implicit_assumption_audit:
#     - Original main() treated every non-bg component as a puzzle piece. That would now wrongly count the
#       template as a giant extra piece, corrupting total_area and frame detection. The variant therefore
#       must explicitly detect and remove the hollow template before running the old assembly logic.
#     - Original frame-locking code assumes that a candidate frame union has bbox exactly (H,W). This still
#       holds because the template is excluded before that search.
#     - Original output was returned immediately after assembly. That is no longer sufficient; after
#       assembly, the solver must verify that the template can contain the rectangle and then overlay it
#       at the centered offset.
#   placement_legality:
#     - Generator reserves the full template bbox and the full frame-group bbox before scattering other
#       pieces, so no object overlaps or clutters those structures.
#     - The template interior is left empty in the input; this makes the centering relation easy to read
#       and guarantees template detection by the solver.
#   visual_observability:
#     - Across examples, the user can see a large empty ring plus a separate smaller rectangular frame
#       assembled from pieces. The output always keeps the ring and inserts the completed rectangle into it.
#     - The rule depends on visible geometry only: hollow rectangle shape and geometric centering.
#   visual_cleanliness:
#     - The template is a single-color border with an empty interior, and scattered pieces are kept away
#       from its bbox. This avoids layered clutter and preserves a clean destination cue.
#   color_inference:
#     - No color-id-based ordering or recoloring rule is used. The template color is simply preserved as-is.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random
from collections import Counter


_TEMPLATE_MARGIN = 2


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


def _overlay_on_template(template_patch: np.ndarray, board: np.ndarray, bg: int) -> np.ndarray:
    out = template_patch.copy()
    TH, TW = out.shape
    H, W = board.shape
    assert TH >= H and TW >= W, "template must be large enough to contain the assembled board"
    dr = TH - H
    dc = TW - W
    assert dr == 2 * _TEMPLATE_MARGIN and dc == 2 * _TEMPLATE_MARGIN, "template and board sizes must use fixed centering margin"
    r0 = dr // 2
    c0 = dc // 2
    assert np.all(out[r0 : r0 + H, c0 : c0 + W] == bg), "template interior under centered board must be empty"
    out[r0 : r0 + H, c0 : c0 + W] = board
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle, then center it inside a template.

    Input structure:
    - bg = most common color; one large non-bg hollow rectangle is a destination template.
    - The remaining non-bg cells are the same jigsaw fragments as in the source task.
    - Optional anchor roles among those fragments still exist: 4 singleton corners, horizontal edge bars,
      and paired vertical edge lines.

    Rule:
    - Detect and remove the hollow rectangular template from the input.
    - Reconstruct the hidden dense LR-symmetric rectangle from the scattered fragments using the original
      edge/frame constraints.
    - Center that assembled rectangle inside the template and merge it with the template's non-bg pixels.

    Key insight: The template is not another puzzle piece; it is the destination frame. Solve the old
    rectangle-assembly problem first, then use geometric centering to place the result into the visible ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    tr0, tc0, tr1, tc1, _template_col = _find_template(g, bg)
    template_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    piece_grid = g.copy()
    piece_grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(piece_grid)))):
        if col == bg:
            continue
        comps = components_mask(piece_grid == col, connectivity=4)
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
    assert total_area > 0, "total_area must be positive: grid must contain non-template pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(piece_grid))) - {bg}):
        for pts in components_mask(piece_grid == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts, h, w):
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
    board = _assemble(types, H, W, req, disallow_touch)
    return _overlay_on_template(template_patch, board, bg).astype(int)


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

_GEN_MAX_RETRIES = 500


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


def _zigzag_frame(H):
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


def _fill_holes(grid, H, W, frame1, frame2, split, fill_pool):
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


def _build_board(H, W):
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


def _make_template_patch(board: np.ndarray, bg: int, template_col: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = board.shape
    TH = H + 2 * _TEMPLATE_MARGIN
    TW = W + 2 * _TEMPLATE_MARGIN
    template = np.full((TH, TW), bg, dtype=int)
    template[0, :] = template_col
    template[-1, :] = template_col
    template[:, 0] = template_col
    template[:, -1] = template_col
    out = template.copy()
    out[_TEMPLATE_MARGIN : _TEMPLATE_MARGIN + H, _TEMPLATE_MARGIN : _TEMPLATE_MARGIN + W] = board
    return template, out


def _make_output() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    abstract_board, abstract_fcs = _build_board(H, W)
    if abstract_board is None:
        return None, None, None, 0, set()

    bg = random.randint(0, 9)
    abstract = sorted(np.unique(abstract_board))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract) + 1:
        return None, None, None, 0, set()

    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    board = np.vectorize(lambda x: cmap[int(x)])(abstract_board)
    real_fcs = {cmap[a] for a in abstract_fcs}
    used = set(map(int, np.unique(board))) | {bg}
    template_col = next(c for c in available[len(abstract) :] if c not in used)
    template, out = _make_template_patch(board, bg, template_col)
    return board.astype(int), template.astype(int), out.astype(int), bg, real_fcs


def _extract_non_frame_patches(board, bg, fcs):
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


def _scatter(board: np.ndarray, template: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    bh, bw = board.shape
    th, tw = template.shape
    rows = random.randint(max(bh, th) + _CANVAS_EXTRA_MIN, max(bh, th) + _CANVAS_EXTRA_MAX)
    cols = random.randint(max(bw, tw) + _CANVAS_EXTRA_MIN, max(bw, tw) + _CANVAS_EXTRA_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    tr, tc = _reserve_rect(canvas, th, tw, bg)
    fr, fc = _reserve_rect(canvas, bh, bw, bg)

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

    canvas[tr : tr + th, tc : tc + tw] = bg
    paste(canvas, template, tr, tc, transparent=bg)

    canvas[fr : fr + bh, fc : fc + bw] = bg
    paste(canvas, _frame_patch(board, bg, fcs), fr, fc, transparent=bg)
    return canvas


def _piece_count(board: np.ndarray) -> int:
    return sum(len(components_mask(board == col, connectivity=4)) for col in np.unique(board))


def _gen_attempt() -> np.ndarray | None:
    board, template, out, bg, fcs = _make_output()
    if board is None or template is None or out is None:
        return None
    if not np.array_equal(board, board[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(board)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(board)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(board) for pts in components_mask(board == col, connectivity=4)):
        return None
    try:
        inp = _scatter(board, template, bg, fcs)
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
