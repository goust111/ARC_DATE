# Variant of e8686506
# Dimension: D15 — 符号指令：图例或提示标记 (symbol_instructions)
# Change: 无图例或符号指令——所有信息从碎片的几何关系和对称性推断... → 小型模板展示目标形状轮廓

# Design analysis
# ---
# spec:
#   core_change:
#     - Input now contains a monochrome hollow-rectangle template in one corner.
#     - The template bbox is exactly the target output size HxW and its interior stays bg.
#     - Solver must read the template first, remove it from the puzzle inventory, then assemble
#       the remaining fragments exactly as in the original task.
#   why_d15_is_still_nontrivial_without_other_rule_changes:
#     - The corner template is an extra non-piece object. If the solver ignores it and treats all
#       non-bg cells as puzzle fragments, total area and component structure become wrong.
#     - So D15 matters operationally even though the assembled rectangle family stays the same:
#       the new solver must first detect and erase the symbol before applying the old logic.
#   concrete_visualization:
#     - Example: for an 8x5 target, the input may have an 8x5 one-cell-thick outline sitting
#       flush to the top-left corner. Elsewhere, the colored frame fragment(s) and interior
#       pieces are scattered as before. The output is still a dense LR-symmetric 8x5 rectangle.
#     - Original flow at the size-selection step:
#         treat every non-bg component as a piece -> total area is polluted by the template -> fail.
#       Variant flow:
#         detect corner outline -> erase it from the piece inventory -> run the original size and
#         assembly logic on the remaining fragments.
#       Difference:
#         the symbol is now a required pre-processing clue rather than decorative noise.
#   complexity_calibration:
#     - Template shape has only one role and one appearance class: corner-anchored hollow rectangle.
#     - Output size family remains the original 5x5 / 6x5 / ... / 10x5 regime, so there is no
#       unrelated second novelty beyond the template itself.
#     - The original frame/jigsaw/symmetry mechanics are preserved, so D15 stays the primary edit.
#   implicit_assumption_audit:
#     - Original `main()` assumed all non-bg components are puzzle pieces. No longer true:
#       template cells must be excluded before counting components or total area.
#     - Original `main()` assumed the total-area heuristic sees only real pieces. No longer true:
#       template cells must be removed before that heuristic is allowed to run.
#     - Original frame search could accidentally accept the template because it is LR-symmetric
#       and has the target bbox. No longer safe: frame search must run on the template-erased grid.
#     - Original requirement extraction over all colors could be polluted by the template color.
#       It now runs on the template-erased component map only.
#   placement_legality:
#     - Generator reserves the entire template bbox while scattering other pieces, so the hollow
#       interior stays empty and no fragment overlaps or touches the symbol.
#     - The frame bbox is reserved separately before other pieces are scattered, preserving the
#       original "frame as a grouped clue" behavior while keeping it disjoint from the template.
#   visual_observability:
#     - The corner outline is visually distinct from ordinary fragments because it is hollow,
#       monochrome, corner-flush, and not part of the scattered assembly.
#     - The symbol contributes a visibly separate object class instead of a hidden metadata flag.
#   visual_cleanliness:
#     - Template sits alone in a corner with an enforced one-cell buffer on the inward sides.
#     - Other pieces never occupy the template interior or overlap its bbox, preventing clutter.
#   color_reasoning:
#     - Template color is arbitrary but unique to the template and does not appear in the output.
#       No color-id ordering or hidden color arithmetic is used anywhere.
#   new_solver_assumptions_guaranteed_by_generator:
#     - Exactly one corner-anchored hollow rectangle template exists.
#     - After removing template cells, the remaining colored area equals the true output area H*W.
#     - The remaining scattered fragments still satisfy the original propagation-solvable assembly.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random


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


def _is_rect_outline_component(pts: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if len(pts) != (2 * h + 2 * w - 4):
        return False
    rel = {(int(r) - r0, int(c) - c0) for r, c in pts}
    for rr in range(h):
        for cc in range(w):
            want = (rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1)
            if ((rr, cc) in rel) != want:
                return False
    return True


def _find_template(g: np.ndarray, bg: int) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    H, W = g.shape
    all_non_bg = int(np.count_nonzero(g != bg))
    candidates = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            if not _is_rect_outline_component(pts):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            touches_tb = (r0 == 0) or (r1 == H - 1)
            touches_lr = (c0 == 0) or (c1 == W - 1)
            if not (touches_tb and touches_lr):
                continue
            th = int(r1 - r0 + 1)
            tw = int(c1 - c0 + 1)
            if all_non_bg - len(pts) != th * tw:
                continue
            candidates.append((col, pts, th, tw))
    assert len(candidates) <= 1, "template outline must be unique"
    if not candidates:
        return None, None
    _, pts, th, tw = candidates[0]
    mask = np.zeros_like(g, dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask, (th, tw)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle using a corner outline hint.

    Input structure:
    - bg is the most frequent color; one corner also contains a monochrome hollow rectangle template.
    - The template's bounding box gives the target output height and width, but carries no color detail.
    - All other non-bg components are the same jigsaw fragments as in the original task, including optional corner/edge anchors.

    Rule:
    - Detect the corner template, read its bbox as the target rectangle size, and ignore it as a puzzle piece.
    - Collect the remaining connected components as fragments and preserve any edge-anchor clues they imply.
    - Pack the fragments into a dense rectangle of the templated size so the final board is exactly LR-symmetric and same-color copies never 4-touch.

    Key insight: The template is not a fragment to place; once that corner outline is stripped away, the remaining pieces obey the original frame, symmetry, and no-touch constraints and assemble uniquely.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    template_mask, template_dims = _find_template(g, bg)
    piece_grid = g.copy()
    if template_mask is not None:
        piece_grid[template_mask] = bg

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

    from collections import Counter

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    if template_dims is not None:
        H, W = template_dims
        assert total_area == H * W, "piece area must match the corner template dimensions"
    else:
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
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W:
            if _is_lr_symmetric([(c1, p1)], H, W):
                frame_parts = [(c1, p1)]
                break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals template dims"

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


# --------------- generator (template-guided variant) ---------------

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

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 16
_CANVAS_MAX = 19
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _valid_dims():
    """Precompute valid (H, W) pairs: unique closest-to-square portrait factorization with W=5."""
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


def _zigzag_frame(H: int):
    """Generate frame bitmap (Hx2 left-half) via zigzag path from top to bottom."""
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


def _fill_holes(grid, H, W, frame1, frame2, split, fill_pool):
    """Fill non-frame cells by assigning colors to 4-conn hole regions."""
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i:i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
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


def _build_output(H: int, W: int):
    """Build an abstract LR-symmetric output grid with zigzag frame and filled holes."""
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

    half = W // 2
    if start_c == 1 and random.random() < 0.5:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < 0.5:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)

    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int], int]:
    """Select dims, build abstract output, then remap to real ARC colors and a template color."""
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set(), 0

    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract) + 1:
        return None, 0, set(), 0

    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    template_col = int(available[len(abstract)])
    real_fcs = {cmap[a] for a in abstract_fcs}
    real_out = np.vectorize(lambda x: cmap[int(x)])(out)
    return real_out, bg, real_fcs, template_col


def _extract_non_frame_patches(out, bg, fcs):
    """Extract non-frame piece patches from the output grid, sorted largest-first."""
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


def _template_patch(H: int, W: int, bg: int, col: int) -> np.ndarray:
    patch = np.full((H, W), bg, dtype=int)
    patch[0, :] = col
    patch[H - 1, :] = col
    patch[:, 0] = col
    patch[:, W - 1] = col
    return patch


def _corner_origins(rows: int, cols: int, h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, cols - w),
        (rows - h, 0),
        (rows - h, cols - w),
    ]


def _scatter(out: np.ndarray, bg: int, fcs: set[int], template_col: int) -> np.ndarray:
    """Place pieces onto a canvas with a reserved corner template and grouped frame."""
    oh, ow = out.shape
    rows_lo = max(_CANVAS_MIN, oh + 6)
    cols_lo = max(_CANVAS_MIN, ow + 8)
    rows = random.randint(rows_lo, _CANVAS_MAX)
    cols = random.randint(cols_lo, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    template = _template_patch(oh, ow, bg, template_col)
    tr, tc = random.choice(_corner_origins(rows, cols, oh, ow))
    canvas[tr : tr + oh, tc : tc + ow] = -1

    block = np.full((oh, ow), -1, dtype=int)
    fr, fc = random_free_location(
        canvas,
        block,
        transparent=bg,
        border=1,
        padding=_SCATTER_PADDING,
        padding_connectivity=8,
    )
    paste(canvas, block, fr, fc, transparent=bg)

    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            border=1,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[tr : tr + oh, tc : tc + ow] = bg
    paste(canvas, template, tr, tc, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[fr : fr + oh, fc : fc + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, validate constraints, scatter, oracle check."""
    out, bg, fcs, template_col = _make_output()
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
        inp = _scatter(out, bg, fcs, template_col)
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
    raise RuntimeError("failed to generate a valid e8686506 D15 input")
