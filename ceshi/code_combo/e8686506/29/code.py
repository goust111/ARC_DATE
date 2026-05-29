# Combo variant of e8686506 (index 29)
# combo_spec: D6/2+D7/1
# Name: 边界变色+加边框
#
# Design analysis
# ----------------
# spec:
#   preserved_core:
#     - input is still a background canvas with scattered monochrome fragments
#     - fragments still assemble into one dense left-right symmetric rectangle
#     - frame anchors, closest-to-square factorization, and propagation-only packing stay unchanged
#   transform:
#     - first reconstruct the original assembled rectangle in the fragments' native colors
#     - then recolor every cell that 4-touches a differently colored assembled neighbor to a fixed seam color
#     - after seam recoloring, embed that rectangle inside a new 1-cell outer ring of another fixed color
#     - only piece-to-piece contacts inside the assembled rectangle turn grey; the added outer frame never causes greying
#   visibility:
#     - seam color is fixed to grey (5) and the outer border color is fixed to 7
#     - both reserved colors are absent from the input pieces and from the pre-transform assembly
#     - generator rejects cases with too few non-seam survivors, because an almost-all-grey interior would hide the two-layer rule
#
# a) Concrete behavior and trace
#   - Original flow:
#       input fragments -> assemble dense symmetric rectangle -> output assembled colors unchanged
#   - Combo flow:
#       input fragments -> assemble the same dense symmetric rectangle -> mark contact-band cells grey -> wrap in a 1-cell frame
#   - Small trace:
#       base assembly:
#         2 2 3
#         2 4 3
#       seam-marked interior:
#         2 5 5
#         5 5 5
#       final output:
#         7 7 7 7 7
#         7 2 5 5 7
#         7 5 5 5 7
#         7 7 7 7 7
#     so the human must distinguish "grey = internal join" from "7 = exterior frame".
#   - Generator impact:
#       the scattered input must still be cut from the untransformed base assembly.
#       If pieces were cut from the greyed or bordered output, seam/frame cells would become fake fragments and the task
#       would no longer be "assemble first, then apply two output-only decorations".
#
# b) Complexity calibration
#   - The base assembly remains the same nontrivial propagation puzzle; the combo only changes the postprocess.
#   - Grey and the border color are both fixed globally instead of sampled per-instance, so 2-3 examples are enough
#     to establish the stable semantics "grey = seam" and "7 = outer ring".
#   - Generator keeps the original irregular frame-and-hole structure, but rejects outputs where too few original-color
#     cells survive after seam painting. That avoids degenerate interiors where the added ring is visible but the seam rule is not.
#
# c) Implicit-assumption audit for main()
#   - Piece extraction, type grouping, dimension inference, and frame detection all rely on the input fragments,
#     not on the transformed output. These assumptions remain valid because the input itself is unchanged.
#   - _build_requirements() still writes original colors into req. This is correct because seam recoloring happens
#     strictly after assembly; forcing grey during packing would destroy the source task's anchors.
#   - _assemble() still uses color equality as a proxy for "same piece family". The old guarantee that distinct
#     pieces of the same color never 4-touch must remain true, otherwise color differences would no longer identify
#     all seam contacts. The generator preserves that constraint exactly.
#   - The original factorization computes the tight assembled rectangle size from piece area. That still gives the
#     inner rectangle only; the added 1-cell border must be applied afterward instead of being folded into area inference.
#   - Returning the raw assembled board is no longer correct. The new postprocess must run after all placement asserts:
#     first seam recoloring on the inner board, then border padding around that result.
#
# d) Placement legality
#   - Scattering stays unchanged: frame pieces occupy one preserved bbox, all other pieces are placed outside with
#     Chebyshev padding 1, so they never overlap or merge accidentally.
#   - The oracle compares main(input) with the bordered-and-greyed output, while scattering still uses the base output.
#     This guarantees both output-only effects are actually exercised.
#
# e) Visual observability
#   - Human-facing cue: every example adds two stable colors absent from the input.
#     Grey draws irregular internal joins; 7 always forms a perfectly rectangular outer ring.
#   - Because the seam rule depends only on visible adjacency and the border rule depends only on perimeter position,
#     the two decorations are separable and inferable.
#
# f) Visual cleanliness
#   - Grey and 7 are both absent from the input and reserved from the base output palette, so seams and the frame do
#     not compete with ordinary fragments.
#   - The frame is a single uniform ring outside the assembled rectangle, so there is no overlap ambiguity between
#     the two boundary layers.
#   - Samples with fewer than three surviving non-grey cells or fewer than two surviving original colors are rejected.
#
# g) Color-change inferability
#   - The only color changes are "contact-band -> grey" and "new outer perimeter -> 7".
#   - No per-piece lookup tables or id-dependent swaps are used.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random


_SEAM_COLOR = 5
_OUTER_BORDER_COLOR = 7
_MIN_SURVIVING_COLOR_CELLS = 3
_MIN_SURVIVING_COLORS = 2


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


def _solve_base_assembly(grid: np.ndarray) -> np.ndarray:
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


def _contact_seam_mask(board: np.ndarray) -> np.ndarray:
    """
    Cells that participate in a 4-neighbor contact with a differently colored cell.

    Because the base puzzle forbids distinct same-color pieces from 4-touching, color inequality is a
    sufficient proxy for "this edge is a seam between two fragments".
    """
    H, W = board.shape
    mask = np.zeros((H, W), dtype=bool)
    diff_ud = board[1:, :] != board[:-1, :]
    diff_lr = board[:, 1:] != board[:, :-1]
    mask[1:, :] |= diff_ud
    mask[:-1, :] |= diff_ud
    mask[:, 1:] |= diff_lr
    mask[:, :-1] |= diff_lr
    return mask


def _apply_seam_transform(board: np.ndarray, seam_color: int = _SEAM_COLOR) -> np.ndarray:
    out = np.array(board, copy=True)
    seam_mask = _contact_seam_mask(board)
    out[seam_mask] = seam_color
    return out


def _add_outer_border(inner: np.ndarray, border_color: int = _OUTER_BORDER_COLOR) -> np.ndarray:
    h, w = inner.shape
    out = np.full((h + 2, w + 2), int(border_color), dtype=int)
    out[1:-1, 1:-1] = inner
    return out


def _apply_combo_transform(board: np.ndarray) -> np.ndarray:
    return _add_outer_border(_apply_seam_transform(board))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense left-right symmetric rectangle, then decorate it with two boundary layers.

    Input structure:
    - The background is the most frequent color; every other color belongs to an original fragment from the unmodified assembly.
    - Neither output-only boundary color appears in the input: grey is reserved for internal joins, and 7 is reserved for the final outer frame.
    - Optional corner markers, horizontal edge bars, and paired vertical edge lines are still encoded with the fragments' native colors.

    Rule:
    - First recover the same tight left-right symmetric rectangle as the source task by packing all non-background components.
    - Then repaint each assembled cell that 4-touches a differently colored assembled neighbor to grey, exposing inter-piece seams.
    - Finally surround that transformed rectangle with a new 1-cell border of color 7, increasing the output by 2 rows and 2 columns.

    Key insight:
    - The scattered pieces still determine only the inner assembly. Both visible boundary layers are deterministic postprocesses:
      grey comes from local piece contacts inside the assembly, while 7 marks the brand-new exterior perimeter.
    """
    base = _solve_base_assembly(grid)
    return _apply_combo_transform(base).astype(int)


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

# === CFG: Retry ===
_GEN_MAX_RETRIES = 4000


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
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool):
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


def _build_base_output(H, W):
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

    _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool)

    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_base_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_base_output(H, W)
    if out is None:
        return None, 0, set()
    bg_choices = [c for c in range(10) if c not in {_SEAM_COLOR, _OUTER_BORDER_COLOR}]
    bg = random.choice(bg_choices)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c not in {bg, _SEAM_COLOR, _OUTER_BORDER_COLOR}]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _extract_non_frame_patches(base_out, bg, fcs):
    patches = []
    for col in sorted(map(int, np.unique(base_out))):
        if col in fcs:
            continue
        for pts in components_mask(base_out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(base_out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = base_out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(base_out, bg, fcs):
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
        frame_mask |= base_out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = base_out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _transformed_output_ok(base_out: np.ndarray, transformed: np.ndarray) -> bool:
    seam_mask = transformed == _SEAM_COLOR
    if not np.any(seam_mask):
        return False
    surviving = ~seam_mask
    if np.count_nonzero(surviving) < _MIN_SURVIVING_COLOR_CELLS:
        return False
    surviving_colors = set(map(int, np.unique(transformed[surviving])))
    if len(surviving_colors) < _MIN_SURVIVING_COLORS:
        return False
    if np.array_equal(base_out, transformed):
        return False
    return True


def _combo_output_ok(base_out: np.ndarray, combo_out: np.ndarray) -> bool:
    expected_shape = (base_out.shape[0] + 2, base_out.shape[1] + 2)
    if combo_out.shape != expected_shape:
        return False
    if not bool((combo_out[0, :] == _OUTER_BORDER_COLOR).all()):
        return False
    if not bool((combo_out[-1, :] == _OUTER_BORDER_COLOR).all()):
        return False
    if not bool((combo_out[:, 0] == _OUTER_BORDER_COLOR).all()):
        return False
    if not bool((combo_out[:, -1] == _OUTER_BORDER_COLOR).all()):
        return False
    return _transformed_output_ok(base_out, combo_out[1:-1, 1:-1])


def _gen_attempt() -> np.ndarray | None:
    base_out, bg, fcs = _make_base_output()
    if base_out is None:
        return None
    if not np.array_equal(base_out, base_out[:, ::-1]):
        return None
    if {_SEAM_COLOR, _OUTER_BORDER_COLOR} & set(map(int, np.unique(base_out))):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(base_out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(base_out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(base_out)
        for pts in components_mask(base_out == col, connectivity=4)
    ):
        return None

    transformed = _apply_combo_transform(base_out)
    if not _combo_output_ok(base_out, transformed):
        return None

    try:
        inp = _scatter(base_out, bg, fcs)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, transformed):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
