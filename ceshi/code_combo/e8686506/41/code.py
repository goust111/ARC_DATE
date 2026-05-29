# Combo variant of e8686506 (index 41)
# combo_spec: D4/3+D18/0
# Name: 对角线对称+棋盘格排列

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# concrete_pixel_behavior:
#   original_task:
#     - infer a portrait rectangle
#     - find 1-2 LR-symmetric 8-connected frame parts spanning that rectangle
#     - pack the remaining pieces into the frame so the final board is LR-symmetric
#   combo_task:
#     - infer a square output
#     - keep the main diagonal as a preserved 8-connected clue chain
#     - keep only one side of one checkerboard parity inside that square bbox
#     - scatter the transpose-side cells of that parity, plus all complementary-parity
#       cells, as detached singleton fragments elsewhere
#     - reconstruct a dense square whose colors satisfy both:
#         1. board[r, c] == board[c, r]
#         2. every 4-neighbor pair has different colors, i.e. checkerboard alternation
#   concrete_6x6_trace:
#     - output uses three roles:
#         D on the main diagonal
#         A on every off-diagonal odd-parity cell
#         B on every off-diagonal even-parity cell
#     - input preserves D and, say, only the lower-triangle A cells inside one 6x6 bbox
#     - the upper-triangle A cells and all B cells are scattered outside the bbox
#     - compared with the source task, the preserved clue is now interior-heavy and
#       one-sided; diagonal symmetry is what tells the solver to complete the missing
#       transpose half, while the checkerboard rule tells it how the remaining parity
#       class must alternate around the diagonal.
#
# complexity_calibration:
#   - A full checkerboard anchor over the whole square would make diagonal symmetry
#     mostly redundant. Keeping only one triangular half of the anchor parity avoids
#     that collapse: the transpose rule now carries real information.
#   - The freedom is kept small on purpose: fixed 6x6 size, exactly three colors,
#     fixed off-diagonal anchor parity, and only two side choices (upper or lower).
#     This keeps 2-4 training examples human-readable while still making the fused
#     behavior visibly different from both the original frame puzzle and a plain crop.
#
# implicit_assumption_audit:
#   - The source solver assumed rectangular outputs; here the closest square factor
#     must be exact because diagonal symmetry needs H == W.
#   - The source frame search looked for LR-symmetric 8-connected parts. That premise
#     is invalid here, so the solver instead searches for an exact main-diagonal chain.
#   - The source corner/bar/vertical-line anchor heuristics are irrelevant and removed.
#   - The old "singletons only on the border" assumption is false twice over here:
#     the diagonal clue and the checkerboard fillers both create interior singletons.
#   - The old generic piece-packing machinery is unnecessary because every 4-connected
#     piece in this combo generator is a singleton; the meaningful structure lives in
#     the preserved 8-connected clue geometry, not in large 4-connected polyominoes.
#   - New assumptions introduced by the combo solver:
#       1. one 8-connected main-diagonal clue of length n spans an n x n bbox
#       2. all other preserved cells inside that bbox share one color, one parity,
#          and lie strictly on one side of the diagonal
#       3. exactly one remaining non-background color exists outside those clue roles
#     generate_input() enforces all three by construction.
#
# placement_legality:
#   - The generator reserves one 6x6 bbox on the canvas for the diagonal chain plus
#     one-sided checker clue. All scattered singleton cells are placed outside it.
#   - `random_free_location(..., padding=1, padding_connectivity=8)` keeps every
#     scattered singleton isolated, so no accidental merges or overlaps can occur.
#   - The reserved bbox is rewritten only once at the end, so clue cells never clash
#     with scattered cells.
#
# visual_observability_and_neatness:
#   - A human sees a square diagonal line plus a spotted triangular half already in
#     place, and many loose single pixels elsewhere. Across examples, the missing half
#     is always the transpose of the preserved half, not a left-right reflection.
#   - The output is visibly checkerboard-like because all 4-neighbor pairs differ, yet
#     the diagonal remains a distinct self-mapped stripe. That makes the fusion legible
#     without relying on hidden color ordering or non-visual metadata.
#
# solver_generator_contract:
#   - `main()` assumes the input follows the diagonal-chain + one-sided checker-hint
#     construction described above.
#   - `generate_input()` oracle-checks every sample with `main(inp) == out` and also
#     validates that the output is diagonal-symmetric, checkerboard-adjacent, and not
#     accidentally LR-symmetric.
# -----------------------------------------------------------------------------


_N = 6
_ANCHOR_PARITY = 1

_CANVAS_MIN = 20
_CANVAS_MAX = 24
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 300


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _infer_square_size(total_area: int) -> int:
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square output shape must be unique"
    h, w = best_dims[0]
    assert h == w, "this combo variant requires a square output"
    return int(h)


def _rel_coords(pts: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _, _ = bbox_coords(pts)
    return {(int(r) - r0, int(c) - c0) for r, c in pts}


def _find_diagonal_anchor(
    comps8_all: list[tuple[int, np.ndarray]],
    n: int,
) -> tuple[int, np.ndarray]:
    target = {(i, i) for i in range(n)}
    matches = []
    for col, pts in comps8_all:
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 + 1 != n or c1 - c0 + 1 != n or len(pts) != n:
            continue
        if _rel_coords(pts) == target:
            matches.append((int(col), pts))
    assert len(matches) == 1, "input must contain exactly one preserved main-diagonal clue"
    return matches[0]


def _is_checkerboard_adjacent(grid: np.ndarray) -> bool:
    return bool(
        np.all(grid[:, :-1] != grid[:, 1:])
        and np.all(grid[:-1, :] != grid[1:, :])
    )


def _build_output(diag_color: int, anchor_color: int, fill_color: int, anchor_parity: int) -> np.ndarray:
    out = np.full((_N, _N), fill_color, dtype=int)
    for r in range(_N):
        for c in range(_N):
            if r == c:
                out[r, c] = diag_color
            elif (r + c) % 2 == anchor_parity:
                out[r, c] = anchor_color
            else:
                out[r, c] = fill_color
    return out


def _hint_mask(side: str) -> np.ndarray:
    mask = np.zeros((_N, _N), dtype=bool)
    for i in range(_N):
        mask[i, i] = True
    for r in range(_N):
        for c in range(_N):
            if r == c or (r + c) % 2 != _ANCHOR_PARITY:
                continue
            if side == "upper" and r < c:
                mask[r, c] = True
            if side == "lower" and r > c:
                mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a square mosaic whose colors are symmetric across the main diagonal and alternate like a checkerboard.

    Input structure:
    - The background is the dominant color; every other cell belongs to a puzzle fragment.
    - One preserved 8-connected clue is an exact main-diagonal chain spanning the final square bbox.
    - Inside that same bbox, only one triangular half of one checkerboard parity is kept in place; the
      transpose-side cells of that parity and all complementary cells are scattered elsewhere as singletons.

    Rule:
    - Infer the square size from total non-background area and locate the preserved main diagonal.
    - Read the in-bbox half-checker clue to determine which off-diagonal parity/color is being hinted.
    - Fill the square so diagonal cells keep the diagonal clue color, all off-diagonal cells of the hinted
      parity use the hinted color on both sides of the diagonal, and the remaining parity uses the last color.

    Key insight: the preserved bbox does not show a complete checkerboard, only one side of it. The main
    diagonal fixes the orientation, and transpose symmetry tells you to copy that one-sided checker clue to
    the opposite triangle; then checkerboard alternation identifies the complementary parity class.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    total_area = int(np.sum(g != bg))
    assert total_area > 0, "grid must contain non-background pieces"
    n = _infer_square_size(total_area)
    assert n == _N, "generator and solver agree on a 6x6 output family"

    non_bg_colors = sorted(int(col) for col in np.unique(g) if int(col) != bg)
    assert len(non_bg_colors) == 3, "combo samples use exactly three non-background colors"

    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in non_bg_colors:
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((col, pts))

    diag_color, diag_pts = _find_diagonal_anchor(comps8_all, n)
    r0, c0, _, _ = bbox_coords(diag_pts)
    bbox = g[r0 : r0 + n, c0 : c0 + n]

    off_diag_clues: list[tuple[int, int, int]] = []
    for r in range(n):
        for c in range(n):
            col = int(bbox[r, c])
            if col == bg or r == c:
                continue
            off_diag_clues.append((r, c, col))
    assert off_diag_clues, "bbox must contain one-sided checker clues in addition to the diagonal"

    anchor_colors = {col for _, _, col in off_diag_clues}
    assert len(anchor_colors) == 1, "all preserved off-diagonal clue cells must share one color"
    anchor_color = int(next(iter(anchor_colors)))
    assert anchor_color != diag_color, "diagonal clue color must differ from checker clue color"

    side_signs = {1 if c > r else -1 for r, c, _ in off_diag_clues}
    assert len(side_signs) == 1, "all preserved checker clues must lie on the same side of the diagonal"

    anchor_parities = {(r + c) % 2 for r, c, _ in off_diag_clues}
    assert len(anchor_parities) == 1, "preserved checker clues must occupy one parity only"
    anchor_parity = int(next(iter(anchor_parities)))

    for r, c, _ in off_diag_clues:
        assert int(bbox[c, r]) == bg, "the transpose half of the checker hint must be absent in the bbox"

    fill_candidates = [col for col in non_bg_colors if col not in {diag_color, anchor_color}]
    assert len(fill_candidates) == 1, "exactly one complementary fill color must remain"
    fill_color = int(fill_candidates[0])

    expected = _build_output(diag_color, anchor_color, fill_color, anchor_parity)
    assert Counter(int(x) for x in g.flat if int(x) != bg) == Counter(int(x) for x in expected.flat), (
        "input color multiplicities must match the reconstructed checkerboard square"
    )
    assert np.array_equal(expected, expected.T), "reconstructed board must be diagonally symmetric"
    assert _is_checkerboard_adjacent(expected), "reconstructed board must alternate by 4-neighborhood"
    assert not np.array_equal(expected, expected[:, ::-1]), "combo output must stay distinct from the LR source task"
    return expected.astype(int)


def _make_output() -> tuple[np.ndarray, int]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    diag_color, anchor_color, fill_color = palette[:3]
    out = _build_output(diag_color, anchor_color, fill_color, _ANCHOR_PARITY)
    assert np.array_equal(out, out.T)
    assert _is_checkerboard_adjacent(out)
    assert not np.array_equal(out, out[:, ::-1])
    return out, bg


def _scatter(out: np.ndarray, bg: int, side: str) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    max_r = rows - _N - 2
    max_c = cols - _N - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for preserved clue bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + _N, off_c : off_c + _N] = -1

    hint_mask = _hint_mask(side)
    for r in range(_N):
        for c in range(_N):
            if hint_mask[r, c]:
                continue
            patch = np.array([[int(out[r, c])]], dtype=int)
            rr, cc = random_free_location(
                canvas,
                patch,
                transparent=bg,
                padding=_SCATTER_PADDING,
                padding_connectivity=8,
            )
            paste(canvas, patch, rr, cc, transparent=bg)

    bbox_region = canvas[off_r : off_r + _N, off_c : off_c + _N]
    bbox_region[:] = bg
    bbox_region[hint_mask] = out[hint_mask]
    return canvas


def _piece_count(grid: np.ndarray, bg: int) -> int:
    return sum(
        len(components_mask(grid == col, connectivity=4))
        for col in np.unique(grid)
        if int(col) != bg
    )


def _bbox_half_hint_score(inp: np.ndarray) -> float:
    bg = get_bg(inp)
    non_bg_colors = [int(col) for col in np.unique(inp) if int(col) != bg]
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in non_bg_colors:
        for pts in components_mask(inp == col, connectivity=8):
            comps8_all.append((col, pts))
    diag_color, diag_pts = _find_diagonal_anchor(comps8_all, _N)
    r0, c0, _, _ = bbox_coords(diag_pts)
    bbox = inp[r0 : r0 + _N, c0 : c0 + _N]

    same_side = 0
    mirror_side = 0
    for r in range(_N):
        for c in range(_N):
            if r == c:
                continue
            col = int(bbox[r, c])
            if col == bg or col == diag_color:
                continue
            if (r < c and int(bbox[c, r]) == bg) or (r > c and int(bbox[c, r]) == bg):
                same_side += 1
            else:
                mirror_side += 1
    total = same_side + mirror_side
    return 0.0 if total == 0 else same_side / total


def _gen_attempt() -> np.ndarray | None:
    out, bg = _make_output()
    side = random.choice(["upper", "lower"])
    try:
        inp = _scatter(out, bg, side)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    if _piece_count(inp, bg) != 36:
        return None
    if _bbox_half_hint_score(inp) != 1.0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
