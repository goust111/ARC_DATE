# Combo variant of e8686506 (index 3)
# combo_spec: D4/1+D5/0
# Name: 双轴对称+8-连通禁同色

# Design analysis
# yaml:
#   concrete_behavior:
#     output:
#       - The assembled rectangle is always 7x5 and obeys both LR and UD reflection.
#         Every sample contains a visible 11-cell monochrome cross frame that touches
#         all four sides through the middle row and middle column.
#       - Outside the cross, the four corner quadrants are filled by one of two small
#         libraries. In "top_lift" mode, each quadrant contains a vertical domino near
#         the top/bottom edge plus an oriented L-triomino near the cross. In
#         "bottom_lift" mode, the domino family is replaced by eight border singletons
#         and the L-triominoes move upward/downward accordingly.
#       - The strengthened D5 rule is visible in the finished board: when one color is
#         reused across multiple pieces, any two distinct same-color components are kept
#         at Chebyshev distance >= 2. Diagonal touching never occurs.
#     input:
#       - The 11-cell cross frame stays grouped as one untranslated patch inside the
#         input, preserving the "recognizable anchor" idea from the source task.
#       - All non-frame pieces are scattered separately with 8-neighborhood padding, so
#         the solver must recover the compact rectangle from disconnected fragments.
#     trace_example:
#       - Original logic: after finding a frame, a top-left fragment only needed to
#         respect left-right mirroring; an equally shaped counterpart could still float
#         between an upper slot and a lower slot if both preserved LR symmetry.
#       - Fused logic here: the chosen library fixes four quadrant roles at once. A
#         fragment placed in the upper half immediately determines the lower-half mirror,
#         and the final board is rejected unless every repeated color also stays clear of
#         diagonal same-color contact. The dual-axis rule and the 8-neighbor rule act on
#         the same quadrant inventory, rather than as unrelated post-filters.
#   complexity_calibration:
#     - The rectangle remains compact and puzzle-like, but the grouped frame is now a
#       cross rather than the old zigzag border because diagonal same-color chaining is
#       illegal under D5/0.
#     - Only two quadrant libraries are used. That is enough variation for training
#       examples to show that multiple fused layouts exist, while keeping the object
#       vocabulary small and learnable.
#     - Piece counts stay in the original visual band: 13 pieces in top_lift mode and
#       17 pieces in bottom_lift mode. Colors stay at exactly four.
#   implicit_assumption_review:
#     - Old _assemble() assumed LR reflection was the only global orbit constraint.
#       That no longer holds; the fused solver must reason in full 4-way orbits.
#     - Old sparse-frame generation relied on diagonal same-color contacts to keep a
#       single color visually connected under 8-connectivity. That violates D5/0, so the
#       fused generator replaces it with a single 4-connected cross component.
#     - The source solver inferred arbitrary output shapes from area. Here the frame
#       library is deliberately fixed to area 35, so the solver now assumes the unique
#       closest-to-square factorization is 7x5 and the generator guarantees that.
#     - Because the quadrant library is fixed, the solver assumes exactly one cross
#       component of area 11, exactly one 4-corner singleton color, and then one of two
#       remaining inventories: {4 vertical dominoes + 4 oriented L-triominoes} or
#       {8 border singletons + 4 oriented L-triominoes}. The generator enforces this.
#   placement_legality:
#     - The generator builds the final 7x5 board first, validates dual symmetry and the
#       8-neighbor same-color gap, extracts each 4-connected component as an immutable
#       piece, then scatters them onto a 13-15 square canvas with Chebyshev padding 1.
#     - The cross frame is pasted as one grouped patch inside a blocked bbox, so other
#       pieces never overlap it or leak into the final rectangle interior.
#   observability:
#     - Humans can see the fused change from a few examples because the output is not
#       merely LR-symmetric: the top half and bottom half are exact mirrors, and the
#       repeated colors form a checkerboard-like spacing with no diagonal same-color
#       kisses.
#     - The two modes visibly differ in where the non-corner border fragments sit: one
#       mode has four 2-cell vertical bars, the other has eight isolated border pixels.
#       That repeated contrast provides enough support for inferring the library.
#   cleanliness:
#     - The output has three clearly separated layers: four corner anchors, one central
#       cross frame, and one quadrant-piece library. No region carries more than one
#       structural role.
#   inferability:
#     - Color remapping is sample-local and unconstrained by numeric ids. The only
#       stable clues are piece shape and position relative to the frame, which are both
#       visually available to a human solver.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


H_OUT = 7
W_OUT = 5
TOTAL_AREA = H_OUT * W_OUT

MODE_TOP = "top_lift"
MODE_BOTTOM = "bottom_lift"

FRAME_ABS = 9
CORNER_ABS = 1
MID_ABS = 2
AUX_ABS = 3

TOP_TEMPLATE = np.array(
    [
        [CORNER_ABS, MID_ABS, FRAME_ABS, MID_ABS, CORNER_ABS],
        [AUX_ABS, MID_ABS, FRAME_ABS, MID_ABS, AUX_ABS],
        [AUX_ABS, AUX_ABS, FRAME_ABS, AUX_ABS, AUX_ABS],
        [FRAME_ABS, FRAME_ABS, FRAME_ABS, FRAME_ABS, FRAME_ABS],
        [AUX_ABS, AUX_ABS, FRAME_ABS, AUX_ABS, AUX_ABS],
        [AUX_ABS, MID_ABS, FRAME_ABS, MID_ABS, AUX_ABS],
        [CORNER_ABS, MID_ABS, FRAME_ABS, MID_ABS, CORNER_ABS],
    ],
    dtype=int,
)

BOTTOM_TEMPLATE = np.array(
    [
        [CORNER_ABS, AUX_ABS, FRAME_ABS, AUX_ABS, CORNER_ABS],
        [MID_ABS, MID_ABS, FRAME_ABS, MID_ABS, MID_ABS],
        [AUX_ABS, MID_ABS, FRAME_ABS, MID_ABS, AUX_ABS],
        [FRAME_ABS, FRAME_ABS, FRAME_ABS, FRAME_ABS, FRAME_ABS],
        [AUX_ABS, MID_ABS, FRAME_ABS, MID_ABS, AUX_ABS],
        [MID_ABS, MID_ABS, FRAME_ABS, MID_ABS, MID_ABS],
        [CORNER_ABS, AUX_ABS, FRAME_ABS, AUX_ABS, CORNER_ABS],
    ],
    dtype=int,
)

CROSS_SIG = (
    7,
    5,
    (
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (5, 2),
        (6, 2),
    ),
)
VERT_DOMINO_SIG = (2, 1, ((0, 0), (1, 0)))
SINGLETON_SIG = (1, 1, ((0, 0),))
L_SIGS = {
    (2, 2, ((0, 0), (1, 0), (1, 1))),
    (2, 2, ((0, 1), (1, 0), (1, 1))),
    (2, 2, ((0, 0), (0, 1), (1, 0))),
    (2, 2, ((0, 0), (0, 1), (1, 1))),
}

TOP_DOMINO_CELLS = {
    (0, 1),
    (1, 1),
    (0, 3),
    (1, 3),
    (5, 1),
    (6, 1),
    (5, 3),
    (6, 3),
}
TOP_L_CELLS = {
    (1, 0),
    (2, 0),
    (2, 1),
    (1, 4),
    (2, 3),
    (2, 4),
    (4, 0),
    (4, 1),
    (5, 0),
    (4, 3),
    (4, 4),
    (5, 4),
}
BOTTOM_SINGLETON_CELLS = {
    (0, 1),
    (0, 3),
    (2, 0),
    (2, 4),
    (4, 0),
    (4, 4),
    (6, 1),
    (6, 3),
}
BOTTOM_L_CELLS = {
    (1, 0),
    (1, 1),
    (2, 1),
    (1, 3),
    (1, 4),
    (2, 3),
    (4, 1),
    (5, 0),
    (5, 1),
    (4, 3),
    (5, 3),
    (5, 4),
}

# --------------- generator config ---------------
_N_COLORS = 4
_MAX_PIECE_AREA = 11
_PIECE_COUNT_ALLOWED = {13, 17}
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 200


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


def _cross_cells() -> set[tuple[int, int]]:
    cells = {(r, 2) for r in range(H_OUT)}
    cells |= {(3, c) for c in range(W_OUT)}
    return cells


def _corner_cells() -> set[tuple[int, int]]:
    return {(0, 0), (0, W_OUT - 1), (H_OUT - 1, 0), (H_OUT - 1, W_OUT - 1)}


def _build_output(mode: str) -> np.ndarray:
    if mode == MODE_TOP:
        out = TOP_TEMPLATE.copy()
    elif mode == MODE_BOTTOM:
        out = BOTTOM_TEMPLATE.copy()
    else:
        raise AssertionError(f"unknown mode: {mode}")
    return out


def _strict_dual_symmetry(grid: np.ndarray) -> bool:
    return np.array_equal(grid, grid[:, ::-1]) and np.array_equal(grid, grid[::-1, :])


def _same_color_component_gap_summary(grid: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for col in sorted(map(int, np.unique(grid))):
        comps = components_mask(grid == col, connectivity=4)
        if len(comps) <= 1:
            continue
        best = 10**9
        comp_pts = [[(int(r), int(c)) for r, c in pts] for pts in comps]
        for i, a in enumerate(comp_pts):
            for b in comp_pts[i + 1 :]:
                gap = min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a for r2, c2 in b)
                best = min(best, gap)
        out[col] = best
    return out


def _assert_output_valid(out: np.ndarray) -> None:
    assert out.shape == (H_OUT, W_OUT)
    assert _strict_dual_symmetry(out), "output must be dual-symmetric"
    gaps = _same_color_component_gap_summary(out)
    assert all(gap >= 2 for gap in gaps.values()), "same-color components must be 8-separated"
    pc = sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))
    assert pc in _PIECE_COUNT_ALLOWED, "piece count must match one of the library modes"
    assert all(
        len(pts) <= _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ), "piece area exceeds generator budget"


def _classify_output(out: np.ndarray) -> str:
    sig_counter = Counter()
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            sig_counter[_piece_sig(pts)] += 1
    if sig_counter[CROSS_SIG] != 1:
        raise AssertionError("expected one cross frame component")
    if sig_counter[VERT_DOMINO_SIG] == 4:
        return MODE_TOP
    if sig_counter[SINGLETON_SIG] == 12:
        return MODE_BOTTOM
    raise AssertionError("output does not match either fused mode")


def _make_output() -> tuple[np.ndarray, int, int]:
    mode = random.choice([MODE_TOP, MODE_BOTTOM])
    out = _build_output(mode)
    _assert_output_valid(out)
    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    cmap = {
        CORNER_ABS: available[0],
        MID_ABS: available[1],
        AUX_ABS: available[2],
        FRAME_ABS: available[3],
    }
    real = np.vectorize(lambda x: cmap[int(x)])(out)
    return real.astype(int), bg, int(cmap[FRAME_ABS])


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_color: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col == frame_color:
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


def _scatter(out: np.ndarray, bg: int, frame_color: int) -> np.ndarray:
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

    for patch in _extract_non_frame_patches(out, bg, frame_color):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[out == frame_color] = frame_color
    return canvas


def _sig_multiset_by_color(grid: np.ndarray, bg: int) -> dict[int, Counter]:
    out: dict[int, Counter] = {}
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        ctr = Counter()
        for pts in components_mask(grid == col, connectivity=4):
            ctr[_piece_sig(pts)] += 1
        out[int(col)] = ctr
    return out


def _reconstruct_from_inventory(grid: np.ndarray) -> np.ndarray:
    bg = get_bg(grid)
    sigs_by_color = _sig_multiset_by_color(grid, bg)

    total_area = sum(sum(len(pts) for pts in components_mask(grid == col, connectivity=4)) for col in sigs_by_color)
    assert total_area == TOTAL_AREA, "fused generator always uses area 35"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert best_dims == [(H_OUT, W_OUT)], "expected the unique closest-to-square portrait factorization 7x5"

    frame_colors = [col for col, ctr in sigs_by_color.items() if ctr == Counter({CROSS_SIG: 1})]
    assert len(frame_colors) == 1, "need exactly one cross-frame color"
    frame_color = frame_colors[0]

    corner_colors = [col for col, ctr in sigs_by_color.items() if col != frame_color and ctr == Counter({SINGLETON_SIG: 4})]
    assert len(corner_colors) == 1, "need exactly one 4-corner singleton color"
    corner_color = corner_colors[0]

    remaining = {col: ctr for col, ctr in sigs_by_color.items() if col not in {frame_color, corner_color}}
    assert len(remaining) == 2, "expected exactly two non-frame, non-corner colors"

    out = np.full((H_OUT, W_OUT), -1, dtype=int)
    for r, c in _cross_cells():
        out[r, c] = frame_color
    for r, c in _corner_cells():
        out[r, c] = corner_color

    mode = None
    mid_color = None
    aux_color = None
    for col, ctr in remaining.items():
        if ctr == Counter({VERT_DOMINO_SIG: 4}):
            mode = MODE_TOP
            mid_color = col
        elif ctr == Counter({SINGLETON_SIG: 8}):
            mode = MODE_BOTTOM
            aux_color = col
    if mode == MODE_TOP:
        assert aux_color is None
        other = [col for col in remaining if col != mid_color]
        assert len(other) == 1
        aux_color = other[0]
        assert remaining[aux_color] == Counter({sig: 1 for sig in L_SIGS}), "top mode needs one of each L orientation"
        for r, c in TOP_DOMINO_CELLS:
            out[r, c] = mid_color
        for r, c in TOP_L_CELLS:
            out[r, c] = aux_color
    else:
        assert aux_color is not None, "bottom mode requires the 8-singleton color"
        other = [col for col in remaining if col != aux_color]
        assert len(other) == 1
        mid_color = other[0]
        assert remaining[mid_color] == Counter({sig: 1 for sig in L_SIGS}), "bottom mode needs one of each L orientation"
        for r, c in BOTTOM_SINGLETON_CELLS:
            out[r, c] = aux_color
        for r, c in BOTTOM_L_CELLS:
            out[r, c] = mid_color

    assert not np.any(out < 0), "template reconstruction left holes"
    assert _classify_output(out) == mode
    _assert_output_valid(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a compact rectangle that is mirror-symmetric on both axes.

    Input structure:
    - The dominant color is background; every other 4-connected monochrome component is one rigid puzzle fragment.
    - One fragment is always a grouped 11-cell cross that already spans the future middle row and middle column.
    - The remaining fragments are scattered corner pixels plus a small quadrant library of dominoes, L-triominoes,
      or border singletons, depending on the sample mode.

    Rule:
    - Recover the dense 7x5 rectangle by identifying the cross frame, the unique corner color, and which of the two
      quadrant libraries is present.
    - Place every fragment without rotation or reflection so the completed board is simultaneously LR- and UD-symmetric.
    - Enforce the strengthened adjacency rule: distinct components of the same color may not touch even diagonally in
      the assembled board.

    Key insight:
    - The puzzle is no longer solved by a single-axis mirror alone. The frame fixes the center cross, the remaining
      inventory chooses one of two four-quadrant templates, and the final legality check rejects any arrangement that
      would let repeated colors meet in an 8-neighborhood.
    """
    g = np.asarray(grid, dtype=int)
    out = _reconstruct_from_inventory(g)
    return out.astype(int)


def _gen_attempt() -> np.ndarray | None:
    out, bg, frame_color = _make_output()
    try:
        inp = _scatter(out, bg, frame_color)
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
