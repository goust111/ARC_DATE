# Combo variant of e8686506 (index 5)
# combo_spec: D4/2+D18/1
# Name: 90度旋转对称+同心环排列

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    The original task searched for a left-right symmetric packed rectangle with
#    an irregular frame and non-touching same-color pieces. This combo changes
#    both of those pillars:
#      - D4/2 forces the assembled target to be an odd square, because every
#        cell must agree with its full 90-degree rotation orbit.
#      - D18/1 replaces the irregular interior by concentric square rings, one
#        color per ring depth, all the way to the center.
#    Example with n=7:
#      - old output logic could place a color only with one mirror mate.
#      - new output is:
#          ring 0: border of the 7x7 square
#          ring 1: border of the inner 5x5 square
#          ring 2: border of the inner 3x3 square
#          ring 3: center singleton
#      - input pieces are the bars that can rebuild those rings:
#          two 1x7 bars + two 5x1 bars,
#          two 1x5 bars + two 3x1 bars,
#          two 1x3 bars + two 1x1 side pixels,
#          one center 1x1 pixel.
#    Difference: instead of generic placement search, the solver now reads the
#    bar-length signatures of each color and renders the ring depths directly.
#
# b) Complexity calibration:
#    Using only one or two rings would be too weak: many square colorings are
#    also 90-degree symmetric, so the fusion would look trivial. Restricting
#    outputs to n in {7, 9} gives 4 or 5 ring depths, which is enough repeated
#    structure for humans to see both "quarter-turn square" and "nested rings"
#    without introducing many free parameters. Every non-center ring yields four
#    pieces, so the input remains visibly fragmented.
#
# c) Hidden-assumption audit:
#    Original assumptions that no longer hold:
#      - output dims are inferred from a closest-to-square portrait factor pair;
#        the combo requires a perfect odd square instead
#      - same-color pieces must stay 4-disconnected after assembly; false here,
#        because bars of one ring fuse into one contiguous ring
#      - a 1-2 piece frame already occupies its final bbox; false here, because
#        every ring is split into scattered bars
#      - optional edge/corner anchors are meaningful; unnecessary now, because
#        the ring-depth signature is sufficient
#    New assumptions introduced:
#      - every non-background component is a solid bar or singleton
#      - each ring color is unique and appears either as four bars/singletons
#        matching one side length, or as the lone center pixel
#      - all ring depths 0..n//2 are present exactly once
#    The generator enforces each of these explicitly.
#
# d) Placement legality:
#    The generator is output-first. It chooses the final odd square, decomposes
#    each ring into four disjoint bars (plus one center cell), then scatters the
#    patches onto a larger canvas with Chebyshev padding 1. `random_free_location`
#    prevents overlap and out-of-bounds placement.
#
# e) Visual observability:
#    Across samples, the user sees only detached bars and a few singletons in
#    the input, but the outputs always become tight odd squares with repeated
#    side lengths shrinking by 2 each layer. The pattern "same-color bars of
#    one length family become one concentric ring" is visible without using any
#    hidden ordering on numeric color ids.
#
# f) Visual cleanliness:
#    The input contains nothing except separated monochrome bars on a clean
#    background. The output contains one object only: a compact square of nested
#    rings. There is no overlap of semantic markers or competing anchor systems.
#
# g) Inferability of changes:
#    Colors are permuted globally and carry no fixed numeric meaning. The role
#    of a color is inferred from repeated shape evidence:
#      - longest horizontal/vertical bars -> outer ring
#      - next shorter family -> next inner ring
#      - ...
#      - lone singleton -> center
#    This gives a human solver enough repeated support in 2-4 examples to infer
#    the ring-depth mapping and the resulting 90-degree symmetric square.
# ---------------------------------------------------------------------------


def _bar_info(pts: np.ndarray) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(len(pts))
    if h == 1 and area == w:
        return "h", w
    if w == 1 and area == h:
        return "v", h
    raise AssertionError("every piece must be a straight filled bar or singleton")


def _render_rings(n: int, colors_by_depth: dict[int, int]) -> np.ndarray:
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            depth = min(r, c, n - 1 - r, n - 1 - c)
            out[r, c] = int(colors_by_depth[depth])
    return out


def _match_depth(comps: list[np.ndarray], n: int) -> int | None:
    bars = [_bar_info(pts) for pts in comps]
    sizes = sorted(int(len(pts)) for pts in comps)
    if len(comps) == 1 and sizes == [1]:
        return n // 2
    if len(comps) != 4:
        return None

    side = sizes[-1]
    if sizes != [side - 2, side - 2, side, side]:
        return None
    depth = (n - side) // 2
    if side < 3 or side % 2 == 0 or depth < 0:
        return None
    # For side >= 5 we expect two long horizontal bars and two shorter vertical bars.
    # For side == 3 the "vertical bars" collapse to two singletons, so only the size
    # multiset remains stable across samples.
    if side >= 5:
        hs = sorted(length for orient, length in bars if orient == "h")
        vs = sorted(length for orient, length in bars if orient == "v")
        if hs != [side, side] or vs != [side - 2, side - 2]:
            return None
    return depth


def _ring_patch_specs(n: int, colors_by_depth: dict[int, int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for depth in range(n // 2):
        side = n - 2 * depth
        col = int(colors_by_depth[depth])

        top = np.full((1, side), col, dtype=int)
        bottom = np.full((1, side), col, dtype=int)
        side_len = side - 2
        left = np.full((side_len, 1), col, dtype=int)
        right = np.full((side_len, 1), col, dtype=int)
        patches.extend([top, bottom, left, right])

    center_col = int(colors_by_depth[n // 2])
    patches.append(np.full((1, 1), center_col, dtype=int))
    random.shuffle(patches)
    return [p.astype(int, copy=False) for p in patches]


def _scatter(patches: list[np.ndarray], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered bar fragments into an odd square of concentric color rings with 90-degree rotational symmetry.

    Input structure:
    - The background is the most common color; every non-background component is a solid horizontal bar, vertical bar, or singleton.
    - Each non-background color represents exactly one ring depth of the final square, from the outer border to the center pixel.

    Rule:
    - Collect all non-background 4-connected components and infer the output size from the total area; it must be an odd perfect square.
    - Identify which color belongs to which ring depth from the component signature: a ring of side `s` appears as two horizontal `1xs` bars and two vertical `(s-2)x1` bars, while the center appears as a lone singleton.
    - Render the compact `n x n` square by painting every cell according to its concentric depth.

    Key insight: the bar lengths already encode the entire nested-ring hierarchy. Once the total area reveals the odd square size, quarter-turn symmetry is automatic because every depth class fills a complete square ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        assert comps, "each non-background color must contribute at least one piece"
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert total_area > 0, "grid must contain non-background pieces"
    n = int(round(total_area ** 0.5))
    assert n * n == total_area, "combo variant requires a square output area"
    assert n % 2 == 1 and n >= 7, "combo variant uses odd square sizes with at least three visible rings"

    depth_to_color: dict[int, int] = {}
    for col, comps in comps_by_color.items():
        depth = _match_depth(comps, n)
        assert depth is not None, "every color must match exactly one concentric-ring signature"
        assert depth not in depth_to_color, "ring depths must be uniquely identifiable"
        depth_to_color[depth] = int(col)

    expected_depths = set(range((n // 2) + 1))
    assert set(depth_to_color) == expected_depths, "all ring depths from border to center must be present"
    out = _render_rings(n, depth_to_color)
    assert np.array_equal(out, np.rot90(out)), "rendered output must be 90-degree rotationally symmetric"
    return out.astype(int)


# --------------- generator (output-first concentric-square combo) ---------------

_SIZE_CHOICES = [7, 9]
_SIZE_WEIGHTS = [45, 55]
_CANVAS_MIN = 15
_CANVAS_MAX = 17
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300


def _build_output(n: int) -> tuple[np.ndarray, dict[int, int], int]:
    ring_count = (n // 2) + 1
    palette = random.sample(list(range(10)), ring_count + 1)
    bg = int(palette[0])
    ring_colors = palette[1:]
    colors_by_depth = {depth: int(ring_colors[depth]) for depth in range(ring_count)}
    out = _render_rings(n, colors_by_depth)
    return out, colors_by_depth, bg


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choices(_SIZE_CHOICES, weights=_SIZE_WEIGHTS, k=1)[0]
    out, colors_by_depth, bg = _build_output(n)
    patches = _ring_patch_specs(n, colors_by_depth)
    try:
        inp = _scatter(patches, bg)
    except ValueError:
        return None

    if len(set(map(int, np.unique(out)))) != (n // 2) + 1:
        return None

    got = main(inp)
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid e8686506 combo input")
