from __future__ import annotations

import numpy as np

from arc_utils import bbox


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []

    for color in np.unique(g):
        mask = g == color
        b = bbox(mask)
        assert b is not None, "present colors must have a bounding box"
        r0, c0, r1, c1 = b
        if int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1):
            candidates.append((int(color), r0, c0, r1, c1))

    assert len(candidates) == 1, "expected a unique solid rectangular marker mask"
    return candidates[0]


def _assert_virtual_axis_symmetry(g: np.ndarray, marker: int, mirror_sum: int) -> None:
    height, width = g.shape
    for r in range(height):
        for c in range(width):
            if int(g[r, c]) == marker:
                continue
            for sr, sc in ((mirror_sum - r, c), (r, mirror_sum - c)):
                if not (0 <= sr < height and 0 <= sc < width):
                    continue
                if int(g[sr, sc]) == marker:
                    continue
                assert int(g[sr, sc]) == int(g[r, c]), "visible cells must obey the virtual fold"


def _recovered_value(g: np.ndarray, marker: int, r: int, c: int, mirror_sum: int) -> int:
    height, width = g.shape
    mr, mc = mirror_sum - r, mirror_sum - c
    values: list[int] = []

    for sr, sc in ((mr, c), (r, mc), (mr, mc)):
        if 0 <= sr < height and 0 <= sc < width and int(g[sr, sc]) != marker:
            values.append(int(g[sr, sc]))

    if values:
        assert len(set(values)) == 1, "visible orbit cells must agree"
        return values[0]

    cr, cc = min(r, mr), min(c, mc)
    sr, sc = cc, cr
    assert 0 <= sr < height and 0 <= sc < width, "cropped orbit must have a visible transposed source"
    assert int(g[sr, sc]) != marker, "transposed source must not be hidden"
    return int(g[sr, sc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the solid marker rectangle from a virtually centered two-axis mirror pattern.

    Input structure:
    - The input is a square crop of a larger pattern mirrored across both center axes.
    - One color is structurally unique because all of its pixels form one solid rectangular marker.
    - The mirror coordinate is two cells outside the visible crop, so each known pixel matches positions with row or column index `shape + 1 - index`.

    Rule:
    - Locate the marker rectangle and make an output grid with the same height and width.
    - For each marker pixel, copy the single visible color from its horizontal, vertical, or 180-degree mirror orbit.
    - If the whole orbit is hidden or outside the crop, use the transposed upper-left representative of that orbit.

    Key insight:
    - The rectangle is an occlusion mask, not an object to preserve; the answer is the missing crop of the virtual mirror pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height == width, "expected a square grid"

    marker, r0, c0, r1, c1 = _solid_rectangle_marker(g)
    mirror_sum = height + 1
    assert mirror_sum == width + 1, "row and column folds must use the same virtual size"
    _assert_virtual_axis_symmetry(g, marker, mirror_sum)

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            out[rr, cc] = _recovered_value(g, marker, r, c, mirror_sum)

    return out

# --------------- generator (o5x) ---------------

# Canvas and virtual symmetry dimensions.
_GEN_PATTERN_COLORS = np.array([1, 2, 3, 4, 5, 6, 8, 9], dtype=int)
_GEN_MARKER = 7
_GEN_CROP_SIZE = 30
_GEN_VIRTUAL_SIZE = 32
_GEN_MIRROR_SUM = _GEN_VIRTUAL_SIZE - 1
_GEN_QUADRANT_SIZE = _GEN_VIRTUAL_SIZE // 2
# Palette sampling and color survival seeding.
_GEN_COLOR_COUNT_CHOICES = [7, 8]
_GEN_DIRICHLET_ALPHA = 0.7
_GEN_COLOR_SEED_MARGIN = 2
_GEN_COLOR_SEED_PIXELS = 3
# Marker rectangle sampling.
_GEN_MARKER_MIN_SIDE = 2
_GEN_MARKER_MAX_SIDE = 10
_GEN_MARKER_MIN_AREA = 16
_GEN_MARKER_SHAPES = tuple(
    (h, w) for h in range(_GEN_MARKER_MIN_SIDE, _GEN_MARKER_MAX_SIDE + 1)
    for w in range(_GEN_MARKER_MIN_SIDE, _GEN_MARKER_MAX_SIDE + 1) if h * w >= _GEN_MARKER_MIN_AREA
)
# Rejection sampling limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_reflect_quadrant(quad: np.ndarray) -> np.ndarray:
    full = np.empty((_GEN_VIRTUAL_SIZE, _GEN_VIRTUAL_SIZE), dtype=int)
    full[:_GEN_QUADRANT_SIZE, :_GEN_QUADRANT_SIZE] = quad
    full[_GEN_QUADRANT_SIZE:, :_GEN_QUADRANT_SIZE] = quad[::-1, :]
    full[:_GEN_QUADRANT_SIZE, _GEN_QUADRANT_SIZE:] = quad[:, ::-1]
    full[_GEN_QUADRANT_SIZE:, _GEN_QUADRANT_SIZE:] = quad[::-1, ::-1]
    return full


def _gen_build_pattern() -> np.ndarray:
    n_colors = int(np.random.choice(_GEN_COLOR_COUNT_CHOICES))
    palette = np.random.choice(_GEN_PATTERN_COLORS, size=n_colors, replace=False)
    weights = np.random.dirichlet(np.full(n_colors, _GEN_DIRICHLET_ALPHA))

    quad = np.random.choice(
        palette, size=(_GEN_QUADRANT_SIZE, _GEN_QUADRANT_SIZE), p=weights
    )
    seed_stop = _GEN_QUADRANT_SIZE - _GEN_COLOR_SEED_MARGIN
    coords = [
        (r, c) for r in range(_GEN_COLOR_SEED_MARGIN, seed_stop)
        for c in range(_GEN_COLOR_SEED_MARGIN, seed_stop)
    ]
    np.random.shuffle(coords)
    for i, color in enumerate(palette):
        start = _GEN_COLOR_SEED_PIXELS * i
        for r, c in coords[start : start + _GEN_COLOR_SEED_PIXELS]:
            quad[r, c] = int(color)

    full = _gen_reflect_quadrant(quad)
    return full[:_GEN_CROP_SIZE, :_GEN_CROP_SIZE].copy()


def _gen_marker_shape() -> tuple[int, int]:
    return _GEN_MARKER_SHAPES[int(np.random.randint(len(_GEN_MARKER_SHAPES)))]


def _gen_ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _gen_boxes_overlap(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return _gen_ranges_overlap(ar0, ar1, br0, br1) and _gen_ranges_overlap(ac0, ac1, bc0, bc1)


def _gen_orbit_has_visible_source(
    r: int, c: int, r0: int, c0: int, r1: int, c1: int
) -> bool:
    for sr, sc in (
        (_GEN_MIRROR_SUM - r, c),
        (r, _GEN_MIRROR_SUM - c),
        (_GEN_MIRROR_SUM - r, _GEN_MIRROR_SUM - c),
    ):
        if 0 <= sr < _GEN_CROP_SIZE and 0 <= sc < _GEN_CROP_SIZE:
            if not (r0 <= sr <= r1 and c0 <= sc <= c1):
                return True
    return False


def _gen_valid_marker_positions(h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(_GEN_CROP_SIZE - h + 1):
        r1 = r0 + h - 1
        for c0 in range(_GEN_CROP_SIZE - w + 1):
            c1 = c0 + w - 1
            box = (r0, c0, r1, c1)
            mirror_box = (_GEN_MIRROR_SUM - r1, _GEN_MIRROR_SUM - c1, _GEN_MIRROR_SUM - r0, _GEN_MIRROR_SUM - c0)
            if _gen_boxes_overlap(box, mirror_box):
                continue
            if all(
                _gen_orbit_has_visible_source(r, c, r0, c0, r1, c1)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
            ):
                positions.append((r0, c0))
    return positions


def _gen_passes_oracle(inp: np.ndarray, expected: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    try:
        marker, mr0, mc0, mr1, mc1 = _solid_rectangle_marker(inp)
        if (marker, mr0, mc0, mr1, mc1) != (_GEN_MARKER, r0, c0, r1, c1):
            return False
        actual = main(inp)
    except AssertionError:
        return False

    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    pattern = _gen_build_pattern()
    h, w = _gen_marker_shape()
    positions = _gen_valid_marker_positions(h, w)
    if not positions:
        return None

    r0, c0 = positions[int(np.random.randint(len(positions)))]
    box = (r0, c0, r0 + h - 1, c0 + w - 1)
    expected = pattern[r0 : r0 + h, c0 : c0 + w].copy()
    inp = pattern.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _GEN_MARKER
    return inp if _gen_passes_oracle(inp, expected, box) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid de493100 input")
