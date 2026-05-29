# Combo variant of 0934a4d8 (index 23)
# combo_spec: D7/2+D9/0+D15/0
# Name: 小网格+少色+条纹纹理

from __future__ import annotations

import numpy as np


# Design analysis for the fused variant
# a) Concrete behavior:
#    Example with N=11 and horizontal stripes. Rows 1..10 of the clean grid are
#    single-color bands such as A B C B A A B C B A, mirrored vertically. Each
#    band is constant across cols 1..10, so the inner block is visually striped
#    and still bilateral. A solid 8-colored rectangle hides part of that block.
#    The original task would recover a hidden pixel from its reflection orbit in
#    a dense random texture; here the hidden pixel usually comes from a whole
#    stripe. If the marker touches the left border, the top border stores a
#    transposed stripe lookup, so border-side hidden cells can be read directly
#    from g[c, r] before falling back to the mirrored stripe.
# b) Complexity calibration:
#    Small 10-15 grids and only 3-4 non-marker colors simplify the visual load,
#    but the stripe sequence is forced to have multiple transitions so the input
#    does not degenerate into one or two giant rectangles. This keeps the task
#    readable while preserving enough structure for humans to infer symmetry.
# c) Broken original assumptions:
#    - "Find the unique solid rectangle color" no longer works because a stripe
#      texture creates many monochrome rectangles. The solver therefore uses the
#      fixed marker color 8 directly and only checks that its footprint is a
#      filled rectangle.
#    - Searching for the smallest valid reflection strip is fragile on a tiny
#      grid with few colors, because accidental symmetry becomes common. The
#      fused generator fixes the exposed border width to 1 and the solver uses
#      that explicit rule.
#    - Reflection orbits may now contain many repeated colors by design, so the
#      solver reasons at row/column level instead of set-union over raw pixels.
# d) Placement legality:
#    Horizontal-stripe cases never place the marker on the top border; vertical
#    cases never place it on the left border. That makes the transposed border
#    lookup one-sided and guarantees the lookup cell is outside the marker.
#    Marker sizes are capped so they fit inside the 10-15 canvas and leave
#    visible evidence in the controlling row/column or its mirror.
# e) Visual observability:
#    Across a few examples, humans can spot three stable cues: the grid is much
#    smaller than the original, only 3-4 colors recur, and the visible pattern
#    is made of mirrored horizontal or vertical bands rather than noisy tiles.
# f) Visual cleanliness:
#    There are only three layers: stripe field, one-cell border lookup strip,
#    and one solid marker rectangle. No overlapping objects or color-encoded
#    sorting is used, so the scene stays readable.
# g) Inferability of changes:
#    No hidden numeric ordering is used. The palette is reduced but colors do
#    not change by arbitrary mapping; they are simply copied from the relevant
#    stripe, mirrored stripe, or the visible transposed lookup strip.


_BORDER = 1
_MARKER = 8
_PALETTE_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

_N_MIN = 10
_N_MAX = 15

_MARKER_DIM_LO = 2
_MARKER_DIM_HI = 5
_MARKER_AREA_LO = 4
_MARKER_AREA_HI = 18

_MAX_RETRIES = 500
_RNG = np.random.default_rng()


def _marker_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g == _MARKER
    assert bool(mask.any())
    coords = np.argwhere(mask)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert int(mask.sum()) == area
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return int(r0), int(c0), int(r1), int(c1)


def _visible_single_color(arr: np.ndarray) -> int | None:
    vals = np.unique(arr[arr != _MARKER])
    if len(vals) == 1:
        return int(vals[0])
    return None


def _mirror_index(idx: int, n: int) -> int:
    return _BORDER + (n - 1) - int(idx)


def _detect_orientation(g: np.ndarray) -> str:
    inner = np.asarray(g, dtype=int)[_BORDER:, _BORDER:]

    horizontal_bad = 0
    for row in inner:
        if len(np.unique(row[row != _MARKER])) > 1:
            horizontal_bad += 1

    vertical_bad = 0
    for col in inner.T:
        if len(np.unique(col[col != _MARKER])) > 1:
            vertical_bad += 1

    assert horizontal_bad != vertical_bad
    return "horizontal" if horizontal_bad < vertical_bad else "vertical"


def _recover_horizontal(g: np.ndarray, r: int, c: int) -> int:
    n = int(g.shape[0])

    if c < _BORDER:
        color = int(g[c, r])
        assert color != _MARKER
        return color

    color = _visible_single_color(g[r, _BORDER:])
    if color is not None:
        return color

    mirror_r = _mirror_index(r, n)
    color = _visible_single_color(g[mirror_r, _BORDER:])
    if color is not None:
        return color

    color = int(g[c, r])
    assert color != _MARKER
    return color


def _recover_vertical(g: np.ndarray, r: int, c: int) -> int:
    n = int(g.shape[0])

    if r < _BORDER:
        color = int(g[c, r])
        assert color != _MARKER
        return color

    color = _visible_single_color(g[_BORDER:, c])
    if color is not None:
        return color

    mirror_c = _mirror_index(c, n)
    color = _visible_single_color(g[_BORDER:, mirror_c])
    if color is not None:
        return color

    color = int(g[c, r])
    assert color != _MARKER
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stripe patch hidden by the solid marker in a small bilateral grid.

    Input structure:
    - A 10-15 square grid with a one-cell exposed border and a single solid
      marker rectangle in color 8.
    - The clean inner block is bilaterally symmetric and uses either horizontal
      or vertical monochrome stripes with only 3-4 non-marker colors.
    - One exposed border line acts as a transposed stripe lookup: top cells
      decode left-border horizontal cases, and left cells decode top-border
      vertical cases.

    Rule:
    - Locate the filled marker rectangle and use its size as the output size.
    - Detect whether the visible stripe texture is horizontal or vertical by
      checking whether rows or columns stay monochrome inside the inner block.
    - Fill each hidden cell from its controlling stripe: compatible border-side
      cells can be read directly from the transposed lookup line, while the
      remaining cells come from the same row/column or its mirrored partner.

    Key insight:
    The hidden region is not arbitrary image content. Because the clean pattern
    is a mirrored stripe field, every marker cell is determined by one row-band
    or one column-band, and the exposed transposed border makes the border-side
    stripe labels explicit rather than leaving them implicit in raw reflections.
    """
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m
    assert _N_MIN <= n <= _N_MAX

    r0, c0, r1, c1 = _marker_bbox(g)
    orientation = _detect_orientation(g)

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            if orientation == "horizontal":
                out[rr, cc] = _recover_horizontal(g, r, c)
            else:
                out[rr, cc] = _recover_vertical(g, r, c)
    return out


def _palindrome_sequence(length: int, palette: np.ndarray, min_unique: int = 2) -> np.ndarray:
    half = (length + 1) // 2
    while True:
        seed = _RNG.choice(palette, size=half, replace=True)
        if length % 2:
            seq = np.concatenate([seed, seed[-2::-1]])
        else:
            seq = np.concatenate([seed, seed[::-1]])
        transitions = int(np.sum(seq[1:] != seq[:-1]))
        if len(np.unique(seq)) >= min_unique and transitions >= 2:
            return seq.astype(int)


def _build_clean(n: int, orientation: str, palette: np.ndarray) -> np.ndarray:
    inner = n - _BORDER
    stripe_min_unique = len(np.unique(palette))
    while True:
        stripes = _palindrome_sequence(inner, palette, min_unique=stripe_min_unique)
        if len(np.unique(stripes)) == stripe_min_unique:
            break

    clean = np.empty((n, n), dtype=int)
    if orientation == "horizontal":
        clean[_BORDER:, _BORDER:] = stripes[:, None]
    else:
        clean[_BORDER:, _BORDER:] = stripes[None, :]
    clean[0, _BORDER:] = stripes
    clean[_BORDER:, 0] = stripes

    clean[0, 0] = int(stripes[0])
    return clean


def _size_pairs(n: int) -> list[tuple[int, int]]:
    dim_hi = min(_MARKER_DIM_HI, n - 1)
    pairs = [
        (h, w)
        for h in range(_MARKER_DIM_LO, dim_hi + 1)
        for w in range(_MARKER_DIM_LO, dim_hi + 1)
        if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
    ]
    assert pairs
    return pairs


def _valid_placements(n: int, h: int, w: int, orientation: str) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    if orientation == "horizontal":
        for r0 in range(_BORDER, n - h + 1):
            for c0 in range(0, n - w + 1):
                if c0 == _BORDER and w == n - _BORDER:
                    continue
                placements.append((r0, c0))
    else:
        for r0 in range(0, n - h + 1):
            if r0 == _BORDER and h == n - _BORDER:
                continue
            for c0 in range(_BORDER, n - w + 1):
                placements.append((r0, c0))
    return placements


def _choose_placement(placements: list[tuple[int, int]], orientation: str) -> tuple[int, int]:
    if orientation == "horizontal":
        edge = [p for p in placements if p[1] == 0]
    else:
        edge = [p for p in placements if p[0] == 0]
    interior = [p for p in placements if p not in edge]

    if edge and interior:
        pool = edge if bool(_RNG.integers(2)) else interior
        return pool[int(_RNG.integers(len(pool)))]
    if edge:
        return edge[int(_RNG.integers(len(edge)))]
    return placements[int(_RNG.integers(len(placements)))]


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(_N_MIN, _N_MAX + 1))
    orientation = str(_RNG.choice(["horizontal", "vertical"]))
    palette_size = int(_RNG.integers(3, 5))
    palette = _RNG.choice(_PALETTE_POOL, size=palette_size, replace=False).astype(int)

    clean = _build_clean(n, orientation, palette)
    size_pairs = _size_pairs(n)
    h, w = size_pairs[int(_RNG.integers(len(size_pairs)))]
    placements = _valid_placements(n, h, w, orientation)
    if not placements:
        return None
    r0, c0 = _choose_placement(placements, orientation)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 0934a4d8 input")
