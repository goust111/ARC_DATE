# Combo variant of 0934a4d8 (index 16)
# combo_spec: D9/0+D11/0
# Name: 少色调色板+边界含解码提示

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox


# Design analysis for the fused variant:
# - D9/0 lowers the visual complexity by restricting the whole non-marker palette
#   to 3 or 4 colors. To stop that simplification from collapsing back into the
#   base task, the visible symmetric pattern is treated as an encoded pattern.
# - D11/0 makes the free boundary informative: the top-left 4x4 corner is a legend.
#   Each legend row with the shape [src, src, dst, dst] encodes one color mapping.
#   The output is the hidden rectangle after recovering encoded colors by symmetry
#   and then applying this legend-driven remapping.
# - The mapping is always a derangement on the same 3-4 colors, so the legend is
#   necessary but the total non-marker palette still stays at 3-4 colors.
# - The original transpose fallback is preserved. The rest of the top/left border
#   keeps the original reflective construction, while placements whose transpose
#   fallback would land inside the legend corner are rejected.
# - New solver assumptions introduced by the fusion:
#   1. marker color is fixed to 8 and forms one solid rectangle;
#   2. the informative border width is 4, because the legend needs room for up to
#      four rows of [src src dst dst];
#   3. every encoded color used in the body appears in the legend exactly once.
#   The generator below enforces all three assumptions.


_N = 30
_LO = 4
_HI = _N - 1
_MARKER = 8
_ALL_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _marker_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g == _MARKER
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert area > 1
    assert int(mask.sum()) == area
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return int(r0), int(c0), int(r1), int(c1)


def _find_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m == _N
    hi = n - 1

    for lo in range(_LO, n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    raise AssertionError("no reflected strip found")


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> set[int]:
    hi = int(g.shape[0]) - 1
    seen: set[tuple[int, int]] = set()
    values: set[int] = set()
    q: deque[tuple[int, int]] = deque([(int(r), int(c))])

    while q:
        rr, cc = q.popleft()
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))

        value = int(g[rr, cc])
        if value != marker:
            values.add(value)

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def _read_legend(g: np.ndarray, lo: int) -> dict[int, int]:
    legend = g[:lo, :lo]
    mid = lo // 2
    mapping: dict[int, int] = {}
    for row in legend:
        left = row[:mid]
        right = row[mid:]
        if bool(np.all(left == left[0])) and bool(np.all(right == right[0])) and int(left[0]) != int(right[0]):
            mapping[int(left[0])] = int(right[0])

    assert 3 <= len(mapping) <= 4
    assert set(mapping) == set(mapping.values())
    assert all(src != dst for src, dst in mapping.items())
    return mapping


def _recover_encoded_patch(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _marker_bbox(g)
    lo = _find_reflected_strip_start(g, _MARKER)
    mapping = _read_legend(g, lo)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo, _MARKER)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                value = int(g[c, r])
                assert value != _MARKER
                out[rr, cc] = value

    return out, mapping


def _decode_patch(encoded: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    decoded = np.empty_like(encoded)
    for color in np.unique(encoded):
        key = int(color)
        assert key in mapping
        decoded[encoded == key] = mapping[key]
    return decoded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle by first reconstructing an encoded symmetric patch,
    then decoding its colors from the legend embedded in the free border.

    Input structure:
    - A 30x30 square grid with one solid marker rectangle in color 8.
    - The large block starting at row/col 4 is horizontally and vertically reflective,
      while the top-left 4x4 corner is a legend rather than ordinary pattern content.
    - The remaining top/left border cells still carry the same mirrored/transposed
      support role as in the base task for border-overlapping marker placements.

    Rule:
    - Locate the color-8 solid rectangle and recover the encoded color under each
      marker cell from its LR/UD reflection orbit; if an orbit is fully hidden, use
      the visible transpose counterpart on the border.
    - Read the legend from the top-left corner: every row shaped like [src src dst dst]
      means "replace encoded color src with output color dst".
    - Apply that legend mapping to the recovered encoded rectangle and return the
      decoded rectangle.

    Key insight:
    Symmetry still tells you which encoded color belongs under the marker, but the
    border legend tells you that those recovered colors are not final yet; they must
    be remapped through the short palette permutation shown in the corner.
    """
    encoded, mapping = _recover_encoded_patch(np.asarray(grid, dtype=int))
    return _decode_patch(encoded, mapping)


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    if r0 < _LO and c0 < _LO:
        return None

    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                tr, tc = c, r
                if r0 <= tr < r1 and c0 <= tc < c1:
                    return None
                if tr < _LO and tc < _LO:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _placement_category(r0: int, c0: int, fallback: list[tuple[int, int]]) -> str:
    if fallback:
        return "fallback_top" if r0 < _LO else "fallback_left"
    if r0 < _LO:
        return "border_top"
    if c0 < _LO:
        return "border_left"
    return "interior"


def _choose_placement(
    placements: list[tuple[int, int, list[tuple[int, int]]]]
) -> tuple[int, int, list[tuple[int, int]]]:
    buckets: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = {}
    for placement in placements:
        category = _placement_category(*placement)
        buckets.setdefault(category, []).append(placement)

    # Keep the solver branches visible in generated samples instead of letting
    # the much more numerous interior placements drown out border/fallback cases.
    categories = sorted(buckets)
    chosen_category = categories[int(_RNG.integers(len(categories)))]
    bucket = buckets[chosen_category]
    return bucket[int(_RNG.integers(len(bucket)))]


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]], colors: np.ndarray) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        coords = list(group)
        if not coords:
            continue
        for coord in coords:
            parent.setdefault(coord, coord)
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(colors))
        for r, c in coords:
            clean[r, c] = color


def _sample_palette_and_mapping() -> tuple[np.ndarray, dict[int, int]]:
    k = int(_RNG.integers(3, 5))
    colors = np.array(_RNG.choice(_ALL_COLORS, size=k, replace=False), dtype=int)
    shift = int(_RNG.integers(1, k))
    permuted = np.roll(colors, -shift)
    mapping = {int(src): int(dst) for src, dst in zip(colors, permuted, strict=True)}
    return colors, mapping


def _write_legend(clean: np.ndarray, colors: np.ndarray, mapping: dict[int, int]) -> None:
    order = list(map(int, _RNG.permutation(colors)))
    for idx, src in enumerate(order):
        clean[idx, :2] = src
        clean[idx, 2:_LO] = mapping[src]

    if len(order) == 3:
        a, b, c = order
        clean[3, :_LO] = np.array([a, b, a, c], dtype=int)


def _build_clean(colors: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(colors, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(colors, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(colors, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    _write_legend(clean, colors, mapping)
    return clean


def _legend_rows_count(clean: np.ndarray) -> int:
    mapping = _read_legend(clean, _LO)
    return len(mapping)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    colors, mapping = _sample_palette_and_mapping()
    clean = _build_clean(colors, mapping)
    if set(np.unique(clean[_LO:, _LO:])) != set(map(int, colors)):
        return None
    if _legend_rows_count(clean) != len(colors):
        return None

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    r0, c0, fallback = _choose_placement(placements)

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups, colors)
    _write_legend(clean, colors, mapping)

    encoded_patch = clean[r0 : r0 + h, c0 : c0 + w].copy()
    if len(np.unique(encoded_patch)) < 2:
        return None

    expected = _decode_patch(encoded_patch, mapping)
    if np.array_equal(expected, encoded_patch):
        return None

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if np.array_equal(main(inp), expected):
            return inp, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            grid, _expected = sample
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
