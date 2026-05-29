# Variant of 0934a4d8
# Dimension: D4 — 恢复变换机制 (transform)
# Change: 通过 LR/UD 反射轨道找到非标记值恢复；若轨道全被遮挡，使用转置回退 g[c,r]... → 恢复后颜色取反（互补映射）

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the D4 variant
# a) Concrete behavior:
#    The hidden rectangle still belongs to a square grid with the same LR/UD mirror
#    structure and the same transpose fallback near the top/left border. The new
#    visible ingredient is a repeated 2-row / 2-column border legend:
#    for every exposed legend position, the two stacked colors (top border) or the
#    two side-by-side colors (left border) form one complementary pair. Solver flow
#    becomes "recover an encoded color from symmetry" -> "swap it to its visible
#    partner from the legend". Example: if a marker cell recovers encoded color 3
#    from its reflection orbit and the border repeatedly shows 3 paired with 7, the
#    output cell is 7 instead of 3. In the original task the same cell would have
#    output 3 directly.
# b) Complexity calibration:
#    The complement map is restricted to an involution on the 8 non-marker colors,
#    i.e. four disjoint pairs. This is richer than a fixed global swap, but still
#    simple enough for humans because every pair is repeated many times in the
#    border legend. The geometric part stays unchanged, so the task differs in one
#    core dimension instead of becoming a new puzzle.
# c) Hidden-assumption review:
#    - `_reflection_orbit_values` still only needs the unique visible encoded value;
#      that assumption remains valid.
#    - The original `main()` wrote recovered values directly into `out`; that no
#      longer holds. Every recovered value must pass through an inferred map.
#    - The transpose fallback still assumes `g[c, r]` is visible and consistent with
#      the hidden orbit. Generator must therefore keep transpose-linked encoded cells
#      equal even after inserting the legend.
# d) Placement legality:
#    Marker placement rules remain the same. The generator keeps the same cached
#    valid placements and the same fallback-cell test, so no rectangle is allowed
#    to create an unrecoverable all-hidden orbit or a transpose cell that is also
#    buried under the marker.
# e) Visual observability:
#    The new rule is visible because the border contains repeated paired colors while
#    the output always uses the partner color of the reconstructed patch. Since the
#    pairing is shown inside the grid itself, the rule can vary per instance without
#    becoming opaque.
# f) Visual cleanliness:
#    The legend is confined to the existing outer border, so the central symmetric
#    pattern remains readable and the marker remains the only solid rectangle.
# g) Color-change inferability:
#    Only four pairings exist in each sample, each repeated several times on both
#    the top and left borders. There are no color-id tricks: the mapping is entirely
#    specified by visible adjacency in the legend.


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))

    assert len(candidates) == 1
    _area, color, b = candidates[0]
    return color, b


def _find_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    for lo in range(n // 2 + 1):
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
    n = int(g.shape[0])
    hi = n - 1
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


def _extract_color_map(g: np.ndarray, lo: int, marker: int) -> dict[int, int]:
    n = int(g.shape[0])
    mapping: dict[int, int] = {}

    def add_pair(a: int, b: int) -> None:
        assert a != marker and b != marker
        assert a != b
        if a in mapping:
            assert mapping[a] == b
        else:
            mapping[a] = b
        if b in mapping:
            assert mapping[b] == a
        else:
            mapping[b] = a

    for c in range(lo, n):
        a = int(g[0, c])
        b = int(g[1, c])
        if a != marker and b != marker:
            add_pair(a, b)

    for r in range(lo, n):
        a = int(g[r, 0])
        b = int(g[r, 1])
        if a != marker and b != marker:
            add_pair(a, b)

    visible_colors = set(map(int, np.unique(g))) - {marker}
    assert visible_colors <= set(mapping)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the marked rectangle by symmetry, then decode the recovered colors through a visible complement legend.

    Input structure:
    - A square colored grid contains one solid rectangular marker hiding an encoded patch.
    - The lower/right main field keeps the same left-right and up-down mirror structure as the original task.
    - The top two rows and left two columns act as a repeated legend: adjacent border colors come in complementary pairs.

    Rule:
    - Detect the unique solid rectangle; that rectangle defines the output size.
    - Infer the mirrored strip start exactly as before, ignoring the marker while checking LR/UD consistency.
    - Recover each hidden encoded pixel from its reflection orbit, or from the transpose cell `g[c, r]` when the full orbit is hidden in the border, then replace that encoded color with its paired border complement.

    Key insight:
    Reflection and transpose still tell you which encoded color was hidden, but the output is one extra step away: the border legend says which partner color each recovered value must flip to.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)
    color_map = _extract_color_map(g, lo, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo, marker)
            if values:
                assert len(values) == 1
                encoded = values.pop()
            else:
                assert r < lo or c < lo
                assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
                encoded = int(g[c, r])
                assert encoded != marker
            out[rr, cc] = color_map[encoded]

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


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
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
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


def _is_legend_cell(r: int, c: int) -> bool:
    return (r < _LO <= c) or (c < _LO <= r)


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
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
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        anchor_colors = {int(clean[r, c]) for r, c in coords if _is_legend_cell(r, c)}
        if anchor_colors:
            assert len(anchor_colors) == 1
            color = anchor_colors.pop()
        else:
            color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _random_color_map() -> tuple[dict[int, int], list[tuple[int, int]]]:
    colors = _PALETTE.copy()
    _RNG.shuffle(colors)

    color_map: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for i in range(0, len(colors), 2):
        a = int(colors[i])
        b = int(colors[i + 1])
        color_map[a] = b
        color_map[b] = a
        if bool(_RNG.integers(2)):
            pairs.append((a, b))
        else:
            pairs.append((b, a))
    return color_map, pairs


def _build_clean(pairs: list[tuple[int, int]]) -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    legend_left = np.array([pairs[i % len(pairs)][0] for i in range(half)], dtype=int)
    legend_right = np.array([pairs[i % len(pairs)][1] for i in range(half)], dtype=int)

    clean[0, _LO : _LO + half] = legend_left
    clean[0, _LO + half : _N] = legend_left[::-1]
    clean[1, _LO : _LO + half] = legend_right
    clean[1, _LO + half : _N] = legend_right[::-1]

    clean[_LO : _LO + half, 0] = legend_left
    clean[_LO + half : _N, 0] = legend_left[::-1]
    clean[_LO : _LO + half, 1] = legend_right
    clean[_LO + half : _N, 1] = legend_right[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _decode_patch(patch: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.empty_like(patch)
    for color in np.unique(patch):
        value = int(color)
        out[patch == value] = color_map[value]
    return out


def _gen_attempt() -> np.ndarray | None:
    color_map, pairs = _random_color_map()
    clean = _build_clean(pairs)
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    fallback_placements = [placement for placement in placements if placement[2]]
    pool = fallback_placements if fallback_placements and bool(_RNG.integers(4)) else placements
    r0, c0, fallback = pool[int(_RNG.integers(len(pool)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = _decode_patch(clean[r0 : r0 + h, c0 : c0 + w], color_map)
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
    raise RuntimeError("failed to generate a valid 0934a4d8 input")
