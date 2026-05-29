# Combo variant of 0934a4d8 (index 24)
# combo_spec: D4/1+D6/0+D11/0
# Name: 颜色映射+可变lo+边界提示

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the combo variant
# a) Concrete behavior:
#    The original task recovered the hidden rectangle directly from LR/UD reflection
#    orbits, with transpose fallback only when the marker swallowed an entire orbit
#    near the top/left border. Here the reasoning chain becomes:
#    1) infer `lo` from the smallest lower/right strip that is mirror-consistent,
#    2) read a color-pair legend from the free border,
#    3) reconstruct the hidden encoded patch by reflection or transpose fallback,
#    4) map every recovered encoded color through the inferred legend.
#    Example with `lo=3`: a hidden cell might reflect to visible color 5, while the
#    top two border rows repeatedly show the pair (5, 2). The solver must output 2.
#    In the original task that same cell would have output 5.
# b) Complexity calibration:
#    `lo` varies only over {2, 3, 4}, so the axis-inference step is real but still
#    learnable from a few examples. The color map is restricted to four disjoint
#    swaps over the 8 non-marker colors, and each pair is repeated many times along
#    both the top and left border. That keeps the added symbolic layer visible
#    without making the puzzle depend on one-off coincidences.
# c) Hidden-assumption review:
#    - The original direct-write assumption is no longer valid: recovered values are
#      encoded colors and must always pass through the border-derived map.
#    - `_find_reflected_strip_start` now matters semantically because the start of
#      the mirrored strip is not fixed. The generator therefore rejects any sample
#      where a smaller accidental `lo` would also appear valid after masking.
#    - Transpose fallback still assumes `g[c, r]` is visible and consistent with the
#      hidden encoded color. When fallback touches legend rows/cols, the generator
#      anchors the entire transpose-linked group to that visible legend color.
#    - `_extract_color_map` assumes the first two border rows/cols remain available;
#      therefore generation keeps `lo >= 2` and repeats all four pairs often enough
#      that partial marker overlap cannot erase the legend.
# d) Placement legality:
#    Marker placement still uses the original validity rule: no fully hidden orbit is
#    allowed unless one coordinate lies in the free border and the transpose cell is
#    outside the marker. The cache is keyed by `(lo, h, w)` because legality depends
#    on the inferred strip width.
# e) Visual observability:
#    Humans can see the difference from the base task in two ways. First, the border
#    width changes across samples, so the symmetry axis must be inferred rather than
#    memorized. Second, the first two border rows/cols visibly form repeated stacked
#    or side-by-side color pairs, making the decode step explicit.
# f) Visual cleanliness:
#    The legend lives only in the existing free border. The central symmetric field
#    remains a clean reflected texture, and the marker stays the only solid rectangle.
#    Extra border rows/cols for `lo=3` or `lo=4` are plain mirrored noise so they
#    help expose `lo` without competing with the legend.
# g) Color-change inferability:
#    The color map is a sample-specific involution with no fixed points. Because all
#    8 non-marker colors participate and each pair repeats several times on both top
#    and left borders, the output-changing rule is recoverable from 2-4 examples
#    without relying on color-id arithmetic or hidden metadata.


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


def _recover_encoded_patch(
    g: np.ndarray,
    marker: int,
    rect: tuple[int, int, int, int],
    lo: int,
) -> np.ndarray:
    r0, c0, r1, c1 = rect
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
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden encoded rectangle from the variable-offset mirror field, then decode it through a border legend.

    Input structure:
    - A square grid contains one solid rectangular marker hiding an encoded patch.
    - The large lower/right region is left-right and up-down symmetric, but its starting offset `lo` varies by sample.
    - The free border is at least two cells thick; its first two rows and first two columns repeatedly display color-pair hints for the decode map, while any extra border width only helps reveal `lo`.

    Rule:
    - Find the unique solid rectangle to determine the output size, then infer `lo` as the smallest row/column offset whose lower/right strips are mirror-consistent when marker cells are ignored.
    - Read the color map from the repeated border pairs in rows 0-1 and columns 0-1.
    - For each marker cell, recover the unique visible encoded color from its LR/UD reflection orbit; if the whole orbit is still hidden because the cell lies in the free border, use the transpose fallback `g[c, r]`. After that, transform the recovered encoded color through the inferred map before writing it to the output.

    Key insight:
    The geometric recovery and the symbolic decoding are both necessary: symmetry tells you which encoded color was hidden, but the border legend tells you that the encoded patch is not the final answer yet.
    """
    g = np.asarray(grid, dtype=int)
    marker, rect = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)
    color_map = _extract_color_map(g, lo, marker)
    encoded = _recover_encoded_patch(g, marker, rect, lo)

    out = np.empty_like(encoded)
    for color in np.unique(encoded):
        value = int(color)
        out[encoded == value] = color_map[value]
    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_HI = _N - 1
_LO_CHOICES = (2, 3, 4)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}
_LO_CURSOR = 0
_FALLBACK_CURSOR = 0


def _orbit(r: int, c: int, lo: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    if lo <= r <= _HI:
        rows.add(lo + _HI - int(r))
    if lo <= c <= _HI:
        cols.add(lo + _HI - int(c))
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(lo: int, r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c, lo)):
                if r >= lo and c >= lo:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int, lo: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (lo, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(lo, r0, c0, h, w)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _is_legend_cell(r: int, c: int, lo: int) -> bool:
    return ((r == 0 or r == 1) and c >= lo) or ((c == 0 or c == 1) and r >= lo)


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]], lo: int) -> None:
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
        anchor_colors = {int(clean[r, c]) for r, c in coords if _is_legend_cell(r, c, lo)}
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


def _make_palindrome(left: np.ndarray, length: int) -> np.ndarray:
    left = np.asarray(left, dtype=int)
    if length % 2 == 0:
        return np.concatenate([left, left[::-1]])
    return np.concatenate([left, left[-2::-1]])


def _build_clean(lo: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    inner_len = _N - lo
    half = (inner_len + 1) // 2
    clean = np.empty((_N, _N), dtype=int)

    for r in range(lo, lo + half):
        for c in range(lo, lo + half):
            color = int(_RNG.choice(_PALETTE))
            for rr, cc in _orbit(r, c, lo):
                clean[rr, cc] = color

    legend_a = np.array([pairs[i % len(pairs)][0] for i in range(half)], dtype=int)
    legend_b = np.array([pairs[i % len(pairs)][1] for i in range(half)], dtype=int)
    clean[0, lo:_N] = _make_palindrome(legend_a, inner_len)
    clean[1, lo:_N] = _make_palindrome(legend_b, inner_len)
    clean[lo:_N, 0] = _make_palindrome(legend_a, inner_len)
    clean[lo:_N, 1] = _make_palindrome(legend_b, inner_len)

    for r in range(2, lo):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, lo:_N] = _make_palindrome(vals, inner_len)
    for c in range(2, lo):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[lo:_N, c] = _make_palindrome(vals, inner_len)

    clean[:lo, :lo] = _RNG.choice(_PALETTE, size=(lo, lo))
    return clean


def _decode_patch(patch: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.empty_like(patch)
    for color in np.unique(patch):
        value = int(color)
        out[patch == value] = color_map[value]
    return out


def _target_generation_branch() -> tuple[int, bool]:
    global _LO_CURSOR, _FALLBACK_CURSOR
    lo = int(_LO_CHOICES[_LO_CURSOR % len(_LO_CHOICES)])
    want_fallback = bool(_FALLBACK_CURSOR % 2)
    _LO_CURSOR += 1
    _FALLBACK_CURSOR += 1
    return lo, want_fallback


def _gen_attempt(lo: int, want_fallback: bool) -> np.ndarray | None:
    color_map, pairs = _random_color_map()
    clean = _build_clean(lo, pairs)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w, lo)
    if not placements:
        return None

    fallback_placements = [placement for placement in placements if placement[2]]
    plain_placements = [placement for placement in placements if not placement[2]]
    if want_fallback:
        pool = fallback_placements
    else:
        pool = plain_placements
    if not pool:
        return None
    r0, c0, fallback = pool[int(_RNG.integers(len(pool)))]

    groups = [_orbit(r, c, lo) | _orbit(c, r, lo) for r, c in fallback]
    _unify_groups(clean, groups, lo)

    expected = _decode_patch(clean[r0 : r0 + h, c0 : c0 + w], color_map)
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        marker, rect = _solid_rectangle_marker(inp)
        assert marker == _MARKER
        assert rect == (r0, c0, r0 + h - 1, c0 + w - 1)
        assert _find_reflected_strip_start(inp, _MARKER) == lo
        assert np.array_equal(_recover_encoded_patch(inp, _MARKER, rect, lo), clean[r0 : r0 + h, c0 : c0 + w])
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    lo, want_fallback = _target_generation_branch()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(lo, want_fallback)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
