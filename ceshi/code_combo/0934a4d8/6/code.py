# Combo variant of 0934a4d8 (index 6)
# combo_spec: D4/1+D11/0
# Name: 颜色映射恢复+边界含解码提示

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for combo D4/1 + D11/0
# a) Concrete behavior:
#    The original task recovers the rectangle directly from LR/UD reflection or,
#    when the whole orbit is covered inside the free border, from the transpose
#    cell g[c, r]. The fused variant keeps that geometry but changes what the
#    border means: rows 0..1 and cols 0..1 are now a repeated mapping legend.
#    Example: suppose the border repeatedly shows vertical/horizontal pairs 3->7,
#    1->4, 9->2, 5->6. If a hidden marker cell reconstructs encoded color 3 from
#    its reflection orbit, the solver must output 7 instead of 3. If that hidden
#    cell has a fully covered orbit near the top border, it first reads encoded 3
#    from g[c, r], then still decodes 3->7. Relative to the original task, the
#    geometric recovery stage is unchanged; the new difference is a mandatory
#    border-driven color decoding stage after recovery.
# b) Complexity calibration:
#    The legend uses only four disjoint color pairs over the 8 non-marker colors.
#    Each pair is repeated many times across both the top and left border, so the
#    mapping is obvious from a few training examples. This keeps the task visibly
#    different from the base puzzle without adding too many independent knobs.
# c) Hidden-assumption review:
#    - `_find_reflected_strip_start` assumes the visible [lo..N-1] segments still
#      satisfy LR/UD consistency. The generator preserves that by mirroring the
#      legend along the same axes.
#    - `_reflection_orbit_values` assumes a hidden cell's visible orbit members,
#      when any exist, agree on exactly one encoded color. The generator still
#      builds a clean symmetric field before inserting the marker, so this holds.
#    - The old `main()` wrote the recovered color directly to output; that is no
#      longer valid. Every recovered encoded value must appear in the inferred
#      border map and be translated through it.
#    - The transpose fallback assumes g[c, r] is visible and non-marker. The same
#      placement legality from the base task is retained, and fallback-linked
#      transpose/orbit groups are unified so that border legend cells do not
#      contradict the hidden encoded patch.
# d) Placement legality:
#    Marker placements reuse the original legality test: fully hidden orbits are
#    only allowed when the cell touches the free border and its transpose is not
#    also covered. The rectangle may still touch any side, but the generator only
#    accepts placements satisfying those recoverability constraints.
# e) Visual observability:
#    A human sees two regular signals at once: the big field is reflection-based,
#    and the border contains stacked / side-by-side repeated color pairings. The
#    output rectangle consistently uses the partner color rather than the visible
#    orbit color, so the extra decoding step is observable from examples.
# f) Visual cleanliness:
#    No extra objects are added inside the main field. The legend stays confined to
#    the already-free border, the marker remains the only solid rectangle, and the
#    top-left 2x2 corner is only a small legend continuation rather than clutter.
# g) Change inferability:
#    The mapping is fully specified by adjacency in the border legend instead of by
#    hidden color ids or metadata. Because every sample uses a complete four-pair
#    involution and the output always applies it, the fused color transformation is
#    both non-degenerate and learnable from 2-4 examples.


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


def _recover_encoded_value(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    values = _reflection_orbit_values(g, r, c, lo, marker)
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
    value = int(g[c, r])
    assert value != marker
    return value


def _decode_patch(patch: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.empty_like(patch)
    for color in np.unique(patch):
        value = int(color)
        out[patch == value] = color_map[value]
    return out


def _recover_encoded_patch(g: np.ndarray, marker_box: tuple[int, int, int, int], lo: int, marker: int) -> np.ndarray:
    r0, c0, r1, c1 = marker_box
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            out[rr, cc] = _recover_encoded_value(g, int(r0 + rr), int(c0 + cc), lo, marker)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the marked rectangle from reflection symmetry, then decode those recovered colors through a border legend.

    Input structure:
    - A square grid contains one solid rectangular marker that hides an encoded patch.
    - The large lower/right field remains left-right and up-down symmetric on a shared inner strip, exactly as in the base task.
    - The free top two rows and left two columns now double as a repeated decoding table: each stacked or side-by-side pair indicates two partner colors.

    Rule:
    - Locate the unique solid rectangle; its bounding box determines the output shape.
    - Infer the mirrored strip start by finding the smallest offset whose lower/right strips are LR/UD consistent when marker cells are ignored.
    - Recover each hidden encoded pixel from its reflection orbit, or from the transpose fallback `g[c, r]` when the full orbit is also covered, then replace that encoded value by its border-paired partner color.

    Key insight:
    The border is no longer just spare symmetric context. It also acts as a legend that tells you how to translate the symmetry-recovered patch colors into the final output colors.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_box = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)
    color_map = _extract_color_map(g, lo, marker)
    encoded_patch = _recover_encoded_patch(g, marker_box, lo, marker)
    return _decode_patch(encoded_patch, color_map)


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
_PREFER_FALLBACK_NEXT = True


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
        for coord in coords:
            find(coord)
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


def _relevant_pair_count(patch: np.ndarray, color_map: dict[int, int]) -> int:
    pairs = {tuple(sorted((value, color_map[value]))) for value in map(int, np.unique(patch))}
    return len(pairs)


def _build_clean(pairs: list[tuple[int, int]]) -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    legend_src = np.array([pairs[i % len(pairs)][0] for i in range(half)], dtype=int)
    legend_dst = np.array([pairs[i % len(pairs)][1] for i in range(half)], dtype=int)

    clean[0, _LO : _LO + half] = legend_src
    clean[0, _LO + half : _N] = legend_src[::-1]
    clean[1, _LO : _LO + half] = legend_dst
    clean[1, _LO + half : _N] = legend_dst[::-1]

    clean[_LO : _LO + half, 0] = legend_src
    clean[_LO + half : _N, 0] = legend_src[::-1]
    clean[_LO : _LO + half, 1] = legend_dst
    clean[_LO + half : _N, 1] = legend_dst[::-1]

    clean[0, 0] = legend_src[0]
    clean[0, 1] = legend_dst[0]
    clean[1, 0] = legend_src[0]
    clean[1, 1] = legend_dst[0]
    return clean


def _choose_marker_placement(require_fallback: bool) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    assert placements
    fallback_placements = [placement for placement in placements if placement[2]]
    plain_placements = [placement for placement in placements if not placement[2]]

    if require_fallback and fallback_placements:
        pool = fallback_placements
    elif not require_fallback and plain_placements:
        pool = plain_placements
    elif fallback_placements:
        pool = fallback_placements
    else:
        pool = placements

    r0, c0, fallback = pool[int(_RNG.integers(len(pool)))]
    return h, w, r0, c0, fallback


def _gen_attempt(require_fallback: bool) -> np.ndarray | None:
    color_map, pairs = _random_color_map()
    clean = _build_clean(pairs)
    h, w, r0, c0, fallback = _choose_marker_placement(require_fallback)

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    encoded_patch = clean[r0 : r0 + h, c0 : c0 + w].copy()
    if np.unique(encoded_patch).size < 2:
        return None
    if _relevant_pair_count(encoded_patch, color_map) < 2:
        return None

    expected = _decode_patch(encoded_patch, color_map)
    if not np.any(expected != encoded_patch):
        return None

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    global _PREFER_FALLBACK_NEXT
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(_PREFER_FALLBACK_NEXT)
        if grid is not None:
            _PREFER_FALLBACK_NEXT = not _PREFER_FALLBACK_NEXT
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
