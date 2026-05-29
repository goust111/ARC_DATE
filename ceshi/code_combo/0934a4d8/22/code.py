# Combo variant of 0934a4d8 (index 22)
# combo_spec: D3/0+D12/2
# Name: 输出完整修复网格+局部对称性破缺

from __future__ import annotations

from collections import Counter

import numpy as np


# Design analysis for the fused variant
# a) Behavior trace:
#    Original 0934a4d8 outputs only the hidden rectangle. Here the input still has
#    one solid rectangular marker, but the desired output is the full repaired grid.
#    We also inject one small asymmetric patch elsewhere: e.g. a 2x2 block in the
#    inner symmetric area is recolored while its LR/UD reflections are left alone.
#    A solver that globally "re-symmetrizes" the whole board would erase that patch,
#    so the correct behavior is: detect the marker, recover only marker cells, and
#    leave every non-marker cell unchanged.
# b) Complexity calibration:
#    The break is local and repeated as a 2x2 block, so humans can notice a designed
#    anomaly, but it touches only a few reflection pairs. This keeps the main task
#    dominated by the marker-recovery logic while making the new rule visible.
# c) Implicit assumptions audit:
#    - The original `_find_reflected_strip_start` assumed zero disagreement outside
#      the marker. That no longer holds, so we infer `lo` by the lowest disagreement
#      rate instead of exact consistency.
#    - Orbit-based recovery still assumes each marker cell has a unique recoverable
#      color. The generator now guarantees that the deliberate break never shares a
#      reflection orbit with any marker cell and never touches transpose fallback
#      donors, so the recovery orbit remains unambiguous.
#    - Returning only the marker crop is no longer valid; `main()` now copies the
#      whole grid and only fills the marked rectangle.
# d) Placement legality:
#    The generator first builds a clean target grid, then places the marker exactly
#    as in the seed task. After that it searches for a legal 2x2 break patch whose
#    full reflected orbit union is disjoint from the marker rectangle.
# e) Visual observability:
#    Across examples, input and output have the same size; only the marker changes.
#    The extra asymmetric 2x2 patch remains identical in input and output, which is
#    the intended evidence that the solver must not "repair" every symmetry defect.
# f) Clean visuals:
#    Only two structured phenomena are present: one solid rectangular occluder and
#    one small off-orbit break. They do not overlap, so the scene stays readable.
# g) New solver assumptions:
#    `main()` assumes the best `lo` candidate has a very low disagreement rate and
#    that each marker orbit has a unique mode color. The generator enforces both.


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = _bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))
    assert len(candidates) == 1
    _area, color, box = candidates[0]
    return color, box


def _orbit(r: int, c: int, lo: int, hi: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    if lo <= r <= hi:
        rows.add(int(lo + hi - r))
    if lo <= c <= hi:
        cols.add(int(lo + hi - c))
    return {(rr, cc) for rr in rows for cc in cols}


def _symmetry_score(g: np.ndarray, lo: int, marker: int) -> tuple[float, int, int]:
    hi = int(g.shape[0] - 1)

    col_strip = g[:, lo : hi + 1]
    col_ref = col_strip[:, ::-1]
    col_known = (col_strip != marker) & (col_ref != marker)
    col_bad = int(np.count_nonzero((col_strip != col_ref) & col_known))
    col_total = int(np.count_nonzero(col_known))

    row_strip = g[lo : hi + 1, :]
    row_ref = row_strip[::-1, :]
    row_known = (row_strip != marker) & (row_ref != marker)
    row_bad = int(np.count_nonzero((row_strip != row_ref) & row_known))
    row_total = int(np.count_nonzero(row_known))

    total = col_total + row_total
    assert total > 0
    return (col_bad + row_bad) / total, col_bad + row_bad, -total


def _find_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m
    scored = [(lo, _symmetry_score(g, lo, marker)) for lo in range(n // 2 + 1)]
    lo, score = min(scored, key=lambda item: item[1])
    assert score[0] < 0.05
    return int(lo)


def _orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> list[int]:
    hi = int(g.shape[0] - 1)
    values: list[int] = []
    for rr, cc in _orbit(int(r), int(c), lo, hi):
        value = int(g[rr, cc])
        if value != marker:
            values.append(value)
    return values


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the cells hidden by the solid marker, but keep the rest of the board exactly as shown.

    Input structure:
    - A 30x30 square grid with one unique solid rectangular marker color covering a missing patch.
    - Most of the board still follows the original task's left-right and up-down symmetry on the
      large strip starting at an unknown offset `lo`.
    - One additional small non-marker patch intentionally breaks that symmetry somewhere else.

    Rule:
    - Detect the rectangular marker and infer `lo` from the strip with the lowest mirror disagreement
      rate, rather than expecting perfect symmetry.
    - For each marker cell, recover its value from the unique dominant color in its reflection orbit;
      if the whole orbit is hidden because the marker reaches the outer border, use the transposed
      fallback cell exactly as in the seed task.
    - Return the full repaired grid: replace only marker cells and preserve every other visible cell,
      including the deliberate asymmetric patch.

    Key insight:
    The non-marker asymmetry is part of the intended pattern, not damage. Symmetry is only a tool for
    identifying the occluded rectangle's contents, so the correct output edits the marker region in
    place and leaves the designed break untouched.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)

    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            values = _orbit_values(g, r, c, lo, marker)
            if values:
                out[r, c] = _mode_color(values)
            else:
                assert r < lo or c < lo
                value = int(g[c, r])
                assert value != marker
                out[r, c] = value
    return out


# --------------- generator ---------------

_N = 30
_LO = 2
_HI = _N - 1

_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

_BREAK_H = 2
_BREAK_W = 2

_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}
_PLACEMENT_POOL_CACHE: dict[
    tuple[int, int],
    tuple[
        list[tuple[int, int, list[tuple[int, int]]]],
        list[tuple[int, int, list[tuple[int, int]]]],
    ],
] = {}
_BRANCH_TOGGLE = 0


def _break_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    ]


def _break_candidates() -> list[tuple[list[tuple[int, int]], set[tuple[int, int]]]]:
    candidates: list[tuple[list[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in range(_LO, _N - _BREAK_H + 1):
        for c0 in range(_LO, _N - _BREAK_W + 1):
            cells = _break_cells(r0, c0)
            orbit_union: set[tuple[int, int]] = set()
            for r, c in cells:
                orbit_union |= _orbit(r, c, _LO, _HI)
            if len(orbit_union) == 16:
                candidates.append((cells, orbit_union))
    return candidates


_BREAK_CANDIDATES = _break_candidates()


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c, _LO, _HI)):
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


def _placement_pools(
    h: int, w: int
) -> tuple[list[tuple[int, int, list[tuple[int, int]]]], list[tuple[int, int, list[tuple[int, int]]]]]:
    key = (h, w)
    if key not in _PLACEMENT_POOL_CACHE:
        direct: list[tuple[int, int, list[tuple[int, int]]]] = []
        fallback: list[tuple[int, int, list[tuple[int, int]]]] = []
        for placement in _valid_placements(h, w):
            if placement[2]:
                fallback.append(placement)
            else:
                direct.append(placement)
        assert direct and fallback
        _PLACEMENT_POOL_CACHE[key] = (direct, fallback)
    return _PLACEMENT_POOL_CACHE[key]


def _choose_placement(h: int, w: int, prefer_fallback: bool) -> tuple[int, int, list[tuple[int, int]]]:
    direct, fallback = _placement_pools(h, w)
    pool = fallback if prefer_fallback else direct
    return pool[int(_RNG.integers(len(pool)))]


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
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_symmetric_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _marker_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _apply_local_break(clean: np.ndarray, marker_cells: set[tuple[int, int]]) -> bool:
    order = _RNG.permutation(len(_BREAK_CANDIDATES))
    for idx in order:
        cells, orbit_union = _BREAK_CANDIDATES[int(idx)]
        if not orbit_union.isdisjoint(marker_cells):
            continue
        old_colors = {int(clean[r, c]) for r, c in cells}
        choices = [int(color) for color in _PALETTE if int(color) not in old_colors]
        if not choices:
            continue
        color = int(_RNG.choice(np.array(choices, dtype=int)))
        for r, c in cells:
            clean[r, c] = color
        return True
    return False


def _gen_attempt(prefer_fallback: bool) -> tuple[np.ndarray, np.ndarray] | None:
    clean = _build_symmetric_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    r0, c0, fallback = _choose_placement(h, w, prefer_fallback)

    groups = [_orbit(r, c, _LO, _HI) | _orbit(c, r, _LO, _HI) for r, c in fallback]
    _unify_groups(clean, groups)

    marker_cells = _marker_cells(r0, c0, h, w)
    if not _apply_local_break(clean, marker_cells):
        return None

    expected = clean.copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if np.array_equal(main(inp), expected):
            return inp, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    global _BRANCH_TOGGLE
    prefer_fallback = bool(_BRANCH_TOGGLE % 2)
    _BRANCH_TOGGLE += 1
    for _ in range(_MAX_RETRIES):
        sample = _gen_attempt(prefer_fallback)
        if sample is not None:
            inp, _expected = sample
            return inp
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
