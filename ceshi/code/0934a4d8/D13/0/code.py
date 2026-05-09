# Variant of 0934a4d8
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 单步处理：识别标记 -> 确定 lo -> 反射恢复 -> 裁剪输出... → 两层标记嵌套恢复

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the D13 variant
# a) Concrete behavior:
#    The input now shows two nested markers with different colors:
#    - an outer rectangular ring marker (color 8),
#    - an inner solid rectangle marker (color 0) strictly inside that ring.
#    The clean grid still follows the original LR+UD reflection law on the [lo..N-1] strip.
#    Solving is intentionally sequential:
#    1. recover the outer ring cells while treating both marker colors as hidden,
#    2. on the partially restored grid, recover the inner rectangle while treating only color 0 as hidden,
#    3. output the clean content of the inner rectangle.
#    Small trace:
#    - original task: one marker M covers a rectangle; any covered cell reads its value from visible orbit mates.
#    - variant: inner-cell orbit mates may all lie in the outer ring, so they are unavailable in the raw input.
#      After step 1 the same orbit mates become clean and step 2 can finish.
# b) Complexity calibration:
#    The inner marker is not independent noise. Its placement is filtered so at least one inner cell depends on
#    outer-ring restoration. This avoids degeneration back to the single-step task.
# c) Implicit assumptions changed from the original:
#    - "there is exactly one solid rectangle marker" is no longer true; the outer marker is a ring, not a solid block.
#    - finding lo must ignore both marker colors, not just one.
#    - recovering the final output from the raw input is invalid because color 8 values around the inner marker are
#      temporary occluders, not true pattern values.
# d) Placement legality:
#    The outer rectangle must satisfy the original fallback legality as if the whole outer bbox were hidden in step 1.
#    The inner rectangle must independently satisfy the original fallback legality for step 2, be strictly nested,
#    and have at least one orbit whose non-inner witnesses all lie in the outer bbox.
# e) Visual observability:
#    Humans can see a nested two-color marker arrangement directly in the input. The output size matches the inner
#    rectangle, making the "peel outer layer, then peel inner layer" behavior observable across examples.
# f) Visual cleanliness:
#    Only one nested marker pair is injected. The clean pattern remains the same bilateral texture as the source task,
#    so the new structure is easy to isolate.
# g) Color inference:
#    No hidden color arithmetic is used. Colors 8 and 0 are reserved marker colors and never appear in the clean
#    pattern, so the two marker roles are explicit visual categories.
# 9) New solver assumptions guaranteed by the generator:
#    - color 8 forms exactly one rectangular ring containing the color-0 rectangle,
#    - color 0 forms exactly one solid rectangle,
#    - both recovery stages have unique orbit/fallback answers,
#    - at least one inner cell requires stage-1 restoration, so the multi-step rule is active.


_OUTER_MARKER = 8
_INNER_MARKER = 0


def _in_rect(r: int, c: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _solid_rectangle_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = g == int(color)
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert area > 1
    assert int(mask.sum()) == area
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return int(r0), int(c0), int(r1), int(c1)


def _nested_markers(g: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    inner = _solid_rectangle_bbox(g, _INNER_MARKER)

    outer_mask = g == _OUTER_MARKER
    outer = bbox(outer_mask)
    assert outer is not None
    or0, oc0, or1, oc1 = map(int, outer)
    ir0, ic0, ir1, ic1 = inner

    assert or0 < ir0 <= ir1 < or1
    assert oc0 < ic0 <= ic1 < oc1

    view = g[or0 : or1 + 1, oc0 : oc1 + 1]
    expected = np.full(view.shape, _OUTER_MARKER, dtype=int)
    expected[ir0 - or0 : ir1 - or0 + 1, ic0 - oc0 : ic1 - oc0 + 1] = _INNER_MARKER
    assert np.array_equal(view, expected)
    return (or0, oc0, or1, oc1), inner


def _find_reflected_strip_start(g: np.ndarray, hidden_colors: set[int]) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1
    hidden = np.isin(g, list(hidden_colors))

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_hidden = hidden[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        strip_hidden_reflected = strip_hidden[:, ::-1]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_hidden = hidden[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        strip_hidden_reflected = strip_hidden[::-1, :]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    raise AssertionError("no reflected strip found")


def _reflection_orbit_values(
    g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]
) -> set[int]:
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
        if value not in hidden_colors:
            values.add(value)

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def _recover_cell(g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]) -> int:
    values = _reflection_orbit_values(g, r, c, lo, hidden_colors)
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
    value = int(g[c, r])
    assert value not in hidden_colors
    return value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel a nested two-color marker in two stages and return the innermost hidden patch.

    Input structure:
    - A square colored pattern still obeys the source task's LR+UD reflection symmetry on a large lower/right strip.
    - The occluder is now nested: a solid inner rectangle (color 0) sits strictly inside a larger outer rectangle whose
      remaining cells form a solid ring (color 8).
    - Some inner-marker cells can only be determined after the surrounding ring has first been restored.

    Rule:
    - Detect the nested marker pair, then infer the reflected strip start while ignoring both marker colors.
    - First restore the outer-ring cells from reflection or transpose witnesses, leaving the inner marker untouched.
    - Then restore the inner rectangle on that partially repaired grid and output the recovered inner patch.

    Key insight:
    The two markers represent two occlusion layers over the same symmetric pattern: the inner rectangle is visible from
    the start, but some of its needed witnesses are themselves hidden by the outer ring, so the outer layer must be
    removed before the inner one can be solved.
    """
    g = np.asarray(grid, dtype=int)
    outer, inner = _nested_markers(g)
    lo = _find_reflected_strip_start(g, {_OUTER_MARKER, _INNER_MARKER})

    stage1 = g.copy()
    or0, oc0, or1, oc1 = outer
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            if _in_rect(r, c, inner):
                continue
            stage1[r, c] = _recover_cell(g, r, c, lo, {_OUTER_MARKER, _INNER_MARKER})

    ir0, ic0, ir1, ic1 = inner
    h = int(ir1 - ir0 + 1)
    w = int(ic1 - ic0 + 1)
    out = np.empty((h, w), dtype=int)
    for rr in range(h):
        for cc in range(w):
            r = int(ir0 + rr)
            c = int(ic0 + cc)
            out[rr, cc] = _recover_cell(stage1, r, c, lo, {_INNER_MARKER})

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# --- CFG: outer marker size ---
_OUTER_DIM_LO = 5
_OUTER_DIM_HI = 10
_OUTER_AREA_LO = 24
_OUTER_AREA_HI = 56

# --- CFG: inner marker size ---
_INNER_DIM_LO = 3
_INNER_DIM_HI = 7
_INNER_AREA_LO = 9
_INNER_AREA_HI = 25

# --- CFG: generation retries ---
_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_OUTER_SIZE_PAIRS = [
    (h, w)
    for h in range(_OUTER_DIM_LO, _OUTER_DIM_HI + 1)
    for w in range(_OUTER_DIM_LO, _OUTER_DIM_HI + 1)
    if _OUTER_AREA_LO <= h * w <= _OUTER_AREA_HI
]
_INNER_SIZE_PAIRS = [
    (h, w)
    for h in range(_INNER_DIM_LO, _INNER_DIM_HI + 1)
    for w in range(_INNER_DIM_LO, _INNER_DIM_HI + 1)
    if _INNER_AREA_LO <= h * w <= _INNER_AREA_HI
]
_OUTER_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}
_INNER_PLACEMENT_CACHE: dict[
    tuple[int, int, int, int], list[tuple[int, int, int, int, list[tuple[int, int]], list[tuple[int, int]]]]
] = {}


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


def _outer_valid_placements(h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (h, w)
    if key not in _OUTER_PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _OUTER_PLACEMENT_CACHE[key] = placements
    return _OUTER_PLACEMENT_CACHE[key]


def _dependent_inner_cells(
    outer_r0: int, outer_c0: int, outer_h: int, outer_w: int, inner_r0: int, inner_c0: int, inner_h: int, inner_w: int
) -> list[tuple[int, int]]:
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    inner = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)
    dependent: list[tuple[int, int]] = []
    for r in range(inner_r0, inner_r0 + inner_h):
        for c in range(inner_c0, inner_c0 + inner_w):
            outside_inner = [(rr, cc) for rr, cc in _orbit(r, c) if not _in_rect(rr, cc, inner)]
            if outside_inner and all(_in_rect(rr, cc, outer) for rr, cc in outside_inner):
                dependent.append((r, c))
    return dependent


def _inner_valid_placements(
    outer_r0: int, outer_c0: int, outer_h: int, outer_w: int
) -> list[tuple[int, int, int, int, list[tuple[int, int]], list[tuple[int, int]]]]:
    key = (outer_r0, outer_c0, outer_h, outer_w)
    if key not in _INNER_PLACEMENT_CACHE:
        placements: list[tuple[int, int, int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
        for inner_h, inner_w in _INNER_SIZE_PAIRS:
            if inner_h > outer_h - 2 or inner_w > outer_w - 2:
                continue
            for inner_r0 in range(outer_r0 + 1, outer_r0 + outer_h - inner_h):
                for inner_c0 in range(outer_c0 + 1, outer_c0 + outer_w - inner_w):
                    fallback = _fallback_cells(inner_r0, inner_c0, inner_h, inner_w)
                    if fallback is None:
                        continue
                    dependent = _dependent_inner_cells(
                        outer_r0, outer_c0, outer_h, outer_w, inner_r0, inner_c0, inner_h, inner_w
                    )
                    if dependent:
                        placements.append((inner_r0, inner_c0, inner_h, inner_w, fallback, dependent))
        _INNER_PLACEMENT_CACHE[key] = placements
    return _INNER_PLACEMENT_CACHE[key]


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
            parent.setdefault(coord, coord)
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_clean() -> np.ndarray:
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


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()

    outer_h, outer_w = _OUTER_SIZE_PAIRS[int(_RNG.integers(len(_OUTER_SIZE_PAIRS)))]
    outer_placements = _outer_valid_placements(outer_h, outer_w)
    if not outer_placements:
        return None
    outer_r0, outer_c0, outer_fallback = outer_placements[int(_RNG.integers(len(outer_placements)))]

    inner_placements = _inner_valid_placements(outer_r0, outer_c0, outer_h, outer_w)
    if not inner_placements:
        return None
    inner_r0, inner_c0, inner_h, inner_w, inner_fallback, dependent = inner_placements[
        int(_RNG.integers(len(inner_placements)))
    ]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in outer_fallback]
    groups += [_orbit(r, c) | _orbit(c, r) for r, c in inner_fallback]
    _unify_groups(clean, groups)

    expected = clean[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w].copy()

    inp = clean.copy()
    inp[outer_r0 : outer_r0 + outer_h, outer_c0 : outer_c0 + outer_w] = _OUTER_MARKER
    inp[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = _INNER_MARKER

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
    raise RuntimeError("failed to generate a valid 0934a4d8 D13 input")
