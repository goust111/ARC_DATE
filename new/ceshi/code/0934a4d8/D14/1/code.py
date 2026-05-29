# Variant of 0934a4d8
# Dimension: D14 — 标记放置约束 (meta)
# Change: 标记可放置在网格任意位置，包括边界和角落区域；但受限于反射轨道的可恢复性约束... → 标记必须跨越对称轴
#
# Design analysis for the D14 variant
# a) Concrete behavior and trace:
#    - The clean grid keeps the original 30x30 bilateral mirror structure with axes
#      between rows 15/16 and columns 15/16 for the [2..29] strip.
#    - The marker is still a single solid rectangle of color 8, but now every valid
#      placement must cover row-pair 15/16 or column-pair 15/16. Visually the block
#      is forced into the middle band of the board instead of being fully peripheral.
#    - Example trace: if the marker spans columns 13..18, then a hidden cell at
#      (10,15) has its LR mirror at (10,16), also hidden because the rectangle crosses
#      the vertical axis. In the original task that pair would usually not both be
#      hidden. The solver still recovers the value from the same reflection orbit
#      because (21,15) and (21,16) remain visible. For rarer cells whose whole orbit is
#      hidden, the old code assumed this only happened near the outer border; the new
#      variant allows the same situation anywhere, so the transpose fallback must be
#      generalized from "border-only" to "any non-marker transpose".
# b) Complexity calibration:
#    - Colors, canvas size, marker size range, and the bilateral scaffold stay fixed.
#      This keeps the task recognizable as 0934a4d8 instead of turning into a new task.
#    - Only the placement distribution changes, but it changes a visually obvious
#      property: the marker always straddles a symmetry axis, so some direct reflected
#      partners are swallowed by the marker itself.
#    - We keep two repeated placement branches, horizontal-axis crossing and
#      vertical-axis crossing, instead of one-off exotic cases.
# c) Hidden-assumption audit of the original main():
#    - _solid_rectangle_marker: still valid; the marker remains a unique solid block.
#    - _find_reflected_strip_start: still valid because marker cells are ignored while
#      testing bilateral consistency.
#    - _reflection_orbit_values: still valid; crossing an axis only changes which orbit
#      members are visible, not what the orbit is.
#    - Original fallback branch:
#          assert r < lo or c < lo
#      This is no longer valid. In the variant, a fully hidden orbit can happen in a
#      central axis-crossing placement, so the fallback must only assume that the
#      transpose cell (c,r) is outside the marker and was color-synchronized during
#      generation.
# d) Placement legality:
#    - The rectangle stays in bounds and remains non-overlapping by construction because
#      it is the only explicit object.
#    - A placement is legal iff it crosses at least one symmetry axis and every fully
#      hidden reflection orbit has a transpose cell outside the marker. Generator-side
#      color unification then guarantees that transpose fallback returns the right value.
# e) Visual observability:
#    - From a few examples, a human can see that the teal rectangle is always centered
#      in one dimension and straddles the mirror line, unlike the source task where it
#      could sit in a corner or border.
#    - Output remains the hidden patch, so the relation between the central marker and
#      the recovered subgrid stays learnable.
# f) Visual cleanliness:
#    - No extra overlays or new symbols are introduced. The only visible difference is
#      marker placement, so the scene stays clean and close to the source task.
# g) Color-change inference:
#    - No new color mapping rule is introduced. Transpose fallback only enforces hidden
#      equality constraints internally during generation; it does not ask the solver to
#      infer any invisible color permutation.

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


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


def _axis_pair(lo: int, n: int) -> tuple[int, int]:
    hi = n - 1
    left = (lo + hi) // 2
    return left, left + 1


def _crosses_axis(start: int, end: int, lo: int, n: int) -> bool:
    axis_lo, axis_hi = _axis_pair(lo, n)
    return int(start) <= axis_lo and int(end) >= axis_hi


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle by combining bilateral reflection with a generalized transpose backup.

    Input structure:
    - A square colored grid contains one solid rectangular marker color that occludes a patch.
    - Outside the marker, the board is mirror-consistent left/right and up/down on the large [lo..N-1] strip.
    - In this variant, the marker must straddle at least one symmetry axis, so some marker cells have reflected partners that are also hidden inside the marker.

    Rule:
    - Detect the unique solid rectangle color and use its bounding box as the output window.
    - Infer the mirrored strip start by finding the smallest offset whose horizontal and vertical reflections match whenever both compared cells are visible.
    - For each hidden cell, read the unique visible color from its reflection orbit; if the full orbit is hidden by the axis-crossing marker, fall back to the transposed cell (c,r), which is guaranteed to stay visible in valid instances.

    Key insight:
    Crossing a symmetry axis can hide both members of a direct mirror pair, but the hidden orbit is still determined either by another visible orbit member or by a deliberately exposed transpose counterpart.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)
    n = int(g.shape[0])

    assert _crosses_axis(r0, r1, lo, n) or _crosses_axis(c0, c1, lo, n)

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
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXIS_LO, _AXIS_HI = _axis_pair(_LO, _N)

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
_PLACEMENT_CACHE: dict[
    tuple[int, int], dict[str, list[tuple[int, int, list[tuple[int, int]]]]]
] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    """Return the LR+UD reflection orbit of (r, c) within the [_LO.._HI] axes."""
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _placement_mode(r0: int, c0: int, h: int, w: int) -> str:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    cross_h = _crosses_axis(r0, r1, _LO, _N)
    cross_v = _crosses_axis(c0, c1, _LO, _N)
    if cross_h and cross_v:
        return "HV"
    if cross_h:
        return "H"
    if cross_v:
        return "V"
    return ""


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    """List marker cells whose whole reflection orbit is hidden and must use transpose.

    Returns None if the placement is invalid for the variant: it must cross at least
    one symmetry axis, and each fully hidden orbit must have its transpose outside the
    marker so the solver can read it.
    """
    mode = _placement_mode(r0, c0, h, w)
    if not mode:
        return None

    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int) -> dict[str, list[tuple[int, int, list[tuple[int, int]]]]]:
    """Return cached valid placements grouped by which symmetry axis is crossed."""
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: dict[str, list[tuple[int, int, list[tuple[int, int]]]]] = {
            "H": [],
            "V": [],
            "HV": [],
        }
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w)
                if fallback is None:
                    continue
                placements[_placement_mode(r0, c0, h, w)].append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
    """Merge coordinate groups with union-find so all members share a random color."""
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


def _build_clean() -> np.ndarray:
    """Build a clean NxN grid with bilateral LR+UD symmetry on the [lo..N-1] axes."""
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
    """Try once to generate a valid input grid. Returns None on failure."""
    clean = _build_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements_by_mode = _valid_placements(h, w)
    available_modes = [mode for mode in ("H", "V", "HV") if placements_by_mode[mode]]
    if not available_modes:
        return None

    mode = available_modes[int(_RNG.integers(len(available_modes)))]
    candidates = placements_by_mode[mode]
    r0, c0, fallback = candidates[int(_RNG.integers(len(candidates)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

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
    raise RuntimeError("failed to generate a valid 0934a4d8 D14 input")
