# Variant of 0934a4d8
# Dimension: D8 — 标记颜色的确定方式 (match_cond)
# Change: 标记是唯一一个外接矩形被完全填满且面积大于1的颜色（实心矩形检测）... → 标记为网格中唯一不参与对称的颜色

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
from arc_utils import bbox


# Design analysis for the D8 variant
# a) Concrete behavior:
#    - The occluder is still a solid rectangle, and the output is still the clean content under it.
#    - The key visible change is not the shape of the marker, but how it must be identified:
#      the correct color is the only one whose removal restores the global LR+UD mirror checks.
#    - To make that observable, generation adds a second solid rectangle candidate: a centered
#      decoy block with a reserved non-marker color. Old "unique solid rectangle color" logic
#      now fails, while "ignore one color and re-check symmetry" isolates the marker.
#    - Small trace:
#      old task: color 8 is the only full solid rectangle -> pick 8 directly -> infer lo -> recover patch.
#      new task: both colors 8 and 9 are full solid rectangles -> old cue is ambiguous.
#      Instead test each color as a wildcard: ignoring 8 makes mirror checks pass, ignoring 9 does not
#      because the real marker still causes mismatches. The rest of the recovery pipeline stays unchanged.
# b) Complexity calibration:
#    - Only one extra decoy rectangle is added. That is enough to break the old cue without cluttering
#      the grid or introducing many branches.
#    - The clean pattern still uses one bilateral symmetry family plus the original transpose fallback,
#      so the reasoning jump is focused on D8 rather than a different reconstruction mechanism.
# c) Implicit assumptions audit:
#    - Any line that used `_solid_rectangle_marker` depended on uniqueness of that cue; this is no longer valid.
#    - `_find_reflected_strip_start` itself is still valid if we reinterpret its second argument as
#      "color to ignore during symmetry validation" rather than "already-known marker color".
#    - Orbit recovery assumes the chosen color is the real occluder and that its bbox is a filled rectangle.
#      The generator still guarantees that after the new symmetry-based identification.
# d) Placement legality:
#    - The marker rectangle keeps the original placement rules and transpose-fallback validity checks.
#    - The centered decoy rectangle is kept disjoint from the marker so it remains perfectly symmetric and
#      preserves its role as an old-rule distractor.
# e) Visual observability:
#    - Across examples, the solver-visible novelty is that there are at least two obvious filled rectangles,
#      but only one color explains the broken symmetry. This is inferable from a few examples because the
#      same "test by symmetry, not by shape" pattern repeats.
# f) Visual cleanliness:
#    - There are only three structural layers: symmetric background, one centered decoy block, one marker block.
#      No overlapping decorations are introduced.
# g) Color-rule inferability:
#    - No hidden arithmetic or ID-based ordering is used. Reserved colors are only used as visible roles:
#      color 8 is the occluder, color 9 is a symmetric distractor, and the background uses 1..7.


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


def _try_reflected_strip_start(g: np.ndarray, ignored: int) -> int | None:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != ignored) & (strip_reflected != ignored)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != ignored) & (strip_reflected != ignored)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    return None


def _symmetry_breaking_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int], int]:
    candidates: list[tuple[int, int]] = []
    for color in np.unique(g):
        lo = _try_reflected_strip_start(g, int(color))
        if lo is None:
            continue
        candidates.append((int(color), int(lo)))

    assert len(candidates) == 1
    color, lo = candidates[0]
    return color, _solid_rectangle_bbox(g, color), lo


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangle hidden by the color that uniquely disrupts the grid's bilateral symmetries.

    Input structure:
    - A square colored grid contains one rectangular occluder and one extra symmetric solid rectangle.
    - Away from the occluder, the clean pattern obeys left-right and up-down reflection on a large lower/right strip.
    - Cells near the top or left border may still need the original transpose fallback when every reflected mate is hidden.

    Rule:
    - Test each color as the possible "ignore this and symmetry works again" color.
    - The marker is the unique color whose removal makes both reflection checks consistent and reveals the strip start.
    - Once that color is identified, recover each covered cell from its reflection orbit; if the entire orbit is still hidden in the outer strip, copy the transposed counterpart instead.

    Key insight:
    The distracting rectangle is visually similar to the occluder, so shape alone is no longer enough.
    The correct marker is the only color that explains every symmetry violation at once.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1), lo = _symmetry_breaking_marker(g)

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


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_BASE_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_DECOY = 9
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: centered decoy rectangle ---
_DECOY_SIZE_PAIRS = [(h, w) for h in (2, 4, 6, 8) for w in (2, 4, 6, 8) if h * w >= 4]

# --- CFG: generation retries ---
_MAX_RETRIES = 400

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


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


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
        color = int(_RNG.choice(_BASE_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_BASE_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(_BASE_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(_BASE_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_BASE_PALETTE, size=(_LO, _LO))
    return clean


def _centered_decoy_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    r0 = (32 - h) // 2
    c0 = (32 - w) // 2
    return r0, c0, r0 + h - 1, c0 + w - 1


def _stamp_decoy(clean: np.ndarray, decoy_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = decoy_bbox
    clean[r0 : r1 + 1, c0 : c1 + 1] = _DECOY


def _old_solid_rectangle_candidates(g: np.ndarray) -> list[int]:
    candidates: list[int] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append(int(color))
    return candidates


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    symmetry_colors = [int(color) for color in np.unique(g) if _try_reflected_strip_start(g, int(color)) is not None]
    old_candidates = _old_solid_rectangle_candidates(g)
    return {
        "symmetry_candidate_count": len(symmetry_colors),
        "old_candidate_count": len(old_candidates),
        "old_has_marker": int(_MARKER in old_candidates),
        "old_has_decoy": int(_DECOY in old_candidates),
    }


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    decoy_h, decoy_w = _DECOY_SIZE_PAIRS[int(_RNG.integers(len(_DECOY_SIZE_PAIRS)))]
    decoy_bbox = _centered_decoy_bbox(decoy_h, decoy_w)
    _stamp_decoy(clean, decoy_bbox)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None

    allowed = []
    for r0, c0, fallback in placements:
        marker_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _rects_overlap(marker_bbox, decoy_bbox):
            allowed.append((r0, c0, fallback))
    if not allowed:
        return None

    r0, c0, fallback = allowed[int(_RNG.integers(len(allowed)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    if len(np.unique(expected)) < 2:
        return None

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    metrics = _variant_metrics(inp)
    if metrics["symmetry_candidate_count"] != 1:
        return None
    if metrics["old_candidate_count"] < 2:
        return None
    if not (metrics["old_has_marker"] and metrics["old_has_decoy"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 D8 input")
