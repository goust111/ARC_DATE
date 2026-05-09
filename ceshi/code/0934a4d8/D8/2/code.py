# Variant of 0934a4d8
# Dimension: D8 — 标记颜色的确定方式 (match_cond)
# Change: 标记是唯一一个外接矩形被完全填满且面积大于1的颜色（实心矩形检测）... → 标记颜色由边界像素指示

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the D8 variant
# a) Concrete behavior:
#    - The occluder is still a solid rectangle, and the hidden content is still
#      recovered from LR/UD reflection orbits with transpose fallback.
#    - The visible change is only in how the solver identifies the occluder
#      color: one corner cell of the outer border is recolored to the marker
#      color, acting as a control pixel. The solver must inspect the corners,
#      find the corner color whose remaining pixels form a solid rectangle, and
#      then recover the covered patch as before.
#    - Small trace:
#      original: "find the only solid-rectangle color" -> use that color as the
#      marker -> recover hidden cells.
#      variant: "scan corners" -> remove one corner candidate from its color
#      mask -> if the rest is a solid rectangle, that corner is the hint for the
#      marker color -> recover hidden cells.
#      The difference is local but decisive: the marker color is no longer
#      inferable from rectangle-ness alone, because the same color also appears
#      as an extra border pixel.
# b) Complexity calibration:
#    - The control signal uses only four salient locations (the corners), so the
#      rule is visible from a few examples and does not create a large search
#      space.
#    - The marker color varies between two unused colors {0, 8}; that is enough
#      to make the hint meaningful without introducing a hard color-mapping task.
# c) Hidden assumptions in the old main():
#    - _solid_rectangle_marker assumed all marker-colored pixels belonged to one
#      solid rectangle. The corner hint breaks that assumption because the same
#      color now has one extra isolated cell on the border.
#    - _find_reflected_strip_start and _reflection_orbit_values only need to
#      know the marker color, not why it was selected, so they remain valid.
#    - The transpose fallback assumes g[c, r] is not marker-colored whenever an
#      orbit is fully hidden. The generator still enforces that by checking
#      placements exactly as before and by keeping the corner hint away from the
#      rectangle's 8-neighborhood.
# d) Placement legality:
#    - The marker rectangle must stay disjoint from the chosen corner's 2x2
#      neighborhood so the hint remains a single isolated pixel.
#    - The original fallback placement constraints are preserved, so every
#      hidden cell is still recoverable.
# e) Visual observability:
#    - Across samples, the corner location changes and the marker color flips
#      between two values, so the border control pixel is a visible repeated
#      cue rather than a hidden implementation detail.
#    - The old "unique solid rectangle color" cue is intentionally destroyed:
#      the marker color now occupies a rectangle plus one separate corner cell.
# f) Visual cleanliness:
#    - Only one extra pixel is added, at a corner; it does not overlap other
#      structures or add clutter inside the patterned area.
# g) Color-change inferability:
#    - There is no arbitrary recoloring rule. The only color dependency is
#      identity: the corner hint and the solid occluder share the same color.
# Solver assumptions guaranteed by the generator:
#    - Exactly one corner color, after removing that corner cell, leaves a solid
#      rectangle of area > 1.
#    - The clean pattern never uses colors 0 or 8, so the hint color is not
#      confused with ordinary pattern cells.
#    - The rectangle does not touch the hint corner's 8-neighborhood, so the
#      control pixel is visually singleton.


def _solid_rectangle_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bbox if mask is a completely filled rectangle of area > 1."""
    b = bbox(mask)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    if area <= 1:
        return None
    if int(mask.sum()) != area:
        return None
    if not bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _corner_hint_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    """Find the marker color from the unique corner hint whose remaining pixels form a solid rectangle."""
    n, m = map(int, g.shape)
    corners = [(0, 0), (0, m - 1), (n - 1, 0), (n - 1, m - 1)]

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for r, c in corners:
        color = int(g[r, c])
        mask = g == color
        if int(mask.sum()) <= 1:
            continue
        masked = mask.copy()
        masked[r, c] = False
        b = _solid_rectangle_bbox(masked)
        if b is not None:
            candidates.append((color, b))

    assert len(candidates) == 1
    return candidates[0]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangular patch hidden by the hinted marker using the grid's reflection symmetries.

    Input structure:
    - A square colored pattern contains one solid rectangular occluder plus one corner control pixel of the same color.
    - The visible pattern is still horizontally and vertically symmetric on a large lower/right strip.
    - The top and left outer strips still expose transpose copies for cells whose full reflection orbit is hidden.

    Rule:
    - Inspect the four corners and find the unique corner color whose pixels, after removing that corner cell, form a solid rectangle; that color is the marker and that rectangle is the output region.
    - Infer the reflected strip start as the smallest row/column offset whose lower/right strips are mirror-consistent when marker-colored cells are ignored.
    - For each marker-covered cell, copy the unique non-marker value from its reflection orbit; if the whole orbit is also covered, read the transposed counterpart from the exposed outer strip.

    Key insight:
    The corner pixel is a control hint, not part of the missing patch. Once that hint reveals the marker color, the hidden rectangle is recovered exactly as in the original symmetry-based task.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _corner_hint_marker(g)
    lo = _find_reflected_strip_start(g, marker)

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


# --------------- generator (D8 variant) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER_COLORS = np.array([0, 8], dtype=int)
_CONTROL_CORNERS = ((0, 0), (0, _HI), (_HI, 0), (_HI, _HI))

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
    tuple[int, int, tuple[int, int]], list[tuple[int, int, list[tuple[int, int]]]]
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


def _intersects_hint_guard(r0: int, c0: int, h: int, w: int, corner: tuple[int, int]) -> bool:
    """Keep the control pixel visually isolated by reserving its 2x2 corner neighborhood."""
    cr, cc = corner
    gr0, gr1 = (0, 1) if cr == 0 else (_N - 2, _N - 1)
    gc0, gc1 = (0, 1) if cc == 0 else (_N - 2, _N - 1)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    return not (r1 < gr0 or r0 > gr1 or c1 < gc0 or c0 > gc1)


def _fallback_cells(r0: int, c0: int, h: int, w: int, corner: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Identify marker cells whose entire orbit is inside the marker (need transpose fallback)."""
    if _intersects_hint_guard(r0, c0, h, w, corner):
        return None

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


def _valid_placements(h: int, w: int, corner: tuple[int, int]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Return all valid (r0, c0, fallback_cells) placements for a h x w marker, cached."""
    key = (h, w, corner)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w, corner)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
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
    marker = int(_RNG.choice(_MARKER_COLORS))
    corner = _CONTROL_CORNERS[int(_RNG.integers(len(_CONTROL_CORNERS)))]
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w, corner)
    if not placements:
        return None
    r0, c0, fallback = placements[int(_RNG.integers(len(placements)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = marker
    inp[corner] = marker

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
    raise RuntimeError("failed to generate a valid 0934a4d8 D8 input")
