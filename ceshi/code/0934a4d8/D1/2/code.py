# Variant of 0934a4d8
# Dimension: D1 — 对称类型 (match_cond)
# Change: 双轴对称（LR + UD），反射轴位于 [lo..N-1] 区间的中心... → 对角线反射对称

from __future__ import annotations

import numpy as np

from arc_utils import bbox


# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   preserved:
#     - canvas stays 30x30 square
#     - the hidden region is still the unique solid rectangle of marker color 8
#     - output is still the original content of that rectangle
#     - generator still works output-first and validates via main(input) == expected
#   changed_core:
#     - replace LR+UD strip symmetry with exactly one diagonal reflection symmetry
#     - the clean grid is symmetric either across the main diagonal
#       (r, c) <-> (c, r) or across the anti-diagonal
#       (r, c) <-> (N-1-c, N-1-r)
#     - every marker cell is recovered from its single diagonal mirror partner
#   concrete_pixel_behavior:
#     - original task used 4-cell reflection orbits inside a lower/right strip and
#       sometimes needed a transpose fallback from the outer border
#     - this variant uses 2-cell diagonal orbits on the whole square, so there is
#       no strip inference and no fallback branch
#     - example: on a 6x6 main-diagonal grid, a hidden cell at (1, 4) must copy the
#       visible value at (4, 1); under the old rule it would have looked for left/right
#       and up/down reflections around a centered axis, which is the wrong geometry
#   complexity_calibration:
#     - using both main and anti diagonal modes keeps a visible task distinction and
#       forces the solver to identify which diagonal is active
#     - the generator rejects ambiguous cases where both diagonal symmetries happen
#       to hold after masking
#   implicit_assumption_review:
#     - original _find_reflected_strip_start depended on a bilateral strip [lo..N-1];
#       this no longer exists, so the entire inference must be replaced
#     - original _reflection_orbit_values assumed horizontal/vertical reflections can
#       enlarge an orbit to size 4; diagonal reflection orbits have size 1 or 2 instead
#     - original fallback branch assumed fully hidden orbits only occur near the outer
#       border and can be recovered by transpose; in the new variant we instead forbid
#       placements whose mirrored partner would also be hidden
#   placement_legality:
#     - for the active diagonal mode, the marker rectangle R must be disjoint from its
#       reflected rectangle mirror(R)
#     - if R intersected mirror(R), some hidden cell would map to another hidden cell
#       (or to itself on the diagonal), making recovery impossible
#     - the generator caches only placements satisfying this disjointness constraint
#   visual_observability:
#     - the whole clean grid visibly mirrors across one diagonal, so the new symmetry
#       type is readable from a few examples without using color ids as metadata
#     - random asymmetric half-grid fillings ensure the image does not collapse into a
#       pattern that also looks LR/UD symmetric
#   solver_generator_contract:
#     - main() assumes exactly one diagonal mode is consistent ignoring marker pixels
#     - main() assumes each marker cell's mirrored partner is visible and non-marker
#     - generate_input() enforces both assumptions before accepting a sample


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


def _reflected_grid(g: np.ndarray, mode: str) -> np.ndarray:
    if mode == "main":
        return g.T
    if mode == "anti":
        return g[::-1, ::-1].T
    raise AssertionError(f"unknown diagonal mode: {mode}")


def _diagonal_modes(g: np.ndarray, marker: int) -> list[str]:
    modes: list[str] = []
    for mode in ("main", "anti"):
        reflected = _reflected_grid(g, mode)
        known = (g != marker) & (reflected != marker)
        if not bool(np.any((g != reflected) & known)):
            modes.append(mode)
    return modes


def _reflect_coord(r: int, c: int, n: int, mode: str) -> tuple[int, int]:
    if mode == "main":
        return int(c), int(r)
    if mode == "anti":
        return int(n - 1 - c), int(n - 1 - r)
    raise AssertionError(f"unknown diagonal mode: {mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangular patch hidden by the solid marker using the grid's diagonal mirror symmetry.

    Input structure:
    - A square colored pattern contains one solid rectangular marker color covering a contiguous patch.
    - Away from the marker, the clean pattern is symmetric across exactly one diagonal of the whole grid.
    - The active mirror can be either the main diagonal or the anti-diagonal, and the marker stays off that diagonal so mirrored counterparts remain visible.

    Rule:
    - Find the unique color whose complete footprint is a filled rectangle; that rectangle defines the output window.
    - Decide which diagonal symmetry is consistent with the visible non-marker cells.
    - For each marker pixel, copy the value from its coordinate reflected across that diagonal and place it into the output rectangle.

    Key insight:
    The rectangle is missing data, not a real object: once the correct diagonal mirror is identified, every hidden cell has exactly one visible twin elsewhere in the grid.
    """
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m

    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    modes = _diagonal_modes(g, marker)
    assert len(modes) == 1
    mode = modes[0]

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            mr, mc = _reflect_coord(r, c, n, mode)
            value = int(g[mr, mc])
            assert value != marker
            out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30

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
_MODES = ("main", "anti")
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[str, int, int], list[tuple[int, int]]] = {}


def _rect_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _reflected_rect(r0: int, c0: int, h: int, w: int, mode: str) -> tuple[int, int, int, int]:
    if mode == "main":
        return c0, r0, c0 + w, r0 + h
    if mode == "anti":
        return _N - (c0 + w), _N - (r0 + h), _N - c0, _N - r0
    raise AssertionError(f"unknown diagonal mode: {mode}")


def _valid_placements(mode: str, h: int, w: int) -> list[tuple[int, int]]:
    key = (mode, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                rect = (r0, c0, r0 + h, c0 + w)
                if not _rect_overlap(rect, _reflected_rect(r0, c0, h, w, mode)):
                    placements.append((r0, c0))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean(mode: str) -> np.ndarray:
    clean = np.empty((_N, _N), dtype=int)

    if mode == "main":
        for r in range(_N):
            vals = _RNG.choice(_PALETTE, size=_N - r)
            clean[r, r:] = vals
            clean[r:, r] = vals
        return clean

    if mode == "anti":
        for r in range(_N):
            vals = _RNG.choice(_PALETTE, size=_N - r)
            cols = np.arange(_N - r)
            clean[r, cols] = vals
            clean[_N - 1 - cols, _N - 1 - r] = vals
        return clean

    raise AssertionError(f"unknown diagonal mode: {mode}")


def _gen_attempt() -> np.ndarray | None:
    mode = _MODES[int(_RNG.integers(len(_MODES)))]
    clean = _build_clean(mode)
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(mode, h, w)
    if not placements:
        return None
    r0, c0 = placements[int(_RNG.integers(len(placements)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        modes = _diagonal_modes(inp, _MARKER)
        if len(modes) != 1 or modes[0] != mode:
            return None
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
