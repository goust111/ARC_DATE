# Combo variant of 0934a4d8 (index 17)
# combo_spec: D1/2+D5/1
# Name: 对角线对称+输出转置

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox


# Design analysis for the fused variant:
# - Core fusion:
#   The base task used LR+UD reflection orbits plus a transpose fallback, then
#   returned the hidden rectangle unchanged. This fusion replaces the symmetry
#   with main-diagonal reflection and changes the output to the recovered patch's
#   transpose. On a clean grid we require g[r, c] == g[c, r] for all cells.
# - Concrete pixel behavior with a small example:
#   Suppose a 3x5 marker hides rows 2..4 and cols 8..12. In the base task,
#   cell (3, 10) would search a horizontal/vertical reflection orbit and the
#   output would stay 3x5. In this fusion, the hidden value at (3, 10) equals
#   the visible value at its diagonal mirror (10, 3), and the final answer must
#   be arranged in a 5x3 output because the hidden 3x5 patch is transposed.
#   Thus output[2, 1] comes from hidden patch cell (1, 2), whose visible twin is
#   grid[c0 + 2, r0 + 1] = grid[10, 3].
# - Complexity calibration:
#   If square markers were allowed freely, many outputs would keep the same shape
#   and the D5/1 change would be much less visible. The generator therefore uses
#   only non-square marker sizes, so the row/column swap becomes an immediately
#   observable cue in every sample.
# - Implicit assumption review against the original solver:
#   1. _find_reflected_strip_start no longer makes sense because there is no
#      lower/right bilateral strip to infer; the whole square obeys one global
#      main-diagonal symmetry instead.
#   2. _reflection_orbit_values used 4-cell LR/UD orbits; diagonal reflection
#      reduces each hidden dependency to one mirrored twin.
#   3. The old transpose fallback branch is not a fallback anymore. The output
#      itself is transposed, so we deliberately read the mirrored twin in the
#      transposed rectangle directly.
#   4. main() now assumes every marker cell's diagonal twin is visible. The
#      generator enforces this by rejecting placements where the marker overlaps
#      its own transposed footprint.
# - Placement legality:
#   For a marker rectangle R = [r0:r0+h, c0:c0+w), the diagonal-mirror footprint
#   is T(R) = [c0:c0+w, r0:r0+h). If R intersects T(R), some hidden cell would
#   mirror to another hidden cell or to itself on the diagonal, so recovery would
#   fail. The generator caches only placements with R ∩ T(R) = empty.
# - Visual observability and cleanliness:
#   The full canvas is visibly symmetric about the main diagonal, and the marker
#   is a single solid rectangle of a reserved color 8. Because the output swaps
#   dimensions, a human can infer from 2-4 examples that the answer is not just
#   the masked rectangle but its transposed content.
# - Change inferability:
#   No hidden metadata, color ordering, or non-visual rule is used. The only cues
#   are geometric: diagonal mirroring in the input and swapped output dimensions.
# - Solver/generator contract introduced by the fusion:
#   1. the clean grid has exact main-diagonal symmetry;
#   2. the marker is the unique solid rectangle;
#   3. the marker rectangle is disjoint from its transposed footprint;
#   4. marker sizes are non-square so the transpose effect stays visible.
#   generate_input() explicitly enforces all four conditions.


_N = 30
_MARKER = 8
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

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
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI and h != w
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int]]] = {}


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


def _main_diagonal_consistent(g: np.ndarray, marker: int) -> bool:
    reflected = g.T
    known = (g != marker) & (reflected != marker)
    return not bool(np.any((g != reflected) & known))


def _rect_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _transposed_rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(c0), int(r0), int(c0 + w), int(r0 + h)


def _valid_placements(h: int, w: int) -> list[tuple[int, int]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                rect = (r0, c0, r0 + h, c0 + w)
                if not _rect_overlap(rect, _transposed_rect(r0, c0, h, w)):
                    placements.append((r0, c0))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean() -> np.ndarray:
    clean = np.empty((_N, _N), dtype=int)
    for r in range(_N):
        vals = _RNG.choice(_PALETTE, size=_N - r)
        clean[r, r:] = vals
        clean[r:, r] = vals
    return clean


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the marker-covered rectangle from main-diagonal symmetry and return its transposed content.

    Input structure:
    - A 30x30 square grid contains one solid rectangular marker in color 8.
    - In the clean version of the grid, every cell matches its main-diagonal mirror:
      g[r, c] == g[c, r].
    - The marker rectangle is placed away from the main diagonal and away from its
      transposed footprint, so each hidden cell has a visible mirrored twin.

    Rule:
    - Find the unique filled rectangle; that marker bbox is the hidden patch.
    - Confirm that the visible non-marker cells are consistent with main-diagonal symmetry.
    - Return the hidden patch with rows and columns swapped: for output cell (rr, cc),
      read the visible twin of hidden cell (r0 + cc, c0 + rr), namely grid[c0 + rr, r0 + cc].

    Key insight:
    Diagonal reflection and output transpose reinforce each other here: the answer is
    not the masked rectangle in place, but the same values seen in the rectangle mirrored
    across the main diagonal, which already has the required transposed orientation.
    """
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m == _N

    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    assert marker == _MARKER
    assert _main_diagonal_consistent(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((w, h), dtype=int)

    for rr in range(w):
        for cc in range(h):
            mr = int(c0 + rr)
            mc = int(r0 + cc)
            value = int(g[mr, mc])
            assert value != marker
            out[rr, cc] = value

    return out


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    r0, c0 = placements[int(_RNG.integers(len(placements)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].T.copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        marker, _bbox = _solid_rectangle_marker(inp)
        if marker != _MARKER:
            return None
        if not _main_diagonal_consistent(inp, _MARKER):
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
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
