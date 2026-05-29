# Combo variant of 0934a4d8 (index 15)
# combo_spec: D1/0+D4/0+D6/1
# Name: 单轴对称+禁止回退+无边界

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
from arc_utils import bbox

# Design analysis
# a) Concrete behavior and trace:
#    - Original task: the clean grid was doubly symmetric on a lower/right strip
#      [lo..N-1], and a hidden cell could use the whole LR+UD reflection orbit.
#      If that orbit was fully covered near the free border, the solver fell back
#      to the transpose cell.
#    - Combo task: there is exactly one global reflection axis on the full grid
#      (lo=0), either LR or UD. A hidden cell now has only one usable mirror
#      partner: (r, N-1-c) for LR or (N-1-r, c) for UD.
#    - Example with N=30, LR case: cell (4, 6) used to be recoverable from
#      {(4,6), (4,23), (25,6), (25,23)} plus maybe transpose. Now only (4,23)
#      matters. If the marker also covered column 23, recovery would fail, so the
#      generator forbids marker intervals that overlap their own mirror image.
# b) Complexity calibration:
#    - Removing one axis and the transpose fallback simplifies the local copying
#      rule, but the task still requires global inference: the solver must decide
#      whether the entire board is LR-only or UD-only.
#    - To prevent degeneration into the original or a visually ambiguous sample,
#      the generator rejects boards that also satisfy the orthogonal symmetry.
# c) Hidden assumptions from the old main():
#    - `_find_reflected_strip_start` assumed some lo and required both LR and UD.
#      Neither assumption survives. Here lo is fixed to 0 by construction, and
#      exactly one axis must match.
#    - `_reflection_orbit_values` assumed multiple reflected witnesses might exist.
#      In the combo version every hidden cell must have exactly one visible mirror
#      mate, so the solver reads a single partner instead of a multi-cell orbit.
#    - The old `else` branch relied on transpose fallback. That path is invalid in
#      this combo and must disappear entirely.
# d) Placement legality:
#    - The marker stays a solid rectangle inside the 30x30 board.
#    - For LR samples, its column interval must be disjoint from the mirrored
#      column interval; for UD samples, the same disjointness is enforced on rows.
#      This guarantees that no cell and its only mirror partner are both hidden.
# e) Visual observability:
#    - Because lo=0, the symmetry is visible all the way to the outermost edge;
#      the first row/column participates like every other row/column.
#    - Across examples, some boards are LR and others are UD, while the opposite
#      direction visibly disagrees on exposed cells.
# f) Visual cleanliness:
#    - There is still only one salient object: a single solid rectangle of marker
#      color 8 hiding a same-sized answer patch.
#    - The background texture is a clean mirror pattern rather than layered cues,
#      so the relation between visible half and hidden half stays legible.
# g) Change inferability:
#    - No color remapping is introduced. Hidden colors are copied directly from a
#      visible mirrored cell, so the only inference burden is structural: detect
#      the active symmetry axis and apply reflection.


_AXES = ("lr", "ud")


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


def _matches_axis(g: np.ndarray, marker: int, axis: str) -> bool:
    n, m = map(int, g.shape)
    assert n == m

    if axis == "lr":
        reflected = g[:, ::-1]
    else:
        reflected = g[::-1, :]

    known = (g != marker) & (reflected != marker)
    return not bool(np.any((g != reflected) & known))


def _detect_axis(g: np.ndarray, marker: int) -> str:
    candidates = [axis for axis in _AXES if _matches_axis(g, marker, axis)]
    assert len(candidates) == 1, f"expected one reflection axis, got {candidates}"
    return candidates[0]


def _mirror_partner(r: int, c: int, axis: str, n: int) -> tuple[int, int]:
    hi = n - 1
    if axis == "lr":
        return int(r), int(hi - c)
    return int(hi - r), int(c)


def _orthogonal_mismatch_count(g: np.ndarray, marker: int, axis: str) -> int:
    other = "ud" if axis == "lr" else "lr"
    if other == "lr":
        reflected = g[:, ::-1]
    else:
        reflected = g[::-1, :]
    known = (g != marker) & (reflected != marker)
    return int(np.count_nonzero((g != reflected) & known))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the rectangle hidden by the solid marker using the grid's one surviving full-grid reflection axis.

    Input structure:
    - A square colored grid contains one solid rectangular marker that covers the answer patch.
    - In the clean grid, every cell belongs to exactly one global mirror system: either the whole board is left-right
      symmetric or the whole board is up-down symmetric.
    - There is no free border strip and no secondary transpose cue; the active symmetry reaches the outer edge.

    Rule:
    - Find the unique color whose complete footprint is a filled rectangle; that rectangle is the output window.
    - Determine whether the visible non-marker cells are consistent with LR or UD reflection on the full grid.
    - For each covered cell, copy the color from its unique partner across that detected axis.

    Key insight:
    The marker hides one side of a single-axis mirror pattern, so reconstruction comes entirely from identifying the
    board's lone reflection direction and reading the corresponding mirrored cells, with no fallback mechanism at all.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    axis = _detect_axis(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            pr, pc = _mirror_partner(r, c, axis, int(g.shape[0]))
            value = int(g[pr, pc])
            assert value != marker
            out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_HI = _N - 1
_HALF = _N // 2

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
_MIN_ORTHOGONAL_MISMATCH = 100

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
_NEXT_AXIS_INDEX = 0


def _marker_interval_is_valid(start: int, span: int) -> bool:
    end = start + span - 1
    for idx in range(start, end + 1):
        mirror = _HI - idx
        if start <= mirror <= end:
            return False
    return True


def _valid_placements(axis: str, h: int, w: int) -> list[tuple[int, int]]:
    key = (axis, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                if axis == "lr":
                    if not _marker_interval_is_valid(c0, w):
                        continue
                else:
                    if not _marker_interval_is_valid(r0, h):
                        continue
                placements.append((r0, c0))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean(axis: str) -> np.ndarray:
    clean = np.empty((_N, _N), dtype=int)

    if axis == "lr":
        left = _RNG.choice(_PALETTE, size=(_N, _HALF))
        clean[:, :_HALF] = left
        clean[:, _HALF:] = left[:, ::-1]
    else:
        top = _RNG.choice(_PALETTE, size=(_HALF, _N))
        clean[:_HALF, :] = top
        clean[_HALF:, :] = top[::-1, :]

    return clean


def _gen_attempt(axis: str) -> np.ndarray | None:
    clean = _build_clean(axis)
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(axis, h, w)
    if not placements:
        return None
    r0, c0 = placements[int(_RNG.integers(len(placements)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        found_axis = _detect_axis(inp, _MARKER)
        if found_axis != axis:
            return None
        if _orthogonal_mismatch_count(inp, _MARKER, found_axis) < _MIN_ORTHOGONAL_MISMATCH:
            return None
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    global _NEXT_AXIS_INDEX

    preferred_axis = _AXES[_NEXT_AXIS_INDEX % len(_AXES)]
    _NEXT_AXIS_INDEX += 1
    axis_order = (preferred_axis, "ud" if preferred_axis == "lr" else "lr")

    retries_per_axis = max(1, _MAX_RETRIES // len(axis_order))
    for axis in axis_order:
        for _ in range(retries_per_axis):
            grid = _gen_attempt(axis)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
