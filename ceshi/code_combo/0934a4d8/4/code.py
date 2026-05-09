# Combo variant of 0934a4d8 (index 4)
# combo_spec: D1/3+D6/0
# Name: 平移对称+可变周期参数

from __future__ import annotations

import numpy as np

from arc_utils import bbox


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete fused behavior and trace
#    - Original seed:
#      1. detect the unique solid rectangular marker;
#      2. infer one bilateral-symmetry offset `lo`;
#      3. recover each hidden cell from an LR/UD reflection orbit, with a
#         transpose fallback when the orbit is fully hidden near the free border.
#    - Fused variant:
#      1. keep the same marker object, but replace mirror symmetry with
#         translation symmetry;
#      2. keep the original "free border starts at lo" idea, but make `lo`
#         vary per sample and let it define where full 2D periodic tiling begins;
#      3. recover hidden cells from adjacent tile copies:
#         * inside [lo..]x[lo..]: same row-residue and col-residue class;
#         * in the top border: same row, same column residue;
#         * in the left border: same column, same row residue.
#      Example on a small 12x12 board:
#      - choose lo=3, row_period=4, col_period=5;
#      - rows/cols 3..11 form a cropped 4x5 repeated tile;
#      - rows 0..2 repeat only horizontally from col 3 onward;
#      - cols 0..2 repeat only vertically from row 3 downward.
#      A hidden cell at (7,9) is restored from cells
#      (3,4), (3,9), (7,4), (11,9), ... with the same residues.
#      A hidden cell at (1,8) is restored from (1,3), (1,8), ... on that same
#      top-border row. This is visually and algorithmically different from the
#      seed's reflection-orbit logic.
#
# b) Complexity calibration
#    - `lo` is restricted to 1..4 and each period is restricted to 3..6.
#      This keeps the hidden parameters few and repeated across examples.
#    - The visible board still has many repeats per period, so humans can spot
#      the tiling by eye and the solver can recover every hidden cell from more
#      than one surviving copy.
#    - The border is not discarded: it changes the observable structure and
#      prevents the task from collapsing into the simpler "whole board is tiled"
#      variant.
#
# c) Implicit-assumption audit against the original solver
#    - `_find_reflected_strip_start`: invalid, because there is no mirror axis.
#    - `_reflection_orbit_values`: invalid, because equal-valued cells are linked
#      by translation classes, not reflections or transposition.
#    - `main()` now assumes:
#      1. there is exactly one solid rectangular marker color;
#      2. the true `lo` is the smallest offset for which the lower/right region
#         supports consistent row/column translations;
#      3. each marker cell has at least one visible classmate in its translation
#         class;
#      4. the marker never covers the top-left lo x lo corner, because that
#         corner has no repeated support.
#      The generator enforces all four conditions and rejects samples otherwise.
#
# d) Placement legality
#    - The marker remains one solid rectangle within the 30x30 board.
#    - It may overlap the top border or the left border, but never the top-left
#      corner, and every covered cell is checked to have a visible recovery copy.
#
# e) Visual observability
#    - Across 2-4 examples, humans can see:
#      * the width of the free border changes;
#      * the interior is a periodic tapestry instead of a mirrored figure;
#      * the top strip and left strip only repeat in one axis.
#    - None of these cues depend on numeric color ordering.
#
# f) Visual cleanliness
#    - There is still only one overlay object: the marker rectangle.
#    - The board is partitioned into three clean regimes: top strip, left strip,
#      and interior block. No extra symbols or layered annotations are added.
#
# g) Change inferability
#    - No arbitrary color remapping is introduced.
#    - The only hidden variables are `lo`, row period, and column period, each
#      supported by repeated visible evidence.
#
# 9) Audit of new solver assumptions
#    - `_find_structure()` assumes smaller fake `lo` values are broken by the
#      border rows/cols. The generator constructs border patterns that disagree
#      with the interior classes they would be forced to match under smaller `lo`.
#    - `_translation_class_values()` assumes top-border cells use horizontal
#      repetition only and left-border cells use vertical repetition only. The
#      generator fills those strips exactly that way.
#    - The `assert len(values) == 1` in `main()` assumes each class is
#      monochromatic and non-empty after masking. The generator uses one exact
#      periodic source pattern and rejects placements that would hide a whole
#      class.


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


def _axis_period_matches(g: np.ndarray, period: int, marker: int, axis: int, lo: int) -> bool:
    n = int(g.shape[axis])
    if period <= 0 or lo + period >= n:
        return False

    if axis == 0:
        a = g[lo : n - period, :]
        b = g[lo + period : n, :]
    else:
        a = g[:, lo : n - period]
        b = g[:, lo + period : n]

    known = (a != marker) & (b != marker)
    return not bool(np.any((a != b) & known))


def _find_period(g: np.ndarray, marker: int, axis: int, lo: int) -> int:
    n = int(g.shape[axis])
    for period in range(1, n - lo):
        if _axis_period_matches(g, period, marker, axis=axis, lo=lo):
            return period
    raise AssertionError("no visible period found")


def _find_structure(g: np.ndarray, marker: int) -> tuple[int, int, int]:
    n, m = map(int, g.shape)
    assert n == m

    max_lo = min(_MAX_LO_SEARCH, n - 2)
    for lo in range(1, max_lo + 1):
        try:
            row_period = _find_period(g, marker, axis=0, lo=lo)
            col_period = _find_period(g, marker, axis=1, lo=lo)
        except AssertionError:
            continue
        return lo, row_period, col_period

    raise AssertionError("no valid lo/period structure found")


def _translation_class_values(
    g: np.ndarray,
    r: int,
    c: int,
    lo: int,
    row_period: int,
    col_period: int,
    marker: int,
) -> set[int]:
    n = int(g.shape[0])
    if r < lo and c < lo:
        return set()

    if r < lo:
        rows = [int(r)]
    else:
        row_start = int(lo + (r - lo) % row_period)
        rows = range(row_start, n, row_period)

    if c < lo:
        cols = [int(c)]
    else:
        col_start = int(lo + (c - lo) % col_period)
        cols = range(col_start, n, col_period)

    values: set[int] = set()
    for rr in rows:
        for cc in cols:
            value = int(g[rr, cc])
            if value != marker:
                values.add(value)
    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the rectangle hidden by the solid marker using offset periodic tiling.

    Input structure:
    - A square colored board contains one solid rectangular marker color covering a patch.
    - The clean board has a variable top/left border width `lo`.
    - On rows and columns `lo..N-1`, the lower-right block is a cropped periodic tiling of one small tile.
      The top border rows repeat only horizontally from column `lo` onward, and the left border columns
      repeat only vertically from row `lo` downward.

    Rule:
    - Find the unique color whose full extent is a solid rectangle; that rectangle is the output window.
    - Infer the smallest `lo` whose lower/right region supports consistent row and column translation periods,
      then infer those smallest visible periods.
    - For each marker cell, collect the visible cells in the same translation class:
      same row-and-column residues in the interior block, same border row with matching column residue in the
      top strip, or same border column with matching row residue in the left strip.

    Key insight:
    The missing patch is one crop of a repeated tile system whose origin is shifted inward by an unknown border
    width, so solving requires discovering both the offset `lo` and the tile periods before copying from nearby
    repeats.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo, row_period, col_period = _find_structure(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _translation_class_values(g, r, c, lo, row_period, col_period, marker)
            assert len(values) == 1
            out[rr, cc] = values.pop()

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO_CHOICES = (1, 2, 3, 4)
_PERIOD_CHOICES = (3, 4, 5, 6)
_MAX_LO_SEARCH = max(_LO_CHOICES)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PERIOD_PAIRS = [(rp, cp) for rp in _PERIOD_CHOICES for cp in _PERIOD_CHOICES]
_PLACEMENT_CACHE: dict[tuple[int, int, int, int, int], list[tuple[int, int]]] = {}


def _has_smaller_cyclic_period_1d(values: np.ndarray) -> bool:
    n = int(values.shape[0])
    for period in range(1, n):
        if bool(np.array_equal(values, np.roll(values, -period))):
            return True
    return False


def _has_smaller_row_period(tile: np.ndarray) -> bool:
    h = int(tile.shape[0])
    for period in range(1, h):
        if bool(np.array_equal(tile, np.roll(tile, -period, axis=0))):
            return True
    return False


def _has_smaller_col_period(tile: np.ndarray) -> bool:
    w = int(tile.shape[1])
    for period in range(1, w):
        if bool(np.array_equal(tile, np.roll(tile, -period, axis=1))):
            return True
    return False


def _sample_exact_cycle(length: int, forbidden: np.ndarray | None = None) -> np.ndarray:
    while True:
        values = _RNG.choice(_PALETTE, size=length)
        if _has_smaller_cyclic_period_1d(values):
            continue
        if forbidden is not None and bool(np.array_equal(values, forbidden)):
            continue
        return values.astype(int)


def _sample_exact_tile(row_period: int, col_period: int) -> np.ndarray:
    while True:
        tile = _RNG.choice(_PALETTE, size=(row_period, col_period))
        if _has_smaller_row_period(tile) or _has_smaller_col_period(tile):
            continue
        return tile.astype(int)


def _build_clean(lo: int, row_period: int, col_period: int) -> np.ndarray:
    clean = np.empty((_N, _N), dtype=int)
    tile = _sample_exact_tile(row_period, col_period)

    for r in range(lo, _N):
        rr = int((r - lo) % row_period)
        for c in range(lo, _N):
            clean[r, c] = int(tile[rr, (c - lo) % col_period])

    for r in range(lo):
        forbidden = tile[(r - lo) % row_period, :]
        pattern = _sample_exact_cycle(col_period, forbidden=forbidden)
        for c in range(lo, _N):
            clean[r, c] = int(pattern[(c - lo) % col_period])

    for c in range(lo):
        forbidden = tile[:, (c - lo) % col_period]
        pattern = _sample_exact_cycle(row_period, forbidden=forbidden)
        for r in range(lo, _N):
            clean[r, c] = int(pattern[(r - lo) % row_period])

    clean[:lo, :lo] = _RNG.choice(_PALETTE, size=(lo, lo))
    return clean


def _in_marker(rr: int, cc: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= rr < r0 + h and c0 <= cc < c0 + w


def _class_has_visible_copy(
    r: int,
    c: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    lo: int,
    row_period: int,
    col_period: int,
) -> bool:
    if r < lo and c < lo:
        return False

    if r < lo:
        rows = [int(r)]
    else:
        row_start = int(lo + (r - lo) % row_period)
        rows = range(row_start, _N, row_period)

    if c < lo:
        cols = [int(c)]
    else:
        col_start = int(lo + (c - lo) % col_period)
        cols = range(col_start, _N, col_period)

    for rr in rows:
        for cc in cols:
            if not _in_marker(rr, cc, r0, c0, h, w):
                return True
    return False


def _placement_is_valid(r0: int, c0: int, h: int, w: int, lo: int, row_period: int, col_period: int) -> bool:
    if r0 < lo and c0 < lo:
        return False
    if r0 < lo and c0 + w > lo:
        pass
    elif c0 < lo and r0 + h > lo:
        pass
    elif r0 >= lo or c0 >= lo:
        pass
    else:
        return False

    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if not _class_has_visible_copy(r, c, r0, c0, h, w, lo, row_period, col_period):
                return False
    return True


def _valid_placements(h: int, w: int, lo: int, row_period: int, col_period: int) -> list[tuple[int, int]]:
    key = (h, w, lo, row_period, col_period)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                if _placement_is_valid(r0, c0, h, w, lo, row_period, col_period):
                    placements.append((r0, c0))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _placement_kind(r0: int, c0: int, lo: int) -> str:
    if r0 < lo:
        return "top_border"
    if c0 < lo:
        return "left_border"
    return "interior"


def _gen_attempt() -> np.ndarray | None:
    lo = int(_RNG.choice(_LO_CHOICES))
    row_period, col_period = _PERIOD_PAIRS[int(_RNG.integers(len(_PERIOD_PAIRS)))]
    clean = _build_clean(lo, row_period, col_period)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w, lo, row_period, col_period)
    if not placements:
        return None
    buckets: dict[str, list[tuple[int, int]]] = {"top_border": [], "left_border": [], "interior": []}
    for r0, c0 in placements:
        buckets[_placement_kind(r0, c0, lo)].append((r0, c0))
    nonempty = [kind for kind, coords in buckets.items() if coords]
    chosen_kind = nonempty[int(_RNG.integers(len(nonempty)))]
    chosen_bucket = buckets[chosen_kind]
    r0, c0 = chosen_bucket[int(_RNG.integers(len(chosen_bucket)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if _find_structure(inp, _MARKER) != (lo, row_period, col_period):
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
