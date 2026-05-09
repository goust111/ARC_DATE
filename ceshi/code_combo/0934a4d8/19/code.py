# Combo variant of 0934a4d8 (index 19)
# combo_spec: D2/3+D8/1
# Name: 空心矩形框标记+对称性破坏识别

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
from arc_utils import bbox


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The true occluder is no longer a solid block. It is a 1-cell-thick hollow
#      rectangular frame, so only the perimeter pixels are hidden and the interior
#      pixels stay visible in the input.
#    - The color cannot be found by "unique hollow frame" either. Generation adds
#      one extra centered hollow frame of another reserved color. Both look like
#      plausible markers by shape, but only the real marker color is responsible
#      for the broken LR+UD symmetry of the grid.
#    - Example trace on a 4x6 bbox:
#      original task: one solid 4x6 block -> detect the only filled rectangle ->
#      recover all 24 cells.
#      D2-only: one hollow 4x6 frame -> detect the only frame -> copy the 8 inner
#      cells directly and recover only the 16 perimeter cells.
#      combo: two hollow frames are visible. Shape-only detection is ambiguous.
#      The solver must test colors: ignoring the decoy still leaves symmetry
#      violations from the true marker, while ignoring the marker restores the
#      bilateral checks and reveals the same `lo`. After that, only frame cells
#      are reconstructed and the interior is copied as-is.
# b) Complexity calibration
#    - The frame thickness stays fixed at 1 cell, matching the D2 variant and
#      keeping the visible concept legible from few examples.
#    - Only one symmetric decoy frame is added. That is enough to make the
#      symmetry-based color test necessary without cluttering the board.
#    - Background colors are restricted to 1..7; marker and decoy use 8 and 9.
#      This keeps the special roles visually stable across examples.
# c) Implicit assumption audit of the original main()
#    - `_solid_rectangle_marker(...)` assumed the marker filled its bbox. False in
#      the combo: we need a hollow-frame test instead.
#    - The old solver also assumed the marker cue was unique by shape. False now:
#      two frame-shaped colors exist, so color identification must be driven by
#      "ignoring this color restores both symmetries".
#    - The old output loop reconstructed every bbox cell. False now because inner
#      cells remain visible; the combo solver must copy the bbox first, then only
#      overwrite frame positions.
#    - `_find_reflected_strip_start(...)` remains usable if its second argument is
#      interpreted as "ignored candidate color" instead of "already-known marker".
#    - The transpose fallback logic is still valid, but only for hidden frame
#      cells whose entire reflection orbit lies on the marker frame.
# d) Placement legality
#    - The marker and the decoy must have disjoint outer bboxes so their roles are
#      visually separable and their masks never merge.
#    - The decoy frame must itself be LR+UD symmetric, so it is centered on the
#      original reflection axes and uses even outer dimensions.
#    - Any marker frame cell whose whole reflection orbit is also on the frame may
#      only occur when a transpose fallback cell `(c, r)` remains visible and
#      non-marker, exactly as in the source task.
# e) Visual observability
#    - A human sees two conspicuous hollow frames. The output is one full rectangle
#      patch, so the repeated lesson across examples is: the correct frame is the
#      one whose color explains the symmetry break, not just any outlined box.
#    - The preserved interior makes the "recover only the border" rule visible.
# f) Visual cleanliness
#    - There are only three layers: symmetric background, one centered decoy
#      frame, and one asymmetric marker frame. No overlaps and no extra glyphs.
# g) Change inferability
#    - No hidden arithmetic or color ordering is used. The only color-role rule is
#      structural: both 8 and 9 can form frames, but exactly one frame color is
#      symmetry-breaking. The recovered colors themselves still come from mirror or
#      transpose counterparts of the same clean pattern.
# New solver assumptions that the generator must guarantee
#    - Exactly two colors form hollow rectangular frames: the marker and the decoy.
#    - Exactly one of those frame colors restores the bilateral symmetry checks
#      when treated as unknown, and that color is the true marker.
#    - Every transpose fallback read lands on a visible non-marker cell.
#    - The clean grid under both frames still obeys the original LR+UD symmetry on
#      `[lo..N-1]`.


BBox = tuple[int, int, int, int]


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_cell(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= r < r0 + h and c0 <= c < c0 + w and (r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1))


def _try_hollow_rectangle_bbox(g: np.ndarray, color: int) -> BBox | None:
    mask = g == int(color)
    b = bbox(mask)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    frame = _frame_mask(h, w)
    if int(mask.sum()) != int(frame.sum()):
        return None
    if not bool(np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], frame)):
        return None
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


def _symmetry_breaking_frame_marker(g: np.ndarray) -> tuple[int, BBox, int]:
    candidates: list[tuple[int, BBox, int]] = []
    for color in np.unique(g):
        frame_bbox = _try_hollow_rectangle_bbox(g, int(color))
        if frame_bbox is None:
            continue
        lo = _try_reflected_strip_start(g, int(color))
        if lo is None:
            continue
        candidates.append((int(color), frame_bbox, int(lo)))

    assert len(candidates) == 1
    return candidates[0]


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
    Restore the rectangular patch outlined by the symmetry-breaking hollow frame.

    Input structure:
    - A square colored grid contains two conspicuous hollow rectangular frames of thickness 1.
    - One frame is a centered symmetric decoy, while the other frame hides only its perimeter and breaks the grid's LR+UD symmetry.
    - The frame interior remains visible, and top/left border cells can still provide transpose fallbacks for fully hidden frame orbits.

    Rule:
    - Test frame-colored candidates by temporarily ignoring that color in the bilateral reflection checks.
    - The true marker is the unique hollow-frame color whose removal makes both reflections consistent and reveals the strip start.
    - Output the marker's full outer bbox: copy the visible interior directly, and reconstruct only the frame cells from their reflection orbit or, when needed, from the visible transposed fallback cell.

    Key insight:
    Shape alone is insufficient because there are two hollow frames. The decisive clue is that only one frame color explains every symmetry violation, and that frame hides only the border of the answer patch.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1), lo = _symmetry_breaking_frame_marker(g)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    frame = _frame_mask(h, w)
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    for rr in range(h):
        for cc in range(w):
            if not frame[rr, cc]:
                continue
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

# --- CFG: colors ---
_BASE_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MARKER = 8
_DECOY = 9

# --- CFG: marker outer-bbox size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: centered decoy outer-bbox size ---
_DECOY_SIZE_PAIRS = [(h, w) for h in (4, 6, 8) for w in (4, 6, 8)]

# --- CFG: generation retries ---
_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_Placement = tuple[int, int, list[tuple[int, int]]]
_PLACEMENT_CACHE: dict[tuple[int, int], tuple[list[_Placement], list[_Placement]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if not _is_frame_cell(r, c, r0, c0, h, w):
                continue
            if all(_is_frame_cell(rr, cc, r0, c0, h, w) for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                if _is_frame_cell(c, r, r0, c0, h, w):
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int) -> tuple[list[_Placement], list[_Placement]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[_Placement] = []
        fallback_placements: list[_Placement] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w)
                if fallback is None:
                    continue
                placement = (r0, c0, fallback)
                placements.append(placement)
                if fallback:
                    fallback_placements.append(placement)
        _PLACEMENT_CACHE[key] = (placements, fallback_placements)
    return _PLACEMENT_CACHE[key]


def _rects_overlap(a: BBox, b: BBox) -> bool:
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


def _centered_decoy_bbox(h: int, w: int) -> BBox:
    # For the [lo..hi] reflection axis, a self-mapped bbox satisfies r0+r1=lo+hi
    # and c0+c1=lo+hi. Using even sizes keeps the centered frame integral.
    r0 = (_LO + _HI - (h - 1)) // 2
    c0 = (_LO + _HI - (w - 1)) // 2
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _paint_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    patch = g[r0 : r0 + h, c0 : c0 + w]
    patch[_frame_mask(h, w)] = color


def _shape_frame_colors(g: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(g) if _try_hollow_rectangle_bbox(g, int(color)) is not None]


def _variant_metrics(g: np.ndarray) -> dict[str, object]:
    frame_colors = _shape_frame_colors(g)
    symmetry_frame_colors = [color for color in frame_colors if _try_reflected_strip_start(g, color) is not None]
    return {
        "shape_frame_colors": tuple(frame_colors),
        "shape_frame_count": len(frame_colors),
        "symmetry_frame_colors": tuple(symmetry_frame_colors),
        "symmetry_frame_count": len(symmetry_frame_colors),
    }


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()

    decoy_h, decoy_w = _DECOY_SIZE_PAIRS[int(_RNG.integers(len(_DECOY_SIZE_PAIRS)))]
    decoy_bbox = _centered_decoy_bbox(decoy_h, decoy_w)
    dr0, dc0, dr1, dc1 = decoy_bbox
    _paint_frame(clean, dr0, dc0, dr1 - dr0 + 1, dc1 - dc0 + 1, _DECOY)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements, _ = _valid_placements(h, w)
    if not placements:
        return None

    allowed_fallback: list[_Placement] = []
    allowed_plain: list[_Placement] = []
    for placement in placements:
        r0, c0, fallback = placement
        marker_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _rects_overlap(marker_bbox, decoy_bbox):
            continue
        if fallback:
            allowed_fallback.append(placement)
        else:
            allowed_plain.append(placement)
    if not allowed_fallback and not allowed_plain:
        return None

    # Keep both reconstruction branches visible across examples instead of
    # letting fallback placements get drowned out inside the full placement set.
    prefer_fallback = bool(_RNG.integers(2))
    if prefer_fallback and allowed_fallback:
        pool = allowed_fallback
    elif not prefer_fallback and allowed_plain:
        pool = allowed_plain
    else:
        pool = allowed_plain if allowed_plain else allowed_fallback
    r0, c0, fallback = pool[int(_RNG.integers(len(pool)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    if len(np.unique(expected)) < 2:
        return None

    inp = clean.copy()
    _paint_frame(inp, r0, c0, h, w, _MARKER)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    metrics = _variant_metrics(inp)
    if metrics["shape_frame_count"] != 2:
        return None
    if metrics["symmetry_frame_count"] != 1:
        return None
    if tuple(metrics["shape_frame_colors"]) != (_MARKER, _DECOY):
        return None
    if tuple(metrics["symmetry_frame_colors"]) != (_MARKER,):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
