# Variant of 0934a4d8
# Dimension: D2 — 标记物（遮挡区域）的形状 (src)
# Change: 实心填充矩形，颜色为8（azure），面积16-36，边长3-9... → 标记为空心矩形框

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The occluder is still the only azure object, but it is now a 1-cell-thick
#      rectangular frame. Its outer bbox keeps the original side-length regime:
#      3-9 by 3-9, bbox area 16-36. The frame hides only the perimeter pixels;
#      the interior cells remain visible and already equal the target patch.
#    - The output stays the frame's outer bbox, not a sparse mask, so the task
#      remains "extract the marked rectangle". The difference is that only border
#      cells must be reconstructed; interior cells are copied as-is.
#    - Example trace: if the bbox is 4x5, the original task would replace all 20
#      cells with color 8 and recover all 20 cells from symmetry. In this variant,
#      only the 14 perimeter cells are 8; the 6 interior cells are visible in the
#      input and should pass straight through to the output. Therefore:
#        old detector: requires mask area == bbox area, so it would miss the frame
#        new detector: requires mask == bbox perimeter, so it finds the marker
#        old recovery: loops over every cell as hidden
#        new recovery: loops over the perimeter as hidden, copies the interior
#      The visible difference is substantial: a hollow azure outline is obvious.
# b) Complexity calibration
#    - The frame thickness is fixed at 1 cell. That keeps the new concept legible
#      from a few ARC examples and avoids ambiguity about which layers are hidden.
#    - Marker color remains unique (8 never appears in the clean pattern), so the
#      solver does not need a separate color-disambiguation mechanism.
#    - Output size, symmetry axes, palette, and transpose fallback stay unchanged.
# c) Implicit assumption audit of the original main()
#    - `_solid_rectangle_marker(...)` relied on every bbox cell being marker-colored.
#      That assumption is false now, so detection must switch to a perimeter test.
#    - `out = np.empty((h, w))` plus "recover every cell" relied on the whole bbox
#      being hidden. That is no longer true; blindly re-inferring interior cells is
#      unnecessary and would add avoidable generator constraints. The fix is to copy
#      the visible bbox first, then overwrite only frame coordinates with recovered
#      values.
#    - The reflection-orbit logic itself remains valid because every hidden pixel is
#      still a pixel of the same underlying symmetric clean grid.
# d) Placement legality
#    - The generator must only place 1-cell-thick rectangular frames.
#    - For any hidden frame cell whose full LR+UD reflection orbit is also on the
#      frame, the transpose fallback cell `(c, r)` must stay visible and non-marker.
#    - To preserve the original fallback story, fully hidden orbits are only allowed
#      for cells touching the top/left free border (`r < lo or c < lo`).
# e) Visual observability
#    - The hollow marker is visibly different from the original solid block because
#      the interior remains multi-colored and readable inside the azure outline.
#    - Across examples, the solver's action is human-legible: "take the outlined
#      rectangle, keep the inside, reconstruct the missing border by symmetry."
# f) Visual cleanliness
#    - There is only one special object: the azure frame. No other layer overlaps it.
#      The visible interior reduces clutter instead of increasing it.
# g) Color-change inference
#    - No new color mapping is introduced. Colors are copied from mirror/transpose
#      counterparts exactly as in the source task.
# New solver assumptions that the generator must guarantee
#    - The unique marker color forms exactly one hollow rectangle of thickness 1.
#    - Any transpose fallback cell read by `main()` is visible (`!= marker`).
#    - The clean grid still satisfies the original LR+UD symmetry on `[lo..N-1]`.


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_cell(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= r < r0 + h and c0 <= c < c0 + w and (r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1))


def _hollow_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        frame = _frame_mask(h, w)
        if int(mask.sum()) != int(frame.sum()):
            continue
        if bool(np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], frame)):
            candidates.append((h * w, int(color), b))

    assert len(candidates) == 1
    _bbox_area, color, b = candidates[0]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the rectangle outlined by the hollow marker frame using the grid's mirror symmetries.

    Input structure:
    - A square colored pattern contains one unique azure rectangular frame that hides only its 1-cell-thick border.
    - The cells strictly inside that frame remain visible and already belong to the desired output patch.
    - The lower/right main strip is horizontally and vertically mirror-consistent, while the top/left free border still exposes transpose fallbacks.

    Rule:
    - Find the unique color whose full extent is a hollow rectangle frame; its outer bounding box is the output window.
    - Infer the mirrored strip start as the smallest offset whose lower/right strips are reflection-consistent when frame pixels are ignored.
    - Copy interior cells directly from the visible bbox, and reconstruct each frame cell from its reflection orbit; if the entire orbit is hidden by the frame, use the visible transposed cell instead.

    Key insight:
    The marker now removes only the perimeter, so the answer is the whole framed rectangle with an already-visible interior and a border recovered from symmetry.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _hollow_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)

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
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker outer-bbox size ---
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
_Placement = tuple[int, int, list[tuple[int, int]]]
_PLACEMENT_CACHE: dict[tuple[int, int], tuple[list[_Placement], list[_Placement]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    """Return the LR+UD reflection orbit of (r, c) within the [_LO.._HI] axes."""
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    """Identify hidden frame cells whose entire reflection orbit stays on the frame.

    Returns None if a placement would need an unavailable transpose fallback.
    """
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
    """Return cached valid placements, split by whether they exercise transpose fallback."""
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


def _paint_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    patch = g[r0 : r0 + h, c0 : c0 + w]
    patch[_frame_mask(h, w)] = color


def _gen_attempt() -> np.ndarray | None:
    """Try once to generate a valid input grid. Returns None on failure."""
    clean = _build_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements, fallback_placements = _valid_placements(h, w)
    if not placements:
        return None

    if fallback_placements and bool(_RNG.random() < 0.35):
        pool = fallback_placements
    else:
        pool = placements
    r0, c0, fallback = pool[int(_RNG.integers(len(pool)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    _paint_frame(inp, r0, c0, h, w, _MARKER)

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
    raise RuntimeError("failed to generate a valid 0934a4d8 input")
