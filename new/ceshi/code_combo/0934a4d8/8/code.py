# Combo variant of 0934a4d8 (index 8)
# combo_spec: D2/0+D3/0
# Name: L/T形标记+输出完整修复网格

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the combo variant "L/T形标记+输出完整修复网格".
# a) Concrete behavior trace:
#    - Original: find the one solid rectangle color, infer its bbox, recover only
#      that rectangle, and return the cropped patch.
#    - New: find the one connected marker component whose exact silhouette is an
#      L or T polyomino, recover every marked cell in place, and return the full
#      repaired 30x30 grid.
#    - Small mental example: if a 6x6 clean grid had an L marker occupying five
#      cells in the top-left, the old solver would fail immediately because there
#      is no filled bbox to crop. The new solver instead treats the five marked
#      coordinates themselves as the destination set; each cell is restored from
#      its LR/UD reflection orbit, or from the transposed border counterpart when
#      the whole orbit is hidden by the marker.
# b) Complexity calibration:
#    - The marker family is intentionally small: only thick orthogonal L/T shapes,
#      with area 16-36 and thickness 2 or 3. This keeps the change visually obvious
#      in 2-4 examples without exploding the geometry search space.
#    - Returning the whole repaired grid is visibly different from the base task:
#      the output preserves all untouched context, so the solver must understand
#      "repair in place" rather than "crop the covered box".
# c) Broken assumptions from the original main():
#    - `_solid_rectangle_marker(...)` no longer applies because the marker bbox is
#      deliberately not full.
#    - `h,w = bbox size` can no longer define the output; the target is an arbitrary
#      cell set, and the output shape must equal the input shape.
#    - The reflection logic itself still holds, but it must iterate over marker
#      coordinates rather than over every cell in a rectangle.
# d) Placement legality enforced by the generator:
#    - The L/T mask is always a single connected component, stays in-bounds, and is
#      the only color-8 component.
#    - For each marked cell, either at least one orbit mate stays visible, or the
#      full orbit is hidden only in the outer border zone where transpose fallback
#      is legal, and the transpose cell is not also marked.
# e) Visual observability:
#    - Thick L/T silhouettes are easy to spot, unlike tiny accidental hooks.
#    - Outputs keep the full canvas, so the "same grid but repaired" relationship is
#      readable by eye, while the non-rectangular mask makes the bbox shortcut fail.
# f) Visual cleanliness:
#    - The clean grid remains the original bilateral pattern; only one large marker
#      overlays it, so there is no multi-object clutter competing with the shape cue.
# g) Post-change solver assumptions:
#    - There is exactly one connected marker component of an allowed L/T shape.
#    - Whenever a marked cell has no visible orbit value, `grid[c, r]` is a visible
#      non-marker transpose witness. `generate_input()` explicitly constructs only
#      such placements.


_MARKER = 8


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _normalize_component(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _classify_marker_shape(mask: np.ndarray) -> str | None:
    if mask.ndim != 2 or bool(np.all(mask)):
        return None

    for rot in range(4):
        m = np.rot90(mask, rot)
        h, w = map(int, m.shape)

        for thickness in range(1, min(h, w)):
            expected = np.zeros_like(m, dtype=bool)
            expected[:, :thickness] = True
            expected[:thickness, :] = True
            if np.array_equal(m, expected):
                return "L"

        for thickness in range(1, min(h, w)):
            if h <= thickness or w <= thickness:
                continue
            if (w - thickness) % 2 != 0:
                continue
            stem_c0 = (w - thickness) // 2
            if stem_c0 == 0 or stem_c0 + thickness == w:
                continue
            expected = np.zeros_like(m, dtype=bool)
            expected[:thickness, :] = True
            expected[thickness:, stem_c0 : stem_c0 + thickness] = True
            if np.array_equal(m, expected):
                return "T"

    return None


def _find_marker_component(g: np.ndarray) -> tuple[int, np.ndarray]:
    mask = g == _MARKER
    assert bool(mask.any())
    comps = _components(mask)
    assert len(comps) == 1
    norm = _normalize_component(comps[0])
    assert _classify_marker_shape(norm) is not None
    return _MARKER, mask


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


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker_mask: np.ndarray) -> set[int]:
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

        if not marker_mask[rr, cc]:
            values.add(int(g[rr, cc]))

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the full symmetric canvas after removing one non-rectangular L/T marker.

    Input structure:
    - A square colored grid contains one connected marker component whose color is unique and whose silhouette is an L or T polyomino, not a filled rectangle.
    - Away from that marker, the clean pattern obeys the same bilateral LR+UD symmetry as the base task on the strip [lo..N-1] in both axes.
    - Cells on the top/left outer border may be recovered through transpose fallback when every reflected copy of a marked cell is also covered.

    Rule:
    - Detect the single marker component by its exact L/T shape rather than by a solid bounding box.
    - Infer the reflected strip start `lo` exactly as in the base task, ignoring marker cells when checking mirror consistency.
    - For every marked coordinate, copy the unique visible value from its horizontal/vertical reflection orbit; if that orbit is fully hidden, use the transposed witness at `(c, r)`. Write the recovered values back into the original grid and output the entire repaired canvas.

    Key insight:
    The essential structure of the task is still "one symmetric image with one occluder", but the occluder is now an irregular connected shape and the target is the whole scene, so the solver must track the precise marked cell set and repair it in place instead of extracting a rectangular crop.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_mask = _find_marker_component(g)
    lo = _find_reflected_strip_start(g, marker)

    out = g.copy()
    for r, c in np.argwhere(marker_mask):
        rr = int(r)
        cc = int(c)
        values = _reflection_orbit_values(g, rr, cc, lo, marker_mask)
        if values:
            assert len(values) == 1
            out[rr, cc] = values.pop()
        else:
            assert rr < lo or cc < lo
            value = int(g[cc, rr])
            assert value != marker
            out[rr, cc] = value

    return out


# --------------- generator ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# --- CFG: marker shape ---
_MARKER_THICKNESSES = (2, 3)
_MARKER_DIM_LO = 4
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_PLACEMENT_CACHE: dict[tuple[tuple[int, int], bytes], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _build_marker_shapes() -> dict[str, list[np.ndarray]]:
    shapes: dict[str, list[np.ndarray]] = {"L": [], "T": []}
    seen: dict[str, set[tuple[tuple[int, int], bytes]]] = {"L": set(), "T": set()}

    for thickness in _MARKER_THICKNESSES:
        for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1):
            for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1):
                if h <= thickness or w <= thickness:
                    continue

                l_mask = np.zeros((h, w), dtype=bool)
                l_mask[:, :thickness] = True
                l_mask[:thickness, :] = True
                if _MARKER_AREA_LO <= int(l_mask.sum()) <= _MARKER_AREA_HI:
                    for rot in range(4):
                        rotated = np.rot90(l_mask, rot).copy()
                        key = _shape_key(rotated)
                        if key not in seen["L"]:
                            seen["L"].add(key)
                            shapes["L"].append(rotated)

                if (w - thickness) % 2 == 0:
                    stem_c0 = (w - thickness) // 2
                    if stem_c0 > 0 and stem_c0 + thickness < w:
                        t_mask = np.zeros((h, w), dtype=bool)
                        t_mask[:thickness, :] = True
                        t_mask[thickness:, stem_c0 : stem_c0 + thickness] = True
                        if _MARKER_AREA_LO <= int(t_mask.sum()) <= _MARKER_AREA_HI:
                            for rot in range(4):
                                rotated = np.rot90(t_mask, rot).copy()
                                key = _shape_key(rotated)
                                if key not in seen["T"]:
                                    seen["T"].add(key)
                                    shapes["T"].append(rotated)

    assert shapes["L"] and shapes["T"]
    return shapes


_MARKER_SHAPES = _build_marker_shapes()


def _translated_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _fallback_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]] | None:
    cells = _translated_cells(mask, r0, c0)
    fallback: list[tuple[int, int]] = []
    for r, c in cells:
        if any((rr, cc) not in cells for rr, cc in _orbit(r, c)):
            continue
        if not (r < _LO or c < _LO):
            return None
        if (c, r) in cells:
            return None
        fallback.append((r, c))
    return fallback


def _valid_placements(mask: np.ndarray) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = _shape_key(mask)
    if key not in _PLACEMENT_CACHE:
        h, w = map(int, mask.shape)
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(mask, r0, c0)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


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


def _marker_cells_match_shape(mask: np.ndarray) -> bool:
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    norm = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _classify_marker_shape(norm) is not None and not bool(np.all(norm))


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    kind = "L" if bool(_RNG.integers(2)) else "T"
    pool = _MARKER_SHAPES[kind]
    marker_mask = pool[int(_RNG.integers(len(pool)))]
    placements = _valid_placements(marker_mask)
    if not placements:
        return None

    with_fallback = [p for p in placements if p[2]]
    without_fallback = [p for p in placements if not p[2]]
    if with_fallback and without_fallback:
        bucket = with_fallback if bool(_RNG.integers(2)) else without_fallback
    else:
        bucket = with_fallback or without_fallback
    r0, c0, fallback = bucket[int(_RNG.integers(len(bucket)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    inp = clean.copy()
    h, w = map(int, marker_mask.shape)
    window = inp[r0 : r0 + h, c0 : c0 + w]
    window[marker_mask] = _MARKER

    try:
        if not _marker_cells_match_shape(inp == _MARKER):
            return None
        out = main(inp)
        if np.array_equal(out, clean):
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
