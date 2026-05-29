# Combo variant of 0934a4d8 (index 10)
# combo_spec: D2/2+D13/0
# Name: 十字形标记+两层嵌套恢复

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for combo D2/2 + D13/0
# a) Concrete behavior:
#    The source task hid one solid rectangle and returned that rectangle's clean contents. Here the hidden
#    geometry becomes a two-stage nested cross:
#    - stage 1 marker: a thick outer cross shell in color 8,
#    - stage 2 marker: a thinner inner cross in color 0, sharing the same center.
#    Example trace on a 9x9 outer / 5x5 inner pair:
#    - original: every hidden cell belonged to one rectangle, so one pass over that bbox was enough.
#    - fused: the visible input contains a plus-shaped azure shell with a smaller black plus inside it.
#      First recover the shell while treating both colors as hidden; this exposes clean witnesses needed
#      by the second pass. Then recover only the inner plus and crop its bounding box.
#    - difference: the occluded support is no longer a rectangle, and at least one inner-plus cell depends
#      on a witness that initially sits under the outer shell, so the second pass is genuinely downstream
#      of the first instead of being an independent shortcut.
# b) Complexity calibration:
#    Only one nested marker pair is injected, both crosses are axis-aligned and concentric, and the arm
#    thicknesses are fixed to a small set (outer thickness 3, inner thickness 1). This makes the new visual
#    rule obvious across 2-4 examples without exploding the hypothesis space. The shape change remains visible
#    because the shell is a real cross silhouette, not a barely perturbed rectangle.
# c) Implicit assumptions changed from the original:
#    - "the marker is the unique solid rectangle" is false: neither stage marker is a solid rectangle.
#    - "every marker cell lies inside one bbox that is fully hidden" is false: corners of the cross bbox stay visible.
#    - "output equals the full hidden rectangle" changes to "output equals the inner cross bbox, with only cross
#      cells needing stage-2 repair".
#    - multi-step means any direct use of raw color-8 cells as clean witnesses would be wrong, so the solver must
#      distinguish hidden colors per stage.
# d) Placement legality:
#    The generator builds the clean symmetric canvas first, then chooses nested cross masks that are fully in-bounds,
#    non-overlapping except by intended nesting, and legal for transpose fallback. Legality is checked on arbitrary
#    masks rather than rectangles:
#    - stage 1 treats the whole outer cross footprint (shell + inner cross) as hidden,
#    - stage 2 treats only the inner cross as hidden,
#    - if a reflection orbit is entirely hidden, the cell must lie in the top/left border and its transpose must
#      remain visible.
# e) Visual observability:
#    A human sees one large plus-shaped occluder surrounding a smaller plus of a different color. The output crop is
#    the inner plus bbox, so across examples the same "peel outer cross, then peel inner cross" story is directly
#    readable from the geometry. No invisible ordering by numeric color id is used: the colors simply mark layer roles.
# f) Visual cleanliness:
#    The clean background keeps the source task's bilateral texture. The only added structure is a single concentric
#    two-color cross pair, which is sparse enough to avoid clutter and rigid enough to be instantly separable from the
#    background pattern.
# g) Inferability of the change:
#    There is no arbitrary recoloring rule. The only new latent relation is stage order, and the generator enforces at
#    least one inner cell whose raw reflection witnesses are all trapped inside the outer cross. That makes the staged
#    dependency behaviorally visible instead of optional.
# 9) New solver assumptions guaranteed by the generator:
#    - color 8 appears exactly on the outer-shell cells of one thick cross,
#    - color 0 appears exactly on the cells of one thinner concentric inner cross,
#    - the union of colors {8,0} is itself a valid thick cross footprint,
#    - both stage recoveries have unique orbit / transpose answers,
#    - at least one inner cell would be mis-solved if stage 2 were attempted directly on the raw input.


_OUTER_MARKER = 8
_INNER_MARKER = 0


def _cross_mask(h: int, w: int, thickness: int) -> np.ndarray:
    assert h % 2 == 1 and w % 2 == 1 and thickness % 2 == 1
    assert 1 <= thickness < min(h, w)
    mask = np.zeros((h, w), dtype=bool)
    cr = h // 2
    cc = w // 2
    rad = thickness // 2
    mask[cr - rad : cr + rad + 1, :] = True
    mask[:, cc - rad : cc + rad + 1] = True
    return mask


def _paint_mask(shape: tuple[int, int], r0: int, c0: int, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = map(int, mask.shape)
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _cross_params(mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = map(int, b)
    crop = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
    h, w = map(int, crop.shape)

    full_rows = np.flatnonzero(np.all(crop, axis=1))
    full_cols = np.flatnonzero(np.all(crop, axis=0))
    assert len(full_rows) > 0
    assert len(full_rows) == len(full_cols)
    thickness = int(len(full_rows))
    assert np.array_equal(full_rows, np.arange(full_rows[0], full_rows[0] + thickness))
    assert np.array_equal(full_cols, np.arange(full_cols[0], full_cols[0] + thickness))
    assert int(full_rows[0]) == h // 2 - thickness // 2
    assert int(full_cols[0]) == w // 2 - thickness // 2

    expected = _cross_mask(h, w, thickness)
    assert np.array_equal(crop, expected)
    return (r0, c0, r1, c1), thickness


def _full_cross_from_bbox(
    shape: tuple[int, int], rect: tuple[int, int, int, int], thickness: int
) -> np.ndarray:
    r0, c0, r1, c1 = rect
    return _paint_mask(shape, int(r0), int(c0), _cross_mask(int(r1 - r0 + 1), int(c1 - c0 + 1), thickness))


def _nested_cross_markers(
    g: np.ndarray,
) -> tuple[tuple[int, int, int, int], int, tuple[int, int, int, int], int]:
    inner_mask = g == _INNER_MARKER
    inner_bbox, inner_t = _cross_params(inner_mask)

    outer_union = (g == _OUTER_MARKER) | inner_mask
    outer_bbox, outer_t = _cross_params(outer_union)

    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    assert or0 < ir0 <= ir1 < or1
    assert oc0 < ic0 <= ic1 < oc1
    assert or0 + or1 == ir0 + ir1
    assert oc0 + oc1 == ic0 + ic1
    assert outer_t > inner_t

    outer_full = _full_cross_from_bbox(g.shape, outer_bbox, outer_t)
    inner_full = _full_cross_from_bbox(g.shape, inner_bbox, inner_t)
    assert np.array_equal(inner_mask, inner_full)
    assert bool(np.all(inner_full <= outer_full))
    assert np.array_equal(g == _OUTER_MARKER, outer_full & ~inner_full)

    return outer_bbox, outer_t, inner_bbox, inner_t


def _find_reflected_strip_start(g: np.ndarray, hidden_colors: set[int]) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1
    hidden = np.isin(g, list(hidden_colors))

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_hidden = hidden[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        strip_hidden_reflected = strip_hidden[:, ::-1]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_hidden = hidden[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        strip_hidden_reflected = strip_hidden[::-1, :]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    raise AssertionError("no reflected strip found")


def _reflection_orbit_values(
    g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]
) -> set[int]:
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
        if value not in hidden_colors:
            values.add(value)

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def _recover_cell(g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]) -> int:
    values = _reflection_orbit_values(g, r, c, lo, hidden_colors)
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    value = int(g[c, r])
    assert value not in hidden_colors
    return value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel a nested pair of cross-shaped markers and return the innermost hidden patch.

    Input structure:
    - A square colored grid still follows the source task's bilateral LR+UD reflection law on the large lower/right strip.
    - The occluder is now a concentric two-layer cross: a thick outer shell in color 8 surrounds a thinner inner cross in color 0.
    - Cross corners inside each bbox can remain visible background pattern; only the cross footprints are actually hidden.

    Rule:
    - Detect the nested outer-shell/inner-core cross pair and infer the reflected strip start while ignoring both marker colors.
    - First restore every outer-shell cell from its reflection orbit, falling back to the transpose witness only when the full orbit is hidden.
    - Then restore the inner cross on the partially repaired grid and output the inner cross bbox with those repaired cells inserted.

    Key insight:
    The two marker colors are two occlusion layers over the same symmetric texture. Some inner-cross cells cannot trust the raw
    input because all of their non-inner orbit witnesses are still sitting under the outer shell, so the outer layer has to be
    solved first before the inner layer has valid evidence.
    """
    g = np.asarray(grid, dtype=int)
    outer_bbox, outer_t, inner_bbox, inner_t = _nested_cross_markers(g)
    lo = _find_reflected_strip_start(g, {_OUTER_MARKER, _INNER_MARKER})

    outer_full = _full_cross_from_bbox(g.shape, outer_bbox, outer_t)
    inner_full = _full_cross_from_bbox(g.shape, inner_bbox, inner_t)
    shell = outer_full & ~inner_full

    stage1 = g.copy()
    for r, c in np.argwhere(shell):
        stage1[int(r), int(c)] = _recover_cell(g, int(r), int(c), lo, {_OUTER_MARKER, _INNER_MARKER})

    ir0, ic0, ir1, ic1 = inner_bbox
    out = stage1[ir0 : ir1 + 1, ic0 : ic1 + 1].copy()
    for r, c in np.argwhere(inner_full):
        out[int(r - ir0), int(c - ic0)] = _recover_cell(stage1, int(r), int(c), lo, {_INNER_MARKER})

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# --- CFG: cross sizes ---
_OUTER_THICKNESS = 3
_INNER_THICKNESS = 1
_OUTER_SIDES = (7, 9, 11)
_INNER_SIDES = (5, 7, 9)

# --- CFG: generation retries ---
_MAX_RETRIES = 600

_RNG = np.random.default_rng()
_OUTER_SIZE_PAIRS = [(h, w) for h in _OUTER_SIDES for w in _OUTER_SIDES]
_INNER_SIZE_PAIRS = [(h, w) for h in _INNER_SIDES for w in _INNER_SIDES]
_PLACEMENT_CACHE: dict[
    tuple[int, int, int, int],
    list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]],
] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - int(r))
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - int(c))
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(hidden: np.ndarray) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(hidden):
        rr = int(r)
        cc = int(c)
        if all(bool(hidden[or_r, or_c]) for or_r, or_c in _orbit(rr, cc)):
            if rr >= _LO and cc >= _LO:
                return None
            if bool(hidden[cc, rr]):
                return None
            cells.append((rr, cc))
    return cells


def _dependent_inner_cells(outer_full: np.ndarray, inner_full: np.ndarray) -> list[tuple[int, int]]:
    dependent: list[tuple[int, int]] = []
    for r, c in np.argwhere(inner_full):
        rr = int(r)
        cc = int(c)
        outside_inner = [(or_r, or_c) for or_r, or_c in _orbit(rr, cc) if not bool(inner_full[or_r, or_c])]
        if outside_inner and all(bool(outer_full[or_r, or_c]) for or_r, or_c in outside_inner):
            dependent.append((rr, cc))
    return dependent


def _nested_placements(
    outer_h: int, outer_w: int, inner_h: int, inner_w: int
) -> list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]:
    key = (outer_h, outer_w, inner_h, inner_w)
    if key not in _PLACEMENT_CACHE:
        placements: list[
            tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]
        ] = []
        if inner_h > outer_h - 2 or inner_w > outer_w - 2:
            _PLACEMENT_CACHE[key] = placements
            return placements

        outer_local = _cross_mask(outer_h, outer_w, _OUTER_THICKNESS)
        inner_local = _cross_mask(inner_h, inner_w, _INNER_THICKNESS)
        dr = (outer_h - inner_h) // 2
        dc = (outer_w - inner_w) // 2

        for outer_r0 in range(_N - outer_h + 1):
            for outer_c0 in range(_N - outer_w + 1):
                inner_r0 = outer_r0 + dr
                inner_c0 = outer_c0 + dc

                outer_full = _paint_mask((_N, _N), outer_r0, outer_c0, outer_local)
                inner_full = _paint_mask((_N, _N), inner_r0, inner_c0, inner_local)
                if not bool(np.all(inner_full <= outer_full)):
                    continue

                outer_fallback = _fallback_cells(outer_full)
                if outer_fallback is None:
                    continue
                inner_fallback = _fallback_cells(inner_full)
                if inner_fallback is None:
                    continue

                dependent = _dependent_inner_cells(outer_full, inner_full)
                if dependent:
                    placements.append((outer_r0, outer_c0, outer_fallback, inner_fallback, dependent))

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


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()

    outer_h, outer_w = _OUTER_SIZE_PAIRS[int(_RNG.integers(len(_OUTER_SIZE_PAIRS)))]
    candidate_inner = [(h, w) for h, w in _INNER_SIZE_PAIRS if h <= outer_h - 2 and w <= outer_w - 2]
    if not candidate_inner:
        return None
    inner_h, inner_w = candidate_inner[int(_RNG.integers(len(candidate_inner)))]

    placements = _nested_placements(outer_h, outer_w, inner_h, inner_w)
    if not placements:
        return None
    outer_r0, outer_c0, outer_fallback, inner_fallback, _dependent = placements[
        int(_RNG.integers(len(placements)))
    ]

    inner_r0 = outer_r0 + (outer_h - inner_h) // 2
    inner_c0 = outer_c0 + (outer_w - inner_w) // 2
    outer_full = _paint_mask((_N, _N), outer_r0, outer_c0, _cross_mask(outer_h, outer_w, _OUTER_THICKNESS))
    inner_full = _paint_mask((_N, _N), inner_r0, inner_c0, _cross_mask(inner_h, inner_w, _INNER_THICKNESS))

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in outer_fallback]
    groups += [_orbit(r, c) | _orbit(c, r) for r, c in inner_fallback]
    _unify_groups(clean, groups)

    expected = clean[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w].copy()

    inp = clean.copy()
    inp[outer_full & ~inner_full] = _OUTER_MARKER
    inp[inner_full] = _INNER_MARKER

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
    raise RuntimeError("failed to generate fused 0934a4d8 D2/2+D13/0 input")
