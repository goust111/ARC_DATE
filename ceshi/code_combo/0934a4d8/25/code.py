# Combo variant of 0934a4d8 (index 25)
# combo_spec: D1/0+D2/0+D14/1
# Name: 单轴对称+L形标记+跨轴放置

from __future__ import annotations

from collections import deque

import numpy as np


# Design analysis for combo D1/0 + D2/0 + D14/1
# a) Concrete behavior and trace:
#    - Source task: one solid rectangle of color 8 hides a full rectangular patch.
#      The clean 30x30 board is symmetric in both LR and UD on the [lo..N-1] strip,
#      so each hidden pixel can read a value from a 4-cell reflection orbit, with a
#      transpose fallback only for rare fully hidden border orbits.
#    - Fused task: the clean board keeps only one reflection direction, either LR or
#      UD. The marker is no longer a full rectangle but one thick L polyomino of
#      color 8, and that L must cross the active symmetry axis.
#    - Example in the LR case: suppose the vertical mirror axis is between columns
#      15 and 16 and an L spans both sides of that axis. A marker cell on the long
#      upright stroke may mirror to an exposed cell and can still be solved directly.
#      But a marker cell on the horizontal arm that covers both axis columns mirrors
#      to another hidden L cell, so the direct LR witness disappears. For those cells
#      the generator pre-synchronizes a visible transpose witness at (c, r), and the
#      solver uses that only when the mirror partner is also marked.
#    - Difference from the source: the solver can no longer crop "all cells in bbox"
#      as hidden, can no longer assume bilateral orbits, and must mix two recovery
#      modes inside one irregular marker crop: direct single-axis mirroring for some
#      L cells, transpose fallback for others.
# b) Complexity calibration:
#    - The symmetry choice is binary (LR or UD), and the marker family is limited to
#      rotated thick L shapes with small thickness set {2,3}. This keeps the visual
#      rule learnable from a few examples without collapsing back to rectangles.
#    - The marker always crosses the active axis and every accepted sample has both
#      directly recoverable cells and transpose-fallback cells. That makes the fused
#      behavior observable instead of an optional corner case.
#    - Output remains a crop around the marker bbox, but because the marker is sparse,
#      visible non-marker cells inside that bbox pass through unchanged. This makes
#      the non-rectangular occluder obvious in both input and output.
# c) Implicit assumption audit of the original main():
#    - "marker = unique filled rectangle" is false. Detection must switch to "one
#      connected color-8 component whose normalized silhouette is an L".
#    - "the same lo works for both LR and UD" is false. We must infer exactly one
#      consistent (axis, lo) pair and reject ambiguous generation cases.
#    - "every bbox cell is hidden" is false. The output crop should start from the
#      visible bbox content and overwrite only marker cells.
#    - "orbit search may use both reflections" is false. Recovery now has only one
#      mirror partner per marker cell, plus transpose fallback when that partner is
#      also marked.
#    - "transpose fallback only happens near the free border" is false. Because the
#      L crosses the axis, doubly hidden pairs can appear in the middle; generator
#      legality must therefore guarantee a visible transpose witness anywhere.
# d) Placement legality enforced by the generator:
#    - The L mask is a single connected polyomino, stays in bounds, and is the only
#      use of color 8.
#    - For LR samples the whole marker lies in columns [lo..N-1]; for UD samples it
#      lies in rows [lo..N-1]. This ensures every marked cell has a defined mirror.
#    - The occupied cells truly cross the chosen axis, at least one marked cell has
#      a visible mirror partner, and at least one marked cell has its mirror partner
#      also inside the marker.
#    - Every doubly hidden marked cell has transpose witness (c, r) outside the
#      marker. Generator-side color unification extends through the active symmetry
#      orbit of that witness so the clean board remains valid.
# e) Visual observability:
#    - Across examples, humans can see exactly one global mirror direction and one
#      conspicuous L-shaped azure occluder straddling that mirror axis.
#    - Inside the marker bbox, some cells remain visible background while the L cells
#      are hidden, so the output visibly corresponds to "repair this L crop" rather
#      than "extract a full filled rectangle".
#    - Because every sample mixes direct-mirror and fallback-repaired L cells, the
#      axis-crossing interaction is behaviorally supported instead of being latent.
# f) Visual cleanliness:
#    - The scene still has a single main object over one regular mirrored texture.
#      No extra overlays, legends, or competing shapes are introduced.
#    - Thick L shapes are sparse and rigid, so they remain easy to distinguish from
#      the colorful background and from their own bbox interior.
# g) Color-change inference:
#    - There is no recoloring rule. Hidden values are copied from geometric witnesses
#      only: either the active mirror partner or a visible transpose witness.
# 9) New solver assumptions guaranteed by the generator:
#    - Color 8 appears on exactly one connected L component.
#    - Exactly one reflection mode (axis, lo) is consistent after ignoring marker
#      cells, and the inactive orthogonal direction has at least one visible mismatch.
#    - Any marker cell whose mirror partner is also marked has a visible transpose
#      witness with the same clean value.
#    - Every accepted sample contains both direct-copy cells and fallback cells, so
#      the fused behavior is active on the generated distribution.


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


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _classify_l_shape(mask: np.ndarray) -> tuple[int, int] | None:
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
                return rot, thickness
    return None


def _find_l_marker(g: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    marker_mask = g == _MARKER
    assert bool(marker_mask.any())
    comps = _components(marker_mask)
    assert len(comps) == 1
    coords = comps[0]
    norm = _normalize_component(coords)
    assert _classify_l_shape(norm) is not None
    return marker_mask, _bbox_from_coords(coords)


def _strip_matches_axis(g: np.ndarray, hidden: np.ndarray, lo: int, axis: str) -> bool:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    if axis == "lr":
        strip = g[:, lo : hi + 1]
        strip_hidden = hidden[:, lo : hi + 1]
        reflected = strip[:, ::-1]
        reflected_hidden = strip_hidden[:, ::-1]
    else:
        strip = g[lo : hi + 1, :]
        strip_hidden = hidden[lo : hi + 1, :]
        reflected = strip[::-1, :]
        reflected_hidden = strip_hidden[::-1, :]

    known = (~strip_hidden) & (~reflected_hidden)
    return not bool(np.any((strip != reflected) & known))


def _find_single_reflection(g: np.ndarray, hidden: np.ndarray) -> tuple[str, int]:
    n, m = map(int, g.shape)
    assert n == m
    candidates: list[tuple[str, int]] = []
    for lo in range(n // 2 + 1):
        for axis in ("lr", "ud"):
            if _strip_matches_axis(g, hidden, lo, axis):
                candidates.append((axis, lo))
    assert len(candidates) == 1, f"expected one reflection mode, got {candidates}"
    return candidates[0]


def _mirror_partner(r: int, c: int, lo: int, axis: str, n: int) -> tuple[int, int] | None:
    hi = n - 1
    if axis == "lr":
        if c < lo:
            return None
        return int(r), int(lo + hi - c)
    if r < lo:
        return None
    return int(lo + hi - r), int(c)


def _axis_pair(lo: int, n: int) -> tuple[int, int]:
    hi = n - 1
    left = (lo + hi) // 2
    return left, left + 1


def _crosses_axis_cells(cells: set[tuple[int, int]], axis: str, lo: int, n: int) -> bool:
    axis_lo, axis_hi = _axis_pair(lo, n)
    if axis == "lr":
        cols = [c for _r, c in cells]
        return min(cols) <= axis_lo and max(cols) >= axis_hi
    rows = [r for r, _c in cells]
    return min(rows) <= axis_lo and max(rows) >= axis_hi


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the L-shaped occluded crop from a board that preserves exactly one reflection axis.

    Input structure:
    - A square colored grid contains one connected L-shaped marker component in color 8 rather than a filled rectangle.
    - The clean board is mirror-consistent on the centered strip [lo..N-1] in exactly one direction: either left-right for all rows or up-down for all columns.
    - The L marker straddles that active symmetry axis, so some marked cells still have visible mirror mates while others mirror only to different marked cells.

    Rule:
    - Detect the unique color-8 L component and use its bounding box as the output crop; cells inside that bbox but outside the L are already visible and should be copied through.
    - Infer the only valid `(axis, lo)` reflection mode after ignoring marker cells.
    - For each marked cell, copy the value from its mirror partner across the active axis when that partner is visible; otherwise read the synchronized transpose witness at `(c, r)`.

    Key insight:
    The fused challenge is not just "find a different shape" or "find a different axis" independently. The L crosses the only mirror axis, which creates a mix of easy cells and doubly hidden mirror pairs inside one crop, so the solver must identify the active axis, track the exact L footprint, and switch to transpose only where direct mirroring is blocked by the shape itself.
    """
    g = np.asarray(grid, dtype=int)
    marker_mask, (r0, c0, r1, c1) = _find_l_marker(g)
    axis, lo = _find_single_reflection(g, marker_mask)
    cells = {(int(r), int(c)) for r, c in np.argwhere(marker_mask)}
    assert _crosses_axis_cells(cells, axis, lo, int(g.shape[0]))

    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    n = int(g.shape[0])
    for r, c in np.argwhere(marker_mask):
        rr = int(r)
        cc = int(c)
        partner = _mirror_partner(rr, cc, lo, axis, n)
        assert partner is not None
        pr, pc = partner
        if not marker_mask[pr, pc]:
            value = int(g[pr, pc])
            assert value != _MARKER
        else:
            value = int(g[cc, rr])
            assert not marker_mask[cc, rr]
            assert value != _MARKER
        out[rr - r0, cc - c0] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXES = ("lr", "ud")
_ROTATIONS = (0, 1, 2, 3)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# --- CFG: marker L geometry ---
_THICKNESSES = (2, 3)
_BBOX_DIM_LO = 4
_BBOX_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_PLACEMENT_CACHE: dict[tuple[str, tuple[tuple[int, int], bytes]], list[tuple[int, int, list[tuple[int, int]], int]]] = {}
_GENERATED_COUNT = 0


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _build_marker_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for thickness in _THICKNESSES:
        for h in range(_BBOX_DIM_LO, _BBOX_DIM_HI + 1):
            for w in range(_BBOX_DIM_LO, _BBOX_DIM_HI + 1):
                if h <= thickness or w <= thickness:
                    continue
                base = np.zeros((h, w), dtype=bool)
                base[:, :thickness] = True
                base[:thickness, :] = True
                area = int(base.sum())
                if not (_MARKER_AREA_LO <= area <= _MARKER_AREA_HI):
                    continue
                for rot in range(4):
                    rotated = np.rot90(base, rot).copy()
                    key = _shape_key(rotated)
                    if key not in seen:
                        seen.add(key)
                        shapes.append(rotated)
    assert shapes
    return shapes


_MARKER_SHAPES = _build_marker_shapes()
_MARKER_SHAPES_BY_ROT: dict[int, list[np.ndarray]] = {rot: [] for rot in _ROTATIONS}
for _shape in _MARKER_SHAPES:
    _rot, _thickness = _classify_l_shape(_shape)
    _MARKER_SHAPES_BY_ROT[_rot].append(_shape)
assert all(_MARKER_SHAPES_BY_ROT.values())


def _translated_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _axis_group(axis: str, r: int, c: int) -> set[tuple[int, int]]:
    group = {(int(r), int(c))}
    partner = _mirror_partner(int(r), int(c), _LO, axis, _N)
    if partner is not None:
        group.add(partner)
    return group


def _build_clean(axis: str) -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    if axis == "lr":
        left = _RNG.choice(_PALETTE, size=(_N, half))
        clean[:, _LO : _LO + half] = left
        clean[:, _LO + half : _N] = left[:, ::-1]
        clean[:, :_LO] = _RNG.choice(_PALETTE, size=(_N, _LO))
    else:
        top = _RNG.choice(_PALETTE, size=(half, _N))
        clean[_LO : _LO + half, :] = top
        clean[_LO + half : _N, :] = top[::-1, :]
        clean[:_LO, :] = _RNG.choice(_PALETTE, size=(_LO, _N))

    return clean


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = find(a)
        rb = find(b)
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


def _placement_info(
    axis: str, mask: np.ndarray, r0: int, c0: int
) -> tuple[list[tuple[int, int]], int] | None:
    cells = _translated_cells(mask, r0, c0)
    if not _crosses_axis_cells(cells, axis, _LO, _N):
        return None

    fallback: list[tuple[int, int]] = []
    direct_count = 0
    for r, c in cells:
        partner = _mirror_partner(r, c, _LO, axis, _N)
        if partner is None:
            return None
        if partner in cells:
            if (c, r) in cells:
                return None
            fallback.append((r, c))
        else:
            direct_count += 1

    if direct_count == 0 or not fallback:
        return None
    return fallback, direct_count


def _valid_placements(axis: str, mask: np.ndarray) -> list[tuple[int, int, list[tuple[int, int]], int]]:
    key = (axis, _shape_key(mask))
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]], int]] = []
        h, w = map(int, mask.shape)
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                if axis == "lr" and c0 < _LO:
                    continue
                if axis == "ud" and r0 < _LO:
                    continue
                info = _placement_info(axis, mask, r0, c0)
                if info is None:
                    continue
                fallback, direct_count = info
                placements.append((r0, c0, fallback, direct_count))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _orthogonal_mismatch_count(g: np.ndarray, hidden: np.ndarray, lo: int, axis: str) -> int:
    other = "ud" if axis == "lr" else "lr"
    n = int(g.shape[0])
    hi = n - 1

    if other == "lr":
        strip = g[:, lo : hi + 1]
        strip_hidden = hidden[:, lo : hi + 1]
        reflected = strip[:, ::-1]
        reflected_hidden = strip_hidden[:, ::-1]
    else:
        strip = g[lo : hi + 1, :]
        strip_hidden = hidden[lo : hi + 1, :]
        reflected = strip[::-1, :]
        reflected_hidden = strip_hidden[::-1, :]

    known = (~strip_hidden) & (~reflected_hidden)
    return int(np.count_nonzero((strip != reflected) & known))


def _expected_output(clean: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    return clean[r0 : r0 + h, c0 : c0 + w].copy()


def _gen_attempt(axis: str, rot: int) -> np.ndarray | None:
    clean = _build_clean(axis)
    shapes = _MARKER_SHAPES_BY_ROT[rot]
    mask = shapes[int(_RNG.integers(len(shapes)))]
    placements = _valid_placements(axis, mask)
    if not placements:
        return None

    r0, c0, fallback, _direct_count = placements[int(_RNG.integers(len(placements)))]

    groups = []
    for r, c in fallback:
        groups.append(_axis_group(axis, r, c) | _axis_group(axis, c, r))
    _unify_groups(clean, groups)

    expected = _expected_output(clean, mask, r0, c0)
    inp = clean.copy()
    translated = _translated_cells(mask, r0, c0)
    for r, c in translated:
        inp[r, c] = _MARKER

    try:
        marker_mask, bbox = _find_l_marker(inp)
        found_axis, found_lo = _find_single_reflection(inp, marker_mask)
        if (found_axis, found_lo) != (axis, _LO):
            return None
        if _orthogonal_mismatch_count(inp, marker_mask, found_lo, found_axis) == 0:
            return None
        if np.array_equal(main(inp), expected):
            br0, bc0, br1, bc1 = bbox
            assert (br1 - br0 + 1, bc1 - bc0 + 1) == mask.shape
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    global _GENERATED_COUNT
    combo_idx = _GENERATED_COUNT % (len(_AXES) * len(_ROTATIONS))
    axis = _AXES[combo_idx % len(_AXES)]
    rot = _ROTATIONS[(combo_idx // len(_AXES)) % len(_ROTATIONS)]
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(axis, rot)
        if grid is not None:
            _GENERATED_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo-25 input")
