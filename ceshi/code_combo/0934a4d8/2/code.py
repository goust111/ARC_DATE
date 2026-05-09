# Combo variant of 0934a4d8 (index 2)
# combo_spec: D1/0+D14/1
# Name: 单轴对称+标记跨轴放置

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
from arc_utils import bbox

# Design analysis
# a) Concrete behavior and trace:
#    - Original 0934a4d8 used a bilateral LR+UD orbit, so a hidden cell could often
#      be recovered from another visible orbit member even when one mirror partner was
#      hidden. This combo removes that redundancy: each cell only has one true mirror.
#    - Because the marker must cross the unique symmetry axis, some mirror pairs are
#      swallowed on both sides. Example with N=30, lo=2, LR mode, marker columns
#      14..17: cells at c=15 and c=16 mirror each other, and c=14/c=17 form a second
#      pair. Both pairs lie inside the marker, so direct reflection is insufficient.
#    - The fused rule therefore stores backup values on the free boundary. In LR mode,
#      each row uses left-border columns 0 and 1 as slots for the center pair and the
#      next outer pair. UD mode is the row/column transpose of that idea.
# b) Complexity calibration:
#    - The board still looks like a large colorful mirror texture with one solid
#      rectangle removed, so it remains recognizably close to the source task.
#    - To make the new behavior visible rather than degenerate, the generator forces
#      the marker to straddle the detected axis and requires at least one cell to need
#      boundary fallback. The orthogonal axis is also checked to be genuinely broken.
#    - The backup system is intentionally low-arity: the cross-axis span is restricted
#      to 3 or 4, so there are at most two fully hidden mirror pairs per row/column.
# c) Hidden-assumption audit of the original main():
#    - The old `_find_reflected_strip_start` assumed both LR and UD were valid. Here we
#      must detect exactly one `(axis, lo)` candidate instead.
#    - The old per-cell logic assumed a visible reflected orbit member always existed or
#      that transpose could rescue border cases. Neither is true after combining D1 and
#      D14, so the solver now has an explicit "mirror hidden -> boundary slot" branch.
#    - The new fallback branch assumes the hidden pair index is at most `lo-1`; the
#      generator enforces that by limiting the marker's cross-axis thickness to 3 or 4.
# d) Placement legality:
#    - Every marker is an in-bounds solid rectangle. In LR mode it must cross the
#      vertical mirror axis; in UD mode it must cross the horizontal mirror axis.
#    - The backup cells live on free border columns or rows, so they never interfere
#      with the mirrored strip and are guaranteed to stay outside the marker.
# e) Visual observability:
#    - Across examples, some inputs are LR-only and some are UD-only, so the solver has
#      to infer the surviving symmetry direction from visible structure.
#    - The marker always straddles that axis, and the boundary backups repeat in the
#      same rows or columns as the covered region, making the extra cue visually legible.
# f) Visual cleanliness:
#    - No new shapes are introduced. The scene is still just a mirrored texture plus a
#      single solid marker; only the free border carries backup colors.
#    - Backup slots use existing palette colors and sit on already unconstrained border
#      cells, avoiding clutter inside the main patterned strip.
# g) Color-change inference:
#    - There is no remapping. Every recovered pixel copies either its visible mirror mate
#      or the equal-valued boundary backup that was written from that hidden mirror pair.
# h) New main() assumption audit:
#    - `_find_single_reflection` expects exactly one valid axis/offset pair; generation
#      rejects accidental extra candidates and accidental bilateral samples.
#    - The mirror-hidden branch expects the marker to cross the detected axis and to use
#      at most two backup slots; generation enforces both properties.
#    - Boundary slot reads expect non-marker values already synchronized with the hidden
#      pair; generation writes those slots after constructing the clean symmetric board.


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


def _strip_matches_axis(g: np.ndarray, marker: int, lo: int, axis: str) -> bool:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    if axis == "lr":
        strip = g[:, lo : hi + 1]
        reflected = strip[:, ::-1]
    else:
        strip = g[lo : hi + 1, :]
        reflected = strip[::-1, :]

    known = (strip != marker) & (reflected != marker)
    return not bool(np.any((strip != reflected) & known))


def _all_reflection_candidates(g: np.ndarray, marker: int) -> list[tuple[str, int]]:
    n, m = map(int, g.shape)
    assert n == m

    candidates: list[tuple[str, int]] = []
    for lo in range(n // 2 + 1):
        for axis in ("lr", "ud"):
            if _strip_matches_axis(g, marker, lo, axis):
                candidates.append((axis, lo))
    return candidates


def _find_single_reflection(g: np.ndarray, marker: int) -> tuple[str, int]:
    candidates = _all_reflection_candidates(g, marker)
    assert len(candidates) == 1, f"expected one reflection mode, got {candidates}"
    return candidates[0]


def _axis_pair(lo: int, n: int) -> tuple[int, int]:
    hi = n - 1
    left = (lo + hi) // 2
    return int(left), int(left + 1)


def _crosses_axis(start: int, end: int, lo: int, n: int) -> bool:
    axis_lo, axis_hi = _axis_pair(lo, n)
    return int(start) <= axis_lo and int(end) >= axis_hi


def _mirror_index(idx: int, lo: int, n: int) -> int:
    return int(lo + (n - 1) - idx)


def _mirror_partner(r: int, c: int, lo: int, axis: str, n: int) -> tuple[int, int] | None:
    hi = n - 1
    if axis == "lr":
        if not (lo <= c <= hi):
            return None
        return int(r), _mirror_index(c, lo, n)
    if not (lo <= r <= hi):
        return None
    return _mirror_index(r, lo, n), int(c)


def _in_marker(r: int, c: int, marker_box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = marker_box
    return r0 <= int(r) <= r1 and c0 <= int(c) <= c1


def _hidden_slot(
    r: int,
    c: int,
    marker_box: tuple[int, int, int, int],
    lo: int,
    axis: str,
    n: int,
) -> int | None:
    partner = _mirror_partner(r, c, lo, axis, n)
    assert partner is not None
    pr, pc = partner
    if not _in_marker(pr, pc, marker_box):
        return None

    axis_lo, _axis_hi = _axis_pair(lo, n)
    if axis == "lr":
        slot = axis_lo - min(c, pc)
    else:
        slot = axis_lo - min(r, pr)
    assert 0 <= slot < lo
    return int(slot)


def _orthogonal_mismatch_count(g: np.ndarray, marker: int, lo: int, axis: str) -> int:
    other = "ud" if axis == "lr" else "lr"
    n = int(g.shape[0])
    hi = n - 1

    if other == "lr":
        strip = g[:, lo : hi + 1]
        reflected = strip[:, ::-1]
    else:
        strip = g[lo : hi + 1, :]
        reflected = strip[::-1, :]

    known = (strip != marker) & (reflected != marker)
    return int(np.count_nonzero((strip != reflected) & known))


def _fallback_cell_count(
    marker_box: tuple[int, int, int, int],
    axis: str,
    lo: int,
    n: int,
) -> int:
    r0, c0, r1, c1 = marker_box
    total = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _hidden_slot(r, c, marker_box, lo, axis, n) is not None:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the marker-sized patch from a grid that keeps exactly one mirror axis and stores cross-axis backups on the free border.

    Input structure:
    - A square colored grid contains one solid rectangular marker color hiding the answer patch.
    - Away from the marker, the clean board is mirror-symmetric on the centered strip [lo..N-1] in exactly one direction:
      either left-right for all rows or up-down for all columns.
    - The marker always straddles that unique symmetry axis, so some hidden cells have their only mirror partner hidden too; the
      corresponding values are duplicated onto visible border slots outside the mirrored strip.

    Rule:
    - Find the unique filled rectangle color and use its bounding box as the output window.
    - Infer the only valid reflection mode `(axis, lo)` while ignoring marker cells.
    - For each hidden cell, read its mirror partner if that partner is visible; otherwise read the deterministic backup slot on the
      free border (left border for LR, top border for UD).

    Key insight:
    With only one symmetry axis, an axis-straddling marker can hide both members of a mirror pair, so solving requires two steps:
    first detect which symmetry survives, then switch from direct mirroring to border-backed recovery exactly where the marker covers
    both sides of that pair.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_box = _solid_rectangle_marker(g)
    r0, c0, r1, c1 = marker_box
    axis, lo = _find_single_reflection(g, marker)
    n = int(g.shape[0])

    if axis == "lr":
        assert _crosses_axis(c0, c1, lo, n)
    else:
        assert _crosses_axis(r0, r1, lo, n)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            partner = _mirror_partner(r, c, lo, axis, n)
            assert partner is not None
            pr, pc = partner
            if not _in_marker(pr, pc, marker_box):
                value = int(g[pr, pc])
                assert value != marker
                out[rr, cc] = value
                continue

            slot = _hidden_slot(r, c, marker_box, lo, axis, n)
            assert slot is not None
            value = int(g[r, slot] if axis == "lr" else g[slot, c])
            assert value != marker
            out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXES = ("lr", "ud")

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36
_CROSS_SPANS = (3, 4)

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_PROFILE_CYCLE: tuple[tuple[str, int], ...] = (("lr", 1), ("ud", 1), ("lr", 2), ("ud", 2))
_PROFILE_INDEX = 0
_SIZE_PAIRS_BY_AXIS: dict[str, list[tuple[int, int]]] = {
    "lr": [
        (h, w)
        for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
        for w in _CROSS_SPANS
        if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
    ],
    "ud": [
        (h, w)
        for h in _CROSS_SPANS
        for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
        if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
    ],
}
_PLACEMENT_CACHE: dict[tuple[str, int, int], list[tuple[int, int]]] = {}


def _valid_placements(axis: str, h: int, w: int) -> list[tuple[int, int]]:
    key = (axis, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                r1 = r0 + h - 1
                c1 = c0 + w - 1
                if axis == "lr":
                    if not _crosses_axis(c0, c1, _LO, _N):
                        continue
                else:
                    if not _crosses_axis(r0, r1, _LO, _N):
                        continue
                placements.append((r0, c0))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _hidden_slots_used(
    marker_box: tuple[int, int, int, int],
    axis: str,
    lo: int,
    n: int,
) -> tuple[int, ...]:
    r0, c0, r1, c1 = marker_box
    slots: set[int] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            slot = _hidden_slot(r, c, marker_box, lo, axis, n)
            if slot is not None:
                slots.add(int(slot))
    return tuple(sorted(slots))


def _next_profile() -> tuple[str, int]:
    global _PROFILE_INDEX
    profile = _PROFILE_CYCLE[_PROFILE_INDEX % len(_PROFILE_CYCLE)]
    _PROFILE_INDEX += 1
    return profile


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


def _write_boundary_backups(clean: np.ndarray, axis: str, r0: int, c0: int, h: int, w: int) -> None:
    axis_lo, _axis_hi = _axis_pair(_LO, _N)
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    if axis == "lr":
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mc = _mirror_index(c, _LO, _N)
                if c > mc or not (c0 <= mc <= c1):
                    continue
                slot = axis_lo - min(c, mc)
                assert 0 <= slot < _LO
                clean[r, slot] = int(clean[r, c])
    else:
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                mr = _mirror_index(r, _LO, _N)
                if r > mr or not (r0 <= mr <= r1):
                    continue
                slot = axis_lo - min(r, mr)
                assert 0 <= slot < _LO
                clean[slot, c] = int(clean[r, c])


def _gen_attempt(axis: str, hidden_slot_count: int) -> np.ndarray | None:
    clean = _build_clean(axis)
    sizes = list(_SIZE_PAIRS_BY_AXIS[axis])
    _RNG.shuffle(sizes)

    r0 = c0 = h = w = -1
    marker_box: tuple[int, int, int, int] | None = None
    for hh, ww in sizes:
        placements = list(_valid_placements(axis, hh, ww))
        _RNG.shuffle(placements)
        for rr, cc in placements:
            box = (rr, cc, rr + hh - 1, cc + ww - 1)
            if len(_hidden_slots_used(box, axis, _LO, _N)) != hidden_slot_count:
                continue
            r0, c0, h, w = rr, cc, hh, ww
            marker_box = box
            break
        if marker_box is not None:
            break
    if marker_box is None:
        return None

    _write_boundary_backups(clean, axis, r0, c0, h, w)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        found_axis, found_lo = _find_single_reflection(inp, _MARKER)
        if (found_axis, found_lo) != (axis, _LO):
            return None
        if _orthogonal_mismatch_count(inp, _MARKER, found_lo, found_axis) == 0:
            return None
        if _fallback_cell_count(marker_box, axis, _LO, _N) == 0:
            return None
        if len(_hidden_slots_used(marker_box, axis, _LO, _N)) != hidden_slot_count:
            return None
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    axis, hidden_slot_count = _next_profile()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(axis, hidden_slot_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
