# Combo variant of 0934a4d8 (index 12)
# combo_spec: D3/2+D6/0
# Name: 输出对称中心区域+可变lo

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox


# Design analysis for combo D3/2 + D6/0
# --------------------------------------
# a) Concrete behavior and trace:
#    The seed puzzle hides one solid rectangle and asks for the clean pixels under
#    that exact rectangle. This combo keeps the same bilateral LR+UD symmetry and
#    the same unique solid marker, but changes two linked pieces:
#    - `lo` is no longer fixed; it is chosen per example from a small set.
#    - the output is no longer the marker bbox; instead it is a fixed 4x6 window
#      centered on the inferred symmetry center.
#    Small trace with N=30 and lo=4:
#    - the symmetric strip is rows/cols 4..29, so the center is between 16 and 17
#    - the output window is rows 15..18 and cols 14..19
#    - a marker such as rows 13..18, cols 12..16 hides the left half of that
#      window, but not every reflection mate, so the hidden core cells can still
#      be restored from visible orbit partners
#    - the old task would output the full 6x5 marker content; the combo outputs
#      only the centered 4x6 core
# b) Complexity calibration:
#    `lo` only varies over {2,4}. That is enough to visibly shift the free top/left
#    border and therefore the symmetry center, but does not create too many parity
#    cases. The output size is fixed and even-sized so it stays exactly centered
#    for both lo values. The marker is forced to overlap the center window by a
#    noticeable amount while leaving at least one visible reflection mate for every
#    hidden target cell, so the new behavior is visible and still solvable.
# c) Implicit-assumption audit of the seed solver:
#    - "output bbox == marker bbox" no longer holds, so main must derive the core
#      bbox from inferred `lo` instead of using the marker bbox.
#    - "fixed lo=2" no longer holds, so the reflection axis must be inferred from
#      the visible mirror-consistent strip each time.
#    - "all queried cells may need transpose fallback" is weaker now: the centered
#      output window lies inside the symmetric strip, so generator construction
#      guarantees each hidden target cell has a visible LR/UD orbit mate. A border
#      transpose fallback branch is still kept for completeness.
# d) Placement legality:
#    The clean grid always satisfies the intended LR symmetry on columns [lo..N-1]
#    and UD symmetry on rows [lo..N-1]. The marker remains a single solid rectangle
#    fully inside the 30x30 canvas. Generator placement is accepted only when:
#    - it overlaps the centered 4x6 output window
#    - it hides several target cells, but not all of them
#    - every hidden target cell has at least one reflected orbit cell outside the
#      marker, so reconstruction is well-defined
#    - the inferred lo from the finished input equals the chosen lo
# e) Visual observability and cleanliness:
#    Across examples, the output shape is always the same 4x6 rectangle while the
#    marker size and placement vary. The only way to align that window is to first
#    notice where the bilateral symmetry begins, which directly exposes the lo
#    shift. Because the marker overlaps the core noticeably, the task does not
#    collapse into "just crop the visible center"; because the marker is still one
#    distinct solid color block, the scene stays clean and readable.


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))

    assert len(candidates) == 1
    _area, color, b = candidates[0]
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


def _reflection_orbit(r: int, c: int, lo: int, n: int) -> set[tuple[int, int]]:
    hi = n - 1
    rows = {int(r)}
    cols = {int(c)}
    if lo <= r <= hi:
        rows.add(lo + hi - int(r))
    if lo <= c <= hi:
        cols.add(lo + hi - int(c))
    return {(rr, cc) for rr in rows for cc in cols}


def _core_bbox(lo: int, n: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0 = lo + (n - lo - h) // 2
    c0 = lo + (n - lo - w) // 2
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _recover_cell(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    value = int(g[r, c])
    if value != marker:
        return value

    orbit = _reflection_orbit(r, c, lo, int(g.shape[0]))
    values = {int(g[rr, cc]) for rr, cc in orbit if int(g[rr, cc]) != marker}
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    fallback = int(g[c, r])
    assert fallback != marker
    return fallback


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the fixed center window of a bilaterally symmetric square after first inferring the symmetry offset.

    Input structure:
    - A 30x30 square grid contains one unique solid rectangular marker color that hides part of the clean pattern.
    - The clean pattern is left-right symmetric on columns [lo..N-1] and up-down symmetric on rows [lo..N-1].
    - The top/left border width `lo` is not fixed across examples, so the symmetry center moves with it.

    Rule:
    - Detect the solid rectangle only to learn the marker color, not to define the output area.
    - Infer `lo` as the smallest offset whose lower/right strip is mirror-consistent when marker pixels are ignored.
    - From that inferred `lo`, compute the fixed 4x6 rectangle centered on the bilateral symmetry center.
    - Output the clean values in that center rectangle: visible cells are copied directly, and any covered cells are restored from their horizontal/vertical reflection orbit.

    Key insight:
    The marker is now just an occluder and no longer tells you what to crop. The real target is a constant-size core anchored at the symmetry center, so the decisive step is to infer `lo`, which simultaneously reveals where to read and how to restore any hidden core pixels.
    """
    g = np.asarray(grid, dtype=int)
    marker, _marker_bbox = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)

    out_h, out_w = _OUT_SHAPE
    r0, c0, r1, c1 = _core_bbox(lo, int(g.shape[0]), out_h, out_w)
    out = np.empty((out_h, out_w), dtype=int)

    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            out[rr, cc] = _recover_cell(g, int(r), int(c), lo, marker)

    return out


# --------------- generator ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO_CHOICES = (2, 4)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: output window ---
_OUT_SHAPE = (4, 6)

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 5
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 25
_MARKER_AREA_HI = 48
_TARGET_HIDE_LO = 8

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}


def _contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _build_clean(lo: int) -> np.ndarray:
    """Build a clean NxN grid with bilateral LR+UD symmetry anchored by `lo`."""
    half = (_N - lo) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[lo:_N, lo:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(lo):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, lo : lo + half] = vals
        clean[r, lo + half : _N] = vals[::-1]

    for c in range(lo):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[lo : lo + half, c] = vals
        clean[lo + half : _N, c] = vals[::-1]

    clean[:lo, :lo] = _RNG.choice(_PALETTE, size=(lo, lo))
    return clean


def _valid_placements(lo: int, h: int, w: int) -> list[tuple[int, int, int]]:
    """Valid placements that hide part of the centered output without hiding any target orbit fully."""
    key = (lo, h, w)
    cached = _PLACEMENT_CACHE.get(key)
    if cached is not None:
        return cached

    out_h, out_w = _OUT_SHAPE
    core = _core_bbox(lo, _N, out_h, out_w)
    placements: list[tuple[int, int, int]] = []

    for r0 in range(_N - h + 1):
        for c0 in range(_N - w + 1):
            marker_box = (r0, c0, r0 + h - 1, c0 + w - 1)
            hidden = _overlap_area(marker_box, core)
            if hidden < _TARGET_HIDE_LO or hidden >= out_h * out_w:
                continue

            valid = True
            for r in range(core[0], core[2] + 1):
                for c in range(core[1], core[3] + 1):
                    if not _contains(marker_box, r, c):
                        continue
                    orbit = _reflection_orbit(r, c, lo, _N)
                    if all(_contains(marker_box, rr, cc) for rr, cc in orbit):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                placements.append((r0, c0, hidden))

    _PLACEMENT_CACHE[key] = placements
    return placements


def _gen_attempt() -> np.ndarray | None:
    lo = int(_RNG.choice(_LO_CHOICES))
    clean = _build_clean(lo)
    out_h, out_w = _OUT_SHAPE
    core = _core_bbox(lo, _N, out_h, out_w)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(lo, h, w)
    if not placements:
        return None

    r0, c0, hidden = placements[int(_RNG.integers(len(placements)))]
    expected = clean[core[0] : core[2] + 1, core[1] : core[3] + 1].copy()

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if hidden < _TARGET_HIDE_LO:
            return None
        if _find_reflected_strip_start(inp, _MARKER) != lo:
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
    raise RuntimeError("failed to generate a valid combo 0934a4d8 input")
