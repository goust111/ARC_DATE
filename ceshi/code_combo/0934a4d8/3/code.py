# Combo variant of 0934a4d8 (index 3)
# combo_spec: D2/1+D5/2
# Name: 多分离标记+紧凑打包输出

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox


# Design analysis for the fused variant:
# - Original behavior had exactly one solid rectangle marker, so `_solid_rectangle_marker`
#   could collapse the whole marker color to one filled bbox and `main()` could crop that
#   single bbox directly. The fusion breaks both assumptions at once: there are now 2-3
#   disconnected marker rectangles and the output is not any one crop, but a packed strip.
# - Concrete example: if the clean grid hides a 4x3 patch at the upper right and a 4x5 patch
#   near the bottom left, the old task would output one crop. Here we recover both patches via
#   the same reflection orbits, then sort the rectangles by reading order and place the two
#   recovered patches into a 4x(3+1+5) strip with a 1-column separator of marker color 8.
# - Complexity is calibrated by keeping the markers independent but not noisy: all marker
#   components share one color and one common height, widths may differ, and the separator is
#   fixed. This makes the packing rule visible from a few examples without adding arbitrary
#   padding conventions or color-dependent ordering.
# - New implicit assumptions introduced by packing are explicit in the generator:
#   1) the marker color is still 8 and appears only in marker rectangles;
#   2) there are 2-3 disconnected solid rectangles of that color;
#   3) all marker rectangles have the same height so horizontal packing needs no extra padding;
#   4) rectangles are separated by at least one empty cell so they remain visually distinct.
# - Multi-marker placement can invalidate the old transpose fallback if another marker covers
#   the transposed source cell. The generator therefore validates the union of all markers:
#   whenever an LR/UD orbit is fully hidden by the union, the transpose cell must stay outside
#   every marker, and the fallback orbit is color-unified accordingly.
# - Visual observability: the input still reads as the original symmetric canvas with azure
#   occluders, but now there are several small occluders. The output visibly concatenates
#   several recovered crops with fixed azure dividers, so both the "multi-source" and
#   "packed-row" behaviors are inferable by inspection.


Rect = tuple[int, int, int, int]


def _marker_rectangles(g: np.ndarray, marker: int) -> list[Rect]:
    mask = g == marker
    n, m = map(int, g.shape)
    seen = np.zeros_like(mask, dtype=bool)
    rects: list[Rect] = []

    for r in range(n):
        for c in range(m):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < n and 0 <= nc < m and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comp_mask = np.zeros_like(mask, dtype=bool)
            for rr, cc in coords:
                comp_mask[rr, cc] = True
            b = bbox(comp_mask)
            assert b is not None
            r0, c0, r1, c1 = b
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            assert len(coords) == area
            assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
            rects.append((int(r0), int(c0), int(r1), int(c1)))

    assert 2 <= len(rects) <= 3
    heights = {r1 - r0 + 1 for r0, _c0, r1, _c1 in rects}
    assert len(heights) == 1
    return sorted(rects, key=lambda b: (b[0], b[1]))


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


def _recover_patch(g: np.ndarray, rect: Rect, lo: int, marker: int) -> np.ndarray:
    r0, c0, r1, c1 = rect
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo, marker)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


def _pack_row(patches: list[np.ndarray], sep: int) -> np.ndarray:
    assert patches
    heights = {int(p.shape[0]) for p in patches}
    assert len(heights) == 1
    h = int(patches[0].shape[0])
    widths = [int(p.shape[1]) for p in patches]
    out_w = int(sum(widths) + len(widths) - 1)
    out = np.full((h, out_w), int(sep), dtype=int)

    cursor = 0
    for patch in patches:
        w = int(patch.shape[1])
        out[:, cursor : cursor + w] = patch
        cursor += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover several small rectangular occlusions from a bilaterally symmetric grid and pack them into one strip.

    Input structure:
    - A square colored pattern with 2-3 disconnected solid rectangles of marker color 8 hiding true cells.
    - Outside the hidden cells, the clean pattern keeps the original task's LR and UD reflection symmetry on the
      lower/right strip, with the top/left border still available for transpose fallback.
    - The marker rectangles all share one height, but may have different widths and positions.

    Rule:
    - Detect the disconnected solid marker rectangles, infer the reflection-strip start exactly as in the base task,
      and recover each hidden rectangle cell-by-cell from its reflection orbit or, if needed, the transposed border cell.
    - Sort the recovered rectangles by reading order of their marker positions.
    - Pack the recovered patches left-to-right into a single row, inserting a 1-column separator of color 8 between
      neighboring patches.

    Key insight:
    Each azure block is just an independent window cut out of one shared symmetric image. Once every window is restored,
    the final answer is not any one crop but the ordered concatenation of all recovered windows.
    """
    g = np.asarray(grid, dtype=int)
    marker = 8
    rects = _marker_rectangles(g, marker)
    lo = _find_reflected_strip_start(g, marker)
    patches = [_recover_patch(g, rect, lo, marker) for rect in rects]
    return _pack_row(patches, marker)


# --------------- generator (o5x) ---------------

_N = 30
_LO = 2
_HI = _N - 1

_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

_MARKER_COUNT_CHOICES = (2, 3)
_MARKER_H_RANGE = (3, 6)
_MARKER_W_RANGE = (3, 7)
_MARKER_AREA_LO = 9
_MARKER_AREA_HI = 24
_MIN_GAP = 1
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_FALLBACK_PROB = 0.5
_WIDTHS_BY_HEIGHT = {
    h: [
        w
        for w in range(_MARKER_W_RANGE[0], _MARKER_W_RANGE[1] + 1)
        if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
    ]
    for h in range(_MARKER_H_RANGE[0], _MARKER_H_RANGE[1] + 1)
}
_FALLBACK_PLACEMENTS: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


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


def _rect_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    return not (
        ar0 + ah + _MIN_GAP <= br0
        or br0 + bh + _MIN_GAP <= ar0
        or ac0 + aw + _MIN_GAP <= bc0
        or bc0 + bw + _MIN_GAP <= ac0
    )


def _marker_mask(rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((_N, _N), dtype=bool)
    for r0, c0, h, w in rects:
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _fallback_cells(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]] | None:
    mask = _marker_mask(rects)
    cells: list[tuple[int, int]] = []
    for r0, c0, h, w in rects:
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if all(mask[rr, cc] for rr, cc in _orbit(r, c)):
                    if r >= _LO and c >= _LO:
                        return None
                    if mask[c, r]:
                        return None
                    cells.append((r, c))
    return cells


def _single_rect_fallback_placements(h: int, w: int) -> list[tuple[int, int]]:
    key = (h, w)
    if key not in _FALLBACK_PLACEMENTS:
        placements: list[tuple[int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                if _fallback_cells([(r0, c0, h, w)]):
                    placements.append((r0, c0))
        _FALLBACK_PLACEMENTS[key] = placements
    return _FALLBACK_PLACEMENTS[key]


def _sample_rectangles() -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]] | None:
    count = int(_RNG.choice(_MARKER_COUNT_CHOICES))
    h = int(_RNG.integers(_MARKER_H_RANGE[0], _MARKER_H_RANGE[1] + 1))
    widths = _WIDTHS_BY_HEIGHT[h]
    if not widths:
        return None

    require_fallback = bool(_RNG.random() < _FALLBACK_PROB)
    require_border = bool(_RNG.integers(2))
    for _ in range(200):
        rects: list[tuple[int, int, int, int]] = []
        sampled_widths = [int(_RNG.choice(widths)) for _ in range(count)]
        order = np.argsort(sampled_widths)[::-1]

        for step, idx in enumerate(order):
            w = int(sampled_widths[int(idx)])
            placed = False
            forced_positions: list[tuple[int, int]] = []
            if require_fallback and step == 0:
                forced_positions = _single_rect_fallback_placements(h, w).copy()
                _RNG.shuffle(forced_positions)
            for _ in range(150):
                if forced_positions:
                    r0, c0 = forced_positions.pop()
                elif require_border and not rects and bool(_RNG.integers(2)):
                    if bool(_RNG.integers(2)):
                        r0 = int(_RNG.integers(0, _LO))
                        c0 = int(_RNG.integers(0, _N - w + 1))
                    else:
                        r0 = int(_RNG.integers(0, _N - h + 1))
                        c0 = int(_RNG.integers(0, _LO))
                else:
                    r0 = int(_RNG.integers(0, _N - h + 1))
                    c0 = int(_RNG.integers(0, _N - w + 1))
                cand = (r0, c0, h, w)
                if any(_rect_too_close(cand, other) for other in rects):
                    continue
                rects.append(cand)
                placed = True
                break
            if not placed:
                break

        if len(rects) != count:
            continue
        if require_border and not any(r0 < _LO or c0 < _LO for r0, c0, _h, _w in rects):
            continue
        fallback = _fallback_cells(rects)
        if fallback is None:
            continue
        if require_fallback and not fallback:
            continue
        if not require_fallback and fallback:
            continue
        return sorted(rects, key=lambda x: (x[0], x[1])), fallback
    return None


def _pack_expected(clean: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    patches = [clean[r0 : r0 + h, c0 : c0 + w].copy() for r0, c0, h, w in rects]
    return _pack_row(patches, _MARKER)


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    sampled = _sample_rectangles()
    if sampled is None:
        return None
    rects, fallback = sampled

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = _pack_expected(clean, rects)
    inp = clean.copy()
    for r0, c0, h, w in rects:
        inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

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
    raise RuntimeError("failed to generate a valid fused 0934a4d8 input")
