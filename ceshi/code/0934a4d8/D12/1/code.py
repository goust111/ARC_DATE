# Variant of 0934a4d8
# Dimension: D12 — 干扰与噪声 (meta)
# Change: 无噪声，可见区域完全满足对称约束... → 添加干扰色块

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox

# Design analysis for the D12 variant
# - Concrete change: keep the original hidden solid marker rectangle, but also add
#   1-2 small monochrome hollow rectangles elsewhere in the grid. They are visible
#   clutter, not part of the clean symmetric pattern, so the visible grid is no
#   longer globally mirror-consistent once the marker alone is ignored.
# - Behavioral trace: in the original task, `_find_reflected_strip_start` succeeds
#   because every non-marker visible cell already matches its LR/UD reflection. In
#   this variant, a hollow distractor introduces a few local mismatches, so the old
#   zero-error search would reject the true `lo=2`. The new solver therefore:
#   1) finds the marker as the only solid rectangle,
#   2) gets a provisional symmetry axis by minimizing reflection mismatches while
#      ignoring only the marker,
#   3) detects the small hollow-rectangle components responsible for those
#      mismatches and masks them,
#   4) re-runs the symmetry recovery while ignoring both marker and noise.
# - Complexity calibration: the noise is restricted to 1-2 one-pixel-thick frames
#   with bbox sizes 3..5 by 3..5. This is visually obvious but small enough that
#   the hidden rectangle remains the dominant object and the puzzle does not become
#   a generic denoising task.
# - Implicit assumptions review:
#   * Marker detection still relies on "true marker is a solid rectangle"; this
#     remains valid because distractors are explicitly non-solid.
#   * Exact bilateral symmetry of the visible area is no longer valid, so the old
#     `assert`-style strip search must be replaced by a best-score search.
#   * Reflection-orbit recovery can no longer trust every non-marker value; it must
#     ignore distractor cells as well.
#   * Transpose fallback must remain unobstructed; the generator protects those
#     cells from noise placement.
# - Placement legality enforced by the generator:
#   * distractor frames are in-bounds, disjoint from the marker bbox and from each
#     other, and do not overwrite any reflection-orbit witness cell or transpose
#     fallback witness cell needed by the solver;
#   * each frame color is chosen so it does not 4-connect to same-colored ambient
#     cells, ensuring the solver sees exactly the intended small rectangular
#     component;
#   * generation rejects symmetry-neutral placements by requiring that ignoring the
#     marker alone still leaves a positive symmetry mismatch score.
# - Visual observability: across 2-4 examples, a human will see the same original
#   "recover the marked rectangle from symmetry" rule plus a new nuisance pattern:
#   tiny empty rectangular outlines that clearly do not match the large bilateral
#   structure. The output remains only the marker-covered patch, not the clutter.
# - New solver assumptions introduced by this variant: hollow distractors are
#   4-connected single-color rectangular frames with bbox side lengths in 3..5, and
#   there are at most two of them. The generator guarantees those assumptions.


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


def _component_cells(g: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> set[tuple[int, int]]:
    color = int(g[start])
    q: deque[tuple[int, int]] = deque([start])
    cells: set[tuple[int, int]] = set()
    seen[start] = True

    while q:
        r, c = q.popleft()
        cells.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and int(g[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))

    return cells


def _rect_frame_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r == r0 or r == r1 or c == c0 or c == c1
    }


def _is_small_hollow_rectangle(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return False
    return cells == _rect_frame_cells(r0, c0, r1, c1)


def _symmetry_residual(g: np.ndarray, lo: int, ignore: np.ndarray) -> np.ndarray:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1
    residual = np.zeros_like(ignore, dtype=bool)

    strip = g[:, lo : hi + 1]
    strip_reflected = strip[:, ::-1]
    ignore_strip = ignore[:, lo : hi + 1]
    ignore_reflected = ignore_strip[:, ::-1]
    bad = (strip != strip_reflected) & (~ignore_strip) & (~ignore_reflected)
    residual[:, lo : hi + 1] |= bad
    residual[:, lo : hi + 1] |= bad[:, ::-1]

    strip = g[lo : hi + 1, :]
    strip_reflected = strip[::-1, :]
    ignore_strip = ignore[lo : hi + 1, :]
    ignore_reflected = ignore_strip[::-1, :]
    bad = (strip != strip_reflected) & (~ignore_strip) & (~ignore_reflected)
    residual[lo : hi + 1, :] |= bad
    residual[lo : hi + 1, :] |= bad[::-1, :]

    return residual


def _symmetry_score(g: np.ndarray, lo: int, ignore: np.ndarray) -> int:
    residual = _symmetry_residual(g, lo, ignore)
    return int(residual.sum())


def _find_reflected_strip_start(g: np.ndarray, ignore: np.ndarray) -> tuple[int, int]:
    n, m = map(int, g.shape)
    assert n == m

    best_lo = -1
    best_score: int | None = None
    for lo in range(n // 2 + 1):
        score = _symmetry_score(g, lo, ignore)
        if best_score is None or score < best_score:
            best_lo = lo
            best_score = score

    assert best_score is not None and best_lo >= 0
    return best_lo, best_score


def _noise_mask(g: np.ndarray, marker: int, lo: int) -> np.ndarray:
    marker_mask = g == marker
    residual = _symmetry_residual(g, lo, marker_mask)
    seen = np.zeros_like(marker_mask, dtype=bool)
    noise = np.zeros_like(marker_mask, dtype=bool)
    count = 0

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or marker_mask[r, c]:
                continue
            cells = _component_cells(g, (r, c), seen)
            if not _is_small_hollow_rectangle(cells):
                continue
            if not any(residual[rr, cc] for rr, cc in cells):
                continue
            count += 1
            for rr, cc in cells:
                noise[rr, cc] = True

    assert count <= 2
    return noise


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, blocked: np.ndarray) -> set[int]:
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

        if not bool(blocked[rr, cc]):
            values.add(int(g[rr, cc]))

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the patch hidden by the solid marker while ignoring small hollow-rectangle distractions.

    Input structure:
    - A square color grid contains one filled rectangular marker that hides the answer patch.
    - The underlying clean pattern still obeys the original left-right and up-down reflection rules on a large lower/right strip, with transpose fallback in the outer top/left border.
    - In addition, 1-2 tiny visible hollow rectangles appear as clutter; they are not part of the symmetric pattern and may locally break the visible reflections.

    Rule:
    - Identify the true marker as the only color whose entire footprint is a filled rectangle.
    - Estimate the reflection strip start from the least-contradictory bilateral symmetry, then detect the small hollow rectangular components that account for the remaining mismatches and ignore them.
    - For each marker cell, recover the unique non-blocked value from its horizontal/vertical reflection orbit; if that orbit is fully hidden inside the marker, read the transposed outer-strip counterpart instead.

    Key insight:
    The answer is still determined by the clean bilateral pattern; the new hollow rectangles are merely asymmetric clutter, so the solver must separate "filled hidden patch marker" from "visible rectangular noise" before using symmetry.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    marker_mask = g == marker

    provisional_lo, _ = _find_reflected_strip_start(g, marker_mask)
    noise_mask = _noise_mask(g, marker, provisional_lo)
    blocked = marker_mask | noise_mask
    lo, score = _find_reflected_strip_start(g, blocked)
    assert score == 0

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo, blocked)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
                assert not bool(blocked[c, r])
                out[rr, cc] = int(g[c, r])

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: distractor rectangles ---
_NOISE_DIM_LO = 3
_NOISE_DIM_HI = 5
_NOISE_COUNT_LO = 1
_NOISE_COUNT_HI = 2
_NOISE_PLACEMENT_ATTEMPTS = 120

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w)
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


def _marker_support_cells(r0: int, c0: int, h: int, w: int, fallback: list[tuple[int, int]]) -> set[tuple[int, int]]:
    support = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            for rr, cc in _orbit(r, c):
                if not (r0 <= rr < r0 + h and c0 <= cc < c0 + w):
                    support.add((rr, cc))
    for r, c in fallback:
        support.add((c, r))
    return support


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _frame_cells_from_hw(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return _rect_frame_cells(r0, c0, r0 + h - 1, c0 + w - 1)


def _adjacent_colors(grid: np.ndarray, cells: set[tuple[int, int]]) -> set[int]:
    colors: set[int] = set()
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in cells:
                colors.add(int(grid[nr, nc]))
    return colors


def _place_noise(inp: np.ndarray, marker_bbox: tuple[int, int, int, int], protected: set[tuple[int, int]]) -> tuple[np.ndarray, int] | None:
    noisy = inp.copy()
    count = int(_RNG.integers(_NOISE_COUNT_LO, _NOISE_COUNT_HI + 1))
    used_bboxes = [marker_bbox]

    for _ in range(count):
        placed = False
        for _attempt in range(_NOISE_PLACEMENT_ATTEMPTS):
            h = int(_RNG.integers(_NOISE_DIM_LO, _NOISE_DIM_HI + 1))
            w = int(_RNG.integers(_NOISE_DIM_LO, _NOISE_DIM_HI + 1))
            r0 = int(_RNG.integers(0, _N - h + 1))
            c0 = int(_RNG.integers(0, _N - w + 1))
            b = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_overlaps(b, prev) for prev in used_bboxes):
                continue

            cells = _frame_cells_from_hw(r0, c0, h, w)
            if cells & protected:
                continue

            neighbor_colors = _adjacent_colors(noisy, cells)
            options = [color for color in _PALETTE if int(color) not in neighbor_colors]
            if not options:
                continue

            color = int(options[int(_RNG.integers(len(options)))])
            for rr, cc in cells:
                noisy[rr, cc] = color
            used_bboxes.append(b)
            placed = True
            break

        if not placed:
            return None

    mismatch_score = _symmetry_score(noisy, _LO, noisy == _MARKER)
    if mismatch_score <= 0:
        return None

    return noisy, count


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    r0, c0, fallback = placements[int(_RNG.integers(len(placements)))]

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    protected = _marker_support_cells(r0, c0, h, w, fallback)
    placed = _place_noise(inp, (r0, c0, r0 + h - 1, c0 + w - 1), protected)
    if placed is None:
        return None
    noisy, _noise_count = placed

    try:
        if np.array_equal(main(noisy), expected):
            return noisy
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 D12 input")
