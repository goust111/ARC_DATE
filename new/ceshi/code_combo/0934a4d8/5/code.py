# Combo variant of 0934a4d8 (index 5)
# combo_spec: D8/1+D12/1
# Name: 对称性破坏识别标记+干扰色块

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox


# Design analysis for the D8/1 + D12/1 combo
# a) Concrete behavior and trace:
#    - The clean 30x30 source pattern still follows the original bilateral LR+UD symmetry on
#      rows/cols [lo..N-1], with transpose fallback for marker cells whose whole reflection
#      orbit stays inside the marker near the top/left border.
#    - The fused input now contains:
#      1) one solid rectangular occluder whose color is not fixed across samples,
#      2) one extra solid rectangle centered on the symmetry axes, so "unique solid rectangle"
#         is no longer a valid identification rule,
#      3) one or two small monochrome hollow rectangles that are visible clutter.
#    - Small trace: suppose the marker color is 6, the centered decoy uses color 8, and two tiny
#      hollow frames use colors 2 and 9. Old logic based on "pick the only solid rectangle" is now
#      ambiguous, and even after choosing color 6 the visible grid is still not perfectly symmetric
#      because the hollow frames create residual mismatches. The fused solver instead tests each
#      color as a possible wildcard, finds the best reflection strip under that assumption,
#      identifies the residual cells that form small hollow rectangles, masks them too, and keeps
#      only the unique color whose masked grid becomes perfectly symmetric. That color's rectangle
#      is then reconstructed by the same reflection-orbit plus transpose-fallback mechanism as the
#      base task.
# b) Complexity calibration:
#    - Noise is limited to 1-2 one-pixel-thick frames with bbox side lengths 3..5. This is enough
#      to visibly break exact symmetry, but small enough that the hidden rectangle remains the main
#      object and the task does not devolve into free-form denoising.
#    - The centered decoy is a single extra filled rectangle, so D8 changes the identification rule
#      without adding many new objects. Marker and decoy colors both vary across examples.
# c) Implicit-assumption audit against the original main():
#    - "The marker is the unique solid rectangle" is explicitly false now because a centered decoy
#      is also a filled rectangle. The marker is still solid so its bbox can define the output crop,
#      but identification now depends on the only color whose removal, together with small hollow-
#      frame residuals, restores zero symmetry error.
#    - "Ignoring marker pixels makes the visible grid exactly symmetric" is broken by D12. The new
#      strip search therefore minimizes residual mismatches first, then re-runs after masking the
#      detected clutter.
#    - "Every non-marker orbit witness is trustworthy" is also broken. Orbit recovery must ignore
#      both the candidate marker and the detected hollow frames.
#    - New solver assumptions: clutter components are 4-connected, monochrome, hollow rectangular
#      frames of size 3..5, and there are at most two of them. The generator enforces these.
# d) Placement legality:
#    - The generator keeps the solid marker in-bounds and only accepts placements where any fully
#      hidden reflection orbit still has a valid transpose witness outside the marker.
#    - Hollow frames are placed in-bounds, disjoint from the marker and from one another, and they
#      may not overwrite any orbit witness or transpose witness needed to recover the hidden patch.
#    - Noise colors are chosen to avoid 4-connecting to adjacent same-colored ambient cells, so the
#      solver sees exactly the intended small rectangular frame components.
# e) Visual observability:
#    - Across examples a human sees the same large symmetric field, two filled rectangles with
#      different roles, and 1-2 tiny empty rectangular outlines that are visibly out of family with
#      the bilateral layout. Because the marker/decoy colors change between examples, the stable cue
#      is "the color whose removal plus those tiny frames restores the symmetry", not a memorized
#      color id or a unique-rectangle shortcut.
# f) Visual cleanliness:
#    - There are only four layers: symmetric background, one filled occluder, one centered filled
#      decoy, and up to two tiny hollow frames. They never overlap, and the hollow frames remain
#      much smaller than either filled rectangle.
# g) Rule inferability:
#    - No invisible ordering or color arithmetic is used. Color identity matters only through visible
#      structure: one color forms the hidden solid rectangle, and small frame-shaped exceptions are
#      explicitly recognizable by shape. The repeated train examples support the same decomposition.


# --------------- solver ---------------

_N = 30
_LO = 2
_HI = _N - 1
_ALL_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

_DECOY_DIM_LO = 4
_DECOY_DIM_HI = 8

_NOISE_DIM_LO = 3
_NOISE_DIM_HI = 5
_NOISE_COUNT_LO = 1
_NOISE_COUNT_HI = 2
_NOISE_PLACEMENT_ATTEMPTS = 120

_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_DECOY_SIZE_PAIRS = [
    (h, w)
    for h in range(_DECOY_DIM_LO, _DECOY_DIM_HI + 1, 2)
    for w in range(_DECOY_DIM_LO, _DECOY_DIM_HI + 1, 2)
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _solid_rectangle_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    mask = g == int(color)
    b = bbox(mask)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    if area <= 1:
        return None
    if int(mask.sum()) != area:
        return None
    if not bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
        return None
    return int(r0), int(c0), int(r1), int(c1)


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
    return int(_symmetry_residual(g, lo, ignore).sum())


def _best_reflected_strip_start(g: np.ndarray, ignore: np.ndarray) -> tuple[int, int]:
    n, m = map(int, g.shape)
    assert n == m

    best_lo = -1
    best_score: int | None = None
    for lo in range(n // 2 + 1):
        score = _symmetry_score(g, lo, ignore)
        if best_score is None or score < best_score:
            best_lo = lo
            best_score = score

    assert best_lo >= 0 and best_score is not None
    return best_lo, best_score


def _noise_mask(g: np.ndarray, candidate_mask: np.ndarray, lo: int) -> tuple[np.ndarray, int] | None:
    residual = _symmetry_residual(g, lo, candidate_mask)
    seen = np.zeros_like(candidate_mask, dtype=bool)
    noise = np.zeros_like(candidate_mask, dtype=bool)
    count = 0

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or candidate_mask[r, c]:
                continue
            cells = _component_cells(g, (r, c), seen)
            if not _is_small_hollow_rectangle(cells):
                continue
            if not any(residual[rr, cc] for rr, cc in cells):
                continue
            count += 1
            if count > _NOISE_COUNT_HI:
                return None
            for rr, cc in cells:
                noise[rr, cc] = True

    return noise, count


def _candidate_infos(
    g: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], int, np.ndarray, int, int]]:
    infos: list[tuple[int, tuple[int, int, int, int], int, np.ndarray, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        candidate_mask = g == color
        provisional_lo, provisional_score = _best_reflected_strip_start(g, candidate_mask)
        noise_info = _noise_mask(g, candidate_mask, provisional_lo)
        if noise_info is None:
            continue
        noise_mask, noise_count = noise_info
        blocked = candidate_mask | noise_mask
        lo, score = _best_reflected_strip_start(g, blocked)
        if score != 0:
            continue
        marker_bbox = _solid_rectangle_bbox(g, color)
        if marker_bbox is None:
            continue
        infos.append((color, marker_bbox, lo, blocked, noise_count, provisional_score))
    return infos


def _marker_candidate(
    g: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], int, np.ndarray, int, int]:
    infos = _candidate_infos(g)
    assert len(infos) == 1
    return infos[0]


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, blocked: np.ndarray) -> set[int]:
    hi = int(g.shape[0] - 1)
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
    Recover the patch hidden by the only solid-rectangle color whose removal, together with tiny frame-shaped clutter, restores bilateral symmetry.

    Input structure:
    - A square color grid contains one true solid rectangular occluder and one extra centered solid rectangle, so "pick the only filled rectangle" is not valid.
    - The clean pattern underneath still follows the original left-right and up-down reflection rules on a large lower/right strip, with transpose fallback near the top/left border.
    - In addition, 1-2 small hollow rectangular color blocks are visible distractions; they are not part of the symmetric pattern and may also break the visible mirror checks.

    Rule:
    - Test each color as the possible "ignore this color and symmetry should come back" candidate.
    - For each trial color, find the least-contradictory reflection strip, detect any residual 3..5 hollow rectangular components as clutter, and keep the unique color whose mask plus those clutter cells yields perfect bilateral consistency.
    - Use that color's solid rectangular bbox as the output crop, then recover each covered cell from its reflection orbit; if every reflected witness is blocked, read the transposed outer-strip witness instead.

    Key insight:
    The centered filled decoy is symmetry-consistent, while the true marker is the only filled-rectangle color whose removal makes every remaining mismatch collapse into a few tiny hollow frames. Once those frames are ignored, the original symmetry reconstruction rule works unchanged.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1), lo, blocked, _noise_count, _provisional_score = _marker_candidate(g)
    marker_mask = g == marker
    assert np.array_equal(marker_mask, marker_mask & blocked)

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


# --------------- generator ---------------


def _choose_colors() -> tuple[int, int, np.ndarray]:
    marker, decoy = [int(color) for color in _RNG.choice(_ALL_COLORS, size=2, replace=False)]
    palette = np.array([int(color) for color in _ALL_COLORS if int(color) not in (marker, decoy)], dtype=int)
    return marker, decoy, palette


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


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]], palette: np.ndarray) -> None:
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
        color = int(_RNG.choice(palette))
        for r, c in coords:
            clean[r, c] = color


def _build_clean(palette: np.ndarray) -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(palette, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(palette, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(palette, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(palette, size=(_LO, _LO))
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


def _bbox_cells(b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return {(r, c) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1)}


def _frame_cells_from_hw(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return _rect_frame_cells(r0, c0, r0 + h - 1, c0 + w - 1)


def _adjacent_colors(grid: np.ndarray, cells: set[tuple[int, int]]) -> set[int]:
    colors: set[int] = set()
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in cells:
                colors.add(int(grid[nr, nc]))
    return colors


def _choose_marker_placement(
    placements: list[tuple[int, int, list[tuple[int, int]]]],
) -> tuple[int, int, list[tuple[int, int]]] | None:
    want_fallback = bool(int(_RNG.integers(0, 2)))
    bucket = [placement for placement in placements if bool(placement[2]) == want_fallback]
    if not bucket:
        bucket = placements
    if not bucket:
        return None
    return bucket[int(_RNG.integers(len(bucket)))]


def _centered_decoy_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    r0 = (_LO + _HI + 1 - h) // 2
    c0 = (_LO + _HI + 1 - w) // 2
    return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))


def _place_decoy(
    clean: np.ndarray,
    protected: set[tuple[int, int]],
    decoy: int,
) -> tuple[int, int, int, int] | None:
    order = list(_DECOY_SIZE_PAIRS)
    _RNG.shuffle(order)
    for h, w in order:
        b = _centered_decoy_bbox(h, w)
        cells = _bbox_cells(b)
        if cells & protected:
            continue
        clean[b[0] : b[2] + 1, b[1] : b[3] + 1] = decoy
        return b
    return None


def _place_noise(
    inp: np.ndarray,
    used_bboxes: list[tuple[int, int, int, int]],
    protected: set[tuple[int, int]],
    palette: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    noisy = inp.copy()
    count = int(_RNG.integers(_NOISE_COUNT_LO, _NOISE_COUNT_HI + 1))
    occupied = list(used_bboxes)

    for _ in range(count):
        placed = False
        for _attempt in range(_NOISE_PLACEMENT_ATTEMPTS):
            h = int(_RNG.integers(_NOISE_DIM_LO, _NOISE_DIM_HI + 1))
            w = int(_RNG.integers(_NOISE_DIM_LO, _NOISE_DIM_HI + 1))
            r0 = int(_RNG.integers(0, _N - h + 1))
            c0 = int(_RNG.integers(0, _N - w + 1))
            b = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_overlaps(b, prev) for prev in occupied):
                continue

            cells = _frame_cells_from_hw(r0, c0, h, w)
            if cells & protected:
                continue

            neighbor_colors = _adjacent_colors(noisy, cells)
            options = [int(color) for color in palette if int(color) not in neighbor_colors]
            if not options:
                continue

            color = int(options[int(_RNG.integers(len(options)))])
            for rr, cc in cells:
                noisy[rr, cc] = color
            occupied.append(b)
            placed = True
            break

        if not placed:
            return None

    return noisy, count


def _combo_metrics(g: np.ndarray) -> dict[str, int]:
    infos = _candidate_infos(g)
    solid_candidates = [int(color) for color in np.unique(g) if _solid_rectangle_bbox(g, int(color)) is not None]
    metrics = {
        "symmetry_candidate_count": len(infos),
        "solid_candidate_count": len(solid_candidates),
        "marker_only_score": -1,
        "noise_count": -1,
        "marker_color": -1,
    }
    if len(infos) == 1:
        marker, _bbox, _lo, _blocked, noise_count, provisional_score = infos[0]
        metrics["marker_only_score"] = int(provisional_score)
        metrics["noise_count"] = int(noise_count)
        metrics["marker_color"] = int(marker)
    return metrics


def _gen_attempt() -> np.ndarray | None:
    marker, decoy, palette = _choose_colors()
    clean = _build_clean(palette)

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    chosen = _choose_marker_placement(placements)
    if chosen is None:
        return None
    r0, c0, fallback = chosen

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in fallback]
    _unify_groups(clean, groups, palette)

    protected = _marker_support_cells(r0, c0, h, w, fallback)
    decoy_bbox = _place_decoy(clean, protected, decoy)
    if decoy_bbox is None:
        return None

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    if len(np.unique(expected)) < 2:
        return None

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = marker

    decoy_cells = _bbox_cells(decoy_bbox)
    placed = _place_noise(
        inp,
        [(r0, c0, r0 + h - 1, c0 + w - 1), decoy_bbox],
        protected | decoy_cells,
        palette,
    )
    if placed is None:
        return None
    noisy, placed_noise_count = placed

    try:
        if not np.array_equal(main(noisy), expected):
            return None
    except Exception:
        return None

    metrics = _combo_metrics(noisy)
    if metrics["symmetry_candidate_count"] != 1:
        return None
    if metrics["solid_candidate_count"] != 2:
        return None
    if metrics["marker_color"] != marker:
        return None
    if metrics["noise_count"] != placed_noise_count:
        return None
    if metrics["marker_only_score"] <= 0:
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 D8/1+D12/1 input")
