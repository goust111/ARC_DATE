# Combo variant of 71e489b6 (index 30)
# combo_spec: D1/4+D4/0
# Name: 方向性射线+仅边界缺陷
#
# Design analysis for combo variant D1/4 + D4/0
# - Concrete behavior:
#   The old task marks interior black holes and leaf pixels with a local 3x3 halo.
#   This combo changes both axes at once:
#   1) D4/0: every marked defect must be a black pixel on a visible blue/black boundary,
#      meaning it has at least one blue 4-neighbor and at least one black 4-neighbor.
#      Fully enclosed black holes are ignored.
#   2) D1/4: each such defect keeps its black color, but orange rays extend orthogonally
#      along the maximal same-color run entered on each side instead of painting a fixed halo.
# - Observability:
#   To keep the boundary rule inferable, the generator uses one thick off-center black
#   rectangle inside a blue field. Then "all black boundary cells are special, interior black
#   cells are not" is directly visible in 2-4 examples. The rectangle is deliberately biased
#   toward one side so some rays are much longer than others.
# - Original-task carry-over:
#   Blue spur absorption is preserved. A few interior blue pixels are inserted inside the black
#   rectangle, and the solver must absorb them before detecting boundary defects.
# - Anti-degeneracy:
#   The generator rejects samples whose output matches the original halo behavior or whose ray
#   lengths are too balanced, preventing collapse back to the source task or to a weak D1-only cue.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS = ROOT / "solvers"
if str(SOLVERS) not in sys.path:
    sys.path.insert(0, str(SOLVERS))

from arc_utils import components_mask


GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MARGIN = 2
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (5, 8)
MIN_BLACK_AREA = 20
MIN_TARGETS = 14
MIN_RAY_GAP = 5
SPUR_COUNT_RANGE = (2, 5)
MIN_SPURS = 2
SPUR_CLEARANCE = 1
MAX_RECT_TRIES = 120
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _nearby_any(
    pixels: list[tuple[int, int]], ref: set[tuple[int, int]], dist: int
) -> bool:
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                if (r + dr, c + dc) in ref:
                    return True
    return False


def _effective_black_neighbors(base: np.ndarray, r: int, c: int) -> int:
    H, W = base.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if base[rr, cc] == 0:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    base = np.array(grid, dtype=int)
    H, W = base.shape
    original = base.copy()
    for r in range(H):
        for c in range(W):
            if original[r, c] != 1:
                continue
            if _effective_black_neighbors(original, r, c) >= 3:
                base[r, c] = 0
    return base


def _boundary_defect(base: np.ndarray, r: int, c: int) -> bool:
    if base[r, c] != 0:
        return False
    blue_n = 0
    black_n = 0
    for rr, cc in _neighbors4(r, c):
        if not _in_bounds(base.shape, rr, cc):
            continue
        if base[rr, cc] == 1:
            blue_n += 1
        elif base[rr, cc] == 0:
            black_n += 1
    return blue_n > 0 and black_n > 0


def _collect_targets(base: np.ndarray) -> set[tuple[int, int]]:
    H, W = base.shape
    targets: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if _boundary_defect(base, r, c):
                targets.add((r, c))
    return targets


def _ray_lengths(base: np.ndarray, r: int, c: int) -> list[int]:
    lengths: list[int] = []
    for rr, cc in _neighbors4(r, c):
        dr = rr - r
        dc = cc - c
        if not _in_bounds(base.shape, rr, cc):
            lengths.append(0)
            continue
        color = int(base[rr, cc])
        n = 0
        while _in_bounds(base.shape, rr, cc) and int(base[rr, cc]) == color:
            n += 1
            rr += dr
            cc += dc
        lengths.append(n)
    return lengths


def _paint_rays(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for r, c in targets:
        for rr, cc in _neighbors4(r, c):
            dr = rr - r
            dc = cc - c
            if not _in_bounds(base.shape, rr, cc):
                continue
            side_color = int(base[rr, cc])
            while _in_bounds(base.shape, rr, cc) and int(base[rr, cc]) == side_color:
                if (rr, cc) not in targets:
                    out[rr, cc] = 7
                rr += dr
                cc += dc
    for r, c in targets:
        out[r, c] = 0
    return out


def _solve_core(grid: np.ndarray) -> np.ndarray:
    base = _absorb_spurs(grid)
    targets = _collect_targets(base)
    return _paint_rays(base, targets)


def _original_like_output(grid: np.ndarray) -> np.ndarray:
    base = _absorb_spurs(grid)
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == 0, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(base.shape, rr, cc) and base[rr, cc] == 0:
                    black_n += 1
            if black_n <= 1:
                preserve.add((r, c))
    out = base.copy()
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(base.shape, rr, cc) and (rr, cc) not in preserve:
                    out[rr, cc] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def _sample_metrics(grid: np.ndarray) -> dict[str, int]:
    base = _absorb_spurs(grid)
    targets = _collect_targets(base)
    gaps: list[int] = []
    max_ray = 0
    for r, c in targets:
        pos = [x for x in _ray_lengths(base, r, c) if x > 0]
        if pos:
            gaps.append(max(pos) - min(pos))
            max_ray = max(max_ray, max(pos))
    return {
        "targets": len(targets),
        "max_gap": max(gaps, default=0),
        "max_ray": max_ray,
        "diff_vs_original": int(
            np.count_nonzero(_solve_core(grid) != _original_like_output(grid))
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only black pixels that lie on a blue/black boundary, then replace the old halo with rays.

    Input structure:
    - Blue (1) background with one thick black rectangle offset from the canvas center.
    - A few interior blue spur pixels may sit inside the black region as removable noise.

    Rule:
    - First absorb any blue spur that is effectively surrounded by black on at least three sides.
    - Then find every black pixel that has at least one blue 4-neighbor and at least one black
      4-neighbor. These boundary black pixels are the only defects.
    - From each defect pixel, fire orange (7) rays in the four orthogonal directions. A ray follows
      the maximal straight run of the first color it enters and stops when that run ends or the
      grid edge is reached. The defect pixels themselves stay black.

    Key insight: target selection is now boundary-based, while the marking geometry is long and
    directional instead of local.
    """
    return _solve_core(np.array(grid, dtype=int))


def _rectangle_target_count(height: int, width: int) -> int:
    return 2 * height + 2 * width - 4


def _place_black_rectangle(H: int, W: int) -> np.ndarray | None:
    height_lo = RECT_HEIGHT_RANGE[0]
    height_hi = min(RECT_HEIGHT_RANGE[1], H - 2 * MARGIN)
    width_lo = RECT_WIDTH_RANGE[0]
    width_hi = min(RECT_WIDTH_RANGE[1], W - 2 * MARGIN)
    dims = [
        (height, width)
        for height in range(height_lo, height_hi + 1)
        for width in range(width_lo, width_hi + 1)
        if height * width >= MIN_BLACK_AREA
    ]
    if not dims:
        return None

    for _ in range(MAX_RECT_TRIES):
        height, width = random.choice(dims)
        r0 = random.randint(MARGIN, H - height - MARGIN)
        c0 = random.randint(MARGIN, W - width - MARGIN)
        r1 = r0 + height
        c1 = c0 + width

        grid = np.ones((H, W), dtype=int)
        grid[r0:r1, c0:c1] = 0

        if _rectangle_target_count(height, width) < MIN_TARGETS:
            continue
        metrics = _sample_metrics(grid)
        if metrics["targets"] < MIN_TARGETS or metrics["max_gap"] < MIN_RAY_GAP:
            continue
        return grid
    return None


def _find_spur_candidates(
    grid: np.ndarray, exclude: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != 0 or (r, c) in exclude:
                continue
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == 0:
                    black_n += 1
            if black_n == 4:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _insert_absorbable_spurs(
    grid: np.ndarray, exclude: set[tuple[int, int]]
) -> np.ndarray | None:
    out = grid.copy()
    chosen: set[tuple[int, int]] = set()
    candidates = _find_spur_candidates(out, exclude)
    want = min(random.randint(*SPUR_COUNT_RANGE), len(candidates))
    if want < MIN_SPURS:
        return None

    for r, c in candidates:
        if _nearby_any([(r, c)], chosen, SPUR_CLEARANCE):
            continue
        out[r, c] = 1
        chosen.add((r, c))
        if len(chosen) == want:
            break

    if len(chosen) < MIN_SPURS:
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    base = _place_black_rectangle(H, W)
    if base is None:
        return None

    metrics = _sample_metrics(base)
    if metrics["diff_vs_original"] == 0:
        return None

    expected = main(base)
    targets = _collect_targets(base)
    inp = _insert_absorbable_spurs(base, targets)
    if inp is None:
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
