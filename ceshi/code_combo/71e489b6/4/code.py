# Combo variant of 71e489b6 (index 4)
# combo_spec: D1/4+D7/0
# Name: 方向性射线+光晕限蓝色区域
#
# Design analysis for combo D1/4 + D7/0
# a) Concrete behavior trace:
#    - Original task: after absorbing blue spurs, each black defect/leaf gets a 3x3
#      Chebyshev halo. That adds diagonals and can overwrite either blue or black.
#    - Fused task: the same defect set is detected, but each defect pixel now emits
#      four orthogonal orange rays. A ray starts one step away from the black source,
#      paints only while it keeps stepping on blue cells, and stops before the first
#      non-blue cell or the grid edge.
#    - Small example:
#          input row:   1 1 1 0 1 1 0 0
#          original:    7 7 7 0 7 7 7 0   (local 3x3 splash, including diagonals in
#                                          neighboring rows)
#          fused row:   7 7 7 0 7 7 0 0   (left/right rays only; they stop at the
#                                          next black cell and do not cross into it)
#      The same source pixel therefore changes both geometry (square -> cross rays)
#      and coverage semantics (blue-only clipping).
# b) Complexity calibration:
#    - The object vocabulary stays small: large blue/black regions, tiny black
#      defects, black leaves, and absorbed blue spurs.
#    - Only two visible controls are added: variable ray length and blue-region
#      clipping. The generator forces repeated long rays and repeated black-boundary
#      stops so the combo does not collapse into a near-local plus-sign task.
# c) Implicit assumption audit of the original main():
#    - The old "paint all 8 neighbors" loop assumed every marker has fixed radius 1;
#      that is invalid and must be replaced by directional scans.
#    - The old painter could overwrite black regions across region borders; under the
#      combo, orange is allowed only on blue cells, so the write predicate changes.
#    - Target detection still assumes small black components are full targets and
#      degree<=1 pixels of larger black components are leaf targets; the generator
#      preserves those structural conditions.
# d) Placement legality:
#    - Defects are placed strictly inside blue interiors and kept separated so they
#      neither merge with each other nor with large black regions.
#    - Leaves are added as single black protrusions attached to large black regions.
#    - Spurs are inserted only at black pixels that the preprocessing step provably
#      reabsorbs, so solver and generator stay consistent.
# e) Visual observability:
#    - Humans can see that orange is no longer a 3x3 blob: it forms straight row/col
#      rays whose lengths differ by available blue run length.
#    - The stopping rule is visible because orange halts exactly at black region
#      boundaries that are already present in the input.
# f) Visual cleanliness:
#    - The generator enforces clearance between defects and rejects weak samples with
#      too few far-orange cells, reducing unreadable overdraw.
#    - Region templates remain axis-aligned and sparse enough that rays are legible.
# g) Inferability of the change:
#    - No hidden ordering or color-id logic is used. Every changed output pixel is
#      explained by visible spatial facts: source defect position, orthogonal line of
#      sight, and blue-vs-black occupancy along that line.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask


BLUE = 1
BLACK = 0
ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Absorb blue pixels that are structurally black-region spurs."""
    src = np.array(grid, dtype=int)
    h, w = src.shape
    out = src.copy()
    for r in range(h):
        for c in range(w):
            if int(src[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if not _in_bounds(h, w, rr, cc):
                    oob += 1
                elif int(src[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                out[r, c] = BLACK
    return out


def _collect_targets(base: np.ndarray) -> set[tuple[int, int]]:
    """Return black defect pixels: tiny components and leaves of larger ones."""
    h, w = base.shape
    targets: set[tuple[int, int]] = set()
    comps = components_mask(base == BLACK, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            targets.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = 0
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if (rr, cc) in comp_set:
                    deg += 1
            if deg <= 1:
                targets.add((r, c))
    return targets


def _paint_rays(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Emit orthogonal orange rays from each target, clipped to blue cells only."""
    h, w = base.shape
    out = np.array(base, dtype=int)
    for r, c in targets:
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            while _in_bounds(h, w, rr, cc) and int(base[rr, cc]) == BLUE:
                out[rr, cc] = ORANGE
                rr += dr
                cc += dc
    for r, c in targets:
        out[r, c] = BLACK
    return out


def _ray_statistics(base: np.ndarray, targets: set[tuple[int, int]]) -> dict[str, int]:
    """Measure the combo-defining behaviors on the clean pre-output grid."""
    h, w = base.shape
    long_dirs = 0
    internal_stop_dirs = 0
    longest_ray = 0

    for r, c in targets:
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            steps = 0
            while _in_bounds(h, w, rr, cc) and int(base[rr, cc]) == BLUE:
                steps += 1
                rr += dr
                cc += dc
            if steps >= 2:
                long_dirs += 1
            longest_ray = max(longest_ray, steps)
            if steps > 0 and _in_bounds(h, w, rr, cc) and int(base[rr, cc]) != BLUE:
                internal_stop_dirs += 1

    out = _paint_rays(base, targets)
    far_orange_cells = 0
    orange_coords = np.argwhere(out == ORANGE)
    target_list = list(targets)
    for r, c in orange_coords:
        if min(abs(int(r) - tr) + abs(int(c) - tc) for tr, tc in target_list) >= 2:
            far_orange_cells += 1

    return {
        "target_count": len(targets),
        "long_dirs": int(long_dirs),
        "internal_stop_dirs": int(internal_stop_dirs),
        "longest_ray": int(longest_ray),
        "far_orange_cells": int(far_orange_cells),
    }


def measure_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    """
    Public helper for validating that generated samples exhibit the fused rule.

    Metrics:
    - far_orange_cells: orange cells at Manhattan distance >= 2 from every target;
      this is impossible under the original local 3x3 halo.
    - internal_stop_dirs: ray directions that visibly stop on a black boundary rather
      than merely running off the canvas.
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    targets = _collect_targets(base)
    return _ray_statistics(base, targets)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects inside blue foreground using clipped directional rays.

    Input structure:
    - Blue (1) foreground regions are interleaved with black (0) regions; blue areas
      may contain tiny black holes, and black areas may contain a few thin blue spurs.

    Rule:
    - First absorb blue spur pixels that are mostly surrounded by black.
    - Then treat every small black component (size <= 3) and every leaf pixel of a
      larger black component as a defect source that must remain black.
    - From each source pixel, extend orange (7) rays in the four orthogonal
      directions, but only across consecutive blue cells; stop before the first
      non-blue cell or the grid boundary.

    Key insight:
    - The marked black pixels are the same structural anomalies as in the base task,
      but the orange annotation is no longer a fixed halo. Its geometry is the set of
      blue line-of-sight runs that emanate from those anomaly pixels.
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    targets = _collect_targets(base)
    return _paint_rays(base, targets)


# --------------- generator (combo D1/4 + D7/0) ---------------

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.6
DEFECT_COUNT_RANGE = (3, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 3
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 4)
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 3
MIN_LEAVES = 1
MIN_SPURS = 2
MIN_FAR_ORANGE_CELLS = 10
MIN_INTERNAL_STOP_DIRS = 4
MIN_LONGEST_RAY = 3
MAX_GEN_ATTEMPTS = 400


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_pool = range(CUT_MARGIN, h - CUT_MARGIN)
        cuts = sorted(random.sample(cut_pool, min(n_cuts, max(1, h - 2 * CUT_MARGIN))))
        boundaries = [0] + cuts + [h]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i] : boundaries[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, h // 3)
        r1 = random.randint(2 * h // 3, h - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, w // 3)
        c1 = random.randint(2 * w // 3, w - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP
                and random.random() < INNER_INSET_PROB):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    else:
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(w // 3, 2 * w // 3)
        hc = random.randint(h // 3, 2 * h // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLUE:
                continue
            if all(int(grid[r + dr, c + dc]) == BLUE for dr, dc in DIRS4):
                coords.append((r, c))
    random.shuffle(coords)
    return coords


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    """Place separated small black defects strictly inside blue regions."""
    h, w = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = DIRS4.copy()
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    if _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        continue
                    ok = True
                    for dr2, dc2 in DIRS4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if not _in_bounds(h, w, rr, cc) or int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                            ok = False
                            break
                    if ok:
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if not _in_bounds(h, w, rr, cc) or int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect_set)
                defects_placed += 1
                remove = _exclusion_zone(defect_set, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Attach a few 1-pixel leaves to large black regions."""
    h, w = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()
    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = random.choice(large_black)
        candidates: list[tuple[int, int]] = []
        for r, c in comp_set:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if not _in_bounds(h, w, rr, cc) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                if _nearby(rr, cc, DEFECT_CLEARANCE, defect_pixels | leaf_pixels):
                    continue
                black_n = 0
                for dr2, dc2 in DIRS4:
                    r2 = rr + dr2
                    c2 = cc + dc2
                    if _in_bounds(h, w, r2, c2) and (int(grid[r2, c2]) == BLACK or (r2, c2) in leaf_pixels):
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
        if candidates:
            lr, lc = random.choice(candidates)
            leaf_pixels.add((lr, lc))
            grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels that become absorbable blue spurs."""
    h, w = grid.shape
    black = grid == BLACK
    nb = np.zeros((h, w), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((h, w), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = _build_partition(h, w)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return None
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return None
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    preserve = defect_pixels | leaf_pixels
    if len(leaf_pixels) < MIN_LEAVES:
        return None
    if len(preserve) < MIN_DEFECTS:
        return None

    output = _paint_rays(grid, preserve)
    metrics = _ray_statistics(grid, preserve)
    if metrics["far_orange_cells"] < MIN_FAR_ORANGE_CELLS:
        return None
    if metrics["internal_stop_dirs"] < MIN_INTERNAL_STOP_DIRS:
        return None
    if metrics["longest_ray"] < MIN_LONGEST_RAY:
        return None

    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, preserve)
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid combo input after {MAX_GEN_ATTEMPTS} attempts")
