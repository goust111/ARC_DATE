# Combo variant of 71e489b6 (index 10)
# combo_spec: D7/0+D12/2
# Name: 光晕限蓝色区域+缺陷中心恢复蓝色
#
# Design analysis for combo D7/0 + D12/2
# a) Concrete trace:
#    - Original interior 1-cell defect:
#        111      777
#        101  ->  707
#        111      777
#      The 3x3 halo paints every neighbor, and the defect center stays black.
#    - Fused interior defect:
#        111      777
#        101  ->  717
#        111      777
#      The ring shape stays Chebyshev-1, but the center is repaired back to blue.
#    - Original 1-cell leaf attached to a black region:
#        0000      0777
#        0011  ->  0077
#        1111      7777
#      The halo spills into black body cells and the leaf center stays black.
#    - Fused leaf:
#        0000      0007
#        0011  ->  0017
#        1111      7777
#      Orange is clipped away on the black side, and the marked leaf cell itself
#      returns to blue, producing the intended "orange frame around a blue repair".
# b) Complexity calibration:
#    - Target detection is unchanged: absorb blue spurs, then mark every small black
#      component (size <= 3) and every degree-0/1 pixel inside larger black components.
#    - Only two rendering switches are changed, both with low discrete complexity:
#      (1) halo paint is allowed only on cleaned blue cells, and (2) target centers
#      become blue instead of black. This keeps the task learnable while making the
#      fused behavior visually distinct from the source puzzle.
# c) Hidden-assumption audit of the original main():
#    - Spur cleanup depended on an in-place sweep over blue pixels with >=3 effective
#      black 4-neighbors. That assumption still holds; the generator inserts only such
#      blue spur pixels.
#    - Target collection assumed all true anomalies are black after cleanup. That still
#      holds because defects and leaves are placed before spur injection.
#    - The original paint loop assumed any non-target cell in a 3x3 neighborhood could
#      be overwritten. D7/0 breaks that: now a halo write is legal only when the clean
#      base cell is blue.
#    - The original final loop forced targets back to black. D12/2 breaks that: every
#      target must instead return to its surrounding blue support color. The generator
#      guarantees all targets originate from blue cells, so "repair to blue" is valid.
# d) Placement legality:
#    - Small defects are carved from blue interior cells whose non-defect 4-neighbors
#      remain blue, so they stay isolated 1-3 cell black components.
#    - Leaves are created by flipping blue cells adjacent to exactly one black cell of
#      a large black region, so they become legal degree-1 tips after conversion.
#    - Clearance checks keep defects and leaves from merging with each other or falling
#      outside the grid.
# e) Visual observability:
#    - Every accepted sample contains several restored centers, so D12/2 is visible as
#      repeated blue reappearance exactly where black defects used to be.
#    - Every accepted sample also contains at least one leaf target, which guarantees
#      clipped halo cells on the black-facing side and makes D7/0 visible.
# f) Visual cleanliness:
#    - The generator keeps targets separated and rejects weak samples with too little
#      clipped-halo evidence, avoiding ambiguous orange clutter.
# g) Inferability:
#    - The output uses only one new visible color (orange). The blue-center behavior is
#      not arbitrary recoloring: it is tied to the same local structures that produce
#      the orange ring, so a human can infer "detect defect, repair center, mark rim"
#      from a few examples.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask


BLUE = 1
BLACK = 0
ORANGE = 7
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, treating OOB as black only for corners."""
    h, w = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _clean_spurs(grid: np.ndarray) -> np.ndarray:
    """Absorb blue spur pixels using the source task's in-place sweep."""
    base = np.array(grid, dtype=int)
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_count(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _collect_targets(base: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return all target pixels plus the small-component and leaf subsets."""
    h, w = base.shape
    preserve: set[tuple[int, int]] = set()
    small_targets: set[tuple[int, int]] = set()
    leaf_targets: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            small_targets.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = 0
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                    deg += 1
            if deg <= 1:
                preserve.add((r, c))
                leaf_targets.add((r, c))

    return preserve, small_targets, leaf_targets


def _paint_original_halo(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    """Reference renderer for the source-task output semantics."""
    h, w = base.shape
    out = np.array(base, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _paint_repair_halo(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange only on blue neighbors and restore target centers to blue."""
    h, w = base.shape
    out = np.array(base, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in preserve:
                    continue
                if int(base[rr, cc]) != BLUE:
                    continue
                out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLUE
    return out


def measure_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify the combo-defining differences on one input.

    Metrics:
    - restored_centers: number of target cells that flip from black to blue.
    - clipped_halo_cells: in-bounds halo cells the original task would color orange
      but the combo refuses because they are not blue.
    - diff_vs_original: total cell count where combo output differs from source-task
      output on the same cleaned target set.
    """
    base = _clean_spurs(np.array(grid, dtype=int))
    preserve, small_targets, leaf_targets = _collect_targets(base)

    clipped: set[tuple[int, int]] = set()
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < base.shape[0] and 0 <= cc < base.shape[1]):
                    continue
                if (rr, cc) in preserve:
                    continue
                if int(base[rr, cc]) != BLUE:
                    clipped.add((rr, cc))

    combo_out = _paint_repair_halo(base, preserve)
    original_out = _paint_original_halo(base, preserve)
    diff_vs_original = int(np.count_nonzero(combo_out != original_out))

    return {
        "target_count": len(preserve),
        "small_target_count": len(small_targets),
        "leaf_target_count": len(leaf_targets),
        "restored_centers": len(preserve),
        "clipped_halo_cells": len(clipped),
        "diff_vs_original": diff_vs_original,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair black defects inside blue regions by drawing an orange ring that stays inside blue.

    Input structure:
    - Large blue (1) and black (0) regions partition the grid.
    - Blue areas may contain tiny black holes, and large black areas may have 1-pixel black leaves poking into neighboring blue.
    - Some black regions also contain blue spur pixels that should be absorbed before defect marking.

    Rule:
    - First turn any blue pixel with at least three effective black 4-neighbors into black, using the same corner treatment as the source task.
    - Then treat every small black component of size <= 3 and every degree-0/1 pixel of a larger black component as a defect target.
    - For each target, color its Chebyshev-1 neighbors orange (7) only where those neighbor cells are still blue after cleanup, and finally restore the target cell itself to blue (1).

    Key insight:
    - The puzzle still detects the same structural black anomalies as before, but the output semantics become "repair plus annotate": orange marks the local rim, while the defect center is visibly healed back to blue and black regions block halo paint.
    """
    base = _clean_spurs(np.array(grid, dtype=int))
    preserve, _, _ = _collect_targets(base)
    return _paint_repair_halo(base, preserve)


# --------------- generator (combo D7/0 + D12/2) ---------------

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ("h_stripes", "rect_inset", "l_shape")
PARTITION_WEIGHTS = (3, 2, 3)
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5
DEFECT_COUNT_RANGE = (2, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 4)
MIN_LEAVES = 1
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_CLIPPED_HALO_CELLS = 3
MAX_GEN_ATTEMPTS = 500


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
    """Create a large-scale blue/black partition consistent with the source task."""
    grid = np.zeros((h, w), dtype=int)
    style = random.choices(PARTITION_STYLES, weights=PARTITION_WEIGHTS, k=1)[0]

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_pool = range(CUT_MARGIN, h - CUT_MARGIN)
        cuts = sorted(random.sample(cut_pool, min(n_cuts, max(0, h - 2 * CUT_MARGIN))))
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
    """Blue pixels whose in-bounds 4-neighbors are all blue."""
    blue = grid == BLUE
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_defects(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    """Place 2-8 small black defects strictly inside blue support."""
    h, w = grid.shape
    interior_blue = _find_interior_blue(grid)
    interior_set = set(interior_blue)
    defect_pixels: set[tuple[int, int]] = set()
    defects_placed = 0
    n_defects = random.randint(*DEFECT_COUNT_RANGE)

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]

        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = list(DIRS4)
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in DIRS4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w:
                            if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
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
                    if 0 <= rr < h and 0 <= cc < w:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
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
    """Place one-cell black leaves on the boundary of large black regions."""
    h, w = grid.shape
    leaf_pixels: set[tuple[int, int]] = set()
    n_leaves = random.randint(*LEAF_COUNT_RANGE)

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [comp for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = set((int(r), int(c)) for r, c in random.choice(large_black))
        candidates: list[tuple[int, int]] = []

        for r, c in comp_set:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                if _nearby(rr, cc, DEFECT_CLEARANCE, defect_pixels):
                    continue

                black_n = 0
                for dr2, dc2 in DIRS4:
                    r2 = rr + dr2
                    c2 = cc + dc2
                    if not (0 <= r2 < h and 0 <= c2 < w):
                        continue
                    if int(grid[r2, c2]) == BLACK or (r2, c2) in leaf_pixels:
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))

        if candidates:
            lr, lc = random.choice(candidates)
            leaf_pixels.add((lr, lc))
            grid[lr, lc] = BLACK

    return leaf_pixels


def _build_expected_output(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    return _paint_repair_halo(base, preserve)


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Black pixels that will be reabsorbed if temporarily turned blue."""
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
    """Generate one valid input for the fused variant."""
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    grid = _build_partition(h, w)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    defect_pixels, defects_placed = _place_defects(grid)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    preserve = defect_pixels | leaf_pixels
    output = _build_expected_output(grid, preserve)

    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, preserve)
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    stats = measure_fusion_behavior(inp)
    if stats["leaf_target_count"] < MIN_LEAVES:
        return None
    if stats["restored_centers"] < MIN_DEFECTS + MIN_LEAVES:
        return None
    if stats["clipped_halo_cells"] < MIN_CLIPPED_HALO_CELLS:
        return None
    if stats["diff_vs_original"] <= stats["restored_centers"]:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
