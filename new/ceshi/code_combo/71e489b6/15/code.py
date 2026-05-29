# Combo variant of 71e489b6 (index 15)
# combo_spec: D3/0+D12/1
# Name: 毛刺品红标记+缺陷中心红色

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

# Design analysis for the combo variant D3/0 + D12/1
# a) Concrete behavior and trace:
#    - Base task flow: blue spur pixels in black areas are first absorbed into black, then black defects are
#      detected, then each detected pixel gets an orange 3x3 halo while the center stays black.
#    - Fused flow:
#      1. Detect the same spur pixels by the original ">=3 effective black 4-neighbors" test.
#      2. For defect analysis only, temporarily flip those spur pixels back to black so the black topology is
#         the same as in the base task.
#      3. Detect small black components (size <= 3) and leaf pixels of large black components on that cleaned grid.
#      4. Output uses three marker layers: defect centers become red (2), their Chebyshev-1 neighborhood becomes
#         orange (7), and spur pixels themselves become magenta (6).
#    - Small trace:
#         input patch            analysis patch         output patch
#         0 0 0                  0 0 0                  0 6 0
#         0 1 0   spur ->        0 0 0   no defect ->   0 0 0
#         0 0 0                  0 0 0                  0 0 0
#      For a black singleton inside blue:
#         1 1 1                  1 1 1                  7 7 7
#         1 0 1   defect ->      1 0 1   target ->      7 2 7
#         1 1 1                  1 1 1                  7 7 7
# b) Complexity calibration:
#    - The rule family stays compact: there are still only two anomaly classes from the original task world.
#      The fusion only changes how each class is rendered.
#    - To avoid a weak variant, the generator enforces at least 2 defects and at least 2 visible spurs, so both
#      D3/0 and D12/1 are present in every accepted sample.
# c) Broken original assumptions:
#    - Original STEP 0 assumed "spur removal mutates the board used for the final output." That is now false.
#      The solver must separate analysis state from render state.
#    - Original final pass assumed defect centers should stay black. That is now false: centers are red.
#    - If the solver were to detect defects on the raw input without temporary spur absorption, spurs could split
#      large black regions and create fake leaves. The fused solver therefore reconstructs the cleaned black layout
#      before target detection.
# d) Placement legality:
#    - Defects are still placed strictly inside blue interiors and kept apart by Chebyshev clearance.
#    - Leaves still grow from large black regions into adjacent blue cells.
#    - Spurs are only placed on black cells that satisfy the original absorption criterion, are not themselves
#      defect/leaf pixels in the cleaned grid, and are buffered away from defect halos for readability.
# e) Visual observability:
#    - A human can now see two distinct cues across examples: red-with-orange-ring always sits on black defects
#      inside or at the edge of blue regions, while magenta marks lone blue spikes embedded in black zones.
#    - Because the generator forces repeated instances of both classes, the color semantics are inferable from
#      2-4 examples rather than hinging on a single isolated event.
# f) Visual neatness:
#    - Spur candidates are kept outside the Chebyshev-1 halo zone of defects, so magenta does not usually collide
#      with orange. The board remains readable as "red center + orange ring" versus "magenta spur."
# g) Color inference:
#    - Only four output colors coexist: original black/blue plus magenta spur marks and red/orange defect marks.
#      Each added color has a stable structural cue; no color depends on hidden numeric ordering.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes any spur pixel should be reinterpreted as black only for defect detection; `generate_input()`
#      constructs spurs exactly via that criterion.
#    - `main()` assumes small natural black regions outside blue are absent; the generator preserves the original
#      min-size constraints for large black regions, so all tiny black components are intentional defects.
#    - `main()` gives defect centers and spur marks precedence over halo paint; the generator spaces these objects
#      out so that precedence rarely needs to resolve clutter.


BLUE = 1
BLACK = 0
RED = 2
MAGENTA = 6
ORANGE = 7

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5
DEFECT_COUNT_RANGE = (2, 9)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 50
LEAF_COUNT_RANGE = (0, 4)
SPUR_COUNT_RANGE = (2, 6)
SPUR_CLEARANCE = 1
# Keep spur marks at least one full cell away from defect halos for cleaner examples.
SPUR_TARGET_BUFFER = 2
MIN_DEFECTS = 2
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 300


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, with out-of-bounds counted only at corners."""
    h, w = grid.shape
    black_neighbors = 0
    oob = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_neighbors += 1
    if oob >= 2:
        black_neighbors += oob
    return int(black_neighbors)


def _find_spurs(grid: np.ndarray) -> set[tuple[int, int]]:
    """Blue pixels that the base task would absorb into black."""
    h, w = grid.shape
    spurs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLUE:
                continue
            if _effective_black_count(grid, r, c) >= 3:
                spurs.add((int(r), int(c)))
    return spurs


def _analysis_grid(grid: np.ndarray, spurs: set[tuple[int, int]]) -> np.ndarray:
    """Temporarily absorb spurs so defect detection sees the original black topology."""
    base = np.array(grid, dtype=int)
    for r, c in spurs:
        base[r, c] = BLACK
    return base


def _collect_defect_pixels(base: np.ndarray) -> set[tuple[int, int]]:
    """Find all black defect pixels: small components and leaves of large components."""
    h, w = base.shape
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == BLACK, connectivity=4)

    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        for r, c in pts:
            degree = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and int(base[rr, cc]) == BLACK:
                    degree += 1
            if degree <= 1:
                preserve.add((int(r), int(c)))

    return preserve


def _paint_output(
    clean_grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    spurs: set[tuple[int, int]],
) -> np.ndarray:
    """Render orange halos, red defect centers, and magenta spurs."""
    h, w = clean_grid.shape
    out = np.array(clean_grid, dtype=int)

    for r, c in defect_pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in defect_pixels and (rr, cc) not in spurs:
                    out[rr, cc] = ORANGE

    for r, c in defect_pixels:
        out[r, c] = RED

    for r, c in spurs:
        out[r, c] = MAGENTA

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark two anomaly classes with separate colors on the same blue/black partition.

    Input structure:
    - Large blue foreground regions and large black regions share the board.
    - Small black holes or thin black leaf tips sit inside the blue structures.
    - Occasional blue spur pixels lie in black zones and are almost surrounded by black.

    Rule:
    - First locate spur pixels using the original absorption test: a blue pixel with at least three effective black
      4-neighbors, counting out-of-bounds only for corner cells.
    - For defect detection only, temporarily treat those spurs as black again. On that cleaned board, every black
      component of size at most three and every degree-0/1 pixel of a larger black component is a defect center.
    - Paint an orange Chebyshev-1 halo around each defect center, recolor the defect centers red, and recolor spur
      pixels magenta instead of deleting them.

    Key insight:
    The board now contains two visible anomaly detectors at once: structural black defects get a red core with an
    orange ring, while blue spikes embedded in black remain in place and are singled out in magenta.
    """
    g = np.array(grid, dtype=int)
    spurs = _find_spurs(g)
    base = _analysis_grid(g, spurs)
    defect_pixels = _collect_defect_pixels(base)
    return _paint_output(base, defect_pixels, spurs)


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    """Return True if any pixel in ref is within Chebyshev distance dist."""
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    """Return all coordinates within Chebyshev distance dist of the given pixels."""
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _build_partition(h: int, w: int) -> np.ndarray:
    """Create a grid partitioned into blue and black macro-regions."""
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(random.sample(range(CUT_MARGIN, h - CUT_MARGIN), min(n_cuts, h - 2 * CUT_MARGIN)))
        bounds = [0] + cuts + [h]
        first_color = random.randint(0, 1)
        for i in range(len(bounds) - 1):
            grid[bounds[i] : bounds[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, h // 3)
        r1 = random.randint(2 * h // 3, h - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, w // 3)
        c1 = random.randint(2 * w // 3, w - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP and random.random() < INNER_INSET_PROB):
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


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    """Place small black defects in blue interiors."""
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
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w and (grid[rr, cc] != BLUE or (rr, cc) in defect_pixels):
                            ok = False
                            break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w and (grid[rr, cc] != BLUE or (rr, cc) in defect_pixels):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(set(defect), DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place leaf protrusions from large black regions into adjacent blue territory."""
    h, w = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [comp for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = set((int(r), int(c)) for r, c in random.choice(large_black))
        candidates = []
        for r, c in comp_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                black_n = sum(
                    1
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= rr + dr2 < h
                    and 0 <= cc + dc2 < w
                    and (int(grid[rr + dr2, cc + dc2]) == BLACK or (rr + dr2, cc + dc2) in leaf_pixels)
                )
                if black_n == 1:
                    candidates.append((int(rr), int(cc)))

        if candidates:
            lr, lc = random.choice(candidates)
            if not _nearby(lr, lc, DEFECT_CLEARANCE, defect_pixels):
                leaf_pixels.add((int(lr), int(lc)))
                grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(
    clean_grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    halo_buffer: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Black cells that can become isolated blue spur marks without colliding with defect cues."""
    rs, cs = np.where(clean_grid == BLACK)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        pt = (int(r), int(c))
        if pt in defect_pixels or pt in halo_buffer:
            continue
        if _effective_black_count(clean_grid, int(r), int(c)) < 3:
            continue
        candidates.append(pt)
    random.shuffle(candidates)
    return candidates


def _select_spurs(candidates: list[tuple[int, int]], count: int) -> set[tuple[int, int]]:
    """Greedily choose well-separated single-cell spurs."""
    chosen: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    for r, c in candidates:
        if (r, c) in blocked:
            continue
        chosen.add((int(r), int(c)))
        blocked |= _exclusion_zone({(int(r), int(c))}, SPUR_CLEARANCE)
        if len(chosen) >= count:
            break
    return chosen


def _build_expected_output(clean_grid: np.ndarray, spurs: set[tuple[int, int]]) -> np.ndarray:
    """Build the fused output from the clean board plus chosen spur locations."""
    defect_pixels = _collect_defect_pixels(clean_grid)
    return _paint_output(clean_grid, defect_pixels, spurs)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the fused variant."""
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    grid = _build_partition(h, w)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    _place_leaves(grid, defect_pixels)
    clean_grid = np.array(grid, dtype=int)
    analysis_defects = _collect_defect_pixels(clean_grid)
    halo_buffer = _exclusion_zone(analysis_defects, SPUR_TARGET_BUFFER)
    candidates = _find_spur_candidates(clean_grid, analysis_defects, halo_buffer)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    spurs = _select_spurs(candidates, n_spurs)
    if len(spurs) < MIN_SPURS:
        return None

    inp = np.array(clean_grid, dtype=int)
    for r, c in spurs:
        inp[r, c] = BLUE

    output = _build_expected_output(clean_grid, spurs)

    # Reject degenerate samples where the fused cues are not both visible.
    if int(np.count_nonzero(output == MAGENTA)) < MIN_SPURS:
        return None
    if int(np.count_nonzero(output == RED)) < MIN_DEFECTS:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
