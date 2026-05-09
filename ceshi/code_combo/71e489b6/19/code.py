# Combo variant of 71e489b6 (index 19)
# combo_spec: D9/0+D10/1
# Name: 灰色噪点干扰+非黑背景

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


# Design analysis for combo D9/0 + D10/1
# a) Concrete behavior trace:
#    - Original local black-in-blue defect:
#        1 1 1        7 7 7
#        1 0 1   ->   7 0 7
#        1 1 1        7 7 7
#      The black cell is haloed because it is a tiny black component.
#    - Fused local black-in-grey defect:
#        5 5 5        7 7 7
#        5 0 5   ->   7 0 7
#        5 5 5        7 7 7
#      The same tiny black shape is still haloed, but only because it is enclosed by grey.
#    - New grey-noise cleanup inside a large black area:
#        0 0 0        0 0 0
#        0 5 0   ->   0 0 0
#        0 0 0        0 0 0
#      This reuses the old "mostly black neighbors imply cleanup" idea, but the nuisance color
#      is now grey rather than blue.
#    - Large black regions still contribute marked leaf tips when they stick one cell into blue.
#      That keeps the structural contrast between "isolated small black in grey" and "tip of a
#      larger black mass entering blue".
# b) Complexity calibration:
#    - The fused task uses only four visible colors overall: black, blue, grey, orange.
#    - There are only two marked target classes: small black components enclosed by grey, and
#      degree-0/1 leaf pixels of large black components that touch blue. Grey inside black is
#      never marked; it is normalized away.
#    - The generator always includes several grey-enclosed defects and several removable grey
#      noise pixels, so neither dimension is latent.
# c) Implicit-assumption audit of the original solver:
#    - Original STEP 0 assumed the nuisance color was blue; that is now false. The cleanup pass
#      must inspect grey cells instead.
#    - Original small-component detection assumed "small black anywhere" was enough. That breaks
#      under D10/1 because black is no longer also the background. We now require small black
#      components to be enclosed by grey, which is visible and structurally meaningful.
#    - Original leaf detection assumed any degree-0/1 pixel in a large black component was a tip.
#      In this combo we also require at least one blue 4-neighbor, so only protrusions into blue
#      are marked and compact large black rectangles do not leak false positives.
#    - A new solver assumption appears: grey pixels that satisfy the cleanup predicate are noise
#      inside large black regions, not meaningful background boundary cells. The generator enforces
#      this by placing grey noise only on black cells with at least three effective black neighbors.
# d) Placement legality:
#    - Large black regions are placed only on grey cells and must touch blue along an edge so leaf
#      pixels can be added legally.
#    - Small defects are carved only from grey interior cells whose remaining 4-neighbors stay grey.
#    - Leaves are added only on blue cells adjacent to a large black component with exactly one
#      black attachment, so they remain one-pixel protrusions instead of thick bridges.
#    - Grey noise is injected only into black cells that the cleanup pass deterministically restores.
# e) Visual observability:
#    - Humans can see the palette shift immediately: grey is the broad background/support color,
#      blue is still a foreground region color, and black now plays the role of anomaly/object color.
#    - The input also shows two different black/grey interactions: tiny black spots in grey get
#      orange halos, while tiny grey holes in black disappear.
# f) Visual cleanliness:
#    - Noise pixels are kept away from orange halos, and defects are separated from large black
#      bodies, so each local motif stays readable.
# g) Change inferability:
#    - The fusion does not rely on numeric color ids. The rule is visible from local enclosure:
#      grey-around-black means "mark", black-around-grey means "absorb", blue next to a black tip
#      means the tip is a protrusion target.
# h) Degeneration check target:
#    - If generator samples lacked grey-enclosed black defects or removable grey noise, the combo
#      could collapse to a trivial recolor. Rejection checks below prevent that.
# i) Solver/generator contract:
#    - `main()` assumes small marked defects are strictly grey-enclosed and leaf tips are blue-adjacent.
#      `generate_input()` guarantees both conditions by construction.


BLACK = 0
BLUE = 1
GREY = 5
ORANGE = 7


# --------------- solver helpers ---------------

def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors; count OOB as black only for corner cells."""
    H, W = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _absorb_grey_noise(grid: np.ndarray) -> np.ndarray:
    """Turn grey cells back to black when they are mostly enclosed by black."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != GREY:
                continue
            if _effective_black_neighbors(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _same_component_neighbors_ok(
    grid: np.ndarray, comp_set: set[tuple[int, int]], surround_color: int
) -> bool:
    """True iff every non-component 4-neighbor is in-bounds and has `surround_color`."""
    H, W = grid.shape
    for r, c in comp_set:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if (rr, cc) in comp_set:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if int(grid[rr, cc]) != surround_color:
                return False
    return True


def _collect_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Collect grey-enclosed tiny black components and blue-adjacent leaf pixels."""
    H, W = grid.shape
    preserve: set[tuple[int, int]] = set()
    for coords in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        comp_set = set(pts)
        if len(pts) <= 3:
            if _same_component_neighbors_ok(grid, comp_set, GREY):
                preserve.update(pts)
            continue
        for r, c in pts:
            degree = 0
            blue_touch = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if (rr, cc) in comp_set:
                        degree += 1
                    elif int(grid[rr, cc]) == BLUE:
                        blue_touch = True
            if degree <= 1 and blue_touch:
                preserve.add((r, c))
    return preserve


def _paint_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    H, W = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark grey-enclosed black defects and black leaf tips, while absorbing grey noise in black.

    Input structure:
    - Grey (5) is the broad background/support color, and blue (1) still forms large foreground regions.
    - Small black components may sit fully inside grey regions, and large black regions may border blue and
      sprout one-pixel black leaf tips into it.
    - Some black regions also contain tiny grey specks that are nuisance holes rather than true defects.

    Rule:
    - First convert any grey pixel with at least three effective black 4-neighbors back to black; these are
      removable grey noise pixels inside black regions.
    - Then detect target black pixels on the cleaned grid:
      tiny black components of size <= 3 are marked only when every non-component 4-neighbor is grey, and
      larger black components contribute only degree-0/1 pixels that touch blue.
    - Paint orange (7) on the Chebyshev-1 neighborhood of every target pixel, while the target pixels
      themselves remain black.

    Key insight: The same black/grey pair has two opposite meanings depending on enclosure: black inside grey
    is a defect to highlight, while grey inside black is noise to erase.
    """
    cleaned = _absorb_grey_noise(np.array(grid, dtype=int))
    preserve = _collect_targets(cleaned)
    return _paint_halo(cleaned, preserve)


# --------------- generator ---------------

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5

MIN_BLUE_REGION = 30
MIN_DEFECTS = 2
MIN_LEAVES = 1
MIN_NOISE = 2
MAX_GEN_ATTEMPTS = 500

BLACK_REGION_COUNT_RANGE = (1, 2)
BLACK_RECT_HEIGHT_RANGE = (4, 8)
BLACK_RECT_WIDTH_RANGE = (5, 8)
BLACK_REGION_TRIES = 150
MIN_BLACK_REGION_TOUCH = 3

DEFECT_COUNT_RANGE = (2, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60

LEAF_COUNT_RANGE = (1, 3)
LEAF_PLACE_TRIES = 40

NOISE_COUNT_RANGE = (2, 6)
SAFETY_MARGIN = 2


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a grey/blue partition analogous to the source task's black/blue layout."""
    grid = np.full((H, W), GREY, dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_count = min(n_cuts, H - 2 * CUT_MARGIN)
        cuts = sorted(random.sample(range(CUT_MARGIN, H - CUT_MARGIN), cut_count))
        boundaries = [0] + cuts + [H]
        first_blue = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            color = BLUE if (first_blue + i) % 2 == 1 else GREY
            grid[boundaries[i] : boundaries[i + 1], :] = color

    elif style == "rect_inset":
        grid[:, :] = GREY
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
        if (
            r1 - r0 > INNER_INSET_MIN_GAP
            and c1 - c0 > INNER_INSET_MIN_GAP
            and random.random() < INNER_INSET_PROB
        ):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = GREY

    else:
        grid[:, :] = GREY
        vc = random.randint(W // 3, 2 * W // 3)
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = BLUE
            grid[:hc, vc:] = BLUE
        else:
            grid[:, vc:] = BLUE
            grid[hc:, :vc] = BLUE

    return grid


def _find_interior_cells(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != color:
                continue
            if all(int(grid[r + dr, c + dc]) == color for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                coords.append((r, c))
    random.shuffle(coords)
    return coords


def _rect_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _place_large_black_regions(grid: np.ndarray) -> set[tuple[int, int]]:
    H, W = grid.shape
    desired = random.randint(*BLACK_REGION_COUNT_RANGE)
    placed_pixels: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for _ in range(desired):
        placed = False
        for _try in range(BLACK_REGION_TRIES):
            h = random.randint(*BLACK_RECT_HEIGHT_RANGE)
            w = random.randint(*BLACK_RECT_WIDTH_RANGE)
            if h >= H - 2 or w >= W - 2:
                continue
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            cells = _rect_cells(r0, c0, h, w)
            cell_set = set(cells)
            if any(int(grid[r, c]) != GREY or (r, c) in blocked for r, c in cells):
                continue

            touch_blue = 0
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W) or (rr, cc) in cell_set:
                        continue
                    if int(grid[rr, cc]) == BLUE:
                        touch_blue += 1
            if touch_blue < MIN_BLACK_REGION_TOUCH:
                continue

            for r, c in cells:
                grid[r, c] = BLACK
            placed_pixels.update(cells)
            blocked |= _exclusion_zone(cells, 1)
            placed = True
            break
        if not placed:
            break

    return placed_pixels


def _place_defects(grid: np.ndarray, interior_grey: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_grey)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_grey:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            seed_r, seed_c = random.choice(interior_grey)
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
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if int(grid[rr, cc]) != GREY or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) != GREY or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_grey[:] = [p for p in interior_grey if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    H, W = grid.shape
    target = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    for _ in range(target):
        black_comps = components_mask(grid == BLACK, connectivity=4)
        large_black = [comp for comp in black_comps if len(comp) > 3]
        if not large_black:
            break

        placed = False
        for _try in range(LEAF_PLACE_TRIES):
            comp_set = {(int(r), int(c)) for r, c in random.choice(large_black)}
            candidates: list[tuple[int, int]] = []
            for r, c in comp_set:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if int(grid[rr, cc]) != BLUE or (rr, cc) in blocked or (rr, cc) in leaf_pixels:
                        continue
                    black_n = 0
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = rr + dr2
                        nc = cc + dc2
                        if 0 <= nr < H and 0 <= nc < W:
                            if int(grid[nr, nc]) == BLACK or (nr, nc) in leaf_pixels:
                                black_n += 1
                    if black_n == 1:
                        candidates.append((rr, cc))

            random.shuffle(candidates)
            for lr, lc in candidates:
                if _nearby(lr, lc, DEFECT_CLEARANCE, blocked | leaf_pixels):
                    continue
                leaf_pixels.add((lr, lc))
                grid[lr, lc] = BLACK
                placed = True
                break
            if placed:
                break

    return leaf_pixels


def _find_noise_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if all(
                0 <= r + dr < H
                and 0 <= c + dc < W
                and int(grid[r + dr, c + dc]) == BLACK
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _count_grey_enclosed_small_black(grid: np.ndarray) -> int:
    count = 0
    for coords in components_mask(grid == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3 and _same_component_neighbors_ok(grid, pts, GREY):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    if not blue_comps:
        return None
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return None

    large_black_pixels = _place_large_black_regions(grid)
    if len(large_black_pixels) < 20:
        return None

    interior_grey = _find_interior_cells(grid, GREY)
    defect_pixels, defects_placed = _place_defects(grid, interior_grey)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    preserve = defect_pixels | leaf_pixels
    output = _paint_halo(grid, preserve)

    blocked = _exclusion_zone(preserve, SAFETY_MARGIN)
    noise_candidates = _find_noise_candidates(grid, blocked)
    target_noise = random.randint(*NOISE_COUNT_RANGE)
    noise_pixels: set[tuple[int, int]] = set()
    noise_blocked = set(blocked)
    for r, c in noise_candidates:
        if (r, c) in noise_blocked:
            continue
        noise_pixels.add((r, c))
        noise_blocked |= _exclusion_zone({(r, c)}, 1)
        if len(noise_pixels) >= target_noise:
            break
    if len(noise_pixels) < MIN_NOISE:
        return None

    inp = grid.copy()
    for r, c in noise_pixels:
        inp[r, c] = GREY

    try:
        solver_out = main(inp)
    except Exception:
        return None

    if not np.array_equal(solver_out, output):
        return None

    cleaned = _absorb_grey_noise(inp)
    if _count_grey_enclosed_small_black(cleaned) < MIN_DEFECTS:
        return None
    if len(noise_pixels) < MIN_NOISE:
        return None
    if not np.all(cleaned[tuple(np.array(list(noise_pixels)).T)] == BLACK):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
