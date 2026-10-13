from __future__ import annotations
import numpy as np
from arc_utils import components_mask
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight black defects in blue foreground by surrounding them with orange.

    Input structure:
    - Blue (1) foreground on black (0) bg; small black holes or protrusions within blue regions; occasional thin blue spurs into black areas.

    Rule:
    - Thin blue pixels mostly surrounded by black are absorbed into the background.
    - Small black components (≤3 cells) and leaf pixels of larger black components are "defects."
    - Each defect gets an orange (7) 8-neighbor halo; the defect pixels themselves stay black.

    Key insight: Only structural anomalies—tiny isolated patches and dangling endpoints—are marked, not bulk black regions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    hole_col = 0
    border_col = 7

    # STEP 0) Fill blue (1) pixels to black (0) when mostly surrounded by black.
    base = g.copy()
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != 1:
                continue
            nb = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == int(hole_col):
                    nb += 1
            # only count outside-as-black for corners (two out-of-bounds sides)
            if oob >= 2:
                nb += int(oob)
            if nb >= 3:
                base[int(r), int(c)] = int(hole_col)

    # STEP 1/2) Collect target black pixels (they must remain black).
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == int(hole_col), connectivity=4)
    for coords in comps:
        pts_list = [(int(r), int(c)) for r, c in coords]
        if len(pts_list) <= 3:
            preserve.update(pts_list)
            continue
        for r, c in pts_list:
            nb = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == int(hole_col):
                    nb += 1
            if nb <= 1:
                preserve.add((int(r), int(c)))

    # STEP 3) Paint orange in the Chebyshev-1 neighborhood of each target set/pixel.
    out = base.copy()
    for coords in comps:
        pts_list = [(int(r), int(c)) for r, c in coords]
        targets = pts_list if len(pts_list) <= 3 else [p for p in pts_list if p in preserve]
        for r, c in targets:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                        out[rr, cc] = int(border_col)

    # Ensure target black pixels remain black.
    for r, c in preserve:
        out[int(r), int(c)] = int(hole_col)

    return out

# --------------- generator (att1) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H_RANGE = (16, 19)           # canvas height bounds
GRID_W_RANGE = (16, 19)           # canvas width bounds
MIN_BLUE_REGION = 30              # minimum blue region size (cells)
MIN_BLACK_REGION = 20             # minimum black region size (cells)
PARTITION_STYLES = ['h_stripes', 'rect_inset', 'l_shape']
STRIPE_CUTS_RANGE = (1, 3)       # number of horizontal cuts for h_stripes
CUT_MARGIN = 3                   # min distance from grid edge for cuts
INSET_MARGIN = 2                 # padding from edge for rect_inset
INNER_INSET_MIN_GAP = 6          # min dimension for nested inset
INNER_INSET_PROB = 0.5           # probability of nested inset
DEFECT_COUNT_RANGE = (2, 9)      # number of defect holes to place
DEFECT_SIZE_WEIGHTS = [6, 2, 1]  # weights for defect sizes [1, 2, 3]
DEFECT_CLEARANCE = 2             # min Chebyshev distance between defects
DEFECT_PLACE_TRIES = 50          # retries per defect placement
LEAF_COUNT_RANGE = (0, 4)        # number of leaf protrusions
SPUR_COUNT_RANGE = (2, 6)        # number of blue spurs in black regions
MIN_DEFECTS = 2                  # reject grid if fewer defects placed
MIN_SPURS = 2                    # reject grid if fewer spurs placed
MAX_GEN_ATTEMPTS = 300           # outer retry limit


def _nearby(r: int, c: int, dist: int, ref: set) -> bool:
    """Return True if any pixel in `ref` is within Chebyshev distance `dist`."""
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels, dist: int) -> set:
    """Return the set of all coords within Chebyshev `dist` of any pixel."""
    zone = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an H x W grid partitioned into blue (1) and black (0) regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == 'h_stripes':
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(random.sample(
            range(CUT_MARGIN, H - CUT_MARGIN), min(n_cuts, H - 2 * CUT_MARGIN)))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i]:boundaries[i + 1], :] = (first_color + i) % 2

    elif style == 'rect_inset':
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0:r1 + 1, c0:c1 + 1] = 1 - bg
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP
                and random.random() < INNER_INSET_PROB):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0:ir1 + 1, ic0:ic1 + 1] = bg

    elif style == 'l_shape':
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(W // 3, 2 * W // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue pixels whose 4-neighbors are all blue or out-of-bounds."""
    H, W = grid.shape
    blue = grid == 1
    # A blue pixel is interior if no in-bounds 4-neighbor is non-blue.
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]   # neighbor below
    eroded[1:, :]  &= blue[:-1, :]  # neighbor above
    eroded[:, :-1] &= blue[:, 1:]   # neighbor right
    eroded[:, 1:]  &= blue[:, :-1]  # neighbor left
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_defects(grid: np.ndarray, interior_blue: list) -> tuple[set, int]:
    """Place 2-9 small black defects in blue-region interiors.

    Returns (defect_pixels, defects_placed). Modifies `interior_blue` in place.
    """
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]

        placed = False
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]

            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            # Grow defect from seed
            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = seed_r + dr, seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    # Check neighbor is surrounded by blue (except the seed)
                    ok = True
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr, cc = nr + dr2, nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if grid[rr, cc] != 1 or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            # Validate: every non-defect 4-neighbor of every defect pixel must be blue
            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if grid[rr, cc] != 1 or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                placed = True
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set) -> set:
    """Place 0-4 leaf protrusions from large black regions into blue regions."""
    H, W = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == 0, connectivity=4)
    large_black = [c for c in black_comps if len(c) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = set((int(r), int(c)) for r, c in random.choice(large_black))

        # Find blue pixels adjacent to this component that would be valid leaves
        candidates = []
        for r, c in comp_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or grid[rr, cc] != 1:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                # Count black 4-neighbors of the candidate pixel
                black_n = sum(
                    1 for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= rr + dr2 < H and 0 <= cc + dc2 < W
                    and (grid[rr + dr2, cc + dc2] == 0 or (rr + dr2, cc + dc2) in leaf_pixels))
                if black_n == 1:
                    candidates.append((rr, cc))

        if candidates:
            lr, lc = random.choice(candidates)
            if not _nearby(lr, lc, DEFECT_CLEARANCE, defect_pixels):
                leaf_pixels.add((lr, lc))
                grid[lr, lc] = 0

    return leaf_pixels


def _build_halo(grid: np.ndarray, preserve: set) -> np.ndarray:
    """Paint orange (7) Chebyshev-1 halo around preserve pixels; keep them black."""
    H, W = grid.shape
    output = grid.copy()
    for r, c in preserve:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    output[rr, cc] = 7
    for r, c in preserve:
        output[r, c] = 0
    return output


def _find_spur_candidates(grid: np.ndarray, exclude: set) -> list[tuple[int, int]]:
    """Find black pixels eligible to become blue spurs (solver's STEP 0 absorbs them).

    A pixel qualifies if it has >= 3 effective black 4-neighbors, matching the
    solver's absorption logic (OOB counts as black only at corners with >= 2 OOB sides).
    """
    H, W = grid.shape
    black = grid == 0

    # Count in-bounds black 4-neighbors via shifts
    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :]  += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:]  += black[:, :-1]

    # Count out-of-bounds sides
    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1; oob[-1, :] += 1; oob[:, 0] += 1; oob[:, -1] += 1

    # Effective count: add oob only when >= 2 (corner pixels)
    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    # Eligible: black, effective >= 3, not in exclude set
    rs, cs = np.where(black & (effective >= 3))
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs)
                  if (r, c) not in exclude]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for task 71e489b6."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    # Step 1: Build region partition
    grid = _build_partition(H, W)

    # Validate region sizes
    for comp in components_mask(grid == 1, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == 0, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    # Step 2: Place defects
    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = 0

    # Step 3: Place leaves
    leaf_pixels = _place_leaves(grid, defect_pixels)

    # Step 4: Build expected output
    preserve = defect_pixels | leaf_pixels
    output = _build_halo(grid, preserve)

    # Step 5: Add spurs to derive input
    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, defect_pixels | leaf_pixels)
    for r, c in candidates[:n_spurs]:
        inp[r, c] = 1
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None

    # Step 6: Oracle check
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
