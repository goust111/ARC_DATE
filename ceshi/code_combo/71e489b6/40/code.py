# Combo variant of 71e489b6 (index 40)
# combo_spec: D3/0+D9/1
# Name: 毛刺品红标记+多色干扰

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


# Design analysis for combo D3/0 + D9/1
# a) Concrete behavior trace
#    Original pipeline:
#      1) blue spur in a black area with >=3 effective black 4-neighbors is absorbed to black;
#      2) black components are analyzed on that cleaned grid;
#      3) small black components and leaf pixels get orange halos.
#    Fused pipeline:
#      1) apply the same local "mostly black around me" test to every non-black pixel in black
#         areas;
#      2) blue matches are remembered as spur markers, but are still treated as black for the
#         structural analysis;
#      3) non-blue/non-black matches are simply deleted to black;
#      4) after orange halos are painted, remembered blue spurs are restored as magenta.
#    Example sketch:
#      input:   0 0 0 0      output:  0 0 0 0
#               0 1 0 3 0              0 6 0 0 0
#               0 0 0 0 0              0 0 0 0 0
#      The blue cell behaves like an old spur structurally, but now stays visible as magenta.
#      The green/red/etc. singleton is absorbed and disappears.
# b) Complexity calibration
#    If every disturbance were recolored instead of split by type, the fusion would collapse into a
#    single generic-noise rule. If we allowed many auxiliary colors or long colored chains, the task
#    would become cluttered and underdetermined. We keep one special retained class (blue -> magenta)
#    and a small palette of disappearing non-blue noise colors, with several repeated examples per
#    sample so both behaviors are visible.
# c) Hidden assumptions audit
#    - Original STEP 0 assumed any absorbed blue spur can be overwritten to black before component
#      detection. That still holds here, but only internally: `main()` records blue spur positions
#      before turning them black in the work grid.
#    - Component detection assumes all true defects are black. The generator therefore never inserts
#      colored noise into blue regions and keeps black-region disturbances separated from defects.
#    - Halo painting assumes target pixels remain black. Magenta spur pixels are not targets, so
#      they are restored only after halo painting.
# d) Placement legality
#    Defects are placed only on blue interior cells whose non-defect 4-neighbors remain blue.
#    Leaves are placed only on blue cells adjacent to a large black component with exactly one black
#    4-neighbor, so they become genuine leaf pixels. Disturbances are inserted only at non-border
#    black cells that satisfy the solver's local absorption predicate and are kept 4-separated from
#    each other and away from defect halos.
# e) Visual observability
#    Across a few samples a human can see two repeated patterns in black regions: blue singleton
#    specks become magenta in place, while red/green/yellow-like singleton specks vanish. Orange
#    still marks the black defects, so the new behavior is visible without relying on color ids as
#    numbers or on hidden ordering.
# f) Visual cleanliness
#    The output palette stays small: black, blue, orange, magenta. Extra input-only colors are
#    limited to a short list and are excluded from orange halo zones, so the solver's two marking
#    mechanisms do not visually collide.
# g) Inferability and degeneration check
#    The generator rejects samples unless they contain multiple blue spurs, multiple non-blue noise
#    pixels, at least two distinct disappearing noise colors, and at least one large-component leaf.
#    That prevents regression to the source task where only orange matters or where one defect branch
#    never appears.
# h) New solver assumptions vs generator guarantees
#    `main()` now assumes any non-blue color it absorbs is isolated black-region noise rather than a
#    semantic foreground color. `generate_input()` guarantees this by inserting those colors only at
#    vetted black-noise coordinates and never inside blue regions or inside the defect preserve set.


BLACK = 0
BLUE = 1
MAGENTA = 6
ORANGE = 7
NOISE_COLORS = (2, 3, 4)

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
LEAF_COUNT_RANGE = (1, 4)
BLUE_SPUR_COUNT_RANGE = (2, 5)
NOISE_COUNT_RANGE = (2, 5)
MIN_DEFECTS = 2
MIN_LEAVES = 1
MIN_BLUE_SPURS = 2
MIN_NOISE = 2
MIN_NOISE_COLOR_KINDS = 2
MAX_GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return int(black_n)


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


def _is_absorbable_noise(grid: np.ndarray, r: int, c: int) -> bool:
    color = int(grid[r, c])
    return color != BLACK and _effective_black_neighbors(grid, r, c) >= 3


def _collect_defect_targets(base: np.ndarray) -> set[tuple[int, int]]:
    h, w = base.shape
    preserve: set[tuple[int, int]] = set()
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = sum((rr, cc) in comp_set for rr, cc in _neighbors4(r, c) if 0 <= rr < h and 0 <= cc < w)
            if deg <= 1:
                preserve.add((r, c))
    return preserve


def _paint_orange_halo(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _halo_zone(preserve: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _exclusion_zone(preserve, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects with orange halos while preserving blue spurs as magenta and erasing other colored noise.

    Input structure:
    - Large blue (1) and black (0) regions are partitioned much like the source task.
    - Small black defects still live inside blue regions, and leaf pixels can still protrude from large black masses.
    - Black regions may also contain disturbance singletons: blue spur pixels and other non-blue colors.

    Rule:
    - Any non-black pixel with at least three effective black 4-neighbors is treated as black-region noise.
    - Blue noise pixels are remembered and shown as magenta (6) in the final output, while all other such colors are absorbed into black.
    - On the internally cleaned black/blue structure, mark each size<=3 black component and each leaf pixel of a larger black component with an orange (7) Chebyshev-1 halo, keeping the defect pixels themselves black.

    Key insight: Two local rules coexist on the same black regions: orange reveals structural black defects, while magenta singles out the specific noise class that is blue instead of simply deleting it.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    base = g.copy()
    spur_marks: set[tuple[int, int]] = set()

    # First reinterpret all locally absorbable non-black pixels as black-region noise.
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) == BLACK:
                continue
            if not _is_absorbable_noise(base, r, c):
                continue
            if int(base[r, c]) == BLUE:
                spur_marks.add((r, c))
            base[r, c] = BLACK

    preserve = _collect_defect_targets(base)
    out = _paint_orange_halo(base, preserve)

    # Spur marks are a second visible annotation layer and take precedence over halo color.
    for r, c in spur_marks:
        out[r, c] = MAGENTA

    return out


def _build_partition(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        pool = list(range(CUT_MARGIN, h - CUT_MARGIN))
        if not pool:
            return grid
        cuts = sorted(random.sample(pool, min(n_cuts, len(pool))))
        bounds = [0] + cuts + [h]
        first_color = random.randint(0, 1)
        for idx in range(len(bounds) - 1):
            grid[bounds[idx] : bounds[idx + 1], :] = (first_color + idx) % 2

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
    blue = grid == BLUE
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = [(int(r), int(c)) for r, c in zip(rs, cs)]
    random.shuffle(coords)
    return coords


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
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
            seed_r, seed_c = random.choice(interior_blue)
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
                    for dr2, dc2 in _neighbors4(nr, nc):
                        rr = dr2
                        cc = dc2
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
                for rr, cc in _neighbors4(r, c):
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w and (grid[rr, cc] != BLUE or (rr, cc) in defect_pixels):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            defect_pixels.update(defect)
            defects_placed += 1
            remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [comp for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = {(int(r), int(c)) for r, c in random.choice(large_black)}
        candidates: list[tuple[int, int]] = []
        for r, c in comp_set:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w) or grid[rr, cc] != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (grid[nr, nc] == BLACK or (nr, nc) in leaf_pixels):
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
        if not candidates:
            continue
        lr, lc = random.choice(candidates)
        if _nearby(lr, lc, DEFECT_CLEARANCE, defect_pixels):
            continue
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK

    return leaf_pixels


def _disturbance_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    rs, cs = np.where(grid == BLACK)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        pt = (int(r), int(c))
        if pt in blocked:
            continue
        # Keep disturbances away from the frame so the local "black-region noise" pattern is
        # visually obvious rather than hinging on the corner/out-of-bounds clause.
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            continue
        if _effective_black_neighbors(grid, pt[0], pt[1]) >= 3:
            candidates.append(pt)
    random.shuffle(candidates)
    return candidates


def _select_nonadjacent(candidates: list[tuple[int, int]], count: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    forbidden = set(blocked)
    for r, c in candidates:
        if (r, c) in forbidden:
            continue
        if any((nr, nc) in forbidden for nr, nc in _neighbors4(r, c)):
            continue
        chosen.append((r, c))
        forbidden.add((r, c))
        for nr, nc in _neighbors4(r, c):
            forbidden.add((nr, nc))
        if len(chosen) >= count:
            break
    return chosen


def _assign_noise_colors(n: int) -> list[int]:
    colors = list(NOISE_COLORS)
    random.shuffle(colors)
    assigned = [colors[0], colors[1]]
    while len(assigned) < n:
        assigned.append(random.choice(colors))
    random.shuffle(assigned)
    return assigned


def _build_expected_output(
    clean_grid: np.ndarray,
    preserve: set[tuple[int, int]],
    blue_spurs: set[tuple[int, int]],
) -> np.ndarray:
    out = _paint_orange_halo(clean_grid, preserve)
    for r, c in blue_spurs:
        out[r, c] = MAGENTA
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    clean = _build_partition(h, w)
    for comp in components_mask(clean == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(clean == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(clean)
    defect_pixels, defects_placed = _place_defects(clean, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        clean[r, c] = BLACK

    leaf_pixels = _place_leaves(clean, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None
    preserve = defect_pixels | leaf_pixels
    halo_blocked = _halo_zone(preserve) | preserve

    total_needed = random.randint(*BLUE_SPUR_COUNT_RANGE) + random.randint(*NOISE_COUNT_RANGE)
    candidates = _disturbance_candidates(clean, halo_blocked)
    chosen = _select_nonadjacent(candidates, total_needed, set())
    if len(chosen) < MIN_BLUE_SPURS + MIN_NOISE:
        return None

    blue_count = min(random.randint(*BLUE_SPUR_COUNT_RANGE), len(chosen) - MIN_NOISE)
    blue_count = max(blue_count, MIN_BLUE_SPURS)
    noise_count = min(random.randint(*NOISE_COUNT_RANGE), len(chosen) - blue_count)
    if noise_count < MIN_NOISE:
        noise_count = MIN_NOISE
        blue_count = len(chosen) - noise_count
    if blue_count < MIN_BLUE_SPURS or noise_count < MIN_NOISE:
        return None

    blue_spurs = set(chosen[:blue_count])
    noise_positions = chosen[blue_count : blue_count + noise_count]
    if len(noise_positions) < MIN_NOISE:
        return None

    input_grid = clean.copy()
    for r, c in blue_spurs:
        input_grid[r, c] = BLUE

    noise_colors = _assign_noise_colors(len(noise_positions))
    if len(set(noise_colors)) < MIN_NOISE_COLOR_KINDS:
        return None
    for (r, c), color in zip(noise_positions, noise_colors):
        input_grid[r, c] = color

    output_grid = _build_expected_output(clean, preserve, blue_spurs)

    # Quantitative anti-degeneration guards: both fused behaviors must be visibly present.
    if int(np.count_nonzero(output_grid == MAGENTA)) < MIN_BLUE_SPURS:
        return None
    if len({int(input_grid[r, c]) for r, c in noise_positions}) < MIN_NOISE_COLOR_KINDS:
        return None

    try:
        if np.array_equal(main(input_grid), output_grid):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
