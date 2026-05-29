# Combo variant of 71e489b6 (index 3)
# combo_spec: D1/0+D7/0
# Name: 十字光晕+光晕限蓝色区域
from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for D1/0 + D7/0
# a) Concrete trace:
#    - Original around a 1-cell defect inside blue:
#        111      777
#        101  ->  707
#        111      777
#    - Fused rule on the same local patch:
#        111      171
#        101  ->  707
#        111      171
#      Only orthogonal neighbors are painted, so diagonals stay blue.
#    - For a leaf that sticks out of a large black region:
#        000      070
#        001  ->  007   (original 3x3 halo would also flood black cells)
#        111      171
#      Under the combo, the halo uses the same cross footprint but can only land on
#      blue support cells. The arm pointing back into the black region disappears,
#      and boundary defects may lose extra arms at the grid edge.
# b) Complexity calibration:
#    - Detection stays identical to the source puzzle: absorb blue spurs, then keep
#      tiny black components plus leaf pixels of large black components.
#    - Only two discrete switches change: square -> cross, and unrestricted paint ->
#      blue-only paint. That keeps the rule learnable while producing a visible
#      difference from both the base task and the single-dimension variants.
# c) Hidden assumptions review:
#    - STEP 0 still assumes removable noise is a blue pixel with >=3 effective black
#      4-neighbors; the generator only injects spurs that satisfy that same local test.
#    - STEP 1/2 still assumes true targets are black after cleanup; generator places
#      all defects/leaves before spur injection so cleanup cannot destroy them.
#    - STEP 3 now assumes a paintable halo arm must land on a blue cell in the cleaned
#      grid. The generator therefore guarantees every target has at least one blue
#      orthogonal neighbor, and at least one target per sample has a clipped arm.
# d) Placement legality:
#    - Small defects are placed on blue cells whose in-bounds 4-neighbors are blue,
#      so they remain isolated black components of size <=3 after conversion.
#    - Leaf candidates are blue cells adjacent to exactly one black neighbor, so once
#      flipped they become degree-1 tips of a large black component.
#    - Defects and leaves use clearance checks to avoid accidental merging/overlap.
# e/f/g) Visual observability and cleanliness:
#    - Every sample has several interior defects, making the cross-vs-square change
#      obvious via missing diagonals.
#    - Every accepted sample also has at least one leaf, making the blue-only clipping
#      observable because one cross arm is cut off by a neighboring black region.
#    - The generator further asks for at least one strongly clipped target (<=2 blue
#      arms) so examples show irregular edge marks instead of only perfect 4-arm crosses.


BLUE = 1
BLACK = 0
ORANGE = 7
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, treating OOB as black only for corners."""
    h, w = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in ORTHO_DIRS:
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
    """Absorb blue spur pixels using the original in-place sweep rule."""
    base = np.array(grid, dtype=int)
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_count(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _collect_preserve(base: np.ndarray) -> set[tuple[int, int]]:
    """Return black pixels that must remain black in the output."""
    h, w = base.shape
    preserve: set[tuple[int, int]] = set()
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = 0
            for dr, dc in ORTHO_DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                    deg += 1
            if deg <= 1:
                preserve.add((r, c))
    return preserve


def _build_output(
    base: np.ndarray,
    preserve: set[tuple[int, int]],
    *,
    shape: str = "cross",
    blue_only: bool = True,
) -> np.ndarray:
    """Paint a halo variant around preserve pixels and keep centers black."""
    if shape == "cross":
        offsets = ORTHO_DIRS
    elif shape == "square":
        offsets = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    else:
        raise ValueError(f"unknown halo shape: {shape}")

    h, w = base.shape
    out = np.array(base, dtype=int)
    for r, c in preserve:
        for dr, dc in offsets:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in preserve:
                continue
            if blue_only and int(base[rr, cc]) != BLUE:
                continue
            out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _count_blue_arms(grid: np.ndarray, cell: tuple[int, int]) -> int:
    """How many orthogonal halo arms would survive the blue-only clipping."""
    h, w = grid.shape
    r, c = cell
    arms = 0
    for dr, dc in ORTHO_DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLUE:
            arms += 1
    return arms


def _find_clipped_targets(grid: np.ndarray, preserve: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Targets whose clipped cross has 1-2 visible arms."""
    return [pt for pt in preserve if 1 <= _count_blue_arms(grid, pt) <= 2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects inside blue structure with orange crosses that stay inside blue.

    Input structure:
    - The canvas contains large blue (1) and black (0) regions, with small black defects embedded in blue.
    - Large black regions may also have thin leaf-like black tips poking into blue.
    - Some blue spur pixels sit inside black regions and should be absorbed before marking.

    Rule:
    - First convert any blue pixel with at least three effective black 4-neighbors into black.
    - Then keep all black components of size <= 3, plus degree-0/1 pixels from larger black components, as defect centers.
    - For each center, paint orange (7) only on the four orthogonal neighbors that are still blue after cleanup; the center itself stays black.

    Key insight:
    - Detection is unchanged from the source task, but the marking is stricter twice over: diagonals are never painted, and black/background cells cannot receive halo paint, so edge defects produce cropped crosses.
    """
    base = _clean_spurs(grid)
    preserve = _collect_preserve(base)
    return _build_output(base, preserve, shape="cross", blue_only=True)


# --------------- generator ---------------

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
INTERIOR_DEFECT_COUNT_RANGE = (2, 7)
EDGE_DEFECT_COUNT_RANGE = (0, 2)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 4)
MIN_LEAVES = 1
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 2
MIN_SPURS = 2
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
    """Create a large-scale blue/black partition similar to the source task."""
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
    """Blue cells whose in-bounds orthogonal neighbors are all blue."""
    h, w = grid.shape
    coords = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLUE:
                continue
            ok = True
            for dr, dc in ORTHO_DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLUE:
                    ok = False
                    break
            if ok:
                coords.append((r, c))
    random.shuffle(coords)
    return coords


def _find_corner_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue corner cells that can host strongly clipped 1-cell defects."""
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    out = []
    for r, c in corners:
        if int(grid[r, c]) != BLUE:
            continue
        ok = True
        for dr, dc in ORTHO_DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLUE:
                ok = False
                break
        if ok:
            out.append((r, c))
    random.shuffle(out)
    return out


def _place_edge_defects(grid: np.ndarray) -> set[tuple[int, int]]:
    """Prefer 1-cell corner defects so some crosses are visibly truncated to 2 arms."""
    defect_pixels: set[tuple[int, int]] = set()
    candidates = _find_corner_blue(grid)
    target_n = random.randint(*EDGE_DEFECT_COUNT_RANGE)
    for r, c in candidates:
        if len(defect_pixels) >= target_n:
            break
        if _nearby(r, c, DEFECT_CLEARANCE, defect_pixels):
            continue
        defect_pixels.add((r, c))
    return defect_pixels


def _place_interior_defects(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    """Place small black defects inside blue interiors."""
    h, w = grid.shape
    n_defects = random.randint(*INTERIOR_DEFECT_COUNT_RANGE)
    defect_pixels = set(reserved)
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS, k=1)[0]

        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if (seed_r, seed_c) in defect_pixels or _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = list(ORTHO_DIRS)
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    if _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        continue
                    ok = True
                    for dr2, dc2 in ORTHO_DIRS:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLUE:
                            ok = False
                            break
                    if ok:
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in ORTHO_DIRS:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w and (
                        int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels
                    ):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels - reserved, defects_placed


def _place_leaves(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place degree-1 black tips, preferring heavily clipped locations."""
    h, w = grid.shape
    desired = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]
    random.shuffle(large_black)

    for comp_set in large_black:
        if len(leaf_pixels) >= desired:
            break
        candidates: list[tuple[int, int, int]] = []
        for r, c in comp_set:
            for dr, dc in ORTHO_DIRS:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(grid[rr, cc]) != BLUE or (rr, cc) in blocked or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                blue_arms = 0
                for dr2, dc2 in ORTHO_DIRS:
                    r2 = rr + dr2
                    c2 = cc + dc2
                    if 0 <= r2 < h and 0 <= c2 < w:
                        if int(grid[r2, c2]) == BLACK or (r2, c2) in leaf_pixels:
                            black_n += 1
                        elif int(grid[r2, c2]) == BLUE:
                            blue_arms += 1
                if black_n == 1 and blue_arms >= 1 and not _nearby(rr, cc, DEFECT_CLEARANCE, blocked | leaf_pixels):
                    candidates.append((blue_arms, rr, cc))
        candidates.sort(key=lambda item: (item[0], random.random()))
        for _, lr, lc in candidates:
            if len(leaf_pixels) >= desired:
                break
            if (lr, lc) in blocked or (lr, lc) in leaf_pixels:
                continue
            leaf_pixels.add((lr, lc))
            grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels that the cleanup step will absorb back from blue to black."""
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or (r, c) in exclude:
                continue
            if _effective_black_count(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _region_sizes_ok(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    return (
        all(len(comp) >= MIN_BLUE_REGION for comp in blue_comps)
        and all(len(comp) >= MIN_BLACK_REGION for comp in black_comps)
    )


def _fusion_metrics(clean_grid: np.ndarray, preserve: set[tuple[int, int]]) -> dict[str, int]:
    """Quantify whether both dimensions of the combo are active on this sample."""
    combo = _build_output(clean_grid, preserve, shape="cross", blue_only=True)
    d1_only = _build_output(clean_grid, preserve, shape="square", blue_only=True)
    d7_only = _build_output(clean_grid, preserve, shape="cross", blue_only=False)
    clipped_targets = _find_clipped_targets(clean_grid, preserve)
    return {
        "d1_diff": int(np.count_nonzero(combo != d1_only)),
        "d7_diff": int(np.count_nonzero(combo != d7_only)),
        "leaf_like_clipped": len(clipped_targets),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    clean = _build_partition(h, w)
    if not _region_sizes_ok(clean):
        return None

    edge_defects = _place_edge_defects(clean)
    for r, c in edge_defects:
        clean[r, c] = BLACK

    interior_blue = _find_interior_blue(clean)
    defect_pixels, defects_placed = _place_interior_defects(clean, interior_blue, edge_defects)
    if defects_placed + len(edge_defects) < MIN_DEFECTS:
        return None
    all_defects = edge_defects | defect_pixels
    for r, c in defect_pixels:
        clean[r, c] = BLACK

    leaf_pixels = _place_leaves(clean, all_defects)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    preserve = all_defects | leaf_pixels
    output = _build_output(clean, preserve, shape="cross", blue_only=True)

    metrics = _fusion_metrics(clean, preserve)
    # Demand multiple black-blocked halo arms so the blue-only clipping is obvious
    # from a small set of examples instead of degenerating into a single-pixel hint.
    if metrics["d1_diff"] <= 0 or metrics["d7_diff"] < 2 or metrics["leaf_like_clipped"] <= 0:
        return None

    inp = clean.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    spur_candidates = _find_spur_candidates(inp, preserve)
    placed_spurs = min(n_spurs, len(spur_candidates))
    if placed_spurs < MIN_SPURS:
        return None
    for r, c in spur_candidates[:placed_spurs]:
        inp[r, c] = BLUE

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
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
