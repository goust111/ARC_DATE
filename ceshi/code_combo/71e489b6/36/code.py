# Combo variant of 71e489b6 (index 36)
# combo_spec: D3/1+D8/0
# Name: 毛刺扩展+小网格

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask

# Design analysis:
# a) Behavioral trace:
#    Original flow: a blue spur pixel with >= 3 effective black 4-neighbors is
#    first absorbed into black, then black components are scanned; every size<=3
#    black component is marked, and leaf pixels of larger black components are
#    marked too. In the fused variant, that first step flips direction: the same
#    kind of blue spur remains blue and grows one orthogonal layer into adjacent
#    black cells. Example on a narrow black stripe:
#      base black stripe:  0 0 0 0 0
#      input with spur:    0 0 1 0 0
#      original STEP0 ->   0 0 0 0 0
#      fused STEP0  ->     0 1 1 1 0
#    On a 10-13 cell board, this can split a black stripe into multiple pieces
#    or erase a would-be leaf, so the later halo stage must run on the grown
#    black topology, not on the absorbed one.
# b) Complexity calibration:
#    The grid is smaller, but the rule should stay inferable. The generator
#    keeps 2-6 true black defects, 2-4 spur seeds, and at most a few leaves.
#    That preserves density without making every halo overlap every spur. At
#    least one chosen spur must change black connectivity/leaf structure, so the
#    D3/1 change is not reduced to a cosmetic recolor.
# c) Hidden assumptions audit:
#    The original in-place absorption loop could mutate topology sequentially
#    without issue because pixels only disappeared. That assumption fails here:
#    if growth were applied in-place, newly grown blue pixels could trigger an
#    unintended cascade. The solver therefore detects all spur seeds from the
#    original snapshot and applies growth simultaneously. The original generator
#    also assumed spurs may be placed near any black target because they vanish;
#    that is no longer safe, so spur footprints are kept away from existing
#    defect/leaf halos unless the intended topology change happens deeper inside
#    a large black region.
# d) Placement legality:
#    Defects are still placed only in strict blue interiors and kept separated so
#    they do not merge with each other or with large black regions. Leaf pixels
#    are only inserted on blue cells adjacent to exactly one black neighbor of a
#    large black component. Spur seeds are only chosen from black cells with
#    >= 3 effective black neighbors, and their one-step growth footprints must
#    stay disjoint, in-bounds, and away from existing targets.
# e) Visual observability:
#    A human can now spot two repeating motifs across examples: tiny black
#    anomalies get orange 3x3 halos, while tiny blue intrusions inside black
#    regions grow into plus/T-shaped blue patches before any halos are decided.
#    In the small grids this often creates new small black fragments or removes
#    black tips, making the growth step visibly relevant.
# f) Visual cleanliness:
#    The generator rejects crowded samples where spur growth sits underneath an
#    existing halo or where multiple spur footprints touch. This keeps the board
#    readable despite the denser 10-13 canvas.
# g) Inferability:
#    Only three colors are used and every change is structural. Orange still
#    means "black anomaly neighborhood"; blue growth always comes from blue
#    inputs embedded in black territory. There is no hidden dependence on color
#    ids beyond those visible spatial roles.

BLACK = 0
BLUE = 1
ORANGE = 7

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
MIN_BLUE_REGION = 14
MIN_BLACK_REGION = 14
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
PARTITION_WEIGHTS = [4, 1, 3]
STRIPE_CUTS_RANGE = (1, 2)
CUT_MARGIN = 2
INSET_MARGIN = 1
INNER_INSET_MIN_GAP = 5
INNER_INSET_PROB = 0.2
DEFECT_COUNT_RANGE = (2, 6)
DEFECT_SIZE_WEIGHTS = [5, 2, 1]
DEFECT_CLEARANCE = 1
DEFECT_PLACE_TRIES = 70
LEAF_COUNT_RANGE = (0, 2)
SPUR_COUNT_RANGE = (2, 4)
SPUR_CLEARANCE = 1
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_GROWTH_CELLS = 5
MIN_VARIANT_DIFF = 4
MIN_TOPOLOGY_SCORE = 1
MAX_PARTITION_ATTEMPTS = 80
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: set[tuple[int, int]] | list[tuple[int, int]],
    dist: int,
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors; corner out-of-bounds sides count as black."""
    H, W = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += int(oob)
    return black_n


def _detect_spur_seeds(grid: np.ndarray) -> set[tuple[int, int]]:
    """Blue cells that qualify for one-step growth into black territory."""
    H, W = grid.shape
    seeds: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLUE:
                continue
            if _effective_black_count(grid, r, c) >= 3:
                seeds.add((r, c))
    return seeds


def _spur_growth_footprint(base: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    """Cells turned blue by one seed, including the seed itself."""
    H, W = base.shape
    r, c = seed
    footprint = {(r, c)}
    for rr, cc in _neighbors4(r, c):
        if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
            footprint.add((rr, cc))
    return footprint


def _grow_spurs_once(
    grid: np.ndarray,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Apply the fused D3/1 rule once, simultaneously.

    Returns:
      grown_grid, spur_seeds, newly-grown blue cells
    """
    snapshot = np.array(grid, dtype=int)
    seeds = _detect_spur_seeds(snapshot)
    grown = snapshot.copy()
    growth: set[tuple[int, int]] = set()

    for seed in seeds:
        for rr, cc in _neighbors4(*seed):
            if 0 <= rr < grown.shape[0] and 0 <= cc < grown.shape[1] and int(snapshot[rr, cc]) == BLACK:
                grown[rr, cc] = BLUE
                growth.add((rr, cc))

    return grown, seeds, growth


def _collect_targets(
    base: np.ndarray,
) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]]]:
    """Collect tiny black components and leaf pixels of larger black components."""
    H, W = base.shape
    comps: list[set[tuple[int, int]]] = []
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        comps.append(pts)
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    black_n += 1
            if black_n <= 1:
                preserve.add((r, c))

    return comps, preserve


def _build_expected_output(base: np.ndarray) -> np.ndarray:
    """Paint the orange Chebyshev-1 halo around all target black pixels."""
    H, W = base.shape
    comps, preserve = _collect_targets(base)
    out = np.array(base, dtype=int)

    for pts in comps:
        targets = pts if len(pts) <= 3 else [p for p in pts if p in preserve]
        for r, c in targets:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                        out[rr, cc] = ORANGE

    for r, c in preserve:
        out[r, c] = BLACK

    return out


def _original_main(grid: np.ndarray) -> np.ndarray:
    """Original task solver, used only as a degeneration check in generation."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    base = g.copy()

    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_count(base, r, c) >= 3:
                base[r, c] = BLACK

    comps, preserve = _collect_targets(base)
    out = base.copy()
    for pts in comps:
        targets = pts if len(pts) <= 3 else [p for p in pts if p in preserve]
        for r, c in targets:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                        out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _black_metrics(grid: np.ndarray) -> dict[str, int]:
    comps, preserve = _collect_targets(grid)
    return {
        "components": len(comps),
        "small_components": sum(1 for pts in comps if len(pts) <= 3),
        "target_pixels": len(preserve),
    }


def _topology_score(before: np.ndarray, after: np.ndarray) -> int:
    mb = _black_metrics(before)
    ma = _black_metrics(after)
    score = 0
    score += abs(ma["components"] - mb["components"])
    score += abs(ma["small_components"] - mb["small_components"])
    score += abs(ma["target_pixels"] - mb["target_pixels"])
    return int(score)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether generated inputs actually exercise the fused behavior.

    Metrics:
    - spur_seeds: number of input blue spur seeds that qualify for growth
    - growth_cells: number of newly blue cells created by one-step growth
    - topology_score: change in black connectivity/target counts after growth
    - variant_vs_original_diff: output Hamming distance against original solver
    """
    grown, seeds, growth = _grow_spurs_once(grid)
    base = np.array(grid, dtype=int)
    for r, c in seeds:
        base[r, c] = BLACK
    variant_out = _build_expected_output(grown)
    original_out = _original_main(grid)
    return {
        "spur_seeds": len(seeds),
        "growth_cells": len(growth),
        "topology_score": _topology_score(base, grown),
        "variant_vs_original_diff": int(np.count_nonzero(variant_out != original_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies after letting compact blue spur pixels grow once.

    Input structure:
    - A compact 10-13 cell grid is partitioned into large blue (1) and black (0) regions.
    - Small black defects sit inside blue regions, and large black regions may have leaf-like black tips.
    - Some isolated blue spur pixels are embedded in black territory and qualify by having at least three effective black 4-neighbors.

    Rule:
    - First detect every qualifying blue spur pixel from the original snapshot and simultaneously expand it one orthogonal step into each adjacent black cell.
    - Then, on that grown grid, treat every black component of size <= 3 as a defect and treat every black pixel with <= 1 black 4-neighbor inside a larger black component as a leaf target.
    - Paint orange (7) over the Chebyshev-1 neighborhood of each target while keeping the target pixels themselves black.

    Key insight:
    - The marked black pixels are determined only after the blue spur growth has reshaped the black regions, so the puzzle couples a blue expansion step with the usual black-defect halo rule.
    """
    grown, _seeds, _growth = _grow_spurs_once(np.array(grid, dtype=int))
    return _build_expected_output(grown)


def _build_partition(H: int, W: int) -> np.ndarray | None:
    """Create a small alternating blue/black partition suited to the fused task."""
    for _ in range(MAX_PARTITION_ATTEMPTS):
        grid = np.zeros((H, W), dtype=int)
        style = random.choices(PARTITION_STYLES, weights=PARTITION_WEIGHTS, k=1)[0]

        if style == "h_stripes":
            n_cuts = random.randint(*STRIPE_CUTS_RANGE)
            cut_pool = list(range(CUT_MARGIN, H - CUT_MARGIN + 1))
            if len(cut_pool) < n_cuts:
                continue
            cuts = sorted(random.sample(cut_pool, n_cuts))
            boundaries = [0] + cuts + [H]
            if any(boundaries[i + 1] - boundaries[i] < 2 for i in range(len(boundaries) - 1)):
                continue
            first_color = random.randint(0, 1)
            for i in range(len(boundaries) - 1):
                grid[boundaries[i] : boundaries[i + 1], :] = (first_color + i) % 2

        elif style == "rect_inset":
            bg = random.randint(0, 1)
            grid[:, :] = bg
            r0 = random.randint(INSET_MARGIN, max(INSET_MARGIN, H // 3))
            r1 = random.randint(max(r0 + 3, 2 * H // 3), H - 2)
            c0 = random.randint(INSET_MARGIN, max(INSET_MARGIN, W // 3))
            c1 = random.randint(max(c0 + 3, 2 * W // 3), W - 2)
            grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
            if (
                r1 - r0 >= INNER_INSET_MIN_GAP
                and c1 - c0 >= INNER_INSET_MIN_GAP
                and random.random() < INNER_INSET_PROB
            ):
                ir0 = random.randint(r0 + 1, r0 + max(1, (r1 - r0) // 3))
                ir1 = random.randint(r0 + max(2, 2 * (r1 - r0) // 3), r1 - 1)
                ic0 = random.randint(c0 + 1, c0 + max(1, (c1 - c0) // 3))
                ic1 = random.randint(c0 + max(2, 2 * (c1 - c0) // 3), c1 - 1)
                if ir1 > ir0 and ic1 > ic0:
                    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

        else:  # l_shape
            bg = random.randint(0, 1)
            grid[:, :] = bg
            vc = random.randint(max(2, W // 3), min(W - 2, 2 * W // 3))
            hc = random.randint(max(2, H // 3), min(H - 2, 2 * H // 3))
            if random.random() < 0.5:
                grid[:, :vc] = 1 - bg
                grid[:hc, vc:] = 1 - bg
            else:
                grid[:, vc:] = 1 - bg
                grid[hc:, :vc] = 1 - bg

        blue_comps = list(components_mask(grid == BLUE, connectivity=4))
        black_comps = list(components_mask(grid == BLACK, connectivity=4))
        if not blue_comps or not black_comps:
            continue
        if min(len(comp) for comp in blue_comps) < MIN_BLUE_REGION:
            continue
        if min(len(comp) for comp in black_comps) < MIN_BLACK_REGION:
            continue
        return grid

    return None


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue pixels whose in-bounds 4-neighbors are all blue."""
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


def _place_defects(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    """Place 2-6 small black defects strictly inside blue interiors."""
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS, k=1)[0]

        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed[0], seed[1], DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [seed]
            if defect_size > 1:
                dirs = _neighbors4(*seed)
                random.shuffle(dirs)
                for nr, nc in dirs:
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for rr, cc in _neighbors4(nr, nc):
                        if (rr, cc) == seed:
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
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
                for rr, cc in _neighbors4(r, c):
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
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
    """Insert a few black leaf protrusions extending from large black regions."""
    H, W = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    large_black = [
        {(int(r), int(c)) for r, c in coords}
        for coords in components_mask(grid == BLACK, connectivity=4)
        if len(coords) > 3
    ]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp = random.choice(large_black)
        candidates: list[tuple[int, int]] = []

        for r, c in comp:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W:
                        if int(grid[ar, ac]) == BLACK or (ar, ac) in leaf_pixels:
                            black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))

        if not candidates:
            continue

        lr, lc = random.choice(candidates)
        if _nearby(lr, lc, 1, defect_pixels):
            continue
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK

    return leaf_pixels


def _select_spur_seeds(
    base: np.ndarray,
    blocked: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Derive the input by turning selected black cells into blue spur seeds.

    Returns:
      input_grid, grown_grid
    """
    H, W = base.shape
    base_metrics = _black_metrics(base)
    candidate_items: list[tuple[int, int, tuple[int, int], set[tuple[int, int]]]] = []

    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLACK:
                continue
            if _effective_black_count(base, r, c) < 3:
                continue
            seed = (r, c)
            footprint = _spur_growth_footprint(base, seed)
            if footprint & blocked:
                continue

            grown = base.copy()
            grown[r, c] = BLUE
            for rr, cc in footprint:
                grown[rr, cc] = BLUE

            after_metrics = _black_metrics(grown)
            topology_changed = (
                after_metrics["components"] != base_metrics["components"]
                or after_metrics["small_components"] != base_metrics["small_components"]
                or after_metrics["target_pixels"] != base_metrics["target_pixels"]
            )
            score = len(footprint) - 1
            if topology_changed:
                score += 8
            score += abs(after_metrics["components"] - base_metrics["components"])
            score += abs(after_metrics["small_components"] - base_metrics["small_components"])
            score += abs(after_metrics["target_pixels"] - base_metrics["target_pixels"])
            candidate_items.append((score, int(topology_changed), seed, footprint))

    if not candidate_items:
        return None

    random.shuffle(candidate_items)
    candidate_items.sort(key=lambda item: (item[0], item[1]), reverse=True)

    target_n = random.randint(*SPUR_COUNT_RANGE)
    chosen: list[tuple[int, int]] = []
    used_zone: set[tuple[int, int]] = set()

    for _score, _topology, seed, footprint in candidate_items:
        if footprint & used_zone:
            continue
        chosen.append(seed)
        used_zone |= _exclusion_zone(list(footprint), SPUR_CLEARANCE)
        if len(chosen) >= target_n:
            break

    if len(chosen) < MIN_SPURS:
        return None

    inp = np.array(base, dtype=int)
    for r, c in chosen:
        inp[r, c] = BLUE

    detected = _detect_spur_seeds(inp)
    if detected != set(chosen):
        return None

    grown, grown_seeds, _growth = _grow_spurs_once(inp)
    if grown_seeds != set(chosen):
        return None

    metrics = _fusion_metrics(inp)
    if metrics["spur_seeds"] < MIN_SPURS:
        return None
    if metrics["growth_cells"] < MIN_GROWTH_CELLS:
        return None
    if metrics["topology_score"] < MIN_TOPOLOGY_SCORE:
        return None
    if metrics["variant_vs_original_diff"] < MIN_VARIANT_DIFF:
        return None

    return inp, grown


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    grid = _build_partition(H, W)
    if grid is None:
        return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    blocked = _exclusion_zone(defect_pixels | leaf_pixels, 1)

    selected = _select_spur_seeds(grid, blocked)
    if selected is None:
        return None
    inp, grown = selected

    expected = _build_expected_output(grown)
    try:
        if np.array_equal(main(inp), expected):
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
