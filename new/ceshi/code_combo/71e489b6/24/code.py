# Combo variant of 71e489b6 (index 24)
# combo_spec: D5/0+D7/0+D12/1
# Name: 类型分色+限蓝色区域+红色中心

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
#    Original task: absorb blue spur pixels inside black regions, then find two
#    kinds of black anomalies in the cleaned grid: (1) small black components of
#    size <= 3 inside blue regions, and (2) leaf pixels of larger black regions.
#    Every anomaly used to receive the same orange Chebyshev-1 halo, and centers
#    stayed black even when the halo crossed into nearby black territory.
#    Fused behavior: the same anomaly detector is kept, but the rendering now has
#    three nested layers. Internal defects become red centers with orange halos;
#    leaf pixels become red centers with magenta halos; and every halo cell must
#    already belong to the cleaned blue area, so halo paint is clipped at the
#    blue/black boundary. Example: if a black leaf protrudes one cell into a blue
#    stripe, the original task would color the adjacent black trunk and diagonals
#    orange; the fused task colors only the surrounding blue cells magenta and
#    turns the leaf pixel itself red.
# b) Complexity calibration:
#    To make all three fused controls visible, each sample must contain both at
#    least one internal defect and at least one leaf, so the output shows orange
#    halos, magenta halos, and red centers together. The generator also enforces
#    a positive clipped-halo count, otherwise D7/0 could become latent and the
#    sample would visually collapse to a simpler color-change variant.
# c) Hidden assumptions audit:
#    Original STEP 3 assumed one halo color and unrestricted overwrite. That no
#    longer holds, so the solver must classify targets before painting and apply
#    type-specific colors. Another original assumption was that every in-bounds
#    halo cell may be painted; the fused rule invalidates that, so halo painting
#    now checks the cleaned base grid and only paints cells whose base color is
#    blue. Mixed orange/magenta overlap has no explicit spec, so the generator
#    guarantees cross-type halo footprints stay disjoint; the solver still uses a
#    deterministic order (internal first, leaves second) for robustness.
# d) Placement legality:
#    Internal defects are sampled only from strict blue interiors, and every
#    non-defect 4-neighbor of every placed defect remains blue, so the defect
#    stays a detached black component. Leaves are placed only on blue cells that
#    touch a large black component on exactly one 4-neighbor, which guarantees
#    they remain true leaves after insertion. Spur pixels are added only on black
#    cells that the absorption pass deterministically flips back to black, and a
#    buffer zone keeps them away from defects/leaves so they do not alter target
#    classification.
# e) Visual observability:
#    Human-solvable samples need repeated structural evidence. The generator
#    therefore rejects any board lacking both halo colors, any board without red
#    centers, and any board where blue-limited clipping removes zero halo cells.
#    Across 2-4 examples, a solver can see that "tiny isolated holes" and "tips
#    attached to big black regions" share the same red center but differ in ring
#    color, while every ring stops at blue-area boundaries.
# f) Visual cleanliness:
#    Cross-type halo overlap is avoided with Chebyshev spacing between target
#    groups, and leaf counts stay modest. This prevents ambiguous orange/magenta
#    mixes and keeps the outputs legible rather than turning whole blue regions
#    into dense marker carpets.
# g) Inferability:
#    The fused palette still uses only five ARC colors overall: black/blue in the
#    input, plus red/orange/magenta in the output. The new colors are tied to
#    visible structure, not hidden numeric ordering: red always marks the anomaly
#    center, orange marks detached tiny black components, and magenta marks leaf
#    tips attached to larger black regions.

# --------------- solver ---------------

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
DEFECT_COUNT_RANGE = (2, 7)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 3)
SPUR_COUNT_RANGE = (2, 6)
TARGET_CLEARANCE = 3
SPUR_EXCLUSION = 2
MIN_DEFECT_GROUPS = 2
MIN_LEAVES = 1
MIN_SPURS = 2
MIN_CLIPPED_CELLS = 3
MAX_GEN_ATTEMPTS = 400

BLUE = 1
RED = 2
MAGENTA = 6
ORANGE = 7
BLACK = 0


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the original absorption pass for blue specks inside black regions."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += int(oob)
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _collect_targets(base: np.ndarray) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Split anomalies into internal defect groups and large-component leaf pixels."""
    H, W = base.shape
    internal_groups: list[set[tuple[int, int]]] = []
    leaf_pixels: set[tuple[int, int]] = set()
    all_targets: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            internal_groups.append(pts)
            all_targets.update(pts)
            continue

        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    black_n += 1
            if black_n <= 1:
                leaf_pixels.add((r, c))
                all_targets.add((r, c))

    return internal_groups, leaf_pixels, all_targets


def _paint_markers(
    base: np.ndarray,
    internal_groups: list[set[tuple[int, int]]],
    leaf_pixels: set[tuple[int, int]],
    *,
    split_colors: bool,
    limit_to_blue: bool,
    red_centers: bool,
) -> np.ndarray:
    """Render the fused markers, optionally toggling the combo controls for analysis."""
    H, W = base.shape
    out = np.array(base, dtype=int)
    all_targets = set().union(*internal_groups) if internal_groups else set()
    all_targets |= set(leaf_pixels)

    def can_paint(rr: int, cc: int) -> bool:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if limit_to_blue and int(base[rr, cc]) != BLUE:
            return False
        return True

    for group in internal_groups:
        for r, c in group:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if can_paint(rr, cc) and (rr, cc) not in all_targets:
                        out[rr, cc] = ORANGE

    leaf_halo = MAGENTA if split_colors else ORANGE
    for r, c in sorted(leaf_pixels):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if can_paint(rr, cc) and (rr, cc) not in all_targets:
                    out[rr, cc] = leaf_halo

    center_color = RED if red_centers else BLACK
    for r, c in sorted(all_targets):
        out[r, c] = center_color
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a sample visibly exercises the fused controls."""
    base = _absorb_blue_spurs(grid)
    internal_groups, leaf_pixels, all_targets = _collect_targets(base)
    fused = _paint_markers(
        base,
        internal_groups,
        leaf_pixels,
        split_colors=True,
        limit_to_blue=True,
        red_centers=True,
    )
    unclipped = _paint_markers(
        base,
        internal_groups,
        leaf_pixels,
        split_colors=True,
        limit_to_blue=False,
        red_centers=True,
    )
    original_like = _paint_markers(
        base,
        internal_groups,
        leaf_pixels,
        split_colors=False,
        limit_to_blue=False,
        red_centers=False,
    )
    return {
        "internal_groups": len(internal_groups),
        "leaf_pixels": len(leaf_pixels),
        "red_centers": int(np.count_nonzero(fused == RED)),
        "orange_halo": int(np.count_nonzero(fused == ORANGE)),
        "magenta_halo": int(np.count_nonzero(fused == MAGENTA)),
        "clipped_cells": int(np.count_nonzero(fused != unclipped)),
        "delta_vs_original": int(np.count_nonzero(fused != original_like)),
        "targets": len(all_targets),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect black anomalies in blue areas, then mark them with red centers and type-dependent clipped halos.

    Input structure:
    - Blue (1) foreground regions sit against black (0) regions, with a few tiny black holes embedded inside blue and a few black leaf tips protruding from larger black masses.
    - Black regions may also contain isolated blue spur pixels that should be absorbed before anomaly detection.

    Rule:
    - First turn any blue pixel with at least three effective black 4-neighbor sides into black, using the same corner-as-black convention as the source task.
    - Next detect two anomaly types in the cleaned grid: every black 4-connected component of size <= 3 is an internal defect, and every pixel of a larger black component with black 4-degree <= 1 is a leaf tip.
    - Paint a Chebyshev-1 halo around each anomaly, but only on cells that are blue in the cleaned grid: internal defects use orange (7), leaf tips use magenta (6), and every anomaly center itself becomes red (2).

    Key insight:
    - The puzzle still hinges on distinguishing detached tiny black holes from dangling black tips, but the output now encodes both the anomaly type and the blue-region boundary: the center is always red, while the surrounding ring both changes color by type and is clipped to the blue area.
    """
    base = _absorb_blue_spurs(grid)
    internal_groups, leaf_pixels, _all_targets = _collect_targets(base)
    return _paint_markers(
        base,
        internal_groups,
        leaf_pixels,
        split_colors=True,
        limit_to_blue=True,
        red_centers=True,
    )


# --------------- generator ---------------

def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an HxW board partitioned into large blue (1) and black (0) regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        population = list(range(CUT_MARGIN, H - CUT_MARGIN))
        cuts = sorted(random.sample(population, min(n_cuts, len(population))))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for idx in range(len(boundaries) - 1):
            grid[boundaries[idx] : boundaries[idx + 1], :] = (first_color + idx) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
        if (
            r1 - r0 > INNER_INSET_MIN_GAP
            and c1 - c0 > INNER_INSET_MIN_GAP
            and random.random() < INNER_INSET_PROB
        ):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    else:
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(W // 3, 2 * W // 3)
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue pixels whose 4-neighbors are all blue or out-of-bounds."""
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


def _place_internal_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    """Place detached 1-3 cell black components inside blue interiors."""
    H, W = grid.shape
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    placed_groups = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]

        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, TARGET_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            frontier = [(seed_r, seed_c)]
            while len(defect) < size and frontier:
                cr, cc = frontier.pop(0)
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels or (nr, nc) in defect:
                        continue
                    if _nearby(nr, nc, TARGET_CLEARANCE, defect_pixels):
                        continue
                    defect.append((nr, nc))
                    frontier.append((nr, nc))
                    if len(defect) >= size:
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

            if valid:
                defect_pixels.update(defect_set)
                placed_groups += 1
                remove = _exclusion_zone(defect_set, TARGET_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, placed_groups


def _leaf_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find blue cells that can become one-step black leaves of large black components."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in comp}
        if len(pts) <= 3:
            continue
        for r, c in pts:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in blocked or _nearby(rr, cc, TARGET_CLEARANCE, blocked):
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W and int(grid[ar, ac]) == BLACK:
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _place_leaves(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place 1-3 single-pixel leaf tips on large black-region boundaries."""
    leaf_pixels: set[tuple[int, int]] = set()
    wanted = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(wanted):
        candidates = _leaf_candidates(grid, blocked | leaf_pixels)
        if not candidates:
            break
        r, c = candidates[0]
        grid[r, c] = BLACK
        leaf_pixels.add((r, c))
    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels eligible to become blue spur noise."""
    H, W = grid.shape
    black = grid == BLACK

    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    coords = [
        (int(r), int(c))
        for r, c in zip(rs, cs)
        if (int(r), int(c)) not in exclude
    ]
    random.shuffle(coords)
    return coords


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return None
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return None
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, placed_groups = _place_internal_defects(grid, interior_blue)
    if placed_groups < MIN_DEFECT_GROUPS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    internal_groups, detected_leafs, all_targets = _collect_targets(grid)
    if len(internal_groups) < MIN_DEFECT_GROUPS or len(detected_leafs) < MIN_LEAVES:
        return None
    if leaf_pixels != detected_leafs:
        return None

    output = _paint_markers(
        grid,
        internal_groups,
        detected_leafs,
        split_colors=True,
        limit_to_blue=True,
        red_centers=True,
    )
    metrics = _behavior_metrics(grid)
    if metrics["orange_halo"] == 0 or metrics["magenta_halo"] == 0:
        return None
    if metrics["red_centers"] != metrics["targets"]:
        return None
    if metrics["clipped_cells"] < MIN_CLIPPED_CELLS:
        return None

    inp = grid.copy()
    protected = _exclusion_zone(all_targets, SPUR_EXCLUSION)
    candidates = _find_spur_candidates(inp, protected)
    n_spurs = min(random.randint(*SPUR_COUNT_RANGE), len(candidates))
    if n_spurs < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
