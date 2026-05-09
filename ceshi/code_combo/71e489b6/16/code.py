# Combo variant of 71e489b6 (index 16)
# combo_spec: D7/1+D8/0
# Name: 互斥光晕+小网格高密度

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
#    Original behavior: after absorbing blue spur pixels inside black regions, every
#    black component of size <= 3 and every leaf pixel of a larger black component
#    receives an orange Chebyshev-1 halo, and halo overlaps stay orange. In the
#    combo, target detection stays the same, but halo painting changes: each small
#    black component is treated as one defect group, each large-component leaf is
#    its own group, and any cell covered by halos from two or more groups is left
#    unchanged. Example on a 3x3 blue patch: two single-pixel defects at opposite
#    corners of the patch used to create one merged orange blob; now the shared
#    middle-adjacent cells keep their original blue/black color, producing a split
#    pattern with visible "missing" overlap cells.
# b) Complexity calibration:
#    Shrinking the grid to 10-13 while still placing 2-6 defects could easily make
#    the task either trivial (if defects are spaced out) or unreadable (if every
#    marker collides everywhere). The generator therefore forces at least one dense
#    overlap cluster but keeps the total defect count bounded and leaf count low,
#    so the mutual-exclusion rule appears repeatedly without turning the whole blue
#    region into noise.
# c) Hidden assumptions audit:
#    The original STEP 3 implicitly assumed "orange if any halo reaches here."
#    That assumption is invalid now; we must count coverage per defect group before
#    painting. Another hidden assumption was that iterating target pixels directly
#    is harmless. That breaks for 2-3 pixel defects because internal overlap within
#    one small component would wrongly cancel itself, so the combo solver must build
#    one footprint per small component. The generator must mirror those group units
#    and ensure overlap is between distinct groups, not only within one component.
# d) Placement legality:
#    Small defects are placed only on strict blue interiors whose non-defect
#    4-neighbors remain blue, so they stay detached from large black regions and
#    from each other under 4-connectivity. Leaves are added only on blue cells that
#    touch a large black component on exactly one side. Spurs are inserted only on
#    black cells that the solver's absorption pass deterministically flips to black.
# e) Visual observability:
#    Every accepted sample must have a nonzero difference between mutual-exclusion
#    output and the original overlap-allowed output, and at least one contested halo
#    cell. This guarantees the combo-specific rule is visible from examples instead
#    of being a latent possibility.
# f) Visual cleanliness:
#    The generator concentrates overlap in one or two compact clusters and limits
#    extra leaves, avoiding a board-wide carpet of partial halos. Colors remain the
#    original three-color system (0, 1, 7), so the only new evidence is structural.
# g) Inferability:
#    Because the palette is unchanged, the rule must be recoverable from geometry
#    alone. Repeated near-neighbor defects on small boards create several examples
#    where expected orange cells are missing exactly at shared halo locations, which
#    is a stable, human-visible cue for "overlap cancels."

# --------------- solver ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
MIN_BLUE_REGION = 24
MIN_BLACK_REGION = 16
PARTITION_STYLES = ["rect_inset", "h_stripe", "l_shape"]
DEFECT_COUNT_RANGE = (2, 6)
DEFECT_SIZE_WEIGHTS = [7, 2, 1]
DEFECT_PLACE_TRIES = 80
LEAF_COUNT_RANGE = (0, 1)
SPUR_COUNT_RANGE = (2, 4)
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_MUTUAL_DIFF = 2
MIN_CONTESTED_CELLS = 1
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the original blue-spur absorption pass."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != 1:
                continue
            black_n = 0
            oob = 0
            for dr, dc in _neighbors4(r, c):
                if not (0 <= dr < H and 0 <= dc < W):
                    oob += 1
                    continue
                if int(base[dr, dc]) == 0:
                    black_n += 1
            if oob >= 2:
                black_n += int(oob)
            if black_n >= 3:
                base[r, c] = 0
    return base


def _collect_target_groups(grid: np.ndarray) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]]]:
    """Return defect groups and the black pixels that must remain black."""
    H, W = grid.shape
    groups: list[set[tuple[int, int]]] = []
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(grid == 0, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            groups.append(pts)
            preserve.update(pts)
            continue

        for r, c in sorted(pts):
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == 0:
                    black_n += 1
            if black_n <= 1:
                groups.append({(r, c)})
                preserve.add((r, c))

    return groups, preserve


def _halo_coverage(
    shape: tuple[int, int],
    groups: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Count how many distinct defect groups cover each non-center halo cell."""
    H, W = map(int, shape)
    coverage: dict[tuple[int, int], int] = {}
    for pts in groups:
        footprint: set[tuple[int, int]] = set()
        for r, c in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                        footprint.add((rr, cc))
        for cell in footprint:
            coverage[cell] = coverage.get(cell, 0) + 1
    return coverage


def _paint_mutual_halo(
    grid: np.ndarray,
    groups: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> np.ndarray:
    """Paint orange only on cells covered by exactly one defect-group halo."""
    out = np.array(grid, dtype=int)
    coverage = _halo_coverage(grid.shape, groups, preserve)
    for (r, c), count in coverage.items():
        if count == 1:
            out[r, c] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def _paint_plain_halo(
    grid: np.ndarray,
    groups: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> np.ndarray:
    """Reference output for the original overlap-allowed rule."""
    out = np.array(grid, dtype=int)
    coverage = _halo_coverage(grid.shape, groups, preserve)
    for r, c in coverage:
        out[r, c] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies in compact blue regions, but cancel any halo cells shared by multiple defects.

    Input structure:
    - A 10-13 by 10-13 board contains large blue (1) foreground zones embedded in black (0), plus 2-6 compact black defects packed tightly inside blue.
    - Some larger black regions may contribute single-pixel leaf tips into blue, and black areas may also contain thin blue spur pixels that should be absorbed before marking.

    Rule:
    - First convert any blue pixel with at least three effective black 4-neighbor sides into black, using the same corner treatment as the source task.
    - Then detect two target types: every black connected component of size <= 3 inside blue, and every leaf pixel (black degree <= 1) of larger black components.
    - Paint orange (7) on the Chebyshev-1 halo of each target group, but if a cell belongs to halos from two or more distinct groups, leave that cell unchanged; target pixels themselves always remain black.

    Key insight:
    - The anomaly detector is still structural, but the visual code is no longer additive: dense neighboring defects create intentional orange gaps exactly where their halos would have overlapped.
    """
    base = _absorb_blue_spurs(grid)
    groups, preserve = _collect_target_groups(base)
    return _paint_mutual_halo(base, groups, preserve)


# --------------- generator ---------------

def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a small board with one large blue region and a large black complement."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "rect_inset":
        r0 = random.randint(1, max(1, H // 4))
        r1 = random.randint(max(r0 + 5, 2 * H // 3), H - 2)
        c0 = random.randint(1, max(1, W // 4))
        c1 = random.randint(max(c0 + 5, 2 * W // 3), W - 2)
        grid[r0:r1 + 1, c0:c1 + 1] = 1

    elif style == "h_stripe":
        height = random.randint(5, H - 2)
        r0 = random.randint(1, H - height - 1)
        grid[r0:r0 + height, :] = 1

    else:
        vc = random.randint(max(4, W // 3), W - 3)
        hc = random.randint(max(4, H // 3), H - 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1
            grid[:hc, :] = 1
        else:
            grid[:, vc:] = 1
            grid[hc:, :] = 1

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue cells whose in-bounds 4-neighbors are all blue."""
    blue = grid == 1
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _valid_defect_group(
    grid: np.ndarray,
    group: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    group_set = set(group)
    if not group_set:
        return False
    for r, c in group:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != 1 or (r, c) in occupied:
            return False
        for rr, cc in _neighbors4(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in group_set:
                continue
            if grid[rr, cc] != 1 or (rr, cc) in occupied:
                return False
    return True


def _cluster_candidates(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Find checkerboard-like interior clusters that guarantee halo overlap."""
    H, W = grid.shape
    patterns = [
        [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        [(0, 1), (1, 0), (1, 2), (2, 1)],
    ]
    clusters: list[list[tuple[int, int]]] = []
    for r0 in range(H - 2):
        for c0 in range(W - 2):
            for pattern in patterns:
                cells = [(r0 + dr, c0 + dc) for dr, dc in pattern]
                if all(grid[r, c] == 1 for r, c in cells):
                    ok = True
                    for r, c in cells:
                        for rr, cc in _neighbors4(r, c):
                            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells and grid[rr, cc] != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        clusters.append(cells)
    random.shuffle(clusters)
    return clusters


def _place_defects(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    """Place 2-6 dense small black defect components inside blue."""
    interior_blue = _find_interior_blue(grid)
    if len(interior_blue) < MIN_DEFECTS:
        return []

    target_count = random.randint(*DEFECT_COUNT_RANGE)
    groups: list[set[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()

    clusters = _cluster_candidates(grid)
    if not clusters:
        return []
    cluster = clusters[0]
    cluster_size = min(target_count, random.randint(2, min(4, len(cluster))))
    chosen_cluster = random.sample(cluster, cluster_size)
    for cell in chosen_cluster:
        groups.append({cell})
        occupied.add(cell)

    interior_set = set(interior_blue)
    while len(groups) < target_count:
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        placed = False
        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if (seed_r, seed_c) in occupied:
                continue

            group = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = seed_r + dr, seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in occupied:
                        continue
                    candidate = group + [(nr, nc)]
                    if _valid_defect_group(grid, candidate, occupied):
                        group = candidate
                        if len(group) >= defect_size:
                            break

            if _valid_defect_group(grid, group, occupied):
                group_set = set(group)
                groups.append(group_set)
                occupied.update(group_set)
                placed = True
                break
        if not placed:
            break

    if len(groups) < MIN_DEFECTS:
        return []

    for group in groups:
        for r, c in group:
            grid[r, c] = 0
    return groups


def _place_leaves(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Optionally add 0-1 leaf tips from a large black region into blue."""
    H, W = grid.shape
    leaf_pixels: set[tuple[int, int]] = set()
    wanted = random.randint(*LEAF_COUNT_RANGE)
    if wanted == 0:
        return leaf_pixels

    black_comps = components_mask(grid == 0, connectivity=4)
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]
    random.shuffle(large_black)

    for comp_set in large_black:
        candidates = []
        for r, c in comp_set:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if grid[rr, cc] != 1 or (rr, cc) in blocked or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W:
                        if grid[ar, ac] == 0 or (ar, ac) in leaf_pixels:
                            black_n += 1
                if black_n == 1 and not _nearby(rr, cc, 1, blocked):
                    candidates.append((rr, cc))

        if candidates:
            pick = random.choice(candidates)
            leaf_pixels.add(pick)
            grid[pick] = 0
            break

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black cells that solver STEP 0 will absorb if turned blue."""
    H, W = grid.shape
    black = grid == 0

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
    coords = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(coords)
    return coords


def _count_contested_cells(
    shape: tuple[int, int],
    groups: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> int:
    coverage = _halo_coverage(shape, groups, preserve)
    return sum(count > 1 for count in coverage.values())


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    clean = _build_partition(H, W)

    blue_comps = list(components_mask(clean == 1, connectivity=4))
    black_comps = list(components_mask(clean == 0, connectivity=4))
    if not blue_comps or not black_comps:
        return None
    if max(len(comp) for comp in blue_comps) < MIN_BLUE_REGION:
        return None
    if max(len(comp) for comp in black_comps) < MIN_BLACK_REGION:
        return None

    defect_groups = _place_defects(clean)
    if len(defect_groups) < MIN_DEFECTS:
        return None

    defect_pixels = set().union(*defect_groups)
    leaf_pixels = _place_leaves(clean, defect_pixels)

    groups, preserve = _collect_target_groups(clean)
    contested = _count_contested_cells(clean.shape, groups, preserve)
    mutual_out = _paint_mutual_halo(clean, groups, preserve)
    plain_out = _paint_plain_halo(clean, groups, preserve)
    diff = int(np.count_nonzero(mutual_out != plain_out))

    if contested < MIN_CONTESTED_CELLS or diff < MIN_MUTUAL_DIFF:
        return None

    inp = clean.copy()
    spur_count = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, defect_pixels | leaf_pixels)
    if min(spur_count, len(candidates)) < MIN_SPURS:
        return None
    for r, c in candidates[:spur_count]:
        inp[r, c] = 1

    try:
        if np.array_equal(main(inp), mutual_out):
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
