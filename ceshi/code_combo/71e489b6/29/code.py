# Combo variant of 71e489b6 (index 29)
# combo_spec: D6/1+D8/0
# Name: 棋盘格分区+小网格高密度

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


# Design analysis for combo D6/1 + D8/0
# a) Concrete behavior trace
#    Original inputs used 2-5 large blue/black regions, so defects typically lived in one or two
#    roomy blue masses. Here the visual carrier changes to a 3x3 checkerboard of coarse blocks on a
#    10-13 by 10-13 board. Example sketch with 4/4/4 bands:
#      base checkerboard           add defects in multiple blue blocks       add a black leaf
#        111100001111               111100001111                            111100001111
#        111100001111               101100001111                            111100001111
#        111100001111      --->     111100001101                 --->       111100001101
#        111100001111               111100001111                            111100001111
#        000011110000               000011110000                            000001110000
#    The solver still absorbs blue spur pixels, then marks every size<=3 black component and every
#    leaf pixel of a larger black component with an orange Chebyshev-1 halo. The fusion change is
#    that those targets now come from several disconnected blue tiles on a much tighter canvas, so
#    orange often spills onto neighboring black tiles or clips against the grid border.
# b) Complexity calibration
#    If the board were shrunk without restructuring the regions, the output would often look like a
#    smaller copy of the original task. If we used a very fine 1x1 checkerboard instead, defects and
#    background would become ambiguous noise. The chosen 3x3 coarse-tile checkerboard keeps only 4-5
#    blue regions per sample, enough for repeated cross-region evidence while still leaving each tile
#    visually readable.
# c) Hidden assumptions audit
#    - Original generator assumption: there are large blue interiors with many interchangeable
#      candidate cells. This fails on 10-13 boards, so defect placement must reason per blue tile.
#    - Original solver assumption: large black regions are a few big masses. That is no longer true,
#      but the code only depends on component size and leaf degree, which still hold for rectangular
#      checkerboard black tiles. Edge tiles are kept at least 3x3 to avoid accidental size<=3 black
#      components or degenerate one-cell-wide "leafy" strips.
#    - STEP 0 assumes blue spurs can be recognized from local 4-neighbor counts alone. The checker-
#      board layout preserves that because spurs are inserted only into black tiles where the same
#      local absorption test is guaranteed to fire.
# d) Placement legality
#    Row and column bands are always partitioned into three segments of size 3-5, so every checker-
#    board tile is a nontrivial rectangle. Defects are validated cell-by-cell: each planned black
#    defect pixel must currently be blue and every non-defect 4-neighbor must also stay blue. Leaves
#    are added only on blue cells with exactly one black 4-neighbor, so they become genuine large-
#    component leaf pixels instead of bridges between checkerboard tiles.
# e) Visual observability
#    A human sees the same orange-halo rule, but now against a repeated coarse checkerboard where
#    defects are scattered across several independent blue blocks. Because at least three distinct
#    blue tiles carry defects in accepted samples, the new layout is visible without reading hidden
#    color ids or scan order.
# f) Visual cleanliness
#    The palette remains {0,1,7}. Clutter is controlled by keeping only 2-6 defects and 1-2 leaves
#    on a 3x3 block checkerboard. Spurs are absorbed before output, so they never compete visually
#    with the actual marked anomalies.
# g) Inferability and degeneration check
#    The fusion does not change the semantic transform; it changes the scene geometry. To prevent
#    degenerate samples that still look like the source task, generation rejects boards unless they
#    exhibit multiple blue components, multiple defect-bearing blue tiles, and several orange cells
#    landing on originally black checkerboard tiles.
# h) New solver assumptions vs generator guarantees
#    `main()` now implicitly relies on all non-defect black checkerboard tiles being larger than 3
#    and having no leaf pixels before explicit leaf insertion. `_build_checkerboard_partition()`
#    guarantees 3x3 to 5x5 rectangles only. It also assumes spur absorption runs before component
#    labeling; `generate_input()` inserts spurs only after the clean expected output is fixed.


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
DEFECT_COUNT_RANGE = (2, 6)
LEAF_COUNT_RANGE = (1, 2)
SPUR_COUNT_RANGE = (2, 5)
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_DEFECT_TILES = 3
MIN_BLUE_COMPONENTS = 4
MIN_ORANGE_ON_BLACK = 6
MAX_GEN_ATTEMPTS = 2000

DEFECT_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (-1, 0), (-2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, -1), (0, -2)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, -1)),
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Match the source-task absorption pass for blue spur pixels."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != 1:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(g[rr, cc]) == 0:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                out[r, c] = 0
    return out


def _collect_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Collect small black components and leaf pixels of larger black components."""
    H, W = grid.shape
    preserve: set[tuple[int, int]] = set()

    for comp in components_mask(grid == 0, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            preserve.update(pts)
            continue

        comp_set = set(pts)
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    black_n += 1
            if black_n <= 1:
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
                    out[rr, cc] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight black anomalies on a compact checkerboard of blue and black regions.

    Input structure:
    - The board is only 10-13 cells tall and wide, partitioned into coarse checkerboard blocks that alternate blue (1) and black (0).
    - Several separated blue blocks contain tiny black defect components, black blocks may poke one-pixel leaves into neighboring blue blocks, and some black blocks contain blue spur pixels that should disappear.

    Rule:
    - First absorb any blue pixel whose effective 4-neighborhood is mostly black, with corner out-of-bounds sides counted the same way as in the source task.
    - Then treat every black connected component of size at most 3, plus every leaf pixel of a larger black component, as a target defect.
    - Paint orange (7) on the full Chebyshev-1 halo around each target while keeping the target pixels themselves black.

    Key insight:
    - The detection criterion is still purely structural, but the compact checkerboard layout creates many independent blue search zones and makes halo spillover onto neighboring black blocks a normal visible outcome.
    """
    base = _absorb_blue_spurs(np.asarray(grid, dtype=int))
    preserve = _collect_targets(base)
    return _paint_halo(base, preserve)


def _three_band_sizes(total: int) -> tuple[int, int, int]:
    """Partition `total` into three bands of size 3-5, preferring 4/5-heavy splits."""
    choices: list[tuple[int, int, int]] = []
    best_score: tuple[int, int] | None = None
    for a in range(3, 6):
        for b in range(3, 6):
            c = total - a - b
            if 3 <= c <= 5:
                sizes = (a, b, c)
                score = (sum(x >= 4 for x in sizes), -max(sizes) + min(sizes))
                if best_score is None or score > best_score:
                    best_score = score
                    choices = [sizes]
                elif score == best_score:
                    choices.append(sizes)
    if not choices:
        raise ValueError(f"cannot split {total} into three 3-5 bands")
    return random.choice(choices)


def _build_checkerboard_partition(H: int, W: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Create a 3x3 coarse checkerboard partition."""
    row_sizes = _three_band_sizes(H)
    col_sizes = _three_band_sizes(W)
    row_edges = [0]
    col_edges = [0]
    for size in row_sizes:
        row_edges.append(row_edges[-1] + size)
    for size in col_sizes:
        col_edges.append(col_edges[-1] + size)

    phase = random.randint(0, 1)
    grid = np.zeros((H, W), dtype=int)
    tile_ids = np.full((H, W), -1, dtype=int)
    tiles: list[dict[str, object]] = []

    for tr in range(3):
        for tc in range(3):
            r0 = row_edges[tr]
            r1 = row_edges[tr + 1]
            c0 = col_edges[tc]
            c1 = col_edges[tc + 1]
            color = 1 if (tr + tc + phase) % 2 == 0 else 0
            tile_id = tr * 3 + tc
            grid[r0:r1, c0:c1] = color
            tile_ids[r0:r1, c0:c1] = tile_id
            tiles.append(
                {
                    "id": tile_id,
                    "color": color,
                    "bounds": (r0, c0, r1, c1),
                    "shape": (r1 - r0, c1 - c0),
                    "checker_rc": (tr, tc),
                }
            )

    return grid, tile_ids, tiles


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
        if int(grid[r, c]) != 1 or (r, c) in occupied:
            return False
        for rr, cc in _neighbors4(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in group_set:
                continue
            if int(grid[rr, cc]) != 1 or (rr, cc) in occupied:
                return False
    return True


def _tile_candidates(
    grid: np.ndarray,
    tile: dict[str, object],
    occupied: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    """Enumerate valid defect components that stay inside one blue checker tile."""
    r0, c0, r1, c1 = tile["bounds"]
    raw: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {}
    for template in DEFECT_TEMPLATES:
        for ar in range(r0, r1):
            for ac in range(c0, c1):
                cells = [(ar + dr, ac + dc) for dr, dc in template]
                if _valid_defect_group(grid, cells, occupied):
                    norm = tuple(sorted(set(cells)))
                    raw[norm] = set(norm)
    candidates = list(raw.values())
    random.shuffle(candidates)
    candidates.sort(key=lambda pts: (len(pts), min(pts)))
    return candidates


def _place_defects(
    grid: np.ndarray,
    tiles: list[dict[str, object]],
    tile_ids: np.ndarray,
) -> tuple[list[set[tuple[int, int]]], set[int]]:
    """Place 2-6 defects across multiple distinct blue checkerboard tiles."""
    blue_tiles = [tile for tile in tiles if int(tile["color"]) == 1]
    random.shuffle(blue_tiles)

    occupied: set[tuple[int, int]] = set()
    groups: list[set[tuple[int, int]]] = []
    defect_tile_ids: set[int] = set()

    candidate_cache = {int(tile["id"]): _tile_candidates(grid, tile, occupied) for tile in blue_tiles}
    usable_tiles = [tile for tile in blue_tiles if candidate_cache[int(tile["id"])]]
    if len(usable_tiles) < MIN_DEFECTS:
        return [], set()

    extra_capacity = sum(1 for tile in usable_tiles if tile["shape"][0] >= 4 and tile["shape"][1] >= 4)
    max_groups = min(6, len(usable_tiles) + extra_capacity)
    target_count = random.randint(MIN_DEFECTS, max_groups)
    distinct_goal = min(len(usable_tiles), MIN_DEFECT_TILES if target_count >= MIN_DEFECT_TILES else target_count)

    for tile in usable_tiles:
        if len(defect_tile_ids) >= distinct_goal:
            break
        tile_id = int(tile["id"])
        candidates = _tile_candidates(grid, tile, occupied)
        if not candidates:
            continue
        pick = random.choice(candidates[: min(4, len(candidates))])
        groups.append(pick)
        occupied.update(pick)
        defect_tile_ids.add(tile_id)

    if len(groups) < MIN_DEFECTS or len(defect_tile_ids) < distinct_goal:
        return [], set()

    while len(groups) < target_count:
        random.shuffle(usable_tiles)
        placed = False
        for tile in usable_tiles:
            tile_id = int(tile["id"])
            candidates = _tile_candidates(grid, tile, occupied)
            if not candidates:
                continue
            weighted = [cand for cand in candidates if len(cand) <= 2]
            source = weighted if weighted else candidates
            pick = random.choice(source[: min(6, len(source))])
            groups.append(pick)
            occupied.update(pick)
            defect_tile_ids.add(tile_id)
            placed = True
            break
        if not placed:
            break

    if len(groups) < MIN_DEFECTS:
        return [], set()

    for group in groups:
        for r, c in group:
            grid[r, c] = 0

    return groups, defect_tile_ids


def _place_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Add 1-2 genuine leaf pixels from large black tiles into blue tiles."""
    H, W = grid.shape
    wanted = random.randint(*LEAF_COUNT_RANGE)
    leaves: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == 0, connectivity=4)
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]
    random.shuffle(large_black)

    for comp_set in large_black:
        if len(leaves) >= wanted:
            break
        candidates = []
        for r, c in comp_set:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if int(grid[rr, cc]) != 1 or (rr, cc) in blocked or (rr, cc) in leaves:
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W:
                        if int(grid[ar, ac]) == 0 or (ar, ac) in leaves:
                            black_n += 1
                if black_n == 1 and not _nearby(rr, cc, 1, blocked):
                    candidates.append((rr, cc))
        random.shuffle(candidates)
        if candidates:
            pick = candidates[0]
            leaves.add(pick)
            grid[pick] = 0

    return leaves


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Black cells that solver STEP 0 will absorb if turned blue."""
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


def _fusion_metrics(
    partition: np.ndarray,
    output: np.ndarray,
    defect_groups: list[set[tuple[int, int]]],
    defect_tile_ids: set[int],
) -> dict[str, int]:
    """Quantify that the checkerboard + dense small-grid behavior is visibly present."""
    blue_components = len(components_mask(partition == 1, connectivity=4))
    orange_on_black = int(np.count_nonzero((output == 7) & (partition == 0)))
    return {
        "blue_components": blue_components,
        "defect_count": len(defect_groups),
        "defect_tiles": len(defect_tile_ids),
        "orange_on_black": orange_on_black,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    partition, tile_ids, tiles = _build_checkerboard_partition(H, W)
    clean = partition.copy()

    defect_groups, defect_tile_ids = _place_defects(clean, tiles, tile_ids)
    if len(defect_groups) < MIN_DEFECTS:
        return None

    defect_pixels = set().union(*defect_groups)
    leaf_pixels = _place_leaves(clean, defect_pixels)
    if len(leaf_pixels) < 1:
        return None

    preserve = defect_pixels | leaf_pixels
    output = _paint_halo(clean, preserve)
    metrics = _fusion_metrics(partition, output, defect_groups, defect_tile_ids)
    if metrics["blue_components"] < MIN_BLUE_COMPONENTS:
        return None
    if metrics["defect_tiles"] < min(metrics["defect_count"], MIN_DEFECT_TILES):
        return None
    if metrics["orange_on_black"] < MIN_ORANGE_ON_BLACK:
        return None

    inp = clean.copy()
    spur_count = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, preserve)
    if min(spur_count, len(candidates)) < MIN_SPURS:
        return None
    for r, c in candidates[:spur_count]:
        inp[r, c] = 1

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
