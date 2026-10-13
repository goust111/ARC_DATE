from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


FLOOR = 7
WALL = 6
MARKER = 9
FG_COLORS = (1, 2, 3, 4)
OUT_BG = 0

GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (12, 16)
THREE_REGION_PROB = 0.55
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == MARKER)
    assert len(pts) == 1, "grid must contain exactly one maroon marker"
    return int(pts[0, 0]), int(pts[0, 1])


def _reachable_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    mr, mc = _find_marker(grid)
    passable = {FLOOR, MARKER, *FG_COLORS}
    seen = np.zeros((h, w), dtype=bool)
    seen[mr, mc] = True
    q: deque[tuple[int, int]] = deque([(mr, mc)])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) in passable:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _reachable_counts(grid: np.ndarray) -> dict[int, int]:
    mask = _reachable_mask(grid)
    return {color: int(np.count_nonzero(mask & (grid == color))) for color in FG_COLORS}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter foreground counts by marker reachability and render them as a descending bottom-aligned histogram.

    Input structure:
    - The grid is mostly orange (7) floor with a single maroon (9) marker pixel.
    - Magenta (6) wall pixels form a connected barrier that splits the passable space into 2-3 regions.
    - Blue, red, green, and yellow pixels may be scattered on the floor across multiple regions.

    Rule:
    - Start from the maroon marker and flood-fill with 4-connectivity through orange floor and colored pixels, but never through magenta walls.
    - Count only the reachable blue/red/green/yellow pixels; colors whose reachable count is zero are excluded.
    - Sort the participating colors by reachable frequency in descending order.
    - Build a black output whose width is the number of participating colors and whose height is the largest reachable count, then draw one bottom-aligned color bar per sorted color.

    Key insight:
    - The walls are not output objects; they act as a filter deciding which visible color pixels belong to the marker's accessible region and therefore which data points contribute to the histogram.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = _reachable_counts(g)
    bars = [(color, count) for color, count in counts.items() if count > 0]
    assert bars, "at least one reachable foreground color is required"
    palette_rank = {color: idx for idx, color in enumerate(FG_COLORS)}
    bars.sort(key=lambda item: (-item[1], palette_rank[item[0]]))

    height = max(count for _, count in bars)
    width = len(bars)
    out = np.full((height, width), OUT_BG, dtype=int)
    for col, (color, count) in enumerate(bars):
        out[height - count :, col] = color
    return out


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            cid = len(comps)
            labels[r, c] = cid
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(comp)
    return labels, comps


def _build_vertical_barrier(h: int, w: int, rng: random.Random, three_regions: bool) -> np.ndarray:
    wall = np.zeros((h, w), dtype=bool)
    left_bound = max(2, w // 4)
    right_bound = min(w - 3, w - 1 - w // 4)
    cur_c = rng.randint(left_bound, right_bound)
    spine_cols = [cur_c]
    wall[0, cur_c] = True

    for r in range(1, h):
        step = rng.choice([-1, 0, 1]) if 0 < r < h - 1 else 0
        nxt_c = min(right_bound, max(left_bound, cur_c + step))
        wall[r, cur_c] = True
        for cc in range(min(cur_c, nxt_c), max(cur_c, nxt_c) + 1):
            wall[r, cc] = True
        cur_c = nxt_c
        spine_cols.append(cur_c)

    if three_regions:
        branch_r = rng.randint(2, h - 3)
        branch_c = spine_cols[branch_r]
        branch_to_left = rng.random() < 0.5
        if branch_to_left:
            wall[branch_r, : branch_c + 1] = True
        else:
            wall[branch_r, branch_c:] = True

    return wall


def _build_barrier(h: int, w: int, rng: random.Random, three_regions: bool) -> np.ndarray:
    if rng.random() < 0.5:
        return _build_vertical_barrier(h, w, rng, three_regions)
    return _build_vertical_barrier(w, h, rng, three_regions).T


def _choose_reachable_counts(capacity: int, n_positive: int, rng: random.Random) -> list[int] | None:
    max_bar = min(9, capacity)
    combos = [
        combo
        for combo in itertools.combinations(range(2, max_bar + 1), n_positive)
        if sum(combo) <= capacity - 2
    ]
    if not combos:
        return None
    return list(rng.choice(combos))


def _place_colors_in_region(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color_counts: dict[int, int],
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    free_cells = [cell for cell in cells if cell not in reserved]
    rng.shuffle(free_cells)
    need = sum(color_counts.values())
    assert need <= len(free_cells)

    idx = 0
    for color, count in color_counts.items():
        for _ in range(count):
            r, c = free_cells[idx]
            idx += 1
            grid[r, c] = color
            reserved.add((r, c))


def _assign_unreachable_colors(
    unreachable_comps: list[list[tuple[int, int]]],
    zero_colors: list[int],
    reachable_colors: list[int],
    reachable_counts: dict[int, int],
    rng: random.Random,
) -> list[tuple[list[tuple[int, int]], dict[int, int]]]:
    assignments = [{color: 0 for color in FG_COLORS} for _ in unreachable_comps]

    pool = list(range(len(unreachable_comps)))
    rng.shuffle(pool)
    for i, color in enumerate(zero_colors):
        target = pool[i % len(pool)]
        assignments[target][color] += 2

    hidden_color = rng.choice(zero_colors)
    hidden_target = rng.randrange(len(unreachable_comps))
    assignments[hidden_target][hidden_color] += rng.randint(1, 4)

    bridged_color = rng.choice(reachable_colors)
    bridge_target = rng.randrange(len(unreachable_comps))
    assignments[bridge_target][bridged_color] += rng.randint(1, 3)

    for color in reachable_colors:
        if color == bridged_color:
            continue
        if rng.random() < 0.45:
            assignments[rng.randrange(len(unreachable_comps))][color] += 1

    total_capacity = sum(len(comp) for comp in unreachable_comps)
    total_need = sum(sum(item.values()) for item in assignments)
    if total_need > total_capacity:
        return []

    for comp, color_counts in zip(unreachable_comps, assignments):
        if sum(color_counts.values()) > len(comp):
            return []

    hidden_total = sum(item[hidden_color] for item in assignments)
    if hidden_total <= min(reachable_counts.values()):
        extra = min(reachable_counts.values()) + 1 - hidden_total
        for idx, comp in sorted(enumerate(unreachable_comps), key=lambda item: -len(item[1])):
            free = len(comp) - sum(assignments[idx].values())
            if free >= extra:
                assignments[idx][hidden_color] += extra
                break
        else:
            return []

    return list(zip(unreachable_comps, assignments))


def _expected_histogram_from_counts(counts: dict[int, int]) -> np.ndarray:
    bars = [(color, count) for color, count in counts.items() if count > 0]
    palette_rank = {color: idx for idx, color in enumerate(FG_COLORS)}
    bars.sort(key=lambda item: (-item[1], palette_rank[item[0]]))
    out = np.full((max(count for _, count in bars), len(bars)), OUT_BG, dtype=int)
    for c, (color, count) in enumerate(bars):
        out[-count:, c] = color
    return out


def _validate_example(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == MARKER) != 1:
        return False
    if np.count_nonzero(grid == WALL) == 0:
        return False

    _, wall_comps = _components(grid == WALL)
    if len(wall_comps) != 1:
        return False

    labels, passable_comps = _components(grid != WALL)
    if len(passable_comps) not in (2, 3):
        return False

    mr, mc = _find_marker(grid)
    marker_label = labels[mr, mc]
    if marker_label < 0:
        return False

    counts = _reachable_counts(grid)
    positives = {color: count for color, count in counts.items() if count > 0}
    if len(positives) < 2:
        return False
    if len(set(positives.values())) != len(positives):
        return False

    global_counts = {color: int(np.count_nonzero(grid == color)) for color in FG_COLORS}
    if any(count < 2 for count in global_counts.values()):
        return False
    if not any(global_counts[color] > 0 and counts[color] == 0 for color in FG_COLORS):
        return False

    out = main(grid)
    if not np.array_equal(out, _expected_histogram_from_counts(counts)):
        return False
    if out.shape[1] != len(positives):
        return False
    if out.shape[0] != max(positives.values()):
        return False

    hidden_colors = [color for color in FG_COLORS if counts[color] == 0 and global_counts[color] > 0]
    if not hidden_colors:
        return False
    hidden_color = hidden_colors[0]
    if global_counts[hidden_color] <= min(positives.values()):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    three_regions = rng.random() < THREE_REGION_PROB
    grid = np.full((h, w), FLOOR, dtype=int)

    wall_mask = _build_barrier(h, w, rng, three_regions)
    grid[wall_mask] = WALL

    labels, passable_comps = _components(grid != WALL)
    if len(passable_comps) not in (2, 3):
        return None
    if len(passable_comps) != (3 if three_regions else 2):
        return None
    if min(len(comp) for comp in passable_comps) < 8:
        return None

    feasible_marker_regions = [i for i, comp in enumerate(passable_comps) if len(comp) >= 12]
    if not feasible_marker_regions:
        return None
    marker_idx = rng.choice(feasible_marker_regions)
    reachable_cells = passable_comps[marker_idx]
    unreachable_comps = [comp for i, comp in enumerate(passable_comps) if i != marker_idx]
    unreachable_cells = [cell for comp in unreachable_comps for cell in comp]
    if len(unreachable_cells) < 8:
        return None

    positive_options = [n for n in (2, 3) if _choose_reachable_counts(len(reachable_cells) - 1, n, rng) is not None]
    if not positive_options:
        return None
    n_positive = rng.choice(positive_options)
    chosen_counts = _choose_reachable_counts(len(reachable_cells) - 1, n_positive, rng)
    if chosen_counts is None:
        return None

    reachable_colors = rng.sample(list(FG_COLORS), n_positive)
    zero_colors = [color for color in FG_COLORS if color not in reachable_colors]
    reachable_count_map = {color: 0 for color in FG_COLORS}
    for color, count in zip(reachable_colors, chosen_counts):
        reachable_count_map[color] = int(count)

    unreachable_plan = _assign_unreachable_colors(
        unreachable_comps,
        zero_colors,
        reachable_colors,
        {color: reachable_count_map[color] for color in reachable_colors},
        rng,
    )
    if not unreachable_plan:
        return None

    marker_cell = rng.choice(reachable_cells)
    grid[marker_cell] = MARKER
    reserved = {marker_cell}
    _place_colors_in_region(
        grid,
        reachable_cells,
        {color: reachable_count_map[color] for color in reachable_colors},
        reserved,
        rng,
    )

    for comp, color_counts in unreachable_plan:
        local_reserved: set[tuple[int, int]] = set()
        filtered = {color: count for color, count in color_counts.items() if count > 0}
        if filtered:
            _place_colors_in_region(grid, comp, filtered, local_reserved, rng)

    return grid if _validate_example(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reachability-filter histogram example")
