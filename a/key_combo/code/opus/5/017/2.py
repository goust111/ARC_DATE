from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _UTILS_DIR.exists():
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, components_mask, in_bounds, neighbors4
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= r < shape[0] and 0 <= c < shape[1]

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        if connectivity != 4:
            raise ValueError("fallback components_mask only supports 4-connectivity")
        seen = np.zeros_like(mask, dtype=bool)
        out: list[np.ndarray] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue
            stack = [(rr, cc)]
            seen[rr, cc] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
        return out


BG = Color.BLACK
DIVIDER = Color.GREY
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
REGION_H_RANGE = (4, 6)
REGION_W_RANGE = (4, 6)
MASK_AREA_RANGE = (5, 8)
MAX_GEN_ATTEMPTS = 500
MAX_LOCAL_ATTEMPTS = 200


def _full_grey_row_indices(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]


def _full_grey_col_indices(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]


def _region_origins(split_r: int, split_c: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, split_c + 1), (split_r + 1, 0), (split_r + 1, split_c + 1)]


def _extract_region_data(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[set[tuple[int, int]]], list[Counter]]:
    row_hits = _full_grey_row_indices(grid)
    col_hits = _full_grey_col_indices(grid)
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey divider row and column"
    split_r = row_hits[0]
    split_c = col_hits[0]
    region_h = split_r
    region_w = split_c
    assert grid.shape == (2 * region_h + 1, 2 * region_w + 1), "dividers must split the grid into four equal regions"

    masks: list[set[tuple[int, int]]] = []
    color_counters: list[Counter] = []
    for r0, c0 in _region_origins(split_r, split_c):
        region = grid[r0 : r0 + region_h, c0 : c0 + region_w]
        assert np.all(region != DIVIDER), "grey divider must stay outside the regions"
        coords = {tuple(map(int, rc)) for rc in np.argwhere(region != BG)}
        assert coords, "each region must contain a visible pattern"
        masks.append(coords)
        color_counters.append(Counter(map(int, region[region != BG].tolist())))

    return split_r, split_c, (region_h, region_w), masks, color_counters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each quadrant's single missing pixel by combining mask union and per-color multiset maxima.

    Input structure:
    - Black background grid with one solid grey row and one solid grey column that split the canvas into four equal corner regions.
    - Each corner region contains a partially erased colored pattern using blue, red, green, and yellow cells.
    - The four regions share the same aligned 4-connected mask, but each region is missing exactly one different mask position.

    Rule:
    - Read each region's non-black coordinates relative to that region's top-left corner and take the union across the four regions to recover the full mask.
    - In each region, the one coordinate present in the full mask but absent locally is the missing position to fill.
    - For every color, compute the target multiplicity as the maximum count of that color seen in any single region.
    - Fill each region's missing position with the unique color whose local count is still one below that target multiplicity.
    - Keep the grey dividers unchanged.

    Key insight:
    - The missing coordinate is determined by cross-region mask union, while the missing color is determined independently by cross-region multiset balancing; neither clue alone is sufficient.
    """
    g = np.asarray(grid, dtype=int)
    split_r, split_c, (region_h, region_w), masks, color_counters = _extract_region_data(g)
    del region_h, region_w

    full_mask = set().union(*masks)
    assert len(full_mask) >= MASK_AREA_RANGE[0], "full mask is unexpectedly small"

    target_counts = Counter()
    for color in PALETTE:
        target_counts[color] = max(counter.get(color, 0) for counter in color_counters)
    target_counts = Counter({color: count for color, count in target_counts.items() if count > 0})

    out = g.copy()
    for (r0, c0), mask, color_counter in zip(_region_origins(split_r, split_c), masks, color_counters, strict=False):
        missing = full_mask - mask
        assert len(missing) == 1, "each region must be missing exactly one mask cell"
        miss_r, miss_c = next(iter(missing))

        deficits = []
        total_gap = 0
        for color, target in target_counts.items():
            gap = int(target - color_counter.get(color, 0))
            if gap > 0:
                deficits.append((color, gap))
                total_gap += gap

        assert len(deficits) == 1 and deficits[0][1] == 1 and total_gap == 1, "missing color must be uniquely determined"
        out[r0 + miss_r, c0 + miss_c] = deficits[0][0]

    return out


def _mask_to_array(shape: tuple[int, int], coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.zeros(shape, dtype=bool)
    if coords:
        rows, cols = zip(*sorted(coords))
        arr[list(rows), list(cols)] = True
    return arr


def _is_connected(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not coords:
        return False
    comps = components_mask(_mask_to_array(shape, coords), connectivity=4)
    return len(comps) == 1 and comps[0].shape[0] == len(coords)


def _bbox_dims(coords: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _is_solid_bbox(coords: set[tuple[int, int]]) -> bool:
    h, w = _bbox_dims(coords)
    return len(coords) == h * w


def _adjacent_background_candidates(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc) and (nr, nc) not in coords:
                out.add((int(nr), int(nc)))
    return out


def _grow_random_shape(region_shape: tuple[int, int], area: int) -> set[tuple[int, int]]:
    rh, rw = region_shape
    start = (random.randrange(rh), random.randrange(rw))
    cells = {start}
    frontier = set()
    for nb in neighbors4(*start):
        if in_bounds(region_shape, nb[0], nb[1]):
            frontier.add((int(nb[0]), int(nb[1])))
    while len(cells) < area and frontier:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        for nb in neighbors4(*pick):
            if in_bounds(region_shape, nb[0], nb[1]) and nb not in cells:
                frontier.add((int(nb[0]), int(nb[1])))
    return cells


def _sample_shape_and_missing(region_shape: tuple[int, int]) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    max_area = min(MASK_AREA_RANGE[1], region_shape[0] * region_shape[1] - 2)
    if max_area < MASK_AREA_RANGE[0]:
        return None

    for _ in range(MAX_LOCAL_ATTEMPTS):
        area = random.randint(MASK_AREA_RANGE[0], max_area)
        cells = _grow_random_shape(region_shape, area)
        if len(cells) != area:
            continue
        if not _is_connected(cells, region_shape):
            continue
        bbox_h, bbox_w = _bbox_dims(cells)
        if bbox_h == 1 or bbox_w == 1:
            continue
        if _is_solid_bbox(cells):
            continue

        removable = []
        for cell in cells:
            reduced = set(cells)
            reduced.remove(cell)
            if not _is_connected(reduced, region_shape):
                continue
            if len(_adjacent_background_candidates(reduced, region_shape)) < 2:
                continue
            removable.append(cell)

        if len(removable) < 4:
            continue
        return cells, random.sample(removable, 4)
    return None


def _random_target_counts(area: int) -> Counter:
    n_active = 4 if area >= 6 and random.random() < 0.65 else 3
    active = random.sample(PALETTE, n_active)
    cuts = sorted(random.sample(range(1, area), n_active - 1))
    parts = []
    prev = 0
    for cut in cuts + [area]:
        parts.append(cut - prev)
        prev = cut
    random.shuffle(parts)
    return Counter({color: count for color, count in zip(active, parts, strict=False)})


def _sample_assignment(mask_cells: list[tuple[int, int]], target_counts: Counter) -> dict[tuple[int, int], int]:
    colors = []
    for color, count in target_counts.items():
        colors.extend([int(color)] * int(count))
    assert len(colors) == len(mask_cells)
    random.shuffle(colors)
    shuffled_cells = mask_cells[:]
    random.shuffle(shuffled_cells)
    return {cell: color for cell, color in zip(shuffled_cells, colors, strict=False)}


def _position_copy_blocked(assignments: list[dict[tuple[int, int], int]], missing_cells: list[tuple[int, int]]) -> bool:
    for i, cell in enumerate(missing_cells):
        truth = assignments[i][cell]
        others = [assignments[j][cell] for j in range(4) if j != i]
        if len(set(others)) < 2:
            return False
        if others.count(truth) > 1:
            return False
    return True


def _sample_assignments(mask_cells: list[tuple[int, int]], target_counts: Counter, missing_cells: list[tuple[int, int]]) -> list[dict[tuple[int, int], int]] | None:
    for _ in range(MAX_LOCAL_ATTEMPTS):
        assignments = [_sample_assignment(mask_cells, target_counts) for _ in range(4)]
        missing_colors = [assignments[i][missing_cells[i]] for i in range(4)]
        if len(set(missing_colors)) == 1:
            continue
        if sum(int(target_counts[color] > 1) for color in missing_colors) < 1:
            continue
        if not _position_copy_blocked(assignments, missing_cells):
            continue
        if len({tuple(assign[cell] for cell in mask_cells) for assign in assignments}) < 2:
            continue
        return assignments
    return None


def _render_instance(
    region_shape: tuple[int, int],
    mask_cells: list[tuple[int, int]],
    missing_cells: list[tuple[int, int]],
    assignments: list[dict[tuple[int, int], int]],
) -> tuple[np.ndarray, np.ndarray]:
    rh, rw = region_shape
    inp = np.full((2 * rh + 1, 2 * rw + 1), BG, dtype=int)
    out = inp.copy()
    inp[rh, :] = DIVIDER
    inp[:, rw] = DIVIDER
    out[rh, :] = DIVIDER
    out[:, rw] = DIVIDER

    for idx, (r0, c0) in enumerate(_region_origins(rh, rw)):
        missing = missing_cells[idx]
        assign = assignments[idx]
        for cell in mask_cells:
            color = assign[cell]
            out[r0 + cell[0], c0 + cell[1]] = color
            if cell != missing:
                inp[r0 + cell[0], c0 + cell[1]] = color
    return inp, out


def _validate_generated_instance(
    inp: np.ndarray,
    out: np.ndarray,
    region_shape: tuple[int, int],
    mask_cells: list[tuple[int, int]],
    missing_cells: list[tuple[int, int]],
    target_counts: Counter,
    assignments: list[dict[tuple[int, int], int]],
) -> bool:
    if len(set(missing_cells)) != 4:
        return False
    mask_set = set(mask_cells)
    if not _is_connected(mask_set, region_shape):
        return False
    if not (_bbox_dims(mask_set)[0] >= 2 and _bbox_dims(mask_set)[1] >= 2):
        return False

    visible_masks = [mask_set - {missing_cells[i]} for i in range(4)]
    if any(not _is_connected(visible, region_shape) for visible in visible_masks):
        return False
    if any(len(_adjacent_background_candidates(visible, region_shape)) < 2 for visible in visible_masks):
        return False

    visible_counters = []
    for i, visible in enumerate(visible_masks):
        counts = Counter(assignments[i][cell] for cell in visible)
        visible_counters.append(counts)

    recovered_target = Counter()
    for color in PALETTE:
        recovered_target[color] = max(counter.get(color, 0) for counter in visible_counters)
    recovered_target = Counter({color: count for color, count in recovered_target.items() if count > 0})
    if recovered_target != target_counts:
        return False

    for i, counts in enumerate(visible_counters):
        deficits = {color: target_counts[color] - counts.get(color, 0) for color in target_counts}
        deficits = {color: gap for color, gap in deficits.items() if gap > 0}
        if deficits != {assignments[i][missing_cells[i]]: 1}:
            return False

    if not _position_copy_blocked(assignments, missing_cells):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    region_shape = (
        random.randint(*REGION_H_RANGE),
        random.randint(*REGION_W_RANGE),
    )
    sampled = _sample_shape_and_missing(region_shape)
    if sampled is None:
        return None
    mask_set, missing_cells = sampled
    mask_cells = sorted(mask_set)

    target_counts = _random_target_counts(len(mask_cells))
    assignments = _sample_assignments(mask_cells, target_counts, missing_cells)
    if assignments is None:
        return None

    inp, out = _render_instance(region_shape, mask_cells, missing_cells, assignments)
    if not _validate_generated_instance(inp, out, region_shape, mask_cells, missing_cells, target_counts, assignments):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        generated = _generate_instance()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid complementary-mask balancing instance")
