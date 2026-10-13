from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
PINK = 6


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            out.append((start, prev))
            start = prev = value
    out.append((start, prev))
    return out


def _region_intervals(length: int, divider_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for run_start, run_end in divider_runs:
        if start <= run_start - 1:
            intervals.append((start, run_start - 1))
        start = run_end + 1
    if start <= length - 1:
        intervals.append((start, length - 1))
    return intervals


def _uniform_size(intervals: list[tuple[int, int]]) -> int | None:
    sizes = [end - start + 1 for start, end in intervals]
    if not sizes or any(size <= 0 for size in sizes):
        return None
    if len(set(sizes)) != 1:
        return None
    return int(sizes[0])


def _find_divider_grid(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int, int]:
    height, width = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], int, int]] = []

    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if not full_rows or not full_cols:
            continue

        row_intervals = _region_intervals(height, _runs(full_rows))
        col_intervals = _region_intervals(width, _runs(full_cols))
        tile_h = _uniform_size(row_intervals)
        tile_w = _uniform_size(col_intervals)
        if tile_h is None or tile_w is None:
            continue
        if len(row_intervals) < 2 or len(col_intervals) < 2:
            continue
        candidates.append((color, row_intervals, col_intervals, tile_h, tile_w))

    assert len(candidates) == 1, "expected one same-color divider grid with equal regions"
    return candidates[0]


def _background_excluding(g: np.ndarray, divider_color: int) -> int:
    vals = [int(x) for x in g[g != int(divider_color)].flat]
    assert vals, "expected non-divider region cells"
    return Counter(vals).most_common(1)[0][0]


def _region_masks(
    g: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    bg: int,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.any(tile == int(bg)), "each region must have visible background"
            fg_colors = set(map(int, np.unique(tile))) - {int(bg)}
            assert len(fg_colors) == 1, "each region must contain one foreground color"
            mask = tile != int(bg)
            assert bool(mask.any()), "each region must contain a foreground pattern"
            masks.append(mask)
    return masks


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """Exclude the shape majority in a divider grid and output the two minority masks' symmetric difference.

    Input structure:
    - One color forms complete horizontal and vertical divider bars, splitting the canvas into equal rectangular regions.
    - Every region contains one small monochrome foreground pattern on the shared background.
    - All but two regions have the same local foreground mask; the two remaining masks are unique and different from each other.

    Rule:
    - Ignore pattern colors and vote by each region's foreground mask in local region coordinates.
    - Select the two masks whose shape frequency is one after the majority mask has been excluded.
    - Return one region-sized grid with pink at cells where exactly one selected mask is foreground and black elsewhere.

    Key insight:
    - The repeated majority pattern is only a screen for finding the two exceptions; the answer is their aligned symmetric difference, not a copy of any region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider, row_intervals, col_intervals, tile_h, tile_w = _find_divider_grid(g)
    bg = _background_excluding(g, divider)
    masks = _region_masks(g, row_intervals, col_intervals, bg)
    assert len(masks) >= 5, "expected enough regions for a strict shape majority"

    by_key = {_mask_key(mask): mask for mask in masks}
    counts = Counter(_mask_key(mask) for mask in masks)
    singles = [key for key, count in counts.items() if count == 1]
    majority_count = max(counts.values())
    majority_keys = [key for key, count in counts.items() if count == majority_count]

    assert len(majority_keys) == 1, "expected a unique majority shape"
    assert majority_count == len(masks) - 2 and majority_count > len(masks) // 2
    assert len(singles) == 2, "expected exactly two minority shapes"

    a = by_key[singles[0]]
    b = by_key[singles[1]]
    assert a.shape == (tile_h, tile_w) and b.shape == (tile_h, tile_w)

    xor = np.logical_xor(a, b)
    assert bool(xor.any()), "minority symmetric difference must be non-empty"
    out = np.full((tile_h, tile_w), BLACK, dtype=int)
    out[xor] = PINK
    return out


# --------------- generator ---------------

GEN_RETRY_LIMIT = 500
PATTERN_COLORS = [1, 2, 3, 4, 7, 8, 9]
DIVIDER_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in NEIGHBORS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    max_area = max(4, min(9, (height * width) // 3))
    area = random.randint(4, max_area)
    cells = {(random.randrange(height), random.randrange(width))}

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _removable_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for cell in cells:
        reduced = set(cells)
        reduced.remove(cell)
        if len(reduced) >= 3 and _cells_connected(reduced):
            out.append(cell)
    return out


def _mutate_mask(mask: np.ndarray) -> np.ndarray | None:
    height, width = mask.shape
    original = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    cells = set(original)

    remove_goal = random.randint(1, min(3, max(1, len(cells) - 3)))
    removed = 0
    for _ in range(remove_goal):
        choices = _removable_cells(cells)
        if not choices:
            break
        cells.remove(random.choice(choices))
        removed += 1
    if removed == 0:
        return None

    add_goal = random.randint(1, 3)
    added = 0
    for _ in range(add_goal):
        candidates = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and (nr, nc) not in original:
                    candidates.add((nr, nc))
        if not candidates:
            break
        cells.add(random.choice(list(candidates)))
        added += 1
    if added == 0 or not _cells_connected(cells):
        return None

    out = np.zeros_like(mask, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _sample_shapes(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(300):
        majority = _random_connected_mask(height, width)
        a = _random_connected_mask(height, width)
        if np.array_equal(a, majority):
            continue

        for _mut in range(80):
            b = _mutate_mask(a)
            if b is None:
                continue
            xor = np.logical_xor(a, b)
            overlap = np.logical_and(a, b)
            a_only = np.logical_and(a, np.logical_not(b))
            b_only = np.logical_and(b, np.logical_not(a))
            if not (overlap.any() and a_only.any() and b_only.any()):
                continue
            if np.array_equal(b, majority) or np.array_equal(a, b):
                continue
            if np.array_equal(xor, majority) or np.array_equal(xor, a) or np.array_equal(xor, b):
                continue
            return majority, a, b
    return None


def _valid_layouts() -> list[tuple[int, int, int, int]]:
    layouts = []
    for rows in range(2, 5):
        for cols in range(2, 5):
            if rows * cols < 5:
                continue
            for tile_h in range(4, 8):
                for tile_w in range(4, 8):
                    height = rows * tile_h + rows - 1
                    width = cols * tile_w + cols - 1
                    if height <= 30 and width <= 30:
                        layouts.append((rows, cols, tile_h, tile_w))
    return layouts


LAYOUTS = _valid_layouts()


def _render_expected(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, dtype=int)
    out[np.logical_xor(a, b)] = PINK
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols, tile_h, tile_w = random.choice(LAYOUTS)
    shapes = _sample_shapes(tile_h, tile_w)
    if shapes is None:
        return None
    majority, minority_a, minority_b = shapes

    divider = random.choice(DIVIDER_COLORS)
    palette = [c for c in PATTERN_COLORS if c != divider]
    if len(palette) < 3:
        return None

    height = rows * tile_h + rows - 1
    width = cols * tile_w + cols - 1
    grid = np.full((height, width), BLACK, dtype=int)

    for i in range(1, rows):
        grid[i * tile_h + i - 1, :] = divider
    for j in range(1, cols):
        grid[:, j * tile_w + j - 1] = divider

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    minority_slots = random.sample(slots, 2)
    majority_slots = [slot for slot in slots if slot not in minority_slots]

    color_count = random.randint(3, min(5, len(palette)))
    colors = random.sample(palette, color_count)
    minority_colors = {
        minority_slots[0]: random.choice(colors),
        minority_slots[1]: random.choice(colors),
    }

    majority_colors: dict[tuple[int, int], int] = {}
    shuffled_majority = majority_slots[:]
    random.shuffle(shuffled_majority)
    required = list(dict.fromkeys(minority_colors.values()))
    for idx, slot in enumerate(shuffled_majority):
        if idx < len(required):
            majority_colors[slot] = required[idx]
        else:
            majority_colors[slot] = random.choice(colors)
    if len(set(majority_colors.values())) == 1 and len(colors) > 1 and len(shuffled_majority) > 1:
        majority_colors[shuffled_majority[-1]] = random.choice([c for c in colors if c != majority_colors[shuffled_majority[0]]])

    for slot in slots:
        rslot, cslot = slot
        r0 = rslot * (tile_h + 1)
        c0 = cslot * (tile_w + 1)
        if slot == minority_slots[0]:
            mask = minority_a
            color = minority_colors[slot]
        elif slot == minority_slots[1]:
            mask = minority_b
            color = minority_colors[slot]
        else:
            mask = majority
            color = majority_colors[slot]
        tile = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]
        tile[mask] = color

    expected = _render_expected(minority_a, minority_b)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed = _region_masks(grid, _find_divider_grid(grid)[1], _find_divider_grid(grid)[2], BLACK)
    counts = Counter(_mask_key(mask) for mask in parsed)
    if sorted(counts.values()) != [1, 1, rows * cols - 2]:
        return None
    if not np.logical_and(minority_a, minority_b).any():
        return None
    if not np.logical_and(minority_a, np.logical_not(minority_b)).any():
        return None
    if not np.logical_and(minority_b, np.logical_not(minority_a)).any():
        return None

    used_majority_colors = set(majority_colors.values())
    if not set(minority_colors.values()).issubset(used_majority_colors):
        return None
    if len(used_majority_colors) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid minority symmetric-difference input")
