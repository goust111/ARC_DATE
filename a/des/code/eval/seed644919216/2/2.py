import random
from collections import deque

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARROW_SHAPES = {
    (0, 1): frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
    (0, -1): frozenset({(1, 2), (1, 1), (1, 0), (0, 0), (2, 0)}),
    (1, 0): frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
    (-1, 0): frozenset({(2, 1), (1, 1), (0, 1), (0, 0), (0, 2)}),
}
PATTERN_TO_DIR = {pattern: direction for direction, pattern in ARROW_SHAPES.items()}


GEN_BG = 0
GEN_ARROW = 2
GEN_DIVIDER = 5
GEN_SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_LAYOUTS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3))
GEN_REGION_MIN = 7
GEN_REGION_MAX = 10
GEN_GRID_MAX = 29
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _divider_color(grid: np.ndarray, bg: int) -> int:
    counts: dict[int, int] = {}
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            color = int(row[0])
            counts[color] = counts.get(color, 0) + 1
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            color = int(col[0])
            counts[color] = counts.get(color, 0) + 1
    if not counts:
        raise AssertionError("missing divider lines")
    return max(counts.items(), key=lambda item: item[1])[0]


def _segments_from_dividers(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_divider in enumerate(mask.tolist()):
        if not is_divider and start is None:
            start = idx
        elif is_divider and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _arrow_candidates(
    region: np.ndarray, bg: int, divider: int | None
) -> list[tuple[set[tuple[int, int]], tuple[int, int], int]]:
    candidates: list[tuple[set[tuple[int, int]], tuple[int, int], int]] = []
    for color in np.unique(region):
        color = int(color)
        if color == bg or (divider is not None and color == divider):
            continue
        for comp in _components(region == color):
            if len(comp) != 5:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            if r1 - r0 != 2 or c1 - c0 != 2:
                continue
            local = frozenset((r - r0, c - c0) for r, c in comp)
            direction = PATTERN_TO_DIR.get(local)
            if direction is not None:
                candidates.append((set(comp), direction, color))
    return candidates


def _find_arrow(region: np.ndarray, bg: int, divider: int | None) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    candidates = _arrow_candidates(region, bg, divider)
    if len(candidates) > 1:
        red_candidates = [candidate for candidate in candidates if candidate[2] == GEN_ARROW]
        if len(red_candidates) == 1:
            candidates = red_candidates
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one arrow, found {len(candidates)}")
    cells, direction, _color = candidates[0]
    return cells, direction


def _mirror_patch(patch: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    if direction[1] != 0:
        return np.fliplr(patch).copy()
    return np.flipud(patch).copy()


def _count_copy_slots(region_shape: tuple[int, int], bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    height, width = region_shape
    r0, c0, r1, c1 = bbox
    patch_h = r1 - r0 + 1
    patch_w = c1 - c0 + 1
    step_r = direction[0] * patch_h
    step_c = direction[1] * patch_w
    count = 0
    cur_r0 = r0
    cur_c0 = c0
    while True:
        next_r0 = cur_r0 + step_r
        next_c0 = cur_c0 + step_c
        if next_r0 < 0 or next_c0 < 0:
            break
        if next_r0 + patch_h > height or next_c0 + patch_w > width:
            break
        count += 1
        cur_r0 = next_r0
        cur_c0 = next_c0
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by divider lines, detect a seed pattern and a direction arrow in
    each region, then grow alternating mirror copies of the seed along the arrow.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider = _divider_color(g, bg)

    row_dividers = np.all(g == divider, axis=1)
    col_dividers = np.all(g == divider, axis=0)
    row_segments = _segments_from_dividers(row_dividers)
    col_segments = _segments_from_dividers(col_dividers)

    out = g.copy()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            arrow_cells, direction = _find_arrow(region, bg, divider)

            seed_mask = (region != bg) & (region != divider)
            for rr, cc in arrow_cells:
                seed_mask[rr, cc] = False
            seed_coords = np.argwhere(seed_mask)
            if seed_coords.size == 0:
                raise AssertionError("region is missing a seed pattern")

            sr0, sc0 = seed_coords.min(axis=0)
            sr1, sc1 = seed_coords.max(axis=0)
            patch = np.full((sr1 - sr0 + 1, sc1 - sc0 + 1), bg, dtype=int)
            for rr, cc in seed_coords.tolist():
                patch[rr - sr0, cc - sc0] = int(region[rr, cc])

            region_out = region.copy()
            for rr, cc in arrow_cells:
                region_out[rr, cc] = bg

            patch_h, patch_w = patch.shape
            step_r = direction[0] * patch_h
            step_c = direction[1] * patch_w
            cur_patch = patch
            cur_r0 = int(sr0)
            cur_c0 = int(sc0)
            while True:
                next_patch = _mirror_patch(cur_patch, direction)
                next_r0 = cur_r0 + step_r
                next_c0 = cur_c0 + step_c
                if next_r0 < 0 or next_c0 < 0:
                    break
                if next_r0 + patch_h > region.shape[0] or next_c0 + patch_w > region.shape[1]:
                    break
                target = region_out[next_r0 : next_r0 + patch_h, next_c0 : next_c0 + patch_w]
                mask = next_patch != bg
                target[mask] = next_patch[mask]
                cur_patch = next_patch
                cur_r0 = next_r0
                cur_c0 = next_c0

            out[r0 : r1 + 1, c0 : c1 + 1] = region_out

    return out


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != GEN_BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grow_shape(height: int, width: int, count: int, rng: random.Random) -> set[tuple[int, int]] | None:
    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    return cells


def _seed_patch(max_h: int, max_w: int, direction: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(120):
        height = rng.randint(2, max_h)
        width = rng.randint(2, max_w)
        area = rng.randint(3, min(height * width, 8))
        cells = _grow_shape(height, width, area, rng)
        if cells is None:
            continue

        patch = np.full((height, width), GEN_BG, dtype=int)
        color_count = 1 if area == 3 else rng.randint(1, min(3, area))
        if area >= 4 and rng.random() < 0.7:
            color_count = max(color_count, 2)
        palette = rng.sample(GEN_SEED_COLORS, color_count)
        for r, c in cells:
            patch[r, c] = rng.choice(palette)
        if len(palette) > 1:
            vals = {int(patch[r, c]) for r, c in cells}
            if len(vals) == 1:
                rr, cc = next(iter(cells))
                patch[rr, cc] = palette[1]

        patch = _crop_patch(patch)
        if direction[1] != 0 and np.array_equal(patch, np.fliplr(patch)):
            continue
        if direction[0] != 0 and np.array_equal(patch, np.flipud(patch)):
            continue
        if _arrow_candidates(patch, GEN_BG, None):
            continue
        return patch
    return None


def _expanded_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    expanded = mask.copy()
    points = np.argwhere(mask)
    for r, c in points.tolist():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded[nr, nc] = True
    return expanded


def _arrow_cells_at(r0: int, c0: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in ARROW_SHAPES[direction]]


def _place_region(height: int, width: int, direction: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, int] | None:
    for _ in range(160):
        if direction[1] != 0:
            patch = _seed_patch(min(4, height), min(4, width // 2), direction, rng)
        else:
            patch = _seed_patch(min(4, height // 2), min(4, width), direction, rng)
        if patch is None:
            continue

        patch_h, patch_w = patch.shape
        if direction[1] != 0 and width < 2 * patch_w:
            continue
        if direction[0] != 0 and height < 2 * patch_h:
            continue

        if direction == (0, 1):
            seed_r0 = rng.randint(0, height - patch_h)
            seed_c0 = 0
        elif direction == (0, -1):
            seed_r0 = rng.randint(0, height - patch_h)
            seed_c0 = width - patch_w
        elif direction == (1, 0):
            seed_r0 = 0
            seed_c0 = rng.randint(0, width - patch_w)
        else:
            seed_r0 = height - patch_h
            seed_c0 = rng.randint(0, width - patch_w)

        region = np.full((height, width), GEN_BG, dtype=int)
        seed_mask = np.zeros((height, width), dtype=bool)
        seed_pixels = patch != GEN_BG
        seed_view = region[seed_r0 : seed_r0 + patch_h, seed_c0 : seed_c0 + patch_w]
        seed_view[seed_pixels] = patch[seed_pixels]
        seed_mask[seed_r0 : seed_r0 + patch_h, seed_c0 : seed_c0 + patch_w] = seed_pixels

        strict_block = _expanded_mask(seed_mask)
        strict_candidates: list[tuple[int, int]] = []
        loose_candidates: list[tuple[int, int]] = []
        for arrow_r0 in range(height - 2):
            for arrow_c0 in range(width - 2):
                cells = _arrow_cells_at(arrow_r0, arrow_c0, direction)
                if any(seed_mask[r, c] for r, c in cells):
                    continue
                loose_candidates.append((arrow_r0, arrow_c0))
                if not any(strict_block[r, c] for r, c in cells):
                    strict_candidates.append((arrow_r0, arrow_c0))
        candidates = strict_candidates or loose_candidates
        if not candidates:
            continue

        arrow_r0, arrow_c0 = rng.choice(candidates)
        for r, c in _arrow_cells_at(arrow_r0, arrow_c0, direction):
            region[r, c] = GEN_ARROW

        bbox = (seed_r0, seed_c0, seed_r0 + patch_h - 1, seed_c0 + patch_w - 1)
        copy_slots = _count_copy_slots(region.shape, bbox, direction)
        if copy_slots < 1:
            continue
        try:
            arrow_cells, found_direction = _find_arrow(region, GEN_BG, None)
        except AssertionError:
            continue
        if found_direction != direction or len(arrow_cells) != 5:
            continue
        return region, copy_slots
    return None


def _sample_sizes(count: int, rng: random.Random) -> list[int]:
    min_sum = count * GEN_REGION_MIN
    max_sum = min(count * GEN_REGION_MAX, GEN_GRID_MAX - (count - 1))
    total = rng.randint(min_sum, max_sum)
    remaining = total
    sizes: list[int] = []
    for idx in range(count):
        slots_left = count - idx - 1
        low = max(GEN_REGION_MIN, remaining - slots_left * GEN_REGION_MAX)
        high = min(GEN_REGION_MAX, remaining - slots_left * GEN_REGION_MIN)
        sizes.append(rng.randint(low, high))
        remaining -= sizes[-1]
    rng.shuffle(sizes)
    return sizes


def _direction_sequence(count: int, rng: random.Random) -> list[tuple[int, int]]:
    directions = list(ARROW_SHAPES.keys())
    seq: list[tuple[int, int]] = []
    while len(seq) < count:
        rng.shuffle(directions)
        seq.extend(directions)
    return seq[:count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count, col_count = rng.choice(GEN_LAYOUTS)
    row_sizes = _sample_sizes(row_count, rng)
    col_sizes = _sample_sizes(col_count, rng)

    height = sum(row_sizes) + (row_count - 1)
    width = sum(col_sizes) + (col_count - 1)
    grid = np.full((height, width), GEN_BG, dtype=int)

    row_bounds: list[tuple[int, int]] = []
    pos = 0
    for idx, size in enumerate(row_sizes):
        row_bounds.append((pos, pos + size - 1))
        pos += size
        if idx < row_count - 1:
            grid[pos, :] = GEN_DIVIDER
            pos += 1

    col_bounds: list[tuple[int, int]] = []
    pos = 0
    for idx, size in enumerate(col_sizes):
        col_bounds.append((pos, pos + size - 1))
        pos += size
        if idx < col_count - 1:
            grid[:, pos] = GEN_DIVIDER
            pos += 1

    directions = _direction_sequence(row_count * col_count, rng)
    max_slots = 0
    direction_idx = 0
    for rr0, rr1 in row_bounds:
        for cc0, cc1 in col_bounds:
            region = _place_region(rr1 - rr0 + 1, cc1 - cc0 + 1, directions[direction_idx], rng)
            direction_idx += 1
            if region is None:
                return None
            region_grid, slots = region
            max_slots = max(max_slots, slots)
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = region_grid

    if max_slots < 2:
        return None

    if _most_common_color(grid) != GEN_BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
