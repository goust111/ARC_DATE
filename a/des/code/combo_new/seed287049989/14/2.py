import random
from math import ceil

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300
SLOT_SIZE = 9
MAX_SLOTS_PER_ROW = 3
GEN_COUNTER = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _read_legend(top_row: np.ndarray) -> list[int]:
    colors: list[int] = []
    in_run = False
    for value in top_row:
        color = int(value)
        if color != BG and not in_run:
            colors.append(color)
            in_run = True
        elif color == BG:
            in_run = False
    assert 1 <= len(colors) <= 4
    assert len(colors) == len(set(colors))
    return colors


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    side = r1 - r0 + 1
    if side < 3 or side != c1 - c0 + 1:
        return None
    if len(component) != 4 * side - 4:
        return None

    cells = set(component)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if ((rr, cc) in cells) != on_border:
                return None
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed interior in place using the color-keyed legend on the top row.

    Input structure:
    - Black is the background.
    - The first row is a legend strip containing 1 to 4 distinct single-pixel colors from left to right, with Black separators between them.
    - Below the legend are several non-overlapping one-cell-thick square frames whose border colors come from the legend; each frame surrounds a non-symmetric interior pattern that may include Black padding.

    Rule:
    - Read the legend colors from left to right and map them to rotations: first color = 0 degrees, second = 90 degrees clockwise, third = 180 degrees, fourth = 270 degrees clockwise.
    - For every square frame, find its border color in that legend and rotate only the interior square by the corresponding amount around the frame center.
    - Keep the frame borders, the legend strip, and the Black background unchanged.

    Key insight:
    - The top-row color order is the only rotation key; frame colors are lookup labels, and each frame's interior must be rotated in place according to that legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _read_legend(g[0])
    rotation_steps = {color: idx for idx, color in enumerate(legend)}

    out = g.copy()
    frames: list[tuple[int, int, int, int, int]] = []
    for color in legend:
        mask = g == color
        mask[0, :] = False
        for comp in _components(mask):
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            frames.append((r0, c0, r1, c1, color))

    for r0, c0, r1, c1, color in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, -rotation_steps[color])

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _pick_legend_count() -> int:
    global GEN_COUNTER
    GEN_COUNTER += 1
    return [1, 2, 3, 4][(GEN_COUNTER - 1) % 4]


def _legend_positions(width: int, count: int) -> list[int]:
    min_span = 2 * count - 1
    extra = width - min_span
    gaps = _random_composition(extra, count + 1)
    pos = gaps[0]
    positions: list[int] = []
    for idx in range(count):
        positions.append(pos)
        if idx < count - 1:
            pos += 2 + gaps[idx + 1]
    return positions


def _grow_connected_mask(size: int, target_cells: int) -> set[tuple[int, int]]:
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    frontier = {start}
    while len(cells) < target_cells and frontier:
        base_r, base_c = random.choice(tuple(frontier))
        options = []
        for dr, dc in NEIGHBORS4:
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                options.append((nr, nc))
        if options:
            cells.add(random.choice(options))
        else:
            frontier.remove((base_r, base_c))
        frontier = {
            (r, c)
            for r, c in cells
            if any(
                0 <= r + dr < size
                and 0 <= c + dc < size
                and (r + dr, c + dc) not in cells
                for dr, dc in NEIGHBORS4
            )
        }
    return cells


def _make_pattern(size: int, palette: list[int]) -> np.ndarray:
    max_cells = min(size * size - 1, size + 3)
    min_cells = min(3, max_cells)
    for _ in range(200):
        cell_count = random.randint(min_cells, max_cells)
        cells = _grow_connected_mask(size, cell_count)
        colors = random.sample(palette, random.randint(1, min(3, len(palette))))

        pattern = np.zeros((size, size), dtype=int)
        for cell in cells:
            pattern[cell] = random.choice(colors)

        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        return pattern

    pattern = np.zeros((size, size), dtype=int)
    pattern[0, 0] = palette[0]
    pattern[0, min(1, size - 1)] = palette[min(1, len(palette) - 1)]
    pattern[min(1, size - 1), 0] = palette[min(2, len(palette) - 1)]
    return pattern


def _choose_layout(frame_count: int) -> tuple[int, int]:
    layouts = []
    for rows in range(1, 4):
        cols = ceil(frame_count / rows)
        if cols <= MAX_SLOTS_PER_ROW and 2 + rows * SLOT_SIZE <= 30 and cols * SLOT_SIZE <= 30:
            layouts.append((rows, cols))
    return random.choice(layouts)


def _gen_attempt() -> np.ndarray | None:
    legend_count = _pick_legend_count()
    legend_colors = random.sample(COLOR_POOL, legend_count)

    max_extra = min(2, 6 - legend_count)
    frame_count = legend_count + random.randint(0, max_extra)
    rows, cols = _choose_layout(frame_count)

    base_h = 2 + rows * SLOT_SIZE
    base_w = cols * SLOT_SIZE
    extra_top = random.randint(0, min(2, 30 - base_h))
    extra_bottom = random.randint(0, min(3, 30 - base_h - extra_top))
    extra_left = random.randint(0, min(3, 30 - base_w))
    extra_right = random.randint(0, min(3, 30 - base_w - extra_left))

    height = base_h + extra_top + extra_bottom
    width = base_w + extra_left + extra_right
    if width < 2 * legend_count - 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    for pos, color in zip(_legend_positions(width, legend_count), legend_colors):
        grid[0, pos] = color

    frame_colors = legend_colors + [random.choice(legend_colors) for _ in range(frame_count - legend_count)]
    random.shuffle(frame_colors)
    slot_indices = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slot_indices)

    pattern_palette = [color for color in COLOR_POOL if color not in legend_colors]
    if len(pattern_palette) < 3:
        return None

    top_offset = 2 + extra_top
    left_offset = extra_left

    for idx in range(frame_count):
        slot_r, slot_c = slot_indices[idx]
        side = random.randint(5, 7)
        inner = side - 2
        pattern = _make_pattern(inner, pattern_palette)

        slot_top = top_offset + slot_r * SLOT_SIZE
        slot_left = left_offset + slot_c * SLOT_SIZE
        local_r = random.randint(1, SLOT_SIZE - side - 1)
        local_c = random.randint(1, SLOT_SIZE - side - 1)
        r0 = slot_top + local_r
        c0 = slot_left + local_c
        r1 = r0 + side
        c1 = c0 + side

        color = frame_colors[idx]
        grid[r0:r1, c0] = color
        grid[r0:r1, c1 - 1] = color
        grid[r0, c0:c1] = color
        grid[r1 - 1, c0:c1] = color
        grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = pattern

    if legend_count == 1:
        return grid

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
