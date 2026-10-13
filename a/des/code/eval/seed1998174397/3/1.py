from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if coords.shape[0] != int(border.sum()):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.array_equal(sub == color, border):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if not np.any(interior == bg):
                continue
            seed_coords = np.argwhere(interior != bg)
            if seed_coords.shape[0] != 1:
                continue
            sr, sc = map(int, seed_coords[0])
            containers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "seed_color": int(interior[sr, sc]),
                }
            )

    assert containers, "expected at least one hollow rectangular container"
    return containers


def _find_legend_patterns(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h not in (2, 3) or w != h + 1:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        first_col = sub[:, 0]
        label_rows = np.flatnonzero(first_col != bg)
        if label_rows.size != 1:
            continue

        pattern = sub[:, 1:]
        if pattern.shape != (h, h) or np.any(pattern == bg):
            continue

        color_count = len(set(map(int, np.unique(pattern))))
        if color_count not in (2, 3):
            continue

        label_color = int(first_col[int(label_rows[0])])
        assert label_color not in mapping, "legend labels must be unique"
        mapping[label_color] = pattern.copy()

    assert mapping, "expected legend entries with label and pattern"
    return mapping


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % pattern.shape[0]
    cols = np.arange(width)[None, :] % pattern.shape[1]
    return pattern[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle using the tiled legend pattern keyed by its seed color.

    Input structure:
    - The background is the most common color.
    - Several hollow rectangular containers are drawn with a uniform border color.
    - Each container interior is background except for one seed pixel.
    - Legend entries are connected components made of one label pixel on the left and a full 2x2 or 3x3 pattern block on the right.

    Rule:
    - Read each legend entry as seed-color -> pattern.
    - For every container, find its seed color, look up the pattern, and tile it from the interior top-left corner.
    - The tiled pattern overwrites the original seed pixel.
    - Borders and legend stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    containers = _find_containers(g, bg)
    legend = _find_legend_patterns(g, bg)

    out = g.copy()
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        seed_color = int(container["seed_color"])
        assert seed_color in legend, "every seed color must appear in the legend"
        pattern = legend[seed_color]
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(pattern, interior_h, interior_w)
    return out


# --------------- generator ---------------

FRAME_COLOR = 5
GRID_HEIGHT_RANGE = (18, 31)
GRID_WIDTH_RANGE = (24, 35)
CONTAINER_COUNT_RANGE = (2, 5)
LEGEND_SIDE_OPTIONS = ("left", "right")
PATTERN_SIZE_OPTIONS = (2, 3)
PATTERN_COLOR_COUNT_OPTIONS = (2, 3)
LEGEND_STRIP_WIDTH = 5
LEGEND_GAP_RANGE = (1, 3)
LEGEND_ENTRY_GAP_RANGE = (1, 3)
CONTAINER_INNER_RANGE = (3, 9)
CONTAINER_PAD = 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _random_pattern(size: int, color_pool: list[int], used: set[tuple[int, ...]]) -> np.ndarray | None:
    options = [count for count in PATTERN_COLOR_COUNT_OPTIONS if count <= min(3, len(color_pool))]
    if not options:
        return None
    for _ in range(200):
        color_count = random.choice(options)
        chosen = random.sample(color_pool, color_count)
        flat = [random.choice(chosen) for _ in range(size * size)]
        if len(set(flat)) != color_count:
            continue
        pattern = np.array(flat, dtype=int).reshape(size, size)
        if not (pattern[:, 1:] != pattern[:, :-1]).any():
            continue
        if not (pattern[1:, :] != pattern[:-1, :]).any():
            continue
        signature = (size, *map(int, pattern.flatten()))
        if signature in used:
            continue
        used.add(signature)
        return pattern
    return None


def _sample_inner_size(pattern_size: int) -> tuple[int, int]:
    for _ in range(50):
        h = random.randint(*CONTAINER_INNER_RANGE)
        w = random.randint(*CONTAINER_INNER_RANGE)
        if random.random() < 0.5 and h % pattern_size == 0 and w % pattern_size == 0:
            continue
        return h, w
    return random.randint(*CONTAINER_INNER_RANGE), random.randint(*CONTAINER_INNER_RANGE)


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_color: int,
    seed_pos: tuple[int, int] | None = None,
) -> tuple[int, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR
    if seed_pos is None:
        seed_r = random.randint(top + 1, bottom - 1)
        seed_c = random.randint(left + 1, right - 1)
    else:
        seed_r, seed_c = seed_pos
    grid[seed_r, seed_c] = seed_color
    return int(seed_r), int(seed_c)


def _place_containers(
    grid: np.ndarray,
    area: tuple[int, int, int, int],
    seed_colors: list[int],
    pattern_sizes: list[int],
) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    _ = (h, w)
    area_r0, area_c0, area_r1, area_c1 = area
    placed_boxes: list[tuple[int, int, int, int]] = []
    out: list[tuple[int, int, int, int, int]] = []
    order = list(range(len(seed_colors)))
    random.shuffle(order)

    for idx in order:
        inner_h, inner_w = _sample_inner_size(pattern_sizes[idx])
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        min_r = area_r0
        max_r = area_r1 - outer_h + 1
        min_c = area_c0
        max_c = area_c1 - outer_w + 1
        if min_r > max_r or min_c > max_c:
            return []

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            padded = (
                max(0, top - CONTAINER_PAD),
                max(0, left - CONTAINER_PAD),
                min(grid.shape[0] - 1, bottom + CONTAINER_PAD),
                min(grid.shape[1] - 1, right + CONTAINER_PAD),
            )
            if any(_boxes_overlap(padded, other) for other in placed_boxes):
                continue
            placed_boxes.append(padded)
            out.append((idx, top, left, inner_h, inner_w))
            placed = True
            break
        if not placed:
            return []
    return out


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [color for color in range(10) if color != FRAME_COLOR]
    bg = random.choice(bg_choices)

    count = random.randint(*CONTAINER_COUNT_RANGE)
    usable = [color for color in range(10) if color not in (bg, FRAME_COLOR)]
    if len(usable) < count + 3:
        return None
    random.shuffle(usable)
    seed_colors = usable[:count]
    pattern_pool = usable[count:]
    if len(pattern_pool) < 3:
        return None

    pattern_sizes = [random.choice(PATTERN_SIZE_OPTIONS) for _ in range(count)]
    entry_gaps = [random.randint(*LEGEND_ENTRY_GAP_RANGE) for _ in range(max(0, count - 1))]
    legend_height = sum(pattern_sizes) + sum(entry_gaps)
    min_h = max(GRID_HEIGHT_RANGE[0], legend_height + 2)
    if min_h > GRID_HEIGHT_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    if grid_w <= LEGEND_STRIP_WIDTH + legend_gap + 8:
        return None

    side = random.choice(LEGEND_SIDE_OPTIONS)
    if side == "left":
        legend_c0 = 0
        main_area = (1, LEGEND_STRIP_WIDTH + legend_gap, grid_h - 2, grid_w - 2)
    else:
        legend_c0 = grid_w - LEGEND_STRIP_WIDTH
        main_area = (1, 1, grid_h - 2, legend_c0 - legend_gap - 1)
    if main_area[1] > main_area[3]:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = grid.copy()

    pattern_used: set[tuple[int, ...]] = set()
    legend_patterns: list[np.ndarray] = []
    for size in pattern_sizes:
        pattern = _random_pattern(size, pattern_pool, pattern_used)
        if pattern is None:
            return None
        legend_patterns.append(pattern)

    entries = list(range(count))
    random.shuffle(entries)
    top_margin = random.randint(0, grid_h - legend_height)
    current_r = top_margin
    for pos, idx in enumerate(entries):
        size = pattern_sizes[idx]
        entry_w = size + 1
        offset_c = random.randint(0, LEGEND_STRIP_WIDTH - entry_w)
        label_row = random.randint(0, size - 1)
        left = legend_c0 + offset_c
        pattern = legend_patterns[idx]
        grid[current_r + label_row, left] = seed_colors[idx]
        grid[current_r : current_r + size, left + 1 : left + 1 + size] = pattern
        expected[current_r + label_row, left] = seed_colors[idx]
        expected[current_r : current_r + size, left + 1 : left + 1 + size] = pattern
        if pos < len(entry_gaps):
            current_r += size + entry_gaps[pos]

    placements = _place_containers(grid, main_area, seed_colors, pattern_sizes)
    if len(placements) != count:
        return None

    for idx, top, left, inner_h, inner_w in placements:
        seed_pos = _draw_container(grid, top, left, inner_h, inner_w, seed_colors[idx])
        _draw_container(expected, top, left, inner_h, inner_w, seed_colors[idx], seed_pos=seed_pos)
        pattern = legend_patterns[idx]
        expected[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = _tile_pattern(pattern, inner_h, inner_w)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
