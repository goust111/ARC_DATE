from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ROOM_GRID_CHOICES = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2))
MAX_GEN_RETRIES = 400
PATTERN_LIBRARY = tuple(
    np.array(bits, dtype=bool)
    for bits in (
        [[1, 0], [0, 1], [1, 0]],
        [[0, 1], [1, 0], [0, 1]],
        [[1, 0], [1, 0], [1, 0]],
        [[0, 1], [0, 1], [0, 1]],
        [[1, 1], [0, 0], [1, 1]],
        [[0, 0], [1, 1], [0, 0]],
        [[1, 0], [0, 1], [0, 1]],
        [[0, 1], [1, 0], [1, 0]],
        [[1, 1], [1, 0], [0, 0]],
        [[0, 0], [0, 1], [1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [1, 0]],
    )
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_legend(grid: np.ndarray) -> dict[int, np.ndarray]:
    band = np.asarray(grid[-3:, :], dtype=int)
    _, w = band.shape
    candidates: list[int] = []

    for c in range(w - 2):
        color = int(band[0, c])
        if color in (BG, WALL):
            continue
        if not np.all(band[:, c] == color):
            continue
        tile = band[:, c + 1 : c + 3]
        vals = {int(v) for v in tile.flat}
        if not vals.issubset({BG, color}) or BG not in vals or color not in vals:
            continue
        candidates.append(c)

    assert candidates, "legend key columns not found"

    best_run: list[int] = []
    run: list[int] = [candidates[0]]
    for c in candidates[1:]:
        if c - run[-1] == 3:
            run.append(c)
        else:
            if len(run) > len(best_run):
                best_run = run
            run = [c]
    if len(run) > len(best_run):
        best_run = run

    assert len(best_run) >= 2, "legend must contain at least two entries"

    mapping: dict[int, np.ndarray] = {}
    for c in best_run:
        color = int(band[0, c])
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = band[:, c + 1 : c + 3] == color
    return mapping


def _find_rooms(grid_wo_legend: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid_wo_legend, dtype=int)
    h, w = g.shape
    rooms: list[np.ndarray] = []

    for comp in _components(g != WALL):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 1] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == w - 1):
            continue
        seeds = [(int(r), int(c)) for r, c in comp if int(g[r, c]) != BG]
        if not seeds:
            continue
        assert len(seeds) == 1, "each room must contain exactly one seed pixel"
        rooms.append(comp)

    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed room with the periodic legend pattern keyed by its seed color.

    Input structure:
    - Background is black, and grey walls partition the upper part of the grid into closed rectangular rooms.
    - Every room interior contains exactly one isolated non-black seed pixel, with all other interior cells black.
    - The bottom three rows contain a legend band made of 3-column entries: a solid color key column followed by a 2-column-by-3-row binary pattern using that same color and black.

    Rule:
    - Read the bottom legend to build a mapping from seed color to its 3x2 boolean pattern tile.
    - Ignore the legend band, then find every non-wall room interior that is enclosed away from the grid boundary.
    - For each room, use its seed color to select the matching tile and periodically fill the room, aligned so the room's top-left interior cell is tile phase (0, 0). The seed pixel is overwritten by the tiling.

    Key insight:
    - The seed color is only a lookup key into the bottom legend; the actual fill comes from periodically repeating the legend's binary pattern with phase anchored at the room's own top-left interior corner.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _extract_legend(g)
    rooms = _find_rooms(g[:-3, :])

    out = g.copy()
    for room in rooms:
        rows = room[:, 0]
        cols = room[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        seed_values = [int(g[r, c]) for r, c in room if int(g[r, c]) != BG]
        seed_color = seed_values[0]
        assert seed_color in mapping, "room seed color must appear in the legend"
        tile = mapping[seed_color]
        mask = tile[(rows - r0) % 3, (cols - c0) % 2]
        out[rows, cols] = np.where(mask, seed_color, BG)

    return out


def _room_slices(row_heights: list[int], col_widths: list[int], top: int, left: int) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]]:
    row_walls = [top]
    cur = top
    for height in row_heights:
        cur += height + 1
        row_walls.append(cur)

    col_walls = [left]
    cur = left
    for width in col_widths:
        cur += width + 1
        col_walls.append(cur)

    rooms: list[tuple[int, int, int, int]] = []
    for i in range(len(row_heights)):
        for j in range(len(col_widths)):
            rooms.append((row_walls[i] + 1, row_walls[i + 1], col_walls[j] + 1, col_walls[j + 1]))
    return row_walls, col_walls, rooms


def _draw_walls(grid: np.ndarray, row_walls: list[int], col_walls: list[int]) -> None:
    r0, r1 = row_walls[0], row_walls[-1]
    c0, c1 = col_walls[0], col_walls[-1]
    for r in row_walls:
        grid[r, c0 : c1 + 1] = WALL
    for c in col_walls:
        grid[r0 : r1 + 1, c] = WALL


def _assign_room_colors(rng: random.Random, room_count: int) -> tuple[list[int], dict[int, np.ndarray]]:
    used_count = rng.randint(2, min(4, room_count, len(PALETTE_COLORS)))
    extra_count = rng.randint(0, min(2, len(PALETTE_COLORS) - used_count, 5 - used_count))
    key_count = used_count + extra_count

    key_colors = rng.sample(PALETTE_COLORS, key_count)
    used_colors = rng.sample(key_colors, used_count)
    patterns = rng.sample(PATTERN_LIBRARY, key_count)
    mapping = {color: pattern.copy() for color, pattern in zip(key_colors, patterns)}

    room_colors = used_colors[:]
    while len(room_colors) < room_count:
        room_colors.append(rng.choice(used_colors))
    rng.shuffle(room_colors)

    return room_colors, mapping


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows, room_cols = rng.choice(ROOM_GRID_CHOICES)
    row_heights = [rng.randint(2, 6) for _ in range(room_rows)]
    col_widths = [rng.randint(2, 6) for _ in range(room_cols)]

    if all(height % 3 == 0 for height in row_heights):
        row_heights[rng.randrange(room_rows)] = rng.choice((2, 4, 5))
    if all(width % 2 == 0 for width in col_widths):
        col_widths[rng.randrange(room_cols)] = rng.choice((3, 5))

    room_count = room_rows * room_cols
    room_colors, mapping = _assign_room_colors(rng, room_count)

    block_h = 1 + sum(row_heights) + room_rows
    block_w = 1 + sum(col_widths) + room_cols
    legend_w = 3 * len(mapping)

    top_pad = rng.randint(0, 3)
    gap = rng.randint(1, 3)
    height = top_pad + block_h + gap + 3
    min_width = max(block_w, legend_w)
    if height > 30 or min_width > 30:
        return None

    width = rng.randint(min_width, min(30, min_width + 6))
    room_left = rng.randint(0, width - block_w)
    legend_left = rng.randint(0, width - legend_w)

    grid = np.full((height, width), BG, dtype=int)
    row_walls, col_walls, rooms = _room_slices(row_heights, col_widths, top_pad, room_left)
    _draw_walls(grid, row_walls, col_walls)

    for (r0, r1, c0, c1), color in zip(rooms, room_colors):
        seed_r = rng.randrange(r0, r1)
        seed_c = rng.randrange(c0, c1)
        grid[seed_r, seed_c] = color

    legend_top = height - 3
    for idx, (color, pattern) in enumerate(mapping.items()):
        c0 = legend_left + 3 * idx
        grid[legend_top:height, c0] = color
        patch = grid[legend_top:height, c0 + 1 : c0 + 3]
        patch[pattern] = color

    try:
        parsed = _extract_legend(grid)
        rooms_found = _find_rooms(grid[:-3, :])
        if len(rooms_found) != room_count:
            return None
        if set(parsed) != set(mapping):
            return None
        for color, pattern in mapping.items():
            if not np.array_equal(parsed[color], pattern):
                return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_RETRIES} retries")
