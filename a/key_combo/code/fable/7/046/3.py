from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
NON_WALL_COLORS = tuple(color for color in range(1, 10) if color != WALL)
GENERATION_ATTEMPTS = 400


def _find_gate_mask(grid: np.ndarray) -> np.ndarray:
    gate_mask = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    for r in range(h):
        for c in range(1, w - 1):
            value = int(grid[r, c])
            if value in (BG, WALL):
                continue
            if int(grid[r, c - 1]) == WALL and int(grid[r, c + 1]) == WALL:
                gate_mask[r, c] = True
    return gate_mask


def _drop_particle(static_col: np.ndarray, occupied: set[int], start_row: int, start_color: int) -> tuple[int, int, int]:
    row = int(start_row)
    color = int(start_color)
    passes = 0
    height = int(static_col.shape[0])

    while True:
        next_row = row + 1
        if next_row >= height:
            return row, color, passes
        if next_row in occupied:
            return row, color, passes

        cell = int(static_col[next_row])
        if cell == WALL:
            return row, color, passes
        if cell == BG:
            row = next_row
            continue

        gate_row = next_row
        final_gate_color = cell
        gate_count = 1
        while gate_row + 1 < height:
            below = int(static_col[gate_row + 1])
            if below in (BG, WALL):
                break
            gate_row += 1
            final_gate_color = below
            gate_count += 1

        landing_row = gate_row + 1
        if landing_row >= height:
            return row, color, passes
        if landing_row in occupied:
            return row, color, passes
        if int(static_col[landing_row]) == WALL:
            return row, color, passes

        row = gate_row
        color = final_gate_color
        passes += gate_count


def _transform(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gate_mask = _find_gate_mask(g)
    particle_mask = (g != BG) & (g != WALL) & ~gate_mask

    static = g.copy()
    static[particle_mask] = BG
    out = static.copy()

    meta: list[dict[str, int]] = []
    h, w = g.shape
    for c in range(w):
        starts = [(int(r), int(g[r, c])) for r in np.where(particle_mask[:, c])[0]]
        starts.sort(reverse=True)
        occupied: set[int] = set()
        static_col = static[:, c]
        for start_row, start_color in starts:
            final_row, final_color, passes = _drop_particle(static_col, occupied, start_row, start_color)
            assert int(static[final_row, c]) == BG, "particle cannot settle on a wall or gate"
            out[final_row, c] = final_color
            occupied.add(final_row)
            meta.append(
                {
                    "col": int(c),
                    "start_row": int(start_row),
                    "final_row": int(final_row),
                    "start_color": int(start_color),
                    "final_color": int(final_color),
                    "passes": int(passes),
                }
            )

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each colored pixel down its own column through grey walls, recoloring it to the last gate it successfully passes.

    Input structure:
    - The grid has a black background, several grey horizontal wall segments, and one colored gate embedded inside each wall.
    - Other colored cells are falling single-pixel particles; gate colors are distinct from particle colors.
    - Walls and gates may overlap by column across multiple rows, creating columns with blocks, single gates, or cascaded gates.

    Rule:
    - Process each column independently under vertical gravity.
    - A particle stops above a grey wall, above an already settled particle, or on the bottom row.
    - When the path below contains one or more consecutive gate cells and there is open space below that gate chain, the particle passes through all of them, recolors to the last gate color, and keeps falling.
    - Walls and gate cells stay fixed; only the falling particles move.

    Key insight:
    - The final color of a particle depends on its descent history, not its input color: a column acts like a stack of solid barriers and pass-through recoloring checkpoints.
    """
    return _transform(grid)


def _sample_wall_rows(n_walls: int) -> tuple[int, list[int]] | None:
    top_row = random.randint(5, 8)
    rows = [top_row]
    for _ in range(n_walls - 1):
        rows.append(rows[-1] + random.randint(2, 4))
    height = rows[-1] + random.randint(3, 6)
    if height > 30:
        return None
    return height, rows


def _sample_gate_extent(center: int, left_limit: int, right_limit: int) -> tuple[int, int] | None:
    left_max = min(3, center - left_limit)
    right_max = min(3, right_limit - center)
    if left_max < 1 or right_max < 1:
        return None
    left = random.randint(1, left_max)
    right = random.randint(1, right_max)
    return center - left, center + right


def _place_wall(grid: np.ndarray, row: int, start: int, end: int, gate_col: int, gate_color: int) -> None:
    grid[row, start : end + 1] = WALL
    grid[row, gate_col] = int(gate_color)


def _choose_palette(n_walls: int) -> tuple[list[int], list[int]]:
    colors = list(NON_WALL_COLORS)
    random.shuffle(colors)
    gate_colors = colors[:n_walls]
    particle_pool = colors[n_walls:]
    particle_color_count = random.randint(2, min(3, len(particle_pool)))
    particle_colors = particle_pool[:particle_color_count]
    return gate_colors, particle_colors


def _choose_particle_rows(limit_row: int, count: int) -> list[int] | None:
    available = list(range(limit_row))
    if len(available) < count:
        return None
    chosen = random.sample(available, count)
    chosen.sort()
    return chosen


def _segments_covering(segments: list[tuple[int, int]], col: int) -> bool:
    return any(start <= col <= end for start, end in segments)


def _build_candidate() -> np.ndarray | None:
    n_walls = random.randint(3, 4)
    dims = _sample_wall_rows(n_walls)
    if dims is None:
        return None
    height, wall_rows = dims

    width = random.randint(12, 18)
    gate_colors, particle_colors = _choose_palette(n_walls)

    cascade_col = random.randint(2, width // 2 - 1)

    cascade_segments: list[tuple[int, int]] = []
    for _ in range(2):
        seg = _sample_gate_extent(cascade_col, 1, width - 2)
        if seg is None:
            return None
        cascade_segments.append(seg)

    cascade_hi = max(end for _, end in cascade_segments)
    single_choices = list(range(cascade_hi + 3, width - 2))
    if not single_choices:
        return None
    single_col = random.choice(single_choices)
    single_seg = _sample_gate_extent(single_col, cascade_hi + 2, width - 2)
    if single_seg is None:
        return None

    wall_specs = [
        (wall_rows[0], cascade_segments[0][0], cascade_segments[0][1], cascade_col, gate_colors[0]),
        (wall_rows[1], cascade_segments[1][0], cascade_segments[1][1], cascade_col, gate_colors[1]),
        (wall_rows[2], single_seg[0], single_seg[1], single_col, gate_colors[2]),
    ]

    if n_walls == 4:
        if random.random() < 0.5:
            extra_col = cascade_col
            seg = _sample_gate_extent(extra_col, 1, width - 2)
        else:
            extra_col = single_col
            seg = _sample_gate_extent(extra_col, single_seg[0], width - 2)
        if seg is None:
            return None
        wall_specs.append((wall_rows[3], seg[0], seg[1], extra_col, gate_colors[3]))

    grid = np.zeros((height, width), dtype=int)
    segments = []
    for row, start, end, gate_col, gate_color in wall_specs:
        _place_wall(grid, row, start, end, gate_col, gate_color)
        segments.append((start, end))

    blocked_col = cascade_col - 1
    if blocked_col < wall_specs[0][1]:
        blocked_col = cascade_col + 1
    if not (wall_specs[0][1] <= blocked_col <= wall_specs[0][2]) or blocked_col == cascade_col:
        return None

    free_columns = [
        c for c in range(1, width - 1) if not _segments_covering(segments, c)
    ]
    free_col = random.choice(free_columns) if free_columns and random.random() < 0.7 else None

    particle_plan: list[tuple[int, int]] = [
        (cascade_col, random.randint(2, 3)),
        (blocked_col, random.randint(1, 2)),
        (single_col, random.randint(1, 2)),
    ]
    if free_col is not None:
        particle_plan.append((free_col, 1))

    top_limit = wall_rows[0]
    for col, count in particle_plan:
        rows = _choose_particle_rows(top_limit, count)
        if rows is None:
            return None
        for row in rows:
            grid[row, col] = random.choice(particle_colors)

    if random.random() < 0.5:
        grid = np.fliplr(grid)

    out, meta = _transform(grid, return_meta=True)
    if not np.array_equal(main(grid), out):
        return None

    passes = [item["passes"] for item in meta]
    columns = [item["col"] for item in meta]
    if not any(p >= 2 for p in passes):
        return None
    if not any(p == 1 for p in passes):
        return None
    if not any(p == 0 and item["final_row"] < height - 1 for p, item in zip(passes, meta)):
        return None
    if not any(item["final_color"] != item["start_color"] for item in meta):
        return None
    if not any(columns.count(col) >= 2 for col in set(columns)):
        return None
    if int(np.count_nonzero(out != grid)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
