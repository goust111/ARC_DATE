from __future__ import annotations

import random

import numpy as np


WALL_COLOR = 5
ALL_COLORS = list(range(10))
MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _group_consecutive(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[int(indices[0])]]
    for idx in indices[1:]:
        idx = int(idx)
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _detect_layout(
    grid: np.ndarray,
) -> tuple[str, list[tuple[int, int]], int, tuple[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    full_cols = [c for c in range(w) if np.all(g[:, c] != bg)]
    full_rows = [r for r in range(h) if np.all(g[r, :] != bg)]
    col_groups = _group_consecutive(full_cols)
    row_groups = _group_consecutive(full_rows)

    if col_groups and all(len(group) == 2 for group in col_groups) and not row_groups:
        orientation = "vertical"
        walls = [(group[0], group[1]) for group in col_groups]
        stripe_mask = np.zeros_like(g, dtype=bool)
        for c0, c1 in walls:
            stripe_mask[:, c0 : c1 + 1] = True
        stripe_values = g[stripe_mask]
        particle_cells = np.argwhere((g != bg) & ~stripe_mask)
        assert particle_cells.shape[0] == 1, "expected one particle outside vertical walls"
        particle = tuple(map(int, particle_cells[0]))
        if particle[1] < walls[0][0]:
            direction = 1
            ordered_walls = walls
        else:
            assert particle[1] > walls[-1][1], "particle must start outside the wall set"
            direction = -1
            ordered_walls = list(reversed(walls))
    elif row_groups and all(len(group) == 2 for group in row_groups) and not col_groups:
        orientation = "horizontal"
        walls = [(group[0], group[1]) for group in row_groups]
        stripe_mask = np.zeros_like(g, dtype=bool)
        for r0, r1 in walls:
            stripe_mask[r0 : r1 + 1, :] = True
        stripe_values = g[stripe_mask]
        particle_cells = np.argwhere((g != bg) & ~stripe_mask)
        assert particle_cells.shape[0] == 1, "expected one particle outside horizontal walls"
        particle = tuple(map(int, particle_cells[0]))
        if particle[0] < walls[0][0]:
            direction = 1
            ordered_walls = walls
        else:
            assert particle[0] > walls[-1][1], "particle must start outside the wall set"
            direction = -1
            ordered_walls = list(reversed(walls))
    else:
        raise AssertionError("expected one set of parallel two-cell-thick walls")

    wall_values, wall_counts = np.unique(stripe_values, return_counts=True)
    wall_color = int(wall_values[np.argmax(wall_counts)])
    particle_color = int(g[particle])
    assert particle_color not in {bg, wall_color}
    return orientation, ordered_walls, wall_color, particle, particle_color, bg


def _paint_line(
    out: np.ndarray,
    src: np.ndarray,
    bg: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    r, c = map(int, start)
    er, ec = map(int, end)
    dr = 0 if r == er else (1 if er > r else -1)
    dc = 0 if c == ec else (1 if ec > c else -1)
    assert dr == 0 or dc == 0, "path segments must be axis-aligned"

    while (r, c) != (er, ec):
        r += dr
        c += dc
        if int(src[r, c]) == bg:
            out[r, c] = int(color)


def _find_vertical_door(
    grid: np.ndarray,
    wall: tuple[int, int],
    wall_color: int,
    bg: int,
    direction: int,
    current_color: int,
) -> tuple[int, int]:
    c0, c1 = map(int, wall)
    matches: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, c0])
        right = int(grid[r, c1])
        if left == wall_color and right == wall_color:
            continue
        assert left not in {bg, wall_color} and right not in {bg, wall_color}
        entry = left if direction > 0 else right
        exit_color = right if direction > 0 else left
        if entry == int(current_color):
            matches.append((r, exit_color))
    assert len(matches) == 1, "each wall must have exactly one door matching the current color"
    return matches[0]


def _find_horizontal_door(
    grid: np.ndarray,
    wall: tuple[int, int],
    wall_color: int,
    bg: int,
    direction: int,
    current_color: int,
) -> tuple[int, int]:
    r0, r1 = map(int, wall)
    matches: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        top = int(grid[r0, c])
        bottom = int(grid[r1, c])
        if top == wall_color and bottom == wall_color:
            continue
        assert top not in {bg, wall_color} and bottom not in {bg, wall_color}
        entry = top if direction > 0 else bottom
        exit_color = bottom if direction > 0 else top
        if entry == int(current_color):
            matches.append((c, exit_color))
    assert len(matches) == 1, "each wall must have exactly one door matching the current color"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Thread the particle through the only color-matching bicolor door in each thick wall and paint the journey segment by segment.

    Input structure:
    - The grid has one dominant background color, several parallel two-cell-thick grey walls, and one single-pixel particle outside the wall set.
    - Every wall spans the full grid in one orientation and contains several one-cell-wide bicolor doors crossing the wall thickness.
    - On each wall, exactly one door has an entry color equal to the particle's current color when that wall is reached.

    Rule:
    - Keep the particle, walls, and door cells unchanged.
    - Move perpendicular to the walls until the particle reaches the corridor cell just before the next wall, then slide along that wall to the unique matching door.
    - Paint only the traversed background cells with the particle's current color, pass through the two door cells, switch to the door's exit color, and continue; after the last wall, keep going straight to the grid edge.

    Key insight:
    - Each bicolor door is both a gate and a state update: the visible route is determined by repeatedly matching the current color to the next wall's entry color, not by nearest distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    orientation, walls, wall_color, particle, current_color, bg = _detect_layout(g)
    out = g.copy()
    current_pos = particle

    if orientation == "vertical":
        direction = 1 if particle[1] < walls[0][0] else -1
        for c0, c1 in walls:
            entry_col = c0 if direction > 0 else c1
            exit_col = c1 if direction > 0 else c0
            before_col = entry_col - direction
            _paint_line(out, g, bg, current_pos, (current_pos[0], before_col), current_color)
            door_row, next_color = _find_vertical_door(g, (c0, c1), wall_color, bg, direction, current_color)
            _paint_line(out, g, bg, (current_pos[0], before_col), (door_row, before_col), current_color)
            current_pos = (door_row, exit_col)
            current_color = next_color

        edge_col = g.shape[1] - 1 if direction > 0 else 0
        _paint_line(out, g, bg, current_pos, (current_pos[0], edge_col), current_color)
    else:
        direction = 1 if particle[0] < walls[0][0] else -1
        for r0, r1 in walls:
            entry_row = r0 if direction > 0 else r1
            exit_row = r1 if direction > 0 else r0
            before_row = entry_row - direction
            _paint_line(out, g, bg, current_pos, (before_row, current_pos[1]), current_color)
            door_col, next_color = _find_horizontal_door(g, (r0, r1), wall_color, bg, direction, current_color)
            _paint_line(out, g, bg, (before_row, current_pos[1]), (before_row, door_col), current_color)
            current_pos = (exit_row, door_col)
            current_color = next_color

        edge_row = g.shape[0] - 1 if direction > 0 else 0
        _paint_line(out, g, bg, current_pos, (edge_row, current_pos[1]), current_color)

    return out


def _sample_wall_axis(n_walls: int) -> tuple[int, list[tuple[int, int]]] | None:
    gaps = [random.randint(2, 5) for _ in range(n_walls + 1)]
    needed = 2 * n_walls + sum(gaps)
    if needed > 30:
        return None
    axis_len = random.randint(max(needed, 12), min(30, needed + 6))
    extra = axis_len - needed
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    cursor = gaps[0]
    walls = []
    for idx in range(n_walls):
        walls.append((cursor, cursor + 1))
        cursor += 2
        cursor += gaps[idx + 1]
    return axis_len, walls


def _sample_chain_colors(n_walls: int, palette: list[int]) -> list[int]:
    chain = [random.choice(palette)]
    for _ in range(n_walls):
        options = [color for color in palette if color != chain[-1]]
        chain.append(random.choice(options))
    return chain


def _sample_chain_positions(n_walls: int, span_len: int, start_pos: int) -> list[int]:
    usable = list(range(1, span_len - 1))
    if random.random() < 0.3:
        return [start_pos] * n_walls

    while True:
        positions = [random.choice(usable) for _ in range(n_walls)]
        prev = start_pos
        if any(pos != prev for pos, prev in zip(positions, [start_pos] + positions[:-1])):
            return positions


def _pick_decoy_pair(current_color: int, palette: list[int]) -> tuple[int, int]:
    entry = random.choice([color for color in palette if color != current_color])
    exit_color = random.choice([color for color in palette if color != entry])
    return entry, exit_color


def _gen_vertical() -> np.ndarray | None:
    n_walls = random.randint(2, 4)
    width_and_walls = _sample_wall_axis(n_walls)
    if width_and_walls is None:
        return None
    width, walls = width_and_walls
    height = random.randint(10, 24)
    direction = random.choice((1, -1))
    bg = random.choice([color for color in ALL_COLORS if color != WALL_COLOR])
    palette = random.sample([color for color in ALL_COLORS if color not in {bg, WALL_COLOR}], random.randint(4, 7))
    chain_colors = _sample_chain_colors(n_walls, palette)
    start_row = random.randint(1, height - 2)
    chain_rows = _sample_chain_positions(n_walls, height, start_row)

    grid = np.full((height, width), bg, dtype=int)
    for c0, c1 in walls:
        grid[:, c0 : c1 + 1] = WALL_COLOR

    particle_col = 0 if direction > 0 else width - 1
    grid[start_row, particle_col] = chain_colors[0]

    for wall_idx, (c0, c1) in enumerate(walls):
        total_doors = random.randint(2, min(4, height - 2))
        chain_row = chain_rows[wall_idx]
        door_rows = {chain_row}
        choices = [row for row in range(1, height - 1) if row != chain_row]
        while len(door_rows) < total_doors:
            door_rows.add(random.choice(choices))

        for row in door_rows:
            if row == chain_row:
                entry = chain_colors[wall_idx]
                exit_color = chain_colors[wall_idx + 1]
            else:
                entry, exit_color = _pick_decoy_pair(chain_colors[wall_idx], palette)

            if direction > 0:
                grid[row, c0] = entry
                grid[row, c1] = exit_color
            else:
                grid[row, c0] = exit_color
                grid[row, c1] = entry

    return grid


def _gen_horizontal() -> np.ndarray | None:
    n_walls = random.randint(2, 4)
    height_and_walls = _sample_wall_axis(n_walls)
    if height_and_walls is None:
        return None
    height, walls = height_and_walls
    width = random.randint(10, 24)
    direction = random.choice((1, -1))
    bg = random.choice([color for color in ALL_COLORS if color != WALL_COLOR])
    palette = random.sample([color for color in ALL_COLORS if color not in {bg, WALL_COLOR}], random.randint(4, 7))
    chain_colors = _sample_chain_colors(n_walls, palette)
    start_col = random.randint(1, width - 2)
    chain_cols = _sample_chain_positions(n_walls, width, start_col)

    grid = np.full((height, width), bg, dtype=int)
    for r0, r1 in walls:
        grid[r0 : r1 + 1, :] = WALL_COLOR

    particle_row = 0 if direction > 0 else height - 1
    grid[particle_row, start_col] = chain_colors[0]

    for wall_idx, (r0, r1) in enumerate(walls):
        total_doors = random.randint(2, min(4, width - 2))
        chain_col = chain_cols[wall_idx]
        door_cols = {chain_col}
        choices = [col for col in range(1, width - 1) if col != chain_col]
        while len(door_cols) < total_doors:
            door_cols.add(random.choice(choices))

        for col in door_cols:
            if col == chain_col:
                entry = chain_colors[wall_idx]
                exit_color = chain_colors[wall_idx + 1]
            else:
                entry, exit_color = _pick_decoy_pair(chain_colors[wall_idx], palette)

            if direction > 0:
                grid[r0, col] = entry
                grid[r1, col] = exit_color
            else:
                grid[r0, col] = exit_color
                grid[r1, col] = entry

    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _gen_vertical() if random.random() < 0.5 else _gen_horizontal()
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    bg = _get_bg(grid)
    changed = (grid == bg) & (out != grid)
    if not np.any(changed):
        return None
    if np.any(out[grid != bg] != grid[grid != bg]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
