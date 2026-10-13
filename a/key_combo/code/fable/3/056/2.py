from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


CELL_SIZE = 7
SEP_THICKNESS = 1
CENTER = CELL_SIZE // 2

DIRECTIONS = ("down", "right", "up", "left")
DIR_TO_DELTA = {
    "down": (1, 0),
    "right": (0, 1),
    "up": (-1, 0),
    "left": (0, -1),
}
ROTATION_K = {
    "down": 0,
    "right": 1,
    "up": 2,
    "left": 3,
}

SEPARATOR_DENSITY = 0.85
GENERATION_RETRIES = 400

BASE_SOLID_MASK = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
BASE_SOLID_MASK[1:, 1] = True
BASE_SOLID_MASK[1:, 5] = True
BASE_SOLID_MASK[6, 1:6] = True
BASE_HOLE = (6, CENTER)
BASE_SLOTS = [(r, CENTER) for r in range(CELL_SIZE - 1)]


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _rotate_coord(coord: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    for _ in range(k % 4):
        r, c = CELL_SIZE - 1 - c, r
    return int(r), int(c)


def _cup_mask(direction: str, leak: bool) -> np.ndarray:
    mask = BASE_SOLID_MASK.copy()
    if leak:
        mask[BASE_HOLE] = False
    return np.rot90(mask, ROTATION_K[direction])


def _ordered_slots_local(direction: str) -> list[tuple[int, int]]:
    k = ROTATION_K[direction]
    return [_rotate_coord(coord, k) for coord in BASE_SLOTS]


def _hole_local(direction: str) -> tuple[int, int]:
    return _rotate_coord(BASE_HOLE, ROTATION_K[direction])


TEMPLATE_INFO: dict[bytes, tuple[str, bool]] = {}
for _direction in DIRECTIONS:
    for _leak in (False, True):
        TEMPLATE_INFO[_cup_mask(_direction, _leak).tobytes()] = (_direction, _leak)


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(separator_lines):
        assert start < line < size - 1, "separator lines must split non-empty regions"
        regions.append((start, line - 1))
        start = line + 1
    assert start < size, "expected a trailing region after the final separator"
    regions.append((start, size - 1))
    return regions


def _detect_separator(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    row_threshold = math.ceil(SEPARATOR_DENSITY * width)
    col_threshold = math.ceil(SEPARATOR_DENSITY * height)

    candidates = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(1, height - 1) if int(np.sum(grid[r] == color)) >= row_threshold]
        cols = [c for c in range(1, width - 1) if int(np.sum(grid[:, c] == color)) >= col_threshold]
        if not rows or not cols:
            continue
        try:
            row_regions = _regions_between(height, rows)
            col_regions = _regions_between(width, cols)
        except AssertionError:
            continue
        if {r1 - r0 + 1 for r0, r1 in row_regions} != {CELL_SIZE}:
            continue
        if {c1 - c0 + 1 for c0, c1 in col_regions} != {CELL_SIZE}:
            continue
        score = sum(int(np.sum(grid[r] == color)) for r in rows) + sum(
            int(np.sum(grid[:, c] == color)) for c in cols
        )
        candidates.append((score, color, rows, cols))

    assert candidates, "expected a separator color with near-solid rows and columns"
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0], "separator color must be structurally unique"
    _score, color, rows, cols = candidates[0]
    return int(color), rows, cols


def _cell_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    sep_color, row_lines, col_lines = _detect_separator(grid)
    row_regions = _regions_between(grid.shape[0], row_lines)
    col_regions = _regions_between(grid.shape[1], col_lines)
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_regions}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert row_sizes == {CELL_SIZE} and col_sizes == {CELL_SIZE}, "expected fixed-size 7x7 cup cells"
    return sep_color, row_regions, col_regions


def _cell_color_and_info(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    sep_color: int,
    bg_color: int,
) -> tuple[int, dict[tuple[int, int], dict]]:
    cell_slices: dict[tuple[int, int], tuple[slice, slice]] = {}
    counts: Counter[int] = Counter()
    for rr, (r0, r1) in enumerate(row_regions):
        for cc, (c0, c1) in enumerate(col_regions):
            rs = slice(r0, r1 + 1)
            cs = slice(c0, c1 + 1)
            cell_slices[(rr, cc)] = (rs, cs)
            cell = grid[rs, cs]
            for value, count in zip(*np.unique(cell, return_counts=True)):
                value = int(value)
                if value not in (bg_color, sep_color):
                    counts[value] += int(count)

    assert counts, "expected a visible cup color"
    cup_color = max(counts, key=counts.get)

    info: dict[tuple[int, int], dict] = {}
    for key, (rs, cs) in cell_slices.items():
        cell = grid[rs, cs]
        cup_mask = (cell == cup_color)
        match = TEMPLATE_INFO.get(cup_mask.tobytes())
        assert match is not None, "each cell must contain one recognizable cup template"
        direction, leak = match
        local_slots = _ordered_slots_local(direction)
        slots = [(rs.start + r, cs.start + c) for r, c in local_slots]
        slot_index = {slot: idx for idx, slot in enumerate(slots)}
        info[key] = {
            "direction": direction,
            "leak": leak,
            "slots": slots,
            "slot_index": slot_index,
            "slice_r": rs,
            "slice_c": cs,
        }
    return int(cup_color), info


def _trace_sink(start: tuple[int, int], cells: dict[tuple[int, int], dict], shape: tuple[int, int]) -> tuple[int, int]:
    n_rows, n_cols = shape
    seen = set()
    cur = start
    while cells[cur]["leak"]:
        assert cur not in seen, "leak chains must terminate"
        seen.add(cur)
        dr, dc = DIR_TO_DELTA[cells[cur]["direction"]]
        nxt = (cur[0] + dr, cur[1] + dc)
        assert 0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols, "leak must point to a neighboring cell"
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let every scattered pixel fall through the separator grid by following each cup's local gravity, leaking into the next cell until a solid cup catches it.

    Input structure:
    - Near-solid separator rows and columns of one color partition the canvas into equal 7x7 cells.
    - Each cell contains one same-colored U-shaped cup, rotated so its open side points up, down, left, or right.
    - Some cups are solid, while others miss the center of their bottom wall and the touching separator line has a matching hole.
    - Colored single pixels appear only on the mouth-side channel of some cups.

    Rule:
    - In every occupied cell, read the pixels in local gravity order: the one already closest to the cup bottom falls first.
    - A solid cup keeps that ordered chain and stacks it against its bottom side.
    - A leaking cup passes the whole chain through its hole into the adjacent cell in the cup-bottom direction; the next cell then uses its own gravity, and the relay can repeat.
    - Cups, separator lines, and holes stay fixed; only the colored pixels move.

    Key insight:
    - The cup orientation is a per-cell gravity dial, and a leak cup is not a destination but a router: follow the hole from cell to cell, then stack the preserved color order only in the first solid cup reached by that chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg_color = _most_common(g)
    sep_color, row_regions, col_regions = _cell_regions(g)
    cup_color, cells = _cell_color_and_info(g, row_regions, col_regions, sep_color, bg_color)

    particle_mask = (g != bg_color) & (g != sep_color) & (g != cup_color)
    out = g.copy()
    out[particle_mask] = bg_color

    sink_loads: dict[tuple[int, int], list[int]] = {}
    for cell_key, cell_info in cells.items():
        rs = cell_info["slice_r"]
        cs = cell_info["slice_c"]
        cell_particles = np.argwhere(particle_mask[rs, cs])
        if cell_particles.size == 0:
            continue

        colors_with_order = []
        for local_r, local_c in cell_particles:
            coord = (rs.start + int(local_r), cs.start + int(local_c))
            assert coord in cell_info["slot_index"], "particles must lie on the cup channel"
            colors_with_order.append((cell_info["slot_index"][coord], int(g[coord])))

        colors_with_order.sort(key=lambda item: item[0], reverse=True)
        chain_colors = [color for _slot, color in colors_with_order]
        sink = _trace_sink(cell_key, cells, (len(row_regions), len(col_regions)))
        assert sink not in sink_loads, "different chains must not merge into the same sink cup"
        sink_loads[sink] = chain_colors

    for sink, chain_colors in sink_loads.items():
        sink_slots = cells[sink]["slots"]
        assert len(chain_colors) <= len(sink_slots), "sink cup does not have enough visible stack slots"
        target_slots = list(reversed(sink_slots[-len(chain_colors) :]))
        for color, (r, c) in zip(chain_colors, target_slots):
            out[r, c] = int(color)

    return out


def _global_cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    return cell[0] * (CELL_SIZE + SEP_THICKNESS), cell[1] * (CELL_SIZE + SEP_THICKNESS)


def _global_slots(cell: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0 = _global_cell_origin(cell)
    return [(r0 + r, c0 + c) for r, c in _ordered_slots_local(direction)]


def _paint_separator_grid(grid: np.ndarray, sep_color: int, n_rows: int, n_cols: int) -> tuple[list[int], list[int]]:
    row_lines = [CELL_SIZE + i * (CELL_SIZE + SEP_THICKNESS) for i in range(n_rows - 1)]
    col_lines = [CELL_SIZE + i * (CELL_SIZE + SEP_THICKNESS) for i in range(n_cols - 1)]
    for row in row_lines:
        grid[row, :] = sep_color
    for col in col_lines:
        grid[:, col] = sep_color
    return row_lines, col_lines


def _paint_cup(grid: np.ndarray, cell: tuple[int, int], cup_color: int, direction: str, leak: bool) -> None:
    r0, c0 = _global_cell_origin(cell)
    mask = _cup_mask(direction, leak)
    cell_view = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
    cell_view[mask] = cup_color


def _open_separator_hole(grid: np.ndarray, cell: tuple[int, int], direction: str, bg_color: int) -> None:
    r0, c0 = _global_cell_origin(cell)
    hole_r, hole_c = _hole_local(direction)
    dr, dc = DIR_TO_DELTA[direction]
    grid[r0 + hole_r + dr, c0 + hole_c + dc] = bg_color


def _neighbors(cell: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR_TO_DELTA.values():
        rr, cc = cell[0] + dr, cell[1] + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            out.append((rr, cc))
    return out


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for direction, delta in DIR_TO_DELTA.items():
        if delta == (dr, dc):
            return direction
    raise ValueError("cells must be orthogonal neighbors")


def _sample_turn_path(n_rows: int, n_cols: int, used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    paths: list[list[tuple[int, int]]] = []
    deltas = list(DIR_TO_DELTA.values())
    for mid_r in range(n_rows):
        for mid_c in range(n_cols):
            mid = (mid_r, mid_c)
            if mid in used:
                continue
            for dr1, dc1 in deltas:
                for dr2, dc2 in deltas:
                    if abs(dr1) + abs(dc1) != 1 or abs(dr2) + abs(dc2) != 1:
                        continue
                    if dr1 == dr2 or dc1 == dc2:
                        continue
                    start = (mid_r + dr1, mid_c + dc1)
                    end = (mid_r + dr2, mid_c + dc2)
                    if not (0 <= start[0] < n_rows and 0 <= start[1] < n_cols):
                        continue
                    if not (0 <= end[0] < n_rows and 0 <= end[1] < n_cols):
                        continue
                    if start in used or end in used or start == end:
                        continue
                    path = [start, mid, end]
                    if random.random() < 0.5:
                        path = list(reversed(path))
                    paths.append(path)
    if not paths:
        return None
    return random.choice(paths)


def _sample_simple_path(n_rows: int, n_cols: int, used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    remaining = [(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) not in used]
    if not remaining:
        return None
    start = random.choice(remaining)
    free_neighbors = [nb for nb in _neighbors(start, n_rows, n_cols) if nb not in used]
    if free_neighbors and random.random() < 0.55:
        return [start, random.choice(free_neighbors)]
    return [start]


def _build_paths(n_rows: int, n_cols: int) -> list[list[tuple[int, int]]]:
    used: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    turn_path = _sample_turn_path(n_rows, n_cols, used)
    if turn_path is not None:
        paths.append(turn_path)
        used.update(turn_path)

    max_extra = min(2, n_rows * n_cols - len(used))
    for _ in range(random.randint(0, max_extra)):
        path = _sample_simple_path(n_rows, n_cols, used)
        if path is None:
            break
        paths.append(path)
        used.update(path)

    if not paths:
        fallback = _sample_simple_path(n_rows, n_cols, used)
        assert fallback is not None
        paths.append(fallback)

    return paths


def _sample_chain_colors(count: int, palette: list[int]) -> list[int]:
    assert palette
    if count == 1:
        return [random.choice(palette)]

    unique_count = random.randint(2, min(count, len(palette)))
    colors = random.sample(palette, unique_count)
    while len(colors) < count:
        colors.append(random.choice(colors))
    random.shuffle(colors)
    return colors


def _source_slot_indices(count: int) -> list[int]:
    choices = list(range(0, 4))
    indices = random.sample(choices, count)
    return sorted(indices)


def _render_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 3)
    n_cols = random.randint(2, 3)
    height = n_rows * CELL_SIZE + (n_rows - 1) * SEP_THICKNESS
    width = n_cols * CELL_SIZE + (n_cols - 1) * SEP_THICKNESS

    bg_color, sep_color, cup_color = random.sample(range(10), 3)
    particle_palette = [color for color in range(10) if color not in (bg_color, sep_color, cup_color)]
    grid = np.full((height, width), bg_color, dtype=int)
    _paint_separator_grid(grid, sep_color, n_rows, n_cols)

    paths = _build_paths(n_rows, n_cols)
    if not any(len(path) >= 3 for path in paths):
        return None

    cell_state = {
        (r, c): {"direction": random.choice(DIRECTIONS), "leak": False}
        for r in range(n_rows)
        for c in range(n_cols)
    }

    source_particles: dict[tuple[int, int], list[int]] = {}
    for path_index, path in enumerate(paths):
        for i, cell in enumerate(path[:-1]):
            cell_state[cell]["direction"] = _direction_between(cell, path[i + 1])
            cell_state[cell]["leak"] = True

        sink = path[-1]
        if len(path) >= 2 and path_index == 0:
            incoming = _direction_between(path[-2], sink)
            sink_choices = [direction for direction in DIRECTIONS if direction != incoming]
            cell_state[sink]["direction"] = random.choice(sink_choices)
        else:
            cell_state[sink]["direction"] = random.choice(DIRECTIONS)
        cell_state[sink]["leak"] = False

        max_count = 4 if path_index == 0 else 3
        min_count = 2 if path_index == 0 else 1
        count = random.randint(min_count, max_count)
        source_particles[path[0]] = _sample_chain_colors(count, particle_palette)

    first_path = paths[0]
    turn_dirs = [
        _direction_between(first_path[i], first_path[i + 1])
        for i in range(len(first_path) - 1)
    ]
    if len(set(turn_dirs)) < 2:
        return None

    for cell, state in cell_state.items():
        _paint_cup(grid, cell, cup_color, state["direction"], state["leak"])
        if state["leak"]:
            _open_separator_hole(grid, cell, state["direction"], bg_color)

    for cell, colors in source_particles.items():
        slots = _global_slots(cell, cell_state[cell]["direction"])
        for slot_index, color in zip(_source_slot_indices(len(colors)), colors):
            r, c = slots[slot_index]
            grid[r, c] = color

    moved = main(grid)
    if np.array_equal(grid, moved):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _render_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leak-cup chain input")
