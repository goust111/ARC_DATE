from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
COLOR_CYCLE = (2, 1, 3, 4, 7)  # red, blue, green, yellow, orange
CYCLE_INDEX = {color: idx for idx, color in enumerate(COLOR_CYCLE)}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400

BOX_ROWS = 3
BOX_COLS = 2
BOX_H_CHOICES = (7, 8)
BOX_W_CHOICES = (9, 10)
BOX_GAP_CHOICES = (1, 2)


def _neighbors(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    h, w = shape
    result: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _infer_grey(grid: np.ndarray) -> int:
    non_bg = {int(color) for color in np.unique(grid) if int(color) != BG}
    grey_candidates = [color for color in non_bg if color not in CYCLE_INDEX]
    assert len(grey_candidates) == 1
    return grey_candidates[0]


def _trace_channel(
    grid: np.ndarray,
    grey: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    shape = grid.shape
    start_neighbors = [pos for pos in _neighbors(start, shape) if int(grid[pos]) == grey]
    end_neighbors = [pos for pos in _neighbors(end, shape) if int(grid[pos]) == grey]
    assert len(start_neighbors) == 1
    assert len(end_neighbors) == 1

    path = [start]
    prev = start
    current = start_neighbors[0]
    visited = {start}

    while True:
        assert current not in visited
        path.append(current)
        visited.add(current)
        if current == end:
            break

        next_steps = []
        for pos in _neighbors(current, shape):
            if pos == prev:
                continue
            value = int(grid[pos])
            if value == grey or pos == end:
                next_steps.append(pos)
        assert len(next_steps) == 1
        prev, current = current, next_steps[0]

    return path


def _turn_count(path: list[tuple[int, int]]) -> int:
    steps = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    ]
    turns = sum(int(prev != cur) for prev, cur in zip(steps, steps[1:]))
    return int(turns)


def _collect_channels(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    grey = _infer_grey(g)
    gate_colors = sorted(int(color) for color in np.unique(g) if int(color) not in {BG, grey})
    assert gate_colors
    assert all(color in CYCLE_INDEX for color in gate_colors)

    channels: list[dict] = []
    traced_grey: set[tuple[int, int]] = set()
    for color in gate_colors:
        gates = [tuple(map(int, pos)) for pos in np.argwhere(g == color)]
        assert len(gates) == 2
        start, end = gates
        path = _trace_channel(g, grey, start, end)
        grey_cells = path[1:-1]
        assert all(int(g[cell]) == grey for cell in grey_cells)
        assert traced_grey.isdisjoint(grey_cells)
        traced_grey.update(grey_cells)
        turns = _turn_count(path)
        assert 0 <= turns <= 4
        channels.append(
            {
                "color": color,
                "gates": (start, end),
                "path": path,
                "turns": turns,
            }
        )

    all_grey = {tuple(map(int, pos)) for pos in np.argwhere(g == grey)}
    assert traced_grey == all_grey
    return channels


def main(grid: np.ndarray) -> np.ndarray:
    """Advance each gate pair along the fixed color cycle by its channel turn count."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for channel in _collect_channels(g):
        color = int(channel["color"])
        turns = int(channel["turns"])
        new_color = COLOR_CYCLE[(CYCLE_INDEX[color] + turns) % len(COLOR_CYCLE)]
        for r, c in channel["gates"]:
            out[r, c] = new_color

    return out


def _polyline_cells(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [vertices[0]]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        dr = int(np.sign(r1 - r0))
        dc = int(np.sign(c1 - c0))
        steps = abs(r1 - r0) + abs(c1 - c0)
        assert steps > 0
        for _ in range(steps):
            last_r, last_c = cells[-1]
            cells.append((last_r + dr, last_c + dc))
    return cells


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1
    assert total >= parts
    if parts == 1:
        return [total]
    cuts = np.sort(RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    bounds = np.concatenate(([0], cuts, [total]))
    return [int(bounds[i + 1] - bounds[i]) for i in range(parts)]


def _sample_axis_lengths(count: int, max_extent: int) -> list[int] | None:
    if count == 0:
        return []
    min_total = 2 if count == 1 else count
    if max_extent < min_total:
        return None
    total = int(RNG.integers(min_total, max_extent + 1))
    return _positive_composition(total, count)


def _segment_counts(turns: int, start_horizontal: bool) -> tuple[int, int]:
    segments = turns + 1
    if start_horizontal:
        return (segments + 1) // 2, segments // 2
    return segments // 2, (segments + 1) // 2


def _sample_vertices(
    box: tuple[int, int, int, int],
    turns: int,
) -> list[tuple[int, int]] | None:
    box_r0, box_c0, box_h, box_w = box
    inner_r0 = box_r0 + 1
    inner_c0 = box_c0 + 1
    inner_r1 = box_r0 + box_h - 2
    inner_c1 = box_c0 + box_w - 2
    max_row_extent = inner_r1 - inner_r0
    max_col_extent = inner_c1 - inner_c0

    for _ in range(80):
        start_horizontal = bool(int(RNG.integers(0, 2)))
        h_count, v_count = _segment_counts(turns, start_horizontal)
        h_lengths = _sample_axis_lengths(h_count, max_col_extent)
        v_lengths = _sample_axis_lengths(v_count, max_row_extent)
        if h_lengths is None or v_lengths is None:
            continue

        row_extent = sum(v_lengths)
        col_extent = sum(h_lengths)
        row_sign = int(RNG.choice((-1, 1))) if row_extent else 1
        col_sign = int(RNG.choice((-1, 1))) if col_extent else 1

        if row_sign > 0:
            start_r = int(RNG.integers(inner_r0, inner_r1 - row_extent + 2))
        else:
            start_r = int(RNG.integers(inner_r0 + row_extent, inner_r1 + 1))
        if col_sign > 0:
            start_c = int(RNG.integers(inner_c0, inner_c1 - col_extent + 2))
        else:
            start_c = int(RNG.integers(inner_c0 + col_extent, inner_c1 + 1))

        vertices = [(start_r, start_c)]
        r, c = start_r, start_c
        h_idx = 0
        v_idx = 0
        horizontal = start_horizontal
        for _ in range(turns + 1):
            if horizontal:
                c += col_sign * h_lengths[h_idx]
                h_idx += 1
            else:
                r += row_sign * v_lengths[v_idx]
                v_idx += 1
            vertices.append((r, c))
            horizontal = not horizontal

        cells = _polyline_cells(vertices)
        if len(cells) != len(set(cells)):
            continue
        return vertices

    return None


def _paint_channel(
    grid: np.ndarray,
    color: int,
    vertices: list[tuple[int, int]],
) -> None:
    cells = _polyline_cells(vertices)
    start = cells[0]
    end = cells[-1]
    grid[start] = color
    grid[end] = color
    for cell in cells[1:-1]:
        grid[cell] = GREY


def _make_boxes(box_h: int, box_w: int, gap: int) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    height = 2 + BOX_ROWS * box_h + (BOX_ROWS - 1) * gap
    width = 2 + BOX_COLS * box_w + (BOX_COLS - 1) * gap
    boxes: list[tuple[int, int, int, int]] = []
    for br in range(BOX_ROWS):
        for bc in range(BOX_COLS):
            r0 = 1 + br * (box_h + gap)
            c0 = 1 + bc * (box_w + gap)
            boxes.append((r0, c0, box_h, box_w))
    return (height, width), boxes


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    variant = int(RNG.integers(0, 8))
    if variant < 4:
        return np.rot90(grid, variant).copy()
    return np.fliplr(np.rot90(grid, variant - 4)).copy()


def _gen_attempt() -> np.ndarray | None:
    box_h = int(RNG.choice(BOX_H_CHOICES))
    box_w = int(RNG.choice(BOX_W_CHOICES))
    gap = int(RNG.choice(BOX_GAP_CHOICES))
    (height, width), boxes = _make_boxes(box_h, box_w, gap)

    grid = np.full((height, width), BG, dtype=int)
    used_box_indices = RNG.choice(len(boxes), size=len(COLOR_CYCLE), replace=False)
    colors = list(COLOR_CYCLE)
    turns = list(range(len(COLOR_CYCLE)))
    RNG.shuffle(colors)
    RNG.shuffle(turns)

    for box_idx, color, turn_count in zip(used_box_indices, colors, turns):
        vertices = _sample_vertices(boxes[int(box_idx)], int(turn_count))
        if vertices is None:
            return None
        _paint_channel(grid, int(color), vertices)

    grid = _apply_random_symmetry(grid)

    channels = _collect_channels(grid)
    if len(channels) != len(COLOR_CYCLE):
        return None
    if sorted(int(channel["turns"]) for channel in channels) != list(range(len(COLOR_CYCLE))):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
