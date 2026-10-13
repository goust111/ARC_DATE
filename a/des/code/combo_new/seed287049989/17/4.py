from __future__ import annotations

import random

import numpy as np


RED = 2
GREY = 5
MAROON = 9

DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
BASE_ARMS = (0, 1)

MIN_PANEL_SIZE = 5
MAX_PANEL_SIZE = 7
PANEL_COUNT_RANGE = (2, 4)
ARM_LENGTH_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (1, 5)
GEN_RETRIES = 400


def _mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in separators + [size]:
        if idx - prev > 1:
            spans.append((prev + 1, idx - 1))
        prev = idx
    return spans


def _separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        row_spans = _spans(h, rows)
        col_spans = _spans(w, cols)
        if not row_spans or not col_spans:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1 or row_sizes != col_sizes:
            continue
        candidates.append((color, row_spans, col_spans))
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _rotate_dir(direction: int, k: int) -> int:
    return (direction + k) % 4


def _walk(corner: tuple[int, int], direction: int, steps: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r0, c0 = corner
    return [(r0 + step * dr, c0 + step * dc) for step in range(1, steps + 1)]


def _l_cells(corner: tuple[int, int], lengths: dict[int, int]) -> set[tuple[int, int]]:
    cells = {corner}
    for direction, length in lengths.items():
        cells.update(_walk(corner, direction, length))
    return cells


def _extract_l(panel: np.ndarray) -> dict[str, object]:
    red_cells = {tuple(map(int, coord)) for coord in np.argwhere(panel == RED)}
    assert len(red_cells) >= 3, "every panel must contain a red L"

    corner = None
    arm_dirs: tuple[int, int] | None = None
    for cell in red_cells:
        neighbors = []
        r, c = cell
        for direction, (dr, dc) in DIRS.items():
            if (r + dr, c + dc) in red_cells:
                neighbors.append(direction)
        orthogonal = [(a, b) for a in neighbors for b in neighbors if a < b and (a - b) % 2 != 0]
        if orthogonal:
            assert corner is None, "L corner must be unique"
            corner = cell
            arm_dirs = orthogonal[0]
    assert corner is not None and arm_dirs is not None, "failed to locate the L corner"

    lengths: dict[int, int] = {}
    tips: dict[int, tuple[int, int]] = {}
    for direction in arm_dirs:
        step = 0
        dr, dc = DIRS[direction]
        while (corner[0] + (step + 1) * dr, corner[1] + (step + 1) * dc) in red_cells:
            step += 1
        assert step >= 1, "every L arm must have positive length"
        lengths[direction] = step
        tips[direction] = (corner[0] + step * dr, corner[1] + step * dc)

    assert _l_cells(corner, lengths) == red_cells, "red pixels must form a clean L"
    return {"corner": corner, "lengths": lengths, "tips": tips}


def _rotation_from_legend(legend_l: dict[str, object], panel_l: dict[str, object]) -> int:
    legend_lengths = legend_l["lengths"]
    panel_lengths = panel_l["lengths"]
    candidates = []
    for k in range(4):
        mapped = {_rotate_dir(direction, k): length for direction, length in legend_lengths.items()}
        if mapped == panel_lengths:
            candidates.append(k)
    assert len(candidates) == 1, "panel L must be a rotation of the legend L"
    return candidates[0]


def _active_direction(lengths: dict[int, int]) -> int:
    best = max(lengths.values())
    winners = [direction for direction, length in lengths.items() if length == best]
    if len(winners) == 1:
        return winners[0]

    vertical = [direction for direction in winners if direction in (0, 2)]
    assert len(vertical) == 1, "equal-arm Ls must have one vertical and one horizontal arm"
    return vertical[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel's maroon markers by using the legend L and the panel L's rotation state.

    Input structure:
    - A background-colored grid is partitioned into equal square panels by single-cell grey separator rows and columns.
    - The top-left panel is the legend: it contains one red L shape, and each arm tip has a distinct colored pixel immediately outside the tip.
    - Every other panel contains the same red L shape in a rotated orientation, plus one or more maroon marker pixels on background cells.

    Rule:
    - Read the legend panel as a map from arm direction to the color attached to that arm tip.
    - For every non-legend panel, identify how much its red L has been rotated relative to the legend L.
    - Choose the panel's active arm: use the longer arm, and if the two arms tie, use the vertical arm.
    - Inverse-rotate that active direction back into the legend orientation, look up the corresponding legend color, and repaint every maroon marker in that panel to that color.
    - Leave the legend colors, the red Ls, the background, and the grey separators unchanged.

    Key insight:
    - The legend is a direction-to-color lookup table, and each panel's L rotation decides which legend direction becomes active.
    """
    g = np.asarray(grid, dtype=int)
    separator, row_spans, col_spans = _separator_layout(g)
    bg = _mode(g[g != separator])

    lr0, lr1 = row_spans[0]
    lc0, lc1 = col_spans[0]
    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    legend_l = _extract_l(legend)

    legend_map: dict[int, int] = {}
    for direction, tip in legend_l["tips"].items():
        dr, dc = DIRS[direction]
        rr, cc = tip[0] + dr, tip[1] + dc
        color = int(legend[rr, cc])
        assert color not in {bg, separator, RED, MAROON}, "legend tip color must be distinct"
        legend_map[direction] = color
    assert len(legend_map) == 2

    out = g.copy()
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if i == 0 and j == 0:
                continue
            panel = g[r0 : r1 + 1, c0 : c1 + 1]
            panel_l = _extract_l(panel)
            rotation = _rotation_from_legend(legend_l, panel_l)
            active = _active_direction(panel_l["lengths"])
            legend_direction = (active - rotation) % 4
            color = legend_map[legend_direction]

            target = out[r0 : r1 + 1, c0 : c1 + 1]
            target[panel == MAROON] = color

    return out


def _panel_origin(index: int, size: int) -> int:
    return index * (size + 1)


def _panel_array(size: int, bg: int) -> np.ndarray:
    return np.full((size, size), bg, dtype=int)


def _valid_corners(
    size: int,
    lengths: dict[int, int],
    include_tip_colors: bool,
) -> list[tuple[int, int]]:
    offsets = {(0, 0)}
    for direction, length in lengths.items():
        dr, dc = DIRS[direction]
        reach = length + (1 if include_tip_colors else 0)
        for step in range(1, reach + 1):
            offsets.add((step * dr, step * dc))

    corners = []
    for r in range(size):
        for c in range(size):
            if all(0 <= r + dr < size and 0 <= c + dc < size for dr, dc in offsets):
                corners.append((r, c))
    return corners


def _paint_l(
    panel: np.ndarray,
    corner: tuple[int, int],
    lengths: dict[int, int],
) -> None:
    for rr, cc in _l_cells(corner, lengths):
        panel[rr, cc] = RED


def _build_panel(
    size: int,
    bg: int,
    lengths: dict[int, int],
    marker_count: int,
    rng: random.Random,
    tip_colors: dict[int, int] | None = None,
) -> np.ndarray:
    panel = _panel_array(size, bg)
    corners = _valid_corners(size, lengths, include_tip_colors=tip_colors is not None)
    assert corners, "panel must have a valid L placement"
    corner = rng.choice(corners)
    _paint_l(panel, corner, lengths)

    if tip_colors is not None:
        for direction, color in tip_colors.items():
            dr, dc = DIRS[direction]
            tip_r = corner[0] + (lengths[direction] + 1) * dr
            tip_c = corner[1] + (lengths[direction] + 1) * dc
            panel[tip_r, tip_c] = color

    free = [tuple(map(int, coord)) for coord in np.argwhere(panel == bg)]
    assert marker_count <= len(free), "not enough free cells for markers"
    for rr, cc in rng.sample(free, marker_count):
        panel[rr, cc] = MAROON
    return panel


def _rotation_schedule(count: int, rng: random.Random) -> list[int]:
    if count >= 4 and rng.random() < 0.5:
        rotations = [0, 1, 2, 3]
    else:
        rotations = [rng.choice((0, 2)), rng.choice((1, 3))]
    while len(rotations) < count:
        rotations.append(rng.randrange(4))
    rng.shuffle(rotations)
    return rotations


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_rows = rng.randint(*PANEL_COUNT_RANGE)
    panel_cols = rng.randint(*PANEL_COUNT_RANGE)
    if panel_rows * panel_cols <= 1:
        return None

    max_size = min(
        (30 - (panel_rows - 1)) // panel_rows,
        (30 - (panel_cols - 1)) // panel_cols,
        MAX_PANEL_SIZE,
    )
    if max_size < MIN_PANEL_SIZE:
        return None
    size = rng.randint(MIN_PANEL_SIZE, max_size)

    arm_len = rng.choice(ARM_LENGTH_CHOICES)
    if size < arm_len + 3:
        return None

    palette = [color for color in range(10) if color not in {RED, GREY, MAROON}]
    bg = rng.choice(palette)
    remaining = [color for color in palette if color != bg]
    legend_colors = rng.sample(remaining, 2)

    legend_rotation = rng.randrange(4)
    legend_lengths = {
        _rotate_dir(direction, legend_rotation): arm_len
        for direction in BASE_ARMS
    }
    legend_tip_colors = {
        _rotate_dir(BASE_ARMS[idx], legend_rotation): legend_colors[idx]
        for idx in range(2)
    }

    height = panel_rows * size + (panel_rows - 1)
    width = panel_cols * size + (panel_cols - 1)
    grid = np.full((height, width), bg, dtype=int)
    for i in range(1, panel_rows):
        grid[_panel_origin(i, size) - 1, :] = GREY
    for j in range(1, panel_cols):
        grid[:, _panel_origin(j, size) - 1] = GREY

    legend_panel = _build_panel(
        size=size,
        bg=bg,
        lengths=legend_lengths,
        marker_count=0,
        rng=rng,
        tip_colors=legend_tip_colors,
    )
    grid[0:size, 0:size] = legend_panel

    nonlegend = panel_rows * panel_cols - 1
    rotations = _rotation_schedule(nonlegend, rng)
    idx = 0
    for i in range(panel_rows):
        for j in range(panel_cols):
            if i == 0 and j == 0:
                continue
            rotation = rotations[idx]
            idx += 1
            lengths = {
                _rotate_dir(direction, rotation): arm_len
                for direction in legend_lengths
            }
            panel = _build_panel(
                size=size,
                bg=bg,
                lengths=lengths,
                marker_count=rng.randint(*MARKER_COUNT_RANGE),
                rng=rng,
            )
            r0 = _panel_origin(i, size)
            c0 = _panel_origin(j, size)
            grid[r0 : r0 + size, c0 : c0 + size] = panel

    try:
        out = main(grid)
    except AssertionError:
        return None

    used_colors = {
        int(out[_panel_origin(i, size) : _panel_origin(i, size) + size, _panel_origin(j, size) : _panel_origin(j, size) + size][
            grid[_panel_origin(i, size) : _panel_origin(i, size) + size, _panel_origin(j, size) : _panel_origin(j, size) + size] == MAROON
        ][0])
        for i in range(panel_rows)
        for j in range(panel_cols)
        if not (i == 0 and j == 0)
    }
    if len(used_colors) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
