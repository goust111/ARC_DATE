from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
AVAILABLE_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400


def _parse_legend(row: np.ndarray) -> dict[int, tuple[str, int]]:
    positions = [idx for idx, value in enumerate(row) if int(value) != BLACK]
    assert positions and len(positions) % 2 == 0

    mapping: dict[int, tuple[str, int]] = {}
    cursor = 0
    while cursor < len(positions):
        left_col = positions[cursor]
        right_col = positions[cursor + 1]
        assert right_col == left_col + 1

        left_color = int(row[left_col])
        right_color = int(row[right_col])
        assert left_color not in (BLACK, GREY)
        assert right_color not in (BLACK, GREY)
        assert left_color != right_color
        assert left_color not in mapping and right_color not in mapping

        mapping[left_color] = ("horizontal", right_color)
        mapping[right_color] = ("vertical", left_color)
        cursor += 2

    return mapping


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True

                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            parts.append((cells, touches_border))

    return parts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3

    legend = _parse_legend(g[0])
    body = g[1:]
    out = g.copy()

    for cells, touches_border in _components(body != GREY):
        if touches_border:
            continue

        marker_cells = [(r, c, int(body[r, c])) for r, c in cells if int(body[r, c]) != BLACK]
        assert len(marker_cells) == 1
        marker_color = marker_cells[0][2]
        assert marker_color in legend

        direction, secondary = legend[marker_color]
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)

        for r, c in cells:
            if direction == "horizontal":
                use_primary = (r - top) % 2 == 0
            else:
                use_primary = (c - left) % 2 == 0
            out[r + 1, c] = marker_color if use_primary else secondary

    return out


def _region_line_positions(start: int, spans: list[int]) -> list[int]:
    positions = [start]
    cursor = start
    for span in spans:
        cursor += span + 1
        positions.append(cursor)
    return positions


def _choose_pairs(rng: random.Random, count: int) -> list[tuple[int, int]]:
    colors = rng.sample(AVAILABLE_COLORS, 2 * count)
    rng.shuffle(colors)
    return [(colors[2 * idx], colors[2 * idx + 1]) for idx in range(count)]


def _region_boxes(row_lines: list[int], col_lines: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for row_idx in range(len(row_lines) - 1):
        for col_idx in range(len(col_lines) - 1):
            boxes.append(
                (
                    row_lines[row_idx] + 1,
                    row_lines[row_idx + 1] - 1,
                    col_lines[col_idx] + 1,
                    col_lines[col_idx + 1] - 1,
                )
            )
    return boxes


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        legend = _parse_legend(grid[0])
    except AssertionError:
        return False

    enclosed = [cells for cells, touches in _components(grid[1:] != GREY) if not touches]
    if not enclosed:
        return False

    directions = set()
    for cells in enclosed:
        markers = [int(grid[r + 1, c]) for r, c in cells if int(grid[r + 1, c]) != BLACK]
        if len(markers) != 1 or markers[0] not in legend:
            return False
        directions.add(legend[markers[0]][0])

    if len(enclosed) > 1 and len(directions) < 2:
        return False

    try:
        transformed = main(grid)
    except AssertionError:
        return False
    return transformed.shape == grid.shape and bool(np.any(transformed != grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_rows = rng.randint(1, 3)
    region_cols = rng.randint(1, 4)
    total_regions = region_rows * region_cols

    pair_count = rng.randint(2, 4)
    legend_pairs = _choose_pairs(rng, pair_count)

    row_spans = [rng.randint(2, 5) for _ in range(region_rows)]
    col_spans = [rng.randint(2, 5) for _ in range(region_cols)]
    frame_height = sum(row_spans) + region_rows + 1
    frame_width = sum(col_spans) + region_cols + 1

    spacer_rows = rng.randint(1, 2)
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(1, 3)

    grid_height = 1 + spacer_rows + top_margin + frame_height + bottom_margin
    min_width = frame_width + 2
    legend_width = 2 * pair_count
    grid_width = max(min_width, legend_width + rng.randint(0, 5))

    frame_top = 1 + spacer_rows + top_margin
    frame_left = rng.randint(1, grid_width - frame_width - 1)

    row_lines = _region_line_positions(frame_top, row_spans)
    col_lines = _region_line_positions(frame_left, col_spans)
    boxes = _region_boxes(row_lines, col_lines)

    grid = np.full((grid_height, grid_width), BLACK, dtype=int)

    legend_start = rng.randint(0, grid_width - legend_width)
    for idx, (left_color, right_color) in enumerate(legend_pairs):
        col = legend_start + 2 * idx
        grid[0, col] = left_color
        grid[0, col + 1] = right_color

    for row in row_lines:
        grid[row, col_lines[0] : col_lines[-1] + 1] = GREY
    for col in col_lines:
        grid[row_lines[0] : row_lines[-1] + 1, col] = GREY

    pair_indices = [rng.randrange(pair_count) for _ in range(total_regions)]
    if total_regions > 1 and pair_count > 1 and len(set(pair_indices)) == 1:
        pair_indices[-1] = (pair_indices[-1] + 1) % pair_count

    directions = [rng.choice(("horizontal", "vertical")) for _ in range(total_regions)]
    if total_regions > 1 and len(set(directions)) == 1:
        directions[-1] = "vertical" if directions[0] == "horizontal" else "horizontal"

    for box_idx, (top, bottom, left, right) in enumerate(boxes):
        pair = legend_pairs[pair_indices[box_idx]]
        direction = directions[box_idx]
        marker_color = pair[0] if direction == "horizontal" else pair[1]

        marker_row = rng.randint(top, bottom)
        marker_col = rng.randint(left, right)
        grid[marker_row, marker_col] = marker_color

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
