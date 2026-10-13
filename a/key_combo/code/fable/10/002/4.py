from __future__ import annotations

import random

import numpy as np


BLACK = 0
SHAPE_COLOR = 8
GUIDE_COLORS = tuple(color for color in range(1, 10) if color != SHAPE_COLOR)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    count = 0
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        count += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            cr, cc = stack.pop()
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return count


def _parse_candidate(grid: np.ndarray, orientation: str, index: int) -> dict[str, np.ndarray | int | str] | None:
    n = grid.shape[0]
    line = grid[index, :].copy() if orientation == "row" else grid[:, index].copy()
    line_colors = sorted({int(value) for value in line if value != BLACK})
    if len(line_colors) < 2:
        return None

    remainder = grid.copy()
    if orientation == "row":
        remainder[index, :] = BLACK
    else:
        remainder[:, index] = BLACK

    shape_mask = remainder != BLACK
    if not np.any(shape_mask):
        return None

    shape_colors = {int(value) for value in remainder[shape_mask]}
    if len(shape_colors) != 1:
        return None
    if _component_count(shape_mask) != 1:
        return None

    return {
        "orientation": orientation,
        "index": index,
        "line": line,
        "shape_mask": shape_mask,
        "shape_color": int(next(iter(shape_colors))),
        "size": n,
    }


def _find_guide(grid: np.ndarray) -> dict[str, np.ndarray | int | str]:
    n = grid.shape[0]
    candidates: list[dict[str, np.ndarray | int | str]] = []
    for r in range(n):
        candidate = _parse_candidate(grid, "row", r)
        if candidate is not None:
            candidates.append(candidate)
    for c in range(n):
        candidate = _parse_candidate(grid, "col", c)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one valid guide line"
    return candidates[0]


def _build_output(candidate: dict[str, np.ndarray | int | str]) -> np.ndarray:
    n = int(candidate["size"])
    orientation = str(candidate["orientation"])
    index = int(candidate["index"])
    line = np.asarray(candidate["line"], dtype=int)
    shape_mask = np.asarray(candidate["shape_mask"], dtype=bool)

    out = np.zeros((n * n, n * n), dtype=int)
    if orientation == "row":
        for block_col, color in enumerate(line):
            if color == BLACK:
                continue
            block = out[index * n : (index + 1) * n, block_col * n : (block_col + 1) * n]
            block[shape_mask] = int(color)
    else:
        for block_row, color in enumerate(line):
            if color == BLACK:
                continue
            block = out[block_row * n : (block_row + 1) * n, index * n : (index + 1) * n]
            block[shape_mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the monochrome input shape into selected blocks of a single output band, recolored by the guide line.

    Input structure:
    - The input is a square n x n tile on a black background.
    - Exactly one full row or one full column is a guide line whose cells may be black or colored.
    - The guide line contains at least two distinct non-black colors.
    - Outside the guide line, all non-black cells belong to one single-color shape.

    Rule:
    - Build an n x n array of black n x n blocks, giving an output of size n^2 x n^2.
    - If the guide is row r, only block-row r can be populated; if it is column c, only block-column c can be populated.
    - For each position along the guide line, a black guide cell leaves the corresponding block empty.
    - A non-black guide cell copies the non-guide shape into that block at the same relative coordinates and recolors the whole copied shape to the guide cell's color.

    Key insight:
    - The same guide line is read three times: its orientation and index choose the visible band, its black slots choose which blocks stay empty, and its color sequence provides the recoloring palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h > 0, "expected a nonempty square tile"

    candidate = _find_guide(g)
    return _build_output(candidate)


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_cells = 3
    max_cells = area - 1 if area > min_cells else area
    target = random.randint(min_cells, max_cells)

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < target:
        choice = random.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nr, nc in _neighbors4(*choice):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _guide_side_options(n: int, orientation: str, index: int) -> list[tuple[int, int]]:
    if orientation == "row":
        options = []
        if index >= 2:
            options.append((0, index))
        if n - index - 1 >= 2:
            options.append((index + 1, n))
        return options

    options = []
    if index >= 2:
        options.append((0, index))
    if n - index - 1 >= 2:
        options.append((index + 1, n))
    return options


def _make_shape_mask(n: int, orientation: str, index: int) -> np.ndarray:
    options = _guide_side_options(n, orientation, index)
    assert options, "need at least one side with thickness >= 2"
    side_start, side_end = random.choice(options)

    if orientation == "row":
        side_height = side_end - side_start
        box_h = random.randint(2, side_height)
        box_w = random.randint(2, n)
        box_r0 = random.randint(side_start, side_end - box_h)
        box_c0 = random.randint(0, n - box_w)
        local_mask = _random_connected_mask(box_h, box_w)
        mask = np.zeros((n, n), dtype=bool)
        mask[box_r0 : box_r0 + box_h, box_c0 : box_c0 + box_w] = local_mask
    else:
        side_width = side_end - side_start
        box_h = random.randint(2, n)
        box_w = random.randint(2, side_width)
        box_r0 = random.randint(0, n - box_h)
        box_c0 = random.randint(side_start, side_end - box_w)
        local_mask = _random_connected_mask(box_h, box_w)
        mask = np.zeros((n, n), dtype=bool)
        mask[box_r0 : box_r0 + box_h, box_c0 : box_c0 + box_w] = local_mask

    if orientation == "row":
        mask[index, :] = False
    else:
        mask[:, index] = False
    return mask


def _make_guide_line(n: int) -> np.ndarray:
    line = np.zeros(n, dtype=int)
    active_count = random.randint(2, n - 1)
    active_positions = random.sample(range(n), active_count)
    palette_size = random.randint(2, min(3, active_count))
    palette = random.sample(GUIDE_COLORS, palette_size)
    colors = palette[:] + [random.choice(palette) for _ in range(active_count - palette_size)]
    random.shuffle(colors)
    for pos, color in zip(active_positions, colors):
        line[pos] = color
    return line


def _explicit_semantic_check(inp: np.ndarray) -> None:
    candidate = _find_guide(inp)
    out = main(inp)
    expected = _build_output(candidate)
    assert np.array_equal(out, expected)

    n = inp.shape[0]
    orientation = str(candidate["orientation"])
    index = int(candidate["index"])
    line = np.asarray(candidate["line"], dtype=int)
    shape_mask = np.asarray(candidate["shape_mask"], dtype=bool)

    assert np.any(line == BLACK), "generator must expose empty-slot behavior"
    assert len({int(value) for value in line if value != BLACK}) >= 2
    assert _component_count(shape_mask) == 1

    for block_row in range(n):
        for block_col in range(n):
            block = out[block_row * n : (block_row + 1) * n, block_col * n : (block_col + 1) * n]
            if orientation == "row":
                active_color = int(line[block_col]) if block_row == index else BLACK
            else:
                active_color = int(line[block_row]) if block_col == index else BLACK

            if active_color == BLACK:
                assert not np.any(block)
            else:
                assert np.all(block[shape_mask] == active_color)
                assert not np.any(block[~shape_mask])


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(4, 5)
    orientation = random.choice(("row", "col"))
    index = random.randrange(n)

    if not _guide_side_options(n, orientation, index):
        return None

    shape_mask = _make_shape_mask(n, orientation, index)
    if not np.any(shape_mask):
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[shape_mask] = SHAPE_COLOR
    line = _make_guide_line(n)

    if orientation == "row":
        grid[index, :] = line
    else:
        grid[:, index] = line

    try:
        _explicit_semantic_check(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
