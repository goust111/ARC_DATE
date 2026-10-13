from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 120


def _spans_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(set(int(idx) for idx in separators))
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in separators:
        if idx - prev > 1:
            spans.append((prev + 1, idx))
        prev = idx
    if length - prev > 1:
        spans.append((prev + 1, length))
    return spans


def _grid_structure(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int], int]:
    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    sep_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    row_spans = _spans_from_separators(g.shape[0], sep_rows)
    col_spans = _spans_from_separators(g.shape[1], sep_cols)

    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected header row/column plus interior cells"
    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "cells must be equally sized"
    cell_size = row_sizes.pop()
    assert cell_size == col_sizes.pop(), "cells must be square"
    return row_spans, col_spans, sep_rows, sep_cols, int(cell_size)


def _template_from_block(block: np.ndarray) -> tuple[np.ndarray, int]:
    mask = (block != BG) & (block != SEP)
    colors = {int(v) for v in np.unique(block[mask])}
    assert mask.any(), "template cell cannot be empty"
    assert len(colors) == 1, "template cell must use a single color"
    return mask, int(next(iter(colors)))


def _solve_with_structure(
    g: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> np.ndarray:
    out = g.copy()
    out[sep_rows, :] = SEP
    out[:, sep_cols] = SEP

    tr0, tr1 = row_spans[0]
    tc0, tc1 = col_spans[0]
    out[tr0:tr1, tc0:tc1] = BG

    row_masks: list[np.ndarray] = []
    row_colors: list[int] = []
    for r0, r1 in row_spans[1:]:
        mask, color = _template_from_block(g[r0:r1, tc0:tc1])
        row_masks.append(mask)
        row_colors.append(color)

    col_masks: list[np.ndarray] = []
    for c0, c1 in col_spans[1:]:
        mask, _color = _template_from_block(g[tr0:tr1, c0:c1])
        col_masks.append(mask)

    for row_idx, (r0, r1) in enumerate(row_spans[1:]):
        for col_idx, (c0, c1) in enumerate(col_spans[1:]):
            block_in = g[r0:r1, c0:c1]
            target_mask = row_masks[row_idx] & col_masks[col_idx]
            block_out = np.full(block_in.shape, BG, dtype=int)
            block_out[target_mask] = row_colors[row_idx]

            keep_mask = target_mask & (block_in != BG) & (block_in != SEP)
            block_out[keep_mask] = block_in[keep_mask]
            out[r0:r1, c0:c1] = block_out

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each inner block by intersecting its row template with its column template.

    Input structure:
    - The background is black, and grey full-row/full-column lines split the canvas into equal square cells.
    - The top row of cells, except the top-left corner, contains monochrome binary column templates.
    - The left column of cells, except the top-left corner, contains monochrome binary row templates.
    - Interior cells are blank or partially filled, and may include stray pixels outside the true pattern.

    Rule:
    - Read each column template from the top header row as the set of non-background pixels inside that cell.
    - Read each row template from the left header column in the same way.
    - For every interior cell, compute the pixelwise intersection of its row-template mask and column-template mask.
    - Paint every intersection pixel with the row-template color, except when the input already has a non-background pixel at that correct position; those existing colors are preserved.
    - Remove every interior pixel that lies outside the intersection result, while keeping the headers and grey separators unchanged.

    Key insight:
    - The first row and first column are not content to transform; they are binary stencils whose overlap determines every interior cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_spans, col_spans, sep_rows, sep_cols, _cell_size = _grid_structure(g)
    return _solve_with_structure(g, row_spans, col_spans, sep_rows, sep_cols)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _grow_blob(size: int) -> np.ndarray:
    area = size * size
    target = random.randint(max(2, area // 3), max(3, (2 * area) // 3))
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    for r, c in cells:
        mask[r, c] = True
    return mask


def _band_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    thickness = random.randint(1, max(1, min(2, size - 1)))
    start = random.randint(0, size - thickness)
    mask[start : start + thickness, :] = True
    if random.random() < 0.6:
        col = random.randrange(size)
        depth = random.randint(max(2, size // 2), size)
        mask[:depth, col] = True
    return _random_transform(mask)


def _l_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    row_len = random.randint(max(2, size // 2), size)
    col_len = random.randint(max(2, size // 2), size)
    mask[0, :row_len] = True
    mask[:col_len, 0] = True
    if random.random() < 0.4 and size >= 4:
        mask[1, : max(2, row_len - 1)] = True
    return _random_transform(mask)


def _diag_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    width = 1 if size <= 4 else random.randint(1, 2)
    for r in range(size):
        for c in range(size):
            if abs(r - c) < width:
                mask[r, c] = True
    if random.random() < 0.5:
        row = random.randrange(size)
        mask[row, : random.randint(max(2, size // 2), size)] = True
    return _random_transform(mask)


def _u_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    if size >= 4 and random.random() < 0.5:
        mask[-2, 1:-1] = True
    return _random_transform(mask)


def _random_mask(size: int) -> np.ndarray:
    builders = (_grow_blob, _band_mask, _l_mask, _diag_mask, _u_mask)
    for _ in range(MASK_ATTEMPTS):
        mask = builders[random.randrange(len(builders))](size)
        count = int(mask.sum())
        if 0 < count < size * size:
            return mask
    raise RuntimeError("failed to build a nontrivial mask")


def _unique_masks(count: int, size: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(MASK_ATTEMPTS * 4):
        mask = _random_mask(size)
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
        if len(masks) == count:
            return masks
    return None


def _valid_dimensions(cell_size: int) -> list[tuple[int, int]]:
    options = []
    for cell_rows in range(3, 7):
        for cell_cols in range(3, 7):
            height = cell_rows * cell_size + (cell_rows - 1)
            width = cell_cols * cell_size + (cell_cols - 1)
            header_colors = (cell_rows - 1) + (cell_cols - 1)
            if max(height, width) <= 30 and header_colors <= len(PALETTE):
                options.append((cell_rows, cell_cols))
    return options


def _build_canvas(cell_rows: int, cell_cols: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = cell_rows * cell_size + (cell_rows - 1)
    width = cell_cols * cell_size + (cell_cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(cell_rows):
        row_spans.append((cursor, cursor + cell_size))
        cursor += cell_size
        if idx != cell_rows - 1:
            grid[cursor, :] = SEP
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(cell_cols):
        col_spans.append((cursor, cursor + cell_size))
        cursor += cell_size
        if idx != cell_cols - 1:
            grid[:, cursor] = SEP
            cursor += 1

    return grid, row_spans, col_spans


def _intersection_matrix(row_masks: list[np.ndarray], col_masks: list[np.ndarray]) -> list[list[np.ndarray]]:
    return [[r_mask & c_mask for c_mask in col_masks] for r_mask in row_masks]


def _scene_is_interesting(intersections: list[list[np.ndarray]]) -> bool:
    flat = [mask for row in intersections for mask in row]
    nonempty = [mask for mask in flat if mask.any()]
    if len(nonempty) < max(3, len(flat) // 2):
        return False
    if len({mask.tobytes() for mask in flat}) < min(4, len(flat)):
        return False
    if not all(any(mask.any() for mask in row) for row in intersections):
        return False
    if not all(any(intersections[r][c].any() for r in range(len(intersections))) for c in range(len(intersections[0]))):
        return False
    return True


def _choose_modes(
    intersections: list[list[np.ndarray]],
) -> dict[tuple[int, int], str]:
    positions = [(r, c) for r in range(len(intersections)) for c in range(len(intersections[0]))]
    nonempty = [pos for pos in positions if intersections[pos[0]][pos[1]].any()]
    noisy_candidates = [
        pos
        for pos in positions
        if int(intersections[pos[0]][pos[1]].sum()) < intersections[pos[0]][pos[1]].size
    ]
    random.shuffle(positions)
    random.shuffle(nonempty)
    random.shuffle(noisy_candidates)

    modes: dict[tuple[int, int], str] = {}
    if nonempty:
        modes[nonempty.pop()] = "col"
    if nonempty:
        modes[nonempty.pop()] = "row"
    if noisy_candidates:
        candidate = noisy_candidates.pop()
        if candidate not in modes:
            modes[candidate] = "noisy"
    for pos in positions:
        if pos not in modes:
            modes[pos] = "empty"
            break

    for pos in positions:
        if pos in modes:
            continue
        mask = intersections[pos[0]][pos[1]]
        choices = ["empty", "noisy"] if not mask.any() else ["empty", "row", "col", "noisy"]
        modes[pos] = random.choice(choices)
    return modes


def _subset(coords: list[tuple[int, int]], full_bias: bool = False) -> list[tuple[int, int]]:
    if len(coords) == 1:
        return coords[:]
    if full_bias and random.random() < 0.5:
        k = len(coords)
    else:
        k = random.randint(1, len(coords) - 1)
    return random.sample(coords, k)


def _paint_internal_block(
    block: np.ndarray,
    expected_mask: np.ndarray,
    row_color: int,
    col_color: int,
    all_header_colors: tuple[int, ...],
    mode: str,
) -> None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(expected_mask)]
    wrong_coords = [(int(r), int(c)) for r, c in np.argwhere(~expected_mask)]

    if mode == "empty":
        return

    if mode == "row" and coords:
        for r, c in _subset(coords):
            block[r, c] = row_color
        return

    if mode == "col" and coords:
        for r, c in _subset(coords, full_bias=True):
            block[r, c] = col_color
        return

    if coords:
        for r, c in _subset(coords, full_bias=True):
            block[r, c] = random.choice((row_color, col_color))

    if wrong_coords:
        wrong_count = random.randint(1, min(3, len(wrong_coords)))
        noise_palette = [color for color in all_header_colors if color not in {row_color, col_color}]
        if not noise_palette:
            noise_palette = list(all_header_colors)
        for r, c in random.sample(wrong_coords, wrong_count):
            block[r, c] = random.choice(noise_palette)


def _expected_output(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> np.ndarray:
    sep_rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    sep_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    return _solve_with_structure(grid, row_spans, col_spans, sep_rows, sep_cols)


def _generate_attempt() -> np.ndarray | None:
    cell_size = random.randint(3, 5)
    dims = _valid_dimensions(cell_size)
    if not dims:
        return None
    cell_rows, cell_cols = random.choice(dims)

    grid, row_spans, col_spans = _build_canvas(cell_rows, cell_cols, cell_size)
    row_count = cell_rows - 1
    col_count = cell_cols - 1

    colors = random.sample(PALETTE, row_count + col_count)
    row_colors = colors[:row_count]
    col_colors = colors[row_count:]

    row_masks = _unique_masks(row_count, cell_size)
    col_masks = _unique_masks(col_count, cell_size)
    if row_masks is None or col_masks is None:
        return None

    intersections = _intersection_matrix(row_masks, col_masks)
    if not _scene_is_interesting(intersections):
        return None

    tr0, tr1 = row_spans[0]
    tc0, tc1 = col_spans[0]
    grid[tr0:tr1, tc0:tc1] = BG

    for idx, ((r0, r1), color, mask) in enumerate(zip(row_spans[1:], row_colors, row_masks)):
        block = grid[r0:r1, tc0:tc1]
        block[:, :] = BG
        block[mask] = color

    for idx, ((c0, c1), color, mask) in enumerate(zip(col_spans[1:], col_colors, col_masks)):
        block = grid[tr0:tr1, c0:c1]
        block[:, :] = BG
        block[mask] = color

    modes = _choose_modes(intersections)
    header_colors = tuple(row_colors + col_colors)
    for r_idx, (r0, r1) in enumerate(row_spans[1:]):
        for c_idx, (c0, c1) in enumerate(col_spans[1:]):
            block = grid[r0:r1, c0:c1]
            block[:, :] = BG
            _paint_internal_block(
                block,
                intersections[r_idx][c_idx],
                row_colors[r_idx],
                col_colors[c_idx],
                header_colors,
                modes[(r_idx, c_idx)],
            )

    expected = _expected_output(grid, row_spans, col_spans)
    interior = expected.copy()
    interior[expected == SEP] = BG
    if int(np.count_nonzero(interior)) < max(8, cell_size * 2):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
