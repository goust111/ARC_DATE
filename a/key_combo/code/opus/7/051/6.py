from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LABEL_COLORS = (BLUE, RED, GREEN)
SPRITE_COLORS = (BLUE, RED, GREEN, ORANGE)
ORTHO_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_RETRIES = 300


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _find_divider_grid(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        full_rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if full_rows and full_cols:
            candidates.append((int(color), full_rows, full_cols))
    assert len(candidates) == 1, "expected exactly one divider color"
    return candidates[0]


def _cell_view(grid: np.ndarray, row_range: tuple[int, int], col_range: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_range
    c0, c1 = col_range
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _cell_payload(cell: np.ndarray, bg: int, divider: int) -> tuple[int | None, np.ndarray]:
    colors = set(map(int, cell.ravel())) - {bg}
    assert divider not in colors, "divider color must not appear inside a cell"
    if not colors:
        return None, np.zeros(cell.shape, dtype=bool)
    assert len(colors) == 1, "each filled cell must contain exactly one sprite color"
    color = int(next(iter(colors)))
    return color, cell == color


def _label_color(cell: np.ndarray, bg: int, divider: int) -> int:
    colors = set(map(int, cell.ravel())) - {bg}
    assert divider not in colors, "divider color must not appear in a label cell"
    assert len(colors) == 1, "each label cell must contain one visible label color"
    return int(next(iter(colors)))


def _flip_to_missing(mask: np.ndarray, src_r: int, src_c: int, dst_r: int, dst_c: int) -> np.ndarray:
    pred = mask
    if src_r != dst_r:
        pred = np.flipud(pred)
    if src_c != dst_c:
        pred = np.fliplr(pred)
    return pred


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each missing mirrored sprite inside the divider lattice and recolor label-matching cells yellow.

    Input structure:
    - A black background contains a regular cell lattice separated by one grey
      horizontal/vertical divider color.
    - The rightmost cell column and bottom cell row are label cells; each label
      cell contains one single-pixel red, blue, or green marker on background.
    - The remaining interior cells are partitioned into 2x2 blocks. In each
      block, three cells contain same-color mirrored sprite variants and one
      cell is empty.

    Rule:
    - Parse the divider lattice into equal-sized cells and treat the last cell
      row/column as labels.
    - For each interior 2x2 block, reflect the three present cell-local sprite
      masks toward the empty corner; all reflected predictions agree, and that
      sprite is painted into the missing cell using the block's sprite color.
    - After all interior cells contain sprites, compare each interior cell's row
      label from the right edge with its column label from the bottom edge.
      Recolor every sprite pixel in cells whose two labels match to yellow.

    Key insight:
    - The puzzle composes two stages: mirror completion makes every interior
      cell meaningful, then the border labels select which completed sprites are
      recolored at row/column equality intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _majority_color(g)
    divider, divider_rows, divider_cols = _find_divider_grid(g, bg)
    row_ranges = _ranges_between(g.shape[0], divider_rows)
    col_ranges = _ranges_between(g.shape[1], divider_cols)
    assert len(row_ranges) >= 3 and len(col_ranges) >= 3, "expected interior cells plus labels"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_ranges}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(row_heights) == 1 and len(col_widths) == 1, "expected a regular cell lattice"

    interior_rows = len(row_ranges) - 1
    interior_cols = len(col_ranges) - 1
    assert interior_rows % 2 == 0 and interior_cols % 2 == 0, "interior must partition into 2x2 blocks"

    out = g.copy()

    for block_r in range(0, interior_rows, 2):
        for block_c in range(0, interior_cols, 2):
            present: list[tuple[int, int, int, np.ndarray]] = []
            missing: list[tuple[int, int]] = []
            for dr in range(2):
                for dc in range(2):
                    rr = block_r + dr
                    cc = block_c + dc
                    cell = _cell_view(g, row_ranges[rr], col_ranges[cc])
                    color, mask = _cell_payload(cell, bg, divider)
                    if color is None:
                        missing.append((dr, dc))
                    else:
                        present.append((dr, dc, color, mask))

            assert len(missing) == 1, "each 2x2 block must have exactly one empty cell"
            assert len(present) == 3, "each 2x2 block must have exactly three sprites"
            colors = {color for _dr, _dc, color, _mask in present}
            assert len(colors) == 1, "all present sprites in a block must share one color"

            miss_r, miss_c = missing[0]
            predictions = [
                _flip_to_missing(mask, dr, dc, miss_r, miss_c)
                for dr, dc, _color, mask in present
            ]
            canonical = predictions[0]
            assert all(np.array_equal(canonical, pred) for pred in predictions[1:]), (
                "mirrored sprite predictions must agree"
            )

            fill_color = present[0][2]
            target_rr = block_r + miss_r
            target_cc = block_c + miss_c
            target = _cell_view(out, row_ranges[target_rr], col_ranges[target_cc])
            assert np.all(target == bg), "missing cell must be empty before fill"
            target[canonical] = fill_color

    row_labels = [
        _label_color(_cell_view(out, row_ranges[r], col_ranges[-1]), bg, divider)
        for r in range(interior_rows)
    ]
    col_labels = [
        _label_color(_cell_view(out, row_ranges[-1], col_ranges[c]), bg, divider)
        for c in range(interior_cols)
    ]

    for r in range(interior_rows):
        for c in range(interior_cols):
            if row_labels[r] != col_labels[c]:
                continue
            cell = _cell_view(out, row_ranges[r], col_ranges[c])
            sprite_mask = cell != bg
            assert np.any(sprite_mask), "every interior cell must contain a sprite after completion"
            cell[sprite_mask] = YELLOW

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHO_STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pos = (nr, nc)
                if pos not in seen:
                    seen.add(pos)
                    stack.append(pos)
    return len(seen) == int(mask.sum())


def _random_sprite_mask(cell_size: int) -> np.ndarray | None:
    max_pixels = min(5, cell_size * cell_size)
    for _ in range(120):
        pixel_count = random.randint(3, max_pixels)
        seed = (random.randrange(cell_size), random.randrange(cell_size))
        cells = {seed}
        while len(cells) < pixel_count:
            frontier = []
            for r, c in cells:
                for dr, dc in ORTHO_STEPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if int(mask.sum()) != pixel_count or not _connected(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if len(
            {
                mask.tobytes(),
                np.fliplr(mask).tobytes(),
                np.flipud(mask).tobytes(),
                np.flipud(np.fliplr(mask)).tobytes(),
            }
        ) < 3:
            continue
        return mask
    return None


def _layout_ranges(cell_count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges = []
    dividers = []
    pos = 0
    for i in range(cell_count):
        ranges.append((pos, pos + cell_size - 1))
        pos += cell_size
        if i != cell_count - 1:
            dividers.append(pos)
            pos += 1
    return ranges, dividers, pos


def _paint_cell(
    grid: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    mask: np.ndarray,
    color: int,
) -> None:
    cell = _cell_view(grid, row_range, col_range)
    cell[mask] = color


def _paint_label(
    grid: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    color: int,
) -> None:
    cell = _cell_view(grid, row_range, col_range)
    center_r = cell.shape[0] // 2
    center_c = cell.shape[1] // 2
    cell[center_r, center_c] = color


def _sample_label_sequences(
    interior_rows: int,
    interior_cols: int,
    missing_cells: set[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        row_labels = [random.choice(LABEL_COLORS) for _ in range(interior_rows)]
        col_labels = [random.choice(LABEL_COLORS) for _ in range(interior_cols)]
        matches = {
            (r, c)
            for r in range(interior_rows)
            for c in range(interior_cols)
            if row_labels[r] == col_labels[c]
        }
        if len(matches) < 2:
            continue
        if len(matches) == interior_rows * interior_cols:
            continue
        if not (matches & missing_cells):
            continue
        if len(set(row_labels + col_labels)) < 2:
            continue
        return row_labels, col_labels
    return None


def _attempt_input() -> tuple[np.ndarray, np.ndarray] | None:
    cell_size = random.choice((3, 4))
    max_blocks = 3 if cell_size == 3 else 2
    block_rows = random.randint(1, max_blocks)
    block_cols = random.randint(1, max_blocks)

    cell_rows = 2 * block_rows + 1
    cell_cols = 2 * block_cols + 1
    row_ranges, divider_rows, height = _layout_ranges(cell_rows, cell_size)
    col_ranges, divider_cols, width = _layout_ranges(cell_cols, cell_size)
    if max(height, width) > 30:
        return None

    solved = np.full((height, width), BLACK, dtype=int)
    solved[divider_rows, :] = GREY
    solved[:, divider_cols] = GREY

    missing_cells: set[tuple[int, int]] = set()
    block_specs = []
    for br in range(block_rows):
        for bc in range(block_cols):
            mask = _random_sprite_mask(cell_size)
            if mask is None:
                return None
            color = random.choice(SPRITE_COLORS)
            missing_corner = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
            missing_cells.add((2 * br + missing_corner[0], 2 * bc + missing_corner[1]))
            block_specs.append((br, bc, mask, color, missing_corner))

    labels = _sample_label_sequences(2 * block_rows, 2 * block_cols, missing_cells)
    if labels is None:
        return None
    row_labels, col_labels = labels

    for r, color in enumerate(row_labels):
        _paint_label(solved, row_ranges[r], col_ranges[-1], color)
    for c, color in enumerate(col_labels):
        _paint_label(solved, row_ranges[-1], col_ranges[c], color)
    _paint_label(solved, row_ranges[-1], col_ranges[-1], row_labels[-1])

    for br, bc, mask, color, _missing_corner in block_specs:
        variants = {
            (0, 0): mask,
            (0, 1): np.fliplr(mask),
            (1, 0): np.flipud(mask),
            (1, 1): np.flipud(np.fliplr(mask)),
        }
        for dr in range(2):
            for dc in range(2):
                rr = 2 * br + dr
                cc = 2 * bc + dc
                _paint_cell(solved, row_ranges[rr], col_ranges[cc], variants[(dr, dc)], color)

    for r in range(2 * block_rows):
        for c in range(2 * block_cols):
            if row_labels[r] == col_labels[c]:
                cell = _cell_view(solved, row_ranges[r], col_ranges[c])
                cell[cell != BLACK] = YELLOW

    inp = solved.copy()
    for br, bc, _mask, _color, missing_corner in block_specs:
        rr = 2 * br + missing_corner[0]
        cc = 2 * bc + missing_corner[1]
        cell = _cell_view(inp, row_ranges[rr], col_ranges[cc])
        cell[:, :] = BLACK

    try:
        observed = main(inp)
    except Exception:
        return None
    if not np.array_equal(observed, solved):
        return None
    return inp, solved


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        attempt = _attempt_input()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
