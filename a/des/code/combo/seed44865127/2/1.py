import random
from collections import Counter

import numpy as np


def _find_separator_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        sep_rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        sep_cols = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
        if sep_rows and sep_cols:
            candidates.append((color, sep_rows, sep_cols))
    assert len(candidates) == 1, "expected one separator color"
    color, sep_rows, sep_cols = candidates[0]
    return int(color), list(map(int, sep_rows)), list(map(int, sep_cols))


def _non_separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep = set(map(int, separators))
    spans = []
    i = 0
    while i < length:
        while i < length and i in sep:
            i += 1
        if i >= length:
            break
        start = i
        while i < length and i not in sep:
            i += 1
        spans.append((int(start), int(i)))
    assert spans, "expected at least one payload span"
    return spans


def _has_reflection_or_half_turn(a: np.ndarray) -> bool:
    return (
        np.array_equal(a, np.fliplr(a))
        or np.array_equal(a, np.flipud(a))
        or np.array_equal(a, np.rot90(a, 2))
    )


def _neighbors4(r: int, c: int, size: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            yield rr, cc


def _grow_connected_cells(size: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], size))
    while len(cells) < count:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], size):
            if nb not in cells:
                frontier.add(nb)
    return cells


def _make_sprite(block_size: int, primary: int, secondary: int) -> np.ndarray | None:
    area = block_size * block_size
    max_secondary = (area - 1) // 2
    all_cells = [(r, c) for r in range(block_size) for c in range(block_size)]

    for _ in range(120):
        secondary_count = random.randint(1, max_secondary)
        if random.random() < 0.5:
            secondary_cells = _grow_connected_cells(block_size, secondary_count)
        else:
            secondary_cells = set(random.sample(all_cells, secondary_count))

        sprite = np.full((block_size, block_size), primary, dtype=int)
        for r, c in secondary_cells:
            sprite[r, c] = secondary

        if _has_reflection_or_half_turn(sprite):
            continue
        if len(np.unique(sprite)) != 2:
            continue
        return sprite
    return None


def _cell_origin(cell_r: int, cell_c: int, block_size: int, sep_width: int) -> tuple[int, int]:
    return cell_r * (block_size + sep_width), cell_c * (block_size + sep_width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap block-grid cell roles between a repeated two-color sprite and a solid fill block.

    Input structure
    - Uniform separator rows and columns partition the grid into equal square cells.
    - Most cells contain the same two-color sprite.
    - The remaining cells are solid blocks using the sprite's secondary color.

    Rule
    - Infer the repeated sprite template.
    - Replace each sprite cell with a solid block of the sprite's primary color.
    - Replace each solid secondary-color cell with the sprite.
    - Leave separator rows and columns unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, sep_rows, sep_cols = _find_separator_structure(g)
    row_spans = _non_separator_spans(g.shape[0], sep_rows)
    col_spans = _non_separator_spans(g.shape[1], sep_cols)

    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "cells must be equal-sized"
    block_size = row_sizes.pop()
    assert block_size == col_sizes.pop(), "cells must be square"

    payload_colors = [int(c) for c in np.unique(g) if int(c) != separator]
    assert len(payload_colors) == 2, "expected exactly two payload colors"

    block_entries = []
    mixed_signatures = Counter()
    signature_to_block: dict[tuple[int, ...], np.ndarray] = {}

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            block = g[r0:r1, c0:c1]
            colors = {int(c) for c in np.unique(block)}
            assert separator not in colors, "separator color cannot appear inside a cell"
            assert colors.issubset(set(payload_colors)), "unexpected payload color"
            block_entries.append((r0, r1, c0, c1, block.copy()))
            if len(colors) == 2:
                sig = tuple(map(int, block.reshape(-1)))
                mixed_signatures[sig] += 1
                signature_to_block[sig] = block.copy()

    assert mixed_signatures, "expected at least one sprite cell"
    top_count = max(mixed_signatures.values())
    top_sigs = [sig for sig, count in mixed_signatures.items() if count == top_count]
    assert len(top_sigs) == 1, "sprite template must be uniquely most common"
    sprite = signature_to_block[top_sigs[0]]

    color_counts = {color: int(np.count_nonzero(sprite == color)) for color in payload_colors}
    assert color_counts[payload_colors[0]] != color_counts[payload_colors[1]], "sprite needs a primary color"
    primary = max(payload_colors, key=lambda color: color_counts[color])
    secondary = next(color for color in payload_colors if color != primary)

    out = g.copy()
    sprite_cells = 0
    solid_cells = 0
    for r0, r1, c0, c1, block in block_entries:
        if np.array_equal(block, sprite):
            out[r0:r1, c0:c1] = primary
            sprite_cells += 1
        else:
            assert np.all(block == secondary), "non-sprite cells must be solid secondary-color blocks"
            out[r0:r1, c0:c1] = sprite
            solid_cells += 1

    assert solid_cells >= 1, "expected at least one solid cell"
    assert sprite_cells > solid_cells, "sprite cells must be the majority"
    return out


# --------------- generator ---------------

COLOR_POOL = list(range(10))
CELL_ROWS_RANGE = (2, 5)
CELL_COLS_RANGE = (2, 5)
BLOCK_SIZE_RANGE = (2, 5)
SEPARATOR_WIDTH_CHOICES = [1, 2]
MAX_GENERATION_ATTEMPTS = 400


def _render_case(
    cell_rows: int,
    cell_cols: int,
    block_size: int,
    sep_width: int,
    separator: int,
    primary: int,
    secondary: int,
    sprite: np.ndarray,
    solid_positions: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height = cell_rows * block_size + (cell_rows - 1) * sep_width
    width = cell_cols * block_size + (cell_cols - 1) * sep_width
    inp = np.full((height, width), separator, dtype=int)
    out = np.full((height, width), separator, dtype=int)

    for cell_r in range(cell_rows):
        for cell_c in range(cell_cols):
            r0, c0 = _cell_origin(cell_r, cell_c, block_size, sep_width)
            r1, c1 = r0 + block_size, c0 + block_size
            if (cell_r, cell_c) in solid_positions:
                inp[r0:r1, c0:c1] = secondary
                out[r0:r1, c0:c1] = sprite
            else:
                inp[r0:r1, c0:c1] = sprite
                out[r0:r1, c0:c1] = primary

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    cell_rows = random.randint(*CELL_ROWS_RANGE)
    cell_cols = random.randint(*CELL_COLS_RANGE)
    block_size = random.randint(*BLOCK_SIZE_RANGE)
    sep_width = random.choice(SEPARATOR_WIDTH_CHOICES)
    separator, primary, secondary = random.sample(COLOR_POOL, 3)

    sprite = _make_sprite(block_size, primary, secondary)
    if sprite is None:
        return None

    cell_positions = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    total_cells = len(cell_positions)
    max_solids = (total_cells - 1) // 2
    if max_solids < 1:
        return None
    solid_count = random.randint(1, max_solids)
    solid_positions = set(random.sample(cell_positions, solid_count))

    inp, expected = _render_case(
        cell_rows=cell_rows,
        cell_cols=cell_cols,
        block_size=block_size,
        sep_width=sep_width,
        separator=separator,
        primary=primary,
        secondary=secondary,
        sprite=sprite,
        solid_positions=solid_positions,
    )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
