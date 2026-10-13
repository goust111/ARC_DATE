import random

import numpy as np


BG = 0
CELL_SIZE = 3
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    lines = sorted(set(map(int, lines)))
    assert lines and lines[0] == 0 and lines[-1] == limit - 1
    regions: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2
        regions.append((a + 1, b - 1))
    return regions


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


TEMPLATE_LIBRARY = [
    _mask_from_coords([(0, 0), (0, 1), (1, 0), (2, 0)]),
    _mask_from_coords([(0, 0), (0, 1), (0, 2), (1, 1)]),
    _mask_from_coords([(0, 0), (1, 0), (1, 1), (2, 1)]),
    _mask_from_coords([(0, 1), (0, 2), (1, 1), (2, 1)]),
    _mask_from_coords([(0, 2), (1, 2), (2, 1), (2, 2)]),
    _mask_from_coords([(0, 0), (1, 0), (1, 1), (1, 2)]),
    _mask_from_coords([(0, 1), (1, 1), (1, 2), (2, 2)]),
    _mask_from_coords([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]),
    _mask_from_coords([(0, 2), (1, 0), (1, 1), (2, 0)]),
    _mask_from_coords([(0, 1), (1, 0), (1, 1), (2, 0)]),
]

for _template in TEMPLATE_LIBRARY:
    assert not np.array_equal(_template, np.rot90(_template, 2))


def _find_table(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, list[tuple[int, int]], list[tuple[int, int]], int]:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
        if len(full_rows) >= 2:
            candidates.append((len(full_rows), color, full_rows))
    assert candidates, "expected a separator-line color"

    candidates.sort(reverse=True)
    line_color = int(candidates[0][1])
    table_top = int(min(candidates[0][2]))

    lower = grid[table_top:, :]
    full_cols = [c for c in range(lower.shape[1]) if np.all(lower[:, c] == line_color)]
    assert full_cols, "expected full separator columns"
    c0, c1 = int(min(full_cols)), int(max(full_cols))
    table = lower[:, c0 : c1 + 1].copy()

    row_lines = [r for r in range(table.shape[0]) if np.all(table[r, :] == line_color)]
    col_lines = [c for c in range(table.shape[1]) if np.all(table[:, c] == line_color)]
    assert row_lines and col_lines
    line_mask = table == line_color
    expected = np.zeros_like(line_mask, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    assert np.array_equal(line_mask, expected), "separator color must form a full rectangular lattice"

    row_regions = _regions_between(row_lines, table.shape[0])
    col_regions = _regions_between(col_lines, table.shape[1])
    assert len(row_regions) % 2 == 1 and len(col_regions) % 2 == 1

    heights = {r1 - r0 + 1 for r0, r1 in row_regions}
    widths = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert len(heights) == 1 and len(widths) == 1
    cell_h = heights.pop()
    cell_w = widths.pop()
    assert cell_h == cell_w == CELL_SIZE

    return table, line_color, row_regions, col_regions, table_top


def _parse_legend(legend: np.ndarray, bg: int, tile: int) -> dict[int, np.ndarray]:
    non_bg = legend[legend != bg]
    assert non_bg.size > 0, "legend must contain markers and templates"

    colors, counts = np.unique(non_bg, return_counts=True)
    best = int(np.argmax(counts))
    ink_color = int(colors[best])
    assert np.sum(counts == counts[best]) == 1, "legend ink color must be uniquely most common in legend band"

    templates: dict[int, np.ndarray] = {}
    covered = np.zeros_like(legend, dtype=bool)

    marker_positions = np.argwhere((legend != bg) & (legend != ink_color))
    for r, c in marker_positions:
        r = int(r)
        c = int(c)
        assert r + tile <= legend.shape[0] and c + 1 + tile <= legend.shape[1], "legend template must fit"
        marker_color = int(legend[r, c])
        assert marker_color not in templates, "marker colors must be unique in legend"
        assert np.all(legend[r + 1 : r + tile, c] == bg), "marker column must be otherwise empty"
        patch = legend[r : r + tile, c + 1 : c + 1 + tile]
        assert np.all(np.isin(patch, [bg, ink_color])), "legend patch may contain only bg and common ink"
        mask = patch == ink_color
        assert mask.any(), "legend template must contain ink"
        assert not np.array_equal(mask, np.rot90(mask, 2)), "templates must break 180-degree symmetry"
        templates[marker_color] = mask
        covered[r, c] = True
        covered[r : r + tile, c + 1 : c + 1 + tile] |= mask

    assert templates, "expected at least one legend entry"
    assert np.array_equal(legend != bg, covered), "legend pixels must be fully explained by marker-template entries"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each control cell's matching legend template into its center-symmetric cell after a 180-degree rotation.

    Input structure:
    - A top legend contains several colored marker pixels, each immediately left of a same-size template drawn in one shared ink color.
    - Below the legend, full separator lines form an odd-by-odd table of square cells with a single center cell.
    - Some table cells contain one colored control pixel; the remaining cells are empty.

    Rule:
    - Read the legend as a color-to-template-mask mapping.
    - Remove all control pixels from the table.
    - For each control pixel, find the cell reflected through the table center, rotate that color's template by 180 degrees, and stamp it there in the control color.
    - Return only the table region, preserving all separator lines.

    Key insight:
    - The macro placement and the micro template orientation undergo the same point-symmetry transform: the destination cell is center-symmetric, and the template itself must also be turned by 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    table, line_color, row_regions, col_regions, table_top = _find_table(g, bg)
    legend = g[:table_top, :]
    templates = _parse_legend(legend, bg, CELL_SIZE)

    out = np.full_like(table, bg)
    out[table == line_color] = line_color

    n_rows = len(row_regions)
    n_cols = len(col_regions)
    for row_idx, (r0, r1) in enumerate(row_regions):
        for col_idx, (c0, c1) in enumerate(col_regions):
            cell = table[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere(cell != bg)
            if coords.size == 0:
                continue
            assert coords.shape[0] == 1, "each occupied cell must contain one control pixel"
            control_color = int(cell[tuple(coords[0])])
            assert control_color in templates, "every control color must be defined by the legend"

            target_row = n_rows - 1 - row_idx
            target_col = n_cols - 1 - col_idx
            tr0, tr1 = row_regions[target_row]
            tc0, tc1 = col_regions[target_col]
            mask = np.rot90(templates[control_color], 2)
            out[tr0 : tr1 + 1, tc0 : tc1 + 1][mask] = control_color

    return out


def _table_shape() -> tuple[int, int]:
    return random.choice([3, 5]), random.choice([3, 5])


def _legend_count(n_cols: int) -> int:
    if n_cols == 3:
        return 2
    return random.randint(2, 4)


def _entry_span(count: int) -> int:
    return count * (CELL_SIZE + 1) + (count - 1)


def _table_size(n_cells: int) -> int:
    return n_cells * CELL_SIZE + (n_cells + 1)


def _cell_bounds(index: int) -> tuple[int, int]:
    start = 1 + index * (CELL_SIZE + 1)
    return start, start + CELL_SIZE - 1


def _draw_table(grid: np.ndarray, top: int, line_color: int, n_rows: int, n_cols: int) -> None:
    table_h = _table_size(n_rows)
    table_w = _table_size(n_cols)
    for k in range(n_rows + 1):
        row = top + k * (CELL_SIZE + 1)
        grid[row, :table_w] = line_color
    for k in range(n_cols + 1):
        col = k * (CELL_SIZE + 1)
        grid[top : top + table_h, col] = line_color


def _gen_controls(n_rows: int, n_cols: int, legend_colors: list[int]) -> list[tuple[int, int, int]]:
    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    center = (n_rows // 2, n_cols // 2)
    force_center = random.random() < 0.5

    min_controls = max(len(legend_colors), 2)
    max_controls = min(len(all_cells), len(legend_colors) + (2 if len(all_cells) == 9 else 4))
    control_count = random.randint(min_controls, max_controls)

    chosen: list[tuple[int, int]] = []
    if force_center:
        chosen.append(center)
    remaining = [cell for cell in all_cells if cell not in chosen]
    random.shuffle(remaining)
    while len(chosen) < control_count:
        chosen.append(remaining.pop())

    random.shuffle(chosen)
    color_bag = legend_colors.copy()
    random.shuffle(color_bag)
    assignments: list[tuple[int, int, int]] = []
    for idx, (r, c) in enumerate(chosen):
        if idx < len(color_bag):
            color = color_bag[idx]
        else:
            color = random.choice(legend_colors)
        assignments.append((r, c, color))

    assert len({color for _r, _c, color in assignments}) == len(legend_colors)
    if force_center:
        assert any((r, c) != center for r, c, _color in assignments)
    return assignments


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = _table_shape()
    legend_count = _legend_count(n_cols)
    table_h = _table_size(n_rows)
    table_w = _table_size(n_cols)

    legend_top = random.randint(0, 1)
    gap = random.randint(1, 2)
    input_h = legend_top + CELL_SIZE + gap + table_h
    input_w = table_w
    if max(input_h, input_w) > 30:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    line_color = palette.pop()
    ink_color = palette.pop()
    legend_colors = random.sample(palette, legend_count)
    templates = random.sample(TEMPLATE_LIBRARY, legend_count)

    grid = np.full((input_h, input_w), BG, dtype=int)

    span = _entry_span(legend_count)
    legend_left = random.randint(0, input_w - span)
    for idx, (color, mask) in enumerate(zip(legend_colors, templates)):
        c0 = legend_left + idx * (CELL_SIZE + 2)
        r0 = legend_top
        grid[r0, c0] = color
        grid[r0 : r0 + CELL_SIZE, c0 + 1 : c0 + 1 + CELL_SIZE][mask] = ink_color

    table_top = legend_top + CELL_SIZE + gap
    _draw_table(grid, table_top, line_color, n_rows, n_cols)

    controls = _gen_controls(n_rows, n_cols, legend_colors)
    for cell_r, cell_c, color in controls:
        r0, r1 = _cell_bounds(cell_r)
        c0, c1 = _cell_bounds(cell_c)
        dr = random.randint(0, CELL_SIZE - 1)
        dc = random.randint(0, CELL_SIZE - 1)
        grid[table_top + r0 + dr, c0 + dc] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (table_h, table_w):
        return None
    if np.array_equal(out, grid[table_top:, :table_w]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
