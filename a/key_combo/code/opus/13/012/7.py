from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
TEMPLATE_BLUE = 1
TEMPLATE_RED = 2
PREFIX_GREEN = 3
PREFIX_YELLOW = 4
DIVIDER = 5
MARKER_COLORS = (6, 7, 8, 9)

TABLE_BANDS = 4
DATA_ROWS = (1, 2, 3)
TEMPLATE_COLORS = (TEMPLATE_BLUE, TEMPLATE_RED)
PREFIX_COLORS = (PREFIX_GREEN, PREFIX_YELLOW)

GEN_CELL_SIZES = (3, 4, 5, 6)
GEN_CELL_SIZE_WEIGHTS = (3, 3, 2, 1)
GEN_ATTEMPTS = 400
GEN_TEMPLATE_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _full_divider_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return rows, cols


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    dividers = set(int(idx) for idx in divider_indices)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in dividers:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _table_cells(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    bg = _most_common_color(grid)
    divider_candidates = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        rows, cols = _full_divider_indices(grid, color)
        if rows and cols:
            divider_candidates.append(int(color))
    assert len(divider_candidates) == 1, "expected one full-row/full-column divider color"
    divider = divider_candidates[0]

    row_divs, col_divs = _full_divider_indices(grid, divider)
    row_runs = _non_divider_runs(grid.shape[0], row_divs)
    col_runs = _non_divider_runs(grid.shape[1], col_divs)
    assert len(row_runs) == TABLE_BANDS and len(col_runs) == TABLE_BANDS, "expected a 4x4 table"

    cells: dict[tuple[int, int], np.ndarray] = {}
    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            cells[(ri, ci)] = grid[r0 : r1 + 1, c0 : c1 + 1]
    return divider, row_runs, col_runs, cells


def _decode_prefix_cell(region: np.ndarray, template: np.ndarray) -> np.ndarray:
    height, width = region.shape
    active_cols = [c for c in range(width) if np.any(region[:, c] != BG)]
    assert active_cols, "prefix cell must contain foreground pixels"
    assert active_cols == list(range(active_cols[-1] + 1)), "prefix must occupy a contiguous left column band"
    prefix_width = active_cols[-1] + 1
    assert prefix_width < width, "prefix must leave some template columns to complete"

    mapping: dict[int, int] = {}
    for r in range(height):
        for c in range(prefix_width):
            cell_value = int(region[r, c])
            template_value = int(template[r, c])
            if cell_value == BG:
                assert template_value == BG, "prefix may omit only template-background pixels"
                continue
            assert cell_value in PREFIX_COLORS, "prefix pixels must use the two prefix colors"
            assert template_value in TEMPLATE_COLORS, "prefix pixels must align to template-colored positions"
            if template_value in mapping:
                assert mapping[template_value] == cell_value, "prefix mapping must stay consistent"
            else:
                mapping[template_value] = cell_value

    assert set(mapping) == set(TEMPLATE_COLORS), "prefix must expose both template colors"
    assert len(set(mapping.values())) == 2, "template colors must map to two distinct prefix colors"
    assert np.any(np.isin(template[:, prefix_width:], TEMPLATE_COLORS)), "prefix must be incomplete"

    completed = np.zeros_like(region)
    for template_color, prefix_color in mapping.items():
        completed[template == template_color] = prefix_color
    return completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each prefix-coded table cell from its column template, then overwrite the completed mask with the unique aligned marker color.

    Input structure:
    - Black background and grey full divider lines split the grid into a uniform 4x4 table.
    - The first table row contains four template cells; each template is a background-with-holes pattern colored by exactly two template colors.
    - Some lower cells contain left-side prefixes using two prefix colors, and some other lower cells contain exactly one marker pixel in a marker color.

    Rule:
    - Segment the grid into table cells using the full divider lines.
    - For each prefix cell, compare its filled prefix columns with the template in the same table column to decode the template-color to prefix-color mapping.
    - Reconstruct the full template-shaped foreground in that cell with the decoded prefix colors.
    - Find the unique marker cell that shares the prefix cell's table row or table column, and recolor every reconstructed foreground pixel to that marker color.
    - Keep the background, divider lines, templates, empty cells, and marker cells unchanged.

    Key insight:
    - The prefix only reveals how to translate the column template, while the final color comes from a separate singleton marker aligned in the table coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _divider, row_runs, col_runs, cells = _table_cells(g)
    templates = [cells[(0, ci)] for ci in range(TABLE_BANDS)]
    for template in templates:
        vals = set(map(int, np.unique(template))) - {BG}
        assert vals == set(TEMPLATE_COLORS), "template cells must use exactly the two template colors"

    markers: dict[tuple[int, int], int] = {}
    prefix_locs: list[tuple[int, int]] = []

    for ri in DATA_ROWS:
        for ci in range(TABLE_BANDS):
            region = cells[(ri, ci)]
            coords = np.argwhere(region != BG)
            if coords.size == 0:
                continue
            colors = set(map(int, region[region != BG]))
            if coords.shape[0] == 1:
                marker_color = int(region[tuple(coords[0])])
                assert marker_color in MARKER_COLORS, "marker cells must use marker colors"
                markers[(ri, ci)] = marker_color
                continue
            assert colors <= set(PREFIX_COLORS), "non-marker data cells must be prefix cells"
            prefix_locs.append((ri, ci))

    assert markers, "expected marker cells"
    assert prefix_locs, "expected at least one prefix cell"

    out = g.copy()
    for ri, ci in prefix_locs:
        template = templates[ci]
        completed = _decode_prefix_cell(cells[(ri, ci)], template)
        aligned = [
            color
            for (marker_ri, marker_ci), color in markers.items()
            if marker_ri == ri or marker_ci == ci
        ]
        assert len(aligned) == 1, "each prefix cell must have exactly one aligned marker"
        final_cell = np.zeros_like(completed)
        final_cell[completed != BG] = aligned[0]

        r0, r1 = row_runs[ri]
        c0, c1 = col_runs[ci]
        out[r0 : r1 + 1, c0 : c1 + 1] = final_cell

    return out


def _cell_origin(index: int, cell_size: int) -> int:
    return int(index) * (int(cell_size) + 1)


def _draw_dividers(grid: np.ndarray, cell_size: int) -> None:
    for idx in range(1, TABLE_BANDS):
        pos = idx * (cell_size + 1) - 1
        grid[pos, :] = DIVIDER
        grid[:, pos] = DIVIDER


def _grow_connected_mask(cell_size: int, target_area: int, rng: np.random.Generator) -> np.ndarray:
    start = (int(rng.integers(cell_size)), int(rng.integers(cell_size)))
    cells = {start}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < cell_size and 0 <= cc < cell_size and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])

    mask = np.zeros((cell_size, cell_size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _candidate_prefix_widths(template: np.ndarray) -> list[int]:
    widths = []
    height, width = template.shape
    for prefix_width in range(1, width):
        if not all(np.any(template[:, c] != BG) for c in range(prefix_width)):
            continue
        left_roles = set(
            map(
                int,
                template[:, :prefix_width][np.isin(template[:, :prefix_width], TEMPLATE_COLORS)],
            )
        )
        if left_roles != set(TEMPLATE_COLORS):
            continue
        if not np.any(np.isin(template[:, prefix_width:], TEMPLATE_COLORS)):
            continue
        if np.all(template[:, prefix_width:] == BG):
            continue
        widths.append(prefix_width)
    return widths


def _make_template(cell_size: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    min_area = max(cell_size + 1, 4)
    max_area = min((cell_size * cell_size) - 2, max(min_area, int(round(cell_size * cell_size * 0.55))))
    for _ in range(GEN_TEMPLATE_ATTEMPTS):
        area = int(rng.integers(min_area, max_area + 1))
        mask = _grow_connected_mask(cell_size, area, rng)
        if np.any(mask.sum(axis=0) == 0):
            continue
        if np.any(mask.sum(axis=1) == 0):
            continue
        coords = np.argwhere(mask)
        if coords.shape[0] < 4:
            continue

        role_bits = rng.integers(0, 2, size=coords.shape[0])
        if 0 not in role_bits or 1 not in role_bits:
            continue
        template = np.zeros((cell_size, cell_size), dtype=int)
        for idx, (r, c) in enumerate(coords):
            template[int(r), int(c)] = TEMPLATE_COLORS[int(role_bits[idx])]

        widths = _candidate_prefix_widths(template)
        if widths:
            return template, widths
    raise RuntimeError("failed to generate a valid template")


def _place_cell(grid: np.ndarray, cell_size: int, ri: int, ci: int, content: np.ndarray) -> None:
    r0 = _cell_origin(ri, cell_size)
    c0 = _cell_origin(ci, cell_size)
    grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = content


def _prefix_from_template(
    template: np.ndarray,
    prefix_width: int,
    mapping: dict[int, int],
) -> np.ndarray:
    region = np.zeros_like(template)
    for template_color, prefix_color in mapping.items():
        region[:, :prefix_width][template[:, :prefix_width] == template_color] = prefix_color
    return region


def _mandatory_row_mode(marker_col: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    mandatory: set[tuple[int, int]] = set()
    open_cols = [ci for ci in range(TABLE_BANDS) if ci != marker_col]
    for ri in DATA_ROWS:
        mandatory.add((ri, int(rng.choice(open_cols))))
    for ci in open_cols:
        mandatory.add((int(rng.choice(DATA_ROWS)), ci))
    return mandatory


def _mandatory_col_mode(marker_row: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    mandatory: set[tuple[int, int]] = set()
    open_rows = [ri for ri in DATA_ROWS if ri != marker_row]
    for ri in open_rows:
        mandatory.add((ri, int(rng.integers(TABLE_BANDS))))
    for ci in range(TABLE_BANDS):
        mandatory.add((int(rng.choice(open_rows)), ci))
    return mandatory


def _choose_prefix_cells(mode: str, fixed_band: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    if mode == "row":
        marker_colors = {
            (ri, fixed_band): color
            for ri, color in zip(DATA_ROWS, rng.choice(MARKER_COLORS, size=len(DATA_ROWS), replace=False))
        }
        eligible = [(ri, ci) for ri in DATA_ROWS for ci in range(TABLE_BANDS) if ci != fixed_band]
        mandatory = _mandatory_row_mode(fixed_band, rng)
    else:
        marker_colors = {
            (fixed_band, ci): color
            for ci, color in enumerate(rng.choice(MARKER_COLORS, size=TABLE_BANDS, replace=False))
        }
        eligible = [(ri, ci) for ri in DATA_ROWS for ci in range(TABLE_BANDS) if ri != fixed_band]
        mandatory = _mandatory_col_mode(fixed_band, rng)

    max_prefix = len(eligible) - 1
    min_prefix = max(len(mandatory), 4)
    target = int(rng.integers(min_prefix, max_prefix + 1))
    prefix_cells = set(mandatory)
    extras = [cell for cell in eligible if cell not in prefix_cells]
    rng.shuffle(extras)
    for cell in extras[: max(0, target - len(prefix_cells))]:
        prefix_cells.add(cell)
    return sorted(prefix_cells), marker_colors


def _expected_output(
    grid: np.ndarray,
    cell_size: int,
    templates: list[np.ndarray],
    prefix_specs: dict[tuple[int, int], tuple[int, dict[int, int]]],
    marker_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for (ri, ci), (_prefix_width, mapping) in prefix_specs.items():
        completed = np.zeros_like(templates[ci])
        for template_color, prefix_color in mapping.items():
            completed[templates[ci] == template_color] = prefix_color
        aligned = [
            color
            for (marker_ri, marker_ci), color in marker_colors.items()
            if marker_ri == ri or marker_ci == ci
        ]
        assert len(aligned) == 1
        cell = np.zeros_like(completed)
        cell[completed != BG] = aligned[0]
        _place_cell(out, cell_size, ri, ci, cell)
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_size = int(rng.choice(GEN_CELL_SIZES, p=np.array(GEN_CELL_SIZE_WEIGHTS, dtype=float) / sum(GEN_CELL_SIZE_WEIGHTS)))
    side = TABLE_BANDS * cell_size + (TABLE_BANDS - 1)
    grid = np.full((side, side), BG, dtype=int)
    _draw_dividers(grid, cell_size)

    templates: list[np.ndarray] = []
    prefix_options: list[list[int]] = []
    seen_templates = set()
    for ci in range(TABLE_BANDS):
        for _ in range(GEN_TEMPLATE_ATTEMPTS):
            template, widths = _make_template(cell_size, rng)
            signature = template.tobytes()
            if signature in seen_templates:
                continue
            seen_templates.add(signature)
            templates.append(template)
            prefix_options.append(widths)
            _place_cell(grid, cell_size, 0, ci, template)
            break
        else:
            return None

    mode = str(rng.choice(np.array(["row", "col"])))
    fixed_band = int(rng.integers(TABLE_BANDS)) if mode == "row" else int(rng.choice(DATA_ROWS))
    prefix_cells, marker_colors = _choose_prefix_cells(mode, fixed_band, rng)

    prefix_specs: dict[tuple[int, int], tuple[int, dict[int, int]]] = {}
    for (ri, ci), marker_color in marker_colors.items():
        marker_cell = np.zeros((cell_size, cell_size), dtype=int)
        mr = int(rng.integers(cell_size))
        mc = int(rng.integers(cell_size))
        marker_cell[mr, mc] = int(marker_color)
        _place_cell(grid, cell_size, ri, ci, marker_cell)

    for ri, ci in prefix_cells:
        widths = prefix_options[ci]
        prefix_width = int(rng.choice(widths))
        if bool(rng.integers(2)):
            mapping = {TEMPLATE_BLUE: PREFIX_GREEN, TEMPLATE_RED: PREFIX_YELLOW}
        else:
            mapping = {TEMPLATE_BLUE: PREFIX_YELLOW, TEMPLATE_RED: PREFIX_GREEN}
        prefix_region = _prefix_from_template(templates[ci], prefix_width, mapping)
        prefix_specs[(ri, ci)] = (prefix_width, mapping)
        _place_cell(grid, cell_size, ri, ci, prefix_region)

    expected = _expected_output(grid, cell_size, templates, prefix_specs, marker_colors)
    try:
        solved = main(grid)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
