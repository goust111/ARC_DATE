from __future__ import annotations

from collections import Counter

import numpy as np


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background color"
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]] | None:
    if not separators:
        return None
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if sep <= start:
            return None
        spans.append((start, sep - 1))
        start = sep + 1
    if start >= length:
        return None
    spans.append((start, length - 1))
    return spans


def _mask_offsets(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "template must contain foreground pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    center_r = (int(r0) + int(r1)) // 2
    center_c = (int(c0) + int(c1)) // 2
    return tuple(sorted((int(r - center_r), int(c - center_c)) for r, c in coords))


def _cell_patch(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _parse_candidate(grid: np.ndarray, bg: int, sep_color: int) -> dict | None:
    if sep_color == bg:
        return None

    height, width = map(int, grid.shape)
    sep_rows = [r for r in range(height) if np.all(grid[r, :] == sep_color)]
    sep_cols = [c for c in range(width) if np.all(grid[:, c] == sep_color)]
    row_spans = _spans(height, sep_rows)
    col_spans = _spans(width, sep_cols)
    if row_spans is None or col_spans is None:
        return None
    if len(row_spans) > 4 or len(col_spans) < 2:
        return None

    row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    if len(row_heights) != 1 or len(col_widths) != 1:
        return None
    cell_h = row_heights.pop()
    cell_w = col_widths.pop()
    if not (5 <= cell_h <= 10 and 5 <= cell_w <= 10):
        return None

    legend: dict[int, tuple[tuple[int, int], ...]] = {}
    for row_span in row_spans:
        patch = _cell_patch(grid, row_span, col_spans[0])
        colors = set(map(int, np.unique(patch))) - {bg}
        if len(colors) != 1:
            return None
        color = colors.pop()
        if color == sep_color or color in legend:
            return None
        mask = patch == color
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0 + 1) > 5 or int(c1 - c0 + 1) > 5:
            return None
        legend[color] = _mask_offsets(mask)

    seeds: list[dict[str, int]] = []
    for row_span in row_spans:
        for col_span in col_spans[1:]:
            patch = _cell_patch(grid, row_span, col_span)
            colors = set(map(int, np.unique(patch))) - {bg}
            if len(colors) != 1:
                return None
            color = colors.pop()
            if color not in legend:
                return None
            coords = np.argwhere(patch == color)
            if len(coords) != 1:
                return None
            sr, sc = map(int, coords[0])
            seeds.append(
                {
                    "color": color,
                    "cell_r0": int(row_span[0]),
                    "cell_r1": int(row_span[1]),
                    "cell_c0": int(col_span[0]),
                    "cell_c1": int(col_span[1]),
                    "seed_r": int(row_span[0] + sr),
                    "seed_c": int(col_span[0] + sc),
                }
            )

    return {"bg": bg, "sep": sep_color, "legend": legend, "seeds": seeds}


def _parse_grid(grid: np.ndarray) -> dict:
    bg = _unique_mode(grid)
    candidates = []
    for color in np.unique(grid):
        parsed = _parse_candidate(grid, bg=bg, sep_color=int(color))
        if parsed is not None:
            candidates.append(parsed)
    assert len(candidates) == 1, "expected one valid separator-grid interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left legend column as a color-to-template lookup, then stamp the matched template around each seed pixel inside its cell.

    Input structure
    - The most common color is the background.
    - Single-pixel separator rows and columns split the grid into equal-sized cells.
    - The leftmost cell column is a legend: each legend cell contains one monochrome template in a unique color.
    - Every non-legend cell contains exactly one non-background seed pixel, whose color matches one legend template.

    Rule
    - Read each legend cell, crop its colored template, and convert the template to offsets relative to its bbox center.
    - For each seed pixel outside the legend, find the legend template with the same color and draw that template centered on the seed.
    - Clip any template pixels that would cross a separator or the grid edge.
    - Keep the legend column and separator lines unchanged.

    Key insight
    - Seed colors are lookup keys into the legend, not direct instructions about a fixed shape.
    """
    g = np.asarray(grid, dtype=int)
    layout = _parse_grid(g)
    out = g.copy()

    for seed in layout["seeds"]:
        color = int(seed["color"])
        for dr, dc in layout["legend"][color]:
            rr = int(seed["seed_r"] + dr)
            cc = int(seed["seed_c"] + dc)
            if seed["cell_r0"] <= rr <= seed["cell_r1"] and seed["cell_c0"] <= cc <= seed["cell_c1"]:
                out[rr, cc] = color
    return out


# --------------- generator ---------------

SEPARATOR_COLOR = 5
GRID_ROWS_RANGE = (2, 5)
GRID_DATA_COLS_RANGE = (2, 6)
CELL_SIZE_RANGE = (5, 11)
MAX_TEMPLATE_SIZE = 5
MAX_GENERATION_ATTEMPTS = 300
PALETTE = tuple(range(10))

TEMPLATE_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("010", "111", "010"),
    ("11", "11"),
    ("100", "010", "001"),
    ("111", "010"),
    ("10", "10", "11"),
    ("10", "11", "01"),
    ("101", "010", "101"),
    ("100", "110", "011"),
    ("111",),
    ("010", "111", "010", "010"),
    ("0100", "1110", "0100"),
    ("1110", "0010"),
)


def _pattern_mask(pattern: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in pattern], dtype=bool)


def _template_pool() -> list[dict]:
    templates: list[dict] = []
    for idx, pattern in enumerate(TEMPLATE_PATTERNS):
        mask = _pattern_mask(pattern)
        offsets = _mask_offsets(mask)
        templates.append({"id": idx, "mask": mask, "offsets": offsets})
    return templates


_TEMPLATES = _template_pool()


def _cell_origin(index: int, size: int) -> int:
    return index * (size + 1)


def _draw_separators(grid: np.ndarray, rows: int, cols: int, cell_h: int, cell_w: int) -> None:
    for row in range(1, rows):
        grid[_cell_origin(row, cell_h) - 1, :] = SEPARATOR_COLOR
    for col in range(1, cols):
        grid[:, _cell_origin(col, cell_w) - 1] = SEPARATOR_COLOR


def _place_mask(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    h, w = map(int, mask.shape)
    region = grid[top : top + h, left : left + w]
    region[mask] = color


def _seed_position(rng: np.random.Generator, cell_h: int, cell_w: int) -> tuple[int, int]:
    if rng.random() < 0.55:
        border = [
            (r, c)
            for r in range(cell_h)
            for c in range(cell_w)
            if r in (0, cell_h - 1) or c in (0, cell_w - 1)
        ]
        return tuple(map(int, border[int(rng.integers(0, len(border)))]))
    return int(rng.integers(0, cell_h)), int(rng.integers(0, cell_w))


def _draw_expected(
    inp: np.ndarray,
    assignments: list[list[int]],
    seeds: list[list[tuple[int, int]]],
    legend_offsets: dict[int, tuple[tuple[int, int], ...]],
    cell_h: int,
    cell_w: int,
) -> np.ndarray:
    out = inp.copy()
    for row in range(len(assignments)):
        for data_col in range(len(assignments[row])):
            color = int(assignments[row][data_col])
            seed_r, seed_c = seeds[row][data_col]
            cell_r0 = _cell_origin(row, cell_h)
            cell_c0 = _cell_origin(data_col + 1, cell_w)
            abs_r = cell_r0 + seed_r
            abs_c = cell_c0 + seed_c
            for dr, dc in legend_offsets[color]:
                rr = abs_r + dr
                cc = abs_c + dc
                if cell_r0 <= rr < cell_r0 + cell_h and cell_c0 <= cc < cell_c0 + cell_w:
                    out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    row_count = int(rng.integers(*GRID_ROWS_RANGE))
    data_cols = int(rng.integers(*GRID_DATA_COLS_RANGE))
    total_cols = data_cols + 1
    cell_h = int(rng.integers(*CELL_SIZE_RANGE))
    cell_w = int(rng.integers(*CELL_SIZE_RANGE))

    bg_choices = [color for color in PALETTE if color != SEPARATOR_COLOR]
    bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])
    legend_colors = [int(c) for c in rng.choice([c for c in PALETTE if c not in {bg, SEPARATOR_COLOR}], size=row_count, replace=False)]

    viable_templates = [tpl for tpl in _TEMPLATES if tpl["mask"].shape[0] <= cell_h and tpl["mask"].shape[1] <= cell_w]
    if len(viable_templates) < row_count:
        return None
    chosen_idx = rng.choice(len(viable_templates), size=row_count, replace=False)
    chosen_templates = [viable_templates[int(i)] for i in chosen_idx]

    height = row_count * cell_h + (row_count - 1)
    width = total_cols * cell_w + (total_cols - 1)
    grid = np.full((height, width), bg, dtype=int)
    _draw_separators(grid, row_count, total_cols, cell_h, cell_w)

    legend_offsets: dict[int, tuple[tuple[int, int], ...]] = {}
    for row, (color, template) in enumerate(zip(legend_colors, chosen_templates, strict=False)):
        mask = template["mask"]
        th, tw = map(int, mask.shape)
        cell_r0 = _cell_origin(row, cell_h)
        cell_c0 = 0
        top = cell_r0 + int(rng.integers(0, cell_h - th + 1))
        left = cell_c0 + int(rng.integers(0, cell_w - tw + 1))
        _place_mask(grid, top, left, mask, color)
        legend_offsets[color] = template["offsets"]

    positions = [(row, col) for row in range(row_count) for col in range(data_cols)]
    rng.shuffle(positions)
    mandatory = list(rng.permutation(np.array(legend_colors, dtype=int)))
    assignments = [[-1 for _ in range(data_cols)] for _ in range(row_count)]
    for (row, col), color in zip(positions[:row_count], mandatory, strict=False):
        assignments[row][col] = int(color)
    for row in range(row_count):
        for col in range(data_cols):
            if assignments[row][col] == -1:
                assignments[row][col] = int(legend_colors[int(rng.integers(0, len(legend_colors)))])

    seeds: list[list[tuple[int, int]]] = []
    for row in range(row_count):
        row_seeds: list[tuple[int, int]] = []
        for col in range(data_cols):
            seed_r, seed_c = _seed_position(rng, cell_h, cell_w)
            row_seeds.append((seed_r, seed_c))
            abs_r = _cell_origin(row, cell_h) + seed_r
            abs_c = _cell_origin(col + 1, cell_w) + seed_c
            grid[abs_r, abs_c] = assignments[row][col]
        seeds.append(row_seeds)

    expected = _draw_expected(grid, assignments, seeds, legend_offsets, cell_h, cell_w)
    if _unique_mode(grid) != bg:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
