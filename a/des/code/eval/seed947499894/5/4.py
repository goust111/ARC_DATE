from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SOURCE_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, MAROON)
COLOR_NAMES = {
    BLACK: "black",
    BLUE: "blue",
    RED: "red",
    GREEN: "green",
    YELLOW: "yellow",
    GREY: "grey",
    MAGENTA: "magenta",
    ORANGE: "orange",
    CYAN: "cyan",
    MAROON: "maroon",
}

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PATTERNS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)

GRID_ROWS_RANGE = (3, 5)
GRID_COLS_RANGE = (3, 5)
CELL_HEIGHT_RANGE = (3, 6)
CELL_WIDTH_RANGE = (3, 6)
PALETTE_SIZE_RANGE = (2, 5)
SOURCE_DENSITY_RANGE = (0.18, 0.48)
GEN_RETRIES = 400


def _interior_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(v) for v in separators) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _parse_cells(grid: np.ndarray) -> tuple[list[list[dict[str, int | None]]], Counter]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    row_spans = _interior_spans(h, sep_rows)
    col_spans = _interior_spans(w, sep_cols)

    assert row_spans and col_spans, "expected grey separator grid"

    cells: list[list[dict[str, int | None]]] = []
    counts: Counter = Counter()
    for r0, r1 in row_spans:
        row: list[dict[str, int | None]] = []
        for c0, c1 in col_spans:
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(window == GREY), "grey separators must stay outside cell interiors"
            colors = [int(v) for v in np.unique(window) if int(v) not in {BLACK}]
            assert len(colors) <= 1, "each cell must be blank or contain exactly one source color"
            source_color = colors[0] if colors else None
            if source_color is not None:
                counts[source_color] += 1
            row.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "color": source_color})
        cells.append(row)
    return cells, counts


def _neighbor_positions(rows: int, cols: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _cell_area(cell: dict[str, int | None]) -> int:
    return (int(cell["r1"]) - int(cell["r0"]) + 1) * (int(cell["c1"]) - int(cell["c0"]) + 1)


def _pick_fill_color(colors: set[int], source_counts: Counter, source_areas: Counter, allow_unresolved: bool = False) -> int | None:
    assert len(colors) == 2
    a, b = tuple(colors)
    count_a = int(source_counts[int(a)])
    count_b = int(source_counts[int(b)])
    if count_a != count_b:
        return int(a) if count_a < count_b else int(b)

    area_a = int(source_areas[int(a)])
    area_b = int(source_areas[int(b)])
    if area_a == area_b:
        if allow_unresolved:
            return None
        raise AssertionError("equal source counts must be resolved by unequal source cell areas")
    return int(a) if area_a < area_b else int(b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grey-partitioned table, source cells broadcast their color to directly adjacent cells.

    Blank cells that receive signals from exactly two distinct source colors are activated and
    filled with the globally rarer of those two source colors. Ties use the area of the source
    cell that sent the signal, preferring the smaller source cell area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cells, source_counts = _parse_cells(g)
    rows, cols = len(cells), len(cells[0])
    out = g.copy()

    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell["color"] is not None:
                continue
            incoming: set[int] = set()
            incoming_areas: Counter = Counter()
            for nr, nc in _neighbor_positions(rows, cols, r, c):
                source = cells[nr][nc]
                if source["color"] is None:
                    continue
                color = int(source["color"])
                incoming.add(color)
                incoming_areas[color] += _cell_area(source)
            if len(incoming) != 2:
                continue
            fill = _pick_fill_color(incoming, source_counts, incoming_areas)
            assert fill is not None
            out[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1] = fill

    return out


def _analyze_assignments(
    assignments: list[list[int | None]],
    cell_areas: list[list[int]] | None = None,
) -> tuple[Counter, list[tuple[int, int, tuple[int, ...]]], list[tuple[int, int, tuple[int, int], int]]]:
    rows, cols = len(assignments), len(assignments[0])
    if cell_areas is None:
        cell_areas = [[1 for _ in range(cols)] for _ in range(rows)]
    counts: Counter = Counter()
    for row in assignments:
        for color in row:
            if color is not None:
                counts[int(color)] += 1

    blanks: list[tuple[int, int, tuple[int, ...]]] = []
    activated: list[tuple[int, int, tuple[int, int], int]] = []
    for r in range(rows):
        for c in range(cols):
            if assignments[r][c] is not None:
                continue
            incoming: set[int] = set()
            incoming_areas: Counter = Counter()
            for nr, nc in _neighbor_positions(rows, cols, r, c):
                color = assignments[nr][nc]
                if color is None:
                    continue
                incoming.add(int(color))
                incoming_areas[int(color)] += int(cell_areas[nr][nc])
            incoming_tuple = tuple(sorted(incoming))
            blanks.append((r, c, incoming_tuple))
            if len(incoming_tuple) == 2:
                chosen = _pick_fill_color(set(incoming_tuple), counts, incoming_areas, allow_unresolved=True)
                if chosen is not None:
                    activated.append((r, c, (incoming_tuple[0], incoming_tuple[1]), chosen))

    return counts, blanks, activated


def _random_spans(count: int, size_range: tuple[int, int]) -> list[int]:
    return [random.randint(size_range[0], size_range[1]) for _ in range(count)]


def _render_table(
    row_heights: list[int],
    col_widths: list[int],
    assignments: list[list[int | None]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    start = 1
    for cell_h in row_heights:
        row_spans.append((start, start + cell_h - 1))
        start += cell_h + 1

    col_spans: list[tuple[int, int]] = []
    start = 1
    for cell_w in col_widths:
        col_spans.append((start, start + cell_w - 1))
        start += cell_w + 1

    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    for _, r1 in row_spans[:-1]:
        grid[r1 + 1, :] = GREY
    for _, c1 in col_spans[:-1]:
        grid[:, c1 + 1] = GREY

    for r, row in enumerate(assignments):
        for c, color in enumerate(row):
            if color is None:
                continue
            r0, r1 = row_spans[r]
            c0, c1 = col_spans[c]
            pattern = _sample_pattern(r1 - r0 + 1, c1 - c0 + 1)
            ph = max(rr for rr, _ in pattern) + 1
            pw = max(cc for _, cc in pattern) + 1
            base_r = random.randint(r0, r1 - ph + 1)
            base_c = random.randint(c0, c1 - pw + 1)
            for dr, dc in pattern:
                grid[base_r + dr, base_c + dc] = int(color)

    return grid, row_spans, col_spans


def _sample_pattern(height: int, width: int) -> tuple[tuple[int, int], ...]:
    fitting = []
    for pattern in PATTERNS:
        ph = max(r for r, _ in pattern) + 1
        pw = max(c for _, c in pattern) + 1
        if ph <= height and pw <= width:
            fitting.append(pattern)
    assert fitting, "each source cell must be large enough for a pattern"
    return random.choice(fitting)


def _expected_output(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    assignments: list[list[int | None]],
) -> np.ndarray:
    out = grid.copy()
    cell_areas = [
        [
            (row_spans[r][1] - row_spans[r][0] + 1) * (col_spans[c][1] - col_spans[c][0] + 1)
            for c in range(len(col_spans))
        ]
        for r in range(len(row_spans))
    ]
    _counts, _blanks, activated = _analyze_assignments(assignments, cell_areas)
    for r, c, _incoming, chosen in activated:
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        out[r0 : r1 + 1, c0 : c1 + 1] = chosen
    return out


def _candidate_palette(want_tie: bool) -> list[int]:
    lo, hi = PALETTE_SIZE_RANGE
    if want_tie:
        lo = max(lo, 3)
    size = random.randint(lo, min(hi, len(SOURCE_COLORS)))
    return random.sample(list(SOURCE_COLORS), size)


def _choose_target(rows: int, cols: int) -> tuple[int, int]:
    return random.randint(1, rows - 2), random.randint(1, cols - 2)


def _build_assignments(rows: int, cols: int, want_tie: bool) -> list[list[int | None]]:
    palette = _candidate_palette(want_tie)
    color_a, color_b = palette[0], palette[1]
    extras = palette[2:]

    assignments: list[list[int | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    target_r, target_c = _choose_target(rows, cols)
    directions = DIR4[:]
    random.shuffle(directions)

    primary_dirs = directions[:2]
    secondary_dirs = directions[2:]
    for color, (dr, dc) in zip((color_a, color_b), primary_dirs):
        assignments[target_r + dr][target_c + dc] = color

    secondary_cells = [(target_r + dr, target_c + dc) for dr, dc in secondary_dirs]
    if want_tie:
        pattern = random.choice(
            [
                (None, None),
                (color_a, color_b),
                (color_b, color_a),
            ]
        )
        for (r, c), color in zip(secondary_cells, pattern):
            if color is not None:
                assignments[r][c] = color
        fill_choices = extras
    else:
        common = random.choice((color_a, color_b))
        for idx, (r, c) in enumerate(secondary_cells):
            if idx == 0 or random.random() < 0.25:
                assignments[r][c] = common
        fill_choices = extras + [common]

    protected = {(target_r, target_c)}
    protected.update(_neighbor_positions(rows, cols, target_r, target_c))

    density = random.uniform(SOURCE_DENSITY_RANGE[0], SOURCE_DENSITY_RANGE[1])
    for r in range(rows):
        for c in range(cols):
            if (r, c) in protected:
                continue
            if random.random() >= density:
                continue
            assignments[r][c] = random.choice(fill_choices)

    return assignments


def _is_interesting(assignments: list[list[int | None]], want_tie: bool, cell_areas: list[list[int]]) -> bool:
    counts, blanks, activated = _analyze_assignments(assignments, cell_areas)
    if not counts or not blanks or not activated:
        return False
    rows, cols = len(assignments), len(assignments[0])
    for r, c, incoming in blanks:
        if len(incoming) != 2:
            continue
        a, b = incoming
        if counts[a] != counts[b]:
            continue
        area_a = sum(
            cell_areas[nr][nc]
            for nr, nc in _neighbor_positions(rows, cols, r, c)
            if assignments[nr][nc] == a
        )
        area_b = sum(
            cell_areas[nr][nc]
            for nr, nc in _neighbor_positions(rows, cols, r, c)
            if assignments[nr][nc] == b
        )
        if area_a == area_b:
            return False
    if len(activated) >= len(blanks):
        return False

    signal_sizes = [len(incoming) for _, _, incoming in blanks]
    if not any(size in {1, 3, 4} for size in signal_sizes):
        return False

    if want_tie:
        return any(
            counts[a] == counts[b]
            and sum(
                cell_areas[nr][nc]
                for nr, nc in _neighbor_positions(rows, cols, r, c)
                if assignments[nr][nc] == a
            )
            != sum(
                cell_areas[nr][nc]
                for nr, nc in _neighbor_positions(rows, cols, r, c)
                if assignments[nr][nc] == b
            )
            for r, c, (a, b), _ in activated
        )
    return any(counts[a] != counts[b] for _, _, (a, b), _ in activated)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1])
    cols = random.randint(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1])
    want_tie = random.random() < 0.5

    assignments = _build_assignments(rows, cols, want_tie)
    row_heights = _random_spans(rows, CELL_HEIGHT_RANGE)
    col_widths = _random_spans(cols, CELL_WIDTH_RANGE)
    cell_areas = [[row_heights[r] * col_widths[c] for c in range(cols)] for r in range(rows)]
    if not _is_interesting(assignments, want_tie, cell_areas):
        return None

    grid, row_spans, col_spans = _render_table(row_heights, col_widths, assignments)
    expected = _expected_output(grid, row_spans, col_spans, assignments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-broadcast ARC input")
