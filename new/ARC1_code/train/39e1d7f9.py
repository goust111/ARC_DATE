from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


Span = tuple[int, int]


def _groups(indices: list[int]) -> list[Span]:
    assert indices, "expected separator lines"
    groups: list[Span] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_between(line_groups: list[Span], limit: int) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))

    assert spans, "expected cells between separator lines"
    sizes = {end - start + 1 for start, end in spans}
    assert len(sizes) == 1, "lattice cells must have a common size"
    return spans


def _separator_color(g: np.ndarray) -> int:
    row_colors = {
        int(g[r, 0])
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0])
    }
    col_colors = {
        int(g[0, c])
        for c in range(g.shape[1])
        if np.all(g[:, c] == g[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one full-row/full-column separator color"
    return candidates.pop()


def _compress_lattice(g: np.ndarray) -> tuple[list[Span], list[Span], np.ndarray]:
    separator = _separator_color(g)
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]

    row_spans = _spans_between(_groups(row_lines), g.shape[0])
    col_spans = _spans_between(_groups(col_lines), g.shape[1])
    cell_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    cell_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert cell_heights == cell_widths, "lattice cells must be square"

    cells = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            values = np.unique(block)
            assert len(values) == 1, "each lattice cell must be uniformly colored"
            cells[r, c] = int(values[0])
    return row_spans, col_spans, cells


def _neighbor_values(cells: np.ndarray, bg: int, r: int, c: int) -> list[tuple[int, int, int]]:
    h, w = cells.shape
    values: list[tuple[int, int, int]] = []
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) == (r, c) or cells[rr, cc] == bg:
                continue
            values.append((rr, cc, int(cells[rr, cc])))
    return values


def _template(cells: np.ndarray) -> tuple[int, tuple[int, int], list[tuple[int, int, int]]]:
    bg = int(get_bg(cells))
    candidates: list[tuple[int, tuple[int, int], list[tuple[int, int, int]]]] = []

    for color in colors_present(cells, exclude={bg}):
        locations = [tuple(map(int, rc)) for rc in np.argwhere(cells == color)]
        decorated: list[tuple[int, int, list[tuple[int, int, int]]]] = []
        isolated = 0
        for r, c in locations:
            neighbors = _neighbor_values(cells, bg, r, c)
            if neighbors:
                decorated.append((r, c, neighbors))
            else:
                isolated += 1

        if len(decorated) == 1 and isolated >= 1:
            r, c, neighbors = decorated[0]
            offsets = [(rr - r, cc - c, value) for rr, cc, value in neighbors]
            assert offsets, "decorated marker must have a nonempty template"
            assert all(value != color for _dr, _dc, value in offsets)
            candidates.append((int(color), (r, c), offsets))

    assert len(candidates) == 1, "expected one marker color with one decorated exemplar"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the colored neighborhood around the one decorated marker to every matching marker in the lattice.

    Input structure
    - A square-cell lattice is separated by full horizontal and vertical separator lines.
    - Each lattice cell is uniformly colored; most cells are background.
    - One marker cell is surrounded by a colored 3x3-neighborhood template, and the same marker color appears elsewhere as isolated cells.

    Rule
    - Compress the lattice to one value per cell.
    - Use the decorated marker's non-background neighbors as relative colored offsets.
    - Stamp those offsets around every cell of the marker color, clipping offsets that fall outside the lattice.

    Key insight
    The separator grid is only scale: the real rule is copying a small colored mask from the exemplar marker to all same-color marker cells.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, cells = _compress_lattice(g)
    marker_color, _center, offsets = _template(cells)

    bg = int(get_bg(cells))
    painted = cells.copy()
    h, w = painted.shape
    for r, c in [tuple(map(int, rc)) for rc in np.argwhere(cells == marker_color)]:
        for dr, dc, color in offsets:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            assert painted[rr, cc] in (bg, color), "template stamps must not conflict"
            painted[rr, cc] = color

    out = g.copy()
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = painted[r, c]
    return out

# --------------- generator (o5x) ---------------

# Lattice size bounds and retry limit.
CELL_COUNT_RANGE = (5, 11)
CELL_SIZE_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 300

# Palette roles.
BACKGROUND_COLOR = 0
COLOR_POOL = np.arange(1, 10)
PALETTE_SIZE = 4
SEPARATOR_COLOR_INDEX = 0
MARKER_COLOR_INDEX = 1
FILL_COLOR_START = 2

# Template shape, isolated marker counts, and neighborhood rules.
RING_PROBABILITY = 0.5
RING_TWO_FILL_PROBABILITY = 0.5
ISOLATED_MARKER_RANGE = (1, 4)
ISOLATION_RADIUS = 1
CARDINAL_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _gen_template_offsets(shape, fill_colors):
    offsets = CARDINAL_OFFSETS if shape == "plus" else CARDINAL_OFFSETS + DIAGONAL_OFFSETS
    if shape == "plus" or len(fill_colors) == 1:
        return [(dr, dc, fill_colors[0]) for dr, dc in offsets]
    return [(dr, dc, fill_colors[0]) for dr, dc in CARDINAL_OFFSETS] + [
        (dr, dc, fill_colors[1]) for dr, dc in DIAGONAL_OFFSETS
    ]


def _gen_stamp_targets(center, offsets, n):
    r, c = center
    targets = [(r + dr, c + dc, color) for dr, dc, color in offsets]
    return [(rr, cc, color) for rr, cc, color in targets if 0 <= rr < n and 0 <= cc < n]


def _gen_isolated_candidates(n, input_nonbg, stamp_cells, offsets):
    blocked = input_nonbg | stamp_cells
    candidates = []
    for r in range(n):
        for c in range(n):
            if (r, c) in blocked:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= ISOLATION_RADIUS for rr, cc in input_nonbg):
                continue
            if any((rr, cc) in blocked for rr, cc, _ in _gen_stamp_targets((r, c), offsets, n)):
                continue
            candidates.append((r, c))
    return candidates


def _gen_expand_lattice(cells: np.ndarray, cell_size: int, separator_color: int) -> np.ndarray:
    n = cells.shape[0]
    side = n * cell_size + (n - 1)
    grid = np.full((side, side), separator_color, dtype=int)
    for r in range(n):
        r0 = r * (cell_size + 1)
        for c in range(n):
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[r, c]
    return grid


def _gen_place_exemplar(n, marker_color, offsets):
    output_cells = np.zeros((n, n), dtype=int)
    exemplar = (int(np.random.randint(1, n - 1)), int(np.random.randint(1, n - 1)))
    output_cells[exemplar] = marker_color

    input_nonbg = {exemplar}
    stamp_cells = set()
    for rr, cc, color in _gen_stamp_targets(exemplar, offsets, n):
        output_cells[rr, cc] = color
        input_nonbg.add((rr, cc))
        stamp_cells.add((rr, cc))
    return output_cells, input_nonbg, stamp_cells


def _gen_place_isolated_markers(output_cells, marker_color, offsets, input_nonbg, stamp_cells):
    n = output_cells.shape[0]
    isolated_stamp_cells = []
    for _ in range(int(np.random.randint(*ISOLATED_MARKER_RANGE))):
        candidates = _gen_isolated_candidates(n, input_nonbg, stamp_cells, offsets)
        if not candidates:
            return None
        marker = candidates[int(np.random.randint(0, len(candidates)))]
        output_cells[marker] = marker_color
        input_nonbg.add(marker)

        for rr, cc, color in _gen_stamp_targets(marker, offsets, n):
            if (rr, cc) in input_nonbg or (rr, cc) in stamp_cells:
                return None
            output_cells[rr, cc] = color
            stamp_cells.add((rr, cc))
            isolated_stamp_cells.append((rr, cc))
    return isolated_stamp_cells


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*CELL_COUNT_RANGE))
    cell_size = int(np.random.randint(*CELL_SIZE_RANGE))
    colors = np.random.choice(COLOR_POOL, size=PALETTE_SIZE, replace=False).astype(int).tolist()
    separator_color = colors[SEPARATOR_COLOR_INDEX]
    marker_color = colors[MARKER_COLOR_INDEX]
    shape = "ring" if np.random.random() < RING_PROBABILITY else "plus"
    fill_colors = (
        colors[FILL_COLOR_START:PALETTE_SIZE]
        if shape == "ring" and np.random.random() < RING_TWO_FILL_PROBABILITY
        else colors[FILL_COLOR_START : FILL_COLOR_START + 1]
    )
    offsets = _gen_template_offsets(shape, fill_colors)

    output_cells, input_nonbg, stamp_cells = _gen_place_exemplar(n, marker_color, offsets)
    isolated_stamp_cells = _gen_place_isolated_markers(output_cells, marker_color, offsets, input_nonbg, stamp_cells)
    if isolated_stamp_cells is None:
        return None

    input_cells = output_cells.copy()
    for rr, cc in isolated_stamp_cells:
        input_cells[rr, cc] = BACKGROUND_COLOR

    bg_count = int(np.count_nonzero(input_cells == BACKGROUND_COLOR))
    max_color_count = max(
        int(np.count_nonzero(input_cells == color)) for color in np.unique(input_cells) if color != BACKGROUND_COLOR
    )
    if bg_count <= max_color_count:
        return None

    inp = _gen_expand_lattice(input_cells, cell_size, separator_color)
    expected = _gen_expand_lattice(output_cells, cell_size, separator_color)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 39e1d7f9 input")
