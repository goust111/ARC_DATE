from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BG = 0
GRID = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 300


@dataclass(frozen=True)
class CellSpec:
    row: int
    col: int
    color: int
    pos: tuple[int, int]


def _line_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [int(i) for i in np.where(np.all(grid == GRID, axis=1))[0]]
    return [int(i) for i in np.where(np.all(grid == GRID, axis=0))[0]]


def _spans_from_lines(size: int, lines: list[int]) -> list[tuple[int, int]]:
    assert lines, "expected at least one grey separator line"
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:])), "separator lines must be 1 cell wide"

    spans: list[tuple[int, int]] = []
    prev = -1
    for line in [*lines, int(size)]:
        start = prev + 1
        stop = int(line)
        assert start < stop, "empty cell block is not allowed"
        spans.append((start, stop))
        prev = int(line)

    lengths = {stop - start for start, stop in spans}
    assert len(lengths) == 1, "cell blocks must have a uniform size"

    if len(lines) >= 2:
        pitch = lines[1] - lines[0]
        assert all((b - a) == pitch for a, b in zip(lines, lines[1:])), "separator spacing must be regular"
        only_len = next(iter(lengths))
        assert pitch == only_len + 1, "separator spacing must match block size plus one grid line"

    return spans


def _partition(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _spans_from_lines(grid.shape[0], _line_indices(grid, axis=0))
    col_spans = _spans_from_lines(grid.shape[1], _line_indices(grid, axis=1))
    assert len(row_spans) >= 1 and len(col_spans) >= 2
    assert len(row_spans) * len(col_spans) > 2, "need template cells plus data cells"
    return row_spans, col_spans


def _cell(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return grid[r0:r1, c0:c1]


def _non_bg_non_grid_mask(cell: np.ndarray) -> np.ndarray:
    return (cell != BG) & (cell != GRID)


def _extract_template(cell: np.ndarray) -> tuple[np.ndarray, int]:
    mask = _non_bg_non_grid_mask(cell)
    coords = np.argwhere(mask)
    assert coords.size > 0, "template cell must contain a visible pattern"
    colors = {int(cell[r, c]) for r, c in coords}
    assert len(colors) == 1, "template cell must be monochrome"
    return coords.astype(int), next(iter(colors))


def _extract_marker(cell: np.ndarray) -> tuple[tuple[int, int], int]:
    coords = np.argwhere(_non_bg_non_grid_mask(cell))
    assert coords.shape[0] == 1, "data cell must contain exactly one marker"
    r, c = map(int, coords[0])
    return (r, c), int(cell[r, c])


def _paint_template(shape: tuple[int, int], coords: np.ndarray, color: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one of two template stamps into each grey-grid cell according to that cell's marker color.

    Input structure:
    - A black canvas is partitioned by 1-pixel-wide grey (5) full-row/full-column grid lines into
      equal-sized blocks.
    - The first two blocks of the top row are template blocks; each contains one monochrome pattern
      on black background and together they define a color-to-pattern dictionary.
    - Every remaining block contains exactly one non-black, non-grey marker pixel whose color matches
      one of the two template colors.

    Rule:
    - Detect the regular grid-line partition and read the first two top-row blocks as template A and
      template B.
    - For every non-template block, inspect its unique marker color.
    - If the marker color equals template A's color, replace that whole block by a black block stamped
      with template A at the same relative coordinates; otherwise stamp template B.
    - Keep grey grid lines and both template blocks unchanged.

    Key insight:
    - The grey lattice provides the block coordinate system, while the top-left two blocks act as a
      local legend that maps marker color to the stamp shape that must be copied into every other block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    row_spans, col_spans = _partition(g)
    template_a, color_a = _extract_template(_cell(g, row_spans[0], col_spans[0]))
    template_b, color_b = _extract_template(_cell(g, row_spans[0], col_spans[1]))
    assert color_a != color_b, "template colors must be distinct"

    out = g.copy()
    for rr, row_span in enumerate(row_spans):
        for cc, col_span in enumerate(col_spans):
            if rr == 0 and cc in (0, 1):
                continue

            block = _cell(g, row_span, col_span)
            _marker_pos, marker_color = _extract_marker(block)
            if marker_color == color_a:
                repl = _paint_template(block.shape, template_a, color_a)
            else:
                assert marker_color == color_b, "marker color must match one of the template colors"
                repl = _paint_template(block.shape, template_b, color_b)
            r0, r1 = row_span
            c0, c1 = col_span
            out[r0:r1, c0:c1] = repl

    return out


SHAPE_LIBRARY = [
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
    np.array([(0, 1), (0, 2), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
]


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    return arr - mins


def _transform_shape(coords: np.ndarray, rotations: int, flip: bool) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    if flip:
        arr[:, 1] *= -1
    for _ in range(int(rotations) % 4):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    return _normalize(arr)


def _random_shape(rng: np.random.Generator, cell_size: int) -> np.ndarray:
    for _ in range(50):
        base = SHAPE_LIBRARY[int(rng.integers(0, len(SHAPE_LIBRARY)))]
        coords = _transform_shape(
            base,
            rotations=int(rng.integers(0, 4)),
            flip=bool(rng.integers(0, 2)),
        )
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h > cell_size or w > cell_size:
            continue
        off_r = int(rng.integers(0, cell_size - h + 1))
        off_c = int(rng.integers(0, cell_size - w + 1))
        return coords + np.array([off_r, off_c], dtype=int)
    raise RuntimeError("failed to sample a template shape")


def _make_layout(rng: np.random.Generator) -> tuple[int, int, int]:
    cell_size = int(rng.integers(3, 6))
    max_cells = min(5, 31 // (cell_size + 1))
    n_rows = int(rng.integers(3, max_cells + 1))
    n_cols = int(rng.integers(3, max_cells + 1))
    return cell_size, n_rows, n_cols


def _spans_for_layout(cell_size: int, count: int) -> list[tuple[int, int]]:
    spans = []
    for idx in range(count):
        start = idx * (cell_size + 1)
        spans.append((start, start + cell_size))
    return spans


def _draw_grid(cell_size: int, n_rows: int, n_cols: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = n_rows * cell_size + (n_rows - 1)
    w = n_cols * cell_size + (n_cols - 1)
    grid = np.full((h, w), BG, dtype=int)
    row_spans = _spans_for_layout(cell_size, n_rows)
    col_spans = _spans_for_layout(cell_size, n_cols)

    for idx in range(n_rows - 1):
        line = row_spans[idx][1]
        grid[line, :] = GRID
    for idx in range(n_cols - 1):
        line = col_spans[idx][1]
        grid[:, line] = GRID
    return grid, row_spans, col_spans


def _cell_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _choose_template_pair(rng: np.random.Generator, cell_size: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        a = _random_shape(rng, cell_size)
        b = _random_shape(rng, cell_size)
        if _cell_set(a) != _cell_set(b):
            return a, b
    raise RuntimeError("failed to sample distinct templates")


def _place_template(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], coords: np.ndarray, color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    block = np.full((r1 - r0, c1 - c0), BG, dtype=int)
    block[coords[:, 0], coords[:, 1]] = int(color)
    grid[r0:r1, c0:c1] = block


def _random_marker_position(
    rng: np.random.Generator,
    cell_size: int,
    preferred: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    if preferred:
        idx = int(rng.integers(0, len(preferred)))
        return tuple(map(int, preferred[idx]))
    return int(rng.integers(0, cell_size)), int(rng.integers(0, cell_size))


def _build_expected(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    template_a: np.ndarray,
    template_b: np.ndarray,
    color_a: int,
    color_b: int,
    specs: list[CellSpec],
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        row_span = row_spans[spec.row]
        col_span = col_spans[spec.col]
        target = np.full((row_span[1] - row_span[0], col_span[1] - col_span[0]), BG, dtype=int)
        if spec.color == color_a:
            target[template_a[:, 0], template_a[:, 1]] = color_a
        else:
            assert spec.color == color_b
            target[template_b[:, 0], template_b[:, 1]] = color_b
        out[row_span[0] : row_span[1], col_span[0] : col_span[1]] = target
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    template_a: np.ndarray,
    template_b: np.ndarray,
    color_a: int,
    color_b: int,
    specs: list[CellSpec],
) -> bool:
    try:
        parsed_rows, parsed_cols = _partition(inp)
        if parsed_rows != row_spans or parsed_cols != col_spans:
            return False

        temp_a_cell = _cell(inp, row_spans[0], col_spans[0])
        temp_b_cell = _cell(inp, row_spans[0], col_spans[1])
        read_a, read_color_a = _extract_template(temp_a_cell)
        read_b, read_color_b = _extract_template(temp_b_cell)
        if read_color_a != color_a or read_color_b != color_b:
            return False
        if _cell_set(read_a) != _cell_set(template_a) or _cell_set(read_b) != _cell_set(template_b):
            return False

        seen_colors = {spec.color for spec in specs}
        if seen_colors != {color_a, color_b}:
            return False

        a_only = _cell_set(template_a) - _cell_set(template_b)
        b_only = _cell_set(template_b) - _cell_set(template_a)
        if not a_only or not b_only:
            return False

        saw_cross_a = False
        saw_cross_b = False
        for spec in specs:
            block_in = _cell(inp, row_spans[spec.row], col_spans[spec.col])
            block_out = _cell(out, row_spans[spec.row], col_spans[spec.col])
            pos, marker_color = _extract_marker(block_in)
            if marker_color != spec.color or pos != spec.pos:
                return False

            if spec.color == color_a:
                want = _paint_template(block_out.shape, template_a, color_a)
                saw_cross_a |= spec.pos in b_only
            else:
                want = _paint_template(block_out.shape, template_b, color_b)
                saw_cross_b |= spec.pos in a_only
            if not np.array_equal(block_out, want):
                return False

        if not (saw_cross_a and saw_cross_b):
            return False
        if np.array_equal(inp, out):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_size, n_rows, n_cols = _make_layout(rng)
    inp, row_spans, col_spans = _draw_grid(cell_size, n_rows, n_cols)

    color_a, color_b = map(int, rng.choice(PALETTE, size=2, replace=False))
    template_a, template_b = _choose_template_pair(rng, cell_size)
    _place_template(inp, row_spans[0], col_spans[0], template_a, color_a)
    _place_template(inp, row_spans[0], col_spans[1], template_b, color_b)

    a_only = list(_cell_set(template_a) - _cell_set(template_b))
    b_only = list(_cell_set(template_b) - _cell_set(template_a))
    if not a_only or not b_only:
        return None

    target_cells = [(r, c) for r in range(n_rows) for c in range(n_cols) if not (r == 0 and c in (0, 1))]
    rng.shuffle(target_cells)
    if len(target_cells) < 2:
        return None

    specs: list[CellSpec] = []
    forced = [
        CellSpec(target_cells[0][0], target_cells[0][1], color_a, _random_marker_position(rng, cell_size, b_only)),
        CellSpec(target_cells[1][0], target_cells[1][1], color_b, _random_marker_position(rng, cell_size, a_only)),
    ]
    specs.extend(forced)

    for row, col in target_cells[2:]:
        color = color_a if int(rng.integers(0, 2)) == 0 else color_b
        if color == color_a:
            bucket = [
                list(b_only),
                list(a_only),
                list(_cell_set(template_a) & _cell_set(template_b)),
                None,
            ]
        else:
            bucket = [
                list(a_only),
                list(b_only),
                list(_cell_set(template_a) & _cell_set(template_b)),
                None,
            ]
        preferred = bucket[int(rng.integers(0, len(bucket)))]
        pos = _random_marker_position(rng, cell_size, preferred)
        specs.append(CellSpec(int(row), int(col), int(color), pos))

    for spec in specs:
        r0, r1 = row_spans[spec.row]
        c0, c1 = col_spans[spec.col]
        block = np.full((r1 - r0, c1 - c0), BG, dtype=int)
        block[spec.pos[0], spec.pos[1]] = spec.color
        inp[r0:r1, c0:c1] = block

    expected = _build_expected(inp, row_spans, col_spans, template_a, template_b, color_a, color_b, specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, actual, row_spans, col_spans, template_a, template_b, color_a, color_b, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
