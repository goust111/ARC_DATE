from __future__ import annotations

import random

import numpy as np


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _uniform_line_groups(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    line_count = int(grid.shape[axis])
    for idx in range(line_count):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            groups.setdefault(color, []).append(int(idx))
    return groups


def _segments(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    cuts = sorted(int(c) for c in cuts)
    assert cuts
    assert all(0 < c < size - 1 for c in cuts)
    assert len(set(cuts)) == len(cuts)

    segments: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        segments.append((int(start), int(cut)))
        start = int(cut + 1)
    segments.append((int(start), int(size)))

    lengths = {int(end - begin) for begin, end in segments}
    assert len(lengths) == 1
    return segments


def _rotation_match(reference: np.ndarray, patch: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(patch, np.rot90(reference, k))]
    assert len(matches) == 1
    return int(matches[0])


def _partition_info(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    row_groups = _uniform_line_groups(grid, axis=0)
    col_groups = _uniform_line_groups(grid, axis=1)

    candidates = []
    for color in sorted(row_groups.keys() & col_groups.keys()):
        rows = sorted(row_groups[color])
        cols = sorted(col_groups[color])
        if len(rows) == len(cols) and len(rows) in (2, 3):
            candidates.append((int(color), rows, cols))

    assert len(candidates) == 1
    sep_color, sep_rows, sep_cols = candidates[0]

    row_segments = _segments(int(grid.shape[0]), sep_rows)
    col_segments = _segments(int(grid.shape[1]), sep_cols)
    assert len(row_segments) == len(col_segments) in (3, 4)

    mask = np.ones(grid.shape, dtype=bool)
    mask[sep_rows, :] = False
    mask[:, sep_cols] = False
    bg = _most_common_color(grid[mask])
    return int(sep_color), int(bg), row_segments, col_segments


def _extract_cells(grid: np.ndarray, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> list[list[np.ndarray]]:
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_segments:
        row: list[np.ndarray] = []
        for c0, c1 in col_segments:
            row.append(np.array(grid[int(r0) : int(r1), int(c0) : int(c1)], dtype=int))
        cells.append(row)
    return cells


def _infer_blank_fill(cells: list[list[np.ndarray]], bg: int) -> tuple[tuple[int, int], np.ndarray]:
    reference: np.ndarray | None = None
    known: dict[tuple[int, int], int] = {}
    blank: tuple[int, int] | None = None

    for r, row in enumerate(cells):
        for c, patch in enumerate(row):
            if np.all(patch == int(bg)):
                assert blank is None
                blank = (int(r), int(c))
                continue

            if reference is None:
                reference = patch.copy()
                known[(int(r), int(c))] = 0
            else:
                known[(int(r), int(c))] = _rotation_match(reference, patch)

    assert reference is not None
    assert blank is not None

    predictions = set()
    for base in range(4):
        for row_step in range(4):
            for col_step in range(4):
                if all((base + r * row_step + c * col_step) % 4 == rot for (r, c), rot in known.items()):
                    br, bc = blank
                    predictions.add(int((base + br * row_step + bc * col_step) % 4))

    assert len(predictions) == 1
    fill_patch = np.rot90(reference, int(next(iter(predictions))))
    return blank, fill_patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique blank partition by inferring the 2D rotation rule shared by all other partitions.

    Input structure:
    - Complete separator rows and columns split the board into a 3x3 or 4x4 array of equal cells.
    - Every non-empty cell contains the same local motif, but rotated by 0/90/180/270 degrees.
    - One target cell is completely blank.

    Rule:
    - Treat each occupied cell as a rotated copy of one reference motif.
    - Infer an affine rotation pattern over cell coordinates:
      orientation(row, col) = base + row * row_step + col * col_step (mod 4).
    - Fill the blank cell with the unique rotation consistent with all observed cells.
    """
    g = np.asarray(grid, dtype=int)
    _sep_color, bg, row_segments, col_segments = _partition_info(g)
    cells = _extract_cells(g, row_segments, col_segments)
    (blank_r, blank_c), fill_patch = _infer_blank_fill(cells, bg)

    out = g.copy()
    r0, r1 = row_segments[int(blank_r)]
    c0, c1 = col_segments[int(blank_c)]
    out[int(r0) : int(r1), int(c0) : int(c1)] = fill_patch
    return out


_PATTERN_LIBRARY = (
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 2, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 2, 0],
            [0, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 2, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    ),
)

for _pattern in _PATTERN_LIBRARY:
    assert len({np.rot90(_pattern, k).tobytes() for k in range(4)}) == 4


def _render_pattern(pattern: np.ndarray, primary: int, accent: int) -> np.ndarray:
    rendered = np.zeros_like(pattern)
    rendered[pattern == 1] = int(primary)
    rendered[pattern == 2] = int(accent)
    return rendered


def _random_base_cell(pattern: np.ndarray, cell_size: int, bg: int, sep: int) -> np.ndarray:
    pattern = np.array(pattern, dtype=int)
    if random.random() < 0.5:
        pattern = np.flipud(pattern)
    if random.random() < 0.5:
        pattern = np.fliplr(pattern)

    pattern_size = int(pattern.shape[0])
    assert cell_size >= pattern_size + 2

    palette = [c for c in range(10) if c not in (int(bg), int(sep))]
    primary, accent_a, accent_b = random.sample(palette, 3)
    accent = accent_a if random.random() < 0.5 else accent_b
    if random.random() < 0.5:
        accent = primary

    top = random.randint(1, cell_size - pattern_size - 1)
    left = random.randint(1, cell_size - pattern_size - 1)

    cell = np.full((cell_size, cell_size), int(bg), dtype=int)
    rendered = _render_pattern(pattern, primary, accent)
    patch = cell[top : top + pattern_size, left : left + pattern_size]
    patch[rendered != 0] = rendered[rendered != 0]
    return cell


def _cell_segments(n_cells: int, cell_size: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for _ in range(n_cells):
        end = start + cell_size
        segments.append((int(start), int(end)))
        start = end + 1
    return segments


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_cells = random.choice((3, 4))
    base_turn = random.randrange(4)
    row_step, col_step = random.choice([(r, c) for r in range(4) for c in range(4) if (r, c) != (0, 0)])

    pattern = random.choice(_PATTERN_LIBRARY)
    min_cell = int(pattern.shape[0]) + 2
    cell_size = random.randint(min_cell, min(8, min_cell + 2))

    bg, sep = random.sample(range(10), 2)
    base_cell = _random_base_cell(pattern, cell_size, bg, sep)

    orientations = np.fromfunction(
        lambda r, c: (base_turn + r * row_step + c * col_step) % 4,
        (n_cells, n_cells),
        dtype=int,
    ).astype(int)
    if len(set(int(v) for v in orientations.ravel())) < 2:
        return None

    blank = (random.randrange(n_cells), random.randrange(n_cells))

    side = n_cells * cell_size + (n_cells - 1)
    output = np.full((side, side), int(bg), dtype=int)
    divider_positions = [cell_size + i * (cell_size + 1) for i in range(n_cells - 1)]
    output[divider_positions, :] = int(sep)
    output[:, divider_positions] = int(sep)

    segments = _cell_segments(n_cells, cell_size)
    for r in range(n_cells):
        for c in range(n_cells):
            r0, r1 = segments[r]
            c0, c1 = segments[c]
            output[r0:r1, c0:c1] = np.rot90(base_cell, int(orientations[r, c]))

    inp = output.copy()
    br0, br1 = segments[blank[0]]
    bc0, bc1 = segments[blank[1]]
    inp[br0:br1, bc0:bc1] = int(bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_case()
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate a valid rotation-grid puzzle")
