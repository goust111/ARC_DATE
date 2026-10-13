import numpy as np

from arc_utils import get_bg


LEFT_MARKER_COL = 1


def _glyph_cells(color: int, width: int) -> set[tuple[int, int]]:
    left = LEFT_MARKER_COL
    right = width - 2
    assert right - left + 1 == 18

    cells: set[tuple[int, int]] = set()

    if color == 1:
        cells.update((0, c) for c in range(left, right + 1))
        for dr in (-1, 1):
            cells.update({(dr, right - 4), (dr, right - 2)})
    elif color == 2:
        cells.update((0, c) for c in range(left, right + 1))
        for dr in (-2, 2):
            cells.update({(dr, left + 5), (dr, left + 7)})
        for dr in (-1, 1):
            cells.add((dr, left + 6))
    elif color == 3:
        cells.update((0, c) for c in range(left, left + 3))
        cells.update((0, c) for c in range(left + 6, right + 1))
        for dr in (-2, 2):
            cells.add((dr, left + 4))
        for dr in (-1, 1):
            cells.update({(dr, left + 3), (dr, left + 5)})
    elif color == 6:
        cells.update((0, c) for c in range(left, right + 1) if c != left + 2)
        for dr in (-2, 3):
            cells.update((dr, c) for c in (left + 1, left + 2, left + 3))
        for dr in (-1, 1, 2):
            cells.update({(dr, left + 1), (dr, left + 3)})
    elif color == 7:
        cells.update((0, c) for c in range(left, right + 1))
        for dr in (-2, -1, 1, 2):
            cells.update({(dr, right - 5), (dr, right - 3)})
    else:
        raise AssertionError(f"unknown glyph color: {color}")

    return cells


def _marker_rows(coords: np.ndarray, width: int) -> list[int] | None:
    by_row: dict[int, list[int]] = {}
    for r0, c0 in coords:
        r, c = int(r0), int(c0)
        by_row.setdefault(r, []).append(c)

    right = width - 2
    if all(sorted(cols) == [LEFT_MARKER_COL, right] for cols in by_row.values()):
        return sorted(by_row)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the complete interior glyph and expand each endpoint marker pair into its color's full glyph.

    Input structure
    - The grid is 20x20 with background inside two same-color full-height vertical frame bars.
    - Inside the frame there is one complete color glyph and one or more other colors appearing only as two endpoint markers in the same row.
    - Marker endpoints are always in the first and last interior columns.

    Rule
    - Keep the vertical frame unchanged.
    - Erase any complete interior glyph.
    - For each endpoint marker pair, use that marker color's learned glyph template and anchor its horizontal bar on the marker row.

    Key insight
    - Color identifies which glyph shape to draw; the two endpoint pixels are not copied objects, they are anchors for reconstructing that color's full symbol.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert (height, width) == (20, 20)

    bg = get_bg(g)
    assert bg == 0
    frame_color = int(g[0, 0])
    assert frame_color != bg
    assert np.all(g[:, 0] == frame_color)
    assert np.all(g[:, -1] == frame_color)
    assert np.count_nonzero(g == frame_color) == 2 * height

    out = g.copy()
    marker_jobs: list[tuple[int, int]] = []
    erased_any = False

    for color0 in np.unique(g):
        color = int(color0)
        if color in (bg, frame_color):
            continue

        coords = np.argwhere(g == color)
        rows = _marker_rows(coords, width)
        if rows is not None:
            for row in rows:
                marker_jobs.append((color, row))
        else:
            assert len(coords) > 2
            out[g == color] = bg
            erased_any = True

    assert erased_any
    assert marker_jobs

    claimed: dict[tuple[int, int], int] = {}
    for color, anchor_row in marker_jobs:
        for dr, c in _glyph_cells(color, width):
            r = anchor_row + dr
            assert 0 <= r < height
            assert LEFT_MARKER_COL <= c <= width - 2
            assert (r, c) not in claimed or claimed[(r, c)] == color
            assert out[r, c] in (bg, color)
            claimed[(r, c)] = color

    for (r, c), color in claimed.items():
        out[r, c] = color

    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
_GEN_SIZE = 20
_GEN_GLYPH_COLORS = [1, 2, 3, 6, 7]
_GEN_FRAME_COLORS = (1, 2, 5, 7, 9)
# Object counts and retry budget.
_GEN_MARKER_COUNT_RANGE = (1, 3)
_GEN_RETRIES = 300


def _gen_abs_cells(color: int, anchor_row: int) -> set[tuple[int, int]]:
    return {(anchor_row + dr, c) for dr, c in _glyph_cells(color, _GEN_SIZE)}


def _gen_valid_rows(color: int) -> list[int]:
    drs = [dr for dr, _ in _glyph_cells(color, _GEN_SIZE)]
    return list(range(-min(drs), _GEN_SIZE - max(drs)))


def _gen_framed_grid(frame_color: int) -> np.ndarray:
    grid = np.zeros((_GEN_SIZE, _GEN_SIZE), dtype=int)
    grid[:, 0] = frame_color
    grid[:, -1] = frame_color
    return grid


def _gen_render_glyph(grid: np.ndarray, color: int, anchor_row: int) -> None:
    for r, c in _gen_abs_cells(color, anchor_row):
        grid[r, c] = color


def _gen_marker_jobs(colors: list[int], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    row_options = [_gen_valid_rows(color) for color in colors]
    choices: list[tuple[tuple[int, ...], list[tuple[int, int]]]] = []

    if len(colors) == 1:
        for row in row_options[0]:
            choices.append(((row,), [(colors[0], row)]))
    else:
        for row0 in row_options[0]:
            cells0 = _gen_abs_cells(colors[0], row0)
            for row1 in row_options[1]:
                if cells0.isdisjoint(_gen_abs_cells(colors[1], row1)):
                    choices.append(((row0, row1), [(colors[0], row0), (colors[1], row1)]))

    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))][1]


def _gen_example_rows(example_color: int, marker_jobs: list[tuple[int, int]]) -> list[int]:
    marker_points = {(row, 1) for _, row in marker_jobs} | {
        (row, _GEN_SIZE - 2) for _, row in marker_jobs
    }
    return [
        row
        for row in _gen_valid_rows(example_color)
        if _gen_abs_cells(example_color, row).isdisjoint(marker_points)
    ]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    marker_count = int(rng.integers(*_GEN_MARKER_COUNT_RANGE))
    active = list(rng.choice(_GEN_GLYPH_COLORS, size=marker_count + 1, replace=False))
    marker_colors = active[:marker_count]
    example_color = active[-1]
    frame_choices = [c for c in _GEN_FRAME_COLORS if c not in active]
    frame_color = int(rng.choice(frame_choices))

    marker_jobs = _gen_marker_jobs(marker_colors, rng)
    if marker_jobs is None:
        return None

    out = _gen_framed_grid(frame_color)
    for color, row in marker_jobs:
        _gen_render_glyph(out, color, row)

    example_rows = _gen_example_rows(example_color, marker_jobs)
    if not example_rows:
        return None
    example_row = int(rng.choice(example_rows))

    inp = _gen_framed_grid(frame_color)
    _gen_render_glyph(inp, example_color, example_row)
    for color, row in marker_jobs:
        inp[row, 1] = color
        inp[row, _GEN_SIZE - 2] = color

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
