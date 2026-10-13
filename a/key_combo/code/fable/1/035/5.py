import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = np.array([1, 2, 3, 6, 7, 8, 9], dtype=int)

MIN_SIZE = 8
MAX_SIZE = 18
ROW_COUNT_CHOICES = np.array([2, 3], dtype=int)
COL_COUNT_CHOICES = np.array([2, 3], dtype=int)
MIN_GAP = 2
MAX_GEN_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _compose_cell(horizontal_color: int | None, vertical_color: int | None, r: int, c: int) -> int:
    if horizontal_color is None and vertical_color is None:
        return BLACK

    horizontal_even = (c % 2) == 0
    vertical_even = (r % 2) == 0

    if horizontal_color is None:
        return int(vertical_color) if vertical_even else GREY
    if vertical_color is None:
        return int(horizontal_color) if horizontal_even else GREY
    if horizontal_even and vertical_even:
        return YELLOW
    if horizontal_even:
        return int(horizontal_color)
    if vertical_even:
        return int(vertical_color)
    return GREY


def _sample_positions(length: int, count: int) -> list[int]:
    values = list(range(1, length))
    evens = [value for value in values if value % 2 == 0]
    odds = [value for value in values if value % 2 == 1]

    for _ in range(100):
        chosen = [int(_RNG.choice(evens)), int(_RNG.choice(odds))]
        if abs(chosen[0] - chosen[1]) < MIN_GAP:
            continue

        while len(chosen) < count:
            candidates = [
                value
                for value in values
                if value not in chosen and all(abs(value - other) >= MIN_GAP for other in chosen)
            ]
            if not candidates:
                break
            chosen.append(int(_RNG.choice(candidates)))

        if len(chosen) == count:
            return sorted(chosen)

    raise RuntimeError("failed to sample edge positions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend left-edge and top-edge markers into alternating stripe rays whose overlaps are colored by phase interference.

    Input structure:
    - A black background contains only single-cell colored markers on the left edge and the top edge.
    - Left-edge markers sit at column 0 and launch horizontal rays to the right; top-edge markers sit at row 0 and launch vertical rays downward.
    - Marker colors are all distinct and never use yellow or grey; the top-left corner stays black.

    Rule:
    - Along any single ray, cells at even distance from that ray's marker keep the marker color and cells at odd distance become grey.
    - Where one horizontal ray and one vertical ray cross, combine their parities instead of overwriting: even/even becomes yellow, exactly one even keeps that even-phase ray's marker color, and odd/odd becomes grey.
    - Cells touched by no ray remain black.

    Key insight:
    - The output is the interference of two independent parity fields, so each crossing color depends on distance parity to both edge markers rather than drawing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must have at least two rows and two columns"
    assert int(g[0, 0]) == BLACK, "top-left corner must stay black to avoid edge ambiguity"
    assert np.all(g[1:, 1:] == BLACK), "only edge markers are allowed in the input"

    row_markers = {r: int(g[r, 0]) for r in range(1, height) if int(g[r, 0]) != BLACK}
    col_markers = {c: int(g[0, c]) for c in range(1, width) if int(g[0, c]) != BLACK}

    assert row_markers, "expected at least one left-edge marker"
    assert col_markers, "expected at least one top-edge marker"

    all_colors = list(row_markers.values()) + list(col_markers.values())
    assert len(all_colors) == len(set(all_colors)), "marker colors must be globally distinct"
    assert all(color not in (BLACK, YELLOW, GREY) for color in all_colors), "reserved colors cannot be markers"

    out = np.full_like(g, BLACK)
    for r in range(height):
        horizontal_color = row_markers.get(r)
        for c in range(width):
            vertical_color = col_markers.get(c)
            out[r, c] = _compose_cell(horizontal_color, vertical_color, r, c)

    return out


def _gen_attempt() -> np.ndarray:
    height = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))
    width = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))

    row_count = int(_RNG.choice(ROW_COUNT_CHOICES))
    col_count = int(_RNG.choice(COL_COUNT_CHOICES))

    rows = _sample_positions(height, row_count)
    cols = _sample_positions(width, col_count)

    colors = MARKER_COLORS.copy()
    _RNG.shuffle(colors)

    grid = np.full((height, width), BLACK, dtype=int)
    for idx, r in enumerate(rows):
        grid[r, 0] = int(colors[idx])
    for idx, c in enumerate(cols):
        grid[0, c] = int(colors[row_count + idx])

    output = main(grid)

    row_parities = {r % 2 for r in rows}
    col_parities = {c % 2 for c in cols}
    assert row_parities == {0, 1}, "generator must show both vertical phases"
    assert col_parities == {0, 1}, "generator must show both horizontal phases"
    assert np.any(output == YELLOW), "generator must expose even/even overlap"
    assert np.any(output == GREY), "generator must expose grey stripe cells"
    assert len(rows) < height - 1, "need inactive rows for vertical-only stripes"
    assert len(cols) < width - 1, "need inactive columns for horizontal-only stripes"

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-interference input")
