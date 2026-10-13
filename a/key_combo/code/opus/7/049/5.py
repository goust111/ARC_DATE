import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()

# Generator bounds.
REGION_HEIGHT_RANGE = (4, 7)  # sampled in [4, 6]
GRID_WIDTH_RANGE = (8, 13)    # sampled in [8, 12]
GREY_MASK_SIZE_RANGE = (5, 11)
COLOR_COUNT_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    return int(rows[0])


def _non_special_colors(grid: np.ndarray) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY))
    assert colors, "expected at least one non-black, non-grey color"
    return colors


def _winner_color(grid: np.ndarray, divider: int) -> int:
    colors = _non_special_colors(grid)
    deltas = []
    for color in colors:
        above = int(np.count_nonzero(grid[:divider] == color))
        below = int(np.count_nonzero(grid[divider + 1 :] == color))
        deltas.append((below - above, color))

    best_delta = max(delta for delta, _ in deltas)
    winners = [color for delta, color in deltas if delta == best_delta]
    assert len(winners) == 1, "expected a unique winning color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the upper grey mask downward and repaint it with the color that grows the most below the divider.

    Input structure:
    - A black-background grid is split by one full-width grey horizontal divider.
    - The upper half contains a 4-connected grey mask plus a few scattered colored pixels.
    - The lower half contains scattered colored pixels and no grey cells.

    Rule:
    - Count each non-black, non-grey color above and below the divider.
    - Choose the unique color with the largest value of below_count minus above_count.
    - Keep the entire upper half unchanged, keep the divider grey, erase the lower half,
      and paint the vertical mirror of the upper grey-mask cells in the winning color.

    Key insight:
    - Geometry and color come from separate cues: the upper grey shape decides where to
      paint, while the upper-vs-lower color population difference decides what color to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    divider = _find_divider_row(g)
    upper = g[:divider]
    lower = g[divider + 1 :]
    assert upper.size > 0 and lower.size > 0, "expected non-empty regions above and below divider"

    winner = _winner_color(g, divider)

    out = np.full((height, width), BLACK, dtype=int)
    out[:divider] = upper
    out[divider, :] = GREY

    grey_coords = np.argwhere(upper == GREY)
    assert grey_coords.size > 0, "expected at least one grey mask pixel above divider"
    for r, c in grey_coords:
        mr = 2 * divider - int(r)
        mc = int(c)
        assert divider < mr < height, "mirrored mask pixel must land in lower half"
        out[mr, mc] = winner

    return out


def _random_connected_mask(height: int, width: int, size: int) -> set[tuple[int, int]] | None:
    start = (int(RNG.integers(height)), int(RNG.integers(width)))
    cells = {start}

    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        frontier = list(dict.fromkeys(frontier))
        cells.add(frontier[int(RNG.integers(len(frontier)))])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return None

    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if bbox_area == len(cells):
        return None
    if any(sum(1 for rr, _ in cells if rr == r) == width for r in range(height)):
        return None
    return cells


def _sample_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert count <= len(cells)
    if count == 0:
        return []
    choice = RNG.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in choice]


def _build_expected_output(grid: np.ndarray, winner: int) -> np.ndarray:
    divider = _find_divider_row(grid)
    out = np.full_like(grid, BLACK)
    out[:divider] = grid[:divider]
    out[divider, :] = GREY
    for r, c in np.argwhere(grid[:divider] == GREY):
        out[2 * divider - int(r), int(c)] = int(winner)
    return out


def _gen_attempt() -> np.ndarray | None:
    region_h = int(RNG.integers(*REGION_HEIGHT_RANGE))
    width = int(RNG.integers(*GRID_WIDTH_RANGE))
    height = 2 * region_h + 1
    divider = region_h

    grey_size = int(RNG.integers(*GREY_MASK_SIZE_RANGE))
    if grey_size >= region_h * width:
        return None

    grey_cells = _random_connected_mask(region_h, width, grey_size)
    if grey_cells is None:
        return None

    mirror_cells = {(2 * divider - r, c) for r, c in grey_cells}
    if not all(divider < r < height for r, _ in mirror_cells):
        return None

    color_count = int(RNG.choice(COLOR_COUNT_CHOICES))
    colors = [int(c) for c in RNG.choice(PALETTE, size=color_count, replace=False)]
    winner_index = int(RNG.integers(color_count))
    winner_color = colors[winner_index]

    upper_counts = [int(RNG.integers(1, 5)) for _ in colors]
    deltas = []
    for idx, upper_count in enumerate(upper_counts):
        if idx == winner_index:
            deltas.append(int(RNG.integers(2, 6)))
        else:
            deltas.append(int(RNG.integers(-min(upper_count, 2), 1)))
    lower_counts = [u + d for u, d in zip(upper_counts, deltas)]

    if sum(1 for d in deltas if d > 0) != 1:
        return None
    if max(deltas) != deltas[winner_index]:
        return None
    if sum(lower > 0 for lower in lower_counts) < 2:
        return None
    if sum(upper_counts) < 3:
        return None

    upper_free = [(r, c) for r in range(region_h) for c in range(width) if (r, c) not in grey_cells]
    lower_free = [(r, c) for r in range(divider + 1, height) for c in range(width) if (r, c) not in mirror_cells]
    if sum(upper_counts) > len(upper_free) or sum(lower_counts) > len(lower_free):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[divider, :] = GREY
    for r, c in grey_cells:
        grid[r, c] = GREY

    chosen_upper = _sample_cells(upper_free, sum(upper_counts))
    chosen_lower = _sample_cells(lower_free, sum(lower_counts))

    cursor = 0
    for color, count in zip(colors, upper_counts):
        for r, c in chosen_upper[cursor : cursor + count]:
            grid[r, c] = color
        cursor += count

    cursor = 0
    for color, count in zip(colors, lower_counts):
        for r, c in chosen_lower[cursor : cursor + count]:
            grid[r, c] = color
        cursor += count

    if np.any(grid[divider + 1 :] == GREY):
        return None
    if np.count_nonzero(grid[:divider] == GREY) != grey_size:
        return None
    if any(np.all(grid[r] == GREY) for r in range(height) if r != divider):
        return None

    expected = _build_expected_output(grid, winner_color)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 5")
