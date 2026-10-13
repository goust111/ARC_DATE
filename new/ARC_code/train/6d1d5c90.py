import numpy as np

from arc_utils import Color
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the left control column and roll the remaining image downward by the red marker's row.

    Input structure
    - The input is a rectangular grid with a one-cell-wide control column on the left.
    - That control column is all pink except for exactly one red marker pixel.
    - The columns to the right are the image that remains in the output.

    Rule
    - Delete the left control column.
    - Let r be the row index of the red marker in that column.
    - Cyclically shift the remaining image downward by r rows.

    Key insight
    - The red marker's vertical position encodes the row-roll amount; the pink
      column is only a control strip and is removed after reading it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 1

    control = g[:, 0]
    marker_rows = np.where(control == Color.RED)[0]
    assert marker_rows.size == 1
    assert np.all((control == Color.PINK) | (control == Color.RED))
    assert np.count_nonzero(control == Color.PINK) == h - 1

    shift = int(marker_rows[0])
    return np.roll(g[:, 1:], shift=shift, axis=0)

# --------------- generator (att2) ---------------

# Canvas/image bounds.
IMAGE_HEIGHT_RANGE = (5, 8)
IMAGE_WIDTH_RANGE = (5, 8)
MAX_IMAGE_CELLS = 54
# Color balancing.
COLOR_PALETTE = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.TEAL, Color.MAROON]
COLOR_COUNT = 6
CELLS_PER_COLOR_RANGE = (4, 9)
# Layout sampling.
PATH_MODES = ["row_snake", "col_snake", "row_runs", "col_runs", "diagonal"]
ORIENTATION_CHOICES = [False, True]
# Control strip and retry behavior.
CONTROL_FILL_COLOR = Color.PINK
CONTROL_MARKER_COLOR = Color.RED
GENERATION_RETRIES = 300


def _bounded_counts(total: int) -> list[int]:
    counts = []
    remaining = total
    lo, hi = CELLS_PER_COLOR_RANGE
    for i in range(COLOR_COUNT):
        left = COLOR_COUNT - i - 1
        low = max(lo, remaining - left * hi)
        high = min(hi, remaining - left * lo)
        value = random.randint(low, high)
        counts.append(value)
        remaining -= value
    random.shuffle(counts)
    return counts


def _cell_path(h: int, w: int) -> list[tuple[int, int]]:
    mode = random.choice(PATH_MODES)
    if mode == "row_snake":
        path = [(r, c) for r in range(h) for c in (range(w) if r % 2 == 0 else range(w - 1, -1, -1))]
    elif mode == "col_snake":
        path = [(r, c) for c in range(w) for r in (range(h) if c % 2 == 0 else range(h - 1, -1, -1))]
    elif mode == "row_runs":
        path = [(r, c) for r in range(h) for c in range(w)]
    elif mode == "col_runs":
        path = [(r, c) for c in range(w) for r in range(h)]
    else:
        path = []
        for diag in range(h + w - 1):
            cells = [(r, diag - r) for r in range(h) if 0 <= diag - r < w]
            if diag % 2:
                cells.reverse()
            path.extend(cells)

    if random.choice(ORIENTATION_CHOICES):
        path = [(h - 1 - r, c) for r, c in path]
    if random.choice(ORIENTATION_CHOICES):
        path = [(r, w - 1 - c) for r, c in path]
    if random.choice(ORIENTATION_CHOICES):
        path.reverse()
    return path


def _make_image(h: int, w: int) -> np.ndarray:
    counts = _bounded_counts(h * w)
    colors = COLOR_PALETTE[:]
    random.shuffle(colors)
    image = np.empty((h, w), dtype=int)
    path = _cell_path(h, w)
    idx = 0
    for color, count in zip(colors, counts):
        for r, c in path[idx : idx + count]:
            image[r, c] = color
        idx += count
    return image


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*IMAGE_HEIGHT_RANGE)
    iw = random.choice([w for w in range(IMAGE_WIDTH_RANGE[0], IMAGE_WIDTH_RANGE[1] + 1) if h * w <= MAX_IMAGE_CELLS])
    image = _make_image(h, iw)

    shift = random.randrange(h)
    grid = np.empty((h, iw + 1), dtype=int)
    grid[:, 0] = CONTROL_FILL_COLOR
    grid[shift, 0] = CONTROL_MARKER_COLOR
    grid[:, 1:] = image

    expected = np.roll(image, shift=shift, axis=0)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
