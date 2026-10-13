import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Shear the rectangular input down-left onto a black canvas.

    Input structure
    A rectangular grid of colored pixels with no internal empty region to infer.

    Rule
    Create a black output with the same height and width W + H - 1, then copy
    each input row r unchanged starting at column H - 1 - r.

    Key insight
    The colors and row contents are preserved; only the row offsets change so
    the left edge forms a descending diagonal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        start = h - 1 - r
        out[r, start : start + w] = g[r]
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette bounds.
MIN_HEIGHT, MAX_HEIGHT = 2, 10
MIN_WIDTH, MAX_WIDTH = 1, 9
MIN_COLOR_COUNT, MAX_COLOR_COUNT = 4, 5
MAX_CELLS, MAX_OUTPUT_WIDTH = 36, 30
PALETTE_VALUES = range(1, 10)

# Fill pattern controls.
FILL_MODES = ("noise", "row_runs", "column_bands", "diagonal_bands")
MAX_ROW_RUN_CUTS = 3
COLUMN_NOISE_PROB = 0.5
MIN_DIAGONAL_BAND, MAX_DIAGONAL_BAND = 1, 3
DIAGONAL_FLIP_PROB = 0.5

# Generation safety.
GENERATION_RETRIES = 300


def _ensure_palette_coverage(arr, palette):
    positions = random.sample(range(arr.size), len(palette))
    rows, cols = np.divmod(positions, arr.shape[1])
    arr[rows, cols] = palette


def _fill_grid(h, w, palette):
    k = len(palette)
    mode = random.choice(FILL_MODES)
    arr = np.zeros((h, w), dtype=int)

    if mode == "noise":
        arr[:] = np.random.choice(palette, size=(h, w))
    elif mode == "row_runs":
        for r in range(h):
            cuts = sorted(random.sample(range(1, w), random.randint(0, min(w - 1, MAX_ROW_RUN_CUTS)))) if w > 1 else []
            starts = [0] + cuts
            ends = cuts + [w]
            last = None
            for c0, c1 in zip(starts, ends):
                choices = [c for c in palette if c != last] or list(palette)
                last = random.choice(choices)
                arr[r, c0:c1] = last
    elif mode == "column_bands":
        for c in range(w):
            arr[:, c] = random.choice(palette)
        if random.random() < COLUMN_NOISE_PROB:
            for r in range(h):
                arr[r, random.randrange(w)] = random.choice(palette)
    else:
        band = random.randint(MIN_DIAGONAL_BAND, MAX_DIAGONAL_BAND)
        phase = random.randrange(k)
        rows, cols = np.indices((h, w))
        arr[:] = np.asarray(palette)[((rows + cols) // band + phase) % k]
        if random.random() < DIAGONAL_FLIP_PROB:
            arr = np.fliplr(arr)

    _ensure_palette_coverage(arr, palette)
    return arr


def _gen_attempt():
    k = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    h = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    widths = [
        w
        for w in range(MIN_WIDTH, MAX_WIDTH + 1)
        if h * w >= k and h * w <= MAX_CELLS and w + h - 1 <= MAX_OUTPUT_WIDTH
    ]
    if not widths:
        return None
    w = random.choice(widths)
    palette = random.sample(PALETTE_VALUES, k)
    return _fill_grid(h, w, palette)


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
