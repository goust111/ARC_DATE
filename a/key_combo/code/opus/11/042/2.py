import random

import numpy as np


BG = 0
MARKER = 2
BAR = 4
PIXEL = 8

HEIGHT = 12
WIDTH = 12
PLAY_BOTTOM = HEIGHT - 3
RENDER_TOP = HEIGHT - 2
RENDER_BOTTOM = HEIGHT - 1
TOP_PLAY_ROWS = tuple(range(HEIGHT - 2))
EVEN_COLS = tuple(c for c in range(WIDTH) if c % 2 == 0)
ACTIVE_CAPACITY = {
    c: tuple(r for r in TOP_PLAY_ROWS if r % 2 == 0 and (r, c) != (0, 0))
    for c in EVEN_COLS
}
INACTIVE_POOLS = {
    "oe": tuple((r, c) for r in TOP_PLAY_ROWS for c in range(WIDTH) if r % 2 == 1 and c % 2 == 0),
    "eo": tuple((r, c) for r in TOP_PLAY_ROWS for c in range(WIDTH) if r % 2 == 0 and c % 2 == 1),
    "oo": tuple((r, c) for r in TOP_PLAY_ROWS for c in range(WIDTH) if r % 2 == 1 and c % 2 == 1),
}
GEN_ATTEMPTS = 500


def _solve_from_mask(grid: np.ndarray, active_mask: np.ndarray, *, keep_inactive: bool = False) -> np.ndarray:
    out = np.full_like(grid, BG)
    out[0, 0] = MARKER

    if keep_inactive:
        inactive_mask = (grid == PIXEL) & ~active_mask
        out[inactive_mask] = PIXEL

    counts = np.count_nonzero(active_mask, axis=0)
    for c, height in enumerate(counts):
        k = int(height)
        if k == 0:
            continue
        top = PLAY_BOTTOM - k + 1
        assert top >= 0
        out[top : PLAY_BOTTOM + 1, c] = PIXEL
        out[RENDER_BOTTOM, c] = BAR
        if k >= 2:
            out[RENDER_TOP, c] = BAR
    return out


def _parity_masks(grid: np.ndarray) -> dict[str, np.ndarray]:
    rr, cc = np.indices(grid.shape)
    cyan = grid == PIXEL
    return {
        "ee": cyan & (rr % 2 == 0) & (cc % 2 == 0),
        "eo": cyan & (rr % 2 == 0) & (cc % 2 == 1),
        "oe": cyan & (rr % 2 == 1) & (cc % 2 == 0),
        "oo": cyan & (rr % 2 == 1) & (cc % 2 == 1),
    }


def _sample_active_counts() -> dict[int, int]:
    while True:
        counts = {c: 0 for c in EVEN_COLS}
        hi_col = random.choice(EVEN_COLS)
        counts[hi_col] = random.randint(3, len(ACTIVE_CAPACITY[hi_col]))

        remaining = [c for c in EVEN_COLS if c != hi_col]
        two_col = random.choice(remaining)
        counts[two_col] = 2

        remaining = [c for c in remaining if c != two_col]
        one_col = random.choice(remaining)
        counts[one_col] = 1

        for c in remaining:
            counts[c] = random.randint(0, min(2, len(ACTIVE_CAPACITY[c])))

        values = list(counts.values())
        total = sum(values)
        if not (6 <= total <= 10):
            continue
        if 0 not in values:
            continue
        if not any(v == 1 for v in values):
            continue
        if not any(v == 2 for v in values):
            continue
        if not any(v >= 3 for v in values):
            continue
        return counts


def _place_active_pixels(grid: np.ndarray, counts: dict[int, int]) -> None:
    for c, k in counts.items():
        if k == 0:
            continue
        rows = random.sample(list(ACTIVE_CAPACITY[c]), k)
        for r in rows:
            grid[r, c] = PIXEL


def _place_inactive_pixels(grid: np.ndarray, active_total: int) -> None:
    total_pixels = random.randint(12, 16)
    inactive_total = total_pixels - active_total
    if inactive_total < 4:
        raise ValueError("inactive pool too small")

    class_counts = {"oe": 1, "eo": 1, "oo": 1}
    leftovers = inactive_total - 3
    classes = list(class_counts)
    for _ in range(leftovers):
        class_counts[random.choice(classes)] += 1

    for name, count in class_counts.items():
        coords = random.sample(list(INACTIVE_POOLS[name]), count)
        for r, c in coords:
            assert grid[r, c] == BG
            grid[r, c] = PIXEL


def _is_valid_input(grid: np.ndarray) -> bool:
    if grid.shape != (HEIGHT, WIDTH):
        return False
    if grid[0, 0] != MARKER:
        return False
    if np.count_nonzero(grid == MARKER) != 1:
        return False
    if np.any(grid[HEIGHT - 2 :, :] == PIXEL):
        return False
    if not set(np.unique(grid)).issubset({BG, MARKER, PIXEL}):
        return False

    masks = _parity_masks(grid)
    if any(np.count_nonzero(masks[name]) == 0 for name in ("eo", "oe", "oo")):
        return False

    active_counts = np.count_nonzero(masks["ee"], axis=0)
    even_counts = [int(active_counts[c]) for c in EVEN_COLS]
    if not any(v == 0 for v in even_counts):
        return False
    if not any(v == 1 for v in even_counts):
        return False
    if not any(v == 2 for v in even_counts):
        return False
    if not any(v >= 3 for v in even_counts):
        return False

    total_cyan = int(np.count_nonzero(grid == PIXEL))
    active_total = int(np.count_nonzero(masks["ee"]))
    inactive_total = total_cyan - active_total
    if not (12 <= total_cyan <= 16):
        return False
    if not (6 <= active_total <= 10):
        return False
    if inactive_total < 4:
        return False
    if abs(active_total - inactive_total) > 4:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out[HEIGHT - 2 :, :] == PIXEL):
        return False
    if np.count_nonzero(out == BAR) == 0:
        return False
    if not np.any(out[RENDER_TOP, :] == BAR):
        return False

    true_out = out
    row_even = masks["ee"] | masks["eo"]
    col_even = masks["ee"] | masks["oe"]
    all_cyan = grid == PIXEL
    alternatives = [
        _solve_from_mask(grid, all_cyan),
        _solve_from_mask(grid, row_even),
        _solve_from_mask(grid, col_even),
        _solve_from_mask(grid, masks["eo"]),
        _solve_from_mask(grid, masks["oe"]),
        _solve_from_mask(grid, masks["oo"]),
        _solve_from_mask(grid, masks["ee"], keep_inactive=True),
    ]
    if any(np.array_equal(true_out, alt) for alt in alternatives):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Let only the cyan pixels on the marker's parity lattice fall, then render each fallen column as a two-cell count bar.

    Input structure:
    - The grid is a black field with a single red marker at the top-left corner and several cyan single-pixel objects above a reserved two-row band.
    - The bottom two rows are initially empty and act as a rendering area.
    - Each cyan pixel is interpreted independently by its row and column coordinates.

    Rule:
    - Use the red marker's row parity and column parity to select the active lattice; only cyan pixels on cells with the same row parity and the same column parity stay active.
    - Delete every non-active cyan pixel, keep the red marker fixed, and let the active cyan pixels fall straight down within their own columns until they stack on the row just above the rendering band or on previously fallen active pixels.
    - For each column with fallen height K, paint the bottom rendering cell yellow when K >= 1 and the cell above it yellow when K >= 2, capping the rendered bar height at 2.

    Key insight:
    - The marker identifies one of the four coordinate-parity lattices, and only that lattice survives to create both the gravity stacks and the bottom count bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3
    assert g[0, 0] == MARKER

    rr, cc = np.indices(g.shape)
    active_mask = (g == PIXEL) & (rr % 2 == 0) & (cc % 2 == 0)
    return _solve_from_mask(g, active_mask)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = np.full((HEIGHT, WIDTH), BG, dtype=int)
        grid[0, 0] = MARKER

        active_counts = _sample_active_counts()
        _place_active_pixels(grid, active_counts)
        try:
            _place_inactive_pixels(grid, sum(active_counts.values()))
        except ValueError:
            continue

        if _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a non-degenerate input")
