import itertools
import numpy as np


GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
HIDDEN_ROW_RANGE = (4, 9)
HIDDEN_COL_RANGE = (4, 9)
CLUTTER_COLOR_COUNT_RANGE = (3, 5)
GEN_RETRIES = 1000


def _outer_ring_color(grid: np.ndarray) -> int:
    ring = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(ring, return_counts=True)
    return int(colors[np.argmax(counts)])


def _pair_candidates(interior: np.ndarray, required_color: int | None = None):
    colors = [int(color) for color in np.unique(interior)]
    masks = {color: interior == color for color in colors}
    candidates = []
    for a, b in itertools.combinations(colors, 2):
        if required_color is not None and required_color not in (a, b):
            continue
        mask = masks[a] | masks[b]
        if not mask.any():
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        subgrid = interior[np.ix_(rows, cols)]
        if np.all((subgrid == a) | (subgrid == b)):
            candidates.append((int(a), int(b), rows.astype(int), cols.astype(int)))
    return candidates


def _find_hidden_pair(interior: np.ndarray, keep_color: int):
    candidates = _pair_candidates(interior, required_color=keep_color)
    if len(candidates) != 1:
        raise ValueError("expected exactly one keep-color candidate pair")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError("expected a 2D grid with a 1-cell border")

    keep_color = _outer_ring_color(g)
    interior = g[1:-1, 1:-1]
    color_a, color_b, rows, cols = _find_hidden_pair(interior, keep_color)
    if keep_color not in (color_a, color_b):
        raise ValueError("border color must belong to the hidden pair")

    hidden = interior[np.ix_(rows, cols)]
    out = np.where(hidden == keep_color, keep_color, 0)
    return out.astype(int)


def _choose_axis_indices(length: int, count: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "contiguous":
        start = int(rng.integers(0, length - count + 1))
        return np.arange(start, start + count, dtype=int)

    for _ in range(64):
        picks = np.sort(rng.choice(length, size=count, replace=False).astype(int))
        if np.any(np.diff(picks) > 1):
            return picks
    start = int(rng.integers(0, length - count + 1))
    return np.arange(start, start + count, dtype=int)


def _make_keep_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    rr, cc = np.indices((height, width))
    for _ in range(64):
        mode = int(rng.integers(0, 6))
        if mode == 0:
            phase = int(rng.integers(0, 2))
            mask = (rr + cc + phase) % 2 == 0
        elif mode == 1:
            period = int(rng.integers(2, 4))
            width_band = int(rng.integers(1, period))
            phase = int(rng.integers(0, period))
            mask = ((rr + phase) % period) < width_band
        elif mode == 2:
            period = int(rng.integers(2, 4))
            width_band = int(rng.integers(1, period))
            phase = int(rng.integers(0, period))
            mask = ((cc + phase) % period) < width_band
        elif mode == 3:
            dist = np.minimum.reduce((rr, cc, height - 1 - rr, width - 1 - cc))
            thickness = int(rng.integers(1, max(2, min(height, width) // 3 + 1)))
            mask = dist < thickness
            if bool(rng.integers(0, 2)):
                mask = ~mask
        elif mode == 4:
            if bool(rng.integers(0, 2)):
                cut = int(rng.integers(width // 2, height + width // 2 + 1))
                mask = rr + cc <= cut
            else:
                offset = int(rng.integers(-(width - 2), height - 1))
                mask = (rr - cc) >= offset
        else:
            mask = np.zeros((height, width), dtype=bool)
            block_count = int(rng.integers(1, 4))
            for _ in range(block_count):
                r0 = int(rng.integers(0, height))
                r1 = int(rng.integers(r0 + 1, height + 1))
                c0 = int(rng.integers(0, width))
                c1 = int(rng.integers(c0 + 1, width + 1))
                mask[r0:r1, c0:c1] = True
            if bool(rng.integers(0, 2)):
                mask = ~mask

        frac = float(mask.mean())
        if 0.2 <= frac <= 0.8:
            return mask

    return (rr + cc) % 2 == 0


def _paint_clutter(interior: np.ndarray, clutter_colors: np.ndarray, rng: np.random.Generator) -> None:
    height, width = interior.shape
    interior[:, :] = rng.choice(clutter_colors, size=(height, width))

    patch_count = int(rng.integers(5, 10))
    for _ in range(patch_count):
        color = int(rng.choice(clutter_colors))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            r0 = int(rng.integers(0, height))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c0 = int(rng.integers(0, width))
            c1 = int(rng.integers(c0 + 1, width + 1))
            interior[r0:r1, c0:c1] = color
        elif kind == 1:
            r0 = int(rng.integers(0, height))
            r1 = min(height, r0 + int(rng.integers(1, 3)))
            c0 = int(rng.integers(0, width - 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            interior[r0:r1, c0:c1] = color
        elif kind == 2:
            c0 = int(rng.integers(0, width))
            c1 = min(width, c0 + int(rng.integers(1, 3)))
            r0 = int(rng.integers(0, height - 1))
            r1 = int(rng.integers(r0 + 1, height + 1))
            interior[r0:r1, c0:c1] = color
        else:
            dot_count = int(rng.integers(4, 13))
            flat = rng.choice(height * width, size=dot_count, replace=False)
            interior.flat[flat] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    inner_h = height - 2
    inner_w = width - 2

    max_rows = min(HIDDEN_ROW_RANGE[1], inner_h - 2)
    max_cols = min(HIDDEN_COL_RANGE[1], inner_w - 2)
    row_count = int(rng.integers(HIDDEN_ROW_RANGE[0], max_rows + 1))
    col_count = int(rng.integers(HIDDEN_COL_RANGE[0], max_cols + 1))

    pair = rng.choice(np.arange(1, 10), size=2, replace=False).astype(int)
    keep_first = bool(rng.integers(0, 2))
    keep_color = int(pair[0] if keep_first else pair[1])
    other_color = int(pair[1] if keep_first else pair[0])

    remaining = np.array([color for color in range(10) if color not in (keep_color, other_color)], dtype=int)
    clutter_color_count = int(
        rng.integers(CLUTTER_COLOR_COUNT_RANGE[0], CLUTTER_COLOR_COUNT_RANGE[1] + 1)
    )
    clutter_colors = rng.choice(remaining, size=clutter_color_count, replace=False).astype(int)

    layout_mode = int(rng.integers(0, 4))
    if layout_mode == 0:
        row_mode, col_mode = "contiguous", "contiguous"
    elif layout_mode == 1:
        row_mode, col_mode = "gapped", "contiguous"
    elif layout_mode == 2:
        row_mode, col_mode = "contiguous", "gapped"
    else:
        row_mode, col_mode = "gapped", "gapped"

    rows = _choose_axis_indices(inner_h, row_count, rng, row_mode)
    cols = _choose_axis_indices(inner_w, col_count, rng, col_mode)
    keep_mask = _make_keep_mask(row_count, col_count, rng)

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = keep_color
    grid[-1, :] = keep_color
    grid[:, 0] = keep_color
    grid[:, -1] = keep_color

    interior = grid[1:-1, 1:-1]
    _paint_clutter(interior, clutter_colors, rng)

    hidden = np.where(keep_mask, keep_color, other_color).astype(int)
    interior[np.ix_(rows, cols)] = hidden

    expected = np.where(hidden == keep_color, keep_color, 0).astype(int)
    if expected.min() == expected.max():
        return None

    candidates = _pair_candidates(interior)
    if len(candidates) != 1:
        return None

    cand_a, cand_b, cand_rows, cand_cols = candidates[0]
    if {cand_a, cand_b} != {keep_color, other_color}:
        return None
    if not np.array_equal(cand_rows, rows) or not np.array_equal(cand_cols, cols):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
