import random

import numpy as np


COLOR_CHOICES = tuple(range(1, 10))
GRID_SIZE_RANGE = (2, 5)
PALETTE_SIZE_RANGE = (2, 5)
GENERATION_RETRIES = 300


def _render_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=int)

    color_blocks = {
        int(color): np.where(g == int(color), int(color), 0)
        for color in np.unique(g)
    }

    for r in range(h):
        for c in range(w):
            block = color_blocks[int(g[r, c])]
            if r % 2 == 1:
                block = np.flipud(block)
            if c % 2 == 1:
                block = np.fliplr(block)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each source pixel into a whole-grid self-mask tile mirrored by slot parity.

    Input structure:
    - The input is a small foreground-only grid with no black cells.
    - Colors may repeat, and each repeated color defines a global mask over the whole input.

    Rule:
    - Replace every source cell (r, c) with an HxW block whose non-black cells mark exactly the positions of the same color in the input.
    - Paint those selected positions with that source cell's color and paint all other positions black.
    - If c is odd, flip the block left-right; if r is odd, flip it top-bottom; if both are odd, apply both reflections before placing the block.

    Key insight:
    - Each output block is the full-grid mask for one color, while the slot coordinates independently choose which mirror view of that mask appears there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 1 <= h <= 30 and 1 <= w <= 30, "input grid must be ARC-sized"
    assert h * h <= 30 and w * w <= 30, "expanded output must stay within 30x30"
    assert np.all(g != 0), "input must be foreground-only"
    return _render_output(g)


def _all_colors_used(grid: np.ndarray, palette: list[int]) -> bool:
    return set(map(int, palette)).issubset(set(map(int, np.unique(grid))))


def _has_visible_reflection(grid: np.ndarray) -> bool:
    for color in np.unique(grid):
        mask = grid == int(color)
        if not np.array_equal(mask, np.flipud(mask)):
            return True
        if not np.array_equal(mask, np.fliplr(mask)):
            return True
    return False


def _ensure_palette_coverage(grid: np.ndarray, palette: list[int]) -> np.ndarray:
    g = grid.copy()
    positions = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1])]
    random.shuffle(positions)
    missing = [color for color in palette if color not in g]
    for color, (r, c) in zip(missing, positions):
        g[r, c] = color
    return g


def _force_visible_reflection(grid: np.ndarray, palette: list[int]) -> np.ndarray:
    g = grid.copy()
    if _has_visible_reflection(g):
        return g

    h, w = g.shape
    color = random.choice(palette)
    g[:, :] = palette[0]
    r = random.randrange(h)
    c = random.randrange(w)
    if (r, c) == (h - 1 - r, c):
        r = 0 if r != 0 else h - 1
    if (r, c) == (r, w - 1 - c):
        c = 0 if c != 0 else w - 1
    g[r, c] = color
    g[h - 1 - r, w - 1 - c] = random.choice([p for p in palette if p != color] or [color])
    return g


def _make_scatter_grid(h: int, w: int, palette: list[int]) -> np.ndarray:
    cells = list(palette)
    cells.extend(random.choices(palette, k=h * w - len(cells)))
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(h, w)


def _make_band_grid(h: int, w: int, palette: list[int]) -> np.ndarray:
    horizontal = random.choice([False, True])
    if horizontal:
        row_colors = [random.choice(palette) for _ in range(h)]
        grid = np.array([[row_colors[r]] * w for r in range(h)], dtype=int)
    else:
        col_colors = [random.choice(palette) for _ in range(w)]
        grid = np.array([[col_colors[c] for c in range(w)] for _ in range(h)], dtype=int)

    edits = random.randint(1, max(2, (h * w) // 3))
    for _ in range(edits):
        r = random.randrange(h)
        c = random.randrange(w)
        grid[r, c] = random.choice(palette)
    return grid


def _make_patch_grid(h: int, w: int, palette: list[int]) -> np.ndarray:
    base = random.choice(palette)
    grid = np.full((h, w), base, dtype=int)
    patch_count = random.randint(2, min(len(palette) + 2, h * w))

    for _ in range(patch_count):
        color = random.choice(palette)
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        r1 = random.randint(r0, h - 1)
        c1 = random.randint(c0, w - 1)
        if random.choice([False, True]):
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            grid[r0, c0] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    palette_size = random.randint(PALETTE_SIZE_RANGE[0], min(PALETTE_SIZE_RANGE[1], h * w, len(COLOR_CHOICES)))
    palette = random.sample(COLOR_CHOICES, palette_size)

    mode = random.choice(["scatter", "bands", "patches"])
    if mode == "scatter":
        grid = _make_scatter_grid(h, w, palette)
    elif mode == "bands":
        grid = _make_band_grid(h, w, palette)
    else:
        grid = _make_patch_grid(h, w, palette)

    grid = _ensure_palette_coverage(grid, palette)
    grid = _force_visible_reflection(grid, palette)

    if not _all_colors_used(grid, palette):
        return None
    if not _has_visible_reflection(grid):
        return None
    if np.any(grid == 0):
        return None

    out = _render_output(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, _render_output(grid)):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
