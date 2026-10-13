import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_RETRIES = 300
RNG = np.random.default_rng()


def _split_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0
    n = h // 2
    return g[:n, :n], g[:n, n:], g[n:, :n], g[n:, n:]


def _tile_value(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[r % tile.shape[0], c % tile.shape[1]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper quadrants only as size controls, then tile the lower quadrants.

    - Output height = non-background count in the top-left quadrant.
    - Output width = non-background count in the top-right quadrant.
    - If the bottom-right quadrant has any foreground pixels, tile that template;
      its background cells are recolored from the bottom-left tile at the same
      repeating coordinates.
    - Otherwise tile the bottom-left quadrant directly.
    """
    q_tl, q_tr, q_bl, q_br = _split_quadrants(grid)

    out_h = int(np.count_nonzero(q_tl != BG))
    out_w = int(np.count_nonzero(q_tr != BG))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    use_template = bool(np.any(q_br != BG))

    for r in range(out_h):
        for c in range(out_w):
            color = _tile_value(q_bl, r, c)
            if use_template:
                mask_value = _tile_value(q_br, r, c)
                out[r, c] = mask_value if mask_value != BG else color
            else:
                out[r, c] = color

    return out


def _sample_control(n: int, count: int, color_choices: np.ndarray) -> np.ndarray:
    quad = np.zeros((n, n), dtype=int)
    positions = RNG.choice(n * n, size=count, replace=False)
    colors = RNG.choice(color_choices, size=count, replace=True)
    quad.flat[positions] = colors
    return quad


def _make_color_tile(n: int) -> np.ndarray:
    tile = np.zeros((n, n), dtype=int)
    palette = RNG.choice(PALETTE, size=int(RNG.integers(2, 5)), replace=False)
    mode = int(RNG.integers(0, 5))

    if mode == 0:
        tile = RNG.choice(palette, size=(n, n), replace=True)
    elif mode == 1:
        row_colors = RNG.choice(palette, size=n, replace=True)
        tile = np.repeat(row_colors[:, None], n, axis=1)
    elif mode == 2:
        col_colors = RNG.choice(palette, size=n, replace=True)
        tile = np.repeat(col_colors[None, :], n, axis=0)
    elif mode == 3:
        a, b = RNG.choice(palette, size=2, replace=len(palette) == 1)
        rr, cc = np.indices((n, n))
        tile = np.where((rr + cc) % 2 == 0, a, b)
    else:
        rr, cc = np.indices((n, n))
        block_h = int(RNG.integers(1, min(4, n) + 1))
        block_w = int(RNG.integers(1, min(4, n) + 1))
        idx = (rr // block_h + cc // block_w) % len(palette)
        tile = palette[idx]

    # Keep the source tile visibly structured and not monochrome.
    if len(np.unique(tile)) == 1:
        r = int(RNG.integers(0, n))
        c = int(RNG.integers(0, n))
        alternatives = [int(x) for x in palette if int(x) != int(tile[r, c])]
        tile[r, c] = alternatives[0] if alternatives else int(PALETTE[0])
    return tile.astype(int)


def _make_template_tile(n: int, forbidden_colors: set[int]) -> np.ndarray:
    template = np.zeros((n, n), dtype=int)
    available = [int(c) for c in PALETTE if int(c) not in forbidden_colors]
    if not available:
        available = [1]
    fg = int(RNG.choice(np.array(available, dtype=int)))
    mode = int(RNG.integers(0, 5))
    rr, cc = np.indices((n, n))

    if mode == 0:
        thick = int(RNG.integers(1, min(3, (n + 1) // 2) + 1))
        mask = (rr < thick) | (rr >= n - thick) | (cc < thick) | (cc >= n - thick)
    elif mode == 1:
        step = int(RNG.integers(2, min(5, n + 1)))
        mask = ((rr + cc) % step) == 0
    elif mode == 2:
        period = int(RNG.integers(2, min(5, n + 1)))
        mask = (rr % period == 0) | (cc % period == 0)
    elif mode == 3:
        center = (n - 1) / 2.0
        radius = float(RNG.uniform(max(0.8, n * 0.18), max(1.2, n * 0.42)))
        mask = (rr - center) ** 2 + (cc - center) ** 2 <= radius**2
    else:
        mask = (np.abs(rr - cc) <= 0) | (np.abs(rr + cc - (n - 1)) <= 0)

    if not np.any(mask):
        mask[0, 0] = True
    if np.all(mask):
        mask[n // 2, n // 2] = False

    template[mask] = fg
    return template


def _gen_attempt() -> np.ndarray | None:
    n = int(RNG.integers(4, 9))
    max_count = n * n - 1
    out_h = int(RNG.integers(1, max_count + 1))
    out_w = int(RNG.integers(1, max_count + 1))

    control_palette = RNG.choice(PALETTE, size=int(RNG.integers(2, 5)), replace=False)
    q_tl = _sample_control(n, out_h, control_palette)
    q_tr = _sample_control(n, out_w, control_palette)
    q_bl = _make_color_tile(n)

    use_template = bool(RNG.integers(0, 2))
    if use_template:
        q_br = _make_template_tile(n, set(map(int, np.unique(q_bl))) | {BG})
    else:
        q_br = np.zeros((n, n), dtype=int)

    grid = np.zeros((2 * n, 2 * n), dtype=int)
    grid[:n, :n] = q_tl
    grid[:n, n:] = q_tr
    grid[n:, :n] = q_bl
    grid[n:, n:] = q_br

    out = main(grid)

    # Reject visually degenerate outputs.
    if out.size == 0:
        return None
    if len(np.unique(out)) == 1 and out.shape[0] * out.shape[1] > 1:
        return None
    if use_template and not np.any(q_br == BG):
        return None
    if use_template and not np.any(q_br != BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
