import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_GEN_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0, "expected at least one foreground pixel"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colors of nested rectangular layers from outside to inside and
    linearize them into a 1xN color strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg = g != _BG
    r0, c0, r1, c1 = _bbox(fg)
    colors: list[int] = []

    while r0 <= r1 and c0 <= c1:
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        unique_window = {int(v) for v in np.unique(window)}
        if _BG not in unique_window and len(unique_window) == 1:
            colors.append(next(iter(unique_window)))
            break

        perimeter = _perimeter_coords(r0, c0, r1, c1)
        vals = {int(g[r, c]) for r, c in perimeter}
        assert _BG not in vals, "layer perimeter must be fully colored"
        assert len(vals) == 1, "each layer perimeter must have one color"
        colors.append(next(iter(vals)))

        if r0 == r1 or c0 == c1:
            break
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1

    return np.array([colors], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(_RNG.integers(2, 8))
    point_center = bool(_RNG.integers(0, 2))

    if point_center:
        inner_h = inner_w = 1
    else:
        inner_h = int(_RNG.integers(2, 6))
        inner_w = int(_RNG.integers(2, 6))

    obj_h = inner_h + 2 * (n_layers - 1)
    obj_w = inner_w + 2 * (n_layers - 1)

    margin_top = int(_RNG.integers(1, 5))
    margin_bottom = int(_RNG.integers(1, 5))
    margin_left = int(_RNG.integers(1, 5))
    margin_right = int(_RNG.integers(1, 5))
    grid_h = obj_h + margin_top + margin_bottom
    grid_w = obj_w + margin_left + margin_right

    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    colors = _RNG.choice(_COLORS, size=n_layers, replace=False).astype(int)

    for depth, color in enumerate(colors):
        inset = depth
        rr0 = margin_top + inset
        cc0 = margin_left + inset
        rr1 = margin_top + obj_h - 1 - inset
        cc1 = margin_left + obj_w - 1 - inset

        if depth == n_layers - 1:
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
        else:
            grid[rr0, cc0 : cc1 + 1] = int(color)
            grid[rr1, cc0 : cc1 + 1] = int(color)
            grid[rr0 : rr1 + 1, cc0] = int(color)
            grid[rr0 : rr1 + 1, cc1] = int(color)

    out = main(grid)
    assert out.shape == (1, n_layers)
    assert np.array_equal(out[0], colors)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")
