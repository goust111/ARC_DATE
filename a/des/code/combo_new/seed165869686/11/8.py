import numpy as np
from numpy.random import default_rng


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
LAYER_OPTIONS = (3, 4, 5, 6)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask))


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _peel_depth(mask: np.ndarray) -> tuple[np.ndarray, int]:
    depth = np.full(mask.shape, -1, dtype=int)
    remaining = mask.copy()
    layers = 0

    while remaining.any():
        frontier = _boundary(remaining)
        assert frontier.any(), "peeling stalled on a non-empty region"
        depth[frontier] = layers
        remaining[frontier] = False
        layers += 1

    return depth, layers


def _legend_info(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    assert grid[h - 1, 0] != BACKGROUND, "expected a bottom-left legend"

    rows: list[int] = []
    r = h - 1
    while r >= 0 and grid[r, 0] != BACKGROUND:
        rows.append(r)
        r -= 1

    assert rows, "legend must contain at least one color"
    assert np.all(grid[: r + 1, 0] == BACKGROUND), "legend must be a single bottom-anchored run"
    if w >= 2:
        assert np.all(grid[rows, 1] == BACKGROUND), "legend cells must be single-pixel markers"

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[rows, 0] = True
    legend_colors = [int(grid[row, 0]) for row in rows]
    return legend_mask, legend_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a nested multicolor region by peel depth using the bottom-left legend.

    Input structure:
    - The background is black.
    - Away from the corner legend, all non-background pixels form one large 4-connected
      multicolor region whose equal-depth layers appear as nested contour bands.
    - The bottom-left corner contains a vertical one-pixel legend in column 0; from
      bottom to top its colors specify the target colors for depths 0, 1, 2, ...

    Rule:
    - Ignore the original colors inside the large region.
    - Repeatedly peel the region inward: pixels touching background are depth 0, the
      next exposed pixels are depth 1, and so on until the center is exhausted.
    - Recolor every region pixel by the legend color assigned to its peel depth.
    - Keep the legend and the background unchanged.

    Key insight:
    - The visible ring colors are decoys; the true signal is each pixel's topological
      depth inside the single connected region, and the corner legend remaps that depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_mask, legend_colors = _legend_info(g)
    region_mask = g != BACKGROUND
    region_mask[legend_mask] = False

    comps = _components(region_mask)
    assert len(comps) == 1, "expected exactly one non-legend foreground region"

    region = np.zeros_like(region_mask, dtype=bool)
    for r, c in comps[0]:
        region[r, c] = True

    depth, layers = _peel_depth(region)
    assert layers <= len(legend_colors), "legend is too short for the region depth"

    out = g.copy()
    for layer in range(layers):
        out[depth == layer] = legend_colors[layer]
    return out


def _corner_notch(mask: np.ndarray, corner: str, rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        return
    h, w = mask.shape
    if corner == "tl":
        mask[:rows, :cols] = False
    elif corner == "tr":
        mask[:rows, w - cols :] = False
    elif corner == "bl":
        mask[h - rows :, :cols] = False
    else:
        mask[h - rows :, w - cols :] = False


def _make_region_mask(target_layers: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SHAPE_ATTEMPTS):
        min_dim = int(rng.choice([2 * target_layers - 1, 2 * target_layers]))
        other_dim = int(rng.integers(min_dim, min(16, min_dim + 4) + 1))
        if rng.random() < 0.5:
            h, w = min_dim, other_dim
        else:
            h, w = other_dim, min_dim

        mask = np.ones((h, w), dtype=bool)
        max_corner = min(2, target_layers - 1, h - 2, w - 2)
        for corner in ("tl", "tr", "bl", "br"):
            if max_corner > 0 and rng.random() < 0.85:
                nr = int(rng.integers(0, max_corner + 1))
                nc = int(rng.integers(0, max_corner + 1))
                _corner_notch(mask, corner, nr, nc)

        if not mask.all() and _component_count(mask) != 1:
            continue

        depth, layers = _peel_depth(mask)
        if layers != target_layers:
            continue
        if any(_component_count(depth == layer) != 1 for layer in range(layers)):
            continue
        return mask, depth

    raise RuntimeError(f"failed to build a valid {target_layers}-layer region")


def _band_colors(legend_colors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    band_colors = legend_colors.copy()
    while True:
        rng.shuffle(band_colors)
        if not np.array_equal(band_colors, legend_colors):
            return band_colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    target_layers = int(rng.choice(LAYER_OPTIONS))
    mask, depth = _make_region_mask(target_layers, rng)
    region_h, region_w = mask.shape

    min_h = max(region_h + 2, target_layers + 2)
    min_w = region_w + 5
    grid_h = int(rng.integers(min_h, min(30, min_h + 7) + 1))
    grid_w = int(rng.integers(min_w, min(30, min_w + 9) + 1))

    top = int(rng.integers(1, grid_h - region_h))
    left = int(rng.integers(3, grid_w - region_w))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend_colors = rng.choice(PALETTE, size=target_layers, replace=False).astype(int)
    band_colors = _band_colors(legend_colors, rng)

    for layer, color in enumerate(legend_colors):
        grid[grid_h - 1 - layer, 0] = int(color)

    patch = grid[top : top + region_h, left : left + region_w]
    for layer, color in enumerate(band_colors):
        patch[depth == layer] = int(color)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "expected a visible recolor"
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
