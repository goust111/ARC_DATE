from __future__ import annotations

import numpy as np

from arc_utils import bbox, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the top-left seed mask into a recursive colored pattern and tile it across the full grid.

    Input structure
    - The background is the dominant color.
    - All foreground lies in a top-left rectangle whose last row and last column are a solid secondary color.
    - Removing that secondary-color row and column leaves a seed region containing background plus one primary color.

    Rule
    - Use the primary-color pixels in the seed as a binary mask.
    - Replace each primary seed cell with a copy of the seed mask in the primary color, and each background seed cell
      with a copy of the seed mask in the secondary color.
    - Repeat and crop that expanded tile to the original grid shape.

    Key insight
    The solid L-shaped secondary-color border is a legend, not output content; it supplies the color used when the
    seed's background cells are expanded.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    fg_bbox = bbox(g != bg)
    assert fg_bbox is not None, "expected a top-left foreground legend"
    r0, c0, r1, c1 = fg_bbox
    assert r0 == 0 and c0 == 0, "foreground legend must be anchored at the top-left corner"

    legend_box = g[: r1 + 1, : c1 + 1]
    seed_h = legend_box.shape[0] - 1
    seed_w = legend_box.shape[1] - 1
    assert seed_h >= 1 and seed_w >= 1, "legend must contain a non-empty seed plus border"

    secondary_border = np.concatenate((legend_box[:seed_h, seed_w], legend_box[seed_h, :]))
    secondary_colors = np.unique(secondary_border)
    assert secondary_colors.size == 1, "expected one solid secondary-color border"
    secondary = int(secondary_colors[0])
    assert secondary != bg, "secondary border must be foreground"

    seed = legend_box[:seed_h, :seed_w]
    primary_colors = np.unique(seed[seed != bg])
    assert primary_colors.size == 1, "seed must contain exactly one primary foreground color"
    primary = int(primary_colors[0])
    assert primary != secondary, "primary and secondary colors must be distinct"
    assert np.all((seed == bg) | (seed == primary)), "seed may contain only background and primary colors"
    assert np.any(seed == bg), "seed must include background cells that expand to the secondary color"

    seed_mask = seed == primary
    H, W = g.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    inner_mask = seed_mask[rows % seed_h, cols % seed_w]
    meta_mask = seed_mask[(rows // seed_h) % seed_h, (cols // seed_w) % seed_w]

    out = np.full(g.shape, bg, dtype=int)
    out[inner_mask & meta_mask] = primary
    out[inner_mask & ~meta_mask] = secondary
    return out

# --------------- generator (att2, o5x) ---------------

# Color palette.
BACKGROUND_COLOR, PALETTE_COLORS = 7, tuple(range(10))
# Seed geometry.
SEED_SIZES = (2, 3)
SEED_DENSITY_MIN, SEED_DENSITY_MAX = 0.33, 0.67
# Canvas layout.
TILE_MULTIPLIERS, GRID_MIN, GRID_MAX = range(3, 6), 7, 16
REMAINDERS_BY_SQUARENESS = {False: (0,), True: (0, 1)}
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _expected_output(seed_mask: np.ndarray, shape: tuple[int, int], bg: int, primary: int, secondary: int) -> np.ndarray:
    """Render the two-level fractal that main() should produce."""
    seed_h, seed_w = seed_mask.shape
    H, W = shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    inner = seed_mask[rows % seed_h, cols % seed_w]
    meta = seed_mask[(rows // seed_h) % seed_h, (cols // seed_w) % seed_w]

    out = np.full((H, W), bg, dtype=int)
    out[inner & meta] = primary
    out[inner & ~meta] = secondary
    return out

def _layout_options(seed_h: int, seed_w: int) -> list[tuple[int, int]]:
    """Return valid canvas shapes for a sampled seed size."""
    options = []
    for k in TILE_MULTIPLIERS:
        for rem in REMAINDERS_BY_SQUARENESS[seed_h == seed_w]:
            H = seed_h * k + rem
            W = seed_w * k + rem
            if GRID_MIN <= H <= GRID_MAX and GRID_MIN <= W <= GRID_MAX and H >= seed_h * seed_h and W >= seed_w * seed_w:
                options.append((H, W))
    return options

def _random_seed_mask(seed_h: int, seed_w: int) -> np.ndarray:
    """Sample a nonempty, nonfull primary mask within density bounds."""
    area = seed_h * seed_w
    lo = max(1, int(np.ceil(area * SEED_DENSITY_MIN)))
    hi = min(area - 1, int(np.floor(area * SEED_DENSITY_MAX)))
    n_primary = random.randint(lo, hi)
    chosen = random.sample(range(area), n_primary)
    mask = np.zeros((seed_h, seed_w), dtype=bool)
    mask.reshape(-1)[chosen] = True
    return mask

def _render_input(seed_mask: np.ndarray, shape: tuple[int, int], bg: int, primary: int, secondary: int) -> np.ndarray:
    """Render the top-left seed legend on a background canvas."""
    seed_h, seed_w = seed_mask.shape
    inp = np.full(shape, bg, dtype=int)
    inp[:seed_h, :seed_w][seed_mask] = primary
    inp[seed_h, : seed_w + 1] = secondary
    inp[:seed_h, seed_w] = secondary
    return inp

def _gen_attempt() -> np.ndarray | None:
    """Build one oracle-checked input grid, or None if validation fails."""
    bg = BACKGROUND_COLOR
    primary, secondary = random.sample([c for c in PALETTE_COLORS if c != bg], 2)

    seed_h = random.choice(SEED_SIZES)
    seed_w = random.choice(SEED_SIZES)
    H, W = random.choice(_layout_options(seed_h, seed_w))

    seed_mask = _random_seed_mask(seed_h, seed_w)
    inp = _render_input(seed_mask, (H, W), bg, primary, secondary)
    expected = _expected_output(seed_mask, (H, W), bg, primary, secondary)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58c02a16 input")
