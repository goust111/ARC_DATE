import numpy as np

from arc_utils import Color, bbox
import random


def _unique_most_frequent_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    assert colors.size > 0, "expected at least one marker pixel"

    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "expected a unique most frequent marker color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the majority interior marker color, then redraw the framed board as that color's frame plus its bounding-box outline.

    Input structure:
    - The top row is a copied palette made from adjacent equal color pairs.
    - Below it is a one-cell-thick rectangular frame around a black interior.
    - Non-black, non-frame interior pixels are marker colors; one marker color has
      the unique highest count.

    Rule:
    - Keep the palette row unchanged.
    - Pick the uniquely most frequent interior marker color.
    - Clear the lower board to black, recolor its outer frame with the picked color,
      and draw the picked color's interior bounding-box border.

    Key insight:
    - The noisy marker cloud is not preserved as pixels; only the majority marker
      color and its spatial extent define the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4, "expected room for a palette row, frame, and interior"

    palette = g[0, :].copy()
    assert width % 2 == 0, "palette row must split into adjacent color pairs"
    palette_pairs = palette.reshape(-1, 2)
    assert np.all(palette_pairs[:, 0] == palette_pairs[:, 1]), "expected paired palette colors"
    palette_colors = set(map(int, palette_pairs[:, 0]))

    frame_color = int(g[1, 0])
    assert frame_color != Color.BLACK, "frame must be non-black"
    assert frame_color in palette_colors, "frame color must appear in the palette row"
    assert np.all(g[1, :] == frame_color), "top of lower frame must be solid"
    assert np.all(g[-1, :] == frame_color), "bottom of lower frame must be solid"
    assert np.all(g[1:, 0] == frame_color), "left side of lower frame must be solid"
    assert np.all(g[1:, -1] == frame_color), "right side of lower frame must be solid"

    interior = g[2:-1, 1:-1]
    marker_mask = (interior != Color.BLACK) & (interior != frame_color)
    marker_values = interior[marker_mask]
    assert set(map(int, marker_values)).issubset(palette_colors), "marker colors must come from the palette row"
    draw_color = _unique_most_frequent_color(marker_values)

    selected_mask = np.zeros(g.shape, dtype=bool)
    selected_mask[2:-1, 1:-1] = interior == draw_color
    selected_bbox = bbox(selected_mask)
    assert selected_bbox is not None, "selected marker color must have a bounding box"
    r0, c0, r1, c1 = selected_bbox

    out = np.zeros_like(g)
    out[0, :] = palette

    out[1, :] = draw_color
    out[-1, :] = draw_color
    out[1:, 0] = draw_color
    out[1:, -1] = draw_color

    out[r0 : r1 + 1, c0] = draw_color
    out[r0 : r1 + 1, c1] = draw_color
    out[r0, c0 : c1 + 1] = draw_color
    out[r1, c0 : c1 + 1] = draw_color

    return out

# --------------- generator (o5x) ---------------
# Board and palette.
GRID_SHAPE = (10, 10)
PALETTE_ROW = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int)
PALETTE_COLORS = [1, 2, 3, 4, 5]

# Interior bbox and marker sampling.
BBOX_HEIGHT_BOUNDS = (2, 6)
BBOX_WIDTH_BOUNDS = (2, 7)
INTERIOR_ROW_BOUNDS = (2, 8)
INTERIOR_COL_BOUNDS = (1, 8)
WINNER_COUNT_BOUNDS = (2, 10)
DIAGONAL_CHOICES = [False, True]
GENERATION_RETRIES = 300

def _gen_draw_frame(grid: np.ndarray, color: int) -> None:
    grid[[1, -1], :] = color
    grid[1:, [0, -1]] = color

def _gen_draw_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, [c0, c1]] = color
    grid[[r0, r1], c0 : c1 + 1] = color

def _gen_blank_grid(frame_color: int) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[0, :] = PALETTE_ROW
    _gen_draw_frame(grid, frame_color)
    return grid

def _gen_attempt() -> np.ndarray | None:
    frame_color, winner_color, loser_color = random.sample(PALETTE_COLORS, 3)

    height = random.randint(*BBOX_HEIGHT_BOUNDS)
    width = random.randint(*BBOX_WIDTH_BOUNDS)
    r0 = random.randint(INTERIOR_ROW_BOUNDS[0], INTERIOR_ROW_BOUNDS[1] - height + 1)
    c0 = random.randint(INTERIOR_COL_BOUNDS[0], INTERIOR_COL_BOUNDS[1] - width + 1)
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    out = _gen_blank_grid(winner_color)
    _gen_draw_outline(out, r0, c0, r1, c1, winner_color)

    inp = _gen_blank_grid(frame_color)

    if random.choice(DIAGONAL_CHOICES):
        required = [(r0, c0), (r1, c1)]
    else:
        required = [(r0, c1), (r1, c0)]

    bbox_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    extra_cells = [cell for cell in bbox_cells if cell not in required]
    winner_count = random.randint(WINNER_COUNT_BOUNDS[0], min(WINNER_COUNT_BOUNDS[1], len(bbox_cells)))
    winner_cells = required + random.sample(extra_cells, winner_count - len(required))

    used = set(winner_cells)
    for r, c in winner_cells:
        inp[r, c] = winner_color

    free_interior = [
        (r, c)
        for r in range(INTERIOR_ROW_BOUNDS[0], INTERIOR_ROW_BOUNDS[1] + 1)
        for c in range(INTERIOR_COL_BOUNDS[0], INTERIOR_COL_BOUNDS[1] + 1)
        if (r, c) not in used
    ]
    loser_count = random.randint(1, min(winner_count - 1, len(free_interior)))
    for r, c in random.sample(free_interior, loser_count):
        inp[r, c] = loser_color

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ff2825db input")
