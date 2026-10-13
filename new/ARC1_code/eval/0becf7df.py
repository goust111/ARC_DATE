import numpy as np

from arc_utils import bbox, find_components, get_bg, is_contiguous, paste
import math
import random


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = [
        comp["bbox"]
        for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False)
        if comp["bbox"] == (0, 0, 1, 1) and comp["area"] == 4
    ]
    assert len(candidates) == 1, "expected one full 2x2 legend at the upper-left corner"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Swap each non-legend foreground color with its row-mate in the 2x2 upper-left legend.

    Input structure
    - A dominant background separates a full 2x2 foreground legend in the
      upper-left corner from the main multicolor foreground object.
    - The main object uses exactly the four colors shown in the legend.

    Rule
    - Leave the legend and background unchanged.
    - Treat each legend row as a bidirectional color pair.
    - Recolor every foreground pixel outside the legend to the other color in
      its legend row, using the original input colors for a simultaneous swap.

    Key insight
    - The small 2x2 block is a symbol instruction, not part of the object: its
      rows define the pairwise color substitutions for the separate object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    r0, c0, r1, c1 = _find_legend(g, bg)
    legend = g[r0 : r1 + 1, c0 : c1 + 1]
    legend_colors = set(map(int, legend.ravel()))
    assert legend.shape == (2, 2)
    assert bg not in legend_colors, "legend must be foreground"
    assert len(legend_colors) == 4, "legend colors must be distinct"

    target_mask = g != bg
    target_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    assert target_mask.any(), "expected a foreground object outside the legend"
    assert set(map(int, g[target_mask])) == legend_colors, "target object must use the legend colors"

    out = g.copy()
    for row in range(2):
        left = int(legend[row, 0])
        right = int(legend[row, 1])
        assert left != right
        out[(g == left) & target_mask] = right
        out[(g == right) & target_mask] = left
    return out

# --------------- generator (o5x) ---------------
# Canvas, legend, and palette.
GRID_SHAPE = (10, 10)
BACKGROUND_COLOR = 0
COLOR_POOL = range(1, 10)
LEGEND_SHAPE = (2, 2)
LEGEND_ROWS, LEGEND_COLS = LEGEND_SHAPE
LEGEND_CELLS = LEGEND_ROWS * LEGEND_COLS

# Object geometry and placement.
OBJECT_ROW0, OBJECT_MIN_COL, OBJECT_RIGHT_PADDING = 1, 3, 1
OBJECT_HEIGHTS, OBJECT_WIDTHS = range(6, 9), range(4, 7)
OBJECT_AREA_MIN, OBJECT_AREA_MAX = 13, 23
DENSITY_MIN, DENSITY_MAX = 0.48, 0.72

# Blob construction, color balance, and retry budget.
VERTICAL_SPINE_PROB = 0.5
INTERIOR_MARGIN = 1
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MIN_PIXELS_PER_COLOR, MAX_PIXELS_PER_COLOR = 2, 8
CONNECTIVITY, MAX_GENERATION_ATTEMPTS = 4, 300


def _gen_valid_dims(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in OBJECT_HEIGHTS:
        for w in OBJECT_WIDTHS:
            density = area / (h * w)
            if DENSITY_MIN <= density <= DENSITY_MAX and h + w - 1 <= area:
                dims.append((h, w))
    return dims


def _gen_spanning_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bend_budget = max(0, target_area - (h + w - 1))

    if random.random() < VERTICAL_SPINE_PROB:
        c = random.randrange(w)
        mask[0, c] = True
        bends_used = 0
        for r in range(1, h):
            prev_c = c
            choices = [0]
            if bends_used < bend_budget:
                if c > 0:
                    choices.append(-1)
                if c + 1 < w:
                    choices.append(1)
            step = random.choice(choices)
            c = prev_c + step
            mask[r, prev_c] = True
            mask[r, c] = True
            bends_used += int(step != 0)
        mask[random.randrange(INTERIOR_MARGIN, h - INTERIOR_MARGIN), :] = True
    else:
        r = random.randrange(INTERIOR_MARGIN, h - INTERIOR_MARGIN)
        mask[r, 0] = True
        bends_used = 0
        for c in range(1, w):
            prev_r = r
            choices = [0]
            if bends_used < bend_budget:
                if r > 1:
                    choices.append(-1)
                if r + 2 < h:
                    choices.append(1)
            step = random.choice(choices)
            r = prev_r + step
            mask[prev_r, c] = True
            mask[r, c] = True
            bends_used += int(step != 0)
        mask[:, random.randrange(INTERIOR_MARGIN, w - INTERIOR_MARGIN)] = True

    while int(mask.sum()) < target_area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    return mask


def _gen_color_counts(total: int) -> list[int]:
    counts = [MIN_PIXELS_PER_COLOR] * LEGEND_CELLS
    remaining = total - sum(counts)
    while remaining > 0:
        choices = [i for i, count in enumerate(counts) if count < MAX_PIXELS_PER_COLOR]
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1
    return counts


def _gen_colored_patch(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    values = []
    for color, count in zip(colors, _gen_color_counts(len(coords))):
        values.extend([color] * count)
    random.shuffle(values)

    patch = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, values):
        patch[r, c] = color
    return patch


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    legend = grid[:LEGEND_ROWS, :LEGEND_COLS]
    target_mask = grid != BACKGROUND_COLOR
    target_mask[:LEGEND_ROWS, :LEGEND_COLS] = False
    for r in range(LEGEND_ROWS):
        left = int(legend[r, 0])
        right = int(legend[r, 1])
        out[(grid == left) & target_mask] = right
        out[(grid == right) & target_mask] = left
    return out


def _gen_object_mask() -> np.ndarray | None:
    area = random.randint(OBJECT_AREA_MIN, OBJECT_AREA_MAX)
    dims = _gen_valid_dims(area)
    if not dims:
        return None
    h, w = random.choice(dims)

    mask = _gen_spanning_mask(h, w, area)
    if int(mask.sum()) != area:
        return None
    if bbox(mask) != (0, 0, h - 1, w - 1):
        return None
    if not is_contiguous(mask, background=BACKGROUND_COLOR, connectivity=CONNECTIVITY):
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(COLOR_POOL, LEGEND_CELLS)
    legend = np.array(colors, dtype=int).reshape(LEGEND_SHAPE)

    mask = _gen_object_mask()
    if mask is None:
        return None

    patch = _gen_colored_patch(mask, colors)
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:LEGEND_ROWS, :LEGEND_COLS] = legend
    max_col = GRID_SHAPE[1] - OBJECT_RIGHT_PADDING - patch.shape[1]
    c0 = random.randint(OBJECT_MIN_COL, max_col)
    grid = paste(grid, patch, OBJECT_ROW0, c0, transparent=BACKGROUND_COLOR)

    expected = _gen_expected_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0becf7df input")
