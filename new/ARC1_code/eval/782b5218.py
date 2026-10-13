import numpy as np

from arc_utils import Color, colors_present, components_mask
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Use the red divider as a wall: fill the 4-connected lower-left side with the single payload color and clear the other side to black.

    Input structure
    - The grid contains black cells, red divider pixels, and exactly one other payload color.
    - Treating red as blocked, all non-red pixels form exactly two 4-connected regions.
    - The lower-left corner is part of the region to fill.

    Rule
    - Preserve every red pixel.
    - Fill every non-red pixel in the same 4-connected region as the lower-left corner with the payload color.
    - Recolor the other non-red region to black.

    Key insight
    The noisy black/payload pattern in each region is irrelevant; the red wall partitions the canvas, and the side anchored at the lower-left corner is the solid foreground side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    payload_colors = colors_present(g, exclude={Color.BLACK, Color.RED})
    assert len(payload_colors) == 1
    payload = payload_colors[0]
    assert g[h - 1, 0] != Color.RED

    regions = components_mask(g != Color.RED, connectivity=4)
    assert len(regions) == 2

    lower_left = np.array([h - 1, 0])
    fill_regions = [coords for coords in regions if np.any(np.all(coords == lower_left, axis=1))]
    assert len(fill_regions) == 1
    fill_region = fill_regions[0]

    out = np.full_like(g, Color.BLACK)
    out[g == Color.RED] = Color.RED
    out[fill_region[:, 0], fill_region[:, 1]] = payload
    return out

# --------------- generator (o5x) ---------------

# Canvas anchors and size.
GRID_SIZE = 10
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
LAST_IDX = GRID_SIZE - 1
LOWER_LEFT, TOP_RIGHT = (LAST_IDX, 0), (0, LAST_IDX)

# Divider constraints.
WALL_COUNT_MIN, WALL_COUNT_MAX = 10, 17
MAX_RED_PER_COLUMN = 2
MIN_WALL_ROW = 1
SMALLER_FRACTION_MIN, SMALLER_FRACTION_MAX = 0.33, 0.49

# Left-to-right divider sampling.
LR_FLAT_PROB = 0.30
LR_FLAT_ROWS = (3, 4, 5, 6)
LR_FLAT_SPANS = (1, 2, 3)
LR_WALK_SPANS = (2, 3)
LR_WALK_LOW_ROWS = {2: (3, 4, 5), 3: (3, 4)}
MAX_ROW_STEP = 1

# Diagonal divider sampling.
DIAGONAL_PROB = 0.35
DIAGONAL_LONG_PROB = 0.80
DIAGONAL_START_ROW = 1
DIAGONAL_LONG_END_COL, DIAGONAL_SHORT_END_COL = 8, 7

# Region noise and retry controls.
NOISE_DENSITY_MIN, NOISE_DENSITY_MAX = 0.40, 0.59
MAX_ATTEMPTS = 300
PAYLOAD_COLORS = tuple(c for c in Color.ALL_COLORS if c not in (Color.BLACK, Color.RED))


def _gen_lr_wall() -> np.ndarray:
    red = np.zeros(GRID_SHAPE, dtype=bool)

    if random.random() < LR_FLAT_PROB:
        row = random.choice(LR_FLAT_ROWS)
        rows = [row] * GRID_SIZE
        span = random.choice(LR_FLAT_SPANS)
        low = min(max(row - random.randint(0, span - 1), MIN_WALL_ROW), LAST_IDX - span)
        allowed_rows = list(range(low, low + span))
    else:
        span = random.choice(LR_WALK_SPANS)
        low = random.choice(LR_WALK_LOW_ROWS[span])
        allowed_rows = list(range(low, low + span))
        rows = [random.choice(allowed_rows)]
        for _ in range(GRID_SIZE - 1):
            choices = [r for r in allowed_rows if abs(r - rows[-1]) <= MAX_ROW_STEP]
            rows.append(random.choice(choices))

    coords = {(r, c) for c, r in enumerate(rows)}
    candidates = []
    for c, r in enumerate(rows):
        for rr in allowed_rows:
            if rr != r and abs(rr - r) == MAX_ROW_STEP:
                candidates.append((rr, c))

    random.shuffle(candidates)
    target_count = random.randint(WALL_COUNT_MIN, WALL_COUNT_MAX)
    for rr, c in candidates:
        if len(coords) >= target_count:
            break
        if sum(1 for _, cc in coords if cc == c) < MAX_RED_PER_COLUMN:
            coords.add((rr, c))

    rr, cc = zip(*coords)
    red[rr, cc] = True
    return red


def _gen_diagonal_wall() -> np.ndarray:
    end_col = DIAGONAL_LONG_END_COL if random.random() < DIAGONAL_LONG_PROB else DIAGONAL_SHORT_END_COL
    down_steps = LAST_IDX - DIAGONAL_START_ROW
    down_gaps = set(random.sample(range(end_col + 1), down_steps))

    red = np.zeros(GRID_SHAPE, dtype=bool)
    r, c = DIAGONAL_START_ROW, LOWER_LEFT[1]
    red[r, c] = True
    for gap in range(end_col + 1):
        if gap in down_gaps:
            r += 1
            red[r, c] = True
        if gap < end_col:
            c += 1
            red[r, c] = True
    return red


def _region_containing(regions: list[np.ndarray], point: tuple[int, int]) -> np.ndarray | None:
    target = np.array(point)
    matches = [coords for coords in regions if np.any(np.all(coords == target, axis=1))]
    return matches[0] if len(matches) == 1 else None


def _gen_regions_from_red(red: np.ndarray):
    if red[LOWER_LEFT] or red[TOP_RIGHT]:
        return None

    red_count = int(red.sum())
    if not WALL_COUNT_MIN <= red_count <= WALL_COUNT_MAX:
        return None
    if any(int(red[:, c].sum()) > MAX_RED_PER_COLUMN for c in range(GRID_SIZE)):
        return None

    touches_left = bool(red[:, LOWER_LEFT[1]].any())
    touches_right = bool(red[:, LAST_IDX].any())
    touches_bottom = bool(red[LAST_IDX, :].any())
    if not ((touches_left and touches_right) or (touches_left and touches_bottom)):
        return None

    if len(components_mask(red, connectivity=8)) != 1:
        return None

    regions = components_mask(~red, connectivity=4)
    if len(regions) != 2:
        return None

    fill = _region_containing(regions, LOWER_LEFT)
    clear = _region_containing(regions, TOP_RIGHT)
    if fill is None or clear is None or fill is clear:
        return None

    non_red = red.size - red_count
    smaller_fraction = min(len(fill), len(clear)) / non_red
    if not SMALLER_FRACTION_MIN <= smaller_fraction <= SMALLER_FRACTION_MAX:
        return None

    return fill, clear


def _paint_noisy_region(inp: np.ndarray, coords: np.ndarray, payload: int) -> None:
    n = len(coords)
    lo = max(1, math.ceil(NOISE_DENSITY_MIN * n))
    hi = min(n - 1, math.floor(NOISE_DENSITY_MAX * n))
    chosen = coords[random.sample(range(n), random.randint(lo, hi))]
    inp[coords[:, 0], coords[:, 1]] = Color.BLACK
    inp[chosen[:, 0], chosen[:, 1]] = payload


def _gen_attempt() -> np.ndarray | None:
    red = _gen_diagonal_wall() if random.random() < DIAGONAL_PROB else _gen_lr_wall()
    regions = _gen_regions_from_red(red)
    if regions is None:
        return None
    fill_region, clear_region = regions

    payload = random.choice(PAYLOAD_COLORS)
    out = np.full(GRID_SHAPE, Color.BLACK, dtype=int)
    out[red] = Color.RED
    out[fill_region[:, 0], fill_region[:, 1]] = payload

    inp = out.copy()
    _paint_noisy_region(inp, fill_region, payload)
    _paint_noisy_region(inp, clear_region, payload)

    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 782b5218 input")
