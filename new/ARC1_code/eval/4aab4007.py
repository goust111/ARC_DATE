from collections import Counter

import numpy as np
import random
from functools import lru_cache


def _unique_known_color(values, label):
    counts = Counter(int(v) for v in values if int(v) != 0)
    assert counts, f"no known color for {label}"
    assert len(counts) == 1, f"conflicting colors for {label}: {counts}"
    return next(iter(counts))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the black occlusions by continuing the visible framed anti-diagonal color table.

    Input structure
    A 28x28 color table has black pixels masking missing values.  The first two
    rows/columns are one constant outer color, row/column 2 form one constant
    inner frame color, and the remaining area is a color field whose entries are
    constant on anti-diagonals, except for the visible exceptional cell at (4,4).

    Rule
    Replace each black pixel in the outer frame with the outer-frame color, each
    black pixel on row/column 2 with the inner-frame color, and each other black
    pixel with the unique non-black color already visible on the same
    anti-diagonal of the main field, ignoring cell (4,4).

    Key insight
    The black pixels are only holes; enough of every affected structural region
    and anti-diagonal remains visible to determine its missing color uniquely.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 6 and W >= 6
    assert np.any(g == 0)

    rows, cols = np.indices(g.shape)
    outer_mask = (rows < 2) | (cols < 2)
    inner_frame_mask = ((rows == 2) & (cols >= 2)) | ((cols == 2) & (rows >= 2))

    outer_color = _unique_known_color(g[outer_mask], "outer frame")
    inner_frame_color = _unique_known_color(g[inner_frame_mask], "inner frame")
    assert g[4, 4] != 0

    diagonal_colors = {}
    for s in range(6, H + W - 1):
        values = []
        for r in range(3, H):
            c = s - r
            if 3 <= c < W and (r, c) != (4, 4):
                values.append(g[r, c])
        if any(int(v) != 0 for v in values):
            diagonal_colors[s] = _unique_known_color(values, f"anti-diagonal {s}")

    out = g.copy()
    for r, c in np.argwhere(g == 0):
        if r < 2 or c < 2:
            out[r, c] = outer_color
        elif r == 2 or c == 2:
            out[r, c] = inner_frame_color
        else:
            assert (r, c) != (4, 4)
            s = int(r + c)
            assert s in diagonal_colors, f"missing color for anti-diagonal {s}"
            out[r, c] = diagonal_colors[s]

    return out

# --------------- generator (o5x) ---------------

import itertools

# Canvas structure and fixed colors.
GRID_SIZE = 28
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
OUTER_WIDTH, INNER_FRAME_INDEX, MAIN_START = 2, 2, 3
OUTER_FRAME_COLOR, INNER_FRAME_COLOR = 1, 4
EXCEPTION_CELL = (4, 4)

# Anti-diagonal color field.
PREAMBLE_COLORS = {6: 3, 7: 2, 8: 1}
PERIODIC_SUM_START, PERIODIC_SUM_STOP = 9, 55
PALETTE_PERIOD_CHOICES = (4, 5, 7, 9)
PALETTE_STEP_ONE_PERIODS, PALETTE_STEP_TWO_PERIODS = (5, 7, 9), (4, 5)
PALETTE_CONFIG_STOP_BASE = 11
STEP_ONE_SEQUENCE_JUMP, STEP_TWO_SEQUENCE_JUMP = 2, 1

# Mask blob area, count, and rectangle bounds.
MASK_AREA_RANGE = (82, 126)
BLOB_COUNT_CHOICES = (2, 3)
RECT_HEIGHTS, RECT_WIDTHS = range(3, 13), range(4, 10)
MAX_FRAME_SPANNING_RECTS, PLACEMENT_CLEARANCE = 2, 1

# Rejection sampling budget.
GENERATION_RETRIES = 300


_ROWS, _COLS = np.indices(GRID_SHAPE)
_OUTER_FRAME_MASK = (_ROWS < OUTER_WIDTH) | (_COLS < OUTER_WIDTH)
_INNER_FRAME_MASK = (
    ((_ROWS == INNER_FRAME_INDEX) & (_COLS >= INNER_FRAME_INDEX))
    | ((_COLS == INNER_FRAME_INDEX) & (_ROWS >= INNER_FRAME_INDEX))
)
_MAIN_FIELD = (slice(MAIN_START, GRID_SIZE), slice(MAIN_START, GRID_SIZE))
_MAIN_SUMS = _ROWS[_MAIN_FIELD] + _COLS[_MAIN_FIELD]


def _gen_palette_and_sequence():
    k = random.choice(PALETTE_PERIOD_CHOICES)
    configs = []
    if k in PALETTE_STEP_ONE_PERIODS:
        configs.extend((1, start) for start in range(1, PALETTE_CONFIG_STOP_BASE - k))
    if k in PALETTE_STEP_TWO_PERIODS:
        configs.extend(
            (2, start)
            for start in range(1, PALETTE_CONFIG_STOP_BASE - 2 * (k - 1))
        )
    step, start = random.choice(configs)
    palette = list(range(start, start + step * k, step))
    start_i = random.randrange(k)
    jump = STEP_ONE_SEQUENCE_JUMP if step == 1 else STEP_TWO_SEQUENCE_JUMP
    seq = [palette[(start_i + jump * i) % k] for i in range(k)]
    return palette, seq


def _gen_output_grid():
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[_OUTER_FRAME_MASK] = OUTER_FRAME_COLOR
    grid[_INNER_FRAME_MASK] = INNER_FRAME_COLOR

    palette, seq = _gen_palette_and_sequence()
    color_lookup = np.zeros(PERIODIC_SUM_STOP, dtype=int)
    for total, color in PREAMBLE_COLORS.items():
        color_lookup[total] = color
    for total in range(PERIODIC_SUM_START, PERIODIC_SUM_STOP):
        color_lookup[total] = seq[(total - PERIODIC_SUM_START) % len(seq)]
    grid[_MAIN_FIELD] = color_lookup[_MAIN_SUMS]

    choices = [color for color in palette if color != PREAMBLE_COLORS[8]]
    grid[EXCEPTION_CELL] = random.choice(choices)
    return grid


@lru_cache(maxsize=None)
def _gen_rect_combos(blob_count, target_area):
    rects = tuple((h, w, h * w) for h in RECT_HEIGHTS for w in RECT_WIDTHS)
    combos = []
    for candidate in itertools.product(rects, repeat=blob_count):
        if sum(area for _, _, area in candidate) == target_area:
            combos.append(tuple((h, w) for h, w, _ in candidate))
    return tuple(combos)


def _gen_rect_spans_frame(r0, c0, h, w):
    touches_frame = r0 < MAIN_START or c0 < MAIN_START
    touches_main = r0 + h > MAIN_START and c0 + w > MAIN_START
    return touches_frame and touches_main


def _gen_constraints_ok(mask):
    if mask[EXCEPTION_CELL]:
        return False

    if np.all(mask[_OUTER_FRAME_MASK]) or np.all(mask[_INNER_FRAME_MASK]):
        return False

    for total in range(min(PREAMBLE_COLORS), PERIODIC_SUM_STOP):
        coords = [
            (r, total - r)
            for r in range(MAIN_START, GRID_SIZE)
            if (
                MAIN_START <= total - r < GRID_SIZE
                and (r, total - r) != EXCEPTION_CELL
            )
        ]
        if coords and all(mask[r, c] for r, c in coords):
            return False
    return True


def _gen_touches_existing(mask, r0, c0, h, w):
    r1 = min(GRID_SIZE, r0 + h + PLACEMENT_CLEARANCE)
    c1 = min(GRID_SIZE, c0 + w + PLACEMENT_CLEARANCE)
    r0 = max(0, r0 - PLACEMENT_CLEARANCE)
    c0 = max(0, c0 - PLACEMENT_CLEARANCE)
    return np.any(mask[r0:r1, c0:c1])


def _gen_place_rect(mask, h, w, want_frame_overlap):
    candidates = []
    for r0 in range(GRID_SIZE - h + 1):
        for c0 in range(GRID_SIZE - w + 1):
            if _gen_rect_spans_frame(r0, c0, h, w) != want_frame_overlap:
                continue
            if _gen_touches_existing(mask, r0, c0, h, w):
                continue
            trial = mask.copy()
            trial[r0:r0 + h, c0:c0 + w] = True
            if _gen_constraints_ok(trial):
                candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    mask[r0:r0 + h, c0:c0 + w] = True
    return True


def _gen_mask():
    target_area = random.randint(*MASK_AREA_RANGE)
    blob_count = random.choice(BLOB_COUNT_CHOICES)
    combos = _gen_rect_combos(blob_count, target_area)
    if not combos:
        return None

    rects = list(random.choice(combos))
    frame_count = random.randint(0, min(MAX_FRAME_SPANNING_RECTS, blob_count))
    frame_flags = [True] * frame_count + [False] * (blob_count - frame_count)
    random.shuffle(frame_flags)
    items = sorted(zip(rects, frame_flags), key=lambda item: item[0][0] * item[0][1], reverse=True)

    mask = np.zeros(GRID_SHAPE, dtype=bool)
    for (h, w), want_frame in items:
        if not _gen_place_rect(mask, h, w, want_frame):
            return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    output = _gen_output_grid()
    mask = _gen_mask()
    if mask is None:
        return None

    inp = output.copy()
    inp[mask] = 0
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 4aab4007")
