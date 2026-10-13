from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, paste
import random


Frame = tuple[int, int, int, int]


def _find_rectangular_frames(g: np.ndarray, bg: int, frame_color: int) -> list[Frame]:
    height = g.shape[0]
    frames: list[Frame] = []

    for r0 in range(height - 2):
        for r1 in range(r0 + 2, height):
            shared_frame_cols = np.flatnonzero((g[r0] == frame_color) & (g[r1] == frame_color))
            for i, c0_raw in enumerate(shared_frame_cols[:-1]):
                c0 = int(c0_raw)
                for c1_raw in shared_frame_cols[i + 1 :]:
                    c1 = int(c1_raw)
                    if c1 - c0 < 2:
                        continue
                    if not bool(np.all(g[r0, c0 : c1 + 1] == frame_color)):
                        continue
                    if not bool(np.all(g[r1, c0 : c1 + 1] == frame_color)):
                        continue
                    if not bool(np.all(g[r0 : r1 + 1, c0] == frame_color)):
                        continue
                    if not bool(np.all(g[r0 : r1 + 1, c1] == frame_color)):
                        continue
                    if not bool(np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg)):
                        continue
                    frames.append((r0, c0, r1, c1))

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow blue rectangular frame with red for even interior area and orange for odd interior area.

    Input structure
    - A black background contains one or more one-cell-thick blue rectangular frames.
    - Each frame encloses a black rectangular interior.
    - Frames may be separate or touch along their outside borders.

    Rule
    - Detect every blue rectangular border in the original input.
    - Compute each frame's interior area.
    - Fill the interior red when that area is even, or orange when it is odd.
    - Leave the blue borders and all exterior background unchanged.

    Key insight
    - The fill color encodes the parity of the empty rectangular interior, so
      frames must be detected as rectangles rather than foreground components.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground_colors = colors_present(g, exclude={bg})
    assert foreground_colors == [Color.BLUE], "expected blue rectangular frames only"

    frames = _find_rectangular_frames(g, bg, Color.BLUE)
    assert frames, "expected at least one hollow rectangular frame"
    frame_pixels = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        frame_pixels[r0, c0 : c1 + 1] = True
        frame_pixels[r1, c0 : c1 + 1] = True
        frame_pixels[r0 : r1 + 1, c0] = True
        frame_pixels[r0 : r1 + 1, c1] = True
    assert bool(np.array_equal(frame_pixels, g == Color.BLUE)), "all blue pixels must belong to frame borders"

    out = g.copy()
    filled = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
        assert not bool(np.any(filled[interior])), "frame interiors must not overlap"
        filled[interior] = True

        interior_area = (r1 - r0 - 1) * (c1 - c0 - 1)
        fill_color = Color.ORANGE if interior_area % 2 else Color.RED
        out[interior] = fill_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and frame bounds.
GRID_SIDE_MIN, GRID_SIDE_MAX = 10, 20
FRAME_COUNT_MIN, FRAME_COUNT_MAX, FRAME_COUNT_MAX_AT_MIN_GRID = 2, 5, 4
FRAME_SIDE_MIN_ODD, FRAME_SIDE_MIN_EVEN, FRAME_SIDE_MAX = 3, 4, 10

# Layout spacing and weighting.
FRAME_GAP = 1
ROW_WEIGHT_FLOOR, ROW_WEIGHT_MARGIN, ROW_WEIGHT_POWER = 0.25, 2, 2
SIDE_WEIGHT_POWER, ROW_GROW_MIN_DIVISOR = 2, 2

# Final random orientation and retry budget.
ROTATION_TURNS_MIN, ROTATION_TURNS_MAX = 0, 3
FLIP_OPTIONS = (False, True)
GENERATION_RETRIES = 300

def _gen_frame_patch(side: int) -> np.ndarray:
    patch = np.zeros((side, side), dtype=int)
    patch[0, :] = Color.BLUE
    patch[-1, :] = Color.BLUE
    patch[:, 0] = Color.BLUE
    patch[:, -1] = Color.BLUE
    return patch

def _gen_expected_output(inp: np.ndarray, frames: list[tuple[int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for r0, c0, side in frames:
        fill = Color.ORANGE if side % 2 else Color.RED
        out[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = fill
    return out

def _gen_split_slack(extra: int, slots: int) -> list[int]:
    cuts = sorted(random.randint(0, extra) for _ in range(slots - 1))
    parts = []
    prev = 0
    for cut in cuts + [extra]:
        parts.append(cut - prev)
        prev = cut
    random.shuffle(parts)
    return parts

def _gen_weighted_choice(items: list[int], weights: list[float]) -> int:
    total = sum(weights)
    pick = random.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += weight
        if pick <= upto:
            return item
    return items[-1]

def _gen_min_side(parity: int) -> int:
    return FRAME_SIDE_MIN_EVEN if parity == 0 else FRAME_SIDE_MIN_ODD

def _gen_row_counts(n: int, k: int) -> list[int] | None:
    min_slot_span = FRAME_SIDE_MIN_ODD + FRAME_GAP
    max_per_row = (n + FRAME_GAP) // min_slot_span
    max_rows = min(k, (n + FRAME_GAP) // min_slot_span)
    feasible_rows = [r for r in range(1, max_rows + 1) if k <= r * max_per_row]
    if not feasible_rows:
        return None

    weights = []
    for rows in feasible_rows:
        max_frames_in_row = (k + rows - 1) // rows
        vertical_avg = (n - FRAME_GAP * (rows - 1)) / rows
        horizontal_avg = (n - FRAME_GAP * (max_frames_in_row - 1)) / max_frames_in_row
        weights.append(
            max(ROW_WEIGHT_FLOOR, min(vertical_avg, horizontal_avg) - ROW_WEIGHT_MARGIN) ** ROW_WEIGHT_POWER
        )

    row_count = _gen_weighted_choice(feasible_rows, weights)
    base = k // row_count
    extra = k % row_count
    counts = [base + 1] * extra + [base] * (row_count - extra)
    random.shuffle(counts)
    return counts

def _gen_choose_row_heights(n: int, row_parities: list[list[int]]) -> list[int] | None:
    row_mins = [max(_gen_min_side(parity) for parity in parities) for parities in row_parities]
    extra = n - FRAME_GAP * (len(row_mins) - 1) - sum(row_mins)
    if extra < 0:
        return None

    heights = row_mins[:]
    min_grow = (extra + ROW_GROW_MIN_DIVISOR - 1) // ROW_GROW_MIN_DIVISOR
    grow_budget = random.randint(min_grow, extra) if extra else 0
    while grow_budget:
        choices = [i for i, height in enumerate(heights) if height < FRAME_SIDE_MAX]
        if not choices:
            break
        row = random.choice(choices)
        heights[row] += 1
        grow_budget -= 1
    return heights

def _gen_choose_sides(n: int, height: int, parities: list[int]) -> list[int] | None:
    remaining = n - FRAME_GAP * (len(parities) - 1)
    sides = []
    for i, parity in enumerate(parities):
        min_side = _gen_min_side(parity)
        future_min = sum(_gen_min_side(p) for p in parities[i + 1 :])
        max_side = min(FRAME_SIDE_MAX, height, remaining - future_min)
        candidates = [side for side in range(min_side, max_side + 1) if side % 2 == parity]
        if not candidates:
            return None
        side = _gen_weighted_choice(candidates, [side**SIDE_WEIGHT_POWER for side in candidates])
        sides.append(side)
        remaining -= side
    random.shuffle(sides)
    return sides

def _gen_layout(n: int, k: int) -> tuple[list[int], list[list[int]]] | None:
    """Choose row heights and per-row frame parities without placing pixels."""
    parity_plan = [0, 1]
    parity_plan.extend(random.randint(0, 1) for _ in range(k - 2))
    random.shuffle(parity_plan)

    row_counts = _gen_row_counts(n, k)
    if row_counts is None:
        return None

    row_parities = []
    cursor = 0
    for count in row_counts:
        row_parities.append(parity_plan[cursor : cursor + count])
        cursor += count
    random.shuffle(row_parities)

    row_heights = _gen_choose_row_heights(n, row_parities)
    if row_heights is None:
        return None
    return row_heights, row_parities

def _gen_place_frames(
    n: int, row_heights: list[int], row_parities: list[list[int]]
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Render non-overlapping square frames from the row layout."""
    inp = np.zeros((n, n), dtype=int)
    frames: list[tuple[int, int, int]] = []

    used_height = sum(row_heights) + FRAME_GAP * (len(row_heights) - 1)
    v_slack = _gen_split_slack(n - used_height, len(row_heights) + 1)
    r = v_slack[0]
    for row_i, (height, parities) in enumerate(zip(row_heights, row_parities)):
        sides = _gen_choose_sides(n, height, parities)
        if sides is None:
            return None

        used_width = sum(sides) + FRAME_GAP * (len(sides) - 1)
        h_slack = _gen_split_slack(n - used_width, len(sides) + 1)
        c = h_slack[0]
        for side_i, side in enumerate(sides):
            r0 = r + random.randint(0, height - side)
            paste(inp, _gen_frame_patch(side), r0, c, transparent=None)
            frames.append((r0, c, side))
            c += side + FRAME_GAP + h_slack[side_i + 1]
        r += height + FRAME_GAP + v_slack[row_i + 1]
    return inp, frames

def _gen_random_orientation(inp: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    turns = random.randint(ROTATION_TURNS_MIN, ROTATION_TURNS_MAX)
    if turns:
        inp = np.rot90(inp, turns)
        expected = np.rot90(expected, turns)
    if random.choice(FLIP_OPTIONS):
        inp = np.fliplr(inp)
        expected = np.fliplr(expected)
    return inp, expected

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    max_k = FRAME_COUNT_MAX_AT_MIN_GRID if n == GRID_SIDE_MIN else FRAME_COUNT_MAX
    k = random.randint(FRAME_COUNT_MIN, max_k)

    layout = _gen_layout(n, k)
    if layout is None:
        return None

    placed = _gen_place_frames(n, *layout)
    if placed is None:
        return None
    inp, frames = placed
    expected = _gen_expected_output(inp, frames)
    inp, expected = _gen_random_orientation(inp, expected)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
