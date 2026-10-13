from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERN_FROM_POINT: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((0, 0), (-1, -1), (-1, 1)),
    "down": ((0, 0), (1, -1), (1, 1)),
    "left": ((0, 0), (-1, -1), (1, -1)),
    "right": ((0, 0), (-1, 1), (1, 1)),
}

ARROW_PATTERN_FROM_BBOX: dict[frozenset[tuple[int, int]], tuple[str, tuple[int, int]]] = {
    frozenset({(1, 1), (0, 0), (0, 2)}): ("up", (1, 1)),
    frozenset({(0, 1), (1, 0), (1, 2)}): ("down", (0, 1)),
    frozenset({(1, 1), (0, 0), (2, 0)}): ("left", (1, 1)),
    frozenset({(1, 0), (0, 1), (2, 1)}): ("right", (1, 0)),
}

SCENARIOS = ("free", "absorb", "stop", "absorb_stop")
MIN_STEPS = {
    "free": 3,
    "absorb": 4,
    "stop": 3,
    "absorb_stop": 5,
}

GEN_RETRIES = 300
GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_RANGE = (3, 4)
STRIPE_LENGTHS = (2, 3, 4)
LANE_GAP = 1
OUTER_MARGIN = 1


@dataclass(frozen=True)
class ArrowSpec:
    direction: str
    point: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    stripe: tuple[tuple[int, int, int], ...]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow's attached color strip forward, absorbing same-color obstacles and stopping on mismatched ones.

    Input structure:
    - The background is the dominant color.
    - One shared arrow color forms several 3-pixel V-shaped arrowheads; the open side of each V gives one of the four cardinal directions.
    - Directly behind each arrow tip there is a 1-pixel-wide strip, perpendicular to the movement direction and 2-4 cells long, whose cells may have different colors.
    - All remaining non-background cells are isolated obstacle pixels.

    Rule:
    - Erase every arrow.
    - For each arrow, move its attached strip one step at a time along the arrow direction while keeping each surviving strip cell on its own parallel track.
    - If a strip cell would enter a same-color obstacle, remove both that strip cell and the obstacle; the rest of the strip continues moving.
    - If any surviving strip cell would enter a different-color obstacle or leave the grid, stop that strip before the step. Keep all unabsorbed obstacles.

    Key insight:
    - The strip behaves like several synchronized pixels moving in parallel: matching obstacles delete only the corresponding track, while a mismatched obstacle blocks the whole strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, arrows = _parse_scene(g)
    return _solve_scene(g, bg, arrows)


def _solve_scene(grid: np.ndarray, bg: int, arrows: list[ArrowSpec]) -> np.ndarray:
    board = np.array(grid, dtype=int, copy=True)
    for arrow in arrows:
        for r, c in arrow.cells:
            board[r, c] = bg
        for r, c, _color in arrow.stripe:
            board[r, c] = bg

    settled: list[tuple[int, int, int]] = []
    for arrow in arrows:
        settled.extend(_advance_strip(board, bg, arrow.stripe, arrow.direction))

    out = board.copy()
    for r, c, color in settled:
        assert out[r, c] == bg, "settled strips must not overlap"
        out[r, c] = color
    return out


def _advance_strip(
    board: np.ndarray,
    bg: int,
    strip: tuple[tuple[int, int, int], ...],
    direction: str,
) -> list[tuple[int, int, int]]:
    dr, dc = DIRS[direction]
    active = list(strip)
    h, w = board.shape

    while active:
        moved: list[tuple[int, int, int]] = []
        absorbed: list[tuple[int, int]] = []
        blocked = False

        for r, c, color in active:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                blocked = True
                break
            cell = int(board[rr, cc])
            if cell == bg:
                moved.append((rr, cc, color))
            elif cell == color:
                absorbed.append((rr, cc))
            else:
                blocked = True
                break

        if blocked:
            break

        for rr, cc in absorbed:
            board[rr, cc] = bg
        active = moved

    return active


def _parse_scene(grid: np.ndarray) -> tuple[int, list[ArrowSpec]]:
    bg = _get_bg(grid)
    arrows_raw = _find_arrows(grid, bg)
    assert arrows_raw, "expected at least one arrow"

    arrow_color_set = {color for color, _direction, _point, _cells in arrows_raw}
    assert len(arrow_color_set) == 1, "all arrows must share one color"

    arrow_cells = {
        (r, c)
        for _color, _direction, _point, cells in arrows_raw
        for r, c in cells
    }

    stripes: list[ArrowSpec] = []
    stripe_cells_seen: set[tuple[int, int]] = set()
    for _color, direction, point, cells in arrows_raw:
        stripe = _extract_strip(grid, bg, arrow_cells, point, direction)
        stripe_coords = {(r, c) for r, c, _cell_color in stripe}
        assert not (stripe_coords & stripe_cells_seen), "strips must be disjoint"
        stripe_cells_seen |= stripe_coords
        stripes.append(ArrowSpec(direction=direction, point=point, cells=cells, stripe=stripe))

    leftover = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) != bg and (r, c) not in arrow_cells and (r, c) not in stripe_cells_seen
    }
    for r, c in leftover:
        for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
                if (rr, cc) == (r, c):
                    continue
                assert (rr, cc) not in leftover, "obstacles must remain isolated single pixels"

    return bg, stripes


def _extract_strip(
    grid: np.ndarray,
    bg: int,
    arrow_cells: set[tuple[int, int]],
    point: tuple[int, int],
    direction: str,
) -> tuple[tuple[int, int, int], ...]:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    anchor_r = point[0] - dr
    anchor_c = point[1] - dc
    assert 0 <= anchor_r < h and 0 <= anchor_c < w, "strip anchor must stay in bounds"
    assert (anchor_r, anchor_c) not in arrow_cells, "strip anchor cannot overlap the arrow"
    assert int(grid[anchor_r, anchor_c]) != bg, "missing strip behind arrow"

    cells: list[tuple[int, int, int]] = []
    if dr == 0:
        top = anchor_r
        while top - 1 >= 0 and int(grid[top - 1, anchor_c]) != bg and (top - 1, anchor_c) not in arrow_cells:
            top -= 1
        bottom = anchor_r
        while bottom + 1 < h and int(grid[bottom + 1, anchor_c]) != bg and (bottom + 1, anchor_c) not in arrow_cells:
            bottom += 1
        assert top <= point[0] <= bottom, "arrow tip row must line up with its strip"
        cells = [(r, anchor_c, int(grid[r, anchor_c])) for r in range(top, bottom + 1)]
    else:
        left = anchor_c
        while left - 1 >= 0 and int(grid[anchor_r, left - 1]) != bg and (anchor_r, left - 1) not in arrow_cells:
            left -= 1
        right = anchor_c
        while right + 1 < w and int(grid[anchor_r, right + 1]) != bg and (anchor_r, right + 1) not in arrow_cells:
            right += 1
        assert left <= point[1] <= right, "arrow tip column must line up with its strip"
        cells = [(anchor_r, c, int(grid[anchor_r, c])) for c in range(left, right + 1)]

    assert 2 <= len(cells) <= 4, "strip length must be between 2 and 4"
    return tuple(cells)


def _find_arrows(grid: np.ndarray, bg: int) -> list[tuple[int, str, tuple[int, int], tuple[tuple[int, int], ...]]]:
    arrows: list[tuple[int, str, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for color, coords in _monochrome_components(grid, bg):
        if len(coords) != 3:
            continue
        r0 = min(r for r, _c in coords)
        c0 = min(c for _r, c in coords)
        rel = frozenset((r - r0, c - c0) for r, c in coords)
        if rel not in ARROW_PATTERN_FROM_BBOX:
            continue
        direction, point_rel = ARROW_PATTERN_FROM_BBOX[rel]
        point = (r0 + point_rel[0], c0 + point_rel[1])
        arrows.append((color, direction, point, tuple(sorted(coords))))
    return arrows


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            components.append((color, coords))

    return components


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _arrow_cells(point: tuple[int, int], direction: str) -> tuple[tuple[int, int], ...]:
    pr, pc = point
    return tuple(sorted((pr + dr, pc + dc) for dr, dc in ARROW_PATTERN_FROM_POINT[direction]))


def _sample_intervals(total: int, sizes: list[int], rng: Generator) -> list[int] | None:
    required = sum(sizes) + LANE_GAP * (len(sizes) - 1) + 2 * OUTER_MARGIN
    if required > total:
        return None
    extra = total - required
    slots = rng.multinomial(extra, [1.0 / (len(sizes) + 1)] * (len(sizes) + 1))

    starts: list[int] = []
    pos = OUTER_MARGIN + int(slots[0])
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx < len(sizes) - 1:
            pos += LANE_GAP + int(slots[idx + 1])
    return starts


def _scenario_list(count: int, rng: Generator) -> list[str]:
    scenarios = list(SCENARIOS)
    rng.shuffle(scenarios)
    return scenarios[:count]


def _direction_list(axis: str, count: int, rng: Generator) -> list[str]:
    if axis == "horizontal":
        directions = ["left", "right"] * ((count + 1) // 2)
    else:
        directions = ["up", "down"] * ((count + 1) // 2)
    rng.shuffle(directions)
    return directions[:count]


def _build_horizontal_object(
    lane_start: int,
    stripe_len: int,
    width: int,
    direction: str,
    scenario: str,
    color_pool: list[int],
    rng: Generator,
) -> dict[str, object] | None:
    stripe_rows = [lane_start + 1 + i for i in range(stripe_len)]
    point_r = stripe_rows[int(rng.integers(0, stripe_len))]
    min_steps = MIN_STEPS[scenario]

    if direction == "right":
        lo = 1
        hi = width - min_steps
        if lo > hi:
            return None
        point_c = int(rng.integers(lo, hi + 1))
        max_steps = width - point_c
        stripe_c = point_c - 1
        step_to_cell = lambda step: stripe_c + step
    else:
        lo = min_steps - 1
        hi = width - 2
        if lo > hi:
            return None
        point_c = int(rng.integers(lo, hi + 1))
        max_steps = point_c + 1
        stripe_c = point_c + 1
        step_to_cell = lambda step: stripe_c - step

    stripe_colors = [int(x) for x in rng.choice(color_pool, size=stripe_len, replace=False)]
    stripe = [(r, stripe_c, color) for r, color in zip(stripe_rows, stripe_colors)]
    obstacles: list[tuple[int, int, int]] = []

    if scenario == "absorb":
        idx = int(rng.integers(0, stripe_len))
        step = int(rng.integers(2, max_steps))
        obstacles.append((stripe_rows[idx], step_to_cell(step), stripe_colors[idx]))
    elif scenario == "stop":
        idx = int(rng.integers(0, stripe_len))
        step = int(rng.integers(2, max_steps + 1))
        mismatch = _choose_other_color(color_pool, stripe_colors[idx], rng)
        obstacles.append((stripe_rows[idx], step_to_cell(step), mismatch))
    elif scenario == "absorb_stop":
        absorb_idx, stop_idx = [int(x) for x in rng.choice(stripe_len, size=2, replace=False)]
        absorb_step = int(rng.integers(2, max_steps - 1))
        stop_step = int(rng.integers(absorb_step + 2, max_steps + 1))
        obstacles.append((stripe_rows[absorb_idx], step_to_cell(absorb_step), stripe_colors[absorb_idx]))
        mismatch = _choose_other_color(color_pool, stripe_colors[stop_idx], rng)
        obstacles.append((stripe_rows[stop_idx], step_to_cell(stop_step), mismatch))

    point = (point_r, point_c)
    return {
        "point": point,
        "direction": direction,
        "arrow": _arrow_cells(point, direction),
        "stripe": stripe,
        "obstacles": obstacles,
    }


def _build_vertical_object(
    lane_start: int,
    stripe_len: int,
    height: int,
    direction: str,
    scenario: str,
    color_pool: list[int],
    rng: Generator,
) -> dict[str, object] | None:
    stripe_cols = [lane_start + 1 + i for i in range(stripe_len)]
    point_c = stripe_cols[int(rng.integers(0, stripe_len))]
    min_steps = MIN_STEPS[scenario]

    if direction == "down":
        lo = 1
        hi = height - min_steps
        if lo > hi:
            return None
        point_r = int(rng.integers(lo, hi + 1))
        max_steps = height - point_r
        stripe_r = point_r - 1
        step_to_cell = lambda step: stripe_r + step
    else:
        lo = min_steps - 1
        hi = height - 2
        if lo > hi:
            return None
        point_r = int(rng.integers(lo, hi + 1))
        max_steps = point_r + 1
        stripe_r = point_r + 1
        step_to_cell = lambda step: stripe_r - step

    stripe_colors = [int(x) for x in rng.choice(color_pool, size=stripe_len, replace=False)]
    stripe = [(stripe_r, c, color) for c, color in zip(stripe_cols, stripe_colors)]
    obstacles: list[tuple[int, int, int]] = []

    if scenario == "absorb":
        idx = int(rng.integers(0, stripe_len))
        step = int(rng.integers(2, max_steps))
        obstacles.append((step_to_cell(step), stripe_cols[idx], stripe_colors[idx]))
    elif scenario == "stop":
        idx = int(rng.integers(0, stripe_len))
        step = int(rng.integers(2, max_steps + 1))
        mismatch = _choose_other_color(color_pool, stripe_colors[idx], rng)
        obstacles.append((step_to_cell(step), stripe_cols[idx], mismatch))
    elif scenario == "absorb_stop":
        absorb_idx, stop_idx = [int(x) for x in rng.choice(stripe_len, size=2, replace=False)]
        absorb_step = int(rng.integers(2, max_steps - 1))
        stop_step = int(rng.integers(absorb_step + 2, max_steps + 1))
        obstacles.append((step_to_cell(absorb_step), stripe_cols[absorb_idx], stripe_colors[absorb_idx]))
        mismatch = _choose_other_color(color_pool, stripe_colors[stop_idx], rng)
        obstacles.append((step_to_cell(stop_step), stripe_cols[stop_idx], mismatch))

    point = (point_r, point_c)
    return {
        "point": point,
        "direction": direction,
        "arrow": _arrow_cells(point, direction),
        "stripe": stripe,
        "obstacles": obstacles,
    }


def _choose_other_color(pool: list[int], blocked: int, rng: Generator) -> int:
    choices = [color for color in pool if color != blocked]
    return int(choices[int(rng.integers(0, len(choices)))])


def _paint_scene(
    height: int,
    width: int,
    bg: int,
    arrow_color: int,
    objects: list[dict[str, object]],
) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for obj in objects:
        for r, c in obj["arrow"]:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if (r, c) in occupied:
                return None
            occupied.add((r, c))
            grid[r, c] = arrow_color

        for r, c, color in obj["stripe"]:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if (r, c) in occupied:
                return None
            occupied.add((r, c))
            grid[r, c] = color

        for r, c, color in obj["obstacles"]:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if (r, c) in occupied:
                return None
            for rr in range(max(0, r - 1), min(height, r + 2)):
                for cc in range(max(0, c - 1), min(width, c + 2)):
                    if (rr, cc) in occupied:
                        return None
            occupied.add((r, c))
            grid[r, c] = color

    return grid


def _gen_attempt(rng: Generator) -> np.ndarray | None:
    axis = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
    count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    stripe_lengths = [int(rng.choice(STRIPE_LENGTHS)) for _ in range(count)]
    lane_sizes = [length + 2 for length in stripe_lengths]

    if axis == "horizontal":
        min_h = max(GRID_MIN, sum(lane_sizes) + LANE_GAP * (count - 1) + 2 * OUTER_MARGIN)
        if min_h > GRID_MAX:
            return None
        height = int(rng.integers(min_h, GRID_MAX + 1))
        width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        lane_starts = _sample_intervals(height, lane_sizes, rng)
    else:
        min_w = max(GRID_MIN, sum(lane_sizes) + LANE_GAP * (count - 1) + 2 * OUTER_MARGIN)
        if min_w > GRID_MAX:
            return None
        width = int(rng.integers(min_w, GRID_MAX + 1))
        height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        lane_starts = _sample_intervals(width, lane_sizes, rng)

    if lane_starts is None:
        return None

    bg, arrow_color = [int(x) for x in rng.choice(np.arange(10), size=2, replace=False)]
    color_pool = [color for color in range(10) if color not in (bg, arrow_color)]
    scenarios = _scenario_list(count, rng)
    directions = _direction_list(axis, count, rng)

    objects: list[dict[str, object]] = []
    for lane_start, stripe_len, direction, scenario in zip(lane_starts, stripe_lengths, directions, scenarios):
        if axis == "horizontal":
            obj = _build_horizontal_object(
                lane_start=lane_start,
                stripe_len=stripe_len,
                width=width,
                direction=direction,
                scenario=scenario,
                color_pool=color_pool,
                rng=rng,
            )
        else:
            obj = _build_vertical_object(
                lane_start=lane_start,
                stripe_len=stripe_len,
                height=height,
                direction=direction,
                scenario=scenario,
                color_pool=color_pool,
                rng=rng,
            )
        if obj is None:
            return None
        objects.append(obj)

    grid = _paint_scene(height, width, bg, arrow_color, objects)
    if grid is None:
        return None

    try:
        parsed_bg, arrows = _parse_scene(grid)
        assert parsed_bg == bg
        _solve_scene(grid, parsed_bg, arrows)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
