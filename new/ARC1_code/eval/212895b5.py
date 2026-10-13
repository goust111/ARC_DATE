from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, get_bg, paste
import random

BBox = tuple[int, int, int, int]


def _find_solid_3x3_square(g: np.ndarray, bg: int) -> BBox:
    candidates: list[BBox] = []
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected one 3x3 square color and one blocker color"

    for color in fg_colors:
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None, "foreground color must be present"
        r0, c0, r1, c1 = b
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if int(mask.sum()) == 9 and np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
            candidates.append((int(r0), int(c0), int(r1), int(c1)))

    assert len(candidates) == 1, "expected exactly one solid 3x3 square"
    return candidates[0]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _non_bg(g: np.ndarray, bg: int, r: int, c: int) -> bool:
    return _in_bounds(g.shape, r, c) and int(g[r, c]) != bg


def _row_pinched(g: np.ndarray, bg: int, r: int, c: int) -> bool:
    return _non_bg(g, bg, r, c - 1) and _non_bg(g, bg, r, c + 1)


def _paint_cell(g: np.ndarray, out: np.ndarray, bg: int, r: int, c: int, color: int) -> None:
    assert int(g[r, c]) == bg, "paint paths may only cover original background"
    assert int(out[r, c]) in (bg, int(color)), "paint paths must not conflict"
    out[r, c] = int(color)


def _paint_red_ray(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    r, c = start
    dr, dc = direction
    while _in_bounds(g.shape, r, c) and int(g[r, c]) == bg and not _row_pinched(g, bg, r, c):
        _paint_cell(g, out, bg, r, c, Color.RED)
        r += dr
        c += dc


def _slice_cells(r: int, c: int, axis: str, order: str) -> list[tuple[int, int]]:
    assert order in ("asc", "desc")
    if axis == "h":
        cols = [c - 1, c, c + 1]
        if order == "desc":
            cols.reverse()
        return [(r, cc) for cc in cols]

    assert axis == "v"
    rows = [r - 1, r, r + 1]
    if order == "desc":
        rows.reverse()
    return [(rr, c) for rr in rows]


def _paint_yellow_ribbon(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    odd_axis: str,
    odd_order: str,
) -> None:
    r, c = start
    dr, dc = direction
    step = 0
    while True:
        cells = [(r, c)] if step % 2 == 0 else _slice_cells(r, c, odd_axis, odd_order)

        stop = False
        for rr, cc in cells:
            if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != bg:
                stop = True
                break
            _paint_cell(g, out, bg, rr, cc, Color.YELLOW)

        if stop:
            break
        r += dr
        c += dc
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the solid 3x3 square into clipped red diagonal corner rays and yellow side-center ribbons.

    Input structure
    - The background is the dominant color.
    - The input has exactly two foreground colors: one forms a solid 3x3 square, and the other forms scattered blocker pixels.
    - The 3x3 square is the source; all original foreground pixels remain unchanged.

    Rule
    - Paint red one-cell diagonal rays outward from the four square corners until a blocker, edge, or row-wise foreground pinch stops the ray.
    - Paint yellow diagonal ribbons from the top, left, bottom, and right side centers in a clockwise pinwheel: NE, NW, SW, and SE.
    - A yellow ribbon alternates a single center cell with a three-cell horizontal or vertical slice; a blocker encountered within a slice stops the ribbon after any earlier cells in that slice are painted.

    Key insight
    - The square defines four corner rays and four side-center ribbons, while the scattered foreground pixels are not targets; they are blockers that clip these local diagonal extrusions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    r0, c0, r1, c1 = _find_solid_3x3_square(g, bg)
    assert r1 - r0 == 2 and c1 - c0 == 2

    out = g.copy()
    rm = r0 + 1
    cm = c0 + 1

    red_specs = [
        ((r0 - 1, c0 - 1), (-1, -1)),
        ((r0 - 1, c1 + 1), (-1, 1)),
        ((r1 + 1, c0 - 1), (1, -1)),
        ((r1 + 1, c1 + 1), (1, 1)),
    ]
    for start, direction in red_specs:
        _paint_red_ray(g, out, bg, start, direction)

    yellow_specs = [
        ((r0 - 1, cm), (-1, 1), "h", "asc"),
        ((rm, c0 - 1), (-1, -1), "v", "desc"),
        ((r1 + 1, cm), (1, -1), "h", "desc"),
        ((rm, c1 + 1), (1, 1), "v", "asc"),
    ]
    for start, direction, odd_axis, odd_order in yellow_specs:
        _paint_yellow_ribbon(g, out, bg, start, direction, odd_axis, odd_order)

    return out

# --------------- generator (o5x) ---------------
# Canvas and source square.
GRID_ROWS_RANGE = (16, 21)
GRID_COLS_RANGE = (17, 23)
SQUARE_SIZE = 3
SQUARE_MARGIN = 5
SQUARE_COLOR = Color.TEAL
BLOCKER_COLOR = Color.GRAY

# Output path constraints.
MAX_RED_RAY_LEN = 8
MAX_YELLOW_RIBBON_STEPS = 5
PAINTED_COUNT_RANGE = (31, 52)
MIN_ACTIVE_EMITTERS = 4
LENGTH_ADJUST_RETRIES = 64

# Blocker population and local cluster shape.
BLOCKER_COUNT_RANGE = (24, 64)
BLOCKER_DENSITY_RANGE = (0.06, 0.13)
EXTRA_BLOCKER_CLUSTER_SIZES = (1, 2, 3)
EXTRA_BLOCKER_CLUSTER_WEIGHTS = (76, 17, 7)
EXTRA_BLOCKER_SEED_TRIES = 80
MAX_BLOCKER_COMPONENT_SIZE = 3
MIN_SINGLETON_COMPONENT_FACTOR = 2

# Whole-generator retry budget.
GENERATION_RETRIES = 300

def _gen_slice_cells(r, c, axis, order):
    if axis == "h":
        cols = [c - 1, c, c + 1]
        if order == "desc":
            cols.reverse()
        return [(r, cc) for cc in cols]

    rows = [r - 1, r, r + 1]
    if order == "desc":
        rows.reverse()
    return [(rr, c) for rr in rows]


def _gen_ribbon_cells_at_step(start, direction, odd_axis, odd_order, step):
    r = start[0] + direction[0] * step
    c = start[1] + direction[1] * step
    if step % 2 == 0:
        return [(r, c)]
    return _gen_slice_cells(r, c, odd_axis, odd_order)


def _gen_max_ray_len(shape, start, direction):
    r, c = start
    dr, dc = direction
    n = 0
    while _in_bounds(shape, r, c):
        n += 1
        r += dr
        c += dc
    return n


def _gen_max_ribbon_steps(shape, start, direction, odd_axis, odd_order):
    step = 0
    while True:
        cells = _gen_ribbon_cells_at_step(start, direction, odd_axis, odd_order, step)
        if any(not _in_bounds(shape, r, c) for r, c in cells):
            return step
        step += 1


def _gen_ribbon_cell_count(steps):
    return sum(1 if step % 2 == 0 else 3 for step in range(steps))


def _gen_total_painted(red_lengths, ribbon_steps):
    return sum(red_lengths) + sum(_gen_ribbon_cell_count(s) for s in ribbon_steps)


def _gen_pick_lengths(red_caps, ribbon_caps):
    red_lengths = [random.randint(1, cap) for cap in red_caps]
    ribbon_steps = [random.randint(1, cap) for cap in ribbon_caps]
    for _ in range(LENGTH_ADJUST_RETRIES):
        active = sum(1 for v in red_lengths + ribbon_steps if v >= 2)
        total = _gen_total_painted(red_lengths, ribbon_steps)
        if PAINTED_COUNT_RANGE[0] <= total <= PAINTED_COUNT_RANGE[1] and active >= MIN_ACTIVE_EMITTERS:
            return red_lengths, ribbon_steps
        if total < PAINTED_COUNT_RANGE[0] or active < MIN_ACTIVE_EMITTERS:
            choices = []
            for i, (v, cap) in enumerate(zip(red_lengths, red_caps)):
                if v < cap:
                    choices.append(("red", i))
            for i, (v, cap) in enumerate(zip(ribbon_steps, ribbon_caps)):
                if v < cap:
                    choices.append(("yellow", i))
            if not choices:
                break
            kind, i = random.choice(choices)
            if kind == "red":
                red_lengths[i] += 1
            else:
                ribbon_steps[i] += 1
            continue
        choices = []
        for i, v in enumerate(red_lengths):
            if v > 1:
                choices.append(("red", i))
        for i, v in enumerate(ribbon_steps):
            if v > 1:
                choices.append(("yellow", i))
        if not choices:
            break
        kind, i = random.choice(choices)
        if kind == "red":
            red_lengths[i] -= 1
        else:
            ribbon_steps[i] -= 1

    return None, None


def _gen_add_paint(painted, color, cells):
    for cell in cells:
        old = painted.get(cell)
        if old is not None and old != color:
            return False
    for cell in cells:
        painted[cell] = color
    return True


def _gen_red_side_neighbors(cells, shape):
    side = set()
    for r, c in cells:
        for cc in (c - 1, c + 1):
            if _in_bounds(shape, r, cc):
                side.add((r, cc))
    return side


def _gen_blocker_component_sizes(blockers):
    unseen = set(blockers)
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in unseen:
                    unseen.remove(nb)
                    stack.append(nb)
        sizes.append(size)
    return sizes


def _gen_discard_neighbors(allowed, cells, shape):
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _in_bounds(shape, nr, nc):
                allowed.discard((nr, nc))


def _gen_place_extra_blockers(blockers, target, allowed, shape):
    blockers = set(blockers)
    allowed = set(allowed) - blockers
    _gen_discard_neighbors(allowed, blockers, shape)

    while len(blockers) < target and allowed:
        remaining = target - len(blockers)
        want = random.choices(EXTRA_BLOCKER_CLUSTER_SIZES, weights=EXTRA_BLOCKER_CLUSTER_WEIGHTS)[0]
        want = min(want, remaining)

        seeds = list(allowed)
        random.shuffle(seeds)
        cluster = None
        for seed in seeds[:EXTRA_BLOCKER_SEED_TRIES]:
            chosen = [seed]
            frontier = [seed]
            local_allowed = allowed - {seed}
            while len(chosen) < want and frontier:
                r, c = random.choice(frontier)
                nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                random.shuffle(nbrs)
                added = False
                for nr, nc in nbrs:
                    if (nr, nc) in local_allowed:
                        chosen.append((nr, nc))
                        frontier.append((nr, nc))
                        local_allowed.remove((nr, nc))
                        added = True
                        break
                if not added:
                    frontier.remove((r, c))
            if len(chosen) == want:
                cluster = chosen
                break

        if cluster is None:
            cluster = [random.choice(tuple(allowed))]

        for cell in cluster:
            blockers.add(cell)
            allowed.discard(cell)

        # Keep independent clusters small and separated from later clusters.
        _gen_discard_neighbors(allowed, cluster, shape)

    return blockers if len(blockers) >= target else None


def _gen_square_bounds(shape):
    h, w = shape
    r0 = random.randint(SQUARE_MARGIN, h - SQUARE_MARGIN - SQUARE_SIZE)
    c0 = random.randint(SQUARE_MARGIN, w - SQUARE_MARGIN - SQUARE_SIZE)
    r1 = r0 + SQUARE_SIZE - 1
    c1 = c0 + SQUARE_SIZE - 1
    rm = r0 + 1
    cm = c0 + 1
    return r0, c0, r1, c1, rm, cm


def _gen_path_specs(square):
    r0, c0, r1, c1, rm, cm = square
    red_specs = [
        ((r0 - 1, c0 - 1), (-1, -1)),
        ((r0 - 1, c1 + 1), (-1, 1)),
        ((r1 + 1, c0 - 1), (1, -1)),
        ((r1 + 1, c1 + 1), (1, 1)),
    ]
    yellow_specs = [
        ((r0 - 1, cm), (-1, 1), "h", "asc"),
        ((rm, c0 - 1), (-1, -1), "v", "desc"),
        ((r1 + 1, cm), (1, -1), "h", "desc"),
        ((rm, c1 + 1), (1, 1), "v", "asc"),
    ]
    return red_specs, yellow_specs


def _gen_layout():
    shape = (random.randint(*GRID_ROWS_RANGE), random.randint(*GRID_COLS_RANGE))
    square = _gen_square_bounds(shape)
    red_specs, yellow_specs = _gen_path_specs(square)
    red_caps = [
        min(MAX_RED_RAY_LEN, _gen_max_ray_len(shape, start, direction))
        for start, direction in red_specs
    ]
    ribbon_caps = [
        min(MAX_YELLOW_RIBBON_STEPS, _gen_max_ribbon_steps(shape, start, direction, axis, order))
        for start, direction, axis, order in yellow_specs
    ]
    if min(red_caps + ribbon_caps) < 1:
        return None

    red_lengths, ribbon_steps = _gen_pick_lengths(red_caps, ribbon_caps)
    if red_lengths is None:
        return None
    return shape, square, red_specs, yellow_specs, red_caps, ribbon_caps, red_lengths, ribbon_steps


def _gen_trace_paths(red_specs, yellow_specs, red_lengths, ribbon_steps, red_caps, ribbon_caps):
    painted = {}
    blockers = set()

    for (start, direction), length, cap in zip(red_specs, red_lengths, red_caps):
        cells = [
            (start[0] + direction[0] * i, start[1] + direction[1] * i)
            for i in range(length)
        ]
        if not _gen_add_paint(painted, Color.RED, cells):
            return None
        if length < cap:
            stop = (start[0] + direction[0] * length, start[1] + direction[1] * length)
            blockers.add(stop)

    for (start, direction, axis, order), steps, cap in zip(yellow_specs, ribbon_steps, ribbon_caps):
        for step in range(steps):
            cells = _gen_ribbon_cells_at_step(start, direction, axis, order, step)
            if not _gen_add_paint(painted, Color.YELLOW, cells):
                return None
        if steps < cap:
            stop_cells = _gen_ribbon_cells_at_step(start, direction, axis, order, steps)
            blockers.add(stop_cells[0])
    return painted, blockers


def _gen_hard_forbidden(square, start_ring, painted, shape):
    r0, c0, r1, c1, _, _ = square
    square_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    red_cells = {cell for cell, color in painted.items() if color == Color.RED}
    red_side = _gen_red_side_neighbors(red_cells, shape)
    path_cells = set(painted)
    return square_cells | start_ring | path_cells | red_side


def _gen_target_blocker_count(shape, blocker_count):
    h, w = shape
    area = h * w
    lo = max(BLOCKER_COUNT_RANGE[0], int(np.ceil(area * BLOCKER_DENSITY_RANGE[0])), blocker_count)
    hi = min(BLOCKER_COUNT_RANGE[1], int(np.floor(area * BLOCKER_DENSITY_RANGE[1])))
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_attempt():
    layout = _gen_layout()
    if layout is None:
        return None
    shape, square, red_specs, yellow_specs, red_caps, ribbon_caps, red_lengths, ribbon_steps = layout

    traced = _gen_trace_paths(red_specs, yellow_specs, red_lengths, ribbon_steps, red_caps, ribbon_caps)
    if traced is None:
        return None
    painted, blockers = traced

    start_ring = {start for start, _ in red_specs}
    start_ring.update(start for start, _, _, _ in yellow_specs)
    hard_forbidden = _gen_hard_forbidden(square, start_ring, painted, shape)
    if blockers & hard_forbidden:
        return None

    target_blockers = _gen_target_blocker_count(shape, len(blockers))
    if target_blockers is None:
        return None

    h, w = shape
    allowed = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in hard_forbidden and (r, c) not in blockers
    }
    blockers = _gen_place_extra_blockers(blockers, target_blockers, allowed, shape)
    if blockers is None:
        return None
    component_sizes = _gen_blocker_component_sizes(blockers)
    if component_sizes and (
        max(component_sizes) > MAX_BLOCKER_COMPONENT_SIZE
        or component_sizes.count(1) * MIN_SINGLETON_COMPONENT_FACTOR < len(component_sizes)
    ):
        return None

    r0, c0, _, _, _, _ = square
    inp = np.zeros(shape, dtype=int)
    paste(inp, np.full((SQUARE_SIZE, SQUARE_SIZE), SQUARE_COLOR, dtype=int), r0, c0)
    for r, c in blockers:
        inp[r, c] = BLOCKER_COLOR

    expected = inp.copy()
    for (r, c), color in painted.items():
        expected[r, c] = color

    actual = main(inp)
    painted_count = int(np.sum(actual != inp))
    if not (PAINTED_COUNT_RANGE[0] <= painted_count <= PAINTED_COUNT_RANGE[1]):
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 212895b5 input")
