from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

import random

from arc_utils import (
    Color,
    STRUCT8,
    colors_present,
    components_mask,
    get_bg,
    in_bounds,
    label_mask,
    neighbors4,
)


def _path_color(g: np.ndarray, bg: int) -> int:
    candidates = []
    for color in colors_present(g, exclude={bg}):
        comps = components_mask(g == color, connectivity=4)
        if len(comps) == 2:
            candidates.append(int(color))

    assert len(candidates) == 1, "expected exactly one color forming two path components"
    return candidates[0]


def _adjacent_path_labels(labels: np.ndarray, r: int, c: int) -> set[int]:
    adjacent = set()
    for rr, cc in neighbors4(r, c):
        if in_bounds(labels.shape, rr, cc) and labels[rr, cc] > 0:
            adjacent.add(int(labels[rr, cc]))
    return adjacent


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the top-connected middle channel between two orange path components teal.

    Input structure
    - The background is black.
    - One foreground color forms exactly two 4-connected one-pixel-wide path components.
    - Optional non-background control pixels may sit in the channel between the two paths.

    Rule
    - Dilate each path component by one cell in the 8-neighborhood and intersect the two dilations; this gives the one-cell-wide middle channel.
    - A control pixel in that channel is a blocker only when its 4-neighbors touch fewer than both path components; otherwise it is part of the channel and is repainted.
    - After removing blockers, paint the channel component connected to the top edge teal and keep all other pixels unchanged.

    Key insight
    - The desired pixels are not found by region filling; they are the geometric overlap of the two paths' one-cell halos, with weakly attached control pixels splitting the channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    path_color = _path_color(g, bg)
    path_labels, n_paths = label_mask(g == path_color, connectivity=4)
    assert n_paths == 2, "expected two path components"

    channel = g != path_color
    for label_id in range(1, n_paths + 1):
        channel &= binary_dilation(path_labels == label_id, structure=STRUCT8)
    assert np.any(channel), "expected a middle channel between the paths"

    blocked = []
    for r_raw, c_raw in np.argwhere(channel & (g != bg)):
        r = int(r_raw)
        c = int(c_raw)
        adjacent = _adjacent_path_labels(path_labels, r, c)
        assert adjacent, "control pixel in channel must touch a path component"
        if len(adjacent) < n_paths:
            blocked.append((r, c))
        else:
            assert len(adjacent) == n_paths, "channel control pixel cannot touch extra path labels"

    passable = channel.copy()
    for r, c in blocked:
        passable[r, c] = False

    channel_labels, n_channels = label_mask(passable, connectivity=4)
    assert n_channels >= 1, "expected at least one passable channel component"
    top_components = [
        label_id for label_id in range(1, n_channels + 1) if np.any(channel_labels[0, :] == label_id)
    ]
    assert len(top_components) == 1, "expected exactly one channel component connected to the top edge"

    paint = channel_labels == top_components[0]
    assert np.any(paint), "expected cells to paint"
    out = g.copy()
    out[paint] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Canvas and channel bounds.
GEN_GRID_SHAPE = (10, 10)
GEN_TOP_ROW = 0
GEN_BOTTOM_ROW = 9
GEN_LEFT_COL = 0
GEN_RIGHT_COL = 9
GEN_MIN_CHANNEL_COL = 1
GEN_MAX_CHANNEL_COL = 8

# Segment and turn sampling.
GEN_MIN_TURNS = 2
GEN_MAX_TURNS = 4
GEN_VERTICAL_MIN = 2
GEN_VERTICAL_MAX = 7
GEN_VERTICAL_TOTAL_MIN = 6
GEN_VERTICAL_TOTAL_MAX = 9
GEN_FULL_HEIGHT_TOTAL = 9
GEN_FULL_HEIGHT_PROB = 0.8
GEN_HORIZONTAL_MIN = 2
GEN_HORIZONTAL_MAX = 4
GEN_HORIZONTAL_BUDGET = 7

# Optional blocker placement.
GEN_BLOCKER_PROB = 0.6
GEN_BLOCKING_BLOCKER_PROB = 0.55

# Retry budget for rejected geometry.
GEN_RETRY_LIMIT = 300


def _gen_walk_segment(path, dr, dc, steps):
    r, c = path[-1]
    for _ in range(steps):
        r += dr
        c += dc
        path.append((r, c))


def _gen_directions(hdir, turns):
    directions = [(1, 0)]
    for i in range(turns):
        directions.append((0, hdir) if i % 2 == 0 else (1, 0))
    return directions


def _gen_partition_total(total, count, minimum, maximum):
    """Randomly split total into count bounded positive steps."""
    steps = []
    remaining = total
    for i in range(count - 1):
        rest = count - i - 1
        hi = min(maximum, remaining - minimum * rest)
        lo = max(minimum, remaining - maximum * rest)
        val = random.randint(lo, hi)
        steps.append(val)
        remaining -= val
    steps.append(remaining)
    return steps


def _gen_vertical_steps(directions):
    v_count = sum(1 for d in directions if d == (1, 0))
    if directions[-1] == (1, 0) and random.random() < GEN_FULL_HEIGHT_PROB:
        total = GEN_FULL_HEIGHT_TOTAL
    else:
        total = random.randint(GEN_VERTICAL_TOTAL_MIN, GEN_VERTICAL_TOTAL_MAX)
    return _gen_partition_total(total, v_count, GEN_VERTICAL_MIN, GEN_VERTICAL_MAX)


def _gen_horizontal_steps(h_count):
    steps = []
    remaining = GEN_HORIZONTAL_BUDGET
    for i in range(h_count):
        hi = min(GEN_HORIZONTAL_MAX, remaining - GEN_HORIZONTAL_MIN * (h_count - i - 1))
        step = random.randint(GEN_HORIZONTAL_MIN, hi)
        steps.append(step)
        remaining -= step
    return steps


def _gen_start_col(hdir, h_steps):
    total_h = sum(h_steps)
    if hdir == 1:
        return random.randint(GEN_MIN_CHANNEL_COL, GEN_MAX_CHANNEL_COL - total_h)
    return random.randint(GEN_MIN_CHANNEL_COL + total_h, GEN_MAX_CHANNEL_COL)


def _gen_channel_path():
    hdir = random.choice([-1, 1])
    turns = random.randint(GEN_MIN_TURNS, GEN_MAX_TURNS)
    directions = _gen_directions(hdir, turns)
    v_steps = _gen_vertical_steps(directions)
    h_steps = _gen_horizontal_steps(sum(1 for d in directions if d[0] == 0))
    start_c = _gen_start_col(hdir, h_steps)

    path = [(GEN_TOP_ROW, start_c)]
    vi = 0
    hi = 0
    for dr, dc in directions:
        if dr:
            steps = v_steps[vi]
            vi += 1
        else:
            steps = h_steps[hi]
            hi += 1
        _gen_walk_segment(path, dr, dc, steps)

    if any(
        r < GEN_TOP_ROW or r > GEN_BOTTOM_ROW or c < GEN_MIN_CHANNEL_COL or c > GEN_MAX_CHANNEL_COL
        for r, c in path
    ):
        return None
    return path


def _gen_left_normal(dr, dc):
    return -dc, dr


def _gen_wall_sets(path):
    left = set()
    right = set()
    channel = set(path)
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))
        for dr, dc in dirs:
            lr, lc = _gen_left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))
        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _gen_left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr, c + dc) for dr, dc in offsets]
                if all(cell not in channel for cell in direct):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _gen_mask_from_cells(cells):
    mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    cells = list(cells)
    if cells:
        rows, cols = zip(*cells)
        mask[rows, cols] = True
    return mask


def _gen_paint_cells(grid, cells, color):
    cells = list(cells)
    if cells:
        rows, cols = zip(*cells)
        grid[rows, cols] = color


def _gen_is_thin_component(mask):
    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return False
    cells = {tuple(map(int, xy)) for xy in comps[0]}
    for r, c in cells:
        deg = sum((rr, cc) in cells for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if deg > 2:
            return False
    return True


def _gen_channel_from_solver_geometry(grid):
    labels, n_paths = label_mask(grid == Color.ORANGE, connectivity=4)
    if n_paths != 2:
        return None, None

    channel = grid != Color.ORANGE
    for label_id in range(1, n_paths + 1):
        channel &= binary_dilation(labels == label_id, structure=STRUCT8)
    return channel, labels


def _gen_render_valid_walls(channel_path):
    """Render orange walls and verify they induce exactly the sampled channel."""
    channel_cells = set(channel_path)
    left, right = _gen_wall_sets(channel_path)
    if left & right or left & channel_cells or right & channel_cells:
        return None, None
    if any(r < GEN_TOP_ROW or r > GEN_BOTTOM_ROW or c < GEN_LEFT_COL or c > GEN_RIGHT_COL for r, c in left | right):
        return None, None

    grid = np.zeros(GEN_GRID_SHAPE, dtype=int)
    _gen_paint_cells(grid, left | right, Color.ORANGE)

    left_mask = _gen_mask_from_cells(left)
    right_mask = _gen_mask_from_cells(right)
    if not _gen_is_thin_component(left_mask) or not _gen_is_thin_component(right_mask):
        return None, None

    channel, labels = _gen_channel_from_solver_geometry(grid)
    if channel is None:
        return None, None
    expected_channel = _gen_mask_from_cells(channel_path)
    if not np.array_equal(channel, expected_channel):
        return None, None
    return grid, labels


def _gen_blocker_candidates(labels, channel_path):
    """Return turn blockers and straight pass-through blocker locations."""
    turn_cells = []
    straight_cells = []
    for i, (r, c) in enumerate(channel_path):
        if i == 0:
            continue
        adj = _adjacent_path_labels(labels, r, c)
        if len(adj) == 1:
            turn_cells.append((r, c))
        elif len(adj) == 2:
            straight_cells.append((r, c))
    return turn_cells, straight_cells


def _gen_maybe_place_blocker(grid, labels, channel_path):
    if random.random() < GEN_BLOCKER_PROB:
        turn_cells, straight_cells = _gen_blocker_candidates(labels, channel_path)
        if turn_cells and straight_cells:
            if random.random() < GEN_BLOCKING_BLOCKER_PROB:
                r, c = random.choice(turn_cells)
            else:
                r, c = random.choice(straight_cells)
            grid[r, c] = Color.MAROON


def _gen_oracle_accepts(grid):
    """Accept only grids whose solver output paints a top-connected channel."""
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(np.any((out == Color.TEAL)[GEN_TOP_ROW, :]))


def _gen_attempt():
    channel_path = _gen_channel_path()
    if channel_path is None:
        return None

    grid, labels = _gen_render_valid_walls(channel_path)
    if grid is None:
        return None

    _gen_maybe_place_blocker(grid, labels, channel_path)
    if not _gen_oracle_accepts(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f1bcbc2c input")
