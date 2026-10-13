import math
import random

import numpy as np

from arc_utils import colors_present, find_components, paste


def _blocked_mask(g: np.ndarray, obstacle: int) -> np.ndarray:
    obstacle_mask = g == int(obstacle)
    cushion = np.zeros_like(obstacle_mask, dtype=bool)
    cushion[:-1, :] = obstacle_mask[1:, :]
    bottom_edge = np.zeros_like(obstacle_mask, dtype=bool)
    bottom_edge[-1, :] = True
    return obstacle_mask | cushion | bottom_edge


def _assert_output_separated(out: np.ndarray, obstacle: int, moving: int) -> None:
    obstacle_mask = out == int(obstacle)
    near_obstacle = np.zeros_like(obstacle_mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue

            src_r0 = max(0, -dr)
            src_r1 = out.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = out.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = out.shape[0] - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = out.shape[1] - max(0, -dc)
            near_obstacle[dst_r0:dst_r1, dst_c0:dst_c1] |= obstacle_mask[
                src_r0:src_r1, src_c0:src_c1
            ]

    assert not np.any((out == int(moving)) & near_obstacle), (
        "settled moving objects must not be 8-connected to obstacles"
    )


def _obstacle_column_heights(g: np.ndarray, obstacle: int) -> list[int]:
    heights = []
    obstacle_mask = g == int(obstacle)
    H, W = map(int, g.shape)
    for c in range(W):
        rows = np.flatnonzero(obstacle_mask[:, c])
        heights.append(0 if len(rows) == 0 else H - int(rows[0]))
    return heights


def _assert_obstacle_profile(g: np.ndarray, obstacle: int) -> None:
    heights = _obstacle_column_heights(g, obstacle)
    occupied = [h for h in heights if h > 0]
    assert occupied, "obstacle must occupy at least one bottom-supported column"

    H, W = map(int, g.shape)
    obstacle_mask = g == int(obstacle)
    total = int(obstacle_mask.sum())
    bottom3 = int(obstacle_mask[max(0, H - 3) :, :].sum())

    assert len(occupied) * 3 >= W, "obstacle must cover a substantial part of the bottom"
    assert sum(occupied) * 4 <= len(occupied) * 13, (
        "obstacle columns must be low on average"
    )
    assert bottom3 * 5 >= total * 3, "obstacle pixels must be mostly near the bottom"


def _infer_roles(g: np.ndarray) -> tuple[int, int, int]:
    colors = colors_present(g)
    assert len(colors) == 3, "expected background, obstacle, and moving colors"

    candidates = []
    for bg in colors:
        for obstacle in colors:
            if obstacle == bg or not np.any(g[-1, :] == obstacle):
                continue
            for moving in colors:
                if moving in (bg, obstacle) or np.any(g[-1, :] == moving):
                    continue
                blocked = _blocked_mask(g, obstacle)
                if np.any(blocked & (g == moving)):
                    continue
                candidates.append((int(bg), int(obstacle), int(moving)))

    assert len(candidates) == 1, "roles must be uniquely determined"
    return candidates[0]


def _movable_components(out: np.ndarray, bg: int, moving: int, blocked: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    moving_mask = out == int(moving)
    moves = []
    for comp in find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[moving]):
        coords = comp["coords"]
        shifted = coords + np.array([1, 0])
        if int(shifted[:, 0].max()) >= int(out.shape[0]):
            continue
        if np.any(blocked[shifted[:, 0], shifted[:, 1]]):
            continue

        other_moving = moving_mask.copy()
        other_moving[coords[:, 0], coords[:, 1]] = False
        if np.any(other_moving[shifted[:, 0], shifted[:, 1]]):
            continue
        moves.append((coords, shifted))
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the moving-color objects downward until they rest above a one-row cushion before the obstacle or bottom edge.

    Input structure:
    - The grid has exactly three colors: empty background, a stationary obstacle/floor color, and a moving object color.
    - The obstacle color is the unique non-background role that touches the bottom edge.
    - Moving pixels start outside the obstacle cells, outside the row directly above each obstacle pixel, and outside the bottom row.

    Rule:
    - Keep the background and obstacle pixels fixed.
    - Treat obstacle pixels, cells directly above obstacle pixels, and the bottom row as blocked.
    - Repeatedly move every 4-connected moving-color object one row down when its translated shape does not hit blocked cells or another moving object.

    Key insight: The visible floor has an invisible one-row cushion above it, so the objects fall under gravity but stop with one empty row before the floor or the grid bottom.
    """
    g = np.asarray(grid, dtype=int)
    bg, obstacle, moving = _infer_roles(g)
    _assert_obstacle_profile(g, obstacle)
    blocked = _blocked_mask(g, obstacle)
    assert not np.any(blocked & (g == moving)), "moving pixels must start in free cells"

    out = g.copy()
    while True:
        moves = _movable_components(out, bg, moving, blocked)
        if not moves:
            break
        for coords, _ in moves:
            out[coords[:, 0], coords[:, 1]] = bg
        for _, shifted in moves:
            out[shifted[:, 0], shifted[:, 1]] = moving

    _assert_output_separated(out, obstacle, moving)
    return out

# --------------- generator (att2) ---------------

# Canvas bounds.
GRID_H_RANGE = (4, 10)
GRID_W_RANGE = (7, 20)

# Obstacle profile constraints.
OBSTACLE_COVERAGE_RANGE = (0.08, 0.52)
POCKET_COUNT_RANGE = (1, 5)
DISTINCT_HEIGHT_COUNT_RANGE = (2, 4)
OBSTACLE_COMPONENT_RANGE = (1, 3)
BOTTOM_BAND_ROWS = 3
BOTTOM_COLS_MIN_MULTIPLIER = 3
COLUMN_AVG_MAX_NUM = 13
COLUMN_AVG_MAX_DEN = 4
BOTTOM3_MIN_NUM = 3
BOTTOM3_MIN_DEN = 5

# Bottom-heavy profile sampling.
PROFILE_RETRIES = 120
BOTTOM_HEAVY_WEIGHT = 5
LOW_COLUMN_EXTRA_PROB = 0.35
LOW_COLUMN_HEIGHT2_PROB = 0.25
TALL_RUN_COUNT_MAX = 3
TALL_RUN_WIDTH2_MIN_W = 10
TALL_RUN_WIDTH2_PROB = 0.25
MEDIUM_RUN_PROB = 0.45
MEDIUM_RUN_WIDTH_MAX = 4
SHORT_OBSTACLE_MAX_H = 4
MEDIUM_RUN_HEIGHT_MIN = 2

# Sparse post profile sampling.
WIDTH_PLACEMENT_RETRIES = 80
SPARSE_POST_MIN_BOTTOM_FRAC = 3
SPARSE_POST_MAX_BOTTOM_FRAC = 0.45
SPARSE_POST_RUN_COUNT_MAX = 3
SPARSE_POST_HEIGHT1_PROB = 0.8

# Moving object sampling and validation.
PALETTE_SIZE = 10
OBJECT_COUNT_RANGE = (1, 3)
OBJECT_AREA_RANGE = (1, 21)
OBJECT_BBOX_MAX_H = 6
OBJECT_BBOX_MAX_W = 7
MIN_LIFT_ROWS = 1
MAX_LIFT_ROWS = 4

# Whole-input retry budget.
GENERATE_RETRIES = 300


def _spread_extra(widths, extra):
    widths = list(widths)
    for _ in range(int(extra)):
        widths[random.randrange(len(widths))] += 1
    return widths


def _blocked_from_heights(heights, H):
    first = []
    for h in heights:
        first.append(H - 1 if h == H else max(0, h - 1))
    return np.array(first, dtype=int)


def _pocket_count(heights):
    min_h = min(heights)
    count = 0
    in_run = False
    for h in heights:
        if h > min_h:
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def _placement_runs(heights, H):
    """Return constant-depth pocket runs as (left, right, first_blocked_row)."""
    first_blocked = _blocked_from_heights(heights, H)
    min_h = min(heights)
    runs = []
    start = None
    depth = None
    for c, h in enumerate(heights + [min_h]):
        d = int(first_blocked[c]) if c < len(heights) else None
        ok = c < len(heights) and h > min_h and d >= 2
        if ok and start is None:
            start = c
            depth = d
        elif ok and d == depth:
            pass
        else:
            if start is not None:
                runs.append((start, c - 1, int(depth)))
                start = None
            if ok:
                start = c
                depth = d
    return runs


def _profile_is_valid(heights, H, W):
    if heights is None or len(heights) != W:
        return False
    min_distinct, max_distinct = DISTINCT_HEIGHT_COUNT_RANGE
    if not (min_distinct <= len(set(heights)) <= max_distinct):
        return False
    min_pockets, max_pockets = POCKET_COUNT_RANGE
    if not (min_pockets <= _pocket_count(heights) <= max_pockets):
        return False

    area = sum(H - h for h in heights if h < H)
    coverage = area / (H * W)
    min_coverage, max_coverage = OBSTACLE_COVERAGE_RANGE
    if not (min_coverage <= coverage <= max_coverage):
        return False
    if not any(h < H for h in heights):
        return False

    probe = np.zeros((H, W), dtype=int)
    for c, h in enumerate(heights):
        if h < H:
            probe[h:, c] = 1
    n_components = len(find_components(probe, bg=0, colors=[1], connectivity=4))
    min_components, max_components = OBSTACLE_COMPONENT_RANGE
    if not (min_components <= n_components <= max_components):
        return False
    return bool(_placement_runs(heights, H))


def _starts_from_actual_heights(actual_heights, H):
    return [H - h if h > 0 else H for h in actual_heights]


def _bottom_profile_is_eval_like(actual_heights, H, W):
    occupied = [h for h in actual_heights if h > 0]
    if not occupied:
        return False

    bottom_cols = len(occupied)
    total = sum(occupied)
    bottom3 = sum(min(h, BOTTOM_BAND_ROWS) for h in occupied)
    return (
        bottom_cols * BOTTOM_COLS_MIN_MULTIPLIER >= W
        and total * COLUMN_AVG_MAX_DEN <= bottom_cols * COLUMN_AVG_MAX_NUM
        and bottom3 * BOTTOM3_MIN_DEN >= total * BOTTOM3_MIN_NUM
    )


def _profile_bottom_heavy(H, W):
    for _ in range(PROFILE_RETRIES):
        actual = [1] * W

        if random.random() < LOW_COLUMN_EXTRA_PROB:
            for c in range(W):
                if random.random() < LOW_COLUMN_HEIGHT2_PROB:
                    actual[c] = 2

        n_runs = random.randint(1, min(TALL_RUN_COUNT_MAX, W))
        for _ in range(n_runs):
            width = 1
            if W >= TALL_RUN_WIDTH2_MIN_W and random.random() < TALL_RUN_WIDTH2_PROB:
                width = 2
            start = random.randint(0, W - width)
            tall = random.randint(min(H, SHORT_OBSTACLE_MAX_H), H)
            for c in range(start, start + width):
                actual[c] = max(actual[c], tall)

        if random.random() < MEDIUM_RUN_PROB:
            width = random.randint(1, min(MEDIUM_RUN_WIDTH_MAX, W))
            start = random.randint(0, W - width)
            medium = random.randint(MEDIUM_RUN_HEIGHT_MIN, min(H, SHORT_OBSTACLE_MAX_H))
            for c in range(start, start + width):
                actual[c] = max(actual[c], medium)

        heights = _starts_from_actual_heights(actual, H)
        if _bottom_profile_is_eval_like(actual, H, W) and _profile_is_valid(heights, H, W):
            return heights
    return None


def _place_width_runs(W, widths):
    total = sum(widths)
    if total > W:
        return None

    for _ in range(WIDTH_PLACEMENT_RETRIES):
        order = list(range(len(widths)))
        random.shuffle(order)
        occupied = [False] * W
        starts = [None] * len(widths)
        ok = True
        for idx in order:
            width = widths[idx]
            candidates = []
            for start in range(W - width + 1):
                if any(occupied[start : start + width]):
                    continue
                left_clear = start == 0 or not occupied[start - 1]
                right_clear = start + width == W or not occupied[start + width]
                if left_clear and right_clear:
                    candidates.append(start)
            if not candidates:
                ok = False
                break
            start = random.choice(candidates)
            starts[idx] = start
            for c in range(start, start + width):
                occupied[c] = True
        if ok:
            return starts
    return None


def _profile_sparse_posts(H, W):
    min_coverage, _ = OBSTACLE_COVERAGE_RANGE
    min_cols = max(
        math.ceil(W / SPARSE_POST_MIN_BOTTOM_FRAC),
        math.ceil(min_coverage * H * W),
    )
    max_cols = min(W - 1, math.floor(SPARSE_POST_MAX_BOTTOM_FRAC * W))
    if min_cols > max_cols:
        return None

    for _ in range(PROFILE_RETRIES):
        total_cols = random.randint(min_cols, max_cols)
        n_runs = random.randint(1, min(SPARSE_POST_RUN_COUNT_MAX, total_cols))
        widths = _spread_extra([1] * n_runs, total_cols - n_runs)

        starts = _place_width_runs(W, widths)
        if starts is None:
            continue

        actual = [0] * W
        for start, width in zip(starts, widths):
            post_h = 1 if random.random() < SPARSE_POST_HEIGHT1_PROB else min(2, H)
            for c in range(start, start + width):
                actual[c] = post_h

        heights = _starts_from_actual_heights(actual, H)
        if _bottom_profile_is_eval_like(actual, H, W) and _profile_is_valid(heights, H, W):
            return heights
    return None


def _make_profile(H, W):
    builders = [_profile_bottom_heavy] * BOTTOM_HEAVY_WEIGHT + [_profile_sparse_posts]
    random.shuffle(builders)
    for builder in builders:
        heights = builder(H, W)
        if _profile_is_valid(heights, H, W):
            return heights
    return None


def _shape_mask(max_h, max_w):
    """Sample a 4-connected object mask with a solid bottom row."""
    max_h = max(1, min(OBJECT_BBOX_MAX_H, int(max_h)))
    max_w = max(1, min(OBJECT_BBOX_MAX_W, int(max_w)))
    choices = []
    weights = []
    for h0 in range(1, max_h + 1):
        for w0 in range(1, max_w + 1):
            lo = w0 + max(0, h0 - 1)
            hi = min(OBJECT_AREA_RANGE[1], h0 * w0)
            if lo <= hi:
                choices.append((h0, w0, lo, hi))
                weights.append(w0 * w0)
    h, w, min_area, max_area = random.choices(choices, weights=weights, k=1)[0]

    target_area = random.randint(min_area, max_area)
    mask = np.zeros((h, w), dtype=bool)
    mask[-1, :] = True
    area = w

    prev_cols = list(range(w))
    for r in range(h - 2, -1, -1):
        c = random.choice(prev_cols)
        mask[r, c] = True
        area += 1
        prev_cols = [c]

    while area < target_area:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if (
                    (r > 0 and mask[r - 1, c])
                    or (r + 1 < h and mask[r + 1, c])
                    or (c > 0 and mask[r, c - 1])
                    or (c + 1 < w and mask[r, c + 1])
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True
        area += 1
    return mask


def _choose_runs(runs, n_objects):
    pool = list(runs)
    chosen = []
    for _ in range(n_objects):
        weights = [(r[1] - r[0] + 1) ** 2 for r in pool]
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(idx))
    return chosen


def _stamp(grid, mask, r0, c0, bg, moving):
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = moving
    paste(grid, patch, r0, c0, transparent=bg)


def _render_obstacles(H, W, heights, bg, obstacle):
    base = np.full((H, W), bg, dtype=int)
    for c, h in enumerate(heights):
        if h < H:
            base[h:, c] = obstacle
    return base


def _place_settled_objects(base, runs, bg, moving):
    """Return settled output plus object specs (mask, row, col, lift)."""
    n_objects = random.randint(OBJECT_COUNT_RANGE[0], min(OBJECT_COUNT_RANGE[1], len(runs)))
    chosen = _choose_runs(runs, n_objects)

    out = base.copy()
    objects = []
    for c0_run, c1_run, free_depth in chosen:
        width = c1_run - c0_run + 1
        mask = _shape_mask(free_depth - 1, width)
        h, w = mask.shape
        c0 = random.randint(c0_run, c1_run - w + 1)
        r0 = free_depth - h
        if r0 < 1:
            return None, None
        shift = random.randint(MIN_LIFT_ROWS, min(MAX_LIFT_ROWS, r0))
        objects.append((mask, r0, c0, shift))
        _stamp(out, mask, r0, c0, bg, moving)
    return out, objects


def _lift_objects(base, objects, bg, moving):
    inp = base.copy()
    for mask, r0, c0, shift in objects:
        _stamp(inp, mask, r0 - shift, c0, bg, moving)
    return inp


def _components_are_valid(inp, bg, moving):
    comps = find_components(inp, bg=bg, colors=[moving], connectivity=4)
    if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
        return False
    min_area, max_area = OBJECT_AREA_RANGE
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if not (
            min_area <= comp["area"] <= max_area
            and r1 - r0 + 1 <= OBJECT_BBOX_MAX_H
            and c1 - c0 + 1 <= OBJECT_BBOX_MAX_W
        ):
            return False
    return True


def _validate_generated_pair(base, inp, out, bg, obstacle, moving):
    blocked = _blocked_mask(base, obstacle)
    if np.any(blocked & (inp == moving)) or np.any(blocked & (out == moving)):
        return False

    try:
        _assert_output_separated(out, obstacle, moving)
    except AssertionError:
        return False

    if not _components_are_valid(inp, bg, moving):
        return False

    counts = np.bincount(inp.ravel(), minlength=PALETTE_SIZE)
    if counts[bg] <= max(counts[obstacle], counts[moving]):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt():
    """Build one output-first sample and validate it against the solver oracle."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    heights = _make_profile(H, W)
    if heights is None:
        return None

    bg, obstacle, moving = random.sample(range(PALETTE_SIZE), 3)
    base = _render_obstacles(H, W, heights, bg, obstacle)
    runs = _placement_runs(heights, H)

    out, objects = _place_settled_objects(base, runs, bg, moving)
    if out is None:
        return None
    inp = _lift_objects(base, objects, bg, moving)

    if not _validate_generated_pair(base, inp, out, bg, obstacle, moving):
        return None
    return inp


def generate_input():
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 825aa9e9 input")
