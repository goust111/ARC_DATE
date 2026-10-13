import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 1
OBSTACLE = 5
DIR_COLORS = {
    (-1, 0): 2,  # up -> red
    (0, -1): 7,  # left -> orange
    (0, 1): 6,   # right -> magenta
    (1, 0): 8,   # down -> cyan
}
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _adjacent_to_obstacle(grid: np.ndarray, pos: tuple[int, int], obstacle: int) -> bool:
    r, c = pos
    return any(
        in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == obstacle
        for nr, nc in neighbors4(r, c)
    )


def _turn_vectors(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(dr, -1), (dr, 1)]
    return [(-1, dc), (1, dc)]


def _marker_seed_directions(
    grid: np.ndarray, marker: tuple[int, int], bg: int, obstacle: int
) -> list[tuple[int, int]]:
    r, c = marker
    dirs = []
    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            continue
        if int(grid[nr, nc]) != bg:
            continue
        if _adjacent_to_obstacle(grid, (nr, nc), obstacle):
            dirs.append((dr, dc))
    return dirs


def _stamp_cross(out: np.ndarray, inp: np.ndarray, marker: tuple[int, int], bg: int) -> None:
    r, c = marker
    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if in_bounds(inp.shape, nr, nc) and int(inp[nr, nc]) == bg:
            out[nr, nc] = DIR_COLORS[(dr, dc)]


def _extend_from_seed(
    out: np.ndarray,
    inp: np.ndarray,
    seed: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    bg: int,
    obstacle: int,
) -> None:
    for tr, tc in _turn_vectors(direction):
        r, c = seed
        while True:
            nr, nc = r + tr, c + tc
            if not in_bounds(inp.shape, nr, nc):
                break
            if int(inp[nr, nc]) != bg:
                break
            if int(out[nr, nc]) != bg:
                break
            if not _adjacent_to_obstacle(inp, (nr, nc), obstacle):
                break
            out[nr, nc] = color
            r, c = nr, nc


def _single_marker_footprint(
    base: np.ndarray, marker: tuple[int, int], direction: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    footprint = {marker}
    outline = set()
    r, c = marker

    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if in_bounds(base.shape, nr, nc) and int(base[nr, nc]) == BG:
            footprint.add((nr, nc))

    seed = (r + direction[0], c + direction[1])
    outline.add(seed)
    footprint.add(seed)

    for tr, tc in _turn_vectors(direction):
        rr, cc = seed
        while True:
            nr, nc = rr + tr, cc + tc
            if not in_bounds(base.shape, nr, nc):
                break
            if int(base[nr, nc]) != BG:
                break
            if not _adjacent_to_obstacle(base, (nr, nc), OBSTACLE):
                break
            outline.add((nr, nc))
            footprint.add((nr, nc))
            rr, cc = nr, nc

    return footprint, outline


def _width_profiles() -> list[list[int]]:
    return [
        [1, 3, 5, 3, 1],
        [1, 3, 5, 5, 3, 1],
        [2, 4, 4, 2],
        [2, 4, 6, 4, 2],
        [1, 3, 4, 3, 1],
        [2, 3, 5, 3, 2],
    ]


def _build_obstacle_patch() -> np.ndarray:
    widths = random.choice(_width_profiles())
    max_w = max(widths)
    patch = np.zeros((len(widths), max_w), dtype=int)

    lefts = []
    current_left = (max_w - widths[0]) // 2
    lefts.append(current_left)
    for width in widths[1:]:
        base_left = (max_w - width) // 2
        candidates = [base_left]
        if lefts[-1] - 1 <= base_left <= lefts[-1] + 1:
            candidates.append(base_left)
        current_left = random.choice(candidates)
        lefts.append(current_left)

    if random.random() < 0.5:
        skew = 0
        for idx in range(len(lefts)):
            if idx > 0:
                skew += random.choice([-1, 0, 1])
                skew = max(-1, min(1, skew))
            lefts[idx] += skew
        min_left = min(lefts)
        max_right = max(left + width for left, width in zip(lefts, widths))
        shift = 0
        if min_left < 0:
            shift -= min_left
        if max_right > max_w:
            shift -= max_right - max_w
        lefts = [left + shift for left in lefts]

    for r, (left, width) in enumerate(zip(lefts, widths)):
        patch[r, left : left + width] = OBSTACLE

    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = patch.T
    return patch


def _candidate_markers(base: np.ndarray) -> list[dict]:
    candidates = []
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BG:
                continue
            marker = (r, c)
            dirs = _marker_seed_directions(base, marker, BG, OBSTACLE)
            if len(dirs) != 1:
                continue
            arm_count = 0
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                if in_bounds(base.shape, nr, nc) and int(base[nr, nc]) == BG:
                    arm_count += 1
            if arm_count < 3:
                continue
            direction = dirs[0]
            footprint, outline = _single_marker_footprint(base, marker, direction)
            if len(outline) < 3:
                continue
            candidates.append(
                {
                    "marker": marker,
                    "direction": direction,
                    "footprint": footprint,
                    "outline": outline,
                }
            )
    random.shuffle(candidates)
    return candidates


def _choose_marker_set(base: np.ndarray) -> list[dict] | None:
    candidates = _candidate_markers(base)
    if len(candidates) < 3:
        return None

    target_count = random.randint(3, 4)
    choices_by_dir: dict[tuple[int, int], list[dict]] = {}
    for cand in candidates:
        choices_by_dir.setdefault(cand["direction"], []).append(cand)

    dir_orders = list(choices_by_dir.keys())
    if len(dir_orders) < 2:
        return None
    random.shuffle(dir_orders)

    for _ in range(120):
        selected: list[dict] = []
        occupied: set[tuple[int, int]] = set()
        random.shuffle(dir_orders)

        preferred_dirs = dir_orders[: random.randint(2, min(4, len(dir_orders)))]
        trial_pool = []
        for direction in preferred_dirs:
            trial_pool.extend(choices_by_dir[direction])
        random.shuffle(trial_pool)
        trial_pool.extend(candidates)

        for cand in trial_pool:
            if cand["marker"] in occupied:
                continue
            if occupied & cand["footprint"]:
                continue
            selected.append(cand)
            occupied |= cand["footprint"]
            if len(selected) == target_count:
                break

        used_dirs = {cand["direction"] for cand in selected}
        if len(selected) >= 3 and len(used_dirs) >= 2:
            return selected

    return None


def _build_input_attempt() -> np.ndarray | None:
    obstacle_patch = _build_obstacle_patch()
    ph, pw = obstacle_patch.shape

    rows = random.randint(max(12, ph + 6), max(13, ph + 10))
    cols = random.randint(max(12, pw + 6), max(13, pw + 10))
    if rows > 30 or cols > 30:
        return None

    base = np.zeros((rows, cols), dtype=int)
    r0 = random.randint(2, rows - ph - 3)
    c0 = random.randint(2, cols - pw - 3)
    base[r0 : r0 + ph, c0 : c0 + pw] = obstacle_patch

    selected = _choose_marker_set(base)
    if selected is None:
        return None

    inp = base.copy()
    for cand in selected:
        r, c = cand["marker"]
        inp[r, c] = MARKER

    if not _valid_generated_input(inp):
        return None
    _ = main(inp)
    return inp


def _valid_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BG, MARKER, OBSTACLE}:
        return False

    obstacle_cells = np.argwhere(grid == OBSTACLE)
    if not (12 <= len(obstacle_cells) <= 20):
        return False

    seen = set()
    stack = [tuple(map(int, obstacle_cells[0]))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nr, nc in neighbors4(*cell):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == OBSTACLE and (nr, nc) not in seen:
                stack.append((nr, nc))
    if len(seen) != len(obstacle_cells):
        return False

    markers = [tuple(map(int, p)) for p in np.argwhere(grid == MARKER)]
    if len(markers) not in (3, 4):
        return False

    full_footprints = []
    used_dirs = set()
    obstacle_only = grid.copy()
    obstacle_only[obstacle_only == MARKER] = BG
    for marker in markers:
        dirs = _marker_seed_directions(obstacle_only, marker, BG, OBSTACLE)
        if len(dirs) != 1:
            return False
        direction = dirs[0]
        used_dirs.add(direction)
        footprint, outline = _single_marker_footprint(obstacle_only, marker, direction)
        if len(outline) < 3:
            return False
        arm_count = sum(
            1
            for dr, dc in CARDINALS
            if in_bounds(grid.shape, marker[0] + dr, marker[1] + dc)
            and int(obstacle_only[marker[0] + dr, marker[1] + dc]) == BG
        )
        if arm_count < 3:
            return False
        full_footprints.append(footprint)

    for i in range(len(full_footprints)):
        for j in range(i + 1, len(full_footprints)):
            if full_footprints[i] & full_footprints[j]:
                return False

    return len(used_dirs) >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp direction-colored crosses on blue markers and extend the obstacle-touching arm along the obstacle contour.

    Input structure:
    - The background is black, with one connected gray obstacle region and 3-4 isolated blue marker pixels.
    - Each marker has exactly one cardinal neighbor cell that is black and 4-adjacent to the gray obstacle.
    - Markers are spaced so their crosses and contour extensions do not overlap.

    Rule:
    - Keep every blue marker unchanged and stamp the four cardinal background neighbors with fixed direction colors:
      up -> red, left -> orange, right -> magenta, down -> cyan.
    - For each marker, find the unique direction whose adjacent black neighbor sits on the obstacle contour.
    - Use that direction's color as the outline color, then extend from that stamped seed cell in the two diagonal contour directions
      that keep stepping through black cells while remaining 4-adjacent to the gray obstacle.

    Key insight:
    The cross always uses the same compass-color mapping, but only the arm that touches the obstacle becomes a longer contour-tracing stroke,
    so spatial orientation around the obstacle determines which color spreads.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    obstacle = OBSTACLE

    markers = [tuple(map(int, p)) for p in np.argwhere(g == MARKER)]
    assert len(markers) > 0

    dirs_by_marker: dict[tuple[int, int], tuple[int, int]] = {}
    obstacle_only = g.copy()
    obstacle_only[obstacle_only == MARKER] = bg
    for marker in markers:
        dirs = _marker_seed_directions(obstacle_only, marker, bg, obstacle)
        assert len(dirs) == 1, "each marker must have exactly one contour-touching direction"
        dirs_by_marker[marker] = dirs[0]

    out = g.copy()
    for marker in markers:
        _stamp_cross(out, g, marker, bg)

    for marker in sorted(markers):
        direction = dirs_by_marker[marker]
        seed = (marker[0] + direction[0], marker[1] + direction[1])
        color = DIR_COLORS[direction]
        out[seed] = color
        _extend_from_seed(out, obstacle_only, seed, direction, color, bg, obstacle)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/003 puzzle 2")
