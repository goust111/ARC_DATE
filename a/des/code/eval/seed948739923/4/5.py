import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_TO_DIR = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}

# Generator configuration.
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
LEGEND_LEN_CHOICES = (3, 4)
LEGEND_COL_CHOICES = (1, 2)
MIN_EVENT_DIST = 3
MIN_SPACE_FOR_PORTAL = 4
MIN_SPACE_FOR_BOUNDARY = 2
MAX_PORTAL_PAIRS = 4
DECOR_WALL_MAX = 3
GENERATE_RETRIES = 400


def _find_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = {int(color): int(np.count_nonzero(g == color)) for color in np.unique(g)}
    left_limit = min(w - 1, max(2, w // 3))
    candidates: list[tuple[int, int, int, int, list[tuple[int, int]], list[int]]] = []

    for c in range(left_limit + 1):
        r = 0
        while r < h:
            if int(g[r, c]) in (BG, WALL):
                r += 1
                continue
            r0 = r
            colors: list[int] = []
            while r < h and int(g[r, c]) not in (BG, WALL):
                colors.append(int(g[r, c]))
                r += 1
            seg_len = len(colors)
            if seg_len not in (3, 4) or len(set(colors)) != seg_len:
                continue

            exact_pattern = (
                counts.get(colors[0], 0) == 2
                and all(counts.get(color, 0) == 1 for color in colors[1:])
            )
            side_bg = 0
            for rr in range(r0, r):
                for dc in (-1, 1):
                    cc = c + dc
                    if 0 <= cc < w and int(g[rr, cc]) == BG:
                        side_bg += 1
            coords = [(rr, c) for rr in range(r0, r)]
            candidates.append((int(exact_pattern), side_bg, -c, -r0, coords, colors))

    if not candidates:
        raise AssertionError("failed to find the legend column")

    candidates.sort(reverse=True)
    _score0, _score1, _score2, _score3, coords, colors = candidates[0]
    return coords, colors


def _find_launcher(
    grid: np.ndarray, first_color: int, legend_coords: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_set = set(legend_coords)
    launchers: list[tuple[int, int]] = []

    for r, c in np.argwhere(g == first_color):
        rr, cc = int(r), int(c)
        if (rr, cc) in legend_set:
            continue
        if rr in (0, h - 1) or cc in (0, w - 1):
            launchers.append((rr, cc))

    assert len(launchers) == 1, "expected exactly one border launcher"
    r, c = launchers[0]
    if r == 0:
        assert 0 < c < w - 1
        return (r, c), SIDE_TO_DIR["top"]
    if r == h - 1:
        assert 0 < c < w - 1
        return (r, c), SIDE_TO_DIR["bottom"]
    if c == 0:
        assert 0 < r < h - 1
        return (r, c), SIDE_TO_DIR["left"]
    assert c == w - 1 and 0 < r < h - 1
    return (r, c), SIDE_TO_DIR["right"]


def _portal_pairs(grid: np.ndarray, legend_colors: list[int]) -> dict[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    legend_set = set(legend_colors)
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for color in np.unique(g):
        col = int(color)
        if col in legend_set or col in (BG, WALL):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == col)]
        assert len(coords) == 2, "each portal color must appear exactly twice"
        a, b = coords
        pairs[a] = b
        pairs[b] = a
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a ray from the edge launcher, teleporting through same-color portal pairs
    and cycling the ray color according to the left-side legend after every teleport.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_coords, legend_colors = _find_legend(g)
    (r, c), (dr, dc) = _find_launcher(g, legend_colors[0], legend_coords)
    portals = _portal_pairs(g, legend_colors)
    portal_cells = set(portals)

    out = g.copy()
    color_index = 0
    h, w = g.shape
    seen_states: set[tuple[int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(color_index))
        if state in seen_states:
            break
        seen_states.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(g[nr, nc]) == WALL:
            break
        if (nr, nc) in portal_cells:
            r, c = portals[(nr, nc)]
            color_index = (color_index + 1) % len(legend_colors)
            continue

        out[nr, nc] = legend_colors[color_index]
        r, c = nr, nc

    return out


def _major_limits(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side in ("left", "right"):
        return w - 1, (-1 if side == "right" else 1)
    return h - 1, (-1 if side == "bottom" else 1)


def _coord_from_minor_major(side: str, minor: int, major: int) -> tuple[int, int]:
    if side in ("left", "right"):
        return minor, major
    return major, minor


def _distance_forward(sign: int, src_major: int, dst_major: int) -> int:
    return sign * (dst_major - src_major)


def _space_to_boundary(sign: int, major: int, max_major: int) -> int:
    return (max_major - major) if sign > 0 else major


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    legend_len = random.choice(LEGEND_LEN_CHOICES)
    legend_col_choices = [c for c in LEGEND_COL_CHOICES if c < w - 2]
    if not legend_col_choices:
        return None
    legend_col = random.choice(legend_col_choices)
    legend_r0 = random.randint(1, h - legend_len - 2)
    legend_colors = random.sample(COLOR_POOL, legend_len)
    for i, color in enumerate(legend_colors):
        grid[legend_r0 + i, legend_col] = color

    side = random.choice(tuple(SIDE_TO_DIR))
    stop_mode = random.choice(("wall", "boundary"))
    max_major, sign = _major_limits(side, grid.shape)

    if side in ("left", "right"):
        blocked_minors = set(range(max(1, legend_r0 - 1), min(h - 1, legend_r0 + legend_len + 1)))
        available_minors = [row for row in range(1, h - 1) if row not in blocked_minors]
    else:
        blocked_minors = set(range(max(1, legend_col - 1), min(w - 1, legend_col + 2)))
        available_minors = [col for col in range(1, w - 1) if col not in blocked_minors]

    remaining_colors = [color for color in COLOR_POOL if color not in legend_colors]
    max_pairs = min(MAX_PORTAL_PAIRS, len(remaining_colors), len(available_minors) - 1)
    if max_pairs < 1:
        return None
    portal_count = random.randint(1, max_pairs)
    portal_colors = random.sample(remaining_colors, portal_count)

    occupied = {(legend_r0 + i, legend_col) for i in range(legend_len)}

    current_minor = random.choice(available_minors)
    current_major = 0 if sign > 0 else max_major
    launcher = _coord_from_minor_major(side, current_minor, current_major)
    if launcher in occupied:
        return None
    grid[launcher] = legend_colors[0]
    occupied.add(launcher)

    used_minors = {current_minor}
    path_minors = {current_minor}

    for idx, portal_color in enumerate(portal_colors):
        remaining_after = portal_count - idx - 1
        entrance_candidates = []
        for major in range(1, max_major):
            if _distance_forward(sign, current_major, major) < MIN_EVENT_DIST:
                continue
            coord = _coord_from_minor_major(side, current_minor, major)
            if coord not in occupied:
                entrance_candidates.append(major)
        if not entrance_candidates:
            return None
        entrance_major = random.choice(entrance_candidates)
        entrance_coord = _coord_from_minor_major(side, current_minor, entrance_major)

        next_minor_choices = [minor for minor in available_minors if minor not in used_minors]
        if not next_minor_choices:
            return None
        next_minor = random.choice(next_minor_choices)

        required_space = (
            MIN_SPACE_FOR_PORTAL
            if remaining_after > 0 or stop_mode == "wall"
            else MIN_SPACE_FOR_BOUNDARY
        )
        exit_candidates = []
        for major in range(1, max_major):
            if _space_to_boundary(sign, major, max_major) < required_space:
                continue
            coord = _coord_from_minor_major(side, next_minor, major)
            if coord not in occupied and coord != entrance_coord:
                exit_candidates.append(major)
        if not exit_candidates:
            return None
        exit_major = random.choice(exit_candidates)
        exit_coord = _coord_from_minor_major(side, next_minor, exit_major)

        grid[entrance_coord] = portal_color
        grid[exit_coord] = portal_color
        occupied.add(entrance_coord)
        occupied.add(exit_coord)

        current_minor = next_minor
        current_major = exit_major
        used_minors.add(current_minor)
        path_minors.add(current_minor)

    if stop_mode == "wall":
        wall_candidates = []
        for major in range(1, max_major):
            if _distance_forward(sign, current_major, major) < MIN_EVENT_DIST:
                continue
            coord = _coord_from_minor_major(side, current_minor, major)
            if coord not in occupied:
                wall_candidates.append(coord)
        if not wall_candidates:
            return None
        wall_coord = random.choice(wall_candidates)
        grid[wall_coord] = WALL
        occupied.add(wall_coord)
    elif _space_to_boundary(sign, current_major, max_major) < MIN_SPACE_FOR_BOUNDARY:
        return None

    decor_candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in occupied:
                continue
            if side in ("left", "right") and r in path_minors:
                continue
            if side in ("top", "bottom") and c in path_minors:
                continue
            decor_candidates.append((r, c))
    random.shuffle(decor_candidates)
    for coord in decor_candidates[: random.randint(0, min(DECOR_WALL_MAX, len(decor_candidates)))]:
        grid[coord] = WALL
        occupied.add(coord)

    try:
        found_legend_coords, found_legend_colors = _find_legend(grid)
        if found_legend_colors != legend_colors or set(found_legend_coords) != {
            (legend_r0 + i, legend_col) for i in range(legend_len)
        }:
            return None
        found_launcher, found_dir = _find_launcher(grid, legend_colors[0], found_legend_coords)
        if found_launcher != launcher or found_dir != SIDE_TO_DIR[side]:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
