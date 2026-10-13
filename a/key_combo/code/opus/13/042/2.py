from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")

HEIGHT_RANGE = (9, 13)
PANEL_WIDTH_RANGE = (5, 8)
WALL_COUNT_OPTIONS = (5, 6, 7)
MAX_GENERATION_ATTEMPTS = 500


def _find_separator_column(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    candidates = [c for c in range(width) if np.all(grid[:, c] == BLUE)]
    assert len(candidates) == 1, "expected exactly one full-height blue divider"
    sep = candidates[0]
    assert 0 < sep < width - 1, "divider must split two non-empty panels"
    assert sep * 2 + 1 == width, "divider must split equal-width panels"
    assert not np.any(grid[:, :sep] == BLUE) and not np.any(grid[:, sep + 1 :] == BLUE), "blue must appear only in the divider"
    return sep


def _extract_compact_state(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_column(g)
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()
    assert left.shape == right.shape, "panels must have identical shape"

    allowed = {BLACK, RED, GREEN}
    assert set(map(int, np.unique(left))).issubset(allowed), "unexpected colors in left panel"
    assert set(map(int, np.unique(right))).issubset(allowed), "unexpected colors in right panel"

    left_markers = np.argwhere(left == RED)
    right_markers = np.argwhere(right == RED)
    assert left_markers.shape[0] == 1 and right_markers.shape[0] == 1, "expected one red marker per panel"
    left_marker = tuple(map(int, left_markers[0]))
    right_marker = tuple(map(int, right_markers[0]))
    assert left[left_marker] == RED and right[right_marker] == RED

    walls = (left == GREEN) | (right == GREEN)
    assert not walls[left_marker] and not walls[right_marker], "markers cannot overlap walls"
    assert left_marker != right_marker, "markers must map to different compact positions"
    assert left_marker[0] != right_marker[0] and left_marker[1] != right_marker[1], "markers must have different rows and columns"

    return walls, [left_marker, right_marker], left, right, sep


def _ray_coverages(
    walls: np.ndarray,
    markers: list[tuple[int, int]],
) -> tuple[list[set[tuple[int, int]]], list[dict[str, int]]]:
    height, width = map(int, walls.shape)
    coverages: list[set[tuple[int, int]]] = []
    lengths: list[dict[str, int]] = []

    for r, c in markers:
        covered: set[tuple[int, int]] = set()
        ray_lengths: dict[str, int] = {}
        for name, (dr, dc) in DIRS.items():
            steps = 0
            rr, cc = r + dr, c + dc
            while 0 <= rr < height and 0 <= cc < width and not walls[rr, cc]:
                covered.add((rr, cc))
                steps += 1
                rr += dr
                cc += dc
            ray_lengths[name] = steps
        coverages.append(covered)
        lengths.append(ray_lengths)

    return coverages, lengths


def _render_output(walls: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(walls.shape, BLACK, dtype=int)
    out[walls] = GREY

    coverages, _ = _ray_coverages(walls, markers)
    shared = coverages[0] & coverages[1]
    singles = (coverages[0] | coverages[1]) - shared

    for r, c in singles:
        if not walls[r, c]:
            out[r, c] = RED
    for r, c in shared:
        if not walls[r, c]:
            out[r, c] = YELLOW
    for r, c in markers:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the two side panels into hidden walls that stop the markers' four-way rays.

    Input structure:
    - The grid has a single full-height blue divider column splitting two equal-width panels.
    - Each panel contains black background, scattered green pixels, and exactly one red marker pixel.
    - The left and right panels are aligned by row and column after removing the divider.

    Rule:
    - Remove the blue divider and align the two panels into one compact grid.
    - Mark a compact cell as a grey wall when either aligned panel has a green pixel there; otherwise keep it black.
    - Map both red markers into the compact grid and extend red rays from each marker up, down, left, and right.
    - A ray stops immediately before the first grey wall in that direction, or at the compact-grid boundary if no wall blocks it.
    - Cells reached by rays from both markers become yellow, while each marker cell itself stays red.

    Key insight:
    - The green pixels are not the final objects: their aligned union creates an emergent wall layout, and that hidden wall layout alone determines where the two markers' rays can travel and intersect.
    """
    walls, markers, _left, _right, _sep = _extract_compact_state(grid)
    return _render_output(walls, markers)


def _max_len(marker: tuple[int, int], direction: str, shape: tuple[int, int]) -> int:
    r, c = marker
    height, width = shape
    if direction == "up":
        return r
    if direction == "down":
        return height - 1 - r
    if direction == "left":
        return c
    return width - 1 - c


def _step(marker: tuple[int, int], direction: str, distance: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return marker[0] + dr * distance, marker[1] + dc * distance


def _choose_markers(height: int, width: int) -> list[tuple[int, int]] | None:
    rows = list(range(1, height - 1))
    cols = list(range(1, width - 1))
    row_pairs = [(a, b) for a in rows for b in rows if abs(a - b) >= 2]
    col_pairs = [(a, b) for a in cols for b in cols if abs(a - b) >= 2]
    if not row_pairs or not col_pairs:
        return None
    r1, r2 = random.choice(row_pairs)
    c1, c2 = random.choice(col_pairs)
    return [(r1, c1), (r2, c2)]


def _segment_constraints(markers: list[tuple[int, int]]) -> dict[str, tuple[tuple[int, str], int, tuple[int, int]]]:
    (r1, c1), (r2, c2) = markers
    cross_a = (r1, c2)
    cross_b = (r2, c1)

    horiz_1 = "right" if c2 > c1 else "left"
    horiz_2 = "left" if c1 < c2 else "right"
    vert_1 = "down" if r2 > r1 else "up"
    vert_2 = "up" if r1 < r2 else "down"

    row_gap = abs(r2 - r1)
    col_gap = abs(c2 - c1)
    assert row_gap >= 2 and col_gap >= 2

    crosses = {
        "a": (((0, horiz_1), col_gap), ((1, vert_2), row_gap), cross_a),
        "b": (((0, vert_1), row_gap), ((1, horiz_2), col_gap), cross_b),
    }
    return crosses


def _choose_wall_lengths(
    markers: list[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[tuple[int, str], int] | None:
    crosses = _segment_constraints(markers)
    max_lens = {(idx, direction): _max_len(markers[idx], direction, shape) for idx in range(2) for direction in DIR_ORDER}

    open_crosses = {"a", "b"} if random.random() < 0.45 else {random.choice(("a", "b"))}

    breaker_limits: dict[tuple[int, str], int] = {}
    open_limits: dict[tuple[int, str], int] = {}
    forced_walls: set[tuple[int, str]] = set()

    for cross_name, (seg1, seg2, _cross_pos) in crosses.items():
        if cross_name in open_crosses:
            open_limits[seg1[0]] = max(open_limits.get(seg1[0], 0), seg1[1])
            open_limits[seg2[0]] = max(open_limits.get(seg2[0], 0), seg2[1])
        else:
            breaker = random.choice((seg1, seg2))
            forced_walls.add(breaker[0])
            breaker_limits[breaker[0]] = min(breaker_limits.get(breaker[0], shape[0] + shape[1]), breaker[1] - 2)

    target_wall_count = random.choice(WALL_COUNT_OPTIONS)
    all_segments = [(idx, direction) for idx in range(2) for direction in DIR_ORDER]
    wall_segments = set(forced_walls)

    candidates = all_segments[:]
    random.shuffle(candidates)
    for key in candidates:
        if len(wall_segments) >= target_wall_count:
            break
        if key in wall_segments:
            continue
        max_len = max_lens[key]
        lower = open_limits.get(key, 0)
        upper = max_len - 1
        if key in breaker_limits:
            upper = min(upper, breaker_limits[key])
        if lower <= upper:
            wall_segments.add(key)

    if len(wall_segments) != target_wall_count:
        return None

    lengths: dict[tuple[int, str], int] = {}
    for key in all_segments:
        max_len = max_lens[key]
        if key not in wall_segments:
            lengths[key] = max_len
            continue

        lower = open_limits.get(key, 0)
        upper = max_len - 1
        if key in breaker_limits:
            upper = min(upper, breaker_limits[key])
        if lower > upper:
            return None
        lengths[key] = random.randint(lower, upper)

    for cross_name, (seg1, seg2, _cross_pos) in crosses.items():
        key1, dist1 = seg1
        key2, dist2 = seg2
        cross_open = lengths[key1] >= dist1 and lengths[key2] >= dist2
        if cross_name in open_crosses and not cross_open:
            return None
        if cross_name not in open_crosses and cross_open:
            return None

    for marker_idx in range(2):
        if max(lengths[(marker_idx, direction)] for direction in DIR_ORDER) < 2:
            return None

    return lengths


def _walls_from_lengths(
    markers: list[tuple[int, int]],
    shape: tuple[int, int],
    lengths: dict[tuple[int, str], int],
) -> list[tuple[int, int]]:
    walls: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for key, accessible_len in lengths.items():
        idx, direction = key
        max_len = _max_len(markers[idx], direction, shape)
        if accessible_len >= max_len:
            continue
        wall = _step(markers[idx], direction, accessible_len + 1)
        assert 0 <= wall[0] < shape[0] and 0 <= wall[1] < shape[1]
        assert wall not in seen, "effective walls must be unique"
        seen.add(wall)
        walls.append(wall)
    return walls


def _assign_walls(
    shape: tuple[int, int],
    walls: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.full(shape, BLACK, dtype=int)
    right = np.full(shape, BLACK, dtype=int)

    categories = ["L", "L", "R", "R", "B"]
    if len(walls) >= 6:
        categories.append("B")
    if len(walls) >= 7:
        categories.append("B")
    assert len(categories) == len(walls)

    shuffled_walls = walls[:]
    random.shuffle(shuffled_walls)
    random.shuffle(categories)

    for (r, c), category in zip(shuffled_walls, categories):
        if category in {"L", "B"}:
            left[r, c] = GREEN
        if category in {"R", "B"}:
            right[r, c] = GREEN

    left[markers[0]] = RED
    right[markers[1]] = RED
    return left, right


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape
    grid = np.full((height, width * 2 + 1), BLACK, dtype=int)
    grid[:, :width] = left
    grid[:, width] = BLUE
    grid[:, width + 1 :] = right
    return grid


def _tuple_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _removal_changes_output(grid: np.ndarray, side: str, cells: set[tuple[int, int]], baseline: np.ndarray, sep: int) -> bool:
    for r, c in cells:
        trial = grid.copy()
        if side == "left":
            trial[r, c] = BLACK
        else:
            trial[r, sep + 1 + c] = BLACK
        try:
            changed = not np.array_equal(main(trial), baseline)
        except Exception:
            changed = True
        if changed:
            return True
    return False


def _semantic_valid(grid: np.ndarray) -> bool:
    try:
        walls, markers, left, right, sep = _extract_compact_state(grid)
        out = main(grid)
        if not np.array_equal(out, _render_output(walls, markers)):
            return False
    except Exception:
        return False

    left_green = int(np.count_nonzero(left == GREEN))
    right_green = int(np.count_nonzero(right == GREEN))
    if not (3 <= left_green <= 8 and 3 <= right_green <= 8):
        return False
    if np.array_equal(left == GREEN, right == GREEN):
        return False

    coverages, lengths = _ray_coverages(walls, markers)
    if np.count_nonzero(out == YELLOW) < 1:
        return False
    if not coverages[0] or not coverages[1]:
        return False
    if not all(max(ray_lengths.values()) >= 2 for ray_lengths in lengths):
        return False

    union_mask = walls
    if np.array_equal(union_mask, left == GREEN) or np.array_equal(union_mask, right == GREEN):
        return False

    left_only = _tuple_coords((left == GREEN) & (right != GREEN))
    right_only = _tuple_coords((right == GREEN) & (left != GREEN))
    if not left_only or not right_only:
        return False

    if not _removal_changes_output(grid, "left", left_only, out, sep):
        return False
    if not _removal_changes_output(grid, "right", right_only, out, sep):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*PANEL_WIDTH_RANGE)
    markers = _choose_markers(height, width)
    if markers is None:
        return None

    lengths = _choose_wall_lengths(markers, (height, width))
    if lengths is None:
        return None

    walls = _walls_from_lengths(markers, (height, width), lengths)
    if len(walls) not in WALL_COUNT_OPTIONS:
        return None

    left, right = _assign_walls((height, width), walls, markers)
    grid = _assemble_input(left, right)
    if _semantic_valid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
