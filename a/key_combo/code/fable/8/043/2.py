from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, get_bg


BACKGROUND = Color.BLACK
VISITED_COLOR = Color.BLUE
START_COLOR = Color.BLUE
MARKER_COLOR = Color.RED
SEPARATOR_COLOR = Color.GREY

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
COUNT_TO_TURN = {1: 0, 2: 1, 3: 2, 4: 3}
TURN_TO_COUNT = {value: key for key, value in COUNT_TO_TURN.items()}
EDGE_NAMES = {UP: "top", RIGHT: "right", DOWN: "bottom", LEFT: "left"}
TURN_REQUIREMENTS = (
    {1, 2, 3},
    {1, 2, 4},
    {1, 3, 4},
    {2, 3, 4},
)
GEN_RETRIES = 800
PATH_RETRIES = 300

_GEN_CALL_COUNT = 0
_SEEN_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(v) for v in separators) + [int(length)]
    bands = []
    for start, stop in zip(bounds, bounds[1:]):
        inner_start = start + 1
        inner_stop = stop
        assert inner_start < inner_stop, "separator lines must leave non-empty regions"
        bands.append((inner_start, inner_stop))
    return bands


def _extract_regions(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = get_bg(g)

    colors = sorted(set(map(int, g.reshape(-1))) - {int(bg)})
    separator_candidates = []
    for color in colors:
        sep_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if sep_rows and sep_cols:
            separator_candidates.append((int(color), sep_rows, sep_cols))
    assert len(separator_candidates) == 1, "expected exactly one separator color"

    separator_color, sep_rows, sep_cols = separator_candidates[0]
    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected multiple regions"

    non_separator = [color for color in colors if color != separator_color]
    start_candidates = [color for color in non_separator if int(np.count_nonzero(g == color)) == 1]
    assert len(start_candidates) == 1, "expected exactly one singleton start color"
    start_color = int(start_candidates[0])
    marker_colors = [color for color in non_separator if color != start_color]
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker_color = int(marker_colors[0])

    counts = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    start_region = None
    initial_dir = None
    allowed = {int(bg), separator_color, start_color, marker_color}
    assert set(map(int, g.reshape(-1))) <= allowed, "unexpected extra colors present"

    for out_r, (r0, r1) in enumerate(row_bands):
        for out_c, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            assert region.size > 0
            assert separator_color not in set(map(int, region.reshape(-1))), "separator leaked into a region"

            marker_count = int(np.count_nonzero(region == marker_color))
            assert 1 <= marker_count <= 4, "every region must contain 1..4 marker pixels"
            counts[out_r, out_c] = marker_count

            start_pixels = np.argwhere(region == start_color)
            assert len(start_pixels) <= 1, "a region may contain at most one start pixel"
            if len(start_pixels) == 0:
                continue

            assert start_region is None, "expected exactly one start region"
            local_r, local_c = map(int, start_pixels[0])
            region_h, region_w = map(int, region.shape)
            edges = []
            if local_r == 0:
                edges.append(UP)
            if local_c == region_w - 1:
                edges.append(RIGHT)
            if local_r == region_h - 1:
                edges.append(DOWN)
            if local_c == 0:
                edges.append(LEFT)
            assert len(edges) == 1, "start pixel must touch exactly one region edge"
            start_region = (out_r, out_c)
            initial_dir = edges[0]

    assert start_region is not None and initial_dir is not None, "missing start pixel"
    return counts, start_region, int(initial_dir)


def _trace_path(
    counts: np.ndarray,
    start_region: tuple[int, int],
    initial_dir: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    rows, cols = map(int, counts.shape)
    r, c = map(int, start_region)
    facing = int(initial_dir)
    visited = []
    move_dirs = []
    seen = set()

    for _ in range(rows * cols + 1):
        if not (0 <= r < rows and 0 <= c < cols):
            return visited, move_dirs
        cell = (int(r), int(c))
        assert cell not in seen, "route must not self-intersect"
        seen.add(cell)
        visited.append(cell)

        turn = COUNT_TO_TURN[int(counts[r, c])]
        facing = (facing + turn) % 4
        move_dirs.append(facing)
        dr, dc = DIRS[facing]
        r += dr
        c += dc

    raise AssertionError("route failed to leave the board")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read per-region marker counts as relative turn instructions and output the visited region summary.

    Input structure
    - A grey full-row/full-column separator lattice partitions the board into an m×n table of rectangular regions.
    - Every region contains 1..4 red marker pixels, and exactly one region also contains a single blue start pixel.
    - The blue start pixel lies on exactly one edge of its region, which encodes the initial facing direction.

    Rule
    - Count the red markers in each region and interpret the count as a relative turn command:
      1=straight, 2=right, 3=turn back, 4=left.
    - Start in the unique blue-marked region with the facing direction given by the touched edge.
    - On each visited region, first update the current facing by that region's command, then move one region in the new direction.
      Continue until the walk leaves the partitioned board.

    Key insight
    - The marker count does not name an absolute direction; it only tells how to rotate the current facing, so the whole path is a stateful chain walk.
    """
    counts, start_region, initial_dir = _extract_regions(grid)
    path, _move_dirs = _trace_path(counts, start_region, initial_dir)

    out = np.full(counts.shape, BACKGROUND, dtype=int)
    for r, c in path:
        out[int(r), int(c)] = VISITED_COLOR
    return out


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    for direction, (er, ec) in DIRS.items():
        if (dr, dc) == (er, ec):
            return direction
    raise AssertionError("path cells must be 4-neighbors")


def _outside_dirs(cell: tuple[int, int], rows: int, cols: int) -> list[int]:
    r, c = map(int, cell)
    out = []
    if r == 0:
        out.append(UP)
    if c == cols - 1:
        out.append(RIGHT)
    if r == rows - 1:
        out.append(DOWN)
    if c == 0:
        out.append(LEFT)
    return out


def _encode_path_counts(
    path: list[tuple[int, int]],
    initial_dir: int,
    exit_dir: int,
) -> tuple[list[int], list[int]]:
    move_dirs = [_direction_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    move_dirs.append(int(exit_dir))

    counts = []
    facing = int(initial_dir)
    for move_dir in move_dirs:
        relative_turn = (int(move_dir) - facing) % 4
        counts.append(TURN_TO_COUNT[relative_turn])
        facing = int(move_dir)
    return counts, move_dirs


def _weighted_choice(scored_moves: list[tuple[float, tuple[int, int]]]) -> tuple[int, int]:
    weights = [max(0.05, score) for score, _move in scored_moves]
    total = sum(weights)
    pick = random.random() * total
    for weight, (_score, move) in zip(weights, scored_moves):
        pick -= weight
        if pick <= 0:
            return move
    return scored_moves[-1][1]


def _random_path(rows: int, cols: int, min_len: int, max_len: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(starts)

    for start in starts:
        path = [start]
        used = {start}
        while len(path) < max_len:
            cur_r, cur_c = path[-1]
            unused_neighbors = []
            for dr, dc in DIRS.values():
                nxt = (cur_r + dr, cur_c + dc)
                if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in used:
                    future_used = used | {nxt}
                    future_degree = 0
                    for fdr, fdc in DIRS.values():
                        rr = nxt[0] + fdr
                        cc = nxt[1] + fdc
                        if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in future_used:
                            future_degree += 1
                    boundary_bonus = 0.4 if _outside_dirs(nxt, rows, cols) else 0.0
                    score = 1.0 + future_degree + boundary_bonus + random.random()
                    unused_neighbors.append((score, nxt))

            can_exit = len(path) >= min_len and bool(_outside_dirs(path[-1], rows, cols))
            if can_exit and (not unused_neighbors or random.random() < 0.35):
                return path
            if not unused_neighbors:
                break

            nxt = _weighted_choice(unused_neighbors)
            path.append(nxt)
            used.add(nxt)

        if len(path) >= min_len and _outside_dirs(path[-1], rows, cols):
            return path
    return None


def _path_requirements_met(
    counts: list[int],
    move_dirs: list[int],
    required_turns: set[int],
) -> bool:
    if len(counts) < 5:
        return False
    if len(set(counts)) < 3:
        return False
    if not required_turns.issubset(set(counts)):
        return False

    per_count_dirs: dict[int, set[int]] = {}
    for count, move_dir in zip(counts, move_dirs):
        per_count_dirs.setdefault(int(count), set()).add(int(move_dir))
    if not any(len(dir_set) >= 2 for dir_set in per_count_dirs.values()):
        return False
    return True


def _generate_path_plan(
    rows: int,
    cols: int,
    required_start_dir: int,
    required_turns: set[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    total = rows * cols
    min_len = 5
    max_len = min(total - 2, random.randint(6, min(10, total - 2)))
    if max_len < min_len:
        return None

    for _ in range(PATH_RETRIES):
        path = _random_path(rows, cols, min_len, max_len)
        if path is None or len(path) > total - 2:
            continue
        exit_dirs = _outside_dirs(path[-1], rows, cols)
        random.shuffle(exit_dirs)
        for exit_dir in exit_dirs:
            counts, move_dirs = _encode_path_counts(path, required_start_dir, exit_dir)
            if _path_requirements_met(counts, move_dirs, required_turns):
                return path, counts, move_dirs
    return None


def _sample_partition_sizes(count: int) -> list[int]:
    return [random.randint(3, 5) for _ in range(count)]


def _band_starts(sizes: list[int]) -> list[int]:
    starts = []
    cursor = 0
    for index, size in enumerate(sizes):
        starts.append(cursor)
        cursor += int(size)
        if index != len(sizes) - 1:
            cursor += 1
    return starts


def _edge_cell(region_h: int, region_w: int, direction: int) -> tuple[int, int]:
    if direction == UP:
        return 0, random.randint(1, region_w - 2)
    if direction == RIGHT:
        return random.randint(1, region_h - 2), region_w - 1
    if direction == DOWN:
        return region_h - 1, random.randint(1, region_w - 2)
    if direction == LEFT:
        return random.randint(1, region_h - 2), 0
    raise AssertionError("invalid direction")


def _place_region_objects(
    grid: np.ndarray,
    top: int,
    left: int,
    region_h: int,
    region_w: int,
    marker_count: int,
    start_dir: int | None,
) -> None:
    forbidden = set()
    if start_dir is not None:
        sr, sc = _edge_cell(region_h, region_w, start_dir)
        grid[top + sr, left + sc] = START_COLOR
        forbidden.add((sr, sc))

    all_cells = [(r, c) for r in range(region_h) for c in range(region_w) if (r, c) not in forbidden]
    assert len(all_cells) >= marker_count
    marker_cells = random.sample(all_cells, marker_count)
    for mr, mc in marker_cells:
        grid[top + mr, left + mc] = MARKER_COLOR


def _expected_output(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for r, c in path:
        out[int(r), int(c)] = VISITED_COLOR
    return out


def _simulate_with_alt_start(
    counts: np.ndarray,
    start_region: tuple[int, int],
    initial_dir: int,
) -> tuple[tuple[tuple[int, int], ...], bool]:
    try:
        path, _move_dirs = _trace_path(counts, start_region, initial_dir)
    except AssertionError:
        return tuple(), False
    return tuple(path), True


def _validate_generated(
    grid: np.ndarray,
    expected_path: list[tuple[int, int]],
    required_turns: set[int],
) -> bool:
    try:
        counts, start_region, initial_dir = _extract_regions(grid)
        path, move_dirs = _trace_path(counts, start_region, initial_dir)
    except AssertionError:
        return False

    if path != expected_path:
        return False

    path_counts = [int(counts[r, c]) for r, c in path]
    if not _path_requirements_met(path_counts, move_dirs, required_turns):
        return False

    if len(path) >= counts.size - 1:
        return False

    off_path = [(r, c) for r in range(counts.shape[0]) for c in range(counts.shape[1]) if (r, c) not in set(path)]
    if len(off_path) < 2:
        return False
    if not any(int(counts[r, c]) in set(path_counts) for r, c in off_path):
        return False

    alt_paths = []
    for alt_dir in (UP, RIGHT, DOWN, LEFT):
        if alt_dir == initial_dir:
            continue
        alt_path, ok = _simulate_with_alt_start(counts, start_region, alt_dir)
        alt_paths.append((alt_path, ok))
    if not any((not ok) or alt_path != tuple(path) for alt_path, ok in alt_paths):
        return False

    expected = _expected_output(counts.shape, expected_path)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    return True


def _gen_attempt(call_index: int) -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    required_start_dir = call_index % 4
    required_turns = TURN_REQUIREMENTS[call_index % len(TURN_REQUIREMENTS)]

    path_plan = _generate_path_plan(rows, cols, required_start_dir, required_turns)
    if path_plan is None:
        return None
    path, path_counts, _move_dirs = path_plan

    counts = np.zeros((rows, cols), dtype=int)
    path_cells = set(path)
    off_path_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in path_cells]
    if len(off_path_cells) < 2:
        return None

    missing_counts = [count for count in (1, 2, 3, 4) if count not in set(path_counts)]
    random.shuffle(off_path_cells)
    for cell, count in zip(off_path_cells, missing_counts):
        counts[cell] = int(count)
    for cell in off_path_cells[len(missing_counts) :]:
        counts[cell] = random.randint(1, 4)
    for cell, count in zip(path, path_counts):
        counts[cell] = int(count)

    if not any(counts[cell] in set(path_counts) for cell in off_path_cells):
        counts[off_path_cells[0]] = random.choice(path_counts)

    row_sizes = _sample_partition_sizes(rows)
    col_sizes = _sample_partition_sizes(cols)
    grid_h = sum(row_sizes) + rows - 1
    grid_w = sum(col_sizes) + cols - 1
    if grid_h > 30 or grid_w > 30:
        return None

    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    for index in range(rows - 1):
        grid[row_starts[index] + row_sizes[index], :] = SEPARATOR_COLOR
    for index in range(cols - 1):
        grid[:, col_starts[index] + col_sizes[index]] = SEPARATOR_COLOR

    start_region = path[0]
    for r in range(rows):
        for c in range(cols):
            start_dir = required_start_dir if (r, c) == start_region else None
            _place_region_objects(
                grid,
                row_starts[r],
                col_starts[c],
                row_sizes[r],
                col_sizes[c],
                int(counts[r, c]),
                start_dir,
            )

    if not _validate_generated(grid, path, required_turns):
        return None

    sig = (tuple(map(int, grid.shape)), grid.tobytes())
    if sig in _SEEN_SIGNATURES:
        return None
    _SEEN_SIGNATURES.add(sig)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for offset in range(GEN_RETRIES):
        grid = _gen_attempt(_GEN_CALL_COUNT + offset)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid turning-compass input")
