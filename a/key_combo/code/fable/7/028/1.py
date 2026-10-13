from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
FILL_COLORS = (3, 4, 5, 6, 7, 8, 9)
TILE_SIZE = 5
STRIDE = TILE_SIZE + 1
CENTER = 2.0
MAX_ATTEMPTS = 400

DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MASK_LIBRARY = {
    (-1, 0): (
        ((0, 2), (1, 1), (1, 2), (1, 3)),
        ((0, 1), (0, 2), (0, 3)),
        ((0, 2), (1, 2)),
    ),
    (1, 0): (
        ((4, 2), (3, 1), (3, 2), (3, 3)),
        ((4, 1), (4, 2), (4, 3)),
        ((4, 2), (3, 2)),
    ),
    (0, -1): (
        ((1, 1), (2, 0), (2, 1), (3, 1)),
        ((1, 0), (2, 0), (3, 0)),
        ((2, 0), (2, 1)),
    ),
    (0, 1): (
        ((1, 3), (2, 4), (2, 3), (3, 3)),
        ((1, 4), (2, 4), (3, 4)),
        ((2, 4), (2, 3)),
    ),
    (-1, -1): (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0)),
    ),
    (-1, 1): (
        ((0, 4), (0, 3), (1, 4)),
        ((0, 4), (1, 4), (1, 3)),
        ((0, 3), (1, 4)),
    ),
    (1, -1): (
        ((4, 0), (4, 1), (3, 0)),
        ((4, 0), (3, 0), (3, 1)),
        ((4, 1), (3, 0)),
    ),
    (1, 1): (
        ((4, 4), (4, 3), (3, 4)),
        ((4, 4), (3, 4), (3, 3)),
        ((4, 3), (3, 4)),
    ),
}


def _sign(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def _constant_color(line: np.ndarray) -> int | None:
    values = np.unique(line)
    if len(values) != 1:
        return None
    return int(values[0])


def _infer_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    row_colors = [_constant_color(grid[r, :]) for r in range(h)]
    col_colors = [_constant_color(grid[:, c]) for c in range(w)]
    candidates = {v for v in row_colors if v is not None} & {v for v in col_colors if v is not None}
    for color in sorted(candidates):
        sep_rows = [r for r, v in enumerate(row_colors) if v == color]
        sep_cols = [c for c, v in enumerate(col_colors) if v == color]
        if len(sep_rows) < 3 or len(sep_cols) < 3:
            continue
        if sep_rows[0] != 0 or sep_rows[-1] != h - 1 or sep_cols[0] != 0 or sep_cols[-1] != w - 1:
            continue
        if any(b - a != STRIDE for a, b in zip(sep_rows, sep_rows[1:])):
            continue
        if any(b - a != STRIDE for a, b in zip(sep_cols, sep_cols[1:])):
            continue
        row_runs = [(r + 1, r + TILE_SIZE) for r in sep_rows[:-1]]
        col_runs = [(c + 1, c + TILE_SIZE) for c in sep_cols[:-1]]
        if all(r1 < h and r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_runs) and all(
            c1 < w and c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_runs
        ):
            return color, row_runs, col_runs
    raise AssertionError("failed to infer a regular 5x5 tile lattice")


def _trace_chain(
    shape: tuple[int, int],
    start: tuple[int, int],
    directions: dict[tuple[int, int], tuple[int, int]],
) -> tuple[list[tuple[int, int]], str]:
    rows, cols = shape
    visited: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    current = start
    while True:
        visited.append(current)
        seen.add(current)
        dr, dc = directions[current]
        nxt = (current[0] + dr, current[1] + dc)
        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
            return visited, "exit"
        if nxt in seen:
            return visited, "revisit"
        current = nxt


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_color, row_runs, col_runs = _infer_lattice(g)
    sep_mask = np.zeros_like(g, dtype=bool)
    for r0, r1 in row_runs:
        sep_mask[r0 : r1 + 1, :] = False
    sep_mask[::STRIDE, :] = True
    sep_mask[:, ::STRIDE] = True

    fill_color = Counter(map(int, g[~sep_mask].ravel())).most_common(1)[0][0]
    rows, cols = len(row_runs), len(col_runs)
    total_tiles = rows * cols

    tile_info: dict[tuple[int, int], dict] = {}
    marker_counts: Counter[int] = Counter()
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (TILE_SIZE, TILE_SIZE)
            colors = set(map(int, np.unique(tile)))
            assert separator_color not in colors
            assert fill_color in colors
            marker_colors = list(colors - {fill_color})
            assert len(marker_colors) == 1, "each tile must contain exactly one fragment color"
            marker_color = int(marker_colors[0])
            coords = np.argwhere(tile == marker_color)
            assert coords.size > 0
            centroid = coords.mean(axis=0)
            direction = (_sign(float(centroid[0] - CENTER)), _sign(float(centroid[1] - CENTER)))
            assert direction != (0, 0), "fragment centroid may not coincide with tile center"
            pos = (tile_r, tile_c)
            tile_info[pos] = {"marker_color": marker_color, "direction": direction}
            marker_counts[marker_color] += 1

    assert len(marker_counts) == 2, "expected one start color and one ordinary pointer color"
    counts = sorted(marker_counts.values())
    assert counts == [1, total_tiles - 1], "expected exactly one unique-color start tile"
    start_color = next(color for color, count in marker_counts.items() if count == 1)
    start_tiles = [pos for pos, info in tile_info.items() if info["marker_color"] == start_color]
    assert len(start_tiles) == 1
    start = start_tiles[0]

    directions = {pos: info["direction"] for pos, info in tile_info.items()}
    visited, stop_reason = _trace_chain((rows, cols), start, directions)
    return {
        "separator_color": separator_color,
        "fill_color": fill_color,
        "rows": rows,
        "cols": cols,
        "start": start,
        "directions": directions,
        "visited": visited,
        "stop_reason": stop_reason,
    }


def _render_output(shape: tuple[int, int], visited: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r, c in visited:
        out[r, c] = RED
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Follow the hidden pointer chain across 5x5 tiles and mark the visited coarse cells.

    Input structure:
    - Full separator rows and columns partition the canvas into an m×n array of 5×5 tiles.
    - Every tile has one dominant fill color plus one small fragment; exactly one fragment uses a unique start color and all others share another marker color.
    - Each fragment's centroid is offset from the tile center in one of the eight neighbor directions.

    Rule:
    - For each tile, compare the fragment centroid with the 5×5 geometric center and take the row/column offset signs as a pointer to one adjacent tile, including diagonals.
    - Start at the unique-color fragment tile and repeatedly jump to the pointed neighbor.
    - Stop before the first jump that would leave the tile array or revisit a previously visited tile, then output an m×n coarse grid with visited cells red and all others black.

    Key insight:
    - The fragment shape itself is not the payload; only the centroid direction matters, turning the tile array into a hidden directed graph with a single visible entry point.
    """
    analysis = _analyze_input(np.asarray(input_grid, dtype=int))
    return _render_output((analysis["rows"], analysis["cols"]), analysis["visited"])


def _neighbors(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    r, c = cell
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _is_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    rows, cols = shape
    r, c = cell
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def _has_turn(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return True
    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return any(deltas[i] != deltas[i - 1] for i in range(1, len(deltas)))


def _find_path(rows: int, cols: int, length: int, stop_mode: str, rng: random.Random) -> list[tuple[int, int]] | None:
    shape = (rows, cols)
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(starts)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if stop_mode == "exit" and not _is_border(shape, path[-1]):
                return None
            if not _has_turn(path):
                return None
            return list(path)

        current = path[-1]
        nxts = [cell for cell in _neighbors(shape, current) if cell not in used]
        rng.shuffle(nxts)
        for nxt in nxts:
            if stop_mode == "exit" and len(path) + 1 == length and not _is_border(shape, nxt):
                continue
            path.append(nxt)
            used.add(nxt)
            result = dfs(path, used)
            if result is not None:
                return result
            used.remove(nxt)
            path.pop()
        return None

    for start in starts:
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _exit_directions(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    r, c = cell
    return [(dr, dc) for dr, dc in DIRS if not (0 <= r + dr < rows and 0 <= c + dc < cols)]


def _revisit_directions(path: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    current = path[-1]
    previous = path[-2]
    visited = set(path[:-1])
    all_dirs = []
    non_previous = []
    for vr, vc in visited:
        dr, dc = vr - current[0], vc - current[1]
        if (dr, dc) in DIRS:
            all_dirs.append((dr, dc))
            if (vr, vc) != previous:
                non_previous.append((dr, dc))
    if non_previous and rng.random() < 0.7:
        return non_previous
    return all_dirs


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 1 + STRIDE * tile_r, 1 + STRIDE * tile_c


def _render_input(
    rows: int,
    cols: int,
    fill_color: int,
    start: tuple[int, int],
    directions: dict[tuple[int, int], tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((rows * STRIDE + 1, cols * STRIDE + 1), fill_color, dtype=int)
    grid[::STRIDE, :] = BLACK
    grid[:, ::STRIDE] = BLACK

    for tile_r in range(rows):
        for tile_c in range(cols):
            marker_color = RED if (tile_r, tile_c) == start else BLUE
            mask = rng.choice(MASK_LIBRARY[directions[(tile_r, tile_c)]])
            r0, c0 = _tile_origin(tile_r, tile_c)
            for dr, dc in mask:
                grid[r0 + dr, c0 + dc] = marker_color
    return grid


def _build_case(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    total_tiles = rows * cols
    if total_tiles < 4:
        return None

    stop_mode = rng.choice(("exit", "revisit"))
    min_length = 2 if total_tiles <= 4 else 3
    candidate_lengths = list(range(min_length, total_tiles))
    rng.shuffle(candidate_lengths)

    path = None
    for length in candidate_lengths:
        path = _find_path(rows, cols, length, stop_mode, rng)
        if path is not None:
            break
    if path is None:
        return None

    directions: dict[tuple[int, int], tuple[int, int]] = {}
    for src, dst in zip(path, path[1:]):
        directions[src] = (dst[0] - src[0], dst[1] - src[1])

    if stop_mode == "exit":
        exit_dirs = _exit_directions((rows, cols), path[-1])
        if not exit_dirs:
            return None
        directions[path[-1]] = rng.choice(exit_dirs)
    else:
        revisit_dirs = _revisit_directions(path, rng)
        if not revisit_dirs:
            return None
        directions[path[-1]] = rng.choice(revisit_dirs)

    for tile_r in range(rows):
        for tile_c in range(cols):
            directions.setdefault((tile_r, tile_c), rng.choice(DIRS))

    fill_color = rng.choice(FILL_COLORS)
    grid = _render_input(rows, cols, fill_color, path[0], directions, rng)
    expected = _render_output((rows, cols), path)
    spec = {
        "rows": rows,
        "cols": cols,
        "start": path[0],
        "path": path,
        "directions": directions,
        "stop_mode": stop_mode,
        "expected": expected,
    }
    return grid, spec


def _validate_generated_case(grid: np.ndarray, spec: dict) -> bool:
    try:
        analysis = _analyze_input(grid)
    except Exception:
        return False
    if analysis["start"] != spec["start"]:
        return False
    if analysis["directions"] != spec["directions"]:
        return False
    if analysis["visited"] != spec["path"]:
        return False
    if analysis["stop_reason"] != spec["stop_mode"]:
        return False
    solved = main(grid)
    if not np.array_equal(solved, spec["expected"]):
        return False
    if solved.shape != (spec["rows"], spec["cols"]):
        return False
    return int(np.sum(solved == RED)) >= 2 and int(np.sum(solved == BLACK)) >= 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        built = _build_case(rng)
        if built is None:
            continue
        grid, spec = built
        if _validate_generated_case(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain ARC input")
