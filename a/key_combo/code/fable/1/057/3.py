from __future__ import annotations

from collections import Counter

import numpy as np


CELL_INNER = 3
CELL_STEP = CELL_INNER + 1
FULL_RING_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
GAP_CELL_BY_DIR = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
RING_MASKS = {}
for _dir_name, (_rr, _cc) in GAP_CELL_BY_DIR.items():
    _mask = FULL_RING_MASK.copy()
    _mask[_rr, _cc] = False
    RING_MASKS[_dir_name] = _mask
del _dir_name, _rr, _cc, _mask

BG_COLOR = 0
SEPARATOR_CHOICES = (5, 8)
RING_COLOR_CHOICES = (1, 2, 3, 4, 6, 7, 9)
GRID_ROWS_RANGE = (4, 7)
GRID_COLS_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray, *, exclude: set[int] | None = None) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    for color, _count in counts.most_common():
        if exclude is None or color not in exclude:
            return int(color)
    raise ValueError("no color available")


def _infer_separator(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue
        if h != CELL_STEP * (len(row_lines) - 1) + 1:
            continue
        if w != CELL_STEP * (len(col_lines) - 1) + 1:
            continue
        if not np.all(np.diff(row_lines) == CELL_STEP):
            continue
        if not np.all(np.diff(col_lines) == CELL_STEP):
            continue
        candidates.append((int(color), row_lines, col_lines))
    assert len(candidates) == 1, "expected exactly one regular separator grid"
    return candidates[0]


def _cell_patch(row_lines: np.ndarray, col_lines: np.ndarray, r: int, c: int) -> tuple[int, int]:
    return int(row_lines[r] + 1), int(col_lines[c] + 1)


def _parse_grid(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray, np.ndarray, dict[tuple[int, int], dict[str, int | str]]]:
    g = np.asarray(grid, dtype=int)
    sep_color, row_lines, col_lines = _infer_separator(g)
    center_rows = row_lines[:-1] + 2
    center_cols = col_lines[:-1] + 2
    center_values = [int(g[r, c]) for r in center_rows for c in center_cols]
    bg = Counter(center_values).most_common(1)[0][0]

    rings: dict[tuple[int, int], dict[str, int | str]] = {}
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            r0, c0 = _cell_patch(row_lines, col_lines, rr, cc)
            patch = g[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER]
            assert patch.shape == (CELL_INNER, CELL_INNER)
            if np.all(patch == bg):
                continue

            colors = set(map(int, patch.ravel())) - {bg}
            assert len(colors) == 1, "ring cell must contain exactly one non-background color"
            color = colors.pop()
            mask = patch == color
            direction = None
            for name, templ in RING_MASKS.items():
                if np.array_equal(mask, templ):
                    direction = name
                    break
            assert direction is not None, "expected a 3x3 hollow ring with one cardinal gap"
            rings[(rr, cc)] = {"color": int(color), "direction": direction}

    return sep_color, bg, len(row_lines) - 1, len(col_lines) - 1, row_lines, col_lines, rings


def _trace_path(
    rows: int,
    cols: int,
    rings: dict[tuple[int, int], dict[str, int | str]],
) -> tuple[list[tuple[int, int]], int]:
    color_counts = Counter(int(info["color"]) for info in rings.values())
    assert len(color_counts) == 2, "expected one start color and one shared ring color"
    start_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(start_colors) == 1, "expected exactly one odd-colored start ring"
    start_color = int(start_colors[0])
    start = next(cell for cell, info in rings.items() if int(info["color"]) == start_color)

    seen: set[tuple[int, int]] = set()
    path: list[tuple[int, int]] = []
    current = start
    while 0 <= current[0] < rows and 0 <= current[1] < cols:
        assert current in rings, "every visited cell must contain a ring"
        assert current not in seen, "pointer chain must not revisit a cell"
        seen.add(current)
        path.append(current)
        dr, dc = DIR_DELTA[str(rings[current]["direction"])]
        current = (current[0] + dr, current[1] + dc)

    assert path, "path must visit at least the start ring"
    return path, start_color


def _full_ring_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((CELL_INNER, CELL_INNER), bg, dtype=int)
    patch[FULL_RING_MASK] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the gap-encoded ring pointers from the odd-colored start and only seal the rings that get executed.

    Input structure:
    - A single-color separator lattice partitions the canvas into regular 3x3 interior cells.
    - Each non-empty cell contains one monochrome hollow 3x3 ring missing exactly one edge midpoint.
    - Exactly one ring has a unique color; it is the start, while all other rings share the same color.

    Rule:
    - Interpret each ring's missing midpoint as a pointer to the neighboring coarse cell in that cardinal direction.
    - Start from the uniquely colored ring and keep stepping from ring to ring, reading the next direction each time, until the pointer exits the coarse-grid boundary.
    - For every visited ring, fill the missing midpoint to make a complete square ring and recolor that entire ring to the start color; leave unvisited rings, separators, and empty cells unchanged.

    Key insight:
    - The gap is not damage to repair globally; it is an instruction stream, so only the rings reached by executing the pointer chain are completed and recolored.
    """
    g = np.asarray(grid, dtype=int)
    _sep, bg, rows, cols, row_lines, col_lines, rings = _parse_grid(g)
    path, start_color = _trace_path(rows, cols, rings)

    out = g.copy()
    start_patch = _full_ring_patch(start_color, bg)
    for rr, cc in path:
        r0, c0 = _cell_patch(row_lines, col_lines, rr, cc)
        out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = start_patch
    return out


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    neighbors = []
    for dr, dc in DIR_DELTA.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def _on_boundary(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def _sample_path(rows: int, cols: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], bool]:
    total = rows * cols
    require_turn = bool(rng.random() < 0.75)
    min_len = 4 if require_turn else 2
    min_len = min(min_len, total - 2)
    max_len = min(total - 2, min_len + int(rng.integers(2, rows + cols + 1)))

    starts = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    if not starts:
        starts = [(r, c) for r in range(rows) for c in range(cols)]

    for _ in range(MAX_PATH_ATTEMPTS):
        start = starts[int(rng.integers(len(starts)))]
        path = [start]
        visited = {start}
        prev_dir = None
        turned = False

        while len(path) < max_len:
            current = path[-1]
            if _on_boundary(current, rows, cols) and len(path) >= min_len and (turned or not require_turn):
                if rng.random() < 0.45 or len(path) == max_len:
                    break

            options = []
            candidates = _coarse_neighbors(current, rows, cols)
            rng.shuffle(candidates)
            for nxt in candidates:
                if nxt in visited:
                    continue
                ndir = (nxt[0] - current[0], nxt[1] - current[1])
                next_turned = turned or (prev_dir is not None and ndir != prev_dir)
                if len(path) + 1 < min_len and _on_boundary(nxt, rows, cols):
                    weight = 1
                elif require_turn and not turned and prev_dir is not None and ndir != prev_dir:
                    weight = 4
                else:
                    weight = 2 if not _on_boundary(nxt, rows, cols) else 1
                for _repeat in range(weight):
                    options.append((nxt, ndir, next_turned))
            if not options:
                break

            nxt, prev_dir, turned = options[int(rng.integers(len(options)))]
            path.append(nxt)
            visited.add(nxt)

        if _on_boundary(path[-1], rows, cols) and len(path) >= min_len and (turned or not require_turn):
            return path, require_turn

    raise RuntimeError("failed to sample a valid pointer path")


def _exit_direction(cell: tuple[int, int], rows: int, cols: int, rng: np.random.Generator) -> str:
    r, c = cell
    options = []
    if r == 0:
        options.append("up")
    if r == rows - 1:
        options.append("down")
    if c == 0:
        options.append("left")
    if c == cols - 1:
        options.append("right")
    assert options, "final path cell must touch the boundary"
    return str(options[int(rng.integers(len(options)))])


def _direction_from_step(a: tuple[int, int], b: tuple[int, int]) -> str:
    delta = (b[0] - a[0], b[1] - a[1])
    for name, step in DIR_DELTA.items():
        if step == delta:
            return name
    raise ValueError("invalid step")


def _ring_patch_with_gap(color: int, bg: int, direction: str) -> np.ndarray:
    patch = np.full((CELL_INNER, CELL_INNER), bg, dtype=int)
    patch[RING_MASKS[direction]] = int(color)
    return patch


def _build_grid(
    rows: int,
    cols: int,
    sep_color: int,
    base_color: int,
    start_color: int,
    path: list[tuple[int, int]],
    decoys: dict[tuple[int, int], str],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.full((rows * CELL_STEP + 1, cols * CELL_STEP + 1), bg, dtype=int)
    grid[::CELL_STEP, :] = sep_color
    grid[:, ::CELL_STEP] = sep_color

    directions: dict[tuple[int, int], str] = {}
    for idx, cell in enumerate(path[:-1]):
        directions[cell] = _direction_from_step(cell, path[idx + 1])
    directions[path[-1]] = _exit_direction(path[-1], rows, cols, rng)

    for cell, direction in directions.items():
        r0, c0 = 1 + cell[0] * CELL_STEP, 1 + cell[1] * CELL_STEP
        color = start_color if cell == path[0] else base_color
        grid[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = _ring_patch_with_gap(color, bg, direction)

    for cell, direction in decoys.items():
        r0, c0 = 1 + cell[0] * CELL_STEP, 1 + cell[1] * CELL_STEP
        grid[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = _ring_patch_with_gap(base_color, bg, direction)

    return grid


def _naive_complete_all(grid: np.ndarray, recolor: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _sep, bg, _rows, _cols, row_lines, col_lines, rings = _parse_grid(g)
    out = g.copy()
    for (rr, cc), info in rings.items():
        r0, c0 = _cell_patch(row_lines, col_lines, rr, cc)
        color = int(info["color"]) if recolor is None else int(recolor)
        out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = _full_ring_patch(color, bg)
    return out


def _naive_recolor_visited_without_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _sep, bg, rows, cols, row_lines, col_lines, rings = _parse_grid(g)
    path, start_color = _trace_path(rows, cols, rings)
    out = g.copy()
    for rr, cc in path:
        r0, c0 = _cell_patch(row_lines, col_lines, rr, cc)
        patch = out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER]
        patch[(patch != bg)] = start_color
    return out


def _validate_instance(grid: np.ndarray, *, require_turn: bool) -> None:
    g = np.asarray(grid, dtype=int)
    _sep, _bg, rows, cols, _row_lines, _col_lines, rings = _parse_grid(g)
    path, start_color = _trace_path(rows, cols, rings)
    assert len(path) >= 2, "path must have at least two rings"
    assert len(rings) > len(path), "at least one unvisited ring is required"
    assert len(rings) < rows * cols, "at least one empty cell is required"
    turns = sum(
        (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        != (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        for i in range(2, len(path))
    )
    if require_turn:
        assert turns > 0, "turn branch was requested"

    out = main(g)
    assert not np.array_equal(out, g), "output must change the grid"
    assert not np.array_equal(out, _naive_complete_all(g, recolor=None)), "filling every ring is invalid"
    assert not np.array_equal(out, _naive_complete_all(g, recolor=start_color)), "recoloring every ring is invalid"
    assert not np.array_equal(out, _naive_recolor_visited_without_fill(g)), "recolor-only shortcut is invalid"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1] + 1))
    cols = int(rng.integers(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1] + 1))
    total = rows * cols
    if total < 6:
        return None

    path, require_turn = _sample_path(rows, cols, rng)
    remaining_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in set(path)]
    if len(remaining_cells) < 2:
        return None

    sep_color = int(SEPARATOR_CHOICES[int(rng.integers(len(SEPARATOR_CHOICES)))])
    ring_palette = [color for color in RING_COLOR_CHOICES if color != sep_color]
    chosen = rng.choice(ring_palette, size=2, replace=False)
    base_color, start_color = int(chosen[0]), int(chosen[1])

    max_decoys = min(len(remaining_cells) - 1, max(2, (2 * len(remaining_cells)) // 3))
    if max_decoys < 1:
        return None
    decoy_count = int(rng.integers(1, max_decoys + 1))
    rng.shuffle(remaining_cells)
    decoy_cells = remaining_cells[:decoy_count]
    direction_names = tuple(DIR_DELTA.keys())
    decoys = {cell: str(direction_names[int(rng.integers(len(direction_names)))]) for cell in decoy_cells}

    grid = _build_grid(rows, cols, sep_color, base_color, start_color, path, decoys, BG_COLOR, rng)
    _validate_instance(grid, require_turn=require_turn)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-gap-pointer task")
