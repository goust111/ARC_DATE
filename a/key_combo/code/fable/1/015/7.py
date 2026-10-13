from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import Color, in_bounds, paste
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        rr0 = int(r0)
        cc0 = int(c0)
        h, w = map(int, patch.shape)
        H, W = dst.shape
        if clip:
            dr0 = max(0, rr0)
            dc0 = max(0, cc0)
            dr1 = min(H, rr0 + h)
            dc1 = min(W, cc0 + w)
            if dr0 >= dr1 or dc0 >= dc1:
                return dst
            pr0 = dr0 - rr0
            pc0 = dc0 - cc0
            pr1 = pr0 + (dr1 - dr0)
            pc1 = pc0 + (dc1 - dc0)
            view = patch[pr0:pr1, pc0:pc1]
            if transparent is None:
                dst[dr0:dr1, dc0:dc1] = view
            else:
                mask = view != int(transparent)
                dst_view = dst[dr0:dr1, dc0:dc1]
                dst_view[mask] = view[mask]
            return dst
        if rr0 < 0 or cc0 < 0 or rr0 + h > H or cc0 + w > W:
            raise ValueError("paste out of bounds")
        if transparent is None:
            dst[rr0 : rr0 + h, cc0 : cc0 + w] = patch
        else:
            view = dst[rr0 : rr0 + h, cc0 : cc0 + w]
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


_DIRS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
_TRANSFORMS = (
    lambda r, c: (r, c),
    lambda r, c: (r, -c),
    lambda r, c: (-r, c),
    lambda r, c: (-r, -c),
    lambda r, c: (c, r),
    lambda r, c: (c, -r),
    lambda r, c: (-c, r),
    lambda r, c: (-c, -r),
)
_CYCLE_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 0)),
)
_COLOR_SCHEMES = (
    {"base": Color.BLACK, "frame": Color.GREY, "dot": Color.RED, "seed": Color.YELLOW},
    {"base": Color.BLACK, "frame": Color.BLUE, "dot": Color.PINK, "seed": Color.YELLOW},
    {"base": Color.TEAL, "frame": Color.GREY, "dot": Color.RED, "seed": Color.MAROON},
    {"base": Color.BLACK, "frame": Color.MAROON, "dot": Color.TEAL, "seed": Color.YELLOW},
)
_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _ordered_transform(cells: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    transform = random.choice(_TRANSFORMS)
    transformed = [transform(r, c) for r, c in cells]
    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    return [(int(r - min_r), int(c - min_c)) for r, c in transformed]


def _cell_neighbors(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in _DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _out_dirs(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in _DIRS:
        if not in_bounds(shape, int(r + dr), int(c + dc)):
            out.append((dr, dc))
    return out


def _row_lines_for(step: int, count: int) -> list[int]:
    return [idx * int(step) for idx in range(int(count) + 1)]


def _parse_grid(
    grid: np.ndarray,
) -> tuple[list[int], list[int], int, dict[tuple[int, int], tuple[int, int]], tuple[int, int], dict[tuple[int, int], tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates = []

    for color in np.unique(g):
        color = int(color)
        row_lines = np.flatnonzero(np.all(g == color, axis=1)).tolist()
        col_lines = np.flatnonzero(np.all(g == color, axis=0)).tolist()
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        if row_lines[0] != 0 or row_lines[-1] != H - 1 or col_lines[0] != 0 or col_lines[-1] != W - 1:
            continue

        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if row_steps.size == 0 or col_steps.size == 0:
            continue
        if len(set(map(int, row_steps.tolist()))) != 1 or len(set(map(int, col_steps.tolist()))) != 1:
            continue

        step = int(row_steps[0])
        if int(col_steps[0]) != step or step < 4:
            continue
        side = step - 1
        if side % 2 == 0:
            continue

        expected_rows = _row_lines_for(step, len(row_lines) - 1)
        expected_cols = _row_lines_for(step, len(col_lines) - 1)
        if expected_rows != row_lines or expected_cols != col_lines:
            continue
        if row_lines[-1] != H - 1 or col_lines[-1] != W - 1:
            continue

        center = side // 2
        directions: dict[tuple[int, int], tuple[int, int]] = {}
        dot_colors: dict[tuple[int, int], int] = {}
        dot_pixels: dict[tuple[int, int], tuple[int, int, int]] = {}
        valid = True

        for rr in range(len(row_lines) - 1):
            for cc in range(len(col_lines) - 1):
                top = row_lines[rr] + 1
                left = col_lines[cc] + 1
                sub = g[top : top + side, left : left + side]
                values, counts = np.unique(sub, return_counts=True)
                if len(values) != 2 or sorted(map(int, counts.tolist())) != [1, side * side - 1]:
                    valid = False
                    break
                dot_idx = int(np.argmin(counts))
                dot_color = int(values[dot_idx])
                locs = np.argwhere(sub == dot_color)
                if locs.shape != (1, 2):
                    valid = False
                    break
                dot_r = int(locs[0, 0])
                dot_c = int(locs[0, 1])
                dr = dot_r - center
                dc = dot_c - center
                if (dr, dc) not in _DIRS:
                    valid = False
                    break
                cell = (rr, cc)
                directions[cell] = (dr, dc)
                dot_colors[cell] = dot_color
                dot_pixels[cell] = (top + dot_r, left + dot_c, dot_color)
            if not valid:
                break

        if not valid:
            continue

        color_counts: dict[int, int] = {}
        for dot_color in dot_colors.values():
            color_counts[dot_color] = color_counts.get(dot_color, 0) + 1
        total_cells = len(dot_colors)
        if total_cells < 4:
            continue
        seed_candidates = [dot_color for dot_color, count in color_counts.items() if count == 1]
        normal_candidates = [dot_color for dot_color, count in color_counts.items() if count == total_cells - 1]
        if len(seed_candidates) != 1 or len(normal_candidates) != 1:
            continue
        seed_color = seed_candidates[0]
        seed_cells = [cell for cell, dot_color in dot_colors.items() if dot_color == seed_color]
        if len(seed_cells) != 1:
            continue
        candidates.append((row_lines, col_lines, side, directions, seed_cells[0], dot_pixels))

    assert len(candidates) == 1, "expected one valid framed lattice with a unique seed dot"
    return candidates[0]


def _trace_path(
    shape: tuple[int, int],
    directions: dict[tuple[int, int], tuple[int, int]],
    seed: tuple[int, int],
) -> tuple[list[tuple[int, int]], int | None]:
    visited: list[tuple[int, int]] = []
    seen: dict[tuple[int, int], int] = {}
    current = tuple(seed)

    while True:
        if current in seen:
            return visited, seen[current]
        seen[current] = len(visited)
        visited.append(current)
        dr, dc = directions[current]
        nxt = (int(current[0] + dr), int(current[1] + dc))
        if not in_bounds(shape, nxt[0], nxt[1]):
            return visited, None
        current = nxt


def _paint_result(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    side: int,
    visited: list[tuple[int, int]],
    cycle_start: int | None,
    dot_pixels: dict[tuple[int, int], tuple[int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    orange_patch = np.full((side, side), Color.ORANGE, dtype=int)
    green_patch = np.full((side, side), Color.GREEN, dtype=int)

    for idx, cell in enumerate(visited):
        rr, cc = cell
        top = row_lines[rr] + 1
        left = col_lines[cc] + 1
        patch = green_patch if cycle_start is not None and idx >= cycle_start else orange_patch
        paste(out, patch, top, left)

    for dot_r, dot_c, dot_color in dot_pixels.values():
        out[int(dot_r), int(dot_c)] = int(dot_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the seed cell's off-center dot pointers and color the tail and loop segments differently.

    Input structure:
    - The whole grid is partitioned by one frame color into equal square cells with a complete outer border.
    - Each cell interior contains a uniform base color plus exactly one off-center dot one step away from the cell center.
    - Exactly one dot has a unique seed color; all other dots share one normal color.

    Rule:
    - Read each dot's offset from the cell center as one of the eight neighbor directions on the cell lattice.
    - Starting from the seed cell, jump cell-to-cell by those directions until the path either leaves the lattice or first re-enters a visited cell.
    - If the path exits, paint every visited cell interior orange; if it loops, paint the loop cells green and the earlier tail cells orange, while keeping every frame pixel and every dot unchanged.

    Key insight:
    - The visible dot is a hidden pointer: the real object to transform is the implicit directed walk over cells, split at the first repeated destination into tail versus cycle.
    """
    g = np.asarray(grid, dtype=int)
    row_lines, col_lines, side, directions, seed, dot_pixels = _parse_grid(g)
    visited, cycle_start = _trace_path((len(row_lines) - 1, len(col_lines) - 1), directions, seed)
    return _paint_result(g, row_lines, col_lines, side, visited, cycle_start, dot_pixels)


def _build_backward_chain(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    blocked: set[tuple[int, int]],
    length: int,
) -> list[tuple[int, int]] | None:
    if length == 0:
        return []

    path = [anchor]
    used = set(blocked)
    used.add(anchor)

    def dfs() -> bool:
        if len(path) == length + 1:
            return True
        current = path[-1]
        options = [nb for nb in _cell_neighbors(current, shape) if nb not in used]
        random.shuffle(options)
        options.sort(
            key=lambda cell: sum(1 for nb in _cell_neighbors(cell, shape) if nb not in used),
            reverse=True,
        )
        for nxt in options:
            used.add(nxt)
            path.append(nxt)
            if dfs():
                return True
            path.pop()
            used.remove(nxt)
        return False

    if not dfs():
        return None
    return list(reversed(path[1:]))


def _build_exit_trace(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], int | None, tuple[int, int]] | None:
    rows, cols = shape
    boundary = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if r in {0, rows - 1} or c in {0, cols - 1}
    ]
    max_len = min(rows * cols, max(4, rows + cols + 1))

    for _ in range(120):
        end = random.choice(boundary)
        out_dirs = _out_dirs(end, shape)
        if not out_dirs:
            continue
        total_len = random.randint(2, max_len)
        tail = _build_backward_chain(shape, end, set(), total_len - 1)
        if tail is None:
            continue
        return tail + [end], None, random.choice(out_dirs)
    return None


def _place_cycle(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    rows, cols = shape
    template = random.choice(_CYCLE_TEMPLATES)
    cycle = _ordered_transform(template)
    h = max(r for r, _ in cycle) + 1
    w = max(c for _, c in cycle) + 1
    if h > rows or w > cols:
        return None
    top = random.randint(0, rows - h)
    left = random.randint(0, cols - w)
    return [(int(r + top), int(c + left)) for r, c in cycle]


def _build_loop_trace(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], int, None] | None:
    rows, cols = shape
    total_cells = rows * cols

    for _ in range(160):
        cycle = _place_cycle(shape)
        if cycle is None:
            continue
        max_tail = min(4, total_cells - len(cycle))
        if max_tail < 0:
            continue
        if max_tail == 0:
            tail_len = 0
        elif random.random() < 0.75:
            tail_len = random.randint(1, max_tail)
        else:
            tail_len = random.randint(0, max_tail)
        tail = _build_backward_chain(shape, cycle[0], set(cycle), tail_len)
        if tail is None:
            continue
        return tail + cycle, len(tail), None
    return None


def _make_direction_map(
    shape: tuple[int, int],
    visited: list[tuple[int, int]],
    cycle_start: int | None,
    final_dir: tuple[int, int] | None,
) -> dict[tuple[int, int], tuple[int, int]]:
    directions = {
        (r, c): random.choice(_DIRS)
        for r in range(shape[0])
        for c in range(shape[1])
    }
    for src, dst in zip(visited, visited[1:]):
        directions[src] = (int(dst[0] - src[0]), int(dst[1] - src[1]))
    if cycle_start is None:
        assert final_dir is not None
        directions[visited[-1]] = final_dir
    else:
        entry = visited[cycle_start]
        last = visited[-1]
        directions[last] = (int(entry[0] - last[0]), int(entry[1] - last[1]))
    return directions


def _render_input(
    shape: tuple[int, int],
    side: int,
    colors: dict[str, int],
    directions: dict[tuple[int, int], tuple[int, int]],
    seed: tuple[int, int],
) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], tuple[int, int, int]]]:
    rows, cols = shape
    step = side + 1
    H = rows * step + 1
    W = cols * step + 1
    grid = np.full((H, W), int(colors["frame"]), dtype=int)
    base_patch = np.full((side, side), int(colors["base"]), dtype=int)
    center = side // 2
    dot_pixels: dict[tuple[int, int], tuple[int, int, int]] = {}

    for rr in range(rows):
        for cc in range(cols):
            patch = base_patch.copy()
            dr, dc = directions[(rr, cc)]
            dot_color = int(colors["seed"] if (rr, cc) == seed else colors["dot"])
            patch[center + dr, center + dc] = dot_color
            top = rr * step + 1
            left = cc * step + 1
            paste(grid, patch, top, left)
            dot_pixels[(rr, cc)] = (top + center + dr, left + center + dc, dot_color)

    return grid, _row_lines_for(step, rows), _row_lines_for(step, cols), dot_pixels


def _attempt_case(branch: str) -> np.ndarray | None:
    colors = dict(random.choice(_COLOR_SCHEMES))
    side = random.choice((3, 5))
    max_cells = 29 // (side + 1)
    rows = random.randint(2, max_cells)
    cols = random.randint(2, max_cells)
    if rows * cols < 4:
        return None

    if branch == "exit":
        trace = _build_exit_trace((rows, cols))
    else:
        trace = _build_loop_trace((rows, cols))
    if trace is None:
        return None

    visited, cycle_start, final_dir = trace
    seed = visited[0]
    directions = _make_direction_map((rows, cols), visited, cycle_start, final_dir)
    grid, row_lines, col_lines, dot_pixels = _render_input((rows, cols), side, colors, directions, seed)
    expected = _paint_result(grid, row_lines, col_lines, side, visited, cycle_start, dot_pixels)

    parsed_rows, parsed_cols, parsed_side, parsed_dirs, parsed_seed, parsed_dots = _parse_grid(grid)
    if parsed_rows != row_lines or parsed_cols != col_lines or parsed_side != side or parsed_seed != seed:
        return None
    if parsed_dirs != directions or parsed_dots != dot_pixels:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = "exit" if _GEN_CALL_COUNT % 2 == 0 else "loop"
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_case(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {branch} pointer-chain grid")
