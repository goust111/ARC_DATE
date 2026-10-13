from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
LIQUID_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _black_component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _color_component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    color = int(grid[start])
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _find_pool(grid: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detect the shared collection pool as the widest grey framed basin:
    full grey side walls and bottom, black interior, and a top border that
    is grey except for 1-cell black entry gaps.
    """
    h, w = grid.shape
    best = None

    for bottom in range(h):
        c = 0
        while c < w:
            if int(grid[bottom, c]) != GREY:
                c += 1
                continue

            left = c
            while c + 1 < w and int(grid[bottom, c + 1]) == GREY:
                c += 1
            right = c
            width = right - left + 1

            if width >= 5:
                top = bottom
                while top - 1 >= 0 and int(grid[top - 1, left]) == GREY and int(grid[top - 1, right]) == GREY:
                    top -= 1
                height = bottom - top + 1
                if height >= 5:
                    if np.all(grid[top : bottom + 1, left] == GREY) and np.all(grid[top : bottom + 1, right] == GREY):
                        interior = grid[top + 1 : bottom, left + 1 : right]
                        if interior.size and np.all(interior == BG):
                            top_row = grid[top, left : right + 1]
                            if np.all((top_row == GREY) | (top_row == BG)):
                                gaps = [cc for cc in range(left + 1, right) if int(grid[top, cc]) == BG]
                                if gaps and len(gaps) < width - 2:
                                    if all(int(grid[top + 1, cc]) == BG for cc in gaps):
                                        cand = (width, height, -top, -left, top, left, bottom, right)
                                        if best is None or cand > best:
                                            best = cand
            c += 1

    assert best is not None, "expected to find a pool"
    _, _, _, _, top, left, bottom, right = best
    return int(top), int(left), int(bottom), int(right)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drain each container into the shared pool.

    - Each liquid block's row count is its contribution height.
    - Its pipe length is the number of connected black pipe cells outside the pool interior.
    - Shorter pipes settle lower in the pool; ties break by container left-to-right order.
    - Containers empty out; pipe interiors remain black.
    """
    g = np.asarray(grid, dtype=int)
    top, left, bottom, right = _find_pool(g)

    pool_interior = np.zeros_like(g, dtype=bool)
    pool_interior[top + 1 : bottom, left + 1 : right] = True
    pipe_mask = (g == BG) & (~pool_interior)
    entry_cols = [c for c in range(left + 1, right) if int(g[top, c]) == BG and int(g[top + 1, c]) == BG]
    assert entry_cols, "pool must have at least one entry gap"

    out = g.copy()
    cleared_liquid: set[tuple[int, int]] = set()
    pipe_seen = np.zeros_like(g, dtype=bool)
    records: list[tuple[int, int, int, int, int]] = []

    for entry_col in entry_cols:
        start = (top, entry_col)
        if pipe_seen[start]:
            continue

        pipe_cells = _black_component(pipe_mask, start)
        for r, c in pipe_cells:
            pipe_seen[r, c] = True

        adjacent_liquid = sorted(
            {
                (nr, nc)
                for r, c in pipe_cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < g.shape[0]
                and 0 <= nc < g.shape[1]
                and int(g[nr, nc]) not in (BG, GREY)
            }
        )
        assert adjacent_liquid, "each pipe should touch exactly one liquid source"

        liquid_cells = _color_component(g, adjacent_liquid[0])
        rows = [r for r, _ in liquid_cells]
        cols = [c for _, c in liquid_cells]
        records.append(
            (
                len(pipe_cells),
                min(cols),
                entry_col,
                int(g[adjacent_liquid[0]]),
                max(rows) - min(rows) + 1,
            )
        )
        cleared_liquid |= liquid_cells

    for r, c in cleared_liquid:
        out[r, c] = BG

    filled = 0
    bottom_row = bottom - 1
    pool_inner_height = bottom - top - 1
    for pipe_len, left_key, entry_col, color, height in sorted(records, key=lambda rec: (rec[0], rec[1])):
        assert filled + height <= pool_inner_height
        r1 = bottom_row - filled
        r0 = r1 - height + 1
        out[r0 : r1 + 1, entry_col] = color
        filled += height

    return out


def _build_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """
    One-bend path: horizontal from the outlet gap, then vertical into the pool.
    Straight pipes are the special case with zero horizontal offset.
    """
    sr, sc = start
    er, ec = end
    cells: list[tuple[int, int]] = []

    step = 1 if ec >= sc else -1
    for c in range(sc, ec + step, step):
        cells.append((sr, c))
    for r in range(sr + 1, er + 1):
        cells.append((r, ec))

    return cells


def _place_pipe_walls(
    grid: np.ndarray,
    path_cells: list[tuple[int, int]],
    all_path_cells: set[tuple[int, int]],
    pool_interior: np.ndarray,
    container_interior: np.ndarray,
) -> None:
    h, w = grid.shape
    for r, c in path_cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in all_path_cells or pool_interior[nr, nc] or container_interior[nr, nc]:
                continue
            if int(grid[nr, nc]) == BG:
                grid[nr, nc] = GREY


def _sample_liquid_heights(n_containers: int) -> list[int] | None:
    heights = [random.randint(1, 5) for _ in range(n_containers)]
    return heights if sum(heights) <= 13 else None


def generate_input() -> np.ndarray:
    """
    Randomly generate U-shaped containers draining into a shared pool.

    Variations cover:
    - 2 to 4 containers
    - liquid heights 1..5
    - container widths and heights
    - straight pipes vs bent pipes
    - left-bending vs right-bending layouts
    - repeated pipe lengths to exercise tie-breaking
    """
    for _ in range(MAX_GEN_ATTEMPTS):
        n_containers = random.randint(2, 4)
        liquid_heights = _sample_liquid_heights(n_containers)
        if liquid_heights is None:
            continue

        inner_widths = [random.choice([3, 5]) for _ in range(n_containers)]
        inner_heights = [random.randint(h, min(6, h + 2)) for h in liquid_heights]
        outlet_row = max(inner_heights) + 1 + random.randint(0, 2)
        top_rows = [outlet_row - h for h in inner_heights]

        left_walls: list[int] = []
        x = 0
        for i, inner_width in enumerate(inner_widths):
            left_walls.append(x)
            x += inner_width + 2
            if i < n_containers - 1:
                x += random.randint(2, 4)

        right_walls = [left + inner_width + 1 for left, inner_width in zip(left_walls, inner_widths)]
        outlet_cols = [left + 1 + inner_width // 2 for left, inner_width in zip(left_walls, inner_widths)]

        bend_sign = random.choice([-1, 1])
        mag_choices = [0, 1, 2]
        if random.random() < 0.5:
            shared_mag = random.choice(mag_choices)
            tied_indices = set(random.sample(range(n_containers), 2))
            bend_magnitudes = [
                shared_mag if i in tied_indices else random.choice(mag_choices)
                for i in range(n_containers)
            ]
        else:
            bend_magnitudes = [random.choice(mag_choices) for _ in range(n_containers)]
        bend_offsets = [bend_sign * mag for mag in bend_magnitudes]
        entry_cols = [col + off for col, off in zip(outlet_cols, bend_offsets)]

        if any(entry_cols[i] >= entry_cols[i + 1] for i in range(n_containers - 1)):
            continue
        if any(entry_cols[i + 1] - entry_cols[i] < 2 for i in range(n_containers - 1)):
            continue

        inner_left = min(entry_cols) - random.randint(1, 2)
        inner_right = max(entry_cols) + random.randint(1, 2)
        min_pool_inner_width = max(inner_width + 2 for inner_width in inner_widths) + 2
        current_inner_width = inner_right - inner_left + 1
        if current_inner_width < min_pool_inner_width:
            extra = min_pool_inner_width - current_inner_width
            inner_left -= extra // 2
            inner_right += extra - extra // 2

        pool_left = inner_left - 1
        pool_right = inner_right + 1

        x_shift = 1 - min(pool_left, left_walls[0])
        left_walls = [x + x_shift for x in left_walls]
        right_walls = [x + x_shift for x in right_walls]
        outlet_cols = [x + x_shift for x in outlet_cols]
        entry_cols = [x + x_shift for x in entry_cols]
        pool_left += x_shift
        pool_right += x_shift

        pool_inner_height = sum(liquid_heights) + random.randint(1, 3)
        pool_top = outlet_row + random.randint(4, 7)
        pool_bottom = pool_top + pool_inner_height + 1

        grid_h = pool_bottom + random.randint(2, 4) + 1
        grid_w = max(pool_right, max(right_walls)) + random.randint(2, 4) + 1
        if grid_h > 30 or grid_w > 30:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        colors = random.sample(LIQUID_COLORS, n_containers)
        container_interior = np.zeros_like(grid, dtype=bool)
        pool_interior = np.zeros_like(grid, dtype=bool)
        pool_interior[pool_top + 1 : pool_bottom, pool_left + 1 : pool_right] = True

        for left, right, top, inner_width, liquid_height, color in zip(
            left_walls, right_walls, top_rows, inner_widths, liquid_heights, colors
        ):
            grid[top : outlet_row + 1, left] = GREY
            grid[top : outlet_row + 1, right] = GREY
            grid[outlet_row, left : right + 1] = GREY
            outlet_col = left + 1 + inner_width // 2
            grid[outlet_row, outlet_col] = BG
            container_interior[top:outlet_row, left + 1 : right] = True
            grid[outlet_row - liquid_height : outlet_row, left + 1 : right] = color

        grid[pool_top : pool_bottom + 1, pool_left] = GREY
        grid[pool_top : pool_bottom + 1, pool_right] = GREY
        grid[pool_bottom, pool_left : pool_right + 1] = GREY
        grid[pool_top, pool_left : pool_right + 1] = GREY
        for entry_col in entry_cols:
            grid[pool_top, entry_col] = BG

        paths: list[list[tuple[int, int]]] = []
        all_path_cells: set[tuple[int, int]] = set()
        valid = True

        for outlet_col, entry_col in zip(outlet_cols, entry_cols):
            path_cells = _build_path((outlet_row, outlet_col), (pool_top, entry_col))
            path_set = set(path_cells)

            if all_path_cells & path_set:
                valid = False
                break

            for r, c in path_cells:
                if (r, c) in ((outlet_row, outlet_col), (pool_top, entry_col)):
                    continue
                if not (0 <= r < grid_h and 0 <= c < grid_w):
                    valid = False
                    break
                if int(grid[r, c]) != BG:
                    valid = False
                    break
            if not valid:
                break

            for r, c in path_set:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in all_path_cells and not pool_interior[nr, nc]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            all_path_cells |= path_set
            paths.append(path_cells)

        if not valid:
            continue

        for path_cells in paths:
            _place_pipe_walls(grid, path_cells, all_path_cells, pool_interior, container_interior)

        try:
            main(grid)
        except Exception:
            continue

        return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
