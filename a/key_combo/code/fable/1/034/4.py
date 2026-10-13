from __future__ import annotations

from collections import deque

import numpy as np


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


TRACE_COLOR = Color.GREEN
GEN_COLOR_POOL = np.array(
    [
        Color.BLUE,
        Color.RED,
        Color.YELLOW,
        Color.GREY,
        Color.PINK,
        Color.ORANGE,
        Color.TEAL,
        Color.MAROON,
    ],
    dtype=int,
)
GEN_MAX_ATTEMPTS = 300
_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid, dtype=int).ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in set(map(int, exclude))]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _find_full_width_bar(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and np.all(row == row[0]):
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1, "expected exactly one full-width foreground bar"
    return candidates[0]


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, int, list[tuple[int, int]], tuple[int, int]]:
    bg = get_bg(grid)
    bar_r, bar_color = _find_full_width_bar(grid, bg)
    assert bar_r > 0, "bar must have a row above it"

    fg_colors = colors_present(grid, exclude={bg})
    assert len(fg_colors) == 2, "expected one bar color and one suspended-pixel color"
    moving_colors = [color for color in fg_colors if color != bar_color]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])
    assert moving_color != TRACE_COLOR and bar_color != TRACE_COLOR, "green is reserved for ray traces"

    floating = [tuple(map(int, coord)) for coord in np.argwhere(grid == moving_color)]
    assert floating, "expected at least one suspended pixel"
    assert all(r < bar_r for r, _c in floating), "suspended pixels must stay above the bar"

    columns = [c for _r, c in floating]
    assert len(columns) == len(set(columns)), "expected at most one suspended pixel per column"

    seeds = [(r, c) for r, c in floating if r == bar_r - 1]
    assert len(seeds) == 1, "expected exactly one seed pixel immediately above the bar"
    seed = seeds[0]

    return bg, bar_r, bar_color, moving_color, floating, seed


def _ray_cells(shape: tuple[int, int], bar_r: int, col: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dc in (-1, 1):
        r = int(bar_r) - 2
        c = int(col) + dc
        while in_bounds(shape, r, c):
            cells.append((int(r), int(c)))
            r -= 1
            c += dc
    return cells


def _activation_depths(
    shape: tuple[int, int],
    bar_r: int,
    seed_col: int,
    floating: list[tuple[int, int]],
) -> dict[int, int]:
    floating_set = set(floating)
    depths = {int(seed_col): 0}
    queue: deque[int] = deque([int(seed_col)])

    while queue:
        src_col = queue.popleft()
        for cell in _ray_cells(shape, bar_r, src_col):
            if cell not in floating_set:
                continue
            hit_col = int(cell[1])
            if hit_col in depths:
                continue
            depths[hit_col] = depths[src_col] + 1
            queue.append(hit_col)

    return depths


def _all_ray_cells(shape: tuple[int, int], bar_r: int, landed_cols: set[int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for col in landed_cols:
        cells.update(_ray_cells(shape, bar_r, col))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let only ray-hit suspended pixels fall onto the bar and mark the activated diagonal fan in green.

    Input structure
    - A dominant background contains one full-width horizontal bar in a single color.
    - Another color appears as isolated suspended pixels above the bar, with at most one such pixel per column.
    - Exactly one suspended pixel already sits one row above the bar and acts as the seed.

    Rule
    - Treat the seed as already landed on the row just above the bar.
    - Every landed pixel emits two upward diagonal rays to the grid edge; any suspended pixel lying on a ray becomes activated, falls vertically to its own column one row above the bar, and then emits the same rays.
    - Repeat until no new suspended pixels are hit. Keep the bar unchanged, keep only never-hit suspended pixels in place, paint all landed pixels in the landing row, and paint every ray cell that is still background green.

    Key insight
    - Falling is not global gravity: a suspended pixel moves only if a previously landed pixel's diagonal ray reaches its exact coordinate, so chain activation determines which columns drop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, bar_r, bar_color, moving_color, floating, seed = _extract_scene(g)
    depths = _activation_depths(g.shape, bar_r, seed[1], floating)
    landed_cols = set(depths)
    ray_cells = _all_ray_cells(g.shape, bar_r, landed_cols)

    out = g.copy()

    for r, c in floating:
        if c in landed_cols:
            out[r, c] = bg

    for r, c in ray_cells:
        if int(out[r, c]) == bg:
            out[r, c] = TRACE_COLOR

    for col in landed_cols:
        out[bar_r - 1, col] = moving_color

    out[bar_r, :] = bar_color
    return out


def _randint(low: int, high: int) -> int:
    return int(_RNG.integers(low, high + 1))


def _sample_colors() -> tuple[int, int]:
    chosen = _RNG.choice(GEN_COLOR_POOL, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _hit_row(bar_r: int, source_col: int, target_col: int) -> int | None:
    dist = abs(int(target_col) - int(source_col))
    if dist == 0:
        return None
    row = int(bar_r) - 1 - dist
    if row < 0:
        return None
    return int(row)


def _child_options(
    width: int,
    bar_r: int,
    seed_col: int,
    parent_col: int,
    used_cols: set[int],
    *,
    forbid_seed_direct: bool,
) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for col in range(width):
        if col in used_cols:
            continue
        row = _hit_row(bar_r, parent_col, col)
        if row is None:
            continue
        if forbid_seed_direct and _hit_row(bar_r, seed_col, col) == row:
            continue
        options.append((int(col), int(row)))
    return options


def _pick_decoy_row(allowed_rows: list[int], hit_rows: set[int]) -> int:
    distances = []
    for row in allowed_rows:
        nearest = min(abs(row - hit_row) for hit_row in hit_rows)
        distances.append((nearest, float(_RNG.random()), int(row)))
    distances.sort(key=lambda item: (item[0], item[1]))
    best_distance = distances[0][0]
    near_rows = [row for distance, _noise, row in distances if distance == best_distance]
    return int(near_rows[_randint(0, len(near_rows) - 1)])


def _make_grid(
    height: int,
    width: int,
    bar_r: int,
    bar_color: int,
    moving_color: int,
    floating_by_col: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[bar_r, :] = int(bar_color)
    for col, row in floating_by_col.items():
        grid[int(row), int(col)] = int(moving_color)
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = _randint(9, 17)
    width = _randint(9, 19)
    bar_low = max(4, height // 2)
    bar_high = height - 2
    if bar_low > bar_high:
        return None
    bar_r = _randint(bar_low, bar_high)

    if width <= 4:
        return None
    seed_left = 2 if width >= 8 else 1
    seed_right = width - 3 if width >= 8 else width - 2
    if seed_left > seed_right:
        return None
    seed_col = _randint(seed_left, seed_right)

    bar_color, moving_color = _sample_colors()
    need_chain = bool(_RNG.random() < 0.75)
    need_decoy = bool(_RNG.random() < 0.8)

    floating_by_col: dict[int, int] = {seed_col: bar_r - 1}

    max_extra = min(5, width - 2)
    min_extra = 2 if need_chain else 1
    if max_extra < min_extra:
        return None
    target_extra = _randint(min_extra, max_extra)

    direct_candidates = [
        col
        for col in range(width)
        if col != seed_col and _hit_row(bar_r, seed_col, col) is not None
    ]
    _RNG.shuffle(direct_candidates)
    if not direct_candidates:
        return None

    direct_cap = min(3, target_extra, len(direct_candidates))
    if direct_cap < 1:
        return None
    direct_count = _randint(1, direct_cap)
    for col in direct_candidates[:direct_count]:
        floating_by_col[int(col)] = int(_hit_row(bar_r, seed_col, int(col)))

    used_cols = set(floating_by_col)
    chain_slots = target_extra - direct_count
    for _ in range(chain_slots):
        prefer_chain_only = need_chain
        parents = [col for col in used_cols if col != seed_col]
        _RNG.shuffle(parents)
        if not parents:
            parents = [seed_col]

        chosen: tuple[int, int] | None = None
        for parent in parents:
            options = _child_options(
                width,
                bar_r,
                seed_col,
                parent,
                used_cols,
                forbid_seed_direct=prefer_chain_only and parent != seed_col,
            )
            if not options:
                options = _child_options(
                    width,
                    bar_r,
                    seed_col,
                    parent,
                    used_cols,
                    forbid_seed_direct=False,
                )
            if options:
                chosen = options[_randint(0, len(options) - 1)]
                break

        if chosen is None:
            break
        col, row = chosen
        floating_by_col[col] = row
        used_cols.add(col)

    if len(floating_by_col) - 1 < min_extra:
        return None

    core_floating = [(row, col) for col, row in floating_by_col.items()]
    core_depths = _activation_depths((height, width), bar_r, seed_col, core_floating)
    if len(core_depths) != len(floating_by_col):
        return None

    if need_chain and max(core_depths.values()) < 2:
        return None

    landed_cols = set(core_depths)
    unused_cols = [col for col in range(width) if col not in floating_by_col]
    _RNG.shuffle(unused_cols)

    decoy_count = 0
    if unused_cols:
        decoy_goal = _randint(1, min(3, len(unused_cols))) if need_decoy else _randint(0, min(2, len(unused_cols)))
        for col in unused_cols:
            hit_rows = {
                bar_r - 1 - abs(col - landed_col)
                for landed_col in landed_cols
                if bar_r - 1 - abs(col - landed_col) >= 0
            }
            if not hit_rows:
                continue
            allowed_rows = [row for row in range(bar_r - 1) if row not in hit_rows]
            if not allowed_rows:
                continue
            floating_by_col[col] = _pick_decoy_row(allowed_rows, hit_rows)
            decoy_count += 1
            if decoy_count >= decoy_goal:
                break

    if need_decoy and decoy_count == 0:
        return None

    total_floating = [(row, col) for col, row in floating_by_col.items()]
    depths = _activation_depths((height, width), bar_r, seed_col, total_floating)
    activated_cols = set(depths)
    if need_chain and max(depths.values()) < 2:
        return None
    if need_decoy and len(floating_by_col) == len(activated_cols):
        return None

    ray_cells = _all_ray_cells((height, width), bar_r, activated_cols)
    if len(ray_cells) > (height * width) // 2:
        return None

    grid = _make_grid(height, width, bar_r, bar_color, moving_color, floating_by_col)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
