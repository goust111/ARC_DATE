import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

TOWER_COLORS = (RED, BLUE, YELLOW)
VERTICAL_SIDES = ("top", "bottom")
HORIZONTAL_SIDES = ("left", "right")
FOCUS_OPTIONS = ("same", "mixed", "multi", "blocked")
SIZE_OPTIONS = (
    ((2, 2), 5),
    ((2, 3), 3),
    ((3, 2), 3),
    ((1, 2), 2),
    ((2, 1), 2),
)

MIXED_COLOR_BY_PAIR = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GRID_H_RANGE = (8, 14)
GRID_W_RANGE = (8, 14)
MAX_GENERATE_RETRIES = 500


def _is_tower(color: int) -> bool:
    return int(color) in TOWER_COLORS


def _emit_path(grid: np.ndarray, side: str, index: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    if side == "top":
        for r in range(1, h - 1):
            if int(grid[r, index]) == GREY:
                break
            cells.append((r, index))
    elif side == "bottom":
        for r in range(h - 2, 0, -1):
            if int(grid[r, index]) == GREY:
                break
            cells.append((r, index))
    elif side == "left":
        for c in range(1, w - 1):
            if int(grid[index, c]) == GREY:
                break
            cells.append((index, c))
    else:
        assert side == "right"
        for c in range(w - 2, 0, -1):
            if int(grid[index, c]) == GREY:
                break
            cells.append((index, c))
    return cells


def _ordered_towers(grid: np.ndarray) -> list[tuple[str, str, int, int]]:
    h, w = grid.shape
    towers: list[tuple[str, str, int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if _is_tower(color):
            towers.append(("vertical", "top", c, color))
    for c in range(1, w - 1):
        color = int(grid[h - 1, c])
        if _is_tower(color):
            towers.append(("vertical", "bottom", c, color))
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if _is_tower(color):
            towers.append(("horizontal", "left", r, color))
    for r in range(1, h - 1):
        color = int(grid[r, w - 1])
        if _is_tower(color):
            towers.append(("horizontal", "right", r, color))
    return towers


def _trace_rays(grid: np.ndarray):
    h, w = grid.shape
    horizontal = [[[] for _ in range(w)] for _ in range(h)]
    vertical = [[[] for _ in range(w)] for _ in range(h)]
    blocked_towers: list[tuple[str, int, int]] = []
    truncated_towers: list[tuple[str, int, int]] = []

    for orientation, side, index, color in _ordered_towers(grid):
        path = _emit_path(grid, side, index)
        if not path:
            blocked_towers.append((side, index, color))
            continue

        full_len = (grid.shape[0] - 2) if orientation == "vertical" else (grid.shape[1] - 2)
        if len(path) < full_len:
            truncated_towers.append((side, index, color))

        target = vertical if orientation == "vertical" else horizontal
        for r, c in path:
            target[r][c].append(color)

    return horizontal, vertical, blocked_towers, truncated_towers


def _mix_pair(a: int, b: int) -> int:
    if a == b:
        return a
    return MIXED_COLOR_BY_PAIR[frozenset((int(a), int(b)))]


def _choose_intersection_color(horizontal_colors: list[int], vertical_colors: list[int]) -> int:
    for h_color in horizontal_colors:
        for v_color in vertical_colors:
            if int(h_color) != int(v_color):
                return _mix_pair(h_color, v_color)
    return int(horizontal_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    horizontal, vertical, _blocked, _truncated = _trace_rays(g)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) == GREY:
                continue
            if horizontal[r][c] and vertical[r][c]:
                out[r, c] = _choose_intersection_color(horizontal[r][c], vertical[r][c])
    return out


def _reachable_rows(height: int, wall_row: int | None, side: str) -> list[int]:
    if side == "top":
        stop = wall_row if wall_row is not None else height - 1
        return list(range(1, stop))
    assert side == "bottom"
    start = wall_row + 1 if wall_row is not None else 1
    return list(range(start, height - 1))


def _reachable_cols(width: int, wall_col: int | None, side: str) -> list[int]:
    if side == "left":
        stop = wall_col if wall_col is not None else width - 1
        return list(range(1, stop))
    assert side == "right"
    start = wall_col + 1 if wall_col is not None else 1
    return list(range(start, width - 1))


def _sample_optional_wall(length: int, force: bool) -> int | None:
    if length < 8:
        return None
    if not force and random.random() < 0.4:
        return None
    low = 3
    high = length - 4
    if low > high:
        return None
    return random.randint(low, high)


def _choose_size_pair(n_cols: int, n_rows: int) -> tuple[int, int] | None:
    valid: list[tuple[int, int]] = []
    weights: list[int] = []
    for (nv, nh), weight in SIZE_OPTIONS:
        if nv <= n_cols and nh <= n_rows:
            valid.append((nv, nh))
            weights.append(weight)
    if not valid:
        return None
    return random.choices(valid, weights=weights, k=1)[0]


def _sample_active_colors(
    n_vertical: int,
    n_horizontal: int,
    focus: str,
    multi_axis: str | None,
) -> tuple[list[int], list[int]]:
    if focus == "same":
        color = random.choice(TOWER_COLORS)
        return [color] * n_vertical, [color] * n_horizontal

    if focus == "mixed":
        v_color, h_color = random.sample(TOWER_COLORS, 2)
        return [v_color] * n_vertical, [h_color] * n_horizontal

    if focus == "multi":
        assert multi_axis in {"vertical", "horizontal"}
        v_color, h_color = random.sample(TOWER_COLORS, 2)
        return [v_color] * n_vertical, [h_color] * n_horizontal

    assert focus == "blocked"
    if random.random() < 0.5:
        color = random.choice(TOWER_COLORS)
        return [color] * n_vertical, [color] * n_horizontal
    v_color, h_color = random.sample(TOWER_COLORS, 2)
    return [v_color] * n_vertical, [h_color] * n_horizontal


def _render_grid(
    height: int,
    width: int,
    wall_row: int | None,
    wall_col: int | None,
    towers: list[tuple[str, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    if wall_row is not None:
        grid[wall_row, 1 : width - 1] = GREY
    if wall_col is not None:
        grid[1 : height - 1, wall_col] = GREY

    used_cells: set[tuple[int, int]] = set()
    for side, index, color in towers:
        if side == "top":
            cell = (0, index)
        elif side == "bottom":
            cell = (height - 1, index)
        elif side == "left":
            cell = (index, 0)
        else:
            cell = (index, width - 1)
        if cell in used_cells:
            raise ValueError("duplicate tower placement")
        used_cells.add(cell)
        grid[cell] = color
    return grid


def _analyze_grid(grid: np.ndarray) -> dict[str, int]:
    horizontal, vertical, blocked_towers, truncated_towers = _trace_rays(grid)
    out = main(grid)
    h, w = grid.shape

    same_cells = 0
    mixed_cells = 0
    multi_cells = 0
    ambiguous_cells = 0

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == GREY:
                continue
            h_colors = horizontal[r][c]
            v_colors = vertical[r][c]
            if not h_colors or not v_colors:
                continue

            diff_results = {_mix_pair(hc, vc) for hc in h_colors for vc in v_colors if hc != vc}
            if diff_results:
                mixed_cells += 1
            else:
                same_cells += 1
            if len(h_colors) + len(v_colors) >= 3:
                multi_cells += 1
            if len(diff_results) > 1:
                ambiguous_cells += 1

    return {
        "tower_count": len(_ordered_towers(grid)),
        "blocked_tower_count": len(blocked_towers),
        "truncated_tower_count": len(truncated_towers),
        "intersection_cells": same_cells + mixed_cells,
        "same_cells": same_cells,
        "mixed_cells": mixed_cells,
        "multi_cells": multi_cells,
        "ambiguous_cells": ambiguous_cells,
        "changed_cells": int(np.count_nonzero(out != grid)),
        "non_black_output_cells": int(np.count_nonzero(out != BLACK)),
    }


def _add_blocked_towers(
    towers: list[tuple[str, int, int]],
    wall_row: int | None,
    wall_col: int | None,
) -> None:
    candidates: list[tuple[str, int]] = []
    if wall_row is not None:
        candidates.extend([("left", wall_row), ("right", wall_row)])
    if wall_col is not None:
        candidates.extend([("top", wall_col), ("bottom", wall_col)])

    random.shuffle(candidates)
    count = random.randint(1, min(2, len(candidates)))
    used = {(side, index) for side, index, _color in towers}
    added = 0
    for side, index in candidates:
        if (side, index) in used:
            continue
        towers.append((side, index, random.choice(TOWER_COLORS)))
        used.add((side, index))
        added += 1
        if added >= count:
            break


def _build_focus_instance() -> np.ndarray | None:
    focus = random.choice(FOCUS_OPTIONS)
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    vertical_side = random.choice(VERTICAL_SIDES)
    horizontal_side = random.choice(HORIZONTAL_SIDES)

    if focus == "multi":
        multi_axis = random.choice(("vertical", "horizontal"))
        wall_row = None if multi_axis == "vertical" else _sample_optional_wall(height, False)
        wall_col = None if multi_axis == "horizontal" else _sample_optional_wall(width, False)
    else:
        multi_axis = None
        force_wall = focus == "blocked"
        wall_row = _sample_optional_wall(height, force_wall and random.random() < 0.7)
        wall_col = _sample_optional_wall(width, force_wall and wall_row is None)
        if force_wall and wall_row is None and wall_col is None:
            if random.random() < 0.5:
                wall_row = _sample_optional_wall(height, True)
            else:
                wall_col = _sample_optional_wall(width, True)

    row_candidates = _reachable_rows(height, wall_row, vertical_side)
    col_candidates = _reachable_cols(width, wall_col, horizontal_side)
    size_pair = _choose_size_pair(len(col_candidates), len(row_candidates))
    if size_pair is None:
        return None
    n_vertical, n_horizontal = size_pair

    vertical_colors, horizontal_colors = _sample_active_colors(
        n_vertical, n_horizontal, focus, multi_axis
    )

    chosen_cols = sorted(random.sample(col_candidates, n_vertical))
    chosen_rows = sorted(random.sample(row_candidates, n_horizontal))

    towers: list[tuple[str, int, int]] = []
    towers.extend((vertical_side, col, color) for col, color in zip(chosen_cols, vertical_colors))
    towers.extend((horizontal_side, row, color) for row, color in zip(chosen_rows, horizontal_colors))

    if focus == "multi":
        if multi_axis == "vertical":
            opposite = "bottom" if vertical_side == "top" else "top"
            viable = [i for i, color in enumerate(vertical_colors) if any(color != h for h in horizontal_colors)]
            if not viable:
                return None
            idx = random.choice(viable)
            towers.append((opposite, chosen_cols[idx], vertical_colors[idx]))
        else:
            opposite = "right" if horizontal_side == "left" else "left"
            viable = [i for i, color in enumerate(horizontal_colors) if any(color != v for v in vertical_colors)]
            if not viable:
                return None
            idx = random.choice(viable)
            towers.append((opposite, chosen_rows[idx], horizontal_colors[idx]))

    if focus == "blocked":
        _add_blocked_towers(towers, wall_row, wall_col)

    grid = _render_grid(height, width, wall_row, wall_col, towers)
    stats = _analyze_grid(grid)
    clear_interior = (height - 2) * (width - 2) - int((height - 2) if wall_col is not None else 0) - int((width - 2) if wall_row is not None else 0)

    if stats["ambiguous_cells"] != 0:
        return None
    if stats["intersection_cells"] < 2:
        return None
    if stats["changed_cells"] != stats["intersection_cells"]:
        return None
    if stats["intersection_cells"] > max(8, clear_interior // 2):
        return None
    if focus == "same" and (stats["same_cells"] == 0 or stats["mixed_cells"] != 0 or stats["multi_cells"] != 0):
        return None
    if focus == "mixed" and (stats["mixed_cells"] == 0 or stats["same_cells"] != 0 or stats["multi_cells"] != 0):
        return None
    if focus == "multi" and stats["multi_cells"] == 0:
        return None
    if focus == "blocked" and stats["blocked_tower_count"] == 0:
        return None
    if (wall_row is not None or wall_col is not None) and stats["truncated_tower_count"] == 0 and stats["blocked_tower_count"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_RETRIES):
        grid = _build_focus_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
