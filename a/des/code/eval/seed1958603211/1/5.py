from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np


BG = 0
GROUND = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


@dataclass(frozen=True)
class Tower:
    col: int
    height: int
    signal_row: int
    signal_color: int


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[Tower], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = map(int, g.shape)
    ground_row = h - 1
    ground_color = int(g[ground_row, 0])
    if not np.all(g[ground_row] == ground_color):
        raise ValueError("bottom row must be a single ground color")

    bg = _most_common_color(g)
    towers: list[Tower] = []
    signal_positions: set[tuple[int, int]] = set()

    for col in range(w):
        height = 1
        row = ground_row - 1
        while row >= 0 and int(g[row, col]) == ground_color:
            height += 1
            row -= 1
        if height < 2:
            continue

        signal_row = ground_row - height
        if signal_row < 0:
            raise ValueError("tower signal must fit inside the grid")
        signal_color = int(g[signal_row, col])
        if signal_color in (bg, ground_color):
            raise ValueError("tower must have a colored signal above its top")

        towers.append(Tower(col=col, height=height, signal_row=signal_row, signal_color=signal_color))
        signal_positions.add((signal_row, col))

    if not towers:
        raise ValueError("expected at least one tower")

    return bg, ground_color, ground_row, towers, signal_positions


def _is_filter_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    *,
    bg: int,
    ground_color: int,
    signal_positions: set[tuple[int, int]],
) -> bool:
    if (row, col) in signal_positions:
        return False
    value = int(grid[row, col])
    return value != bg and value != ground_color


def _path_filters(
    grid: np.ndarray,
    tower: Tower,
    target_col: int,
    *,
    bg: int,
    ground_color: int,
    ground_row: int,
    signal_positions: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    filters: dict[tuple[int, int], int] = {}
    c0, c1 = sorted((tower.col, target_col))
    for col in range(c0, c1 + 1):
        if _is_filter_cell(
            grid,
            tower.signal_row,
            col,
            bg=bg,
            ground_color=ground_color,
            signal_positions=signal_positions,
        ):
            filters[(tower.signal_row, col)] = int(grid[tower.signal_row, col])

    for row in range(tower.signal_row + 1, ground_row):
        if _is_filter_cell(
            grid,
            row,
            target_col,
            bg=bg,
            ground_color=ground_color,
            signal_positions=signal_positions,
        ):
            filters[(row, target_col)] = int(grid[row, target_col])

    result = [(row, col, color) for (row, col), color in filters.items()]
    result.sort(key=lambda item: (item[0], item[1]))
    return result


def _tower_coverage(
    grid: np.ndarray,
    tower: Tower,
    *,
    ground_color: int,
) -> tuple[list[int], bool]:
    _, width = map(int, grid.shape)
    covered = [tower.col]
    blocked = False

    for direction in (-1, 1):
        for step in range(1, tower.height + 1):
            col = tower.col + direction * step
            if not (0 <= col < width):
                break
            if int(grid[tower.signal_row, col]) == ground_color:
                blocked = True
                break
            covered.append(int(col))

    covered = sorted(set(covered))
    return covered, blocked


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg, ground_color, ground_row, towers, signal_positions = _parse_scene(g)

    out = g.copy()
    _, width = map(int, g.shape)
    best_height = np.full(width, -1, dtype=int)
    coverage_count = np.zeros(width, dtype=int)
    blocked_exists = False
    filtered_exists = False
    multi_filter_exists = False
    tower_infos: list[dict[str, object]] = []

    for tower in sorted(towers, key=lambda item: item.col):
        covered_cols, blocked = _tower_coverage(g, tower, ground_color=ground_color)
        blocked_exists = blocked_exists or blocked

        path_info: dict[int, dict[str, object]] = {}
        for col in covered_cols:
            filters = _path_filters(
                g,
                tower,
                col,
                bg=bg,
                ground_color=ground_color,
                ground_row=ground_row,
                signal_positions=signal_positions,
            )
            color = tower.signal_color
            if filters:
                lowest = max(filters, key=lambda item: item[0])
                color = int(lowest[2])
                filtered_exists = True
            if len(filters) >= 2:
                multi_filter_exists = True

            path_info[int(col)] = {
                "filters": filters,
                "color": int(color),
            }
            coverage_count[int(col)] += 1

            if tower.height > int(best_height[col]):
                best_height[col] = tower.height
                out[ground_row, col] = int(color)

        tower_infos.append(
            {
                "tower": tower,
                "covered_cols": covered_cols,
                "paths": path_info,
            }
        )

    meta = {
        "bg": bg,
        "ground_color": ground_color,
        "ground_row": ground_row,
        "towers": towers,
        "tower_infos": tower_infos,
        "blocked_exists": blocked_exists,
        "overlap_exists": bool(np.any(coverage_count >= 2)),
        "filtered_exists": filtered_exists,
        "multi_filter_exists": multi_filter_exists,
        "coverage_count": coverage_count,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project tower signals onto the ground.

    Towers are vertical grey columns that rise from the grey ground row, with one
    colored signal pixel directly above each top. A tower covers its own column
    plus up to `height` columns on each side along the signal row; propagation in a
    direction stops before the first other grey tower body it would enter.

    For every covered ground column, paint the tower's signal color unless the
    path crosses one or more colored horizontal filter cells, in which case the
    lowest crossed filter determines the ground color. If multiple towers reach
    the same ground column, the taller tower wins.
    """
    out, _meta = _solve(np.asarray(grid, dtype=int))
    return out


def _sample_spaced_columns(rng: np.random.Generator, width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(100):
        rng.shuffle(candidates)
        chosen: list[int] = []
        for col in candidates:
            if all(abs(col - prev) >= 2 for prev in chosen):
                chosen.append(int(col))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _build_base_grid(
    h: int,
    w: int,
    towers: list[tuple[int, int]],
    signal_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = GROUND
    for (col, height), color in zip(towers, signal_colors):
        top = h - 1 - height + 1
        grid[top:, col] = GROUND
        grid[top - 1, col] = int(color)
    return grid


def _layout_matches(
    towers: list[Tower],
    *,
    blocked_required: bool,
    overlap_required: bool,
    meta: dict[str, object],
) -> bool:
    if bool(meta["blocked_exists"]) != bool(blocked_required):
        return False
    if bool(meta["overlap_exists"]) != bool(overlap_required):
        return False

    heights = [tower.height for tower in towers]
    return len(set(heights)) == len(heights)


def _gen_tower_layout(
    rng: np.random.Generator,
    *,
    blocked_required: bool,
    overlap_required: bool,
) -> np.ndarray | None:
    for _ in range(200):
        h = int(rng.integers(11, 19))
        w = int(rng.integers(15, 25))
        tower_count = int(rng.integers(2, 5))
        max_height = min(8, h - 1)
        if max_height < tower_count + 1:
            continue

        height_pool = np.arange(2, max_height + 1)
        heights = [int(x) for x in rng.choice(height_pool, size=tower_count, replace=False)]
        columns = _sample_spaced_columns(rng, w, tower_count)
        if columns is None:
            continue

        signal_colors = [int(x) for x in rng.choice(PALETTE, size=tower_count, replace=False)]
        base = _build_base_grid(h, w, list(zip(columns, heights)), signal_colors)
        _out, meta = _solve(base)
        towers = list(meta["towers"])
        if _layout_matches(
            towers,
            blocked_required=blocked_required,
            overlap_required=overlap_required,
            meta=meta,
        ):
            return base
    return None


def _bg_run_containing(grid: np.ndarray, row: int, col: int) -> tuple[int, int] | None:
    if int(grid[row, col]) != BG:
        return None
    width = int(grid.shape[1])
    left = col
    while left - 1 >= 0 and int(grid[row, left - 1]) == BG:
        left -= 1
    right = col
    while right + 1 < width and int(grid[row, right + 1]) == BG:
        right += 1
    if right - left + 1 < 2:
        return None
    return int(left), int(right)


def _path_opportunities(
    grid: np.ndarray,
    meta: dict[str, object],
) -> list[dict[str, int]]:
    ground_row = int(meta["ground_row"])
    opportunities: list[dict[str, int]] = []
    for tower_index, info in enumerate(meta["tower_infos"]):
        tower = info["tower"]
        for target_col in info["covered_cols"]:
            if int(target_col) == int(tower.col):
                continue
            for row in range(int(tower.signal_row) + 1, ground_row):
                run = _bg_run_containing(grid, row, int(target_col))
                if run is None:
                    continue
                left, right = run
                opportunities.append(
                    {
                        "tower_index": int(tower_index),
                        "target_col": int(target_col),
                        "row": int(row),
                        "left": int(left),
                        "right": int(right),
                    }
                )
    return opportunities


def _place_filter_segment(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    row: int,
    target_col: int,
    left: int,
    right: int,
    color: int,
) -> None:
    max_len = min(7, right - left + 1)
    min_len = 2
    if max_len < min_len:
        raise ValueError("filter segment needs at least two cells")
    seg_len = int(rng.integers(min_len, max_len + 1))
    start_min = max(left, target_col - seg_len + 1)
    start_max = min(target_col, right - seg_len + 1)
    start = int(rng.integers(start_min, start_max + 1))
    grid[row, start : start + seg_len] = int(color)


def _available_filter_colors(grid: np.ndarray) -> list[int]:
    used = set(map(int, np.unique(grid)))
    return [color for color in PALETTE if color not in used]


def _add_single_filter(
    rng: np.random.Generator,
    grid: np.ndarray,
    meta: dict[str, object],
) -> np.ndarray | None:
    opportunities = _path_opportunities(grid, meta)
    if not opportunities:
        return None

    filter_colors = _available_filter_colors(grid)
    if not filter_colors:
        return None

    choice = opportunities[int(rng.integers(len(opportunities)))]
    color = int(filter_colors[int(rng.integers(len(filter_colors)))])
    out = grid.copy()
    _place_filter_segment(
        rng,
        out,
        row=int(choice["row"]),
        target_col=int(choice["target_col"]),
        left=int(choice["left"]),
        right=int(choice["right"]),
        color=color,
    )
    return out


def _add_multi_filters(
    rng: np.random.Generator,
    grid: np.ndarray,
    meta: dict[str, object],
) -> np.ndarray | None:
    grouped: dict[tuple[int, int], list[dict[str, int]]] = {}
    for opp in _path_opportunities(grid, meta):
        key = (int(opp["tower_index"]), int(opp["target_col"]))
        grouped.setdefault(key, []).append(opp)

    candidates = [rows for rows in grouped.values() if len({int(item["row"]) for item in rows}) >= 2]
    if not candidates:
        return None

    filter_colors = _available_filter_colors(grid)
    if len(filter_colors) < 2:
        return None
    filter_colors = [int(color) for color in rng.choice(filter_colors, size=2, replace=False)]

    rows = candidates[int(rng.integers(len(candidates)))]
    unique_by_row: dict[int, dict[str, int]] = {}
    for item in rows:
        unique_by_row.setdefault(int(item["row"]), item)
    chosen_rows = sorted(unique_by_row)
    row_a, row_b = sorted(rng.choice(chosen_rows, size=2, replace=False))

    out = grid.copy()
    first = unique_by_row[int(row_a)]
    second = unique_by_row[int(row_b)]
    _place_filter_segment(
        rng,
        out,
        row=int(first["row"]),
        target_col=int(first["target_col"]),
        left=int(first["left"]),
        right=int(first["right"]),
        color=int(filter_colors[0]),
    )
    _place_filter_segment(
        rng,
        out,
        row=int(second["row"]),
        target_col=int(second["target_col"]),
        left=int(second["left"]),
        right=int(second["right"]),
        color=int(filter_colors[1]),
    )
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    blocked_required, overlap_required = (
        (False, False),
        (False, True),
        (True, True),
    )[int(rng.integers(0, 3))]
    filter_mode = int(rng.integers(0, 3))

    base = _gen_tower_layout(
        rng,
        blocked_required=blocked_required,
        overlap_required=overlap_required,
    )
    if base is None:
        return None

    grid = base.copy()
    _base_out, meta = _solve(grid)
    if filter_mode == 1:
        updated = _add_single_filter(rng, grid, meta)
        if updated is None:
            return None
        grid = updated
    elif filter_mode == 2:
        updated = _add_multi_filters(rng, grid, meta)
        if updated is None:
            return None
        grid = updated

    values, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(value): int(count) for value, count in zip(values, counts)}
    bg_count = count_by_color.get(BG, 0)
    if any(bg_count <= count for color, count in count_by_color.items() if color != BG):
        return None

    _out, final_meta = _solve(grid)
    if bool(final_meta["blocked_exists"]) != blocked_required:
        return None
    if bool(final_meta["overlap_exists"]) != overlap_required:
        return None
    if filter_mode == 0 and bool(final_meta["filtered_exists"]):
        return None
    if filter_mode == 1 and (not bool(final_meta["filtered_exists"]) or bool(final_meta["multi_filter_exists"])):
        return None
    if filter_mode == 2 and not bool(final_meta["multi_filter_exists"]):
        return None

    towers = list(final_meta["towers"])
    if len(towers) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower-and-filter scene")
