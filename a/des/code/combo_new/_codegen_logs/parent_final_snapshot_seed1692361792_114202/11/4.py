import random

import numpy as np


BLACK = 0
GRAY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATE_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_scene(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int]]]:
    bg = _most_common_color(grid)
    non_bg = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(non_bg) >= 2, "expected wall color and at least one seed color"

    wall_color = max(non_bg, key=lambda color: int(np.count_nonzero(grid == color)))
    assert int(np.count_nonzero(grid == wall_color)) >= 2, "wall structure must have area >= 2"

    seed_mask = (grid != bg) & (grid != wall_color)
    seed_coords = np.argwhere(seed_mask)
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in seed_coords]
    assert seeds, "expected at least one seed"

    seed_colors = [color for _r, _c, color in seeds]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    for color in seed_colors:
        assert int(np.count_nonzero(grid == color)) == 1, "each seed color must appear once"

    return bg, wall_color, seeds


def _solve(grid: np.ndarray, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, wall_color, seeds = _extract_scene(g)
    h, w = g.shape

    best_dist = np.full((h, w), -1, dtype=int)
    best_color = np.full((h, w), bg, dtype=int)
    tied = np.zeros((h, w), dtype=bool)
    claim_count = np.zeros((h, w), dtype=int)
    stop_counts = {"wall": 0, "boundary": 0, "seed": 0}

    for r, c, color in seeds:
        for dr, dc in DIRECTIONS:
            step = 1
            while True:
                nr = r + dr * step
                nc = c + dc * step
                if not _in_bounds(g.shape, nr, nc):
                    stop_counts["boundary"] += 1
                    break

                value = int(g[nr, nc])
                if value == bg:
                    claim_count[nr, nc] += 1
                    if step > best_dist[nr, nc]:
                        best_dist[nr, nc] = step
                        best_color[nr, nc] = color
                        tied[nr, nc] = False
                    elif step == best_dist[nr, nc] and best_color[nr, nc] != color:
                        tied[nr, nc] = True
                    step += 1
                    continue

                if value == wall_color:
                    stop_counts["wall"] += 1
                else:
                    stop_counts["seed"] += 1
                break

    out = g.copy()
    fill_mask = (g == bg) & (best_dist >= 1) & ~tied
    out[fill_mask] = best_color[fill_mask]

    if not return_stats:
        return out

    contest_mask = claim_count >= 2
    stats = {
        "seed_count": len(seeds),
        "filled_cells": int(np.count_nonzero(fill_mask)),
        "contest_cells": int(np.count_nonzero(contest_mask)),
        "tie_cells": int(np.count_nonzero(tied)),
        "farthest_win_cells": int(np.count_nonzero(contest_mask & ~tied & (best_dist >= 1))),
        "wall_stop_rays": int(stop_counts["wall"]),
        "boundary_stop_rays": int(stop_counts["boundary"]),
        "seed_stop_rays": int(stop_counts["seed"]),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each colored seed along the corridor axes and let the farther ray win shared cells.

    Input structure:
    - The background is black.
    - Gray one-cell-thick horizontal or vertical wall segments form fixed obstacles.
    - Several isolated colored single-pixel seeds sit in the open corridor cells.

    Rule:
    - Keep the wall pixels and the original seed pixels unchanged.
    - From every seed, extend four orthogonal rays through black cells until the next cell is outside the grid, a wall, or another seed.
    - Each visited black cell is claimed by the seed color together with its step distance from that seed.
    - If multiple seeds claim the same black cell, keep the color from the strictly larger step distance; equal step distances cancel and leave the cell black.

    Key insight:
    - The overlap priority is deliberately reversed: the farther seed overrides the nearer one, and exact distance ties erase the fill.
    """
    return _solve(grid)


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _segment_cells(kind: str, fixed: int, start: int, end: int) -> list[tuple[int, int]]:
    if kind == "h":
        return [(fixed, c) for c in range(start, end + 1)]
    return [(r, fixed) for r in range(start, end + 1)]


def _can_place_seed(
    grid: np.ndarray,
    pos: tuple[int, int],
    seed_positions: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
    avoid_wall_adjacency: bool = True,
) -> bool:
    r, c = pos
    if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK or pos in reserved_open:
        return False
    if r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1:
        return False
    for sr, sc in seed_positions:
        if max(abs(sr - r), abs(sc - c)) <= 1:
            return False
    if avoid_wall_adjacency:
        for nr, nc in _expanded({pos}, grid.shape, radius=1):
            if int(grid[nr, nc]) == GRAY:
                return False
    return True


def _can_place_segment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    if not all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK for r, c in cell_set):
        return False
    if cell_set & reserved_open:
        return False
    seed_buffer = _expanded(set(seed_positions), grid.shape, radius=1)
    if cell_set & seed_buffer:
        return False
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GRAY)}
    wall_buffer = _expanded(wall_cells, grid.shape, radius=1)
    if cell_set & wall_buffer:
        return False
    return True


def _reserve_line(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        return {(r, c) for c in range(min(a[1], b[1]) + 1, max(a[1], b[1]))}
    c = a[1]
    return {(r, c) for r in range(min(a[0], b[0]) + 1, max(a[0], b[0]))}


def _place_contest_pair(
    grid: np.ndarray,
    colors: list[int],
    seed_positions: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
) -> tuple[bool, int]:
    h, w = grid.shape
    want_tie = random.choice([False, True])
    orientation = random.choice(["h", "v"])
    gaps = [3, 5, 7] if want_tie else [2, 4, 6]

    for _ in range(120):
        if orientation == "h":
            valid_gaps = [gap for gap in gaps if gap + 5 <= w]
            if not valid_gaps:
                break
            gap = random.choice(valid_gaps)
            r = random.randint(1, h - 2)
            c1 = random.randint(1, w - gap - 3)
            c2 = c1 + gap + 1
            p1 = (r, c1)
            p2 = (r, c2)
        else:
            valid_gaps = [gap for gap in gaps if gap + 5 <= h]
            if not valid_gaps:
                break
            gap = random.choice(valid_gaps)
            c = random.randint(1, w - 2)
            r1 = random.randint(1, h - gap - 3)
            r2 = r1 + gap + 1
            p1 = (r1, c)
            p2 = (r2, c)

        if not _can_place_seed(grid, p1, seed_positions, reserved_open):
            continue
        if not _can_place_seed(grid, p2, seed_positions + [p1], reserved_open):
            continue

        grid[p1] = colors[0]
        grid[p2] = colors[1]
        seed_positions.extend([p1, p2])
        reserved_open.update(_reserve_line(p1, p2))
        return want_tie, 2

    raise RuntimeError("failed to place the contest pair")


def _place_boundary_seed(
    grid: np.ndarray,
    color: int,
    seed_positions: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
) -> None:
    h, w = grid.shape

    for dr, dc in random.sample(DIRECTIONS, k=len(DIRECTIONS)):
        for _ in range(120):
            reserved_rows = {r for r, _c in reserved_open}
            reserved_cols = {c for _r, c in reserved_open}
            used_rows = {r for r, _c in seed_positions}
            used_cols = {c for _r, c in seed_positions}
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if r in used_rows or c in used_cols or r in reserved_rows or c in reserved_cols:
                continue
            if not _can_place_seed(grid, (r, c), seed_positions, reserved_open):
                continue

            ray: list[tuple[int, int]] = []
            nr = r + dr
            nc = c + dc
            while _in_bounds(grid.shape, nr, nc):
                ray.append((nr, nc))
                nr += dr
                nc += dc

            if len(ray) < 2 or set(ray) & reserved_open:
                continue

            grid[r, c] = color
            seed_positions.append((r, c))
            reserved_open.update(ray)
            return

    raise RuntimeError("failed to place the boundary-stopped seed")


def _place_wall_seed_and_barrier(
    grid: np.ndarray,
    color: int,
    seed_positions: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
) -> None:
    h, w = grid.shape

    for dr, dc in random.sample(DIRECTIONS, k=len(DIRECTIONS)):
        for _ in range(200):
            reserved_rows = {r for r, _c in reserved_open}
            reserved_cols = {c for _r, c in reserved_open}
            used_rows = {r for r, _c in seed_positions}
            used_cols = {c for _r, c in seed_positions}

            if dr == 0:
                span_choices = [span for span in (2, 3, 4, 5) if span + 5 <= w]
                if not span_choices:
                    continue
                span = random.choice(span_choices)
                r = random.randint(2, h - 3)
                if r in used_rows or r in reserved_rows:
                    continue
                if dc > 0:
                    c_seed = random.randint(1, w - span - 4)
                    c_wall = c_seed + span + 1
                else:
                    c_seed = random.randint(span + 2, w - 2)
                    c_wall = c_seed - span - 1
                if c_seed in used_cols or c_seed in reserved_cols:
                    continue
                pos = (r, c_seed)
                path = _reserve_line(pos, (r, c_wall))

                seg_len = random.randint(4, min(8, h))
                top_min = max(0, r - seg_len + 1)
                top_max = min(r, h - seg_len)
                if top_min > top_max:
                    continue
                top = random.randint(top_min, top_max)
                barrier = _segment_cells("v", c_wall, top, top + seg_len - 1)
            else:
                span_choices = [span for span in (2, 3, 4, 5) if span + 5 <= h]
                if not span_choices:
                    continue
                span = random.choice(span_choices)
                c = random.randint(2, w - 3)
                if c in used_cols or c in reserved_cols:
                    continue
                if dr > 0:
                    r_seed = random.randint(1, h - span - 4)
                    r_wall = r_seed + span + 1
                else:
                    r_seed = random.randint(span + 2, h - 2)
                    r_wall = r_seed - span - 1
                if r_seed in used_rows or r_seed in reserved_rows:
                    continue
                pos = (r_seed, c)
                path = _reserve_line(pos, (r_wall, c))

                seg_len = random.randint(4, min(8, w))
                left_min = max(0, c - seg_len + 1)
                left_max = min(c, w - seg_len)
                if left_min > left_max:
                    continue
                left = random.randint(left_min, left_max)
                barrier = _segment_cells("h", r_wall, left, left + seg_len - 1)

            if path & reserved_open:
                continue
            if not _can_place_seed(grid, pos, seed_positions, reserved_open):
                continue
            if not _can_place_segment(grid, barrier, reserved_open, seed_positions):
                continue

            grid[pos] = color
            for cell in barrier:
                grid[cell] = GRAY
            seed_positions.append(pos)
            reserved_open.update(path)
            return

    raise RuntimeError("failed to place the wall-stopped seed")


def _add_random_walls(
    grid: np.ndarray,
    reserved_open: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    wall_count: int,
) -> None:
    h, w = grid.shape

    for _ in range(wall_count):
        placed = False
        for _ in range(160):
            kind = random.choice(["h", "v"])
            if kind == "h":
                length = random.randint(3, min(10, w))
                row = random.randint(0, h - 1)
                col0 = random.randint(0, w - length)
                cells = _segment_cells("h", row, col0, col0 + length - 1)
            else:
                length = random.randint(3, min(10, h))
                col = random.randint(0, w - 1)
                row0 = random.randint(0, h - length)
                cells = _segment_cells("v", col, row0, row0 + length - 1)

            if not _can_place_segment(grid, cells, reserved_open, seed_positions):
                continue

            for cell in cells:
                grid[cell] = GRAY
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to place an auxiliary wall")


def _add_extra_seeds(
    grid: np.ndarray,
    colors: list[int],
    seed_positions: list[tuple[int, int]],
    reserved_open: set[tuple[int, int]],
) -> int:
    h, w = grid.shape
    added = 0

    for color in colors:
        placed = False
        for _ in range(120):
            reserved_rows = {r for r, _c in reserved_open}
            reserved_cols = {c for _r, c in reserved_open}
            used_rows = {r for r, _c in seed_positions}
            used_cols = {c for _r, c in seed_positions}
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if r in used_rows or c in used_cols or r in reserved_rows or c in reserved_cols:
                continue
            if not _can_place_seed(grid, (r, c), seed_positions, reserved_open):
                continue

            grid[r, c] = color
            seed_positions.append((r, c))
            added += 1
            placed = True
            break

        if not placed:
            break

    return added


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = list(SEED_COLORS)
    random.shuffle(colors)
    seed_positions: list[tuple[int, int]] = []
    reserved_open: set[tuple[int, int]] = set()

    try:
        want_tie, color_index = _place_contest_pair(grid, colors, seed_positions, reserved_open)
        _place_boundary_seed(grid, colors[color_index], seed_positions, reserved_open)
        color_index += 1
        _place_wall_seed_and_barrier(grid, colors[color_index], seed_positions, reserved_open)
        color_index += 1

        extra_walls = random.randint(1, 3)
        _add_random_walls(grid, reserved_open, seed_positions, extra_walls)

        extra_seed_budget = random.randint(0, 2)
        _add_extra_seeds(
            grid,
            colors[color_index : color_index + extra_seed_budget],
            seed_positions,
            reserved_open,
        )
    except RuntimeError:
        return None

    out, stats = _solve(grid, return_stats=True)
    if np.array_equal(out, grid):
        return None
    if stats["seed_count"] < 3:
        return None
    if stats["filled_cells"] < 8:
        return None
    if stats["contest_cells"] < 2 or stats["farthest_win_cells"] < 1:
        return None
    if stats["wall_stop_rays"] < 1 or stats["boundary_stop_rays"] < 1 or stats["seed_stop_rays"] < 2:
        return None
    if want_tie and stats["tie_cells"] < 1:
        return None
    if not want_tie and stats["tie_cells"] != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-and-wall ray puzzle")
