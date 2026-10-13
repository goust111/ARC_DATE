from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _seed_list(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != BLACK) & (grid != GREY))
    ]


def _bfs_distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0

    while q:
        r, c = q.popleft()
        next_dist = int(dist[r, c]) + 1
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(passable.shape, rr, cc):
                continue
            if not passable[rr, cc] or dist[rr, cc] >= 0:
                continue
            dist[rr, cc] = next_dist
            q.append((rr, cc))

    return dist


def _solve_with_metadata(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seeds = _seed_list(g)
    assert seeds, "expected at least one non-grey seed pixel"

    passable = g != GREY
    distance_maps = [_bfs_distances(passable, (r, c)) for r, c, _color in seeds]

    out = g.copy()
    for r, c in np.argwhere(g == BLACK):
        best_dist: int | None = None
        best_colors: set[int] = set()
        for (_sr, _sc, color), dist_map in zip(seeds, distance_maps):
            dist = int(dist_map[r, c])
            if dist < 0:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_colors = {int(color)}
            elif dist == best_dist:
                best_colors.add(int(color))

        if best_dist is None:
            continue
        out[r, c] = GREY if len(best_colors) > 1 else next(iter(best_colors))

    return out, seeds, distance_maps


def _naive_partition(grid: np.ndarray, seeds: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == BLACK):
        best_dist: int | None = None
        best_colors: set[int] = set()
        for sr, sc, color in seeds:
            dist = abs(int(r) - sr) + abs(int(c) - sc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_colors = {int(color)}
            elif dist == best_dist:
                best_colors.add(int(color))
        out[r, c] = GREY if len(best_colors) > 1 else next(iter(best_colors))
    return out


def _semantic_stats(grid: np.ndarray) -> dict[str, object]:
    out, seeds, distance_maps = _solve_with_metadata(grid)
    baseline = _naive_partition(grid, seeds)

    different_color_ties = 0
    same_color_ties = 0
    unreachable = 0
    geodesic_mismatches = 0

    for r, c in np.argwhere(np.asarray(grid) == BLACK):
        best_dist: int | None = None
        best_seed_count = 0
        best_colors: set[int] = set()
        for (_sr, _sc, color), dist_map in zip(seeds, distance_maps):
            dist = int(dist_map[r, c])
            if dist < 0:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_seed_count = 1
                best_colors = {int(color)}
            elif dist == best_dist:
                best_seed_count += 1
                best_colors.add(int(color))

        if best_dist is None:
            unreachable += 1
            continue

        if int(out[r, c]) != int(baseline[r, c]):
            geodesic_mismatches += 1

        if len(best_colors) > 1:
            different_color_ties += 1
        elif best_seed_count > 1:
            same_color_ties += 1

    return {
        "output": out,
        "different_color_ties": different_color_ties,
        "same_color_ties": same_color_ties,
        "unreachable": unreachable,
        "geodesic_mismatches": geodesic_mismatches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition black cells by nearest seed using four-connected geodesic distance around grey walls.

    Input structure:
    - Black background contains grey wall pixels and several colored single-pixel seeds.
    - Walls may create bent corridors, sealed pockets, and multiple disconnected passable regions.

    Rule:
    - For each black cell, measure shortest path length to every seed using only four-connected moves through black cells and seed cells, never through grey walls.
    - If one nearest color wins, or several nearest seeds share the same color, paint the cell with that color.
    - If nearest seeds of different colors tie, paint the cell grey; if no seed is reachable, keep the cell black.
    - Preserve every original wall pixel and every original seed pixel.

    Key insight:
    - The partition is determined by geodesic wavefronts constrained by walls, so ties and ownership changes depend on detours and sealed pockets rather than straight-line proximity.
    """
    return _solve_with_metadata(grid)[0]


def _draw_box(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = GREY
    grid[top + height - 1, left : left + width] = GREY
    grid[top : top + height, left] = GREY
    grid[top : top + height, left + width - 1] = GREY


def _rotated(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, random.randrange(4)).copy()


def _make_same_color_room(color: int) -> np.ndarray:
    inner_len = random.choice((5, 7, 9, 11))
    room = np.full((3, inner_len + 2), BLACK, dtype=int)
    _draw_box(room, 0, 0, *room.shape)
    room[1, 1] = int(color)
    room[1, -2] = int(color)
    return _rotated(room)


def _make_pocket() -> np.ndarray:
    inner_h = random.randint(2, 4)
    inner_w = random.randint(2, 4)
    room = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    _draw_box(room, 0, 0, *room.shape)
    if random.random() < 0.5:
        room = room.T.copy()
    return room


def _make_diff_color_room(color_a: int, color_b: int) -> np.ndarray | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        inner_h = random.randint(7, 11)
        inner_w = random.randint(9, 14)
        room = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
        _draw_box(room, 0, 0, inner_h + 2, inner_w + 2)

        wall_c = random.randint(3, inner_w - 2)
        door_r = random.choice((1, 2, inner_h - 1, inner_h))
        room[1 : inner_h + 1, wall_c] = GREY
        room[door_r, wall_c] = BLACK

        left_cols = sorted({1, max(1, wall_c - 1), max(1, wall_c - 2)})
        right_cols = sorted({inner_w, min(inner_w, wall_c + 1), min(inner_w, wall_c + 2)})

        left_candidates = [
            (r, c)
            for r in range(1, inner_h + 1)
            for c in left_cols
            if room[r, c] == BLACK
        ]
        right_candidates = [
            (r, c)
            for r in range(1, inner_h + 1)
            for c in right_cols
            if room[r, c] == BLACK
        ]

        left_candidates.sort(key=lambda rc: abs(rc[0] - door_r), reverse=True)
        right_candidates.sort(key=lambda rc: abs(rc[0] - door_r))
        left_candidates = left_candidates[:18]
        right_candidates = right_candidates[:18]

        for left_seed in left_candidates:
            for right_seed in right_candidates:
                trial = room.copy()
                trial[left_seed] = int(color_a)
                trial[right_seed] = int(color_b)
                stats = _semantic_stats(trial)
                if (
                    int(stats["different_color_ties"]) >= 2
                    and int(stats["geodesic_mismatches"]) >= 2
                    and int(stats["unreachable"]) == 0
                ):
                    return _rotated(trial)

    return None


def _rect_is_free(
    rect: tuple[int, int, int, int],
    used: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    pad: int,
) -> bool:
    top, left, height, width = rect
    rows, cols = shape
    if top < 1 or left < 1 or top + height > rows - 1 or left + width > cols - 1:
        return False

    for other_top, other_left, other_h, other_w in used:
        separated = (
            top + height + pad <= other_top
            or other_top + other_h + pad <= top
            or left + width + pad <= other_left
            or other_left + other_w + pad <= left
        )
        if not separated:
            return False

    return True


def _place_rect(
    shape: tuple[int, int],
    height: int,
    width: int,
    used: list[tuple[int, int, int, int]],
    *,
    pad: int = 2,
) -> tuple[int, int] | None:
    rows, cols = shape
    if rows - height - 1 < 1 or cols - width - 1 < 1:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(1, rows - height - 1)
        left = random.randint(1, cols - width - 1)
        rect = (top, left, height, width)
        if _rect_is_free(rect, used, shape, pad):
            used.append(rect)
            return top, left

    return None


def _build_candidate() -> np.ndarray | None:
    rows = random.randint(22, 29)
    cols = random.randint(22, 30)
    grid = np.full((rows, cols), BLACK, dtype=int)

    same_color, diff_a, diff_b, outside_color = random.sample(SEED_COLORS, 4)
    same_room = _make_same_color_room(same_color)
    diff_room = _make_diff_color_room(diff_a, diff_b)
    if diff_room is None:
        return None

    pocket_count = 1 if random.random() < 0.6 else 2
    pockets = [_make_pocket() for _ in range(pocket_count)]

    used: list[tuple[int, int, int, int]] = []
    placed: list[tuple[np.ndarray, tuple[int, int]]] = []

    for patch in (diff_room, same_room, *pockets):
        location = _place_rect(grid.shape, patch.shape[0], patch.shape[1], used)
        if location is None:
            return None
        placed.append((patch, location))

    outside_mask = np.ones(grid.shape, dtype=bool)
    for patch, (top, left) in placed:
        height, width = patch.shape
        grid[top : top + height, left : left + width] = patch
        outside_mask[top : top + height, left : left + width] = False

    outside_cells = np.argwhere(outside_mask & (grid == BLACK))
    if len(outside_cells) == 0:
        return None

    seed_r, seed_c = map(int, random.choice(outside_cells.tolist()))
    grid[seed_r, seed_c] = int(outside_color)
    return grid


def _is_valid_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    stats = _semantic_stats(g)
    out = np.asarray(stats["output"], dtype=int)
    seed_mask = (g != BLACK) & (g != GREY)

    if not np.array_equal(out[g == GREY], g[g == GREY]):
        return False
    if not np.array_equal(out[seed_mask], g[seed_mask]):
        return False
    if not np.any(out != g):
        return False

    if int(stats["different_color_ties"]) < 2:
        return False
    if int(stats["same_color_ties"]) < 1:
        return False
    if int(stats["unreachable"]) < 4:
        return False
    if int(stats["geodesic_mismatches"]) < 3:
        return False

    black_count = int(np.sum(g == BLACK))
    if int(stats["unreachable"]) > max(16, black_count // 4):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _is_valid_example(grid):
            return grid
    raise RuntimeError("failed to generate a non-degenerate geodesic partition example")
