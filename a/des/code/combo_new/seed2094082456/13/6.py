from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ROOM_LAYOUTS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
ROOM_SIDE_RANGE = (6, 9)
MAX_GENERATION_ATTEMPTS = 300
MAX_ROOM_ATTEMPTS = 120


def _clockwise(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _segments_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(set(int(x) for x in separators))
    assert all(0 <= s < length for s in separators)
    assert all(b - a > 1 for a, b in zip(separators, separators[1:])), "separator lines must be single-cell wide"

    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators + [length]:
        if start < sep:
            spans.append((start, sep))
        start = sep + 1
    assert spans, "separator lines must leave at least one room"
    return spans


def _room_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == MAROON)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == MAROON)]
    assert sep_rows or sep_cols, "expected at least one maroon separator line"

    maroon_mask = grid == MAROON
    valid_maroon = np.zeros_like(maroon_mask, dtype=bool)
    if sep_rows:
        valid_maroon[sep_rows, :] = True
    if sep_cols:
        valid_maroon[:, sep_cols] = True
    assert np.array_equal(maroon_mask, valid_maroon), "maroon pixels must form full separator rows/columns"

    row_spans = _segments_from_separators(h, sep_rows)
    col_spans = _segments_from_separators(w, sep_cols)
    return row_spans, col_spans, sep_rows, sep_cols


def _trace_room(room: np.ndarray) -> dict[str, Any]:
    h, w = room.shape
    assert h >= 1 and w >= 1
    assert not np.any(room == MAROON)
    assert np.all(np.isin(room, [BLACK, GREY, *SEED_COLORS]))

    seed_pos = np.argwhere(np.isin(room, SEED_COLORS))
    assert seed_pos.shape == (1, 2), "each room must contain exactly one seed"
    sr, sc = map(int, seed_pos[0])
    color = int(room[sr, sc])

    straight_by_dir: dict[tuple[int, int], list[tuple[int, int]]] = {}
    straight_occ = room != BLACK
    bounces: list[dict[str, Any]] = []

    for dr, dc in DIRS4:
        cells: list[tuple[int, int]] = []
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(room[r, c])
            if value == BLACK:
                cells.append((r, c))
                r += dr
                c += dc
                continue
            if value == GREY:
                turn = _clockwise(dr, dc)
                bounces.append(
                    {
                        "from_dir": (dr, dc),
                        "wall": (int(r), int(c)),
                        "turn_dir": turn,
                    }
                )
            break
        straight_by_dir[(dr, dc)] = cells
        for cell in cells:
            straight_occ[cell] = True

    for bounce in bounces:
        wall_r, wall_c = bounce["wall"]
        dr, dc = bounce["turn_dir"]
        r, c = wall_r + dr, wall_c + dc
        cells: list[tuple[int, int]] = []
        while 0 <= r < h and 0 <= c < w and not straight_occ[r, c]:
            cells.append((int(r), int(c)))
            r += dr
            c += dc
        bounce["cells"] = cells

    return {
        "seed": (sr, sc),
        "color": color,
        "straight_by_dir": straight_by_dir,
        "bounces": bounces,
    }


def _solve_room(room: np.ndarray) -> np.ndarray:
    traced = _trace_room(room)
    out = room.copy()
    color = traced["color"]

    for cells in traced["straight_by_dir"].values():
        for r, c in cells:
            out[r, c] = color

    for bounce in traced["bounces"]:
        for r, c in bounce["cells"]:
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each room's colored seed into four rays that bounce clockwise once off grey walls.

    Input structure:
    - The background is black.
    - Full-row and full-column maroon separator lines split the grid into independent rectangular rooms.
    - Each room contains exactly one seed pixel in red, blue, green, or yellow, plus some grey wall pixels on black background.

    Rule:
    - In every room, emit rays from the seed upward, rightward, downward, and leftward.
    - A ray paints each visited black cell with the seed color until it reaches the room boundary or a non-black obstacle.
    - If the first obstacle is a grey wall, the ray stops before that wall, rotates 90 degrees clockwise at the wall, then continues once from the wall's clockwise side; after the bounce it stops at the room boundary or before any non-black pixel.
    - Separator lines, seeds, and grey wall pixels keep their original colors.

    Key insight:
    - Maroon lines define separate rectangular chambers, and inside each chamber the solution is a deterministic two-stage ray cast: straight growth first, then at most one clockwise turn around any encountered wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, _sep_rows, _sep_cols = _room_spans(g)
    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            room = g[r0:r1, c0:c1]
            solved = _solve_room(room)
            out[r0:r1, c0:c1] = solved
    return out


def _sample_room_sizes(rng: random.Random, count: int) -> list[int]:
    lo, hi = ROOM_SIDE_RANGE
    return [rng.randint(lo, hi) for _ in range(count)]


def _build_base_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    r = 0
    for i, size in enumerate(row_sizes):
        row_spans.append((r, r + size))
        r += size
        if i != len(row_sizes) - 1:
            grid[r, :] = MAROON
            r += 1

    col_spans: list[tuple[int, int]] = []
    c = 0
    for i, size in enumerate(col_sizes):
        col_spans.append((c, c + size))
        c += size
        if i != len(col_sizes) - 1:
            grid[:, c] = MAROON
            c += 1

    return grid, row_spans, col_spans


def _path_to_edge(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _generate_room(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, dict[str, Any]] | None:
    for _ in range(MAX_ROOM_ATTEMPTS):
        room = np.full((h, w), BLACK, dtype=int)
        seed_color = rng.choice(SEED_COLORS)
        seed = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        room[seed] = seed_color

        dir_indices = list(range(4))
        rng.shuffle(dir_indices)
        bounce_count = rng.randint(1, 4)
        bounce_dirs = set(dir_indices[:bounce_count])

        primary_walls: set[tuple[int, int]] = set()
        blocker_walls: set[tuple[int, int]] = set()
        bounce_plan: dict[tuple[int, int], dict[str, Any]] = {}

        valid = True
        for idx, direction in enumerate(DIRS4):
            if idx not in bounce_dirs:
                continue

            dr, dc = direction
            turn = _clockwise(dr, dc)
            want_blocker = bool(rng.randint(0, 1))
            candidates: list[dict[str, Any]] = []

            max_straight = len(_path_to_edge((seed[0] + dr, seed[1] + dc), direction, h, w))
            for dist in range(1, max_straight):
                wall = (seed[0] + dist * dr, seed[1] + dist * dc)
                full_bounce = _path_to_edge((wall[0] + turn[0], wall[1] + turn[1]), turn, h, w)
                if want_blocker:
                    for blocker in full_bounce[1:]:
                        candidates.append(
                            {
                                "wall": wall,
                                "blocker": blocker,
                                "expected": full_bounce[: full_bounce.index(blocker)],
                            }
                        )
                elif full_bounce:
                    candidates.append({"wall": wall, "blocker": None, "expected": full_bounce})

            if not candidates:
                valid = False
                break

            choice = rng.choice(candidates)
            wall = choice["wall"]
            blocker = choice["blocker"]
            if wall in primary_walls or wall in blocker_walls:
                valid = False
                break
            if blocker is not None and (blocker == seed or blocker in primary_walls or blocker in blocker_walls):
                valid = False
                break

            primary_walls.add(wall)
            if blocker is not None:
                blocker_walls.add(blocker)
            bounce_plan[direction] = choice

        if not valid:
            continue

        all_walls = primary_walls | blocker_walls
        if seed in all_walls or len(all_walls) != len(primary_walls) + len(blocker_walls):
            continue

        for r, c in all_walls:
            room[r, c] = GREY

        traced = _trace_room(room)
        traced_bounces = {tuple(item["from_dir"]): item for item in traced["bounces"]}
        if set(traced_bounces) != set(bounce_plan):
            continue

        bounce_sets: list[set[tuple[int, int]]] = []
        for direction, plan in bounce_plan.items():
            actual = traced_bounces[direction]["cells"]
            if actual != plan["expected"]:
                valid = False
                break
            if not actual:
                valid = False
                break
            bounce_set = set(actual)
            if any(bounce_set & other for other in bounce_sets):
                valid = False
                break
            bounce_sets.append(bounce_set)
        if not valid:
            continue

        open_dirs = [direction for direction in DIRS4 if direction not in bounce_plan]
        if not open_dirs:
            # Allow all-bounce rooms, but only when they still paint enough cells.
            pass

        solved = _solve_room(room)
        added = int(np.count_nonzero((room == BLACK) & (solved == seed_color)))
        if added < 6:
            continue
        if np.count_nonzero(room == GREY) < 1:
            continue

        info = {
            "seed_color": seed_color,
            "bounce_dirs": set(bounce_plan),
            "open_dirs": set(open_dirs),
            "has_blocked_bounce": any(plan["blocker"] is not None for plan in bounce_plan.values()),
            "added_cells": added,
        }
        return room, info
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows, room_cols = rng.choice(ROOM_LAYOUTS)
    row_sizes = _sample_room_sizes(rng, room_rows)
    col_sizes = _sample_room_sizes(rng, room_cols)
    grid, row_spans, col_spans = _build_base_grid(row_sizes, col_sizes)

    room_infos: list[dict[str, Any]] = []
    room_colors: list[int] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            generated = _generate_room(rng, r1 - r0, c1 - c0)
            if generated is None:
                return None
            room, info = generated
            grid[r0:r1, c0:c1] = room
            room_infos.append(info)
            room_colors.append(info["seed_color"])

    if len(set(room_colors)) == 1 and len(room_colors) > 1:
        alt_colors = [color for color in SEED_COLORS if color != room_colors[0]]
        pick = rng.randrange(len(room_infos))
        r_index = pick // len(col_spans)
        c_index = pick % len(col_spans)
        r0, r1 = row_spans[r_index]
        c0, c1 = col_spans[c_index]
        seed_pos = np.argwhere(np.isin(grid[r0:r1, c0:c1], SEED_COLORS))
        assert seed_pos.shape == (1, 2)
        sr, sc = map(int, seed_pos[0])
        grid[r0 + sr, c0 + sc] = rng.choice(alt_colors)
        room_infos[pick]["seed_color"] = int(grid[r0 + sr, c0 + sc])

    total_open = sum(len(info["open_dirs"]) for info in room_infos)
    total_bounce = sum(len(info["bounce_dirs"]) for info in room_infos)
    if total_open == 0 or total_bounce == 0:
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None
    if not np.any(solved != grid):
        return None

    painted = int(np.count_nonzero((grid == BLACK) & (solved != BLACK)))
    if painted < 18:
        return None

    if max(grid.shape) > 30 or max(solved.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
