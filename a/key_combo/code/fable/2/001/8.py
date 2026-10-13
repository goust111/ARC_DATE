from __future__ import annotations

import random

import numpy as np


_BG = 0
_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_DIR_FROM_DELTA = {delta: name for name, delta in _DIRS.items()}
_MAX_GENERATION_ATTEMPTS = 300


def _find_lattice(grid: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates = []
    for color in sorted(set(map(int, np.unique(grid)))):
        row_lines = [r for r in range(height) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(width) if np.all(grid[:, c] == color)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue

        row_steps = {row_lines[i + 1] - row_lines[i] for i in range(len(row_lines) - 1)}
        col_steps = {col_lines[i + 1] - col_lines[i] for i in range(len(col_lines) - 1)}
        if len(row_steps) != 1 or len(col_steps) != 1:
            continue

        row_step = row_steps.pop()
        col_step = col_steps.pop()
        cell_h = row_step - 1
        cell_w = col_step - 1
        if cell_h != cell_w or cell_h < 3 or cell_h % 2 == 0:
            continue
        if row_lines[0] != 0 or row_lines[-1] != height - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != width - 1:
            continue
        if row_lines != list(range(0, height, row_step)):
            continue
        if col_lines != list(range(0, width, col_step)):
            continue
        candidates.append((int(color), int(cell_h), row_lines, col_lines))

    assert len(candidates) == 1, "expected exactly one separator lattice"
    return candidates[0]


def _analyze_cell(patch: np.ndarray) -> tuple[int, int | None, np.ndarray]:
    size = int(patch.shape[0])
    mid = size // 2
    bg = int(patch[0, 0])
    color = int(patch[mid, mid])
    assert color != bg, "cross center must be colored"

    expected = np.full_like(patch, bg)
    expected[mid, :] = color
    expected[:, mid] = color

    missing = []
    rays = {
        "up": patch[:mid, mid],
        "down": patch[mid + 1 :, mid],
        "left": patch[mid, :mid],
        "right": patch[mid, mid + 1 :],
    }
    for name, ray in rays.items():
        present = bool(np.all(ray == color))
        if not present:
            missing.append(name)
            if name == "up":
                expected[:mid, mid] = bg
            elif name == "down":
                expected[mid + 1 :, mid] = bg
            elif name == "left":
                expected[mid, :mid] = bg
            else:
                expected[mid, mid + 1 :] = bg

    assert len(missing) in (0, 1), "each cross must be complete or miss exactly one arm"
    assert np.array_equal(patch, expected), "unexpected cell pattern"
    return color, (None if not missing else missing[0]), patch != bg


def _parse_scene(grid: np.ndarray) -> dict:
    line_color, cell_size, row_lines, col_lines = _find_lattice(grid)
    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1

    cells: dict[tuple[int, int], dict] = {}
    complete = []
    body_colors = set()
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1]
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1]
            patch = grid[r0:r1, c0:c1]
            color, missing, mask = _analyze_cell(patch)
            cells[(rr, cc)] = {
                "color": int(color),
                "missing": missing,
                "mask": mask,
                "slice": (r0, r1, c0, c1),
            }
            if missing is None:
                complete.append((rr, cc))
            else:
                body_colors.add(int(color))

    assert len(complete) == 1, "expected exactly one complete start cross"
    start = complete[0]
    start_color = int(cells[start]["color"])
    assert len(body_colors) == 1, "all incomplete crosses must share one color"
    body_color = next(iter(body_colors))
    assert body_color != start_color, "start cross must be a different color"

    return {
        "grid": np.asarray(grid, dtype=int),
        "line_color": line_color,
        "cell_size": cell_size,
        "shape": (cell_rows, cell_cols),
        "cells": cells,
        "start": start,
        "start_color": start_color,
        "body_color": body_color,
    }


def _trace_route(scene: dict) -> list[tuple[int, int]]:
    cell_rows, cell_cols = scene["shape"]
    cells = scene["cells"]
    current = scene["start"]
    route = [current]
    seen = {current}

    while True:
        r, c = current
        candidates = []
        for dr, dc in _DIRS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < cell_rows and 0 <= nc < cell_cols):
                continue
            required = _DIR_FROM_DELTA[(-dr, -dc)]
            if cells[(nr, nc)]["missing"] == required:
                candidates.append((nr, nc))

        assert len(candidates) <= 1, "route must be unambiguous"
        if not candidates:
            return route

        nxt = candidates[0]
        assert nxt not in seen, "route must not cycle"
        route.append(nxt)
        seen.add(nxt)
        current = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique reverse-relay chain of incomplete crosses to the start cross's color.

    Input structure:
    - Full-width and full-height separator lines partition the canvas into equal odd-sized square cells.
    - Each cell contains a centered cross on a blank background; exactly one cross is complete and uses a unique start color.
    - Every other cell contains the same body-colored cross with exactly one missing arm.

    Rule:
    - Parse the lattice into cells and locate the unique complete start cross.
    - From the current cell, move to the unique orthogonal neighbor whose missing arm points back to the current cell.
    - Repeat until no such neighbor exists, then recolor every visited cross pixel to the start color while preserving each cross's missing-arm shape.

    Key insight:
    - A missing arm encodes where that cross came from, not where to go next, so the route is decoded by reading the arrows in reverse.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene = _parse_scene(g)
    route = _trace_route(scene)

    out = g.copy()
    for cell_pos in route:
        cell = scene["cells"][cell_pos]
        r0, r1, c0, c1 = cell["slice"]
        patch = out[r0:r1, c0:c1]
        patch[cell["mask"]] = scene["start_color"]
    return out


def _render_cross(size: int, color: int, missing: str | None, *, bg: int = _BG) -> np.ndarray:
    patch = np.full((size, size), int(bg), dtype=int)
    mid = size // 2
    patch[mid, :] = int(color)
    patch[:, mid] = int(color)
    if missing == "up":
        patch[:mid, mid] = int(bg)
    elif missing == "down":
        patch[mid + 1 :, mid] = int(bg)
    elif missing == "left":
        patch[mid, :mid] = int(bg)
    elif missing == "right":
        patch[mid, mid + 1 :] = int(bg)
    elif missing is not None:
        raise ValueError(f"unknown direction: {missing}")
    return patch


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(1, len(path) - 1):
        prev_delta = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        next_delta = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if prev_delta != next_delta:
            turns += 1
    return turns


def _sample_path(cell_rows: int, cell_cols: int, length: int, min_turns: int) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    random.shuffle(all_cells)

    def dfs(path: list[tuple[int, int]], seen: set[tuple[int, int]], turns: int) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return list(path) if turns >= min_turns else None

        current = path[-1]
        prev_delta = None if len(path) < 2 else (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        remaining_after_move = length - len(path) - 1

        options = []
        for dr, dc in _DIRS.values():
            nr, nc = current[0] + dr, current[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < cell_rows and 0 <= nc < cell_cols):
                continue
            if nxt in seen:
                continue
            new_turns = turns + int(prev_delta is not None and (dr, dc) != prev_delta)
            if new_turns + remaining_after_move < min_turns:
                continue
            degree = 0
            for dr2, dc2 in _DIRS.values():
                rr, cc = nr + dr2, nc + dc2
                if 0 <= rr < cell_rows and 0 <= cc < cell_cols and (rr, cc) not in seen and (rr, cc) != current:
                    degree += 1
            turn_bonus = 1 if prev_delta is not None and (dr, dc) != prev_delta else 0
            options.append((turn_bonus, degree, nxt, new_turns))

        random.shuffle(options)
        options.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _turn_bonus, _degree, nxt, new_turns in options:
            path.append(nxt)
            seen.add(nxt)
            result = dfs(path, seen, new_turns)
            if result is not None:
                return result
            seen.remove(nxt)
            path.pop()
        return None

    for start in all_cells:
        result = dfs([start], {start}, 0)
        if result is not None:
            return result
    return None


def _sample_straight_path(cell_rows: int, cell_cols: int, length: int) -> list[tuple[int, int]] | None:
    orientations = []
    if length <= cell_cols:
        orientations.append("horizontal")
    if length <= cell_rows:
        orientations.append("vertical")
    if not orientations:
        return None

    orientation = random.choice(orientations)
    if orientation == "horizontal":
        rr = random.randrange(cell_rows)
        cc0 = random.randint(0, cell_cols - length)
        step = random.choice([-1, 1])
        cols = list(range(cc0, cc0 + length))
        if step < 0:
            cols.reverse()
        return [(rr, cc) for cc in cols]

    cc = random.randrange(cell_cols)
    rr0 = random.randint(0, cell_rows - length)
    step = random.choice([-1, 1])
    rows = list(range(rr0, rr0 + length))
    if step < 0:
        rows.reverse()
    return [(rr, cc) for rr in rows]


def _build_specs(
    cell_rows: int,
    cell_cols: int,
    path: list[tuple[int, int]],
    body_color: int,
    start_color: int,
) -> dict[tuple[int, int], tuple[int, str | None]] | None:
    index_of = {pos: idx for idx, pos in enumerate(path)}
    route_set = set(path)
    specs: dict[tuple[int, int], tuple[int, str | None]] = {}

    for rr in range(cell_rows):
        for cc in range(cell_cols):
            pos = (rr, cc)
            if pos == path[0]:
                specs[pos] = (int(start_color), None)
                continue

            if pos in route_set:
                prev_pos = path[index_of[pos] - 1]
                delta = (prev_pos[0] - rr, prev_pos[1] - cc)
                specs[pos] = (int(body_color), _DIR_FROM_DELTA[delta])
                continue

            forbidden = set()
            for dr, dc in _DIRS.values():
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in route_set:
                    forbidden.add(_DIR_FROM_DELTA[(nr - rr, nc - cc)])
            options = [name for name in _DIRS if name not in forbidden]
            if not options:
                return None
            specs[pos] = (int(body_color), random.choice(options))

    return specs


def _validate_specs(
    cell_rows: int,
    cell_cols: int,
    path: list[tuple[int, int]],
    specs: dict[tuple[int, int], tuple[int, str | None]],
    start_color: int,
    body_color: int,
) -> bool:
    if len(path) < 3 or len(path) != len(set(path)):
        return False
    if specs[path[0]] != (start_color, None):
        return False
    route_set = set(path)
    for idx, pos in enumerate(path[1:], start=1):
        prev_pos = path[idx - 1]
        expected = _DIR_FROM_DELTA[(prev_pos[0] - pos[0], prev_pos[1] - pos[1])]
        if specs[pos] != (body_color, expected):
            return False

    for rr in range(cell_rows):
        for cc in range(cell_cols):
            pos = (rr, cc)
            color, missing = specs[pos]
            if pos not in route_set and color != body_color:
                return False
            if pos not in route_set and missing is None:
                return False

    for idx, current in enumerate(path):
        incoming = []
        for dr, dc in _DIRS.values():
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < cell_rows and 0 <= nc < cell_cols):
                continue
            needed = _DIR_FROM_DELTA[(-dr, -dc)]
            if specs[(nr, nc)][1] == needed:
                incoming.append((nr, nc))
        expected_next = [] if idx == len(path) - 1 else [path[idx + 1]]
        if incoming != expected_next:
            return False
    return True


def _render_grid(
    cell_rows: int,
    cell_cols: int,
    cell_size: int,
    line_color: int,
    specs: dict[tuple[int, int], tuple[int, str | None]],
) -> np.ndarray:
    height = cell_rows * cell_size + cell_rows + 1
    width = cell_cols * cell_size + cell_cols + 1
    grid = np.full((height, width), int(line_color), dtype=int)
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            r0 = 1 + rr * (cell_size + 1)
            c0 = 1 + cc * (cell_size + 1)
            color, missing = specs[(rr, cc)]
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = _render_cross(cell_size, color, missing)
    return grid


def _expected_output(
    grid: np.ndarray,
    cell_size: int,
    path: list[tuple[int, int]],
    start_color: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rr, cc in path:
        r0 = 1 + rr * (cell_size + 1)
        c0 = 1 + cc * (cell_size + 1)
        patch = out[r0 : r0 + cell_size, c0 : c0 + cell_size]
        patch[patch != _BG] = int(start_color)
    return out


def _choose_layout() -> tuple[int, int, int]:
    cell_size = random.choice([3, 5, 5, 7])
    if cell_size == 3:
        return cell_size, random.randint(4, 7), random.randint(4, 7)
    if cell_size == 5:
        return cell_size, random.randint(3, 4), random.randint(3, 4)
    return cell_size, 3, 3


def _gen_attempt() -> np.ndarray | None:
    cell_size, cell_rows, cell_cols = _choose_layout()
    max_len = min(cell_rows * cell_cols - 1, (cell_rows * cell_cols * 2) // 3, 9)
    if max_len < 3:
        return None

    path = None
    turn_floor = 0
    if random.random() < 0.25:
        straight_cap = min(max_len, max(cell_rows, cell_cols))
        if straight_cap >= 3:
            path_len = random.randint(3, straight_cap)
            path = _sample_straight_path(cell_rows, cell_cols, path_len)
    if path is None:
        path_len = random.randint(3, max_len)
        turn_floor = min(path_len - 2, random.choice([0, 1, 1, 2]))
        path = _sample_path(cell_rows, cell_cols, path_len, turn_floor)
        if path is None:
            return None

    line_color, body_color, start_color = random.sample(range(1, 10), 3)
    specs = _build_specs(cell_rows, cell_cols, path, body_color, start_color)
    if specs is None:
        return None
    if not _validate_specs(cell_rows, cell_cols, path, specs, start_color, body_color):
        return None
    if _count_turns(path) < turn_floor:
        return None

    grid = _render_grid(cell_rows, cell_cols, cell_size, line_color, specs)
    expected = _expected_output(grid, cell_size, path, start_color)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
