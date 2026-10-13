from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
DIR_VEC = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
OPPOSITE = {"right": "left", "left": "right", "down": "up", "up": "down"}
ARROW_OFFSETS = {
    "right": [(0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)],
    "left": [(0, 0), (0, -1), (0, -2), (-1, -2), (1, -2)],
    "down": [(0, 0), (1, 0), (2, 0), (2, -1), (2, 1)],
    "up": [(0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1)],
}
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(base: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    br, bc = base
    return [(br + dr, bc + dc) for dr, dc in ARROW_OFFSETS[direction]]


def _arrow_tip(base: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIR_VEC[direction]
    return int(base[0] + 2 * dr), int(base[1] + 2 * dc)


def _present_arrow_colors(g: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen = {BG, GREY}
    for value in g.ravel():
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _find_arrows(g: np.ndarray, color: int) -> list[dict]:
    arrows: list[dict] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != int(color):
                continue
            for direction in DIR_VEC:
                cells = _arrow_cells((r, c), direction)
                if all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == int(color) for rr, cc in cells):
                    arrows.append(
                        {
                            "base": (int(r), int(c)),
                            "direction": direction,
                            "cells": frozenset(cells),
                            "tip": _arrow_tip((r, c), direction),
                        }
                    )
    return arrows


def _select_arrow_pair(g: np.ndarray, color: int) -> tuple[dict, dict]:
    color_cells = {tuple(map(int, pt)) for pt in np.argwhere(g == int(color))}
    arrows = _find_arrows(g, color)
    fallback: tuple[dict, dict] | None = None

    for i, a in enumerate(arrows):
        for b in arrows[i + 1 :]:
            if OPPOSITE[a["direction"]] != b["direction"]:
                continue
            if a["cells"] & b["cells"]:
                continue
            covered = set(a["cells"]) | set(b["cells"])
            if covered == color_cells:
                return a, b
            if fallback is None:
                fallback = (a, b)

    assert fallback is not None, "each arrow color must form one opposite-facing pair"
    return fallback


def _blocked(g: np.ndarray, r: int, c: int) -> bool:
    return not _in_bounds(g.shape, r, c) or int(g[r, c]) == GREY


def _sidestep_candidates(
    g: np.ndarray, pos: tuple[int, int], direction: str, target: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = pos
    if direction in ("left", "right"):
        if target[0] > r:
            ordered = [(1, 0), (-1, 0)]
        elif target[0] < r:
            ordered = [(-1, 0), (1, 0)]
        else:
            ordered = [(1, 0), (-1, 0)]
    else:
        if target[1] > c:
            ordered = [(0, 1), (0, -1)]
        elif target[1] < c:
            ordered = [(0, -1), (0, 1)]
        else:
            ordered = [(0, 1), (0, -1)]

    return [(r + dr, c + dc) for dr, dc in ordered if not _blocked(g, r + dr, c + dc)]


def _mark_cell(out: np.ndarray, path: set[tuple[int, int]], r: int, c: int, color: int) -> None:
    current = int(out[r, c])
    assert current != GREY, "paths cannot pass through grey walls"
    assert current in (BG, int(color)), "valid channels do not overwrite other arrows or paths"
    out[r, c] = int(color)
    path.add((int(r), int(c)))


def _advance_path(
    g: np.ndarray,
    out: np.ndarray,
    path: set[tuple[int, int]],
    pos: tuple[int, int],
    direction: str,
    target: tuple[int, int],
    color: int,
) -> tuple[int, int]:
    dr, dc = DIR_VEC[direction]
    nr, nc = int(pos[0] + dr), int(pos[1] + dc)
    if _blocked(g, nr, nc):
        options = _sidestep_candidates(g, pos, direction, target)
        assert options, "blocked path must have a visible one-cell bypass"
        nr, nc = options[0]

    _mark_cell(out, path, nr, nc, color)
    return int(nr), int(nc)


def _clear_join(g: np.ndarray, a: tuple[int, int], b: tuple[int, int], axis: str) -> bool:
    if axis == "h":
        if a[0] != b[0]:
            return False
        r = int(a[0])
        c0, c1 = sorted((int(a[1]), int(b[1])))
        return all(int(g[r, c]) != GREY for c in range(c0, c1 + 1))

    if a[1] != b[1]:
        return False
    c = int(a[1])
    r0, r1 = sorted((int(a[0]), int(b[0])))
    return all(int(g[r, c]) != GREY for r in range(r0, r1 + 1))


def _join_paths(
    g: np.ndarray,
    out: np.ndarray,
    path: set[tuple[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
    axis: str,
    color: int,
) -> None:
    assert _clear_join(g, a, b, axis), "final gap must be clear of grey walls"
    if axis == "h":
        r = int(a[0])
        c0, c1 = sorted((int(a[1]), int(b[1])))
        for c in range(c0, c1 + 1):
            _mark_cell(out, path, r, c, color)
    else:
        c = int(a[1])
        r0, r1 = sorted((int(a[0]), int(b[0])))
        for r in range(r0, r1 + 1):
            _mark_cell(out, path, r, c, color)


def _paint_pair_path(g: np.ndarray, out: np.ndarray, a: dict, b: dict, color: int) -> set[tuple[int, int]]:
    axis = "h" if a["direction"] in ("left", "right") else "v"
    pos_a = tuple(a["tip"])
    pos_b = tuple(b["tip"])
    target_a = pos_b
    target_b = pos_a
    path = {pos_a, pos_b}
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(4 * g.shape[0] * g.shape[1]):
        if _clear_join(g, pos_a, pos_b, axis):
            _join_paths(g, out, path, pos_a, pos_b, axis, color)
            return path

        state = (pos_a, pos_b)
        assert state not in seen, "path simulation must make progress"
        seen.add(state)

        pos_a = _advance_path(g, out, path, pos_a, a["direction"], target_a, color)
        if _clear_join(g, pos_a, pos_b, axis):
            _join_paths(g, out, path, pos_a, pos_b, axis, color)
            return path

        pos_b = _advance_path(g, out, path, pos_b, b["direction"], target_b, color)

    raise AssertionError("paired arrows did not meet")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-colored opposing arrows with a routed path around grey walls.

    Input structure:
    - The background is black, and grey pixels form irregular obstacle walls.
    - Every non-black non-grey color appears as exactly two five-pixel arrows.
    - The two arrows of a color point in opposite horizontal or vertical directions, with their tips facing the channel between them.

    Rule:
    - For each same-colored arrow pair, start at the center tip of both arrows and extend in each arrow's pointing direction.
    - A path normally advances one pixel along its arrow direction; when the next pixel is grey or outside the grid, it turns one perpendicular pixel toward the partner arrow, then resumes the original direction.
    - When the two routed paths become aligned on the same row or column, fill the remaining clear gap with that color while preserving the arrow shapes and grey walls.

    Key insight:
    - The arrow shape gives both the color pairing and the main travel direction, while every obstacle bypass is biased toward the matching arrow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for color in _present_arrow_colors(g):
        a, b = _select_arrow_pair(g, color)
        _paint_pair_path(g, out, a, b, color)

    return out


# ------------------------------ generator ------------------------------

GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 200
MIN_SIDE = 18
MAX_SIDE = 30


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _spaced_sample(lo: int, hi: int, count: int, gap: int = 3) -> list[int] | None:
    if count == 0:
        return []
    if hi < lo:
        return None
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count:
        return None
    for _ in range(80):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _cells_clear(
    grid: np.ndarray,
    reserved: np.ndarray,
    cells: set[tuple[int, int]] | list[tuple[int, int]],
    pad: int,
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not _in_bounds(grid.shape, int(r), int(c)):
            return False
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        if np.any(grid[r0:r1, c0:c1] != BG):
            return False
        if np.any(reserved[r0:r1, c0:c1]):
            return False
    return True


def _reserve(mask: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        mask[r0:r1, c0:c1] = True


def _horizontal_wall_cells(r: int, c: int, shift: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {(int(r), int(c))}
    away = -int(shift)
    if away:
        for k in range(1, random.randint(2, 3)):
            rr = int(r + away * k)
            if _in_bounds(shape, rr, int(c)):
                cells.add((rr, int(c)))
    return cells


def _vertical_wall_cells(r: int, c: int, shift: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {(int(r), int(c))}
    away = -int(shift)
    if away:
        for k in range(1, random.randint(2, 3)):
            cc = int(c + away * k)
            if _in_bounds(shape, int(r), cc):
                cells.add((int(r), cc))
    return cells


def _commit_candidate(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    a: dict,
    b: dict,
    blockers: set[tuple[int, int]],
) -> bool:
    arrow_cells = set(a["cells"]) | set(b["cells"])
    if not _cells_clear(grid, reserved, arrow_cells, pad=1):
        return False
    if not _cells_clear(grid, reserved, blockers, pad=0):
        return False
    if arrow_cells & blockers:
        return False

    trial = grid.copy()
    _paint_cells(trial, blockers, GREY)
    _paint_cells(trial, arrow_cells, color)
    out = trial.copy()

    try:
        path = _paint_pair_path(trial, out, a, b, color)
    except AssertionError:
        return False

    for r, c in path - arrow_cells:
        if int(grid[r, c]) != BG or bool(reserved[r, c]):
            return False

    grid[:, :] = trial
    _reserve(reserved, path, pad=1)
    return True


def _try_horizontal_pair(grid: np.ndarray, reserved: np.ndarray, color: int, turns: int) -> bool:
    h, w = grid.shape
    if w < 18:
        return False

    if turns == 0:
        r_left = r_right = random.randint(2, h - 3)
    else:
        top = random.randint(2, h - 3 - turns)
        if random.choice([True, False]):
            r_left, r_right = top, top + turns
        else:
            r_left, r_right = top + turns, top

    left_base_c = random.randint(1, max(1, w // 3 - 2))
    min_right_base = max((2 * w) // 3, left_base_c + 12)
    if min_right_base > w - 2:
        return False
    right_base_c = random.randint(min_right_base, w - 2)

    a = {
        "base": (r_left, left_base_c),
        "direction": "right",
        "cells": frozenset(_arrow_cells((r_left, left_base_c), "right")),
        "tip": _arrow_tip((r_left, left_base_c), "right"),
    }
    b = {
        "base": (r_right, right_base_c),
        "direction": "left",
        "cells": frozenset(_arrow_cells((r_right, right_base_c), "left")),
        "tip": _arrow_tip((r_right, right_base_c), "left"),
    }

    blockers: set[tuple[int, int]] = set()
    if turns:
        start_c = int(a["tip"][1])
        end_c = int(b["tip"][1])
        mid = (start_c + end_c) // 2
        left_turns = random.randint(0, turns)
        right_turns = turns - left_turns
        left_cols = _spaced_sample(start_c + 2, mid - 2, left_turns)
        right_cols = _spaced_sample(mid + 2, end_c - 2, right_turns)
        if left_cols is None or right_cols is None:
            return False

        row = r_left
        shift = _sign(r_right - r_left)
        for c in left_cols:
            blockers.update(_horizontal_wall_cells(row, c, shift, grid.shape))
            row += shift

        row = r_right
        shift = _sign(r_left - r_right)
        for c in sorted(right_cols, reverse=True):
            blockers.update(_horizontal_wall_cells(row, c, shift, grid.shape))
            row += shift

    return _commit_candidate(grid, reserved, color, a, b, blockers)


def _try_vertical_pair(grid: np.ndarray, reserved: np.ndarray, color: int, turns: int) -> bool:
    h, w = grid.shape
    if h < 18:
        return False

    if turns == 0:
        c_top = c_bottom = random.randint(2, w - 3)
    else:
        left = random.randint(2, w - 3 - turns)
        if random.choice([True, False]):
            c_top, c_bottom = left, left + turns
        else:
            c_top, c_bottom = left + turns, left

    top_base_r = random.randint(1, max(1, h // 3 - 2))
    min_bottom_base = max((2 * h) // 3, top_base_r + 12)
    if min_bottom_base > h - 2:
        return False
    bottom_base_r = random.randint(min_bottom_base, h - 2)

    a = {
        "base": (top_base_r, c_top),
        "direction": "down",
        "cells": frozenset(_arrow_cells((top_base_r, c_top), "down")),
        "tip": _arrow_tip((top_base_r, c_top), "down"),
    }
    b = {
        "base": (bottom_base_r, c_bottom),
        "direction": "up",
        "cells": frozenset(_arrow_cells((bottom_base_r, c_bottom), "up")),
        "tip": _arrow_tip((bottom_base_r, c_bottom), "up"),
    }

    blockers: set[tuple[int, int]] = set()
    if turns:
        start_r = int(a["tip"][0])
        end_r = int(b["tip"][0])
        mid = (start_r + end_r) // 2
        top_turns = random.randint(0, turns)
        bottom_turns = turns - top_turns
        top_rows = _spaced_sample(start_r + 2, mid - 2, top_turns)
        bottom_rows = _spaced_sample(mid + 2, end_r - 2, bottom_turns)
        if top_rows is None or bottom_rows is None:
            return False

        col = c_top
        shift = _sign(c_bottom - c_top)
        for r in top_rows:
            blockers.update(_vertical_wall_cells(r, col, shift, grid.shape))
            col += shift

        col = c_bottom
        shift = _sign(c_top - c_bottom)
        for r in sorted(bottom_rows, reverse=True):
            blockers.update(_vertical_wall_cells(r, col, shift, grid.shape))
            col += shift

    return _commit_candidate(grid, reserved, color, a, b, blockers)


def _try_place_pair(grid: np.ndarray, reserved: np.ndarray, color: int, force_turn: bool | None = None) -> bool:
    for _ in range(PAIR_ATTEMPTS):
        orientation = random.choice(["h", "v"])
        if force_turn is None:
            turns = 0 if random.choice([True, False]) else random.randint(1, 4)
        else:
            turns = random.randint(1, 4) if force_turn else 0

        if orientation == "h":
            ok = _try_horizontal_pair(grid, reserved, color, turns)
        else:
            ok = _try_vertical_pair(grid, reserved, color, turns)
        if ok:
            return True
    return False


def _add_extra_grey(grid: np.ndarray, reserved: np.ndarray) -> None:
    h, w = grid.shape
    for _ in range(random.randint(1, 5)):
        cells: set[tuple[int, int]] = set()
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        cells.add((r, c))
        for _step in range(random.randint(1, 4)):
            dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nr, nc = r + dr, c + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1:
                r, c = nr, nc
                cells.add((r, c))
        if _cells_clear(grid, reserved, cells, pad=1):
            _paint_cells(grid, cells, GREY)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    pair_count = random.randint(1, 3)
    colors = random.sample(GEN_COLORS, pair_count)
    force_turn_index = random.randrange(pair_count) if random.choice([True, False]) else None

    for i, color in enumerate(colors):
        force_turn = True if i == force_turn_index else None
        if not _try_place_pair(grid, reserved, color, force_turn=force_turn):
            return None

    _add_extra_grey(grid, reserved)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid routed-arrow input")
