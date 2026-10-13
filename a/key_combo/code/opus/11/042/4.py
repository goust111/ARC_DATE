from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MOVER = 1
BRACKET = 2
PATH = 3
BLOCKER = 8

MOVE_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SPRITES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)

BRACKET_SHAPES = {
    "down": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3)],
    "up": [(0, 0), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
    "right": [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (3, 1)],
    "left": [(0, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
}

GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                _in_bounds(mask.shape, nr, nc)
                and mask[nr, nc]
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _green_start_pixels(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == PATH)]
    assert len(coords) == 2, "expected a 2-pixel green start bar"
    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    h, w = grid.shape

    def is_border(rc: tuple[int, int]) -> bool:
        r, c = rc
        return r == 0 or c == 0 or r == h - 1 or c == w - 1

    border = [rc for rc in coords if is_border(rc)]
    assert len(border) == 1, "exactly one start-bar pixel must touch the border"
    border_rc = border[0]
    inner_rc = coords[1] if coords[0] == border_rc else coords[0]
    return border_rc, inner_rc


def _trace_corridor(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    border_rc, inner_rc = _green_start_pixels(g)
    dr = inner_rc[0] - border_rc[0]
    dc = inner_rc[1] - border_rc[1]
    assert abs(dr) + abs(dc) == 1

    r, c = inner_rc
    length = 2
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "corridor path must not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        cell = int(g[nr, nc])
        if cell == BLOCKER:
            options: list[tuple[int, int]] = []
            for tdr, tdc in ((-dc, dr), (dc, -dr)):
                tr, tc = r + tdr, c + tdc
                if not _in_bounds(g.shape, tr, tc):
                    continue
                turn_cell = int(g[tr, tc])
                if turn_cell in (BG, PATH):
                    options.append((tdr, tdc))
            assert len(options) == 1, "each blocker collision must force one turn"
            dr, dc = options[0]
            continue

        assert cell == BG, "path may only traverse background cells"
        out[nr, nc] = PATH
        length += 1
        r, c = nr, nc

    return out, length


def _detect_bracket_direction(grid: np.ndarray) -> str:
    coords = np.argwhere(grid == BRACKET)
    assert len(coords) == 6, "expected a single 6-cell red bracket"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if (h, w) == (2, 4):
        row_counts = mask.sum(axis=1).tolist()
        if row_counts == [4, 2] and mask[1, 0] and mask[1, 3]:
            return "down"
        if row_counts == [2, 4] and mask[0, 0] and mask[0, 3]:
            return "up"
    if (h, w) == (4, 2):
        col_counts = mask.sum(axis=0).tolist()
        if col_counts == [4, 2] and mask[0, 1] and mask[3, 1]:
            return "right"
        if col_counts == [2, 4] and mask[0, 0] and mask[3, 0]:
            return "left"
    raise AssertionError("unrecognized red bracket orientation")


def _move_object(base_grid: np.ndarray, direction: str, distance: int) -> np.ndarray:
    g = np.asarray(base_grid, dtype=int)
    out = g.copy()
    mover = np.argwhere(g == MOVER)
    assert 3 <= len(mover) <= 4, "expected one 3-4 cell blue object"
    assert _is_connected(g == MOVER), "blue object must be 4-connected"

    out[mover[:, 0], mover[:, 1]] = BG
    dr, dc = MOVE_VECS[direction]
    dest = mover + np.array([[dr * distance, dc * distance]], dtype=int)

    for rr, cc in dest:
        assert _in_bounds(g.shape, int(rr), int(cc)), "translated object left the grid"
        assert out[int(rr), int(cc)] == BG, "translated object must land on background"

    out[dest[:, 0], dest[:, 1]] = MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the blocker corridor to measure a path length, then shift the bracket-lane object by that amount.

    Input structure:
    - The background is black.
    - A 2-pixel green start bar touches the left border, and cyan blocker pixels define a one-cell-wide corridor in the left area.
    - A single red bracket component in the right area opens in one cardinal direction, and one blue 4-connected object sits in that bracket's lane.

    Rule:
    - Start from the border pixel of the green bar, pass through the other green pixel, and keep walking one cell at a time.
    - If the next cell is a blocker, turn to the unique free perpendicular direction; if the next step would leave the grid, stop. Paint every traversed background cell green and count the full path length including the two start-bar pixels.
    - Detect the red bracket opening direction, then translate the blue object by the counted path length along that opening direction. Clear the blue object's old position and keep the bracket fixed.

    Key insight:
    - The corridor contributes only a number, while the bracket contributes only an axis and sign; both abstract properties are required to determine the final blue-object placement.
    """
    traced, length = _trace_corridor(grid)
    direction = _detect_bracket_direction(traced)
    return _move_object(traced, direction, length)


def _place_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    for r, c in coords:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BG:
            return False
    for r, c in coords:
        grid[r, c] = color
    return True


def _sprite_cells(top: int, left: int, sprite: np.ndarray) -> list[tuple[int, int]]:
    return [
        (top + r, left + c)
        for r in range(sprite.shape[0])
        for c in range(sprite.shape[1])
        if sprite[r, c]
    ]


def _build_corridor_design(
    h: int, w: int, total_length: int, two_turn: bool
) -> dict[str, object] | None:
    extra = total_length - 2
    left_limit = max(4, w // 2)
    choices: list[dict[str, object]] = []

    if not two_turn:
        for a in range(extra + 1):
            b = extra - a
            if b < 1 or 2 + a >= left_limit:
                continue
            for vertical in ("up", "down"):
                start_row = b if vertical == "up" else h - 1 - b
                if not (1 <= start_row <= h - 2):
                    continue
                turn_col = 1 + a
                path = [(start_row, 1 + step) for step in range(1, a + 1)]
                if vertical == "up":
                    path += [(start_row - step, turn_col) for step in range(1, b + 1)]
                    wrong = (start_row + 1, turn_col)
                else:
                    path += [(start_row + step, turn_col) for step in range(1, b + 1)]
                    wrong = (start_row - 1, turn_col)
                blockers = [(start_row, turn_col + 1), wrong]
                if wrong in path:
                    continue
                choices.append(
                    {
                        "start_row": start_row,
                        "path": path,
                        "blockers": blockers,
                    }
                )
    else:
        for a in range(extra + 1):
            b = extra - 1 - 2 * a
            if b < 1 or 2 + a >= left_limit:
                continue
            for vertical in ("up", "down"):
                if vertical == "up":
                    row_min, row_max = b + 1, h - 2
                else:
                    row_min, row_max = 1, h - b - 2
                if row_min > row_max:
                    continue
                for start_row in range(row_min, row_max + 1):
                    turn_col = 1 + a
                    path = [(start_row, 1 + step) for step in range(1, a + 1)]
                    if vertical == "up":
                        path += [(start_row - step, turn_col) for step in range(1, b + 1)]
                        end_row = start_row - b
                        wrong1 = (start_row + 1, turn_col)
                        ahead2 = (end_row - 1, turn_col)
                    else:
                        path += [(start_row + step, turn_col) for step in range(1, b + 1)]
                        end_row = start_row + b
                        wrong1 = (start_row - 1, turn_col)
                        ahead2 = (end_row + 1, turn_col)
                    path += [(end_row, turn_col - step) for step in range(1, turn_col + 1)]
                    blockers = [(start_row, turn_col + 1), wrong1, ahead2, (end_row, turn_col + 1)]
                    if len(set(path)) != len(path):
                        continue
                    if wrong1 in path or ahead2 in path or (end_row, turn_col + 1) in path:
                        continue
                    choices.append(
                        {
                            "start_row": start_row,
                            "path": path,
                            "blockers": blockers,
                        }
                    )

    if not choices:
        return None
    return random.choice(choices)


def _add_distractor_blockers(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    left_limit: int,
) -> None:
    h, _w = grid.shape
    for _ in range(random.randint(0, 2)):
        candidates = []
        for r in range(h):
            for c in range(1, left_limit):
                if grid[r, c] != BG or (r, c) in occupied:
                    continue
                if any(
                    (r + dr, c + dc) in occupied
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ):
                    continue
                candidates.append((r, c))
        if not candidates:
            return
        cell = random.choice(candidates)
        grid[cell] = BLOCKER
        occupied.add(cell)


def _place_corridor(grid: np.ndarray, total_length: int) -> dict[str, object] | None:
    h, w = grid.shape
    allow_two_turn = total_length >= 4 and random.random() < 0.45
    for two_turn in ([allow_two_turn, False] if allow_two_turn else [False, True]):
        design = _build_corridor_design(h, w, total_length, two_turn)
        if design is None:
            continue

        start_row = int(design["start_row"])
        path = list(design["path"])
        blockers = list(design["blockers"])

        trial = grid.copy()
        trial[start_row, 0] = PATH
        trial[start_row, 1] = PATH
        if not _place_cells(trial, blockers, BLOCKER):
            continue

        occupied = {(start_row, 0), (start_row, 1), *path, *blockers}
        _add_distractor_blockers(trial, occupied, max(4, w // 2))

        traced, measured = _trace_corridor(trial)
        expected = trial.copy()
        for r, c in path:
            expected[r, c] = PATH
        if measured != total_length or not np.array_equal(traced, expected):
            continue
        grid[:, :] = trial
        return {"path": path, "blockers": blockers, "start_row": start_row, "two_turn": two_turn}
    return None


def _object_anchor(direction: str, r0: int, c0: int) -> tuple[int, int]:
    if direction == "down":
        return r0 + 1, c0 + 1
    if direction == "up":
        return r0 - 1, c0 + 1
    if direction == "right":
        return r0 + 1, c0 + 1
    return r0 + 1, c0 - 1


def _place_bracket_and_object(
    grid: np.ndarray, direction: str, distance: int, sprite: np.ndarray
) -> dict[str, object] | None:
    h, w = grid.shape
    mid = w // 2
    bracket = BRACKET_SHAPES[direction]
    sprite_h, sprite_w = sprite.shape

    rows: list[int]
    cols: list[int]
    if direction == "down":
        rows = list(range(1, h - distance - sprite_h - 1))
        cols = list(range(max(mid - 1, 1), w - 4))
    elif direction == "up":
        rows = list(range(distance + 1, h - 2))
        cols = list(range(max(mid - 1, 1), w - 4))
    elif direction == "right":
        rows = list(range(1, h - 4))
        cols = list(range(max(mid - 2, 1), w - distance - sprite_w - 1))
    else:
        rows = list(range(1, h - 4))
        cols = list(range(max(mid + distance + 1, distance + 2), w - 1))

    if not rows or not cols:
        return None

    random.shuffle(rows)
    random.shuffle(cols)

    for r0 in rows:
        for c0 in cols:
            bracket_cells = [(r0 + dr, c0 + dc) for dr, dc in bracket]
            top, left = _object_anchor(direction, r0, c0)
            mover_cells = _sprite_cells(top, left, sprite)
            if not mover_cells or len(mover_cells) < 3 or not _is_connected(sprite):
                return None
            dr, dc = MOVE_VECS[direction]
            moved_cells = [(r + dr * distance, c + dc * distance) for r, c in mover_cells]
            all_cells = bracket_cells + mover_cells + moved_cells
            if not all(_in_bounds(grid.shape, r, c) for r, c in all_cells):
                continue
            if direction == "left" and min(c for _r, c in moved_cells) < max(mid - 1, 4):
                continue
            if len(set(bracket_cells)) != len(bracket_cells):
                continue
            if set(bracket_cells) & set(mover_cells):
                continue
            if set(bracket_cells) & set(moved_cells):
                continue
            if any(grid[r, c] != BG for r, c in bracket_cells + mover_cells):
                continue
            grid2 = grid.copy()
            if not _place_cells(grid2, bracket_cells, BRACKET):
                continue
            if not _place_cells(grid2, mover_cells, MOVER):
                continue
            return {
                "grid": grid2,
                "direction": direction,
                "mover_cells": mover_cells,
                "bracket_cells": bracket_cells,
            }
    return None


def _alt_output(base_after_trace: np.ndarray, direction: str, distance: int) -> np.ndarray | None:
    try:
        return _move_object(base_after_trace, direction, distance)
    except AssertionError:
        return None


def _nondegenerate(inp: np.ndarray, direction: str, distance: int) -> bool:
    traced, measured = _trace_corridor(inp)
    if measured != distance:
        return False
    expected = _move_object(traced, direction, distance)

    for alt_dir in MOVE_VECS:
        if alt_dir == direction:
            continue
        candidate = _alt_output(traced, alt_dir, distance)
        if candidate is not None and np.array_equal(candidate, expected):
            return False

    for alt_dist in range(3, 7):
        if alt_dist == distance:
            continue
        candidate = _alt_output(traced, direction, alt_dist)
        if candidate is not None and np.array_equal(candidate, expected):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(15, 19)
    grid = np.full((h, w), BG, dtype=int)

    total_length = random.choices([3, 4, 5, 6], weights=[1, 2, 2, 3])[0]
    corridor_meta = _place_corridor(grid, total_length)
    if corridor_meta is None:
        return None

    sprite = random.choice(SPRITES)
    placements = []
    for direction in ("up", "down", "left", "right"):
        placement = _place_bracket_and_object(grid, direction, total_length, sprite)
        if placement is not None:
            placements.append(placement)
    if not placements:
        return None
    placement = random.choice(placements)
    direction = str(placement["direction"])

    inp = placement["grid"]
    traced, measured = _trace_corridor(inp)
    if measured != total_length:
        return None
    if _detect_bracket_direction(inp) != direction:
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if Counter(map(int, inp.flatten()))[PATH] != 2:
        return None
    if not _nondegenerate(inp, direction, total_length):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
