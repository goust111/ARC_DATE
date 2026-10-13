import math
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5


DIR_PATTERNS = (
    ((0, 1), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (0, 1), (0, 1)),
    ((0, 1), (0, 1), (1, 0), (0, 1)),
    ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
)

MAX_GEN_ATTEMPTS = 1200


def _rotate_cw(vec: tuple[int, int], turns: int = 1) -> tuple[int, int]:
    dr, dc = vec
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _rotate_ccw(vec: tuple[int, int], turns: int = 1) -> tuple[int, int]:
    dr, dc = vec
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _gcd_step(values: Iterable[int]) -> int:
    vals = sorted(set(int(v) for v in values))
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    step = 0
    for diff in diffs:
        step = math.gcd(step, int(diff))
    assert step > 0
    return int(step)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_border(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _primitive_period(seq: list[tuple[int, int]]) -> int:
    n = len(seq)
    for k in range(1, n + 1):
        if n < 2 * k:
            continue
        if all(seq[i] == seq[i % k] for i in range(n)):
            return k
    raise AssertionError("failed to infer a repeated period")


def _path_from_holes(hole_set: set[tuple[int, int]], lattice_h: int, lattice_w: int) -> list[tuple[int, int]]:
    assert hole_set
    degrees = {}
    for cell in hole_set:
        degrees[cell] = sum(nei in hole_set for nei in _neighbors4(cell))
    assert all(deg in (1, 2) for deg in degrees.values())
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2
    border_endpoints = [cell for cell in endpoints if _is_border(cell, lattice_h, lattice_w)]
    assert len(border_endpoints) == 1
    start = border_endpoints[0]

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [nei for nei in _neighbors4(cur) if nei in hole_set and nei != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(hole_set)
    assert not _is_border(ordered[-1], lattice_h, lattice_w)
    return ordered


def _infer_blue_set(
    visible_blues: list[tuple[int, int]], lattice_h: int, lattice_w: int
) -> tuple[str, int, set[tuple[int, int]]]:
    assert len(visible_blues) >= 3
    diff_vals = {r - c for r, c in visible_blues}
    sum_vals = {r + c for r, c in visible_blues}
    if len(diff_vals) == 1:
        key = int(next(iter(diff_vals)))
        blue_set = {
            (r, c)
            for r in range(lattice_h)
            for c in range(lattice_w)
            if r - c == key
        }
        return "diff", key, blue_set
    if len(sum_vals) == 1:
        key = int(next(iter(sum_vals)))
        blue_set = {
            (r, c)
            for r in range(lattice_h)
            for c in range(lattice_w)
            if r + c == key
        }
        return "sum", key, blue_set
    raise AssertionError("blue dots must form one lattice diagonal")


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, BLUE, GREY}

    occupied = np.argwhere(g != BG)
    assert occupied.size > 0
    row_coords = sorted(set(int(r) for r in occupied[:, 0]))
    col_coords = sorted(set(int(c) for c in occupied[:, 1]))
    row_step = _gcd_step(row_coords)
    col_step = _gcd_step(col_coords)
    r0, r1 = min(row_coords), max(row_coords)
    c0, c1 = min(col_coords), max(col_coords)
    lattice_rows = list(range(r0, r1 + 1, row_step))
    lattice_cols = list(range(c0, c1 + 1, col_step))
    assert len(lattice_rows) >= 6 and len(lattice_cols) >= 6

    row_to_idx = {r: i for i, r in enumerate(lattice_rows)}
    col_to_idx = {c: i for i, c in enumerate(lattice_cols)}
    for r, c in occupied:
        assert int(r) in row_to_idx and int(c) in col_to_idx

    lattice_h = len(lattice_rows)
    lattice_w = len(lattice_cols)
    hole_set = set()
    visible_blues = []
    lattice_values = {}
    for ri, r in enumerate(lattice_rows):
        for ci, c in enumerate(lattice_cols):
            value = int(g[r, c])
            lattice_values[(ri, ci)] = value
            if value == BG:
                hole_set.add((ri, ci))
            elif value == BLUE:
                visible_blues.append((ri, ci))
            else:
                assert value == GREY

    path = _path_from_holes(hole_set, lattice_h, lattice_w)
    blue_kind, blue_key, blue_set = _infer_blue_set(visible_blues, lattice_h, lattice_w)
    blue_flags = [cell in blue_set for cell in path]
    assert sum(blue_flags) >= 1

    dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    assert all(abs(dr) + abs(dc) == 1 for dr, dc in dirs)

    normalized = []
    turns = 0
    for i, step in enumerate(dirs):
        if blue_flags[i]:
            turns += 1
        normalized.append(_rotate_ccw(step, turns))

    period = _primitive_period(normalized)
    pattern = normalized[:period]
    assert all(normalized[i] == pattern[i % period] for i in range(len(normalized)))

    return {
        "grid": g,
        "lattice_rows": lattice_rows,
        "lattice_cols": lattice_cols,
        "lattice_h": lattice_h,
        "lattice_w": lattice_w,
        "path": path,
        "blue_set": blue_set,
        "blue_kind": blue_kind,
        "blue_key": blue_key,
        "blue_flags": blue_flags,
        "pattern": pattern,
        "period": period,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the eaten-out lattice trail by continuing its periodic gait and mark the future bites in red.

    Input structure:
    - A black background contains an equally spaced lattice of single grey dots, plus a sparse diagonal of blue dots on the same lattice.
    - A worm is not drawn directly; instead, already eaten lattice sites have been removed to background, forming one lattice-adjacent negative-space trail that starts on the border and stops inside the field.

    Rule:
    - Recover the ordered trail on the lattice, infer the repeating unit-step gait from the shown bites, and continue that gait from the trail tip.
    - Whenever the worm lands on a site belonging to the blue-dot diagonal, the remaining gait rotates clockwise by ninety degrees from that point onward.
    - Keep all existing dots and old holes unchanged, and color every future eaten lattice site red until the next step would leave the grid.

    Key insight:
    - The missing dots themselves encode the worm's path and cycle, while the surviving blue diagonal reveals which missing turn points must also be treated as blue triggers.
    """
    info = _parse_grid(np.asarray(grid, dtype=int))
    out = info["grid"].copy()

    turns = sum(info["blue_flags"])
    step_index = (len(info["path"]) - 1) % info["period"]
    cur = info["path"][-1]

    future = []
    while True:
        move = _rotate_cw(info["pattern"][step_index], turns)
        nxt = (cur[0] + move[0], cur[1] + move[1])
        if not (0 <= nxt[0] < info["lattice_h"] and 0 <= nxt[1] < info["lattice_w"]):
            break
        future.append(nxt)
        cur = nxt
        if cur in info["blue_set"]:
            turns += 1
        step_index = (step_index + 1) % info["period"]

    for r, c in future:
        out[info["lattice_rows"][r], info["lattice_cols"][c]] = RED
    return out


def _diagonal_cells(kind: str, key: int, lattice_h: int, lattice_w: int) -> set[tuple[int, int]]:
    if kind == "diff":
        return {
            (r, c)
            for r in range(lattice_h)
            for c in range(lattice_w)
            if r - c == key
        }
    return {
        (r, c)
        for r in range(lattice_h)
        for c in range(lattice_w)
        if r + c == key
    }


def _diagonal_candidates(kind: str, lattice_h: int, lattice_w: int) -> list[int]:
    candidates = []
    if kind == "diff":
        for key in range(-(lattice_w - 1), lattice_h):
            if len(_diagonal_cells(kind, key, lattice_h, lattice_w)) >= 5:
                candidates.append(key)
    else:
        for key in range(lattice_h + lattice_w - 1):
            if len(_diagonal_cells(kind, key, lattice_h, lattice_w)) >= 5:
                candidates.append(key)
    return candidates


def _start_positions(first_step: tuple[int, int], lattice_h: int, lattice_w: int) -> list[tuple[int, int]]:
    dr, dc = first_step
    if (dr, dc) == (0, 1):
        return [(r, 0) for r in range(lattice_h)]
    if (dr, dc) == (0, -1):
        return [(r, lattice_w - 1) for r in range(lattice_h)]
    if (dr, dc) == (1, 0):
        return [(0, c) for c in range(lattice_w)]
    if (dr, dc) == (-1, 0):
        return [(lattice_h - 1, c) for c in range(lattice_w)]
    raise AssertionError(first_step)


def _simulate_path(
    start: tuple[int, int],
    pattern: list[tuple[int, int]],
    blue_set: set[tuple[int, int]],
    lattice_h: int,
    lattice_w: int,
) -> list[tuple[int, int]] | None:
    positions = [start]
    seen = {start}
    turns = 1 if start in blue_set else 0
    step_index = 0
    while True:
        move = _rotate_cw(pattern[step_index], turns)
        nxt = (positions[-1][0] + move[0], positions[-1][1] + move[1])
        if not (0 <= nxt[0] < lattice_h and 0 <= nxt[1] < lattice_w):
            return positions
        if nxt in seen:
            return None
        positions.append(nxt)
        seen.add(nxt)
        if nxt in blue_set:
            turns += 1
        step_index = (step_index + 1) % len(pattern)
        if len(positions) > lattice_h * lattice_w:
            return None


def _render_input(
    lattice_h: int,
    lattice_w: int,
    spacing: int,
    top: int,
    left: int,
    blue_set: set[tuple[int, int]],
    observed_path: list[tuple[int, int]],
) -> np.ndarray:
    height = top + 1 + spacing * (lattice_h - 1) + random.randint(1, 3)
    width = left + 1 + spacing * (lattice_w - 1) + random.randint(1, 3)
    grid = np.zeros((height, width), dtype=int)

    for r in range(lattice_h):
        for c in range(lattice_w):
            pr = top + r * spacing
            pc = left + c * spacing
            grid[pr, pc] = BLUE if (r, c) in blue_set else GREY

    for r, c in observed_path:
        pr = top + r * spacing
        pc = left + c * spacing
        grid[pr, pc] = BG

    return grid


def _future_from_path(
    full_path: list[tuple[int, int]], observed_len: int
) -> list[tuple[int, int]]:
    return full_path[observed_len:]


def _choose_observation_cut(
    full_path: list[tuple[int, int]],
    blue_set: set[tuple[int, int]],
    lattice_h: int,
    lattice_w: int,
    period: int,
    want_future_turn: bool,
) -> int | None:
    blue_indices = [i for i, cell in enumerate(full_path) if cell in blue_set]
    if not blue_indices:
        return None
    first_blue = blue_indices[0]
    if first_blue < 2 * period:
        return None

    cuts = []
    for cut in range(first_blue + 2, len(full_path) - 1):
        observed = full_path[: cut + 1]
        if _is_border(observed[-1], lattice_h, lattice_w):
            continue
        if len(observed) - 1 < 2 * period:
            continue
        observed_blue = sum(cell in blue_set for cell in observed)
        if observed_blue != 1:
            continue
        future = full_path[cut + 1 :]
        if len(future) < 2:
            continue
        future_turns = sum(cell in blue_set for cell in future)
        if want_future_turn and future_turns == 0:
            continue
        if not want_future_turn and future_turns != 0:
            continue
        cuts.append(cut)
    if not cuts:
        return None
    return random.choice(cuts)


def _check_semantics(
    grid: np.ndarray,
    expected_observed: list[tuple[int, int]],
    expected_blue_set: set[tuple[int, int]],
    expected_pattern: list[tuple[int, int]],
    expected_future: list[tuple[int, int]],
) -> bool:
    try:
        parsed = _parse_grid(grid)
    except AssertionError:
        return False
    if parsed["path"] != expected_observed:
        return False
    if parsed["blue_set"] != expected_blue_set:
        return False
    if parsed["pattern"] != expected_pattern:
        return False

    out = main(grid)
    actual_red = {
        (ri, ci)
        for ri, r in enumerate(parsed["lattice_rows"])
        for ci, c in enumerate(parsed["lattice_cols"])
        if int(out[r, c]) == RED
    }
    if actual_red != set(expected_future):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _generate_candidate(want_future_turn: bool) -> np.ndarray | None:
    lattice_h = random.randint(7, 10)
    lattice_w = random.randint(7, 10)
    base_pattern = list(random.choice(DIR_PATTERNS))
    global_rot = random.randrange(4)
    pattern = [_rotate_cw(step, global_rot) for step in base_pattern]

    blue_kind = random.choice(("diff", "sum"))
    blue_key = random.choice(_diagonal_candidates(blue_kind, lattice_h, lattice_w))
    blue_set = _diagonal_cells(blue_kind, blue_key, lattice_h, lattice_w)

    starts = _start_positions(pattern[0], lattice_h, lattice_w)
    random.shuffle(starts)
    starts = [cell for cell in starts if cell not in blue_set]
    for start in starts:
        full_path = _simulate_path(start, pattern, blue_set, lattice_h, lattice_w)
        if full_path is None or len(full_path) < 2 * len(pattern) + 4:
            continue
        cut = _choose_observation_cut(
            full_path, blue_set, lattice_h, lattice_w, len(pattern), want_future_turn
        )
        if cut is None:
            continue

        observed = full_path[: cut + 1]
        future = _future_from_path(full_path, len(observed))
        if len(blue_set - set(observed)) < 3:
            continue

        spacing = random.choice((2, 3))
        max_top = 30 - (spacing * (lattice_h - 1) + 2)
        max_left = 30 - (spacing * (lattice_w - 1) + 2)
        if max_top < 1 or max_left < 1:
            continue
        top = random.randint(1, max_top)
        left = random.randint(1, max_left)
        grid = _render_input(lattice_h, lattice_w, spacing, top, left, blue_set, observed)
        if max(grid.shape) > 30:
            continue
        if not _check_semantics(grid, observed, blue_set, pattern, future):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        want_future_turn = random.random() < 0.55
        grid = _generate_candidate(want_future_turn=want_future_turn)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 蛀纹延拓 instance")
