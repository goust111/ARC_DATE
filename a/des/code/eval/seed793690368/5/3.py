from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
GREEN = 3
GREY = 5

CW = 1
CCW = -1

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GEN_ATTEMPTS = 500
GEN_BG_CHOICES = (0, 1, 4, 6, 7, 8, 9)
GEN_SIZE_MIN = 10
GEN_SIZE_MAX = 18
GEN_SEED_MARGIN = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _cell_blocked(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    return (not _in_bounds(grid.shape, r, c)) or int(grid[r, c]) != int(bg)


def _rotate_dir(dir_idx: int, turn: int) -> int:
    return (int(dir_idx) + (1 if int(turn) == CW else -1)) % 4


def _find_side_marker(grid: np.ndarray, side: str) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    if side == "top":
        row = 0
        candidates = [
            (c, (int(grid[row, c]), int(grid[row, c + 1])))
            for c in range(w - 1)
            if {int(grid[row, c]), int(grid[row, c + 1])} == {RED, GREEN}
        ]
        assert len(candidates) == 1, "expected one top edge marker"
        c, pair = candidates[0]
        turn = CW if pair == (RED, GREEN) else CCW
        return turn, [(row, int(c)), (row, int(c + 1))]

    if side == "bottom":
        row = h - 1
        candidates = [
            (c, (int(grid[row, c]), int(grid[row, c + 1])))
            for c in range(w - 1)
            if {int(grid[row, c]), int(grid[row, c + 1])} == {RED, GREEN}
        ]
        assert len(candidates) == 1, "expected one bottom edge marker"
        c, pair = candidates[0]
        turn = CW if pair == (GREEN, RED) else CCW
        return turn, [(row, int(c)), (row, int(c + 1))]

    if side == "left":
        col = 0
        candidates = [
            (r, (int(grid[r, col]), int(grid[r + 1, col])))
            for r in range(h - 1)
            if {int(grid[r, col]), int(grid[r + 1, col])} == {RED, GREEN}
        ]
        assert len(candidates) == 1, "expected one left edge marker"
        r, pair = candidates[0]
        turn = CW if pair == (GREEN, RED) else CCW
        return turn, [(int(r), col), (int(r + 1), col)]

    if side == "right":
        col = w - 1
        candidates = [
            (r, (int(grid[r, col]), int(grid[r + 1, col])))
            for r in range(h - 1)
            if {int(grid[r, col]), int(grid[r + 1, col])} == {RED, GREEN}
        ]
        assert len(candidates) == 1, "expected one right edge marker"
        r, pair = candidates[0]
        turn = CW if pair == (RED, GREEN) else CCW
        return turn, [(int(r), col), (int(r + 1), col)]

    raise AssertionError(f"unknown side: {side}")


def _parse_markers(grid: np.ndarray) -> tuple[int, set[tuple[int, int]]]:
    turns = []
    cells: set[tuple[int, int]] = set()
    for side in ("top", "bottom", "left", "right"):
        turn, marker_cells = _find_side_marker(grid, side)
        turns.append(int(turn))
        cells.update(marker_cells)

    assert len(set(turns)) == 1, "all edge markers must encode the same turn direction"
    red_count = int(np.sum(grid == RED))
    green_count = int(np.sum(grid == GREEN))
    assert red_count == 4 and green_count == 4, "expected four red cells and four green cells"
    assert {
        tuple(map(int, xy)) for xy in np.argwhere((grid == RED) | (grid == GREEN))
    } == cells, "red-green cells should belong only to the edge markers"
    return int(turns[0]), cells


def _parse_seeds(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, int | bool]], Counter[tuple[int, int]]]:
    seed_mask = (grid != int(bg)) & (grid != RED) & (grid != GREEN) & (grid != GREY)
    seed_coords = [tuple(map(int, xy)) for xy in np.argwhere(seed_mask)]
    assert seed_coords, "expected at least one colored seed"

    seen_colors: set[int] = set()
    grey_use: Counter[tuple[int, int]] = Counter()
    states: list[dict[str, int | bool]] = []

    for r, c in sorted(seed_coords):
        color = int(grid[r, c])
        assert color not in seen_colors, "seed colors must be unique"
        seen_colors.add(color)

        neighbors = []
        for dir_idx, (dr, dc) in enumerate(DIRS):
            rr = int(r + dr)
            cc = int(c + dc)
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY:
                neighbors.append((dir_idx, (rr, cc)))
        assert len(neighbors) == 1, "each seed must have exactly one grey direction marker"

        dir_idx, grey_cell = neighbors[0]
        grey_use[grey_cell] += 1
        states.append(
            {
                "r": int(r),
                "c": int(c),
                "dir": int(dir_idx),
                "color": int(color),
                "active": True,
                "steps": 0,
                "turns": 0,
            }
        )

    return states, grey_use


def _simulate_growth(
    start_grid: np.ndarray, states: list[dict[str, int | bool]], turn: int, bg: int
) -> tuple[np.ndarray, dict[str, int | list[int]]]:
    out = start_grid.copy()
    same_target_conflicts = 0
    preblocked = 0
    rounds = 0

    while any(bool(state["active"]) for state in states):
        rounds += 1
        assert rounds <= out.size, "growth should terminate within the grid area"

        claims: dict[tuple[int, int], list[int]] = {}
        for idx, state in enumerate(states):
            if not bool(state["active"]):
                continue
            dr, dc = DIRS[int(state["dir"])]
            nr = int(state["r"]) + dr
            nc = int(state["c"]) + dc
            if _cell_blocked(out, nr, nc, bg):
                preblocked += 1
                continue
            claims.setdefault((nr, nc), []).append(idx)

        moved: set[int] = set()
        for target, claimers in claims.items():
            claimers = sorted(claimers)
            winner = int(claimers[0])
            if len(claimers) > 1:
                same_target_conflicts += len(claimers) - 1
            state = states[winner]
            out[target] = int(state["color"])
            state["r"], state["c"] = int(target[0]), int(target[1])
            state["steps"] = int(state["steps"]) + 1
            moved.add(winner)

        for idx, state in enumerate(states):
            if not bool(state["active"]):
                continue
            dr, dc = DIRS[int(state["dir"])]
            nr = int(state["r"]) + dr
            nc = int(state["c"]) + dc
            if not _cell_blocked(out, nr, nc, bg):
                continue

            new_dir = _rotate_dir(int(state["dir"]), turn)
            dr2, dc2 = DIRS[new_dir]
            nr2 = int(state["r"]) + dr2
            nc2 = int(state["c"]) + dc2
            if _cell_blocked(out, nr2, nc2, bg):
                state["active"] = False
            else:
                state["dir"] = int(new_dir)
                state["turns"] = int(state["turns"]) + 1

        if not moved and not any(bool(state["active"]) for state in states):
            break

    info = {
        "rounds": int(rounds),
        "same_target_conflicts": int(same_target_conflicts),
        "preblocked": int(preblocked),
        "steps": [int(state["steps"]) for state in states],
        "turns": [int(state["turns"]) for state in states],
        "seed_count": len(states),
        "total_added": int(sum(int(state["steps"]) for state in states)),
    }
    return out, info


def _solve(grid: np.ndarray, *, return_info: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg not in {RED, GREEN, GREY}, "background must differ from marker colors"

    turn, marker_cells = _parse_markers(g)
    states, grey_use = _parse_seeds(g, bg)
    grey_cells = [tuple(map(int, xy)) for xy in np.argwhere(g == GREY)]
    assert grey_cells, "expected at least one grey direction marker"
    assert set(grey_cells) == set(grey_use), "every grey pixel must belong to one seed"
    assert all(count == 1 for count in grey_use.values()), "grey markers must be unique to one seed"

    cleaned = g.copy()
    cleaned[g == GREY] = int(bg)
    for r, c in marker_cells:
        cleaned[int(r), int(c)] = int(bg)
    assert not bool(np.any((cleaned == RED) | (cleaned == GREEN) | (cleaned == GREY)))

    out, info = _simulate_growth(cleaned, states, turn, bg)
    if return_info:
        return out, info
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all colored seeds synchronously, using grey neighbors as initial directions and
    red-green edge pairs as the shared turn direction.

    Input structure:
    - The most common color is the background.
    - Each grid edge contains one red-green adjacent marker. Reading red->green in the
      clockwise border direction means clockwise turning; the opposite order means
      counterclockwise. All four markers agree.
    - Each interior seed is a unique non-marker color with exactly one 4-neighbor grey
      cell that encodes its initial heading.

    Rule:
    - Remove the grey direction cells and all edge markers.
    - Every active seed advances one cell per round in its current heading, painting
      background cells with its own color.
    - After moving, if the next cell in the same heading is blocked, rotate 90 degrees
      using the shared turn direction. If that rotated next cell is also blocked, the
      seed stops.
    - Earlier claims keep their cells; same-round claim ties are resolved by seed
      reading order, which the generator avoids.
    """
    return _solve(grid)


def _paint_side_marker(grid: np.ndarray, side: str, offset: int, turn: int) -> None:
    h, w = grid.shape
    if side == "top":
        if turn == CW:
            grid[0, offset], grid[0, offset + 1] = RED, GREEN
        else:
            grid[0, offset], grid[0, offset + 1] = GREEN, RED
        return

    if side == "bottom":
        if turn == CW:
            grid[h - 1, offset], grid[h - 1, offset + 1] = GREEN, RED
        else:
            grid[h - 1, offset], grid[h - 1, offset + 1] = RED, GREEN
        return

    if side == "left":
        if turn == CW:
            grid[offset, 0], grid[offset + 1, 0] = GREEN, RED
        else:
            grid[offset, 0], grid[offset + 1, 0] = RED, GREEN
        return

    if side == "right":
        if turn == CW:
            grid[offset, w - 1], grid[offset + 1, w - 1] = RED, GREEN
        else:
            grid[offset, w - 1], grid[offset + 1, w - 1] = GREEN, RED
        return

    raise AssertionError(f"unknown side: {side}")


def _can_place_cells(grid: np.ndarray, cells: list[tuple[int, int]], bg: int, margin: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != int(bg):
            return False

    for r, c in cells:
        r0 = max(0, int(r - margin))
        r1 = min(h, int(r + margin + 1))
        c0 = max(0, int(c - margin))
        c1 = min(w, int(c + margin + 1))
        if np.any(grid[r0:r1, c0:c1] != int(bg)):
            return False
    return True


def _seed_count_for_shape(h: int, w: int) -> int:
    max_seeds = min(4, max(2, (h * w) // 45))
    return random.randint(2, int(max_seeds))


def _place_random_seeds(grid: np.ndarray, bg: int, colors: list[int], n_seeds: int) -> bool:
    h, w = grid.shape
    candidates = [
        (r, c, dir_idx)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        for dir_idx in range(4)
    ]
    random.shuffle(candidates)

    placed = 0
    for color in colors[:n_seeds]:
        found = False
        for r, c, dir_idx in candidates:
            dr, dc = DIRS[dir_idx]
            grey_cell = (int(r + dr), int(c + dc))
            if not _can_place_cells(grid, [(r, c), grey_cell], bg, GEN_SEED_MARGIN):
                continue
            grid[r, c] = int(color)
            grid[grey_cell] = GREY
            placed += 1
            found = True
            break
        if not found:
            return False
    return placed == n_seeds


def _interesting(info: dict[str, int | list[int]]) -> bool:
    steps = [int(v) for v in info["steps"]]
    turns = [int(v) for v in info["turns"]]
    seed_count = int(info["seed_count"])
    total_added = int(info["total_added"])
    if int(info["same_target_conflicts"]) != 0 or int(info["preblocked"]) != 0:
        return False
    if total_added < max(8, 3 * seed_count):
        return False
    if max(turns) == 0:
        return False
    if sum(step >= 2 for step in steps) < min(2, seed_count):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    bg = random.choice(GEN_BG_CHOICES)
    turn = random.choice([CW, CCW])

    grid = np.full((h, w), int(bg), dtype=int)
    _paint_side_marker(grid, "top", random.randint(1, w - 3), turn)
    _paint_side_marker(grid, "bottom", random.randint(1, w - 3), turn)
    _paint_side_marker(grid, "left", random.randint(1, h - 3), turn)
    _paint_side_marker(grid, "right", random.randint(1, h - 3), turn)

    seed_palette = [color for color in GEN_BG_CHOICES if color != int(bg)]
    random.shuffle(seed_palette)
    n_seeds = _seed_count_for_shape(h, w)
    if len(seed_palette) < n_seeds:
        return None
    if not _place_random_seeds(grid, bg, seed_palette, n_seeds):
        return None

    try:
        out, info = _solve(grid, return_info=True)
    except AssertionError:
        return None

    if not _interesting(info):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-growth ARC input")
