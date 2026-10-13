from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

DIR_BY_COLOR = {
    GREEN: (0, -1),   # left
    TEAL: (0, 1),     # right
    PINK: (-1, 0),    # up
    YELLOW: (1, 0),   # down
}

DIR_NAMES = ("U", "R", "D", "L")
DIR_STEP = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
DIR_COLOR = {
    "U": PINK,
    "R": TEAL,
    "D": YELLOW,
    "L": GREEN,
}

GEN_RETRY_LIMIT = 300


def _in_activity(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 1 <= int(r) < height and 0 <= int(c) < width


def _parse_tape(grid: np.ndarray) -> list[int]:
    top = np.asarray(grid[0], dtype=int)
    length = 0
    while length < top.size and int(top[length]) != BLACK:
        color = int(top[length])
        assert color in DIR_BY_COLOR, "tape colors must encode directions"
        length += 1
    assert length > 0, "expected a nonempty instruction tape"
    assert np.all(top[length:] == BLACK), "instruction tape must be contiguous from the left"
    return [int(color) for color in top[:length]]


def _trace_run(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[dict[str, int | tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must have a tape row and an activity area"

    tape = _parse_tape(g)
    allowed = {BLACK, RED, GREY, GREEN, YELLOW, PINK, TEAL}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors in grid"
    assert not np.any(np.isin(g[1:], [GREEN, YELLOW, PINK, TEAL])), "direction colors must stay on the tape row"
    assert not np.any(g[0] == RED), "start must be in the activity area"
    assert not np.any(g[0] == GREY), "obstacles must be in the activity area"

    start_coords = np.argwhere(g[1:] == RED)
    assert start_coords.shape[0] == 1, "expected exactly one red start pixel"
    start_r = int(start_coords[0, 0] + 1)
    start_c = int(start_coords[0, 1])
    assert _in_activity(g.shape, start_r, start_c)

    assert np.all((g[1:] == BLACK) | (g[1:] == RED) | (g[1:] == GREY)), "activity area must contain only black, red, and grey"

    trail: set[tuple[int, int]] = {(start_r, start_c)}
    current = (start_r, start_c)
    metadata: list[dict[str, int | tuple[int, int]]] = []

    for color in tape:
        dr, dc = DIR_BY_COLOR[int(color)]
        moved = 0
        stop_reason = -1
        start_pos = current

        while True:
            nr = int(current[0] + dr)
            nc = int(current[1] + dc)
            if not _in_activity(g.shape, nr, nc):
                stop_reason = 0  # boundary
                break
            if int(g[nr, nc]) == GREY:
                stop_reason = 1  # obstacle
                break
            if (nr, nc) in trail:
                stop_reason = 2  # trail
                break
            current = (nr, nc)
            trail.add(current)
            moved += 1

        metadata.append(
            {
                "color": int(color),
                "start_r": int(start_pos[0]),
                "start_c": int(start_pos[1]),
                "end_r": int(current[0]),
                "end_c": int(current[1]),
                "moved": int(moved),
                "stop_reason": int(stop_reason),
            }
        )

    return trail, metadata


def _opposite(name: str) -> str:
    return {"U": "D", "D": "U", "L": "R", "R": "L"}[name]


def _turn(name: str, delta: int) -> str:
    idx = DIR_NAMES.index(name)
    return DIR_NAMES[(idx + int(delta)) % 4]


def _path_cells(start: tuple[int, int], dr: int, dc: int, length: int) -> list[tuple[int, int]]:
    return [(int(start[0] + dr * step), int(start[1] + dc * step)) for step in range(1, int(length) + 1)]


def _make_core_layout(
    rng: np.random.Generator,
) -> tuple[int, int, list[str], tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    height = int(rng.integers(9, 21))
    width = int(rng.integers(10, 21))
    if width < 8:
        return None

    d1 = str(rng.choice(np.array(DIR_NAMES)))
    d2 = _turn(d1, int(rng.choice(np.array([-1, 1]))))
    core_dirs = [d1, d2, _opposite(d1), _opposite(d2), _opposite(d2)]

    max_len = min(height - 3, width - 3, 7)
    if max_len < 2:
        return None
    len1 = int(rng.integers(2, max_len + 1))
    len2 = int(rng.integers(2, max_len + 1))

    stop_patterns = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    obs1, obs2, obs3 = stop_patterns[int(rng.integers(len(stop_patterns)))]

    s0 = DIR_STEP[d1]
    s1 = DIR_STEP[d2]
    s2 = DIR_STEP[_opposite(d1)]
    s3 = DIR_STEP[_opposite(d2)]

    candidates: list[tuple[tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]]] = []
    for start_r in range(1, height):
        for start_c in range(width):
            start = (start_r, start_c)
            seg1 = _path_cells(start, s0[0], s0[1], len1)
            p1 = seg1[-1]
            seg2 = _path_cells(p1, s1[0], s1[1], len2)
            p2 = seg2[-1]
            seg3 = _path_cells(p2, s2[0], s2[1], len1)
            p3 = seg3[-1]
            seg4 = _path_cells(p3, s3[0], s3[1], len2 - 1)
            path = [start] + seg1 + seg2 + seg3 + seg4

            if len(set(path)) != len(path):
                continue
            if not all(_in_activity((height, width), r, c) for r, c in path):
                continue

            blockers: list[tuple[int, int]] = []
            checks = [
                (p1, s0, obs1),
                (p2, s1, obs2),
                (p3, s2, obs3),
            ]
            valid = True
            for end, step, use_obstacle in checks:
                nxt = (int(end[0] + step[0]), int(end[1] + step[1]))
                if use_obstacle:
                    if not _in_activity((height, width), nxt[0], nxt[1]):
                        valid = False
                        break
                    blockers.append(nxt)
                elif _in_activity((height, width), nxt[0], nxt[1]):
                    valid = False
                    break
            if not valid:
                continue

            if len(set(blockers)) != len(blockers):
                continue
            trail_set = set(path)
            if any(blocker in trail_set for blocker in blockers):
                continue

            candidates.append((start, set(blockers), path))

    if not candidates:
        return None

    start, blockers, path = candidates[int(rng.integers(len(candidates)))]
    return height, width, core_dirs, start, blockers, path


def _add_harmless_obstacles(
    rng: np.random.Generator,
    grid: np.ndarray,
    trail: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> None:
    height, width = map(int, grid.shape)
    target = int(rng.integers(0, 5))
    if target == 0:
        return

    forbidden = set(trail) | set(reserved)
    candidates: list[tuple[int, int]] = []
    for r in range(1, height):
        for c in range(width):
            if (r, c) in forbidden or int(grid[r, c]) != BLACK:
                continue
            if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in trail):
                continue
            candidates.append((r, c))
    if not candidates:
        return

    rng.shuffle(candidates)
    for r, c in candidates[:target]:
        grid[r, c] = GREY


def _quality_ok(
    grid: np.ndarray,
    trail: set[tuple[int, int]],
    metadata: list[dict[str, int | tuple[int, int]]],
) -> bool:
    height, width = map(int, grid.shape)
    area = (height - 1) * width
    trail_size = len(trail)
    if trail_size < 8 or trail_size > area // 2:
        return False

    reasons = {int(item["stop_reason"]) for item in metadata}
    if 2 not in reasons:
        return False
    if 0 not in reasons and 1 not in reasons:
        return False
    if not any(int(item["moved"]) == 0 for item in metadata):
        return False
    if not any(int(item["moved"]) > 0 for item in metadata):
        return False
    if int(np.count_nonzero(grid == GREY)) < 1:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the top-row direction tape and paint the full red self-blocking trail.

    Input structure:
    - The top row contains one contiguous instruction tape starting at the left edge;
      green means left, teal means right, pink means up, and yellow means down.
    - Rows below the tape form the activity area and contain one red start pixel plus
      zero or more grey obstacle pixels on a black background.

    Rule:
    - Read the tape from left to right.
    - For each instruction, slide the token in that direction until the next cell would
      leave the activity area, hit a grey obstacle, or hit an already red trail cell.
    - Every visited activity cell, including the initial start, stays red permanently.
    - Keep the tape row and the grey obstacles unchanged in the output.

    Key insight:
    - Earlier red trail cells are not just the answer being drawn; they become real
      blockers that can shorten or fully cancel later slides.
    """
    g = np.asarray(grid, dtype=int)
    trail, _metadata = _trace_run(g)
    out = g.copy()
    for r, c in trail:
        out[int(r), int(c)] = RED
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rng = np.random.default_rng()
        core = _make_core_layout(rng)
        if core is None:
            continue

        height, width, core_dirs, start, blockers, core_path = core
        extra_count = int(rng.integers(0, 4))
        extra_dirs = [str(rng.choice(np.array(DIR_NAMES))) for _ in range(extra_count)]
        tape_dirs = core_dirs + extra_dirs
        if len(tape_dirs) >= width:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[0, : len(tape_dirs)] = np.array([DIR_COLOR[name] for name in tape_dirs], dtype=int)
        grid[start[0], start[1]] = RED
        for r, c in blockers:
            grid[r, c] = GREY

        base_trail, base_meta = _trace_run(grid)
        if not _quality_ok(grid, base_trail, base_meta):
            continue

        reserved = set(blockers)
        reserved.add(start)
        _add_harmless_obstacles(rng, grid, base_trail, reserved)

        final_trail, final_meta = _trace_run(grid)
        if final_trail != base_trail:
            continue
        if not _quality_ok(grid, final_trail, final_meta):
            continue

        expected = grid.copy()
        for r, c in final_trail:
            expected[r, c] = RED
        if not np.array_equal(main(grid), expected):
            continue

        if len(core_path) < 8:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for self-blocking trail")
