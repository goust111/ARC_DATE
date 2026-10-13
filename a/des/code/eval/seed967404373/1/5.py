from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

DIR_TO_COLOR = {
    (-1, 0): GREEN,
    (0, 1): YELLOW,
    (1, 0): ORANGE,
    (0, -1): MAROON,
}
COLOR_TO_DIR = {color: direction for direction, color in DIR_TO_COLOR.items()}
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DRAW_COLORS = [GREEN, YELLOW, MAGENTA, ORANGE, CYAN, MAROON]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_mode(values: list[int] | np.ndarray) -> int:
    counts = Counter(map(int, values))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique mode"
    return int(ranked[0][0])


def _divider_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey divider"
    return int(rows[0])


def _parse_locks(g: np.ndarray, divider: int) -> list[tuple[tuple[int, int], int, int]]:
    bottom = g[divider + 1 :, :]
    grey_comps = _components4(bottom == GREY)
    locks: list[tuple[int, tuple[int, int], int, int]] = []
    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != 3 or not (5 <= w <= 8):
            continue

        sub = bottom[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == GREY):
            continue
        if not np.all(sub[-1, :] == GREY):
            continue
        if not np.all(sub[:, 0] == GREY):
            continue
        if not np.all(sub[:, -1] == GREY):
            continue

        interior = list(map(int, sub[1, 1:-1]))
        if not (3 <= len(interior) <= 6):
            continue
        if any(v == GREY for v in interior):
            continue

        dir_col = c0 - 1
        if dir_col < 0:
            continue
        marker = int(bottom[r0 + 1, dir_col])
        if marker not in COLOR_TO_DIR:
            continue

        locks.append(
            (
                c0,
                COLOR_TO_DIR[marker],
                len(interior),
                _unique_mode(interior),
            )
        )

    assert locks, "expected at least one lock"
    locks.sort(key=lambda item: item[0])
    return [(direction, steps, color) for _c0, direction, steps, color in locks]


def _simulate_maze(
    maze: np.ndarray,
    instructions: list[tuple[tuple[int, int], int, int]],
) -> np.ndarray:
    h, w = maze.shape
    start_pts = np.argwhere(maze == RED)
    assert len(start_pts) == 1, "expected exactly one red start"
    cur = tuple(map(int, start_pts[0]))
    prev: tuple[int, int] | None = None
    passable = (maze == BLUE) | (maze == RED)

    out = maze.copy()
    for heading, steps, paint in instructions:
        out[cur] = paint
        dr, dc = heading
        for _ in range(steps):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and bool(passable[nr, nc]):
                nxt = (nr, nc)
            else:
                candidates = []
                for rr, cc in _neighbors4(cur[0], cur[1]):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if prev is not None and (rr, cc) == prev:
                        continue
                    if bool(passable[rr, cc]):
                        candidates.append((rr, cc))
                assert len(candidates) == 1, "blocked step must have a unique alternative"
                nxt = candidates[0]
            prev, cur = cur, nxt
            out[cur] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read movement instructions from boxed locks and paint the cursor trail on the maze.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _divider_row(g)
    maze = g[:divider, :].copy()
    instructions = _parse_locks(g, divider)
    return _simulate_maze(maze, instructions)


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(dc, dr), (-dc, -dr)]


def _make_color_sequence(length: int, direction: tuple[int, int], force_mixed: bool) -> tuple[list[int], int]:
    dir_color = DIR_TO_COLOR[direction]
    major = random.choice([c for c in DRAW_COLORS if c != dir_color])
    min_major = length // 2 + 1
    if force_mixed or random.random() < 0.75:
        max_major = length - 1
    else:
        max_major = length
    if max_major < min_major:
        max_major = length
    major_count = random.randint(min_major, max_major)
    seq = [major] * major_count
    others = [c for c in DRAW_COLORS if c != major]
    while len(seq) < length:
        seq.append(random.choice(others))
    random.shuffle(seq)
    assert _unique_mode(seq) == major
    return seq, major


def _make_instruction(
    length: int,
    direction: tuple[int, int],
    last_move: tuple[int, int] | None,
    force_detour: bool,
    force_mixed: bool,
) -> dict:
    reverse_start = last_move is not None and direction == _opposite(last_move)
    has_detour = force_detour or reverse_start or random.random() < 0.45

    if has_detour:
        detour_len = random.randint(1, min(2, length - 1))
        prefix = random.randint(0, length - detour_len)
        if reverse_start:
            prefix = 0
        if prefix == length - detour_len:
            suffix = 0
        else:
            suffix = length - prefix - detour_len
        if prefix == 0 and suffix == 0:
            suffix = 1
            detour_len -= 1
        detour_dir = random.choice(_perpendiculars(direction))
        moves = [direction] * prefix + [detour_dir] * detour_len + [direction] * suffix
    else:
        moves = [direction] * length

    seq, major = _make_color_sequence(length, direction, force_mixed)
    return {
        "direction": direction,
        "steps": length,
        "paint": major,
        "sequence": seq,
        "moves": moves,
    }


def _is_valid_extension(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = nxt
    if not (0 <= r < h and 0 <= c < w):
        return False
    if nxt in occupied or nxt in reserved_empty:
        return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) == cur:
            continue
        if (nr, nc) in occupied:
            return False
    return True


def _build_walk(
    top_h: int,
    width: int,
    instructions: list[dict],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    total_steps = sum(inst["steps"] for inst in instructions)
    for _ in range(250):
        start = (
            random.randint(2, top_h - 3),
            random.randint(2, width - 3),
        )
        walk = [start]
        occupied = {start}
        reserved_empty: set[tuple[int, int]] = set()
        branchable: list[tuple[tuple[int, int], tuple[int, int]]] = []
        ok = True

        for inst in instructions:
            heading = inst["direction"]
            for move in inst["moves"]:
                cur = walk[-1]
                nxt = (cur[0] + move[0], cur[1] + move[1])
                if not _is_valid_extension(nxt, cur, occupied, reserved_empty, top_h, width):
                    ok = False
                    break

                if move != heading:
                    fr, fc = cur[0] + heading[0], cur[1] + heading[1]
                    if 0 <= fr < top_h and 0 <= fc < width:
                        blocked = (fr, fc)
                        if blocked in occupied or blocked in reserved_empty:
                            ok = False
                            break
                        reserved_empty.add(blocked)
                else:
                    branchable.append((cur, heading))

                occupied.add(nxt)
                walk.append(nxt)

            if not ok:
                break

        if ok and len(walk) == total_steps + 1:
            return walk, reserved_empty, branchable
    return None


def _extend_branch(
    source: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    local = set(occupied)
    for branch_dir in random.sample(_perpendiculars(direction), 2):
        for length in random.sample([1, 2], 2):
            cells: list[tuple[int, int]] = []
            cur = source
            ok = True
            for _ in range(length):
                nxt = (cur[0] + branch_dir[0], cur[1] + branch_dir[1])
                if not _is_valid_extension(nxt, cur, local, reserved_empty, h, w):
                    ok = False
                    break
                local.add(nxt)
                cells.append(nxt)
                cur = nxt
            if ok:
                return cells
    return None


def _add_branches(
    walk: list[tuple[int, int]],
    branchable: list[tuple[tuple[int, int], tuple[int, int]]],
    reserved_empty: set[tuple[int, int]],
    top_h: int,
    width: int,
) -> list[tuple[int, int]] | None:
    occupied = set(walk)
    branches: list[tuple[int, int]] = []
    random.shuffle(branchable)
    target = random.randint(1, 3)
    for source, heading in branchable:
        if target == 0:
            break
        cells = _extend_branch(source, heading, occupied, reserved_empty, top_h, width)
        if cells is None:
            continue
        branches.extend(cells)
        occupied.update(cells)
        target -= 1
    if not branches:
        return None
    return branches


def _expected_top(
    top: np.ndarray,
    walk: list[tuple[int, int]],
    instructions: list[dict],
) -> np.ndarray:
    out = top.copy()
    idx = 0
    for inst in instructions:
        color = int(inst["paint"])
        steps = int(inst["steps"])
        for r, c in walk[idx : idx + steps + 1]:
            out[r, c] = color
        idx += steps
    return out


def _place_locks(grid: np.ndarray, top_h: int, instructions: list[dict]) -> None:
    divider = top_h
    grid[divider, :] = GREY
    frame_top = divider + 2
    used_width = sum(inst["steps"] + 3 for inst in instructions)
    offset = random.randint(1, grid.shape[1] - used_width - 1)
    x = offset
    for inst in instructions:
        direction = inst["direction"]
        steps = inst["steps"]
        seq = inst["sequence"]
        grid[frame_top + 1, x] = DIR_TO_COLOR[direction]
        left = x + 1
        right = left + steps + 1
        grid[frame_top, left : right + 1] = GREY
        grid[frame_top + 2, left : right + 1] = GREY
        grid[frame_top : frame_top + 3, left] = GREY
        grid[frame_top : frame_top + 3, right] = GREY
        grid[frame_top + 1, left + 1 : right] = np.array(seq, dtype=int)
        x = right + 1


def _make_attempt() -> np.ndarray | None:
    lock_count = random.choice([3, 4])
    lengths = [random.randint(3, 6) for _ in range(lock_count)]
    total_steps = sum(lengths)

    instructions: list[dict] = []
    turn_idx = random.randrange(lock_count)
    mixed_idx = random.randrange(lock_count)
    last_move: tuple[int, int] | None = None
    for i, length in enumerate(lengths):
        direction = random.choice(CARDINALS)
        inst = _make_instruction(
            length=length,
            direction=direction,
            last_move=last_move,
            force_detour=(i == turn_idx),
            force_mixed=(i == mixed_idx),
        )
        instructions.append(inst)
        last_move = inst["moves"][-1]

    if not any(any(move != inst["direction"] for move in inst["moves"]) for inst in instructions):
        return None

    used_width = sum(length + 3 for length in lengths)
    width = max(22, used_width + random.randint(2, 6))
    top_h = random.randint(12, min(18, max(12, total_steps + 2)))
    if width < 7 or top_h < 7:
        return None

    built = _build_walk(top_h, width, instructions)
    if built is None:
        return None
    walk, reserved_empty, branchable = built
    branches = _add_branches(walk, branchable, reserved_empty, top_h, width)
    if branches is None:
        return None

    total_h = top_h + 6
    grid = np.full((total_h, width), BLACK, dtype=int)
    top = grid[:top_h, :]
    for r, c in walk:
        top[r, c] = BLUE
    for r, c in branches:
        top[r, c] = BLUE
    top[walk[0]] = RED
    _place_locks(grid, top_h, instructions)

    expected = _expected_top(top.copy(), walk, instructions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-lock puzzle")
