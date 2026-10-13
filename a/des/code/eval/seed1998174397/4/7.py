from __future__ import annotations

from collections import Counter, deque
from itertools import permutations
import random

import numpy as np


def _most_common_color(g: np.ndarray) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _wall_color(g: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(g[g != bg], return_counts=True)
    assert len(vals) > 0, "expected at least one non-background color"
    return int(vals[int(np.argmax(cnts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _door_sides(gap: tuple[int, int], orient: str) -> list[tuple[int, int]]:
    r, c = gap
    if orient == "v":
        return [(r, c - 1), (r, c + 1)]
    return [(r - 1, c), (r + 1, c)]


def _find_doors(g: np.ndarray, bg: int, wall: int) -> list[dict]:
    """
    Detect locked door gaps.

    A door is a background gap on a wall line with a single colored lock marker
    one cell away along the wall direction:
      - vertical wall gap: left/right are non-wall; above/below are (lock, wall) or (wall, lock)
      - horizontal wall gap: up/down are non-wall; left/right are (lock, wall) or (wall, lock)
    """
    h, w = g.shape
    doors: list[dict] = []
    seen_gaps: set[tuple[int, int]] = set()
    seen_locks: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg:
                continue

            candidates: list[tuple[str, tuple[int, int]]] = []

            # Vertical wall gap: passage is left <-> right, lock is above or below.
            if 0 < c < w - 1 and int(g[r, c - 1]) != wall and int(g[r, c + 1]) != wall:
                if r >= 2 and r + 1 < h:
                    if (
                        int(g[r - 1, c]) not in (bg, wall)
                        and int(g[r - 2, c]) == wall
                        and int(g[r + 1, c]) == wall
                    ):
                        candidates.append(("v", (r - 1, c)))
                if r + 2 < h and r - 1 >= 0:
                    if (
                        int(g[r + 1, c]) not in (bg, wall)
                        and int(g[r + 2, c]) == wall
                        and int(g[r - 1, c]) == wall
                    ):
                        candidates.append(("v", (r + 1, c)))

            # Horizontal wall gap: passage is up <-> down, lock is left or right.
            if 0 < r < h - 1 and int(g[r - 1, c]) != wall and int(g[r + 1, c]) != wall:
                if c >= 2 and c + 1 < w:
                    if (
                        int(g[r, c - 1]) not in (bg, wall)
                        and int(g[r, c - 2]) == wall
                        and int(g[r, c + 1]) == wall
                    ):
                        candidates.append(("h", (r, c - 1)))
                if c + 2 < w and c - 1 >= 0:
                    if (
                        int(g[r, c + 1]) not in (bg, wall)
                        and int(g[r, c + 2]) == wall
                        and int(g[r, c - 1]) == wall
                    ):
                        candidates.append(("h", (r, c + 1)))

            if not candidates:
                continue

            assert len(candidates) == 1, "each door gap must have exactly one lock"
            orient, lock = candidates[0]
            gap = (int(r), int(c))
            assert gap not in seen_gaps, "duplicate door gap detection"
            assert lock not in seen_locks, "duplicate lock detection"

            doors.append(
                {
                    "gap": gap,
                    "lock": (int(lock[0]), int(lock[1])),
                    "color": int(g[lock]),
                    "orient": orient,
                    "opened": False,
                    "sides": _door_sides(gap, orient),
                }
            )
            seen_gaps.add(gap)
            seen_locks.add(lock)

    colors = [door["color"] for door in doors]
    assert len(colors) == len(set(colors)), "door colors must be unique"
    return doors


def _find_keys(g: np.ndarray, bg: int, wall: int, lock_positions: set[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    keys: dict[int, tuple[int, int]] = {}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v in (bg, wall):
                continue
            if (r, c) in lock_positions:
                continue
            assert v not in keys, "key colors must be unique"
            keys[v] = (int(r), int(c))
    return keys


def _component_labels(out: np.ndarray, wall: int, doors: list[dict]) -> np.ndarray:
    h, w = out.shape
    blocked = out == wall
    for door in doors:
        if not door["opened"]:
            gr, gc = door["gap"]
            lr, lc = door["lock"]
            blocked[gr, gc] = True
            blocked[lr, lc] = True

    labels = np.full((h, w), -1, dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or labels[r, c] >= 0:
                continue
            q = deque([(int(r), int(c))])
            labels[r, c] = cid
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not _in_bounds(out.shape, nr, nc):
                        continue
                    if blocked[nr, nc] or labels[nr, nc] >= 0:
                        continue
                    labels[nr, nc] = cid
                    q.append((int(nr), int(nc)))
            cid += 1
    return labels


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    return []


def _segment_clear(
    out: np.ndarray,
    wall: int,
    doors: list[dict],
    start: tuple[int, int],
    target_gap: tuple[int, int],
) -> bool:
    pts = _segment_points(start, target_gap)
    if not pts:
        return False

    blocked = set()
    for door in doors:
        if door["opened"]:
            continue
        blocked.add(door["gap"])
        blocked.add(door["lock"])
    blocked.discard(target_gap)

    for r, c in pts:
        if int(out[r, c]) == wall:
            return False
        if (r, c) in blocked and (r, c) != start:
            return False
    return True


def _draw_segment(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    for r, c in _segment_points(a, b):
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-matched locked gaps between walled rooms, drawing straight key-to-door paths.

    The solver:
    - detects the dominant background and the dominant non-background wall color,
    - finds all wall gaps that have a single adjacent colored lock marker,
    - treats those gaps as initially blocked,
    - repeatedly opens any door whose color matches an unused key inside a currently
      connected passable region touching that door,
    - draws a straight row/column segment from the key to the gap and removes the lock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    wall = _wall_color(g, bg)
    doors = _find_doors(g, bg, wall)
    assert doors, "expected at least one locked door"

    lock_positions = {door["lock"] for door in doors}
    keys = _find_keys(g, bg, wall, lock_positions)
    assert keys, "expected at least one key"

    door_by_color = {int(door["color"]): door for door in doors}
    assert set(keys) == set(door_by_color), "keys and door colors must match exactly"

    out = g.copy()
    key_used = {int(color): False for color in keys}

    while True:
        labels = _component_labels(out, wall, doors)
        changed = False

        for color in sorted(keys):
            if key_used[color]:
                continue

            door = door_by_color[color]
            if door["opened"]:
                key_used[color] = True
                continue

            kr, kc = keys[color]
            comp = int(labels[kr, kc])
            if comp < 0:
                continue

            touches = False
            for sr, sc in door["sides"]:
                if _in_bounds(out.shape, sr, sc) and int(labels[sr, sc]) == comp:
                    touches = True
                    break
            if not touches:
                continue

            gap = door["gap"]
            if not _segment_clear(out, wall, doors, keys[color], gap):
                continue

            _draw_segment(out, keys[color], gap, color)
            lr, lc = door["lock"]
            out[lr, lc] = bg
            door["opened"] = True
            key_used[color] = True
            changed = True
            break

        if not changed:
            break

    return out


def _simulate_assignment(k: int, assign: dict[int, int]) -> tuple[bool, dict[int, int], list[list[tuple[int, int]]]]:
    opened = [False] * k  # door 0 is the outer-left door; doors 1..k-1 are internal separators.
    used_round: dict[int, int] = {}
    rounds: list[list[tuple[int, int]]] = []
    round_idx = 0

    while True:
        comps: list[list[int]] = []
        room = 1
        while room <= k:
            comp = [room]
            while room < k and opened[room]:
                room += 1
                comp.append(room)
            comps.append(comp)
            room += 1

        ops: list[tuple[int, int]] = []
        for comp in comps:
            boundary: set[int] = set()
            if comp[0] == 1 and not opened[0]:
                boundary.add(0)
            if comp[0] > 1 and not opened[comp[0] - 1]:
                boundary.add(comp[0] - 1)
            if comp[-1] < k and not opened[comp[-1]]:
                boundary.add(comp[-1])

            for r in comp:
                if r in used_round:
                    continue
                d = int(assign[r])
                if d in boundary and not opened[d]:
                    ops.append((int(r), d))

        if not ops:
            break

        round_idx += 1
        rounds.append(list(ops))
        for r, d in ops:
            used_round[int(r)] = round_idx
            opened[d] = True

    return all(opened), used_round, rounds


def _intervening_rooms(room: int, door: int) -> list[int]:
    if door == 0:
        return list(range(1, room))
    if room <= door:
        return list(range(room + 1, door + 1))
    return list(range(door + 1, room))


def _assignment_candidates(k: int) -> dict[int, list[tuple[dict[int, int], list[list[tuple[int, int]]]]]]:
    grouped: dict[int, list[tuple[dict[int, int], list[list[tuple[int, int]]]]]] = {}
    for perm in permutations(range(k), k):
        assign = {room + 1: int(perm[room]) for room in range(k)}
        ok, used_round, rounds = _simulate_assignment(k, assign)
        if not ok or len(rounds) < 2:
            continue

        safe = True
        for room, door in assign.items():
            t = used_round.get(room)
            if t is None:
                safe = False
                break
            for mid in _intervening_rooms(room, door):
                tm = used_round.get(mid)
                if tm is None or tm >= t:
                    safe = False
                    break
            if not safe:
                break

        if safe:
            grouped.setdefault(len(rounds), []).append((assign, rounds))

    assert grouped, f"expected at least one valid dependency pattern for k={k}"
    return grouped


_ASSIGNMENT_CACHE = {k: _assignment_candidates(k) for k in range(3, 7)}


def _choose_assignment(k: int) -> tuple[dict[int, int], list[list[tuple[int, int]]]]:
    grouped = _ASSIGNMENT_CACHE[k]
    n_rounds = random.choice(sorted(grouped))
    return random.choice(grouped[n_rounds])


def _build_base_sample() -> np.ndarray:
    bg = 0
    wall = 5

    k = random.randint(3, 6)
    assign, _rounds = _choose_assignment(k)

    room_h = random.randint(4, 6)
    widths = [random.randint(4, 6) for _ in range(k)]
    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    left_margin = random.randint(2, 5)
    right_margin = random.randint(2, 5)

    building_h = room_h + 2
    building_w = sum(widths) + k + 1
    h = top_margin + building_h + bottom_margin
    w = left_margin + building_w + right_margin

    grid = np.full((h, w), bg, dtype=int)

    top = top_margin
    left = left_margin
    bottom = top + building_h - 1
    right = left + building_w - 1

    wall_x = [left]
    cur = left
    for width in widths:
        cur += width + 1
        wall_x.append(cur)

    grid[top : bottom + 1, left : right + 1] = bg
    grid[top, left : right + 1] = wall
    grid[bottom, left : right + 1] = wall
    for x in wall_x:
        grid[top : bottom + 1, x] = wall

    corridor_r = random.randint(top + 2, bottom - 2)
    door_positions = {door: (corridor_r, wall_x[door]) for door in range(k)}

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], k)
    door_colors = {door: colors[door] for door in range(k)}

    for door, (dr, dc) in door_positions.items():
        grid[dr, dc] = bg
        lock_offset = random.choice([-1, 1])
        grid[dr + lock_offset, dc] = door_colors[door]

    for room_idx in range(k):
        room = room_idx + 1
        x0 = wall_x[room_idx] + 1
        x1 = wall_x[room_idx + 1] - 1
        key_c = random.randint(x0, x1)
        grid[corridor_r, key_c] = door_colors[assign[room]]

    assert _most_common_color(grid) == bg
    assert _wall_color(grid, bg) == wall
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a corridor of rectangular rooms with unique lock colors and a
    dependency pattern that forces at least two rounds of door openings.
    """
    for _ in range(300):
        grid = _build_base_sample()
        krot = random.randint(0, 3)
        if krot:
            grid = np.rot90(grid, krot)

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        return grid

    raise AssertionError("failed to generate a valid sample")
