from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
GREY = 5
COLORS = tuple(c for c in range(10) if c not in (AZURE, GREY))
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or not bool(mask[rr, cc]):
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full_2x2(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 == 1 and c1 - c0 == 1


def _find_key_lock_pairs(grid: np.ndarray) -> list[dict]:
    pairs: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (AZURE, GREY):
            continue

        keys: list[tuple[int, int]] = []
        locks: list[np.ndarray] = []
        for comp in _components(grid == color):
            if int(comp.shape[0]) == 1:
                r, c = comp[0]
                keys.append((int(r), int(c)))
            elif _is_full_2x2(comp):
                locks.append(comp)
            else:
                raise AssertionError("each colored mark must be either a key pixel or a 2x2 lock")

        assert len(keys) == 1, "each color needs exactly one single-pixel key"
        assert len(locks) == 1, "each color needs exactly one 2x2 lock"
        pairs.append({"color": color, "key": keys[0], "lock": locks[0]})

    assert pairs, "expected at least one key-lock color"
    pairs.sort(key=lambda item: item["key"])
    return pairs


def _lock_entry_cells(grid: np.ndarray, lock: np.ndarray) -> set[tuple[int, int]]:
    height, width = map(int, grid.shape)
    entries: set[tuple[int, int]] = set()
    lock_set = {(int(r), int(c)) for r, c in lock}
    for r, c in lock_set:
        for rr, cc in _neighbors4(r, c):
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if (rr, cc) in lock_set:
                continue
            if int(grid[rr, cc]) == AZURE:
                entries.add((rr, cc))
    assert entries, "lock room must have at least one Azure entry cell next to the lock"
    return entries


def _shortest_path_to_any_entry(
    grid: np.ndarray,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    found: tuple[int, int] | None = None
    while q:
        r, c = q.popleft()
        if (r, c) in targets:
            found = (r, c)
            break
        for rr, cc in _neighbors4(r, c):
            nxt = (rr, cc)
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if nxt in prev:
                continue
            if int(grid[rr, cc]) != AZURE:
                continue
            prev[nxt] = (r, c)
            q.append(nxt)

    assert found is not None, "key must have an Azure path to its matching lock room"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each colored key through Azure corridors to its same-colored 2x2 lock.

    Input structure:
    - The grid background is Azure.
    - Grey one-pixel orthogonal walls form walled rooms and one-pixel corridor routes.
    - Every non-wall, non-background color appears as exactly one single-pixel key and one separate 2x2 lock block.

    Rule:
    - Match each key to the 2x2 lock of the same color.
    - For each pair, find a shortest 4-connected route from the key through Azure cells to an Azure cell adjacent to the lock block.
    - Color every Azure cell on that route with the key color, erase the original key cell back to Azure, and leave all Grey walls and 2x2 locks unchanged.

    Key insight:
    The color gives the key-lock pairing, but the visible wall network determines which Azure corridor pixels belong to the route.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "input grid exceeds ARC size limit"

    out = g.copy()
    for pair in _find_key_lock_pairs(g):
        color = int(pair["color"])
        key = pair["key"]
        entries = _lock_entry_cells(g, pair["lock"])
        path = _shortest_path_to_any_entry(g, key, entries)

        for r, c in path:
            if int(g[r, c]) == AZURE:
                out[r, c] = color
        out[key] = AZURE

    return out


def _draw_room(grid: np.ndarray, top: int, left: int, room_h: int, room_w: int) -> None:
    bottom = top + room_h + 1
    right = left + room_w + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY


def _paint_lock(grid: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in (r0, r0 + 1) for c in (c0, c0 + 1)]
    for r, c in cells:
        grid[r, c] = color
    return cells


def _draw_horizontal_pair(
    grid: np.ndarray,
    top: int,
    left: int,
    room_h: int,
    room_w: int,
    gap: int,
    color: int,
    direction: int,
) -> dict:
    bottom = top + room_h + 1
    a_left = left
    a_right = a_left + room_w + 1
    b_left = a_right + gap + 1
    b_right = b_left + room_w + 1

    _draw_room(grid, top, a_left, room_h, room_w)
    _draw_room(grid, top, b_left, room_h, room_w)

    door_r = random.randint(top + 2, bottom - 2)
    for c in range(a_right, b_left + 1):
        grid[door_r - 1, c] = GREY
        grid[door_r, c] = AZURE
        grid[door_r + 1, c] = GREY

    lock_top = door_r + random.choice((-1, 0))
    lock_top = max(top + 1, min(lock_top, bottom - 2))

    if direction > 0:
        key = (door_r, a_right - 1)
        target = (door_r, b_left + 1)
        lock = _paint_lock(grid, lock_top, target[1] + 1, color)
        path = [(door_r, c) for c in range(a_right, b_left + 2)]
    else:
        key = (door_r, b_left + 1)
        target = (door_r, a_right - 1)
        lock = _paint_lock(grid, lock_top, target[1] - 2, color)
        path = [(door_r, c) for c in range(b_left, a_right - 2, -1)]

    grid[key] = color
    return {
        "color": color,
        "key": key,
        "lock": lock,
        "path": path,
        "door_row": door_r,
        "connector_col": a_right + 1,
    }


def _draw_vertical_pair(
    grid: np.ndarray,
    top: int,
    left: int,
    room_h: int,
    room_w: int,
    gap: int,
    color: int,
    direction: int,
) -> dict:
    right = left + room_w + 1
    a_top = top
    a_bottom = a_top + room_h + 1
    b_top = a_bottom + gap + 1
    b_bottom = b_top + room_h + 1

    _draw_room(grid, a_top, left, room_h, room_w)
    _draw_room(grid, b_top, left, room_h, room_w)

    door_c = random.randint(left + 2, right - 2)
    for r in range(a_bottom, b_top + 1):
        grid[r, door_c - 1] = GREY
        grid[r, door_c] = AZURE
        grid[r, door_c + 1] = GREY

    lock_left = door_c + random.choice((-1, 0))
    lock_left = max(left + 1, min(lock_left, right - 2))

    if direction > 0:
        key = (a_bottom - 1, door_c)
        target = (b_top + 1, door_c)
        lock = _paint_lock(grid, target[0] + 1, lock_left, color)
        path = [(r, door_c) for r in range(a_bottom, b_top + 2)]
    else:
        key = (b_top + 1, door_c)
        target = (a_bottom - 1, door_c)
        lock = _paint_lock(grid, target[0] - 2, lock_left, color)
        path = [(r, door_c) for r in range(b_top, a_bottom - 2, -1)]

    grid[key] = color
    return {
        "color": color,
        "key": key,
        "lock": lock,
        "path": path,
        "door_col": door_c,
        "connector_row": a_bottom + 1,
    }


def _connect_vertical(grid: np.ndarray, col: int, door_rows: list[int]) -> None:
    rows = set(door_rows)
    for r in range(min(rows), max(rows) + 1):
        grid[r, col] = AZURE
        if r not in rows:
            grid[r, col - 1] = GREY
            grid[r, col + 1] = GREY


def _connect_horizontal(grid: np.ndarray, row: int, door_cols: list[int]) -> None:
    cols = set(door_cols)
    for c in range(min(cols), max(cols) + 1):
        grid[row, c] = AZURE
        if c not in cols:
            grid[row - 1, c] = GREY
            grid[row + 1, c] = GREY


def _expected_from_records(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = grid.copy()
    painted: dict[tuple[int, int], int] = {}
    for rec in records:
        color = int(rec["color"])
        for r, c in rec["path"]:
            assert int(grid[r, c]) == AZURE
            if (r, c) in painted and painted[(r, c)] != color:
                raise AssertionError("generated routes must not overlap")
            painted[(r, c)] = color
            out[r, c] = color
        out[rec["key"]] = AZURE
        for r, c in rec["lock"]:
            out[r, c] = color
    return out


def _gen_horizontal() -> np.ndarray | None:
    count = random.randint(2, 4)
    room_h = 4 if count == 4 else random.choice((4, 5))
    room_w = random.randint(4, 6)
    max_gap = 30 - 2 * room_w - 5
    if max_gap < 4:
        return None
    gap = random.randint(4, max_gap)

    suite_h = room_h + 2
    vgap_options = [g for g in (1, 2) if count * suite_h + (count - 1) * g <= 30]
    if not vgap_options:
        return None
    vgap = random.choice(vgap_options)
    min_h = count * suite_h + (count - 1) * vgap
    min_w = 2 * room_w + gap + 5
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    top0 = random.randint(0, height - min_h)
    left = random.randint(0, width - min_w)

    grid = np.full((height, width), AZURE, dtype=int)
    records: list[dict] = []
    colors = random.sample(COLORS, count)
    for idx, color in enumerate(colors):
        top = top0 + idx * (suite_h + vgap)
        direction = random.choice((-1, 1))
        records.append(_draw_horizontal_pair(grid, top, left, room_h, room_w, gap, color, direction))

    _connect_vertical(grid, records[0]["connector_col"], [rec["door_row"] for rec in records])
    return _finalize_generated(grid, records)


def _gen_vertical() -> np.ndarray | None:
    count = random.randint(2, 4)
    room_w = 4 if count == 4 else random.choice((4, 5))
    room_h = random.randint(4, 6)
    max_gap = 30 - 2 * room_h - 5
    if max_gap < 4:
        return None
    gap = random.randint(4, max_gap)

    suite_w = room_w + 2
    hgap_options = [g for g in (1, 2) if count * suite_w + (count - 1) * g <= 30]
    if not hgap_options:
        return None
    hgap = random.choice(hgap_options)
    min_h = 2 * room_h + gap + 5
    min_w = count * suite_w + (count - 1) * hgap
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    top = random.randint(0, height - min_h)
    left0 = random.randint(0, width - min_w)

    grid = np.full((height, width), AZURE, dtype=int)
    records: list[dict] = []
    colors = random.sample(COLORS, count)
    for idx, color in enumerate(colors):
        left = left0 + idx * (suite_w + hgap)
        direction = random.choice((-1, 1))
        records.append(_draw_vertical_pair(grid, top, left, room_h, room_w, gap, color, direction))

    _connect_horizontal(grid, records[0]["connector_row"], [rec["door_col"] for rec in records])
    return _finalize_generated(grid, records)


def _finalize_generated(grid: np.ndarray, records: list[dict]) -> np.ndarray | None:
    try:
        expected = _expected_from_records(grid, records)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_horizontal() if random.randrange(2) == 0 else _gen_vertical()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock corridor input")
