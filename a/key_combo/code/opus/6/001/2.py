from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TRACK = 8
PATH = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        c0, c1 = c1, c0
    grid[row, c0 : c1 + 1] = color


def _paint_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, col] = color


def _track_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == TRACK) | (grid == PATH)


def _find_endpoints(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == PATH)
    assert coords.shape == (2, 2)
    a = tuple(map(int, coords[0]))
    b = tuple(map(int, coords[1]))
    return a, b


def _decode_periodic_rule(
    grid: np.ndarray,
) -> tuple[str, int, int, tuple[int, int, int, int]]:
    frame_box = _bbox(grid == GREY)
    assert frame_box is not None
    r0, c0, r1, c1 = frame_box
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = frame.shape
    assert h >= 5 and w >= 5

    border = np.zeros_like(frame, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(frame[border] == GREY)
    assert not np.any(frame[~border] == GREY)

    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]

    blue_rows = sorted(int(r) for r in np.unique(np.argwhere(interior == BLUE)[:, 0])) if np.any(interior == BLUE) else []
    blue_cols = sorted(int(c) for c in np.unique(np.argwhere(interior == BLUE)[:, 1])) if np.any(interior == BLUE) else []

    candidates: list[tuple[str, int, int]] = []

    if len(blue_rows) == 2 and np.all(interior[blue_rows, :] == BLUE) and np.count_nonzero(interior == BLUE) == 2 * interior.shape[1]:
        top, bot = blue_rows
        stripe_rows = list(range(ir0 + top + 1, ir0 + bot))
        red_coords = np.argwhere(grid == RED)
        assert red_coords.size > 0
        red_row_set = {int(r) for r, _c in red_coords}
        red_col_set = {int(c) for _r, c in red_coords}
        assert red_col_set == set(range(ic0, ic1 + 1))
        for period in (2, 3, 4):
            for offset in range(period):
                expected = {r for r in stripe_rows if r % period == offset}
                if expected == red_row_set:
                    candidates.append(("row", period, offset))

    if len(blue_cols) == 2 and np.all(interior[:, blue_cols] == BLUE) and np.count_nonzero(interior == BLUE) == 2 * interior.shape[0]:
        left, right = blue_cols
        stripe_cols = list(range(ic0 + left + 1, ic0 + right))
        red_coords = np.argwhere(grid == RED)
        assert red_coords.size > 0
        red_row_set = {int(r) for r, _c in red_coords}
        red_col_set = {int(c) for _r, c in red_coords}
        assert red_row_set == set(range(ir0, ir1 + 1))
        for period in (2, 3, 4):
            for offset in range(period):
                expected = {c for c in stripe_cols if c % period == offset}
                if expected == red_col_set:
                    candidates.append(("col", period, offset))

    assert len(candidates) == 1
    orient, period, offset = candidates[0]
    return orient, period, offset, frame_box


def _passable_mask(
    grid: np.ndarray,
    *,
    use_periodic: bool,
    use_blockers: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    track = _track_mask(g)
    blocked = np.zeros(g.shape, dtype=bool)

    if use_periodic:
        orient, period, offset, _frame_box = _decode_periodic_rule(g)
        if orient == "row":
            rows = np.arange(g.shape[0]) % period == offset
            blocked |= track & rows[:, None]
        else:
            cols = np.arange(g.shape[1]) % period == offset
            blocked |= track & cols[None, :]

    if use_blockers:
        for r, c in np.argwhere(g == ORANGE):
            for rr in range(int(r) - 1, int(r) + 2):
                for cc in range(int(c) - 1, int(c) + 2):
                    if _in_bounds(g.shape, rr, cc) and track[rr, cc]:
                        blocked[rr, cc] = True

    return track & ~blocked


def _adjacency(passable: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(passable):
        node = (int(r), int(c))
        nbrs = []
        for rr, cc in _neighbors4(*node):
            if _in_bounds(passable.shape, rr, cc) and passable[rr, cc]:
                nbrs.append((rr, cc))
        out[node] = nbrs
    return out


def _bfs_path(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors4(*cur):
            if _in_bounds(passable.shape, *nxt) and passable[nxt] and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    assert goal in prev
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _count_simple_paths(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    limit: int = 2,
) -> int:
    adj = _adjacency(passable)
    seen = {start}
    count = 0

    def dfs(node: tuple[int, int]) -> None:
        nonlocal count
        if count >= limit:
            return
        if node == goal:
            count += 1
            return
        for nxt in adj[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            dfs(nxt)
            seen.remove(nxt)
            if count >= limit:
                return

    dfs(start)
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the frame's periodic row/column mask, remove blocked track choices, and mark the unique safe route.

    Input structure:
    - Black background contains a teal orthogonal track network with exactly two maroon endpoint pixels on that network.
    - One grey rectangular frame contains two parallel blue bars plus red periodic stripes that encode either forbidden rows or forbidden columns.
    - One or more orange blocker pixels sit off the route and disable nearby track cells through their 8-neighborhoods.

    Rule:
    - Read the frame: horizontal blue bars mean the red stripes encode forbidden global rows, while vertical blue bars mean they encode forbidden global columns.
    - Infer the unique period and offset whose arithmetic progression matches exactly the red stripe rows or columns inside the frame.
    - Mark as impassable every teal/maroon track pixel on those forbidden rows or columns, and also every teal/maroon track pixel inside any orange blocker's 8-neighborhood.
    - In the remaining 4-connected passable track graph, find the unique simple path between the two maroon endpoints and recolor its teal pixels to maroon.
    - Erase the whole grey frame and all blue/red contents to black; keep orange blockers unchanged.

    Key insight:
    - The framed stripe pattern is not drawn onto the canvas as artwork; it globally defines which rows or columns of the existing track become unusable before pathfinding.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start, goal = _find_endpoints(g)
    _orient, _period, _offset, frame_box = _decode_periodic_rule(g)
    passable = _passable_mask(g, use_periodic=True, use_blockers=True)
    assert passable[start] and passable[goal]
    assert _count_simple_paths(passable, start, goal, limit=2) == 1

    path = _bfs_path(passable, start, goal)
    out = g.copy()
    for r, c in path:
        if out[r, c] == TRACK:
            out[r, c] = PATH

    r0, c0, r1, c1 = frame_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def _encode_frame(grid: np.ndarray, period: int, offset: int) -> None:
    fr, fc = 1, 1
    fh, fw = 9, 6
    grid[fr : fr + fh, fc : fc + fw] = BG
    grid[fr, fc : fc + fw] = GREY
    grid[fr + fh - 1, fc : fc + fw] = GREY
    grid[fr : fr + fh, fc] = GREY
    grid[fr : fr + fh, fc + fw - 1] = GREY

    left = fc + 1
    right = fc + fw - 2
    top_bar = fr + 1
    bot_bar = fr + fh - 2
    grid[top_bar, left : right + 1] = BLUE
    grid[bot_bar, left : right + 1] = BLUE

    for r in range(top_bar + 1, bot_bar):
        if r % period == offset:
            grid[r, left : right + 1] = RED


def _interval_free(
    lo: int,
    hi: int,
    intervals: list[tuple[int, int]],
    pad: int = 1,
) -> bool:
    for a, b in intervals:
        if not (hi + pad < a or b + pad < lo):
            return False
    return True


def _reserve_interval(lo: int, hi: int, intervals: list[tuple[int, int]]) -> None:
    intervals.append((int(lo), int(hi)))


def _make_canonical_row_case() -> np.ndarray:
    for _ in range(2000):
        h = random.randint(23, 29)
        w = random.randint(26, 30)
        period = random.choice([3, 4])

        frame_rows = list(range(3, 8))
        offsets = [off for off in range(period) if sum(r % period == off for r in frame_rows) >= 2]
        offset = random.choice(offsets)

        rf_candidates = [r for r in range(11, h - 5) if r % period == offset]
        if not rf_candidates:
            continue
        rf = random.choice(rf_candidates)
        r0 = rf + 1
        r1 = rf + 2
        blocker_row = r1 + 1
        if blocker_row >= h - 1:
            continue

        n_periodic = 2
        n_blocker = random.choice([1, 2])
        module_kinds = ["P"] * n_periodic + ["B"] * n_blocker
        random.shuffle(module_kinds)

        c0 = random.randint(7, 8)
        x = c0 + 1
        modules: list[tuple[str, int, int]] = []
        ok = True
        for kind in module_kinds:
            x += random.randint(1, 2)
            span = random.randint(2, 4) if kind == "P" else random.randint(3, 5)
            lo = x
            hi = x + span
            modules.append((kind, lo, hi))
            x = hi
        c_end = x + random.randint(1, 3)
        if c_end > w - 2:
            ok = False
        upper_specs = sorted((lo, hi) for kind, lo, hi in modules if kind == "P")
        lower_specs = sorted((lo, hi) for kind, lo, hi in modules if kind == "B")
        for specs in (upper_specs, lower_specs):
            for (a0, a1), (b0, b1) in zip(specs, specs[1:]):
                if a1 + 1 >= b0:
                    ok = False
        if not ok:
            continue

        grid = np.full((h, w), BG, dtype=int)
        _encode_frame(grid, period, offset)

        upper_intervals: list[tuple[int, int]] = []
        lower_intervals: list[tuple[int, int]] = []
        blocker_cols: list[int] = []

        _paint_h(grid, r0, c0, c_end, TRACK)

        for kind, lo, hi in modules:
            if kind == "P":
                _paint_v(grid, lo, rf, r0, TRACK)
                _paint_h(grid, rf, lo, hi, TRACK)
                _paint_v(grid, hi, rf, r0, TRACK)
                _reserve_interval(lo, hi, upper_intervals)
            else:
                _paint_v(grid, lo, r0, r1, TRACK)
                _paint_h(grid, r1, lo, hi, TRACK)
                _paint_v(grid, hi, r0, r1, TRACK)
                _reserve_interval(lo, hi, lower_intervals)
                mid = random.randint(lo + 1, hi - 1)
                grid[blocker_row, mid] = ORANGE
                blocker_cols.append(mid)

        upper_stub_count = random.randint(10, 14)

        for _ in range(upper_stub_count):
            for _ in range(80):
                anchor = random.randint(c0 + 2, c_end - 2)
                if not _interval_free(anchor, anchor, upper_intervals, pad=1):
                    continue
                _paint_v(grid, anchor, rf, r0, TRACK)
                _reserve_interval(anchor, anchor, upper_intervals)
                break

        grid[r0, c0] = PATH
        grid[r0, c_end] = PATH

        track_pixels = int(np.count_nonzero(_track_mask(grid)))
        if track_pixels < 40:
            continue
        if not _validate_instance(grid):
            continue
        return grid

    raise RuntimeError("failed to generate canonical row case")


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        start, goal = _find_endpoints(grid)
        orient, period, offset, frame_box = _decode_periodic_rule(grid)
        if orient not in {"row", "col"} or period not in {2, 3, 4} or not (0 <= offset < period):
            return False

        both = _passable_mask(grid, use_periodic=True, use_blockers=True)
        per_only = _passable_mask(grid, use_periodic=True, use_blockers=False)
        blk_only = _passable_mask(grid, use_periodic=False, use_blockers=True)

        if not (both[start] and both[goal] and per_only[start] and per_only[goal] and blk_only[start] and blk_only[goal]):
            return False

        if _count_simple_paths(both, start, goal, limit=2) != 1:
            return False
        if _count_simple_paths(per_only, start, goal, limit=2) < 2:
            return False
        if _count_simple_paths(blk_only, start, goal, limit=2) < 2:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        r0, c0, r1, c1 = frame_box
        if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return False

        path = _bfs_path(both, start, goal)
        expected = grid.copy()
        for r, c in path:
            if expected[r, c] == TRACK:
                expected[r, c] = PATH
        expected[r0 : r1 + 1, c0 : c1 + 1] = BG
        if not np.array_equal(out, expected):
            return False

        if np.count_nonzero(grid == ORANGE) < 1 or np.count_nonzero(grid == ORANGE) > 3:
            return False
        if np.count_nonzero(grid == PATH) != 2:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            grid = _make_canonical_row_case()
        except RuntimeError:
            continue
        grid = _apply_random_transform(grid)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate valid periodic fence path input")
