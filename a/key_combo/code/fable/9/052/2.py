from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_colors(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _get_bg(grid)
    non_bg = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert len(non_bg) == 2, f"expected exactly two foreground colors, got {non_bg}"
    top_touching = [c for c in non_bg if bool(np.any(grid[0] == int(c)))]
    assert len(top_touching) == 1, f"expected one top-touching foreground color, got {top_touching}"
    water = int(top_touching[0])
    shelf = int(non_bg[0] if non_bg[1] == water else non_bg[1])
    return bg, water, shelf


def _reservoirs(grid: np.ndarray, water: int) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    out: list[tuple[int, int, int]] = []
    cols = [int(c) for c in range(W) if int(grid[0, c]) == int(water)]
    assert cols, "expected at least one top-edge reservoir"
    assert all(cols[i + 1] - cols[i] > 1 for i in range(len(cols) - 1)), "reservoirs must be single columns"
    for c in cols:
        h = 0
        while h < H and int(grid[h, c]) == int(water):
            h += 1
        assert h >= 1, "reservoir height must be positive"
        out.append((h - 1, c, h))
    return out


def _shelf_segments(grid: np.ndarray, shelf: int) -> dict[int, list[tuple[int, int]]]:
    H, W = grid.shape
    segs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        c = 0
        while c < W:
            if int(grid[r, c]) != int(shelf):
                c += 1
                continue
            c0 = c
            while c + 1 < W and int(grid[r, c + 1]) == int(shelf):
                c += 1
            segs[r].append((c0, c))
            c += 1
    return segs


def _find_segment(segs: dict[int, list[tuple[int, int]]], r: int, c: int) -> tuple[int, int] | None:
    for a, b in segs.get(int(r), []):
        if int(a) <= int(c) <= int(b):
            return int(a), int(b)
    return None


def _simulate(grid: np.ndarray, *, with_events: bool = False):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg, water, shelf = _find_colors(g)
    reservoirs = _reservoirs(g, water)
    segs = _shelf_segments(g, shelf)

    out = g.copy()
    q = deque((int(r), int(c), int(budget), int(idx)) for idx, (r, c, budget) in enumerate(reservoirs))
    seen: set[tuple[int, int, int, int]] = set()
    events = {
        "reservoir_heights": {int(idx): int(b) for idx, (_r, _c, b) in enumerate(reservoirs)},
        "splits": [],
        "pools": [],
        "ends": [],
    }

    while q:
        r, c, budget, source_id = q.popleft()
        state = (int(r), int(c), int(budget), int(source_id))
        if state in seen:
            continue
        seen.add(state)

        rr = int(r)
        cc = int(c)
        out[rr, cc] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= H:
                events["ends"].append((int(source_id), int(rr), int(cc), int(budget)))
                break

            if int(g[nr, cc]) == int(shelf):
                seg = _find_segment(segs, nr, cc)
                assert seg is not None, "shelf hit must belong to a horizontal segment"
                a, b = seg
                if int(budget) > 0:
                    next_budget = int(budget - 1)
                    children = []
                    for nc in (int(a - 1), int(b + 1)):
                        if 0 <= nc < W and int(g[rr, nc]) != int(shelf):
                            children.append((rr, nc, next_budget, source_id))
                    events["splits"].append((int(source_id), int(rr), int(cc), int(budget), int(a), int(b)))
                    for child in children:
                        q.append(child)
                else:
                    for x in range(int(a), int(b) + 1):
                        if int(g[rr, x]) != int(shelf):
                            out[rr, x] = int(water)
                    events["pools"].append((int(source_id), int(rr), int(a), int(b)))
                break

            rr = nr
            if int(g[rr, cc]) != int(shelf):
                out[rr, cc] = int(water)

    if with_events:
        return out, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate budget-limited waterfall cascades released from top-edge reservoir columns.

    Input structure:
    - The background is the most common color.
    - Several single-column water reservoirs touch the top edge; each reservoir is a vertical bar of one foreground color.
    - Lower in the grid, the other foreground color forms separated horizontal shelf segments.

    Rule:
    - Each reservoir releases one downward stream from its bottom cell, carrying a budget equal to the reservoir height.
    - A falling stream colors every background cell it passes through.
    - When a stream meets a shelf from above: if its budget is positive, decrease the budget by one and spawn two child streams from the cells just outside the shelf's left and right ends; if its budget is zero, spread water across the full shelf span on the row above the shelf and stop there.

    Key insight:
    - Reservoir height is not decorative: it is the exact number of shelf layers that this waterfall family may bypass before a later collision becomes a terminal pool.
    """
    return _simulate(grid, with_events=False)


def _bounded_counts(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if total < parts * lo or total > parts * hi:
        return None
    counts = [lo] * parts
    extra = total - parts * lo
    while extra:
        idxs = [i for i, v in enumerate(counts) if v < hi]
        i = random.choice(idxs)
        add = random.randint(1, min(hi - counts[i], extra))
        counts[i] += add
        extra -= add
    random.shuffle(counts)
    return counts


def _touches_existing_shelf(shelf_cells: set[tuple[int, int]], r: int, a: int, b: int) -> bool:
    for c in range(int(a), int(b) + 1):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r + dr), int(c + dc)) in shelf_cells:
                    return True
    return False


def _add_shelf(
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
    r: int,
    a: int,
    b: int,
) -> bool:
    if a > b:
        return False
    if _touches_existing_shelf(shelf_cells, r, a, b):
        return False
    for c in range(int(a), int(b) + 1):
        if (int(r), int(c)) in reserved_bg:
            return False
    for c in range(int(a), int(b) + 1):
        shelf_cells.add((int(r), int(c)))
    shelves.append((int(r), int(a), int(b)))
    return True


def _place_split_shelf(
    H: int,
    L: int,
    R: int,
    row: int,
    col: int,
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[int, tuple[int, int]] | None:
    for _ in range(50):
        shelf_row = random.randint(int(row + 2), min(H - 2, int(row + 4)))
        if shelf_row <= int(row + 1):
            continue
        if any((rr, int(col)) in shelf_cells for rr in range(int(row + 1), int(shelf_row))):
            continue

        max_left = min(3, int(col - (L + 1)))
        max_right = min(3, int((R - 1) - col))
        if max_left < 0 or max_right < 0:
            continue
        left_ext = random.randint(0, max_left)
        right_ext = random.randint(0, max_right)
        a = int(col - left_ext)
        b = int(col + right_ext)
        if b - a + 1 < 2:
            if b + 1 <= R - 1:
                b += 1
            elif a - 1 >= L + 1:
                a -= 1
        if b - a + 1 < 2:
            continue
        if a <= L or b >= R:
            continue
        if not _add_shelf(shelf_cells, shelves, reserved_bg, shelf_row, a, b):
            continue

        for rr in range(int(row + 1), int(shelf_row)):
            reserved_bg.add((int(rr), int(col)))
        spread_row = int(shelf_row - 1)
        reserved_bg.add((spread_row, int(a - 1)))
        reserved_bg.add((spread_row, int(b + 1)))
        return spread_row, (int(a - 1), int(b + 1))
    return None


def _place_pool_shelf(
    H: int,
    L: int,
    R: int,
    row: int,
    col: int,
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(50):
        shelf_row = random.randint(int(row + 2), min(H - 1, int(row + 4)))
        if shelf_row <= int(row + 1):
            continue
        if any((rr, int(col)) in shelf_cells for rr in range(int(row + 1), int(shelf_row))):
            continue

        max_left = min(4, int(col - L))
        max_right = min(4, int(R - col))
        left_ext = random.randint(0, max_left)
        right_ext = random.randint(0, max_right)
        a = int(col - left_ext)
        b = int(col + right_ext)
        if b - a + 1 < 2:
            if b < R:
                b += 1
            elif a > L:
                a -= 1
        if b - a + 1 < 2:
            continue
        if not _add_shelf(shelf_cells, shelves, reserved_bg, shelf_row, a, b):
            continue

        for rr in range(int(row + 1), int(shelf_row)):
            reserved_bg.add((int(rr), int(col)))
        return True
    return False


def _plan_side_branch(
    H: int,
    L: int,
    R: int,
    row: int,
    col: int,
    budget: int,
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if budget < 0:
        return True
    if budget == 0:
        if random.random() < 0.7:
            return _place_pool_shelf(H, L, R, row, col, shelf_cells, shelves, reserved_bg)
        return True

    if random.random() < 0.45:
        return True

    placed = _place_split_shelf(H, L, R, row, col, shelf_cells, shelves, reserved_bg)
    if placed is None:
        return False
    spread_row, children = placed
    child = random.choice(children)
    if budget - 1 == 0 and random.random() < 0.8:
        return _place_pool_shelf(H, L, R, spread_row, child, shelf_cells, shelves, reserved_bg)
    return True


def _plan_mainline(
    H: int,
    L: int,
    R: int,
    row: int,
    col: int,
    budget: int,
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if budget == 0:
        return _place_pool_shelf(H, L, R, row, col, shelf_cells, shelves, reserved_bg)

    placed = _place_split_shelf(H, L, R, row, col, shelf_cells, shelves, reserved_bg)
    if placed is None:
        return False
    spread_row, children = placed
    continue_child = random.choice(children)
    side_child = int(children[0] if children[1] == continue_child else children[1])
    if not _plan_side_branch(H, L, R, spread_row, side_child, budget - 1, shelf_cells, shelves, reserved_bg):
        return False
    return _plan_mainline(H, L, R, spread_row, continue_child, budget - 1, shelf_cells, shelves, reserved_bg)


def _add_distractors(
    H: int,
    W: int,
    shelf_cells: set[tuple[int, int]],
    shelves: list[tuple[int, int, int]],
    reserved_bg: set[tuple[int, int]],
) -> None:
    for _ in range(random.randint(0, 2)):
        for _ in range(30):
            r = random.randint(3, H - 1)
            length = random.randint(2, min(6, W))
            a = random.randint(0, W - length)
            b = a + length - 1
            if _add_shelf(shelf_cells, shelves, reserved_bg, r, a, b):
                break


def _colors_for_generation() -> tuple[int, int, int]:
    bg = random.choice([0, 7, 8])
    water = random.choice([1, 6, 9])
    shelf = random.choice([2, 3, 4, 5])
    while len({bg, water, shelf}) < 3:
        bg = random.choice([0, 7, 8])
        water = random.choice([1, 6, 9])
        shelf = random.choice([2, 3, 4, 5])
    return int(bg), int(water), int(shelf)


def _build_candidate() -> np.ndarray | None:
    num_sources = random.choice([2, 3])
    heights_pool = [1, 2, 3]
    if num_sources == 2:
        heights = random.sample(heights_pool, 2)
    else:
        heights = random.sample(heights_pool, 3)

    lane_gap = random.randint(2, 3)
    widths = _bounded_counts(random.randint(num_sources * 6, num_sources * 8), num_sources, 6, 8)
    if widths is None:
        return None
    layout_w = sum(widths) + lane_gap * (num_sources - 1)
    if layout_w > 30:
        return None
    W = random.randint(layout_w, 30)

    max_height = max(heights)
    min_H = max(16, 6 + 4 * (max_height + 1))
    H = random.randint(min_H, 30)

    zones: list[tuple[int, int, int]] = []
    c0 = random.randint(0, W - layout_w)
    for w in widths:
        L = c0
        R = c0 + w - 1
        source_col = random.randint(L + 1, R - 1)
        zones.append((L, R, source_col))
        c0 = R + 1 + lane_gap

    bg, water, shelf = _colors_for_generation()
    grid = np.full((H, W), bg, dtype=int)
    shelf_cells: set[tuple[int, int]] = set()
    shelves: list[tuple[int, int, int]] = []
    reserved_bg: set[tuple[int, int]] = set()

    for idx, (L, R, source_col) in enumerate(zones):
        h = int(heights[idx])
        for r in range(h):
            grid[r, source_col] = water
            reserved_bg.add((r, source_col))
        if not _plan_mainline(H, L, R, h - 1, source_col, h, shelf_cells, shelves, reserved_bg):
            return None

    _add_distractors(H, W, shelf_cells, shelves, reserved_bg)
    for r, a, b in shelves:
        grid[r, a : b + 1] = shelf
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out, events = _simulate(grid, with_events=True)
    except Exception:
        return False

    bg, water, shelf = _find_colors(grid)
    if np.array_equal(grid, out):
        return False
    if int(np.sum(out == water)) <= int(np.sum(grid == water)):
        return False
    if len(set(events["reservoir_heights"].values())) < 2:
        return False

    shelf_mask = grid == shelf
    if not np.array_equal(out[shelf_mask], grid[shelf_mask]):
        return False

    split_count = Counter(int(src) for src, *_rest in events["splits"])
    pool_count = Counter(int(src) for src, *_rest in events["pools"])
    for src, h in events["reservoir_heights"].items():
        if int(split_count[int(src)]) < int(h):
            return False
        if int(pool_count[int(src)]) < 1:
            return False

    H, W = grid.shape
    for r in range(H):
        c = 0
        while c < W:
            if int(grid[r, c]) != int(shelf):
                c += 1
                continue
            c0 = c
            while c + 1 < W and int(grid[r, c + 1]) == int(shelf):
                c += 1
            for rr in range(max(0, r - 1), min(H, r + 2)):
                for cc in range(max(0, c0 - 1), min(W, c + 2)):
                    if rr == r and c0 <= cc <= c:
                        continue
                    if int(grid[rr, cc]) == int(shelf):
                        return False
            c += 1

    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_candidate()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate valid reservoir-budget waterfall input")
