from __future__ import annotations
from collections import Counter, deque
import numpy as np
from arc_utils import STRUCT8, components_mask
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate water falling from top-edge sources, pooling on shelves, and cascading off edges.

    Input structure:
    - bg = most common color; water = rarest non-bg; shelf = other non-bg.
    - Sources: water clusters touching top row. Shelves: horizontal segments.

    Rule:
    - Water falls down from sources, painting bg cells as water.
    - On hitting a shelf, it spreads across the shelf span plus one cell past each end.
    - From those overflow positions, water resumes falling independently.

    Key insight: Each shelf splits a stream into two at its edges, creating a branching waterfall tree.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non = [int(c) for c in cnt.keys() if int(c) != int(bg)]
    assert not (not non)
    water = int(min(non, key=lambda c: (int(cnt[int(c)]), int(c))))
    other = [int(c) for c in non if int(c) != int(water)]
    assert not (not other)
    shelf = int(other[0])

    out = g.copy()

    # sources: 8-connected components of water that touch the top edge; start from their bottommost row
    sources: list[tuple[int, int]] = []
    for coords in components_mask(g == int(water), connectivity=8):
        if coords.size == 0:
            continue
        if not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))

    # shelf segments (from original grid g)
    shelf_segs: list[list[tuple[int, int]]] = [[] for _ in range(H)]
    for r in range(H):
        c = 0
        while c < W:
            if int(g[int(r), int(c)]) != int(shelf):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < W and int(g[int(r), int(c + 1)]) == int(shelf):
                c += 1
            shelf_segs[int(r)].append((int(c0), int(c)))
            c += 1

    def find_shelf_seg(r: int, c: int) -> tuple[int, int] | None:
        for a, b in shelf_segs[int(r)]:
            if int(a) <= int(c) <= int(b):
                return (int(a), int(b))
        return None

    def spread_limits(row: int, start_col: int, min_col: int, max_col: int) -> tuple[int, int]:
        """
        Expand horizontally only through contiguous bg cells on the current row.

        This prevents water from "jumping" over blockers that sit on the spread row.
        """
        left = int(start_col)
        while left - 1 >= int(min_col) and int(g[int(row), int(left - 1)]) == int(bg):
            left -= 1

        right = int(start_col)
        while right + 1 <= int(max_col) and int(g[int(row), int(right + 1)]) == int(bg):
            right += 1

        return (int(left), int(right))

    seen: set[tuple[int, int]] = set()
    q = deque((int(r), int(c)) for r, c in sources)

    while q:
        r, c = q.popleft()
        if (int(r), int(c)) in seen:
            continue
        seen.add((int(r), int(c)))

        rr, cc = int(r), int(c)
        out[rr, cc] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= H:
                break
            below = int(g[int(nr), int(cc)])

            if below == int(shelf):
                seg = find_shelf_seg(nr, cc)
                if seg is None:
                    break
                a, b = seg
                left_bound = int(max(0, int(a - 1)))
                right_bound = int(min(int(W - 1), int(b + 1)))
                left, right = spread_limits(rr, cc, left_bound, right_bound)
                for x in range(left, right + 1):
                    if int(g[int(rr), int(x)]) == int(bg):
                        out[int(rr), int(x)] = int(water)
                for nc in {left, right}:
                    if int(nc) != int(cc):
                        q.append((int(rr), int(nc)))
                break

            rr = nr
            out[int(rr), int(cc)] = int(water)

    return out

# --------------- generator (cascade tree top-down construction) ---------------

def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for the waterfall cascade task."""
    BG, WATER, SHELF = 7, 6, 2

    H = random.randint(8, 30)
    W = random.randint(5, 30)

    num_sources = random.choice([1, 2]) if W >= 14 else 1
    if num_sources == 2:
        src1 = random.randint(0, W - 8)
        src2 = random.randint(src1 + 7, W - 1)
        source_cols = [src1, src2]
    else:
        source_cols = [random.randint(0, W - 1)]

    # Track cells reserved for shelf
    shelf_cells: set[tuple[int, int]] = set()
    # Cells that must remain bg (overflow positions, fall paths, source positions)
    reserved_bg: set[tuple[int, int]] = set()

    # Reserve source positions
    for sc in source_cols:
        reserved_bg.add((0, sc))
        reserved_bg.add((1, sc))

    all_shelves: list[tuple[int, int, int]] = []  # (row, a, b)

    def try_add_shelf(r: int, a: int, b: int, is_cascade: bool = False) -> bool:
        """Try to add a shelf segment. Check no conflict with existing shelves or reserved bg cells."""
        cells = [(r, c) for c in range(a, b + 1)]
        for cell in cells:
            if cell in shelf_cells or cell in reserved_bg:
                return False
        for cell in cells:
            shelf_cells.add(cell)
        all_shelves.append((r, a, b))
        return True

    def reserve_overflow_and_fall(spread_row: int, left: int, right: int, incoming_col: int):
        """Reserve overflow positions and their fall paths as must-be-bg."""
        for nc in {left, right}:
            if nc != incoming_col:
                # The overflow position itself must be bg
                reserved_bg.add((spread_row, nc))
                # Fall path below must be bg until next shelf or grid bottom
                # We'll just reserve the overflow position for now;
                # fall path conflicts are handled by checking during shelf placement

    max_depth = random.randint(2, 5)
    if H < max_depth * 2 + 3:
        max_depth = max(2, (H - 3) // 2)

    # Build cascade tree top-down
    # Each stream: (current_row, col, depth)
    streams: list[tuple[int, int, int]] = [(1, sc, 0) for sc in source_cols]
    cascade_shelves: list[tuple[int, int, int]] = []
    processed = set()

    while streams:
        next_streams = []
        random.shuffle(streams)
        for (row, col, depth) in streams:
            if (row, col) in processed:
                continue
            processed.add((row, col))

            if depth >= max_depth or row + 2 >= H:
                continue

            min_shelf_row = row + 2
            max_shelf_row = min(H - 1, row + 8)
            if min_shelf_row > max_shelf_row:
                continue

            # Try a few times to place a shelf
            placed = False
            for _try in range(10):
                shelf_row = random.randint(min_shelf_row, max_shelf_row)
                spread_row = shelf_row - 1
                if spread_row <= row:
                    continue

                # Check fall path from (row+1, col) to (shelf_row-1, col) is clear
                blocked = False
                for r_check in range(row + 1, shelf_row):
                    if (r_check, col) in shelf_cells:
                        blocked = True
                        break
                if blocked:
                    continue

                # Generate shelf segment containing col
                shelf_len = random.randint(2, min(8, W))
                max_start = col
                min_start = max(0, col - shelf_len + 1)
                if min_start > max_start:
                    continue
                a = random.randint(min_start, max_start)
                b = min(a + shelf_len - 1, W - 1)

                # Check shelf cells don't conflict
                conflict = False
                for c in range(a, b + 1):
                    if (shelf_row, c) in shelf_cells or (shelf_row, c) in reserved_bg:
                        conflict = True
                        break
                if conflict:
                    continue

                # Compute overflow positions
                left = max(0, a - 1)
                right = min(W - 1, b + 1)

                # Check overflow positions on spread row are not shelf
                overflow_ok = True
                for nc in {left, right}:
                    if nc != col:
                        if (spread_row, nc) in shelf_cells:
                            overflow_ok = False
                            break
                if not overflow_ok:
                    continue

                # Add the shelf
                if not try_add_shelf(shelf_row, a, b, is_cascade=True):
                    continue

                cascade_shelves.append((shelf_row, a, b))

                # Reserve overflow positions and fall path cells as bg
                reserve_overflow_and_fall(spread_row, left, right, col)

                # Reserve the fall path from current position to shelf
                for r_res in range(row + 1, shelf_row):
                    reserved_bg.add((r_res, col))

                # Add new streams from overflow
                for nc in {left, right}:
                    if nc != col:
                        next_streams.append((spread_row, nc, depth + 1))

                placed = True
                break

            # If no shelf placed, stream just falls to bottom (that's ok)

        streams = next_streams

    if len(cascade_shelves) < 3:
        return None

    # Build the grid
    grid = np.full((H, W), BG, dtype=int)

    # Place sources
    for sc in source_cols:
        grid[0, sc] = WATER
        grid[1, sc] = WATER

    # Place all shelves
    for (r, a, b) in all_shelves:
        for c in range(a, b + 1):
            grid[r, c] = SHELF

    # Add 0-5 distractor shelves (carefully avoiding reserved_bg and shelf_cells)
    num_distractors = random.randint(0, 5)
    for _ in range(num_distractors):
        for _attempt in range(20):
            dr = random.randint(1, H - 1)
            dl = random.randint(1, min(8, W))
            da = random.randint(0, W - dl)
            db = da + dl - 1
            # Check no conflict with shelf_cells or reserved_bg
            conflict = False
            for c in range(da, db + 1):
                if (dr, c) in shelf_cells or (dr, c) in reserved_bg:
                    conflict = True
                    break
            if not conflict:
                # Also check distractor doesn't intersect any active water fall path
                # by ensuring it's not on a column where water falls through
                if try_add_shelf(dr, da, db):
                    for c in range(da, db + 1):
                        grid[dr, c] = SHELF
                    break

    # Verify with solver
    try:
        out = main(grid.copy())
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    # Verify shelf cells are preserved in output
    for (r, a, b) in all_shelves:
        for c in range(a, b + 1):
            if out[r, c] != SHELF:
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
