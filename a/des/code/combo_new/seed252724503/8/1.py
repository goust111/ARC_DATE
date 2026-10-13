import random

import numpy as np


BG = 0
FRAME = 1
GRAY = 5
FILL_POOL = (2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for rr in range(r0, r1 + 1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        for cc in range(c0, c1 + 1):
            expected.add((r0, cc))
            expected.add((r1, cc))
        if set(comp) == expected:
            frames.append((r0, c0, r1, c1))
    return frames


def _parse_top_legend(grid: np.ndarray) -> list[tuple[int, int]]:
    row = grid[0]
    w = int(grid.shape[1])
    items: list[tuple[int, int]] = []
    c = 0
    while c < w:
        while c < w and int(row[c]) == BG:
            c += 1
        if c >= w:
            break
        color = int(row[c])
        if color in (BG, FRAME, GRAY):
            return []
        j = c + 1
        count = 0
        while j < w and int(row[j]) == GRAY:
            count += 1
            j += 1
        if count == 0:
            return []
        items.append((count, color))
        if j < w and int(row[j]) != BG:
            return []
        c = j + 1
    if len(items) < 2:
        return []
    if len({count for count, _ in items}) != len(items):
        return []
    return items


def _parse_left_legend(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    items: list[tuple[int, int]] = []
    r = 0
    while r < h:
        while r < h and int(grid[r, 0]) == BG:
            r += 1
        if r >= h:
            break
        color = int(grid[r, 0])
        if color in (BG, FRAME, GRAY):
            return []
        c = 1
        count = 0
        while c < w and int(grid[r, c]) == GRAY:
            count += 1
            c += 1
        if count == 0:
            return []
        if c < w and bool(np.any(grid[r, c:] != BG)):
            return []
        items.append((count, color))
        r += 1
    if len(items) < 2:
        return []
    if len({count for count, _ in items}) != len(items):
        return []
    return items


def _read_legend(grid: np.ndarray) -> dict[int, int]:
    candidates: list[list[tuple[int, int]]] = []
    top = _parse_top_legend(grid)
    if top:
        candidates.append(top)
    left = _parse_left_legend(grid)
    if left:
        candidates.append(left)
    assert candidates, "expected a legend on the top or left edge"
    items = max(candidates, key=lambda xs: (len(xs), sum(count for count, _ in xs)))
    return {count: color for count, color in items}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue enclosed rectangle with the legend color keyed by its interior gray-dot count.

    Input structure:
    - The background is black.
    - A legend sits on the top edge or left edge. Each legend item has one colored square and
      an adjacent run of gray pixels; different items use different run lengths.
    - Elsewhere on the grid, blue pixels form hollow rectangular frames. Each frame interior
      contains some gray pixels scattered on black.

    Rule:
    - Read the legend as a mapping from "number of gray pixels" to "fill color".
    - Find every blue hollow rectangle and count the gray pixels strictly inside it.
    - Replace the entire interior of that rectangle, including both gray and black cells, with
      the legend color whose key matches the count.
    - Keep the blue border and the legend unchanged.

    Key insight:
    - The gray pixels are count encodings. First infer a count-to-color lookup table from the
      legend, then apply the same counting rule inside each independent enclosed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _read_legend(g)
    frames = _find_frames(g)
    assert frames, "expected at least one blue rectangular frame"

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        dot_count = int(np.sum(interior == GRAY))
        assert dot_count in mapping, "every frame count must appear in the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = int(mapping[dot_count])
    return out


def _top_legend_width(counts: list[int]) -> int:
    return sum(1 + count for count in counts) + max(0, len(counts) - 1)


def _left_legend_height(counts: list[int]) -> int:
    return 2 * len(counts) - 1


def _draw_top_legend(grid: np.ndarray, counts: list[int], mapping: dict[int, int], start_col: int) -> None:
    c = start_col
    for idx, count in enumerate(counts):
        grid[0, c] = int(mapping[count])
        grid[0, c + 1 : c + 1 + count] = GRAY
        c += 1 + count
        if idx != len(counts) - 1:
            c += 1


def _draw_left_legend(grid: np.ndarray, counts: list[int], mapping: dict[int, int], start_row: int) -> None:
    r = start_row
    for count in counts:
        grid[r, 0] = int(mapping[count])
        grid[r, 1 : 1 + count] = GRAY
        r += 2


def _frame_dims_for_count(count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for inner_h in range(2, 7):
        for inner_w in range(2, 7):
            if inner_h * inner_w >= count + 1:
                dims.append((inner_h + 2, inner_w + 2))
    return dims


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(int(occupied.shape[0]), r0 + h + 1)
    cc1 = min(int(occupied.shape[1]), c0 + w + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, count: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME

    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for r, c in random.sample(cells, count):
        grid[r, c] = GRAY


def _attempt_generate() -> np.ndarray | None:
    legend_size = random.randint(2, 5)
    counts = list(range(1, legend_size + 1))
    colors = random.sample(FILL_POOL, legend_size)
    mapping = {count: color for count, color in zip(counts, colors)}
    orientation = random.choice(("top", "left"))
    frame_count = random.randint(2, min(5, legend_size + 1))

    if orientation == "top":
        legend_w = _top_legend_width(counts)
        h = random.randint(14, 30)
        w = random.randint(max(14, legend_w), 30)
        grid = np.zeros((h, w), dtype=int)
        start_col = random.randint(0, w - legend_w)
        _draw_top_legend(grid, counts, mapping, start_col)
        min_frame_row = 2
    else:
        legend_h = _left_legend_height(counts)
        start_row = random.randint(0, 2)
        min_h = start_row + legend_h + 6
        legend_w = 1 + counts[-1]
        h = random.randint(max(14, min_h), 30)
        w = random.randint(max(14, legend_w + 4), 30)
        grid = np.zeros((h, w), dtype=int)
        _draw_left_legend(grid, counts, mapping, start_row)
        min_frame_row = start_row + legend_h + 1

    if min_frame_row >= h - 4:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:min_frame_row, :] = True

    frame_counts = [random.choice(counts) for _ in range(frame_count)]
    if frame_count > 1 and len(set(frame_counts)) == 1 and len(counts) > 1:
        idx = random.randrange(frame_count)
        alternatives = [count for count in counts if count != frame_counts[idx]]
        frame_counts[idx] = random.choice(alternatives)
    random.shuffle(frame_counts)

    placed: list[tuple[int, int, int, int, int]] = []
    for count in frame_counts:
        dims = _frame_dims_for_count(count)
        success = False
        for _ in range(PLACE_RETRIES):
            fh, fw = random.choice(dims)
            if min_frame_row > h - fh - 1 or w - fw - 2 < 0:
                continue
            r0 = random.randint(min_frame_row, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if not _can_place(occupied, r0, c0, fh, fw):
                continue
            _paint_frame(grid, r0, c0, fh, fw, count)
            occupied[r0 : r0 + fh, c0 : c0 + fw] = True
            placed.append((r0, c0, fh, fw, count))
            success = True
            break
        if not success:
            return None

    expected = grid.copy()
    for r0, c0, fh, fw, count in placed:
        expected[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = int(mapping[count])

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
