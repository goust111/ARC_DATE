import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _get_bg(grid)
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if len(rows) and len(cols):
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one full-row/full-column cross"
    axis_color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "cross must have one row axis and one column axis"
    return axis_color, int(rows[0]), int(cols[0])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _trace_cycle(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        next_cells = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(next_cells) == 1, (cur, next_cells)
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _classify_cycle(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    straight = set()
    turns: dict[tuple[int, int], int] = {}
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            straight.add((r, c))
            continue
        sign = 1 if cross > 0 else -1
        turns[(r, c)] = sign
        counts[sign] += 1

    assert counts[1] > 0 and counts[-1] > 0, counts
    assert counts[1] != counts[-1], counts
    convex_sign = 1 if counts[1] > counts[-1] else -1
    convex = {coord for coord, sign in turns.items() if sign == convex_sign}
    concave = set(turns) - convex
    assert len(convex) >= 2 and len(concave) >= 1
    return straight, convex, concave


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect the traced loop into the other quadrants while suppressing concave bends.

    Input structure:
    - The grid has a black background, one full blue row, and one full blue column forming a cross.
    - Exactly one quadrant contains a one-pixel-wide 4-connected green orthogonal cycle away from the cross.
    - Every cycle pixel has degree 2, so the object is a closed loop rather than a branch or filled region.

    Rule:
    - Locate the blue mirror axes and trace the green cycle in the populated quadrant.
    - Classify each cycle pixel as straight, convex bend, or concave bend using its local turn and the majority/minority turn sign on the loop.
    - Keep straight pixels green and reflect them across the row axis, column axis, and both axes; keep convex bends green in place but draw their three reflected copies in yellow; recolor concave bends red and do not reflect them.

    Key insight:
    - The output is not a plain quadrant copy: the loop must be traced first so that only straight pixels and majority-turn bends participate in the symmetry completion.
    """
    g = np.asarray(grid, dtype=int)
    axis_color, axis_r, axis_c = _find_cross_axes(g)

    cycle_mask = g == GREEN
    cycle_mask[axis_r, :] = False
    cycle_mask[:, axis_c] = False
    components = _connected_components(cycle_mask)
    assert len(components) == 1, "expected one off-axis green cycle"
    path = _trace_cycle(components[0])
    straight, convex, concave = _classify_cycle(path)

    out = g.copy()
    for r, c in concave:
        out[r, c] = RED

    for coords, mirror_color in ((straight, GREEN), (convex, YELLOW)):
        for r, c in coords:
            for rr, cc in ((2 * axis_r - r, c), (r, 2 * axis_c - c), (2 * axis_r - r, 2 * axis_c - c)):
                if not _in_bounds(g.shape, rr, cc):
                    continue
                if rr == axis_r or cc == axis_c:
                    continue
                out[rr, cc] = mirror_color

    out[axis_r, :] = axis_color
    out[:, axis_c] = axis_color
    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _choose_interval(width: int) -> tuple[int, int] | None:
    if width < 6:
        return None
    span = random.randint(2, max(2, width - 4))
    start = random.randint(1, width - span - 2)
    return start, start + span - 1


def _disjoint_intervals(width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(50):
        first = _choose_interval(width)
        second = _choose_interval(width)
        if first is None or second is None:
            return None
        (a0, a1), (b0, b1) = first, second
        if a1 + 1 < b0 or b1 + 1 < a0:
            return first, second
    return None


def _make_cycle_mask(height: int, width: int) -> np.ndarray | None:
    if height < 5 or width < 6:
        return None

    mode = random.choices(["top", "bottom", "both"], weights=[3, 3, 2])[0]
    top = [0] * width
    bottom = [height - 1] * width

    if mode == "top":
        interval = _choose_interval(width)
        if interval is None:
            return None
        start, end = interval
        depth = random.randint(1, max(1, (height - 3) // 2))
        for c in range(start, end + 1):
            top[c] = depth
    elif mode == "bottom":
        interval = _choose_interval(width)
        if interval is None:
            return None
        start, end = interval
        depth = random.randint(1, max(1, (height - 3) // 2))
        for c in range(start, end + 1):
            bottom[c] = height - 1 - depth
    else:
        intervals = _disjoint_intervals(width)
        if intervals is None:
            return None
        (t0, t1), (b0, b1) = intervals
        top_depth = random.randint(1, max(1, (height - 4) // 2))
        bottom_depth_max = min(max(1, (height - 4) // 2), height - 3 - top_depth)
        if bottom_depth_max < 1:
            return None
        bottom_depth = random.randint(1, bottom_depth_max)
        for c in range(t0, t1 + 1):
            top[c] = top_depth
        for c in range(b0, b1 + 1):
            bottom[c] = height - 1 - bottom_depth

    if any(top[c] + 2 > bottom[c] for c in range(width)):
        return None

    path = _path_from_profiles(top, bottom)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in path:
        mask[r, c] = True

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=random.randrange(4))

    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    try:
        traced = _trace_cycle(coords)
        straight, convex, concave = _classify_cycle(traced)
    except AssertionError:
        return None
    if not straight or len(convex) < 2 or len(concave) < 1:
        return None
    return mask


def _quadrant_bounds(shape: tuple[int, int], axis_r: int, axis_c: int, quadrant: int) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    if quadrant == 0:
        return (1, axis_r - 2), (1, axis_c - 2)
    if quadrant == 1:
        return (1, axis_r - 2), (axis_c + 2, w - 2)
    if quadrant == 2:
        return (axis_r + 2, h - 2), (1, axis_c - 2)
    return (axis_r + 2, h - 2), (axis_c + 2, w - 2)


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[top + rr, left + cc] = color


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(15, 23)
        w = random.randint(15, 23)
        axis_r = random.randint(6, h - 7)
        axis_c = random.randint(6, w - 7)
        quadrant = random.randrange(4)

        row_bounds, col_bounds = _quadrant_bounds((h, w), axis_r, axis_c, quadrant)
        max_loop_h = row_bounds[1] - row_bounds[0] + 1
        max_loop_w = col_bounds[1] - col_bounds[0] + 1
        if max_loop_h < 5 or max_loop_w < 6:
            continue

        loop_h = random.randint(5, min(10, max_loop_h))
        loop_w = random.randint(6, min(12, max_loop_w))
        mask = _make_cycle_mask(loop_h, loop_w)
        if mask is None:
            continue

        mask_h, mask_w = mask.shape
        max_top = row_bounds[1] - mask_h + 1
        max_left = col_bounds[1] - mask_w + 1
        if max_top < row_bounds[0] or max_left < col_bounds[0]:
            continue
        top = random.randint(row_bounds[0], max_top)
        left = random.randint(col_bounds[0], max_left)

        grid = np.full((h, w), BLACK, dtype=int)
        grid[axis_r, :] = BLUE
        grid[:, axis_c] = BLUE
        _place_mask(grid, mask, top, left, GREEN)

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        if not np.any(out == RED) or not np.any(out == YELLOW):
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
