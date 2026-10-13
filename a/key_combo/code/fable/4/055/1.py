import bisect
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)

RIGHT_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
LEFT_TURN = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    if ar == br and ac != bc:
        return RIGHT if bc > ac else LEFT
    if ac == bc and ar != br:
        return DOWN if br > ar else UP
    raise AssertionError("expected distinct axis-aligned points")


def _turn(incoming: tuple[int, int], sign: int) -> tuple[int, int]:
    return RIGHT_TURN[incoming] if sign < 0 else LEFT_TURN[incoming]


def _color_to_turn_sign(color: int) -> int:
    if color == YELLOW:
        return -1
    if color == RED:
        return 1
    raise AssertionError(f"unexpected corner color: {color}")


def _build_visibility(points: list[tuple[int, int]]):
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in points:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)
    for values in rows.values():
        values.sort()
    for values in cols.values():
        values.sort()
    return rows, cols


def _nearest_corner(
    point: tuple[int, int],
    direction: tuple[int, int],
    rows: dict[int, list[int]],
    cols: dict[int, list[int]],
) -> tuple[int, int] | None:
    r, c = point
    dr, dc = direction
    if dr == 0:
        options = rows[r]
        if dc > 0:
            idx = bisect.bisect_right(options, c)
            return None if idx >= len(options) else (r, options[idx])
        idx = bisect.bisect_left(options, c) - 1
        return None if idx < 0 else (r, options[idx])
    options = cols[c]
    if dr > 0:
        idx = bisect.bisect_right(options, r)
        return None if idx >= len(options) else (options[idx], c)
    idx = bisect.bisect_left(options, r) - 1
    return None if idx < 0 else (options[idx], c)


def _canonical_cycle(cycle: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    vertices = cycle[:-1]
    candidates = []
    for seq in (vertices, list(reversed(vertices))):
        for shift in range(len(seq)):
            candidates.append(tuple(seq[shift:] + seq[:shift]))
    return min(candidates)


def _trace_candidate(
    start: tuple[int, int],
    incoming0: tuple[int, int],
    turn_signs: dict[tuple[int, int], int],
    rows: dict[int, list[int]],
    cols: dict[int, list[int]],
) -> list[tuple[int, int]] | None:
    total = len(turn_signs)
    current = start
    incoming = incoming0
    seen_vertices = {start}
    occupied = {start}
    cycle = [start]

    while len(cycle) <= total:
        outgoing = _turn(incoming, turn_signs[current])
        nxt = _nearest_corner(current, outgoing, rows, cols)
        if nxt is None:
            return None

        segment = _cells_between(current, nxt)
        if nxt == start:
            if len(seen_vertices) != total:
                return None
            if outgoing != incoming0:
                return None
            if any(cell in occupied for cell in segment[1:-1]):
                return None
            return cycle + [start]

        if nxt in seen_vertices:
            return None
        if any(cell in occupied for cell in segment[1:]):
            return None

        occupied.update(segment[1:])
        seen_vertices.add(nxt)
        cycle.append(nxt)
        current = nxt
        incoming = outgoing

    return None


def _reconstruct_cycle(points: list[tuple[int, int]], colors: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    assert points, "expected at least one corner"
    rows, cols = _build_visibility(points)
    turn_signs = {point: _color_to_turn_sign(colors[point]) for point in points}

    solutions: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for start in sorted(points):
        for incoming0 in DIRS:
            cycle = _trace_candidate(start, incoming0, turn_signs, rows, cols)
            if cycle is None:
                continue
            solutions[_canonical_cycle(cycle)] = cycle

    assert len(solutions) == 1, "corner set must determine a unique loop"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the hidden orthogonal loop by connecting every colored corner in order.

    Input structure:
    - The background is black.
    - Every non-background cell is a single corner marker: yellow for convex corners
      and red for concave corners of one hidden one-pixel-wide orthogonal loop.
    - Straight path pixels are missing; multiple corners may share rows or columns.

    Rule:
    - Recover the unique simple closed loop that visits every corner exactly once,
      turns only at those corners, and admits a traversal where yellow corners are
      right turns while red corners are left turns.
    - Keep the original corner colors.
    - Fill every straight pixel between consecutive corners with grey.

    Key insight:
    - Once the traversal turn direction is fixed by each corner color, every corner's
      outgoing segment must continue to the nearest corner in the required row/column
      direction; the unique cycle is the only globally non-self-intersecting trace.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, YELLOW}), "unexpected colors"

    points = [tuple(map(int, rc)) for rc in np.argwhere(g != BLACK)]
    assert len(points) >= 4, "expected at least four corners"

    colors = {point: int(g[point]) for point in points}
    cycle = _reconstruct_cycle(points, colors)

    out = g.copy()
    for a, b in zip(cycle, cycle[1:]):
        for r, c in _cells_between(a, b)[1:-1]:
            out[r, c] = GREY
    return out


def _append_vertex(vertices: list[tuple[int, int]], point: tuple[int, int]) -> None:
    if not vertices or vertices[-1] != point:
        vertices.append(point)


def _sample_side_intervals(span: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    if span < count * 2 + (count + 1) * 2:
        return None

    candidates = [(a, b) for a in range(2, span - 3) for b in range(a + 2, span - 1)]
    if len(candidates) < count:
        return None

    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if span - picks[-1][1] < 2:
            continue
        ok = True
        prev_end = 0
        for idx, (a, b) in enumerate(picks):
            if idx == 0 and a < 2:
                ok = False
                break
            if a - prev_end < 2:
                ok = False
                break
            prev_end = b
        if ok:
            return picks
    return None


def _build_vertices(
    height: int,
    width: int,
    top: list[tuple[int, int, int]],
    right: list[tuple[int, int, int]],
    bottom: list[tuple[int, int, int]],
    left: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    vertices: list[tuple[int, int]] = [(0, 0)]

    for a, b, depth in sorted(top):
        _append_vertex(vertices, (0, a))
        _append_vertex(vertices, (depth, a))
        _append_vertex(vertices, (depth, b))
        _append_vertex(vertices, (0, b))
    _append_vertex(vertices, (0, width))

    for a, b, depth in sorted(right):
        _append_vertex(vertices, (a, width))
        _append_vertex(vertices, (a, width - depth))
        _append_vertex(vertices, (b, width - depth))
        _append_vertex(vertices, (b, width))
    _append_vertex(vertices, (height, width))

    for a, b, depth in sorted(bottom, reverse=True):
        _append_vertex(vertices, (height, b))
        _append_vertex(vertices, (height - depth, b))
        _append_vertex(vertices, (height - depth, a))
        _append_vertex(vertices, (height, a))
    _append_vertex(vertices, (height, 0))

    for a, b, depth in sorted(left, reverse=True):
        _append_vertex(vertices, (b, 0))
        _append_vertex(vertices, (b, depth))
        _append_vertex(vertices, (a, depth))
        _append_vertex(vertices, (a, 0))

    return vertices


def _turn_counts(vertices: list[tuple[int, int]]) -> dict[int, int] | None:
    counts = {-1: 0, 1: 0}
    n = len(vertices)
    for i, cur in enumerate(vertices):
        prev = vertices[i - 1]
        nxt = vertices[(i + 1) % n]
        incoming = _direction(prev, cur)
        outgoing = _direction(cur, nxt)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            return None
        counts[-1 if cross < 0 else 1] += 1
    return counts


def _vertices_to_grids(vertices: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(vertices) < 4 or len(set(vertices)) != len(vertices):
        return None

    occupied = {vertices[0]}
    for idx, (a, b) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        if _direction(a, b) not in DIRS:
            return None
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 2:
            return None
        segment = _cells_between(a, b)
        body = segment[1:-1] if idx == len(vertices) - 1 else segment[1:]
        if any(cell in occupied for cell in body):
            return None
        occupied.update(body)

    counts = _turn_counts(vertices)
    if counts is None or abs(counts[-1] - counts[1]) != 4:
        return None

    corners = vertices
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return None

    height = max(r for r, _ in occupied) + 1
    width = max(c for _, c in occupied) + 1
    inp = np.full((height, width), BLACK, dtype=int)
    out = np.full((height, width), BLACK, dtype=int)

    for r, c in occupied:
        out[r, c] = GREY

    convex_sign = -1 if counts[-1] > counts[1] else 1
    n = len(vertices)
    for i, cur in enumerate(vertices):
        prev = vertices[i - 1]
        nxt = vertices[(i + 1) % n]
        incoming = _direction(prev, cur)
        outgoing = _direction(cur, nxt)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        sign = -1 if cross < 0 else 1
        color = YELLOW if sign == convex_sign else RED
        inp[cur] = color
        out[cur] = color

    return inp, out


def _transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    rotations = random.randrange(4)
    if rotations:
        inp = np.rot90(inp, rotations)
        out = np.rot90(out, rotations)
    return inp.copy(), out.copy()


def _embed(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    final_h = h + random.randint(0, min(6, 30 - h))
    final_w = w + random.randint(0, min(6, 30 - w))
    top = random.randint(0, final_h - h)
    left = random.randint(0, final_w - w)
    canvas = np.full((final_h, final_w), BLACK, dtype=int)
    canvas[top : top + h, left : left + w] = grid
    return canvas


def _gen_side_notches(span: int, depth_cap: int, count: int) -> list[tuple[int, int, int]] | None:
    intervals = _sample_side_intervals(span, count)
    if intervals is None:
        return None
    return [(a, b, random.randint(2, depth_cap)) for a, b in intervals]


def _gen_attempt() -> np.ndarray | None:
    outer_h = random.randint(8, 16)
    outer_w = random.randint(8, 16)
    top_bottom_depth = min(6, outer_h - 4)
    left_right_depth = min(6, outer_w - 4)
    if top_bottom_depth < 2 or left_right_depth < 2:
        return None

    total_notches = random.choices([0, 1, 2, 3, 4, 5], weights=[1, 3, 4, 4, 2, 1])[0]
    counts = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    side_order = ["top", "right", "bottom", "left"]
    for _ in range(total_notches):
        choices = [side for side in side_order if counts[side] < 2]
        counts[random.choice(choices)] += 1

    if total_notches == 0 and random.random() < 0.8:
        counts[random.choice(side_order)] = 1

    top = _gen_side_notches(outer_w, top_bottom_depth, counts["top"])
    right = _gen_side_notches(outer_h, left_right_depth, counts["right"])
    bottom = _gen_side_notches(outer_w, top_bottom_depth, counts["bottom"])
    left = _gen_side_notches(outer_h, left_right_depth, counts["left"])
    if None in (top, right, bottom, left):
        return None

    vertices = _build_vertices(outer_h, outer_w, top, right, bottom, left)
    pair = _vertices_to_grids(vertices)
    if pair is None:
        return None
    inp, out = pair
    inp, out = _transform_pair(inp, out)
    inp = _embed(inp)
    out = _embed(out)

    # Preserve a shared placement after padding.
    if inp.shape != out.shape:
        pad_h = max(inp.shape[0], out.shape[0])
        pad_w = max(inp.shape[1], out.shape[1])
        shared_inp = np.full((pad_h, pad_w), BLACK, dtype=int)
        shared_out = np.full((pad_h, pad_w), BLACK, dtype=int)
        shared_inp[: inp.shape[0], : inp.shape[1]] = inp
        shared_out[: out.shape[0], : out.shape[1]] = out
        inp, out = shared_inp, shared_out

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid loop-reconstruction input")
