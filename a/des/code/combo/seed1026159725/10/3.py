import random
from collections import Counter

import numpy as np


RED = 2
BLUE = 1
GREY = 5
ORANGE = 7

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}

PASSABLE = {RED, BLUE, GREY, ORANGE}
GEN_COLOR_POOL = [0, 3, 4, 6, 8, 9]

STEP = 4
MACRO_RANGE = (4, 7)
EDGE_COUNT_RANGE = (4, 8)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _passable_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in PASSABLE:
            out.append((int(nr), int(nc)))
    return out


def _node_turn(grid: np.ndarray, r: int, c: int, bg: int) -> int | None:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return None

    cross = [int(grid[r, c]), int(grid[r - 1, c]), int(grid[r + 1, c]), int(grid[r, c - 1]), int(grid[r, c + 1])]
    if any(v not in PASSABLE for v in cross):
        return None

    tl = int(grid[r - 1, c - 1])
    tr = int(grid[r - 1, c + 1])
    bl = int(grid[r + 1, c - 1])
    br = int(grid[r + 1, c + 1])
    if tl == tr == bl == br == bg:
        return None

    if tl == bl == br != tr and tl not in PASSABLE and tr not in PASSABLE and tl != bg and tr != bg:
        return 1
    if tr == bl == br != tl and tl not in PASSABLE and tr not in PASSABLE and tl != bg and tr != bg:
        return -1
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey network from the red start to the blue end.

    At decorated 3x3 node intersections, the odd-colored top corner gives the turn:
    top-right means clockwise, top-left means counterclockwise. At undecorated
    intersections, continue straight. Paint only the traversed grey path orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    starts = np.argwhere(g == RED)
    ends = np.argwhere(g == BLUE)
    assert starts.shape[0] == 1, "expected one red start"
    assert ends.shape[0] == 1, "expected one blue end"

    start = tuple(map(int, starts[0]))
    end = tuple(map(int, ends[0]))

    out = g.copy()
    prev: tuple[int, int] | None = None
    cur = start
    visited_states: set[tuple[int, int, int, int]] = set()
    paint: set[tuple[int, int]] = set()

    for _ in range(g.size * 4):
        if cur == end:
            break

        if int(g[cur]) == GREY:
            paint.add(cur)

        if prev is None:
            nbrs = _passable_neighbors(g, *cur)
            assert len(nbrs) == 1, "start must be a path endpoint"
            nxt = nbrs[0]
        else:
            incoming = (cur[0] - prev[0], cur[1] - prev[1])
            state = (cur[0], cur[1], incoming[0], incoming[1])
            assert state not in visited_states, "path traversal looped"
            visited_states.add(state)

            nbrs = _passable_neighbors(g, *cur)
            turn = _node_turn(g, cur[0], cur[1], bg)
            if turn is not None:
                idx = DIR_TO_IDX[incoming]
                dr, dc = DIRS[(idx + turn) % 4]
                nxt = (cur[0] + dr, cur[1] + dc)
                assert nxt in nbrs and nxt != prev, "node turn must lead to a valid path cell"
            elif len(nbrs) >= 3:
                nxt = (cur[0] + incoming[0], cur[1] + incoming[1])
                assert nxt in nbrs, "plain intersection must allow straight movement"
            else:
                options = [p for p in nbrs if p != prev]
                assert len(options) == 1, "non-intersection path cells must have one forward continuation"
                nxt = options[0]

        prev, cur = cur, nxt
    else:
        raise AssertionError("failed to reach the blue end")

    assert cur == end, "did not terminate on the blue end"
    for r, c in paint:
        out[r, c] = ORANGE
    return out


def _add_vec(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[0], -d[1])


def _transition_kind(prev_dir: tuple[int, int], next_dir: tuple[int, int]) -> str:
    pi = DIR_TO_IDX[prev_dir]
    ni = DIR_TO_IDX[next_dir]
    if pi == ni:
        return "straight"
    if ni == (pi + 1) % 4:
        return "cw"
    if ni == (pi - 1) % 4:
        return "ccw"
    raise AssertionError("self-avoiding walk should not reverse")


def _macro_to_pixel(x: int) -> int:
    return 2 + STEP * x


def _draw_segment(grid: np.ndarray, cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        for c in range(c0, c1 + 1):
            grid[r, c] = color
            cells.add((int(r), int(c)))
    else:
        assert a[1] == b[1]
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        for r in range(r0, r1 + 1):
            grid[r, c] = color
            cells.add((int(r), int(c)))


def _sample_walk(mh: int, mw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[str]] | None:
    max_edges = min(EDGE_COUNT_RANGE[1], mh * mw - 1)
    if max_edges < EDGE_COUNT_RANGE[0]:
        return None

    for _ in range(200):
        edge_count = random.randint(EDGE_COUNT_RANGE[0], max_edges)
        start = (random.randrange(mh), random.randrange(mw))
        vertices = [start]
        dirs: list[tuple[int, int]] = []
        used = {start}

        for _step in range(edge_count):
            cur = vertices[-1]
            candidates = []
            weights = []
            for d in DIRS:
                nxt = _add_vec(cur, d)
                if not (0 <= nxt[0] < mh and 0 <= nxt[1] < mw):
                    continue
                if nxt in used:
                    continue
                candidates.append(d)
                if dirs and d == dirs[-1]:
                    weights.append(4)
                else:
                    weights.append(2)
            if not candidates:
                break
            d = random.choices(candidates, weights=weights, k=1)[0]
            nxt = _add_vec(cur, d)
            dirs.append(d)
            vertices.append(nxt)
            used.add(nxt)

        if len(dirs) != edge_count:
            continue

        kinds = [_transition_kind(dirs[i - 1], dirs[i]) for i in range(1, len(dirs))]
        if "straight" in kinds and "cw" in kinds and "ccw" in kinds:
            return vertices, dirs, kinds
    return None


def _max_stub_length(center: tuple[int, int], d: tuple[int, int], blocked: set[tuple[int, int]], shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = center
    length = 0
    while True:
        nr = r + d[0] * (length + 1)
        nc = c + d[1] * (length + 1)
        if not (0 <= nr < h and 0 <= nc < w):
            return length
        if (nr, nc) in blocked:
            return length
        length += 1


def _paint_node(grid: np.ndarray, center: tuple[int, int], body: int, marker: int, kind: str) -> None:
    r, c = center
    if kind == "cw":
        grid[r - 1, c - 1] = body
        grid[r - 1, c + 1] = marker
    else:
        assert kind == "ccw"
        grid[r - 1, c - 1] = marker
        grid[r - 1, c + 1] = body
    grid[r + 1, c - 1] = body
    grid[r + 1, c + 1] = body


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    mh = random.randint(*MACRO_RANGE)
    mw = random.randint(*MACRO_RANGE)
    walk = _sample_walk(mh, mw)
    if walk is None:
        return None

    vertices, dirs, kinds = walk
    h = STEP * mh + 1
    w = STEP * mw + 1

    bg = random.choice(GEN_COLOR_POOL)
    node_colors = [c for c in GEN_COLOR_POOL if c != bg]
    body, marker = random.sample(node_colors, 2)

    grid = np.full((h, w), bg, dtype=int)
    main_route: set[tuple[int, int]] = set()

    pix_vertices = [(_macro_to_pixel(r), _macro_to_pixel(c)) for r, c in vertices]
    for a, b in zip(pix_vertices, pix_vertices[1:], strict=False):
        _draw_segment(grid, main_route, a, b, GREY)

    occupied = set(main_route)
    for i in range(1, len(vertices) - 1):
        center = pix_vertices[i]
        kind = kinds[i - 1]
        used_dirs = {_opp(dirs[i - 1]), dirs[i]}
        unused_dirs = [d for d in DIRS if d not in used_dirs]
        if len(unused_dirs) != 2:
            return None

        for d in unused_dirs:
            max_len = _max_stub_length(center, d, occupied, grid.shape)
            if max_len < 2:
                return None
            stub_len = 2
            end = (center[0] + d[0] * stub_len, center[1] + d[1] * stub_len)
            _draw_segment(grid, occupied, center, end, GREY)

        if kind in ("cw", "ccw"):
            _paint_node(grid, center, body, marker, kind)

    start = pix_vertices[0]
    end = pix_vertices[-1]
    grid[start] = RED
    grid[end] = BLUE

    out = grid.copy()
    for r, c in main_route:
        if out[r, c] == GREY:
            out[r, c] = ORANGE

    if _get_bg(grid) != bg:
        return None
    if np.count_nonzero(grid == RED) != 1 or np.count_nonzero(grid == BLUE) != 1:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid network puzzle")
