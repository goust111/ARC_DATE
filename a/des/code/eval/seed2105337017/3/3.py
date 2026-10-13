import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PATH = 5
MARKER_POOL = (3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _cross(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    return d1[0] * d2[1] - d1[1] * d2[0]


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    length = max(abs(r1 - r0), abs(c1 - c0))
    return [(r0 + step * dr, c0 + step * dc) for step in range(length + 1)]


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    mask = grid == PATH
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a grey path"

    degrees: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in coords.tolist():
        pos = (int(r), int(c))
        deg = len(_neighbors4(mask, pos[0], pos[1]))
        assert deg in (1, 2), "path must be a simple non-branching line"
        degrees[pos] = deg
        if deg == 1:
            endpoints.append(pos)

    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start: tuple[int, int] | None = None
    left_color: int | None = None
    right_color: int | None = None
    for endpoint in endpoints:
        (nr, nc) = _neighbors4(mask, endpoint[0], endpoint[1])[0]
        direction = (nr - endpoint[0], nc - endpoint[1])
        ldr, ldc = _left_of(direction)
        rdr, rdc = _right_of(direction)
        lp = (endpoint[0] + ldr, endpoint[1] + ldc)
        rp = (endpoint[0] + rdr, endpoint[1] + rdc)
        if not (_in_bounds(grid.shape, *lp) and _in_bounds(grid.shape, *rp)):
            continue
        pair = {int(grid[lp]), int(grid[rp])}
        if pair == {RED, BLUE}:
            start = endpoint
            left_color = int(grid[lp])
            right_color = int(grid[rp])
            break

    assert start is not None and left_color is not None and right_color is not None

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nb for nb in _neighbors4(mask, cur[0], cur[1]) if nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path walk must be unambiguous"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == int(mask.sum()), "path must be fully connected"
    return path, left_color, right_color


def _paint_if_bg(out: np.ndarray, r: int, c: int, color: int) -> None:
    if _in_bounds(out.shape, r, c) and int(out[r, c]) == BG:
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the grey path from the endpoint marked by the red/blue pair.

    For each travelled segment, paint one-cell-wide bands on the left and right
    with the current side colors. At each 90-degree turn, read the marker placed
    on the exterior diagonal: a left turn updates the left-side color, and a
    right turn updates the right-side color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    path, left_color, right_color = _trace_path(g)

    turn_updates: dict[int, tuple[int, int]] = {}
    for idx in range(1, len(path) - 1):
        prev = path[idx - 1]
        cur = path[idx]
        nxt = path[idx + 1]
        d_prev = (cur[0] - prev[0], cur[1] - prev[1])
        d_next = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_prev == d_next:
            continue
        turn = _cross(d_prev, d_next)
        assert turn in (-1, 1), "turns must be 90 degrees"
        mr = cur[0] + d_prev[0] - d_next[0]
        mc = cur[1] + d_prev[1] - d_next[1]
        assert _in_bounds(g.shape, mr, mc), "turn marker must be in bounds"
        marker_color = int(g[mr, mc])
        assert marker_color not in (BG, PATH), "turn marker must be colored"
        turn_updates[idx] = (turn, marker_color)

    cur_left = left_color
    cur_right = right_color
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        direction = (b[0] - a[0], b[1] - a[1])
        ldr, ldc = _left_of(direction)
        rdr, rdc = _right_of(direction)
        for r, c in (a, b):
            _paint_if_bg(out, r + ldr, c + ldc, cur_left)
            _paint_if_bg(out, r + rdr, c + rdc, cur_right)
        if idx + 1 in turn_updates:
            turn, marker_color = turn_updates[idx + 1]
            if turn > 0:
                cur_left = marker_color
            else:
                cur_right = marker_color

    return out


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for idx in range(1, len(path) - 1):
        d_prev = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        d_next = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if d_prev != d_next:
            indices.append(idx)
    return indices


def _build_canonical_path(height: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cols = list(range(2, width - 2, 3))
    rows = list(range(2, height - 2, 3))
    max_bends = min(len(cols) - 1, len(rows), 3)
    if max_bends < 1:
        return None

    bends = rng.randint(1, max_bends)
    turn_rows = sorted(rng.sample(rows, bends))
    col_sequence = rng.sample(cols, bends + 1)

    anchors: list[tuple[int, int]] = [(0, col_sequence[0])]
    for idx in range(bends):
        anchors.append((turn_rows[idx], col_sequence[idx]))
        anchors.append((turn_rows[idx], col_sequence[idx + 1]))
    anchors.append((height - 1, col_sequence[-1]))

    path: list[tuple[int, int]] = []
    for start, end in zip(anchors, anchors[1:], strict=False):
        segment = _line_points(start, end)
        if path:
            segment = segment[1:]
        path.extend(segment)

    if len(path) != len(set(path)):
        return None
    return path


def _place_markers(grid: np.ndarray, path: list[tuple[int, int]], rng: random.Random) -> bool:
    direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    ldr, ldc = _left_of(direction)
    rdr, rdc = _right_of(direction)
    left_pos = (path[0][0] + ldr, path[0][1] + ldc)
    right_pos = (path[0][0] + rdr, path[0][1] + rdc)
    if not (_in_bounds(grid.shape, *left_pos) and _in_bounds(grid.shape, *right_pos)):
        return False
    if rng.random() < 0.5:
        grid[left_pos] = RED
        grid[right_pos] = BLUE
    else:
        grid[left_pos] = BLUE
        grid[right_pos] = RED

    turn_ids = _turn_indices(path)
    marker_colors = list(rng.sample(MARKER_POOL, len(turn_ids)))
    for idx, color in zip(turn_ids, marker_colors, strict=False):
        prev = path[idx - 1]
        cur = path[idx]
        nxt = path[idx + 1]
        d_prev = (cur[0] - prev[0], cur[1] - prev[1])
        d_next = (nxt[0] - cur[0], nxt[1] - cur[1])
        mr = cur[0] + d_prev[0] - d_next[0]
        mc = cur[1] + d_prev[1] - d_next[1]
        if not _in_bounds(grid.shape, mr, mc):
            return False
        if int(grid[mr, mc]) != BG:
            return False
        grid[mr, mc] = int(color)
    return True


def _validate_input(grid: np.ndarray) -> bool:
    try:
        path, _, _ = _trace_path(grid)
        if not _turn_indices(path):
            return False
        out = main(grid)
    except AssertionError:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(11, 18)
    width = rng.randint(11, 18)
    path = _build_canonical_path(height, width, rng)
    if path is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r, c in path:
        grid[r, c] = PATH

    if not _place_markers(grid, path, rng):
        return None

    rotated = np.rot90(grid, k=rng.randrange(4)).copy()
    if not _validate_input(rotated):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid path-band ARC input")
