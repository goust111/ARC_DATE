from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
SQUARE_SIZE = 2
ARM_LENGTHS = (1, 2, 3)
DIRECTIONS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

# Generator controls.
NODE_COUNT_RANGE = (3, 6)
STEP_RANGE = (6, 9)
MARGIN_RANGE = (1, 4)
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _classify_arm(square_r: int, square_c: int, extras: set[tuple[int, int]]) -> str | None:
    if not extras or len(extras) > 3:
        return None

    rows = sorted(r for r, _ in extras)
    cols = sorted(c for _, c in extras)

    if len(set(rows)) == 1:
        arm_r = rows[0]
        if arm_r in (square_r, square_r + 1) and cols == list(range(cols[0], cols[0] + len(cols))):
            if cols[0] == square_c + SQUARE_SIZE:
                return "right"
            if cols[-1] == square_c - 1:
                return "left"

    if len(set(cols)) == 1:
        arm_c = cols[0]
        if arm_c in (square_c, square_c + 1) and rows == list(range(rows[0], rows[0] + len(rows))):
            if rows[0] == square_r + SQUARE_SIZE:
                return "down"
            if rows[-1] == square_r - 1:
                return "up"

    return None


def _identify_signpost(component: dict[str, object]) -> dict[str, int | str]:
    color = int(component["color"])
    coords = [tuple(rc) for rc in component["coords"]]
    coord_set = set(coords)
    if not 5 <= len(coord_set) <= 7:
        raise AssertionError("unexpected signpost size")

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    candidates: list[dict[str, int | str]] = []
    for sr in range(r0, r1):
        for sc in range(c0, c1):
            square = {
                (sr, sc),
                (sr, sc + 1),
                (sr + 1, sc),
                (sr + 1, sc + 1),
            }
            if not square <= coord_set:
                continue
            extras = coord_set - square
            direction = _classify_arm(sr, sc, extras)
            if direction is None:
                continue
            candidates.append(
                {
                    "color": color,
                    "square_r": sr,
                    "square_c": sc,
                    "direction": direction,
                }
            )

    if len(candidates) != 1:
        raise AssertionError("expected exactly one 2x2 square-and-arm signpost interpretation")
    return candidates[0]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _candidate_distance(src: dict[str, int | str], dst: dict[str, int | str], direction: str) -> int | None:
    sr = int(src["square_r"])
    sc = int(src["square_c"])
    dr = int(dst["square_r"])
    dc = int(dst["square_c"])

    if direction == "right":
        if dc <= sc + 1 or not _intervals_overlap(sr, sr + 1, dr, dr + 1):
            return None
        return dc - (sc + 1)
    if direction == "left":
        if dc + 1 >= sc or not _intervals_overlap(sr, sr + 1, dr, dr + 1):
            return None
        return sc - (dc + 1)
    if direction == "down":
        if dr <= sr + 1 or not _intervals_overlap(sc, sc + 1, dc, dc + 1):
            return None
        return dr - (sr + 1)
    if direction == "up":
        if dr + 1 >= sr or not _intervals_overlap(sc, sc + 1, dc, dc + 1):
            return None
        return sr - (dr + 1)
    raise AssertionError("unknown direction")


def _outgoing_edge(signs: list[dict[str, int | str]], idx: int) -> int | None:
    src = signs[idx]
    direction = str(src["direction"])
    candidates: list[tuple[int, int]] = []
    for j, dst in enumerate(signs):
        if j == idx:
            continue
        dist = _candidate_distance(src, dst, direction)
        if dist is not None:
            candidates.append((dist, j))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise AssertionError("ambiguous nearest signpost in pointer direction")
    return candidates[0][1]


def _topological_order(signs: list[dict[str, int | str]]) -> list[int]:
    n = len(signs)
    outgoing = [_outgoing_edge(signs, i) for i in range(n)]
    indegree = [0] * n
    for dst in outgoing:
        if dst is not None:
            indegree[dst] += 1

    ready = [i for i, deg in enumerate(indegree) if deg == 0]
    order: list[int] = []
    while ready:
        if len(ready) != 1:
            raise AssertionError("expected a unique topological start at every step")
        cur = ready.pop()
        order.append(cur)
        nxt = outgoing[cur]
        if nxt is None:
            continue
        indegree[nxt] -= 1
        if indegree[nxt] < 0:
            raise AssertionError("negative indegree indicates an invalid graph")
        if indegree[nxt] == 0:
            ready.append(nxt)

    if len(order) != n:
        raise AssertionError("signpost graph must be acyclic and fully traversable")
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Read 2x2 signposts as a directed graph and return their color order as one row."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, bg=BG)
    assert components, "expected at least one signpost"
    signs = [_identify_signpost(comp) for comp in components]
    order = _topological_order(signs)
    return np.array([[int(signs[i]["color"]) for i in order]], dtype=int)


def _sign_cells(square_r: int, square_c: int, direction: str, arm_length: int, anchor: int) -> list[tuple[int, int]]:
    cells = [
        (square_r, square_c),
        (square_r, square_c + 1),
        (square_r + 1, square_c),
        (square_r + 1, square_c + 1),
    ]
    if direction == "right":
        arm_r = square_r + anchor
        cells.extend((arm_r, square_c + 1 + step) for step in range(1, arm_length + 1))
    elif direction == "left":
        arm_r = square_r + anchor
        cells.extend((arm_r, square_c - step) for step in range(1, arm_length + 1))
    elif direction == "down":
        arm_c = square_c + anchor
        cells.extend((square_r + 1 + step, arm_c) for step in range(1, arm_length + 1))
    elif direction == "up":
        arm_c = square_c + anchor
        cells.extend((square_r - step, arm_c) for step in range(1, arm_length + 1))
    else:
        raise AssertionError("unknown direction")
    return cells


def _positions_to_signs(
    positions: list[tuple[int, int]], directions: list[str], colors: list[int]
) -> list[dict[str, int | str]]:
    return [
        {
            "color": int(colors[i]),
            "square_r": int(positions[i][0]),
            "square_c": int(positions[i][1]),
            "direction": directions[i],
        }
        for i in range(len(positions))
    ]


def _step(delta: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return -delta, 0
    if direction == "down":
        return delta, 0
    if direction == "left":
        return 0, -delta
    if direction == "right":
        return 0, delta
    raise AssertionError("unknown direction")


def _square_overlap(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    return abs(pos_a[0] - pos_b[0]) < SQUARE_SIZE and abs(pos_a[1] - pos_b[1]) < SQUARE_SIZE


def _build_path_positions(rng: np.random.Generator, count: int) -> tuple[list[tuple[int, int]], list[str]] | None:
    positions = [(0, 0)]
    directions: list[str] = []
    for step_index in range(count - 1):
        prev_dir = directions[-1] if directions else None
        choices = [d for d in DIRECTIONS if d != OPPOSITE.get(prev_dir)]
        placed = False
        for _ in range(80):
            direction = str(rng.choice(choices))
            delta = int(rng.integers(STEP_RANGE[0], STEP_RANGE[1] + 1))
            dr, dc = _step(delta, direction)
            new_pos = (positions[-1][0] + dr, positions[-1][1] + dc)
            if any(_square_overlap(new_pos, old) for old in positions):
                continue
            positions.append(new_pos)
            directions.append(direction)
            placed = True
            break
        if not placed:
            return None
    return positions, directions


def _choose_sink_direction(
    rng: np.random.Generator, positions: list[tuple[int, int]], colors: list[int], directions: list[str]
) -> str | None:
    provisional = directions + ["up"]
    base_signs = _positions_to_signs(positions, provisional, colors)
    last_idx = len(positions) - 1
    empty_dirs = []
    for direction in DIRECTIONS:
        base_signs[last_idx]["direction"] = direction
        if _outgoing_edge(base_signs, last_idx) is None:
            empty_dirs.append(direction)
    if not empty_dirs:
        return None
    return str(rng.choice(empty_dirs))


def _path_is_unambiguous(positions: list[tuple[int, int]], colors: list[int], directions: list[str]) -> bool:
    signs = _positions_to_signs(positions, directions, colors)
    expected = [i + 1 for i in range(len(signs) - 1)] + [None]
    actual = [_outgoing_edge(signs, i) for i in range(len(signs))]
    return actual == expected


def _render_candidate(
    rng: np.random.Generator, positions: list[tuple[int, int]], colors: list[int], directions: list[str]
) -> np.ndarray | None:
    painted: dict[tuple[int, int], int] = {}
    for i, (square_r, square_c) in enumerate(positions):
        arm_length = int(rng.choice(ARM_LENGTHS))
        anchor = int(rng.integers(0, 2))
        for cell in _sign_cells(square_r, square_c, directions[i], arm_length, anchor):
            if cell in painted:
                return None
            painted[cell] = int(colors[i])

    rows = [r for r, _ in painted]
    cols = [c for _, c in painted]
    top = min(rows) - int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    left = min(cols) - int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    bottom = max(rows) + int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    right = max(cols) + int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

    grid = np.full((bottom - top + 1, right - left + 1), BG, dtype=int)
    for (r, c), color in painted.items():
        grid[r - top, c - left] = color
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(NODE_COUNT_RANGE[0], NODE_COUNT_RANGE[1] + 1))
    built = _build_path_positions(rng, count)
    if built is None:
        return None
    positions, directions = built

    colors = [int(c) for c in rng.choice(np.arange(1, 10), size=count, replace=False)]
    sink_direction = _choose_sink_direction(rng, positions, colors, directions)
    if sink_direction is None:
        return None
    directions = directions + [sink_direction]

    if not _path_is_unambiguous(positions, colors, directions):
        return None

    grid = _render_candidate(rng, positions, colors, directions)
    if grid is None:
        return None

    expected = np.array([colors], dtype=int)
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signpost graph input")
