import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
GRID_SIZE_RANGE = (18, 25)
WALL_COUNT_CHOICES = (1, 2, 3)
WALL_COUNT_WEIGHTS = (2, 4, 3)
INNER_SIZE_RANGE = (3, 8)
OBJECT_PLACE_TRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_wall(grid: np.ndarray, coords: list[tuple[int, int]]):
    comp = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    boundary = _frame_boundary(r0, c0, r1, c1)
    if any(cell not in boundary for cell in comp):
        return None

    missing = sorted(boundary - comp)
    if len(missing) not in (1, 2):
        return None

    rows = {r for r, _ in missing}
    cols = {c for _, c in missing}
    side = None
    if len(rows) == 1 and next(iter(rows)) == r0:
        ordered = sorted(c for _, c in missing)
        if ordered[0] <= c0 or ordered[-1] >= c1 or ordered[-1] - ordered[0] + 1 != len(ordered):
            return None
        side = "top"
    elif len(rows) == 1 and next(iter(rows)) == r1:
        ordered = sorted(c for _, c in missing)
        if ordered[0] <= c0 or ordered[-1] >= c1 or ordered[-1] - ordered[0] + 1 != len(ordered):
            return None
        side = "bottom"
    elif len(cols) == 1 and next(iter(cols)) == c0:
        ordered = sorted(r for r, _ in missing)
        if ordered[0] <= r0 or ordered[-1] >= r1 or ordered[-1] - ordered[0] + 1 != len(ordered):
            return None
        side = "left"
    elif len(cols) == 1 and next(iter(cols)) == c1:
        ordered = sorted(r for r, _ in missing)
        if ordered[0] <= r0 or ordered[-1] >= r1 or ordered[-1] - ordered[0] + 1 != len(ordered):
            return None
        side = "right"
    else:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gate_len": len(missing),
    }


def _longest_run(values: np.ndarray, color: int) -> int:
    best = 0
    cur = 0
    for v in values.tolist():
        if int(v) == color:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _indicator_length(grid: np.ndarray, wall: dict) -> int:
    r0, c0, r1, c1 = wall["bbox"]
    side = wall["side"]
    if side == "top":
        if r0 < 2:
            return 0
        return _longest_run(grid[r0 - 2, c0 : c1 + 1], RED)
    if side == "bottom":
        if r1 + 2 >= grid.shape[0]:
            return 0
        return _longest_run(grid[r1 + 2, c0 : c1 + 1], RED)
    if side == "left":
        if c0 < 2:
            return 0
        return _longest_run(grid[r0 : r1 + 1, c0 - 2], RED)
    if c1 + 2 >= grid.shape[1]:
        return 0
    return _longest_run(grid[r0 : r1 + 1, c1 + 2], RED)


def _paint_stripes(out: np.ndarray, wall: dict, count: int) -> None:
    if count <= 0:
        return
    r0, c0, r1, c1 = wall["bbox"]
    side = wall["side"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if side == "top":
        span = min(count, inner_h)
        out[r0 + 1 : r0 + 1 + span, c0 + 1 : c1] = GREEN
    elif side == "bottom":
        span = min(count, inner_h)
        out[r1 - span : r1, c0 + 1 : c1] = GREEN
    elif side == "left":
        span = min(count, inner_w)
        out[r0 + 1 : r1, c0 + 1 : c0 + 1 + span] = GREEN
    else:
        span = min(count, inner_w)
        out[r0 + 1 : r1, c1 - span : c1] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gated rectangular wall interior with stripes whose count is encoded by the nearby outside marker.

    Input structure:
    - The background is black on a grid of roughly 18x18 to 25x25.
    - One or more blue 1-pixel-thick rectangular walls appear; each wall has exactly one 1-2 pixel gate on one side.
    - A red 1-pixel-thick indicator strip sits outside the same side as the gate, parallel to that side and separated from the wall by one black cell.

    Rule:
    - Detect each blue wall as an almost-complete rectangular frame and find which side contains the gate.
    - Read the length L of the red strip that belongs to that wall by looking two cells outside the gated side.
    - If the gate is on the top or bottom, draw L full horizontal green rows inside the wall starting from that side.
    - If the gate is on the left or right, draw L full vertical green columns inside the wall starting from that side.
    - Leave the original blue walls and red strips unchanged.

    Key insight:
    The fill direction comes from the gate side, while the number of stripes comes from the separate red strip outside the same wall rather than from the gate width or alignment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g == BLUE):
        wall = _detect_wall(g, comp)
        if wall is None:
            continue
        count = _indicator_length(g, wall)
        _paint_stripes(out, wall, count)

    return out


def _choose_strip_start(
    rng: random.Random,
    span_start: int,
    span_end: int,
    length: int,
    gate_start: int,
    gate_end: int,
) -> int:
    starts = list(range(span_start, span_end - length + 2))
    preferred = [s for s in starts if not (s == gate_start and s + length - 1 == gate_end)]
    return rng.choice(preferred or starts)


def _footprint_bounds(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    if side == "top":
        fr0 = r0 - 3
    elif side == "bottom":
        fr1 = r1 + 3
    elif side == "left":
        fc0 = c0 - 3
    else:
        fc1 = c1 + 3
    return fr0, fc0, fr1, fc1


def _place_wall(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    side: str,
) -> bool:
    count = rng.randint(1, 4)
    gate_len = rng.randint(1, 2)
    min_inner = max(count, gate_len, INNER_SIZE_RANGE[0])
    inner_h = rng.randint(min_inner, INNER_SIZE_RANGE[1])
    inner_w = rng.randint(min_inner, INNER_SIZE_RANGE[1])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    h, w = inp.shape

    if side == "top":
        row_lo, row_hi = 3, h - outer_h - 1
        col_lo, col_hi = 1, w - outer_w - 1
    elif side == "bottom":
        row_lo, row_hi = 1, h - outer_h - 3
        col_lo, col_hi = 1, w - outer_w - 1
    elif side == "left":
        row_lo, row_hi = 1, h - outer_h - 1
        col_lo, col_hi = 3, w - outer_w - 1
    else:
        row_lo, row_hi = 1, h - outer_h - 1
        col_lo, col_hi = 1, w - outer_w - 3

    if row_hi < row_lo or col_hi < col_lo:
        return False

    for _ in range(OBJECT_PLACE_TRIES):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        fr0, fc0, fr1, fc1 = _footprint_bounds(r0, c0, r1, c1, side)
        if occupied[fr0 : fr1 + 1, fc0 : fc1 + 1].any():
            continue

        if side in ("top", "bottom"):
            gate_start = rng.randint(c0 + 1, c1 - gate_len)
            gate_end = gate_start + gate_len - 1
            strip_start = _choose_strip_start(rng, c0 + 1, c1 - 1, count, gate_start, gate_end)
            strip_end = strip_start + count - 1
            strip_row = r0 - 2 if side == "top" else r1 + 2
            inp[strip_row, strip_start : strip_end + 1] = RED
            out[strip_row, strip_start : strip_end + 1] = RED
        else:
            gate_start = rng.randint(r0 + 1, r1 - gate_len)
            gate_end = gate_start + gate_len - 1
            strip_start = _choose_strip_start(rng, r0 + 1, r1 - 1, count, gate_start, gate_end)
            strip_end = strip_start + count - 1
            strip_col = c0 - 2 if side == "left" else c1 + 2
            inp[strip_start : strip_end + 1, strip_col] = RED
            out[strip_start : strip_end + 1, strip_col] = RED

        inp[r0, c0 : c1 + 1] = BLUE
        inp[r1, c0 : c1 + 1] = BLUE
        inp[r0 : r1 + 1, c0] = BLUE
        inp[r0 : r1 + 1, c1] = BLUE
        out[r0, c0 : c1 + 1] = BLUE
        out[r1, c0 : c1 + 1] = BLUE
        out[r0 : r1 + 1, c0] = BLUE
        out[r0 : r1 + 1, c1] = BLUE

        if side == "top":
            inp[r0, gate_start : gate_end + 1] = BLACK
            out[r0, gate_start : gate_end + 1] = BLACK
            out[r0 + 1 : r0 + 1 + count, c0 + 1 : c1] = GREEN
        elif side == "bottom":
            inp[r1, gate_start : gate_end + 1] = BLACK
            out[r1, gate_start : gate_end + 1] = BLACK
            out[r1 - count : r1, c0 + 1 : c1] = GREEN
        elif side == "left":
            inp[gate_start : gate_end + 1, c0] = BLACK
            out[gate_start : gate_end + 1, c0] = BLACK
            out[r0 + 1 : r1, c0 + 1 : c0 + 1 + count] = GREEN
        else:
            inp[gate_start : gate_end + 1, c1] = BLACK
            out[gate_start : gate_end + 1, c1] = BLACK
            out[r0 + 1 : r1, c1 - count : c1] = GREEN

        occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
        return True

    return False


def _validate_input(inp: np.ndarray, expected: np.ndarray, wall_count: int) -> bool:
    blue_comps = _components(inp == BLUE)
    red_comps = _components(inp == RED)
    if len(blue_comps) != wall_count or len(red_comps) != wall_count:
        return False
    if any(_detect_wall(inp, comp) is None for comp in blue_comps):
        return False
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    return not np.array_equal(inp, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    inp = np.full((h, w), BLACK, dtype=int)
    out = inp.copy()
    occupied = np.zeros((h, w), dtype=bool)

    wall_count = rng.choices(WALL_COUNT_CHOICES, weights=WALL_COUNT_WEIGHTS, k=1)[0]
    sides = rng.sample(SIDES, wall_count)
    rng.shuffle(sides)
    for side in sides:
        if not _place_wall(inp, out, occupied, rng, side):
            return None

    if GREEN in np.unique(inp):
        return None
    if not _validate_input(inp, out, wall_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
