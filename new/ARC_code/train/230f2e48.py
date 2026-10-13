from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, in_bounds, neighbors4
import random


def _line_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size == 3, "expected one line color and two marker colors"
    max_count = int(counts.max())
    winners = vals[counts == max_count]
    assert winners.size == 1, "line color must be uniquely most common"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each straight red rod at its black marker, leaving the gray-anchored side fixed.

    Input structure:
    - The background is the dominant color.
    - Each foreground object is a straight horizontal or vertical 4-connected rod.
    - Each rod contains many pixels of one line color, one endpoint marker, and one
      interior control marker.

    Rule:
    - For every rod, keep the segment from the endpoint marker through the control
      marker unchanged.
    - Move the line-color tail on the far side of the control marker to a
      perpendicular ray starting next to that marker.
    - Choose the perpendicular direction with more space to the grid edge.

    Key insight:
    - The endpoint marker identifies the fixed side of the rod, while the control
      marker is the hinge; the tail folds into the larger open half-plane.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    line_color = _line_color(g, bg)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert components, "expected at least one marked rod"

    clear: list[tuple[int, int]] = []
    paint: list[tuple[int, int]] = []

    for comp in components:
        coords = [tuple(map(int, coord)) for coord in comp["coords"]]
        coord_set = set(coords)
        marker_coords = [p for p in coords if int(g[p]) != line_color]
        assert len(marker_coords) == 2, "each rod must have exactly two markers"

        degrees = {
            p: sum((nr, nc) in coord_set for nr, nc in neighbors4(*p))
            for p in marker_coords
        }
        endpoints = [p for p, degree in degrees.items() if int(degree) == 1]
        hinges = [p for p, degree in degrees.items() if int(degree) == 2]
        assert len(endpoints) == 1 and len(hinges) == 1, "markers must be endpoint and hinge"
        endpoint = endpoints[0]
        hinge = hinges[0]

        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        assert len(rows) == 1 or len(cols) == 1, "each object must be a straight rod"

        if len(rows) == 1:
            assert endpoint[0] == hinge[0], "horizontal rod markers must share a row"
            fixed_sign = int(np.sign(endpoint[1] - hinge[1]))
            assert fixed_sign != 0, "endpoint and hinge must be distinct"
            tail_step = (0, -fixed_sign)
            up_space = int(hinge[0])
            down_space = int(H - 1 - hinge[0])
            assert up_space != down_space, "perpendicular direction must be unique"
            fold_step = (-1, 0) if up_space > down_space else (1, 0)
        else:
            assert endpoint[1] == hinge[1], "vertical rod markers must share a column"
            fixed_sign = int(np.sign(endpoint[0] - hinge[0]))
            assert fixed_sign != 0, "endpoint and hinge must be distinct"
            tail_step = (-fixed_sign, 0)
            left_space = int(hinge[1])
            right_space = int(W - 1 - hinge[1])
            assert left_space != right_space, "perpendicular direction must be unique"
            fold_step = (0, -1) if left_space > right_space else (0, 1)

        tail: list[tuple[int, int]] = []
        step = 1
        while True:
            p = (
                int(hinge[0]) + int(tail_step[0]) * step,
                int(hinge[1]) + int(tail_step[1]) * step,
            )
            if p not in coord_set:
                break
            assert int(g[p]) == line_color, "tail must contain only line-color pixels"
            tail.append(p)
            step += 1
        assert tail, "each hinge must have a moving tail"

        for step in range(1, len(tail) + 1):
            p = (
                int(hinge[0]) + int(fold_step[0]) * step,
                int(hinge[1]) + int(fold_step[1]) * step,
            )
            assert in_bounds(g.shape, *p), "folded tail must remain in bounds"
            paint.append(p)
        clear.extend(tail)

    assert len(set(clear)) == len(clear), "moving tails must not overlap"
    assert len(set(paint)) == len(paint), "folded tails must not collide"
    clear_set = set(clear)
    for p in paint:
        assert int(g[p]) == int(bg) or p in clear_set, "fold target must be empty"

    out = g.copy()
    for p in clear:
        out[p] = bg
    for p in paint:
        out[p] = line_color
    return out

# --------------- generator (o5x) ---------------

# ARC colors.
BG, RED, HINGE, END = 7, 2, 0, 5

# Canvas and rod-count sampling.
GRID_SIZE_MIN, GRID_SIZE_MAX = 8, 16
ROD_COUNT_MIN, ROD_COUNT_MAX = 1, 7
ROD_DENSITY_DIVISOR = 28

# Rod geometry sampling.
HORIZONTAL, VERTICAL = "h", "v"
ORIENTATIONS = (HORIZONTAL, VERTICAL)
FIXED_SIGNS = (-1, 1)
ROD_LEN_MIN, ROD_LEN_MAX = 3, 12
FIXED_LEN_MIN, FIXED_LEN_MAX = 2, 9
TAIL_LEN_MAX = 5

# Placement and retry controls.
GAP_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATE_ATTEMPTS = 300

Record = dict[str, object]
RecordKey = tuple[int, int, int, str]

_RECORD_CACHE: dict[int, list[tuple[RecordKey, Record]]] = {}


def _expanded(cells: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    out = set(cells)
    shape = (n, n)
    for r, c in cells:
        for dr, dc in GAP_DIRECTIONS:
            nr, nc = r + dr, c + dc
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _fold_steps(n: int, orient: str, hinge: tuple[int, int], fixed_sign: int):
    """Return fixed-arm and fold steps, or None for a centered hinge."""
    r, c = hinge
    if orient == HORIZONTAL:
        up, down = r, n - 1 - r
        if up == down:
            return None
        return (0, fixed_sign), ((-1, 0) if up > down else (1, 0))

    left, right = c, n - 1 - c
    if left == right:
        return None
    return (fixed_sign, 0), ((0, -1) if left > right else (0, 1))


def _ray_cells(origin: tuple[int, int], step: tuple[int, int], count: int) -> list[tuple[int, int]]:
    r, c = origin
    dr, dc = step
    return [(r + dr * k, c + dc * k) for k in range(1, count + 1)]


def _record(n: int, orient: str, fixed_len: int, tail_len: int, hinge: tuple[int, int], fixed_sign: int):
    steps = _fold_steps(n, orient, hinge, fixed_sign)
    if steps is None:
        return None

    fixed_step, fold_step = steps
    tail_step = (-fixed_step[0], -fixed_step[1])
    fixed = _ray_cells(hinge, fixed_step, fixed_len - 1)
    tail = _ray_cells(hinge, tail_step, tail_len)
    folded = _ray_cells(hinge, fold_step, tail_len)

    all_cells = fixed + tail + folded + [hinge]
    shape = (n, n)
    if any(not in_bounds(shape, rr, cc) for rr, cc in all_cells):
        return None

    endpoint = fixed[-1]
    fixed_red = fixed[:-1]
    input_cells = {hinge, endpoint, *fixed_red, *tail}
    envelope = input_cells | set(folded)
    if len(envelope) != len(input_cells) + len(folded):
        return None

    return {
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_red": fixed_red,
        "tail": tail,
        "folded": folded,
        "envelope": envelope,
        "red_count": fixed_len + tail_len - 2,
    }


def _all_records(n: int) -> list[tuple[RecordKey, Record]]:
    if n in _RECORD_CACHE:
        return _RECORD_CACHE[n]

    records: list[tuple[RecordKey, Record]] = []
    for orient in ORIENTATIONS:
        for total_len in range(ROD_LEN_MIN, ROD_LEN_MAX + 1):
            fixed_min = max(FIXED_LEN_MIN, total_len - TAIL_LEN_MAX)
            fixed_max = min(FIXED_LEN_MAX, total_len - 1)
            for fixed_len in range(fixed_min, fixed_max + 1):
                tail_len = total_len - fixed_len
                key = (total_len, fixed_len, tail_len, orient)
                for r in range(n):
                    for c in range(n):
                        for fixed_sign in FIXED_SIGNS:
                            rec = _record(n, orient, fixed_len, tail_len, (r, c), fixed_sign)
                            if rec is not None:
                                records.append((key, rec))
    _RECORD_CACHE[n] = records
    return records


def _choose_record(n: int, blocked: set[tuple[int, int]], rod_count: int, current_red: int, is_last: bool):
    by_key: dict[RecordKey, list[Record]] = {}
    for key, rec in _all_records(n):
        total_len = key[0]
        red_count = total_len - 2
        if is_last and current_red + red_count <= rod_count:
            continue
        if rec["envelope"] & blocked:
            continue
        by_key.setdefault(key, []).append(rec)

    if not by_key:
        return None

    totals = sorted({key[0] for key in by_key})
    total_len = random.choice(totals)
    fixed_lengths = sorted({key[1] for key in by_key if key[0] == total_len})
    fixed_len = random.choice(fixed_lengths)
    orientations = sorted({key[3] for key in by_key if key[:2] == (total_len, fixed_len)})
    orient = random.choice(orientations)
    tail_len = total_len - fixed_len
    return random.choice(by_key[(total_len, fixed_len, tail_len, orient)])


def _paint(records: list[Record], n: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    for rec in records:
        hinge = rec["hinge"]
        endpoint = rec["endpoint"]
        inp[hinge] = HINGE
        out[hinge] = HINGE
        inp[endpoint] = END
        out[endpoint] = END
        for p in rec["fixed_red"]:
            inp[p] = RED
            out[p] = RED
        for p in rec["tail"]:
            inp[p] = RED
        for p in rec["folded"]:
            out[p] = RED
    return inp, out


def _place_records(n: int, rod_count: int) -> list[Record] | None:
    """Sample non-touching rods and their fold envelopes for one grid."""
    blocked: set[tuple[int, int]] = set()
    records: list[Record] = []
    red_total = 0

    for i in range(rod_count):
        rec = _choose_record(n, blocked, rod_count, red_total, i == rod_count - 1)
        if rec is None:
            return None
        records.append(rec)
        red_total += int(rec["red_count"])
        blocked |= _expanded(rec["envelope"], n)
    return records


def _valid_generated_pair(inp: np.ndarray, out: np.ndarray, rod_count: int) -> bool:
    counts = {int(v): int(np.count_nonzero(inp == v)) for v in (BG, RED, HINGE, END)}
    if counts[RED] <= rod_count or counts[BG] <= counts[RED]:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    max_rods = max(ROD_COUNT_MIN, min(ROD_COUNT_MAX, (n * n) // ROD_DENSITY_DIVISOR))
    rod_count = random.randint(ROD_COUNT_MIN, max_rods)

    records = _place_records(n, rod_count)
    if records is None:
        return None
    inp, out = _paint(records, n)
    if not _valid_generated_pair(inp, out, rod_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 230f2e48 input")
