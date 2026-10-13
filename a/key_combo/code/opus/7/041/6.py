from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
HINGE = 1
ROD = 2
FILL = 3
ENDPOINT = 5

HORIZONTAL = "h"
VERTICAL = "v"

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MAX_GENERATE_ATTEMPTS = 400
EXTRA_ROD_ATTEMPTS = 250


def _neighbors(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    out: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        comp = _neighbors(mask, (rr, cc))
        for pr, pc in comp:
            seen[pr, pc] = True
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], dirs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _full_rectangle_info(coords: list[tuple[int, int]]) -> tuple[int, int, bool]:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return height, width, len(coords) == height * width


def _fold_step(orientation: str, hinge: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    r, c = hinge
    if orientation == HORIZONTAL:
        up = r
        down = h - 1 - r
        if up == down:
            return None
        return (-1, 0) if up > down else (1, 0)

    left = c
    right = w - 1 - c
    if left == right:
        return None
    return (0, -1) if left > right else (0, 1)


def _make_record(
    orientation: str,
    hinge: tuple[int, int],
    fixed_sign: int,
    fixed_len: int,
    tail_len: int,
    shape: tuple[int, int],
) -> dict[str, object] | None:
    if fixed_len < 1 or tail_len < 1:
        return None

    fold_step = _fold_step(orientation, hinge, shape)
    if fold_step is None:
        return None

    hr, hc = hinge
    if orientation == HORIZONTAL:
        endpoint = (hr, hc + fixed_sign * fixed_len)
        fixed_red = [(hr, hc + fixed_sign * step) for step in range(1, fixed_len)]
        input_tail = [(hr, hc - fixed_sign * step) for step in range(1, tail_len + 1)]
    else:
        endpoint = (hr + fixed_sign * fixed_len, hc)
        fixed_red = [(hr + fixed_sign * step, hc) for step in range(1, fixed_len)]
        input_tail = [(hr - fixed_sign * step, hc) for step in range(1, tail_len + 1)]

    folded_tail = [(hr + fold_step[0] * step, hc + fold_step[1] * step) for step in range(1, tail_len + 1)]
    cells = [endpoint, hinge, *fixed_red, *input_tail, *folded_tail]

    h, w = shape
    if len(set(cells)) != len(cells):
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None

    return {
        "orientation": orientation,
        "endpoint": endpoint,
        "hinge": hinge,
        "fixed_sign": fixed_sign,
        "fixed_len": fixed_len,
        "tail_len": tail_len,
        "fixed_red": fixed_red,
        "input_tail": input_tail,
        "folded_tail": folded_tail,
    }


def _record_cells(record: dict[str, object], *, folded: bool) -> set[tuple[int, int]]:
    tail_key = "folded_tail" if folded else "input_tail"
    cells = {
        tuple(record["endpoint"]),
        tuple(record["hinge"]),
        *[tuple(p) for p in record["fixed_red"]],
        *[tuple(p) for p in record[tail_key]],
    }
    return cells


def _render_records(records: list[dict[str, object]], shape: tuple[int, int], *, folded: bool) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for record in records:
        er, ec = record["endpoint"]
        hr, hc = record["hinge"]
        grid[er, ec] = ENDPOINT
        grid[hr, hc] = HINGE
        for r, c in record["fixed_red"]:
            grid[r, c] = ROD
        tail = record["folded_tail"] if folded else record["input_tail"]
        for r, c in tail:
            grid[r, c] = ROD
    return grid


def _parse_rods(grid: np.ndarray) -> list[dict[str, object]]:
    rods: list[dict[str, object]] = []
    for comp in _components(grid != BG):
        coord_set = set(comp)
        endpoints = [p for p in comp if int(grid[p]) == ENDPOINT]
        hinges = [p for p in comp if int(grid[p]) == HINGE]
        if len(endpoints) != 1 or len(hinges) != 1:
            raise AssertionError("each rod must contain one endpoint and one hinge")
        endpoint = endpoints[0]
        hinge = hinges[0]

        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        if len(rows) == 1 and len(cols) >= 3:
            orientation = HORIZONTAL
            if endpoint[0] != hinge[0]:
                raise AssertionError("horizontal rod markers must share a row")
            fixed_sign = -1 if endpoint[1] < hinge[1] else 1
            fixed_len = abs(endpoint[1] - hinge[1])
            fixed_red = [(hinge[0], hinge[1] + fixed_sign * step) for step in range(1, fixed_len)]
            tail_step = (0, -fixed_sign)
        elif len(cols) == 1 and len(rows) >= 3:
            orientation = VERTICAL
            if endpoint[1] != hinge[1]:
                raise AssertionError("vertical rod markers must share a column")
            fixed_sign = -1 if endpoint[0] < hinge[0] else 1
            fixed_len = abs(endpoint[0] - hinge[0])
            fixed_red = [(hinge[0] + fixed_sign * step, hinge[1]) for step in range(1, fixed_len)]
            tail_step = (-fixed_sign, 0)
        else:
            raise AssertionError("each input object must be a straight rod")

        expected_fixed = {endpoint, hinge, *fixed_red}
        if not expected_fixed.issubset(coord_set):
            raise AssertionError("fixed segment must be contiguous")
        for p in fixed_red:
            if int(grid[p]) != ROD:
                raise AssertionError("fixed segment interior must be red")

        input_tail: list[tuple[int, int]] = []
        step = 1
        while True:
            nr = hinge[0] + tail_step[0] * step
            nc = hinge[1] + tail_step[1] * step
            nxt = (nr, nc)
            if nxt not in coord_set:
                break
            if int(grid[nxt]) != ROD:
                raise AssertionError("tail pixels must be red")
            input_tail.append(nxt)
            step += 1
        if not input_tail:
            raise AssertionError("every rod needs a movable tail")

        if coord_set != expected_fixed | set(input_tail):
            raise AssertionError("rod contains unexpected pixels")

        record = _make_record(orientation, hinge, fixed_sign, fixed_len, len(input_tail), grid.shape)
        if record is None:
            raise AssertionError("failed to reconstruct folded rod")
        rods.append(record)
    return rods


def _fold_only(grid: np.ndarray) -> np.ndarray:
    rods = _parse_rods(np.asarray(grid, dtype=int))
    return _render_records(rods, np.asarray(grid).shape, folded=True)


def _fill_square_regions(folded: np.ndarray) -> np.ndarray:
    out = folded.copy()
    for comp in _components(folded == BG):
        height, width, is_full = _full_rectangle_info(comp)
        if is_full and height == width:
            for r, c in comp:
                out[r, c] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked rod, then fill only the square black regions enclosed by the folded rods and the grid boundary.

    Input structure:
    - The grid is black and contains several straight red rods.
    - Each rod has one gray endpoint marker at an end and one blue hinge marker in its interior.
    - Rods may be horizontal or vertical, and different rods cooperate only after folding.

    Rule:
    - Keep the segment from the gray endpoint through the blue hinge fixed for every rod.
    - Rotate the red tail on the far side of the hinge by 90 degrees toward the side with more space to the grid edge.
    - After all rods are folded, split the remaining black cells into 4-connected regions and fill a region green only when it is a completely filled square block inside its own bounding box.

    Key insight:
    - No single rod creates the target structure alone; the solver must first fold every rod globally, then judge the geometry of the emergent enclosed black regions.
    """
    folded = _fold_only(np.asarray(grid, dtype=int))
    return _fill_square_regions(folded)


def _sample_base_records(shape: tuple[int, int]) -> list[dict[str, object]] | None:
    n = shape[0]

    non_square_options = [
        (height, width)
        for height in range(2, 5)
        for width in range(3, 6)
        if height != width and 7 <= height + width + 1 <= 11
    ]
    top_height, top_width = random.choice(non_square_options)
    square_side = random.randint(3, 5)

    right_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)

    c_right = n - 1 - right_margin
    c_left = c_right - top_width - 1
    if c_left <= (n - 1) // 2:
        return None

    r_bottom = n - 1 - bottom_margin
    r_top = r_bottom - square_side - 1
    if r_top <= (n - 1) // 2:
        return None

    records = [
        _make_record(VERTICAL, (top_height, c_left), -1, top_height, top_width, shape),
        _make_record(VERTICAL, (top_height, c_right), -1, top_height, top_width, shape),
        _make_record(HORIZONTAL, (r_top, square_side), -1, square_side, square_side, shape),
        _make_record(HORIZONTAL, (r_bottom, square_side), -1, square_side, square_side, shape),
    ]
    if any(record is None for record in records):
        return None
    return [record for record in records if record is not None]


def _sample_extra_record(
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> dict[str, object] | None:
    n = shape[0]
    for _ in range(EXTRA_ROD_ATTEMPTS):
        orientation = random.choice([HORIZONTAL, VERTICAL])
        fixed_len = random.randint(2, 6)
        tail_len = random.randint(2, 5)
        if not (7 <= fixed_len + tail_len + 1 <= 11):
            continue

        fixed_sign = random.choice([-1, 1])
        if orientation == HORIZONTAL:
            row = random.randint(2, n - 3)
            col = random.randint(2, n - 3)
            hinge = (row, col)
        else:
            row = random.randint(2, n - 3)
            col = random.randint(2, n - 3)
            hinge = (row, col)

        record = _make_record(orientation, hinge, fixed_sign, fixed_len, tail_len, shape)
        if record is None:
            continue

        input_cells = _record_cells(record, folded=False)
        output_cells = _record_cells(record, folded=True)

        if any(r in {0, n - 1} or c in {0, n - 1} for r, c in input_cells | output_cells):
            continue
        if input_cells & blocked_input:
            continue
        if output_cells & blocked_output:
            continue
        return record
    return None


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _validate_generated_input(grid: np.ndarray, expected_rod_count: int) -> bool:
    try:
        rods = _parse_rods(grid)
    except AssertionError:
        return False
    if len(rods) != expected_rod_count:
        return False

    folded = _fold_only(grid)
    regions = []
    for comp in _components(folded == BG):
        height, width, is_full = _full_rectangle_info(comp)
        if is_full and min(height, width) >= 2:
            regions.append((height, width))

    has_square = any(h == w for h, w in regions)
    has_non_square = any(h != w for h, w in regions)
    if not (has_square and has_non_square):
        return False

    out = main(grid)
    if out.shape != grid.shape or not np.any(out == FILL):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(18, 22)
    shape = (n, n)

    records = _sample_base_records(shape)
    if records is None:
        return None

    base_input = set().union(*(_record_cells(record, folded=False) for record in records))
    base_output = set().union(*(_record_cells(record, folded=True) for record in records))
    blocked_input = _expand(base_input, shape, DIR8)
    blocked_output = _expand(base_output, shape, DIR8)

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        extra = _sample_extra_record(shape, blocked_input, blocked_output)
        if extra is None:
            return None
        records.append(extra)
        extra_input = _record_cells(extra, folded=False)
        extra_output = _record_cells(extra, folded=True)
        blocked_input |= _expand(extra_input, shape, DIR8)
        blocked_output |= _expand(extra_output, shape, DIR8)

    inp = _render_records(records, shape, folded=False)
    inp = _random_symmetry(inp)
    if not _validate_generated_input(inp, len(records)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
