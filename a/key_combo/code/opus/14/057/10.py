from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
PAIR_COUNT_RANGE = (2, 4)
TRIANGLE_LENGTH_RANGE = (2, 6)

MAX_GENERATION_ATTEMPTS = 400
MAX_FILL_RATIO = 0.35


def _triangle_area(length: int) -> int:
    return (length + 1) * (length + 2) // 2


def _triangle_cells(start_r: int, start_c: int, length: int, direction: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for k in range(length + 1):
        r = start_r + k
        if direction == 1:
            cols = range(start_c, start_c + k + 1)
        else:
            cols = range(start_c - k, start_c + 1)
        for c in cols:
            cells.append((r, c))
    return cells


def _diagonal_cells(start_r: int, start_c: int, length: int, direction: int) -> list[tuple[int, int]]:
    return [(start_r + k, start_c + direction * k) for k in range(length + 1)]


def _triangle_end(start_r: int, start_c: int, length: int, direction: int) -> tuple[int, int]:
    return start_r + length, start_c + direction * length


def _extract_pairs(grid: np.ndarray) -> list[dict[str, int]]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)
    pairs: list[dict[str, int]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        pts = sorted((int(r), int(c)) for r, c in coords)
        (r0, c0), (r1, c1) = pts
        dr = r1 - r0
        dc = c1 - c0
        assert dr > 0 and abs(dc) == dr, "same-color endpoints must define a strict 45-degree diagonal"
        direction = 1 if dc > 0 else -1
        pairs.append(
            {
                "color": color,
                "start_r": r0,
                "start_c": c0,
                "length": dr,
                "direction": direction,
            }
        )
    return sorted(pairs, key=lambda item: (item["start_r"], item["start_c"], item["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a right triangle under each same-colored diagonal endpoint pair.

    Input structure:
    - A black background contains 2-4 foreground colors.
    - Each foreground color appears exactly twice.
    - The two same-colored pixels lie on one strict 45-degree diagonal, with one above the other.

    Rule:
    - For each color, treat the higher pixel as the start and the lower pixel as the end.
    - If the end is down-right, fill rows downward with horizontal runs that grow one cell to the right per row.
    - If the end is down-left, fill rows downward with horizontal runs that grow one cell to the left per row.
    - Continue until the end row, so the last row's far diagonal tip lands exactly on the end pixel.

    Key insight:
    - A same-colored endpoint pair does not encode a diagonal line to connect; it encodes both the triangle's direction and its row-by-row growth length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    out = g.copy()
    for pair in _extract_pairs(g):
        color = pair["color"]
        start_r = pair["start_r"]
        start_c = pair["start_c"]
        length = pair["length"]
        direction = pair["direction"]

        end_r, end_c = _triangle_end(start_r, start_c, length, direction)
        assert int(g[end_r, end_c]) == color, "derived endpoint must match the visible lower endpoint"

        for k in range(length + 1):
            r = start_r + k
            if direction == 1:
                c0, c1 = start_c, start_c + k
            else:
                c0, c1 = start_c - k, start_c
            row = out[r, c0 : c1 + 1]
            assert np.all((row == BACKGROUND) | (row == color)), "triangle fills must not conflict"
            out[r, c0 : c1 + 1] = color

    return out


def _expanded_bbox(start_r: int, start_c: int, length: int, direction: int) -> tuple[int, int, int, int]:
    end_r, end_c = _triangle_end(start_r, start_c, length, direction)
    c0 = min(start_c, end_c)
    c1 = max(start_c, end_c)
    return start_r - 1, c0 - 1, end_r + 1, c1 + 1


def _candidate_positions(
    shape: tuple[int, int], reserved: np.ndarray, length: int, direction: int
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []

    row_values = range(1, h - length - 1)
    if direction == 1:
        col_values = range(1, w - length - 1)
    else:
        col_values = range(length + 1, w - 1)

    for start_r in row_values:
        for start_c in col_values:
            r0, c0, r1, c1 = _expanded_bbox(start_r, start_c, length, direction)
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                continue
            if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            candidates.append((start_r, start_c))
    return candidates


def _sample_directions(n_pairs: int) -> list[int]:
    if n_pairs == 1:
        return [random.choice((-1, 1))]
    directions = [-1, 1]
    while len(directions) < n_pairs:
        directions.append(random.choice((-1, 1)))
    random.shuffle(directions)
    return directions


def _sample_lengths(n_pairs: int, max_length: int) -> list[int]:
    assert max_length >= TRIANGLE_LENGTH_RANGE[0]
    lo = TRIANGLE_LENGTH_RANGE[0]
    lengths = [random.randint(lo, max_length) for _ in range(n_pairs)]
    if len(set(lengths)) == 1 and max_length > lo:
        idx = random.randrange(n_pairs)
        lengths[idx] = lo if lengths[idx] != lo else max_length
    return lengths


def _render_output(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        rows, cols = np.array(_triangle_cells(spec["start_r"], spec["start_c"], spec["length"], spec["direction"])).T
        out[rows, cols] = spec["color"]
    return out


def _render_input(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    inp = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = spec["color"]
        start = (spec["start_r"], spec["start_c"])
        end = _triangle_end(spec["start_r"], spec["start_c"], spec["length"], spec["direction"])
        inp[start] = color
        inp[end] = color
    return inp


def _wrong_diagonal_output(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in _extract_pairs(out):
        rows, cols = np.array(
            _diagonal_cells(pair["start_r"], pair["start_c"], pair["length"], pair["direction"])
        ).T
        out[rows, cols] = pair["color"]
    return out


def _wrong_rectangle_output(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in _extract_pairs(out):
        end_r, end_c = _triangle_end(pair["start_r"], pair["start_c"], pair["length"], pair["direction"])
        c0 = min(pair["start_c"], end_c)
        c1 = max(pair["start_c"], end_c)
        out[pair["start_r"] : end_r + 1, c0 : c1 + 1] = pair["color"]
    return out


def _validate_endpoint_spacing(grid: np.ndarray) -> bool:
    endpoints = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BACKGROUND)]
    for i, (r0, c0, color0) in enumerate(endpoints):
        for r1, c1, color1 in endpoints[i + 1 :]:
            if color0 == color1:
                continue
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _validate_generated(inp: np.ndarray, out: np.ndarray, specs: list[dict[str, int]]) -> bool:
    if inp.shape != out.shape:
        return False
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not _validate_endpoint_spacing(inp):
        return False
    if np.count_nonzero(inp) != 2 * len(specs):
        return False
    if np.array_equal(out, _wrong_diagonal_output(inp)):
        return False
    if np.array_equal(out, _wrong_rectangle_output(inp)):
        return False
    if not np.array_equal(main(inp), out):
        return False

    extracted = _extract_pairs(inp)
    if len(extracted) != len(specs):
        return False
    if len(specs) > 1 and len({spec["direction"] for spec in specs}) < 2:
        return False
    if len(specs) > 1 and len({spec["length"] for spec in specs}) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    max_length = min(TRIANGLE_LENGTH_RANGE[1], h - 3, w - 3)
    if max_length < TRIANGLE_LENGTH_RANGE[0]:
        return None

    lengths = _sample_lengths(n_pairs, max_length)
    directions = _sample_directions(n_pairs)
    colors = random.sample(PALETTE, n_pairs)

    if sum(_triangle_area(length) for length in lengths) > int(MAX_FILL_RATIO * h * w):
        return None

    placement_order = list(range(n_pairs))
    random.shuffle(placement_order)
    placement_order.sort(key=lambda idx: lengths[idx], reverse=True)

    reserved = np.zeros((h, w), dtype=bool)
    placed_specs: list[dict[str, int]] = []

    for idx in placement_order:
        length = lengths[idx]
        direction = directions[idx]
        candidates = _candidate_positions((h, w), reserved, length, direction)
        if not candidates:
            return None
        start_r, start_c = random.choice(candidates)
        r0, c0, r1, c1 = _expanded_bbox(start_r, start_c, length, direction)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_specs.append(
            {
                "color": colors[idx],
                "start_r": start_r,
                "start_c": start_c,
                "length": length,
                "direction": direction,
            }
        )

    out = _render_output((h, w), placed_specs)
    inp = _render_input((h, w), placed_specs)
    return inp if _validate_generated(inp, out, placed_specs) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
