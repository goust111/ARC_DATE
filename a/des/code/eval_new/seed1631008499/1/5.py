import random
from collections import Counter, deque

import numpy as np


BG = 0
RULER = 5
TICK = 9
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _find_ruler(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    best: tuple[int, int, int, int, int, int] | None = None
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size < 5:
            continue
        if rows.size != int(rows[-1] - rows[0] + 1):
            continue
        values = {int(v) for v in grid[rows, c]}
        if not values.issubset({RULER, TICK}) or int(TICK) not in values:
            continue
        tick_rows = rows[grid[rows, c] == TICK]
        if tick_rows.size < 3:
            continue
        diffs = np.diff(tick_rows)
        if diffs.size == 0 or len({int(d) for d in diffs}) != 1:
            continue
        unit = int(diffs[0])
        if unit < 2:
            continue
        candidate = (int(rows.size), -int(c), int(c), int(rows[0]), int(rows[-1]), int(unit))
        if best is None or candidate > best:
            best = candidate
    assert best is not None, "failed to find the ruler"
    _score, _neg_c, col, top, bottom, unit = best
    return int(col), int(top), int(bottom), int(unit)


def _parse_objects(
    grid: np.ndarray, bg: int, ruler_col: int, ruler_top: int, ruler_bottom: int, unit: int
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, tuple[int, int, int, int]]]:
    bars: dict[int, tuple[int, int, int, int]] = {}
    squares: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (bg, RULER, TICK):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            area = int(len(comp))
            if height == 1 and area == width and width % unit == 0 and ruler_top <= r0 <= ruler_bottom:
                assert color not in bars, f"duplicate bar for color {color}"
                bars[color] = (int(r0), int(c0), int(r1), int(c1))
            elif height == width and area == height * width and height in (2, 3) and c0 > ruler_col:
                assert color not in squares, f"duplicate square for color {color}"
                squares[color] = (int(r0), int(c0), int(r1), int(c1))
    assert bars, "no bars found"
    assert bars.keys() == squares.keys(), "bars and squares must match by color"
    return bars, squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure horizontal bars with a ticked ruler, then expand the matching colored squares by that measured amount.

    Input structure:
    - The background is black.
    - A single vertical ruler stands on the left: grey cells form the shaft, and maroon cells mark evenly spaced ticks.
    - The middle region contains one-pixel-tall colored horizontal bars, each in a unique color.
    - The right region contains one small solid square per bar color; each square has side length 2 or 3.

    Rule:
    - Read the ruler by finding the constant row spacing between consecutive maroon ticks; that spacing is one measurement unit.
    - For each colored bar, divide its pixel length by the ruler unit to obtain its measurement value.
    - Find the square with the same color and expand its bounding box outward by that many layers on all four sides.
    - Output only the expanded squares on a black background, preserving the original grid size and each square's center.

    Key insight:
    - The maroon tick spacing defines an implicit unit of length, and the bar lengths must be converted into that unit before applying the square expansions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    ruler_col, ruler_top, ruler_bottom, unit = _find_ruler(g, bg)
    bars, squares = _parse_objects(g, bg, ruler_col, ruler_top, ruler_bottom, unit)

    out = np.full_like(g, bg)
    for color, bar_box in bars.items():
        bar_width = int(bar_box[3] - bar_box[1] + 1)
        amount = int(bar_width // unit)
        sq_r0, sq_c0, sq_r1, sq_c1 = squares[color]
        rr0 = int(sq_r0 - amount)
        cc0 = int(sq_c0 - amount)
        rr1 = int(sq_r1 + amount)
        cc1 = int(sq_c1 + amount)
        assert 0 <= rr0 <= rr1 < g.shape[0]
        assert 0 <= cc0 <= cc1 < g.shape[1]
        assert bool((out[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg).all()), "expanded squares must not overlap"
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _place_squares(
    height: int,
    width: int,
    square_start_col: int,
    sizes: list[int],
    amounts: list[int],
) -> list[tuple[int, int, int, int]] | None:
    placements: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    final_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(sizes)), key=lambda i: -(sizes[i] + 2 * amounts[i]))
    for idx in order:
        side = int(sizes[idx])
        amount = int(amounts[idx])
        row_lo = int(amount)
        row_hi = int(height - side - amount)
        col_lo = int(max(square_start_col, amount))
        col_hi = int(width - side - amount)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        candidates = [
            (int(r0), int(c0))
            for r0 in range(row_lo, row_hi + 1)
            for c0 in range(col_lo, col_hi + 1)
        ]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            final_box = (int(r0 - amount), int(c0 - amount), int(r0 + side - 1 + amount), int(c0 + side - 1 + amount))
            if any(_boxes_overlap(final_box, other, pad=1) for other in final_boxes):
                continue
            chosen = (int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1))
            placements[idx] = chosen
            final_boxes.append(final_box)
            break
        if chosen is None:
            return None
    return [box for box in placements if box is not None]


def _expected_output(
    shape: tuple[int, int],
    placements: list[tuple[int, int, int, int]],
    colors: list[int],
    amounts: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color, box, amount in zip(colors, placements, amounts):
        r0, c0, r1, c1 = box
        out[int(r0 - amount) : int(r1 + amount) + 1, int(c0 - amount) : int(c1 + amount) + 1] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice([2, 3])
    unit = random.choice([2, 3])
    colors = random.sample(OBJECT_COLORS, n_objects)
    sizes = [random.choice([2, 3]) for _ in range(n_objects)]
    amount_hi = 5 if n_objects == 2 else 4
    amounts = [random.randint(1, amount_hi) for _ in range(n_objects)]

    height = random.randint(18, 30)
    start_row = random.randint(1, 4)
    max_ticks = 1 + (height - 2 - start_row) // unit
    if max_ticks < n_objects + 1:
        return None
    tick_count = random.randint(n_objects + 1, min(max_ticks, n_objects + 4))
    tick_rows = [int(start_row + i * unit) for i in range(tick_count)]
    bar_rows = random.sample(tick_rows, n_objects)

    ruler_col = random.randint(1, 3)
    bar_start_col = int(ruler_col + 1)
    bar_widths = [int(unit * amount) for amount in amounts]
    bar_end_max = max(int(bar_start_col + width - 1) for width in bar_widths)
    square_start_col = int(bar_end_max + random.randint(2, 4))
    width_min = max(int(square_start_col + side + amount) for side, amount in zip(sizes, amounts))
    width = random.randint(max(20, width_min), 30) if width_min <= 30 else -1
    if width == -1:
        return None

    placements = _place_squares(height, width, square_start_col, sizes, amounts)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[int(tick_rows[0]) : int(tick_rows[-1]) + 1, int(ruler_col)] = RULER
    for r in tick_rows:
        grid[int(r), int(ruler_col)] = TICK

    for color, row, width_px in zip(colors, bar_rows, bar_widths):
        grid[int(row), int(bar_start_col) : int(bar_start_col + width_px)] = int(color)

    for color, box in zip(colors, placements):
        r0, c0, r1, c1 = box
        grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)

    expected = _expected_output(grid.shape, placements, colors, amounts)
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-measure expansion task")
