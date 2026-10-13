import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_SIDES = (13, 15, 17)
LENGTH_OPTIONS = (3, 5, 7, 9)
LENGTH_WEIGHTS = (7, 5, 3, 1)
COUNT_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500
MAX_PLACEMENT_TRIES = 250


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _parse_bars(grid: np.ndarray, color: int, expected_orientation: str) -> list[dict]:
    bars = []
    for coords in _find_components(grid, color):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = int(coords.shape[0])
        if expected_orientation == "H":
            assert height == 1 and width == area, "blue components must be one-pixel-thick horizontal bars"
        else:
            assert width == 1 and height == area, "red components must be one-pixel-thick vertical bars"
        length = max(height, width)
        assert length % 2 == 1 and 3 <= length <= 9, "bars must have odd length 3..9"
        bars.append(
            {
                "color": int(color),
                "orientation": expected_orientation,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "length": int(length),
            }
        )
    return bars


def _bar_input_cells(bar: dict) -> set[tuple[int, int]]:
    r, c = bar["center"]
    half = bar["length"] // 2
    if bar["orientation"] == "H":
        return {(int(r), int(c + d)) for d in range(-half, half + 1)}
    return {(int(r + d), int(c)) for d in range(-half, half + 1)}


def _bar_rotated_cells(bar: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = bar["center"]
    half = bar["length"] // 2
    cells = set()
    if bar["orientation"] == "H":
        for nr in range(r - half, r + half + 1):
            if 0 <= nr < h and 0 <= c < w:
                cells.add((int(nr), int(c)))
    else:
        for nc in range(c - half, c + half + 1):
            if 0 <= r < h and 0 <= nc < w:
                cells.add((int(r), int(nc)))
    return cells


def _render_rotated_output(bars: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    blue_cov = np.zeros((h, w), dtype=int)
    red_cov = np.zeros((h, w), dtype=int)
    for bar in bars:
        rotated = _bar_rotated_cells(bar, shape)
        if bar["color"] == BLUE:
            for r, c in rotated:
                blue_cov[r, c] += 1
        else:
            for r, c in rotated:
                red_cov[r, c] += 1

    out = np.full((h, w), BLACK, dtype=int)
    out[(blue_cov > 0) & (red_cov == 0)] = BLUE
    out[(red_cov > 0) & (blue_cov == 0)] = RED
    out[(blue_cov > 0) & (red_cov > 0)] = GREEN
    return out, blue_cov, red_cov


def _render_input_overlay(bars: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    blue_cov = np.zeros((h, w), dtype=int)
    red_cov = np.zeros((h, w), dtype=int)
    for bar in bars:
        for r, c in _bar_input_cells(bar):
            if bar["color"] == BLUE:
                blue_cov[r, c] += 1
            else:
                red_cov[r, c] += 1
    out = np.full((h, w), BLACK, dtype=int)
    out[(blue_cov > 0) & (red_cov == 0)] = BLUE
    out[(red_cov > 0) & (blue_cov == 0)] = RED
    out[(blue_cov > 0) & (red_cov > 0)] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate horizontal blue bars and vertical red bars 90 degrees around their centers, then mark rotated overlaps green.

    Input structure:
    - The grid has black background with one-pixel-thick blue and red bars.
    - Every blue component is a horizontal bar and every red component is a vertical bar.
    - Each bar has odd length, so its center pixel is well-defined.

    Rule:
    - Detect every blue horizontal bar and red vertical bar, recording each bar's center and length.
    - Rotate every bar by 90 degrees about its own center while keeping the same canvas, center, and length.
    - Clear the canvas to black and redraw the rotated bars.
    - A pixel covered only by rotated blue stays blue, only by rotated red stays red, and by both becomes green.

    Key insight:
    - The crucial intersections are not the input bars themselves but the crossings created only after swapping each bar's orientation around its fixed center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED}

    blue_bars = _parse_bars(g, BLUE, "H")
    red_bars = _parse_bars(g, RED, "V")
    assert blue_bars and red_bars

    out, _blue_cov, _red_cov = _render_rotated_output(blue_bars + red_bars, g.shape)
    return out


def _cells_are_separated(cell_sets: list[set[tuple[int, int]]]) -> bool:
    for i, first in enumerate(cell_sets):
        for second in cell_sets[i + 1 :]:
            for r, c in first:
                for nr, nc in _neighbors8(r, c):
                    if (nr, nc) in second:
                        return False
    return True


def _sample_side(total_bars: int) -> int:
    if total_bars >= 9:
        choices = [15, 17]
    elif total_bars >= 8:
        choices = [13, 15, 17]
    else:
        choices = list(GRID_SIDES)
    return random.choice(choices)


def _sample_length(allowed: tuple[int, ...] = LENGTH_OPTIONS) -> int:
    options = [length for length in LENGTH_OPTIONS if length in allowed]
    weights = [LENGTH_WEIGHTS[LENGTH_OPTIONS.index(length)] for length in options]
    return random.choices(options, weights=weights, k=1)[0]


def _bar_fits(bar: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = bar["center"]
    half = bar["length"] // 2
    if bar["orientation"] == "H":
        return half <= r < h - half and half <= c < w - half
    return half <= r < h - half and half <= c < w - half


def _candidate_bar(shape: tuple[int, int], color: int, allowed_lengths: tuple[int, ...] = LENGTH_OPTIONS) -> dict:
    h, w = shape
    length = _sample_length(allowed_lengths)
    half = length // 2
    r = random.randint(half, h - 1 - half)
    c = random.randint(half, w - 1 - half)
    return {
        "color": int(color),
        "orientation": "H" if color == BLUE else "V",
        "center": (int(r), int(c)),
        "length": int(length),
    }


def _can_add_bar(bar: dict, placed: list[dict], shape: tuple[int, int]) -> bool:
    if not _bar_fits(bar, shape):
        return False

    new_input = _bar_input_cells(bar)
    existing_inputs = [_bar_input_cells(existing) for existing in placed]
    if not _cells_are_separated(existing_inputs + [new_input]):
        return False

    new_rotated = _bar_rotated_cells(bar, shape)
    for existing in placed:
        if existing["color"] != bar["color"]:
            continue
        if new_rotated & _bar_rotated_cells(existing, shape):
            return False
    return True


def _forced_overlap_pair(shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    for _ in range(MAX_PLACEMENT_TRIES):
        mode = random.choice(("blue_short", "red_short"))
        if mode == "blue_short":
            blue_length = _sample_length((3, 5))
            blue_half = blue_length // 2
            red_allowed = tuple(length for length in LENGTH_OPTIONS if length // 2 >= blue_half + 2)
            if not red_allowed:
                continue
            red_length = _sample_length(red_allowed)
            red_half = red_length // 2
            delta_c = random.randint(blue_half + 2, red_half)
            delta_r = random.randint(-blue_half, blue_half)

            rb = random.randint(red_half, h - 1 - red_half)
            rr = rb + delta_r
            if not (red_half <= rr < h - red_half and blue_half <= rb < h - blue_half):
                continue

            sign = random.choice((-1, 1))
            cb_low = max(blue_half, red_half - sign * delta_c)
            cb_high = min(w - 1 - blue_half, w - 1 - red_half - sign * delta_c)
            if cb_low > cb_high:
                continue
            cb = random.randint(cb_low, cb_high)
            cr = cb + sign * delta_c
        else:
            red_length = _sample_length((3, 5))
            red_half = red_length // 2
            blue_allowed = tuple(length for length in LENGTH_OPTIONS if length // 2 >= red_half + 2)
            if not blue_allowed:
                continue
            blue_length = _sample_length(blue_allowed)
            blue_half = blue_length // 2
            delta_r = random.randint(red_half + 2, blue_half)
            delta_c = random.randint(-red_half, red_half)

            cb = random.randint(blue_half, w - 1 - blue_half)
            cr = cb + delta_c
            if not (blue_half <= cb < w - blue_half and red_half <= cr < w - red_half):
                continue

            sign = random.choice((-1, 1))
            rb_low = max(blue_half, red_half - sign * delta_r)
            rb_high = min(h - 1 - blue_half, h - 1 - red_half - sign * delta_r)
            if rb_low > rb_high:
                continue
            rb = random.randint(rb_low, rb_high)
            rr = rb + sign * delta_r

        blue_bar = {
            "color": BLUE,
            "orientation": "H",
            "center": (int(rb), int(cb)),
            "length": int(blue_length),
        }
        red_bar = {
            "color": RED,
            "orientation": "V",
            "center": (int(rr), int(cr)),
            "length": int(red_length),
        }
        if _can_add_bar(blue_bar, [], shape) and _can_add_bar(red_bar, [blue_bar], shape):
            out, blue_cov, red_cov = _render_rotated_output([blue_bar, red_bar], shape)
            if int(((blue_cov > 0) & (red_cov > 0)).sum()) >= 1 and np.any(out == GREEN):
                return [blue_bar, red_bar]
    return None


def _build_grid_from_bars(bars: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for bar in bars:
        for r, c in _bar_input_cells(bar):
            grid[r, c] = int(bar["color"])
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BLACK, BLUE, RED}

    blue_bars = _parse_bars(grid, BLUE, "H")
    red_bars = _parse_bars(grid, RED, "V")
    assert 3 <= len(blue_bars) <= 5
    assert 3 <= len(red_bars) <= 5

    all_bars = blue_bars + red_bars
    input_cells = [_bar_input_cells(bar) for bar in all_bars]
    assert _cells_are_separated(input_cells), "input bars must be 8-separated"

    out, blue_cov, red_cov = _render_rotated_output(all_bars, grid.shape)
    assert int(blue_cov.max(initial=0)) <= 1, "rotated blue bars should remain individually distinguishable"
    assert int(red_cov.max(initial=0)) <= 1, "rotated red bars should remain individually distinguishable"

    green_mask = (blue_cov > 0) & (red_cov > 0)
    assert int(green_mask.sum()) >= 1, "rotation must create at least one blue/red overlap"
    assert np.any((blue_cov > 0) & (red_cov == 0)), "need visible blue-only rotated pixels"
    assert np.any((red_cov > 0) & (blue_cov == 0)), "need visible red-only rotated pixels"

    pair_overlap_counts = []
    for blue_bar in blue_bars:
        overlaps = 0
        blue_rot = _bar_rotated_cells(blue_bar, grid.shape)
        for red_bar in red_bars:
            if blue_rot & _bar_rotated_cells(red_bar, grid.shape):
                overlaps += 1
        pair_overlap_counts.append(overlaps)
    for red_bar in red_bars:
        overlaps = 0
        red_rot = _bar_rotated_cells(red_bar, grid.shape)
        for blue_bar in blue_bars:
            if red_rot & _bar_rotated_cells(blue_bar, grid.shape):
                overlaps += 1
        pair_overlap_counts.append(overlaps)
    assert any(count == 0 for count in pair_overlap_counts), "not every bar should participate in an overlap"
    assert any(count > 0 for count in pair_overlap_counts), "at least one bar must participate in an overlap"

    original_overlay = _render_input_overlay(all_bars, grid.shape)
    assert not np.array_equal(out, original_overlay), "removing rotation must change the result"
    assert not np.array_equal(out, grid), "output must differ from input"
    assert np.array_equal(main(grid), out), "main() must implement the intended transformation"


def _attempt_generate() -> np.ndarray | None:
    blue_count = random.choice(COUNT_OPTIONS)
    red_count = random.choice(COUNT_OPTIONS)
    side = _sample_side(blue_count + red_count)
    shape = (side, side)

    placed = _forced_overlap_pair(shape)
    if placed is None:
        return None

    need_blue = blue_count - 1
    need_red = red_count - 1
    color_order = [BLUE] * need_blue + [RED] * need_red
    random.shuffle(color_order)

    for color in color_order:
        added = False
        for _ in range(MAX_PLACEMENT_TRIES):
            allowed_lengths = LENGTH_OPTIONS
            if side == 13 and color == BLUE and sum(bar["color"] == BLUE for bar in placed) >= 3:
                allowed_lengths = (3, 5, 7)
            candidate = _candidate_bar(shape, color, allowed_lengths)
            if _can_add_bar(candidate, placed, shape):
                placed.append(candidate)
                added = True
                break
        if not added:
            return None

    grid = _build_grid_from_bars(placed, shape)
    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-overlap ARC input")
