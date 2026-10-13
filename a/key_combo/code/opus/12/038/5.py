from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

BAR_COLORS = (RED, TEAL)
GEN_RETRIES = 400
BAR_COUNT_RANGE = (2, 5)
GRID_HEIGHT_RANGE = (9, 13)
GRID_WIDTH_MAX = 18
MARKER_TOTAL_MAX = 7


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _bar_distance(bar: dict[str, int], r: int, c: int) -> int:
    return abs(c - int(bar["col"])) + max(0, int(bar["top"]) - r)


def _extract_bars(grid: np.ndarray) -> tuple[list[dict[str, int]], int]:
    h, w = grid.shape
    result_col = w - 2
    if w < 4:
        raise AssertionError("expected a penultimate result column and operand area")
    if not np.all(grid[:, result_col] == BLACK):
        raise AssertionError("penultimate column must be an empty result slot")

    bars: list[dict[str, int]] = []
    bar_mask = np.zeros_like(grid, dtype=bool)
    for c in range(w):
        if c == result_col:
            continue
        color = int(grid[h - 1, c])
        if color not in BAR_COLORS:
            continue
        r = h - 1
        while r >= 0 and int(grid[r, c]) == color:
            r -= 1
        top = r + 1
        bar_mask[top:h, c] = True
        bars.append({"color": color, "col": c, "top": top, "height": h - top})

    if not bars:
        raise AssertionError("expected at least one operand bar")

    color_mask = np.isin(grid, BAR_COLORS)
    if not np.array_equal(color_mask, bar_mask):
        raise AssertionError("all red/teal cells must belong to bottom-aligned bars")

    bars.sort(key=lambda bar: int(bar["col"]))
    cols = [int(bar["col"]) for bar in bars]
    if len(cols) != len(set(cols)):
        raise AssertionError("bars must occupy distinct columns")
    if any(int(bar["height"]) < 1 for bar in bars):
        raise AssertionError("bar heights must be positive")
    if any(cols[i + 1] - cols[i] < 2 for i in range(len(cols) - 1)):
        raise AssertionError("bars must be separated by at least one empty column")
    return bars, result_col


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid == GREEN)]
    for r, c in markers:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BLACK:
                raise AssertionError("green markers must be isolated from other objects")
    return markers


def _analyze_input(grid: np.ndarray) -> tuple[list[dict[str, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    palette = set(int(v) for v in np.unique(g))
    if not palette.issubset({BLACK, RED, GREEN, TEAL}):
        raise AssertionError("input contains unexpected colors")
    bars, result_col = _extract_bars(g)
    markers = _extract_markers(g)
    if not any(int(bar["color"]) == TEAL for bar in bars):
        raise AssertionError("expected at least one teal bar")
    if not any(int(bar["color"]) == RED for bar in bars):
        raise AssertionError("expected at least one red bar")
    return bars, markers, result_col


def _assign_markers(
    bars: list[dict[str, int]], markers: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    assignments: list[tuple[int, int, int, int]] = []
    for r, c in markers:
        distances = [_bar_distance(bar, r, c) for bar in bars]
        best = min(distances)
        winners = [i for i, dist in enumerate(distances) if dist == best]
        if len(winners) != 1:
            raise AssertionError("every marker must have a unique nearest bar")
        bar_index = winners[0]
        assignments.append((r, c, int(bars[bar_index]["color"]), bar_index))
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each green marker by its nearest bar, then compute the grey result bar from gain-adjusted bar arithmetic.

    Input structure:
    - A black background contains bottom-aligned one-cell-wide vertical bars in teal and red.
    - Teal bars contribute positively and red bars negatively; the penultimate column is an empty result slot.
    - Green single-pixel markers are scattered above or between the bars and are isolated from every other object.

    Rule:
    - For each green marker, find the unique nearest bar by Manhattan distance to the closest pixel of that bar, then recolor the marker to that bar's color.
    - Compute positive total = teal bar heights + number of recolored teal markers, and negative total = red bar heights + number of recolored red markers.
    - Keep all original bars unchanged and fill the result slot from the bottom with a grey bar whose height is positive total minus negative total.

    Key insight:
    - Marker placement changes the arithmetic because each marker joins the nearest bar's side before the final height difference is computed.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bars, markers, result_col = _analyze_input(g)
    assignments = _assign_markers(bars, markers)

    teal_total = sum(int(bar["height"]) for bar in bars if int(bar["color"]) == TEAL)
    red_total = sum(int(bar["height"]) for bar in bars if int(bar["color"]) == RED)
    teal_total += sum(1 for _r, _c, color, _idx in assignments if color == TEAL)
    red_total += sum(1 for _r, _c, color, _idx in assignments if color == RED)

    result_height = teal_total - red_total
    if not (1 <= result_height <= h):
        raise AssertionError("result height must stay within the grid")

    out = g.copy()
    for r, c, color, _bar_index in assignments:
        out[r, c] = color
    out[h - result_height : h, result_col] = GREY
    return out


def _bounded_composition(total: int, parts: int, upper: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None

    values: list[int] = []
    remaining = total
    for i in range(parts):
        slots_left = parts - i - 1
        lo = max(1, remaining - slots_left * upper)
        hi = min(upper, remaining - slots_left)
        values.append(random.randint(lo, hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _sample_bar_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 3))
    if len(candidates) < count:
        return None
    for _ in range(80):
        cols = sorted(random.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)):
            return cols
    return None


def _sample_bar_specs(height: int, count: int) -> list[tuple[int, int]] | None:
    max_height = min(6, height - 2)
    teal_count = random.randint(1, count - 1)
    red_count = count - teal_count
    if max_height < 1:
        return None

    base_choices = [d for d in (-2, -1, 0, 1, 1, 2, 2, 3, 4) if d <= height - 2]
    if not base_choices:
        return None

    for _ in range(120):
        base_result = random.choice(base_choices)
        red_lo = max(red_count, teal_count - base_result)
        red_hi = min(red_count * max_height, teal_count * max_height - base_result)
        if red_lo > red_hi:
            continue
        red_sum = random.randint(red_lo, red_hi)
        teal_sum = red_sum + base_result
        teal_heights = _bounded_composition(teal_sum, teal_count, max_height)
        red_heights = _bounded_composition(red_sum, red_count, max_height)
        if teal_heights is None or red_heights is None:
            continue
        specs = [(TEAL, h) for h in teal_heights] + [(RED, h) for h in red_heights]
        random.shuffle(specs)
        return specs
    return None


def _build_bar_grid(height: int, width: int) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    bar_count = random.randint(*BAR_COUNT_RANGE)
    min_width = max(2 * bar_count + 3, 9)
    if min_width > GRID_WIDTH_MAX:
        return None
    width = random.randint(min_width, GRID_WIDTH_MAX)
    cols = _sample_bar_columns(width, bar_count)
    specs = _sample_bar_specs(height, bar_count)
    if cols is None or specs is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    bars: list[dict[str, int]] = []
    for col, (color, bar_height) in zip(cols, specs):
        top = height - bar_height
        grid[top:height, col] = color
        bars.append({"color": color, "col": col, "top": top, "height": bar_height})
    bars.sort(key=lambda bar: int(bar["col"]))
    return grid, bars


def _candidate_map(
    grid: np.ndarray, bars: list[dict[str, int]]
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    result_col = w - 2
    by_bar = {i: [] for i in range(len(bars))}
    by_color = {RED: [], TEAL: []}
    for r in range(h - 1):
        for c in range(w):
            if c >= result_col:
                continue
            if int(grid[r, c]) != BLACK:
                continue
            isolated = True
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BLACK:
                    isolated = False
                    break
            if not isolated:
                continue
            distances = [_bar_distance(bar, r, c) for bar in bars]
            best = min(distances)
            winners = [i for i, dist in enumerate(distances) if dist == best]
            if len(winners) != 1:
                continue
            bar_index = winners[0]
            by_bar[bar_index].append((r, c))
            by_color[int(bars[bar_index]["color"])].append((r, c))
    return by_bar, by_color


def _select_marker_gain(
    base_result: int, height: int, teal_available: int, red_available: int
) -> tuple[int, int] | None:
    feasible = []
    for teal_gain in range(1, min(4, teal_available) + 1):
        for red_gain in range(1, min(4, red_available) + 1):
            total_markers = teal_gain + red_gain
            if total_markers > MARKER_TOTAL_MAX or teal_gain == red_gain:
                continue
            final_result = base_result + teal_gain - red_gain
            if 1 <= final_result <= height:
                feasible.append((teal_gain, red_gain, final_result))
    if not feasible:
        return None

    modes = ["rescue", "shrink", "boost", "boost"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "rescue":
            choices = [(t, r) for t, r, final in feasible if base_result <= 0 < final]
        elif mode == "shrink":
            choices = [(t, r) for t, r, final in feasible if base_result > 1 and 1 <= final < base_result]
        else:
            choices = [(t, r) for t, r, final in feasible if final > base_result]
        if choices:
            return random.choice(choices)
    return random.choice([(t, r) for t, r, _final in feasible])


def _pick_markers(
    by_color: dict[int, list[tuple[int, int]]], teal_gain: int, red_gain: int
) -> list[tuple[int, int, int]] | None:
    targets = [TEAL] * teal_gain + [RED] * red_gain
    for _ in range(60):
        random.shuffle(targets)
        chosen: list[tuple[int, int, int]] = []
        blocked: set[tuple[int, int]] = set()
        ok = True
        for color in targets:
            pool = by_color[color][:]
            random.shuffle(pool)
            cell = next((pos for pos in pool if pos not in blocked), None)
            if cell is None:
                ok = False
                break
            r, c = cell
            chosen.append((r, c, color))
            blocked.add((r, c))
            blocked.update(_neighbors8(r, c))
        if ok:
            return chosen
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bars, markers, result_col = _analyze_input(grid)
        if len(markers) < 2:
            return False
        assignments = _assign_markers(bars, markers)
        teal_markers = sum(1 for _r, _c, color, _idx in assignments if color == TEAL)
        red_markers = sum(1 for _r, _c, color, _idx in assignments if color == RED)
        if teal_markers == 0 or red_markers == 0:
            return False
        base_result = sum(int(bar["height"]) if int(bar["color"]) == TEAL else -int(bar["height"]) for bar in bars)
        final_result = base_result + teal_markers - red_markers
        if final_result == base_result:
            return False
        if not (1 <= final_result <= grid.shape[0]):
            return False
        out = main(grid)
        grey_cells = np.argwhere(out[:, result_col] == GREY)
        if grey_cells.size == 0:
            return False
        if len(grey_cells) != final_result:
            return False
        for r, c, color, _idx in assignments:
            if int(out[r, c]) != color:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    built = _build_bar_grid(height, GRID_WIDTH_MAX)
    if built is None:
        return None
    grid, bars = built
    by_bar, by_color = _candidate_map(grid, bars)
    if any(not cells for cells in by_bar.values()):
        return None
    if not by_color[TEAL] or not by_color[RED]:
        return None

    base_result = sum(int(bar["height"]) if int(bar["color"]) == TEAL else -int(bar["height"]) for bar in bars)
    gains = _select_marker_gain(base_result, height, len(by_color[TEAL]), len(by_color[RED]))
    if gains is None:
        return None
    teal_gain, red_gain = gains
    markers = _pick_markers(by_color, teal_gain, red_gain)
    if markers is None:
        return None

    inp = grid.copy()
    for r, c, _assigned_color in markers:
        inp[r, c] = GREEN
    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid proximity-gain bar arithmetic input")
