from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


PALETTE = list(range(10))
BAR_ORIENTATIONS = ("H", "V")
PRIMARY_PAIR_MODES = ("HV", "HH", "VV")
PRIMARY_PAIR_WEIGHTS = (4, 1, 1)
GRID_MIN = 18
GRID_MAX = 29
BAR_COUNT_CHOICES = (2, 3, 4, 5)
BAR_COUNT_WEIGHTS = (6, 5, 3, 1)
PRIMARY_LENGTH_RANGE = (7, 14)
EXTRA_LENGTH_RANGE = (5, 12)
OVERLAP_MIN = 3
OVERLAP_MAX = 48
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                colors.add(int(grid[r, c]))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            assert len(colors) == 1, "each bar must be monochromatic"
            arr = np.array(coords, dtype=int)
            r1, c1 = arr.min(axis=0)
            r2, c2 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "color": next(iter(colors)),
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(len(coords)),
                }
            )

    return comps


def _bar_cells(bar: tuple[str, int, int, int, int]) -> list[tuple[int, int]]:
    orientation, r, c, length, _color = bar
    if orientation == "H":
        return [(r, c + dc) for dc in range(length)]
    return [(r + dr, c) for dr in range(length)]


def _field_mask(shape: tuple[int, int], bar: tuple[str, int, int, int, int]) -> np.ndarray:
    orientation, r, c, length, _color = bar
    radius = (length - 1) // 2
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    if orientation == "H":
        for offset in range(radius + 1):
            left = c + offset
            right = c + length - 1 - offset
            rows = (r,) if offset == 0 else (r - offset, r + offset)
            for rr in rows:
                if 0 <= rr < h:
                    lo = max(0, left)
                    hi = min(w - 1, right)
                    if lo <= hi:
                        mask[rr, lo : hi + 1] = True
    else:
        for offset in range(radius + 1):
            top = r + offset
            bottom = r + length - 1 - offset
            cols = (c,) if offset == 0 else (c - offset, c + offset)
            for cc in cols:
                if 0 <= cc < w:
                    lo = max(0, top)
                    hi = min(h - 1, bottom)
                    if lo <= hi:
                        mask[lo : hi + 1, cc] = True

    return mask


def _bar_mask(shape: tuple[int, int], bar: tuple[str, int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in _bar_cells(bar):
        mask[r, c] = True
    return mask


def _coverage_map(shape: tuple[int, int], bars: list[tuple[str, int, int, int, int]]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for bar in bars:
        cover += _field_mask(shape, bar).astype(int)
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the background cells covered by at least two invisible bar-centered diamonds in teal.

    Input structure:
    - The grid has one background color and several one-cell-thick solid horizontal or vertical bars.
    - Each bar is monochromatic, bars use distinct non-background colors, and bars do not touch.
    - Teal does not appear in the input and is reserved for the interference region in the output.

    Rule:
    - Keep every original bar unchanged.
    - Each bar of length L defines an invisible solid diamond field of radius floor((L - 1) / 2), with the bar as its center line and each step away shrinking both ends by one cell.
    - Paint a background cell teal iff it belongs to at least two such invisible fields; cells covered by zero or one field stay background.

    Key insight:
    - The visible output is not any single bar's envelope; it is only the overlap of multiple reconstructed hidden diamond fields.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert not np.any(g == Color.TEAL), "teal is reserved for output only"

    bg = _get_bg(g)
    bars_raw = _find_components(g, bg)
    bars: list[tuple[str, int, int, int, int]] = []

    for comp in bars_raw:
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert area == height * width, "each foreground object must be solid"
        assert height == 1 or width == 1, "each foreground object must be a straight bar"
        orientation = "H" if height == 1 else "V"
        length = max(height, width)
        bars.append((orientation, int(r0), int(c0), int(length), int(comp["color"])))

    cover = _coverage_map(g.shape, bars)
    out = g.copy()
    out[(g == bg) & (cover >= 2)] = Color.TEAL
    return out


def _sample_bar(
    shape: tuple[int, int],
    color: int,
    orientation: str | None = None,
    length_range: tuple[int, int] = PRIMARY_LENGTH_RANGE,
) -> tuple[str, int, int, int, int] | None:
    h, w = shape
    orientation = orientation or random.choice(BAR_ORIENTATIONS)

    legal_lengths = []
    for length in range(length_range[0], length_range[1]):
        radius = (length - 1) // 2
        if orientation == "H":
            if h >= 2 * radius + 1 and w >= length:
                legal_lengths.append(length)
        else:
            if h >= length and w >= 2 * radius + 1:
                legal_lengths.append(length)
    if not legal_lengths:
        return None

    length = random.choice(legal_lengths)
    radius = (length - 1) // 2
    if orientation == "H":
        r = random.randint(radius, h - radius - 1)
        c = random.randint(0, w - length)
    else:
        r = random.randint(0, h - length)
        c = random.randint(radius, w - radius - 1)
    return (orientation, r, c, length, color)


def _bars_touch(bar_a: tuple[str, int, int, int, int], bar_b: tuple[str, int, int, int, int]) -> bool:
    cells_b = set(_bar_cells(bar_b))
    for r, c in _bar_cells(bar_a):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in cells_b:
                    return True
    return False


def _can_add_bar(candidate: tuple[str, int, int, int, int], bars: list[tuple[str, int, int, int, int]]) -> bool:
    return all(not _bars_touch(candidate, bar) for bar in bars)


def _overlap_bg_count(shape: tuple[int, int], bars: list[tuple[str, int, int, int, int]]) -> int:
    cover = _coverage_map(shape, bars)
    bar_pixels = np.zeros(shape, dtype=bool)
    for bar in bars:
        bar_pixels |= _bar_mask(shape, bar)
    return int(np.count_nonzero((cover >= 2) & ~bar_pixels))


def _render_input(shape: tuple[int, int], bg: int, bars: list[tuple[str, int, int, int, int]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for orientation, r, c, length, color in bars:
        if orientation == "H":
            grid[r, c : c + length] = color
        else:
            grid[r : r + length, c] = color
    return grid


def _pick_primary_bars(
    shape: tuple[int, int], colors: list[int]
) -> list[tuple[str, int, int, int, int]] | None:
    mode = random.choices(PRIMARY_PAIR_MODES, weights=PRIMARY_PAIR_WEIGHTS, k=1)[0]
    first_orientation = mode[0]
    second_orientation = mode[1]

    for _ in range(140):
        first = _sample_bar(shape, colors[0], orientation=first_orientation)
        if first is None:
            return None

        for _ in range(220):
            second = _sample_bar(shape, colors[1], orientation=second_orientation)
            if second is None or not _can_add_bar(second, [first]):
                continue
            overlap = _overlap_bg_count(shape, [first, second])
            if OVERLAP_MIN <= overlap <= OVERLAP_MAX:
                return [first, second]

    return None


def _add_extra_bars(
    shape: tuple[int, int],
    bars: list[tuple[str, int, int, int, int]],
    colors: list[int],
) -> list[tuple[str, int, int, int, int]] | None:
    for color in colors:
        placed = None
        for _ in range(160):
            candidate = _sample_bar(shape, color, length_range=EXTRA_LENGTH_RANGE)
            if candidate is None or not _can_add_bar(candidate, bars):
                continue
            trial = bars + [candidate]
            if _overlap_bg_count(shape, trial) > OVERLAP_MAX:
                continue
            placed = candidate
            break
        if placed is None:
            return None
        bars.append(placed)
    return bars


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    bg = random.choice([color for color in PALETTE if color != Color.TEAL])
    bar_count = random.choices(BAR_COUNT_CHOICES, weights=BAR_COUNT_WEIGHTS, k=1)[0]
    palette = [color for color in PALETTE if color not in {bg, Color.TEAL}]
    random.shuffle(palette)
    colors = palette[:bar_count]

    bars = _pick_primary_bars(shape, colors[:2])
    if bars is None:
        return None

    if bar_count > 2:
        bars = _add_extra_bars(shape, bars, colors[2:])
        if bars is None:
            return None

    inp = _render_input(shape, bg, bars)
    out = main(inp)
    overlap = int(np.count_nonzero(out == Color.TEAL))
    if not (OVERLAP_MIN <= overlap <= OVERLAP_MAX):
        return None

    if len(_find_components(inp, bg)) != bar_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
