from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8

SOURCE_TO_COLUMN = {
    RED: 3,
    BLUE: 4,
    GREEN: 5,
}

FILL_BY_SUM = {
    1: YELLOW,
    2: ORANGE,
    3: MAGENTA,
}

EDGE_SPANS = ((2, 3), (5, 6))
MULTI_SPANS = ((3, 4), (4, 5), (2, 4), (3, 5), (4, 6))
ALL_SPANS = EDGE_SPANS + MULTI_SPANS
COUNT_PROFILES = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (3, 1, 1),
    (1, 1, 3),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (3, 2, 1),
    (1, 2, 3),
    (4, 1, 1),
    (2, 2, 2),
    (4, 2, 1),
)
MAX_RETRIES = 300
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _value_components(grid: np.ndarray, value: int) -> list[np.ndarray]:
    return _components_from_mask(grid == int(value))


def _hole_components(crop: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = crop.shape
    zero_mask = crop == int(bg)
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zero_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and zero_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    holes = zero_mask & ~exterior
    return _components_from_mask(holes)


def _fill_color(total: int) -> int:
    return FILL_BY_SUM.get(int(total), CYAN)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase left-edge color codes, count their color-coded landing columns, and fill each gray cavity from that landing-count sum.

    Input structure:
    - The background is black, the leftmost column contains red/blue/green source pixels, and gray rectangular borders enclose black cavities.
    - Each gray object is an isolated 4-connected frame whose interior occupies 2-3 rows and 2-3 columns.
    - The cavities lie around columns 2-6 so that each one covers one or more of the target landing columns 3, 4, and 5.

    Rule:
    - Treat each left-edge source pixel as sliding horizontally on its own row to a fixed color-coded landing column: red->3, blue->4, green->5, without drawing the landing pixel.
    - Count how many source pixels land in each target column.
    - For every enclosed black cavity inside a gray frame, collect the cavity's occupied columns and sum the landing counts from those columns.
    - Fill that cavity with yellow for sum 1, orange for sum 2, magenta for sum 3, and cyan for sum 4 or more; gray borders stay gray and source pixels are erased.

    Key insight:
    - The colored left-edge pixels matter only indirectly: color determines a landing column, landing-column counts become numbers, and those numbers determine each cavity's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = BG

    landing_counts = {3: 0, 4: 0, 5: 0}
    for r in range(h):
        color = int(g[r, 0])
        if color in SOURCE_TO_COLUMN:
            landing_counts[SOURCE_TO_COLUMN[color]] += 1

    out = g.copy()
    for r in range(h):
        if int(out[r, 0]) in SOURCE_TO_COLUMN:
            out[r, 0] = bg

    for comp in _value_components(g, GRAY):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        for hole in _hole_components(crop, bg):
            cols = {int(c0 + c) for _, c in hole}
            total = sum(landing_counts.get(col, 0) for col in cols)
            color = _fill_color(total)
            for rr, cc in hole:
                out[int(r0 + rr), int(c0 + cc)] = color

    return out


def _region_sum(span: tuple[int, int], counts: dict[int, int]) -> int:
    start, end = span
    return sum(counts.get(col, 0) for col in range(start, end + 1))


def _place_frame(grid: np.ndarray, top: int, span: tuple[int, int], inner_h: int) -> None:
    hole_c0, hole_c1 = span
    border_r0 = top
    border_r1 = top + inner_h + 1
    border_c0 = hole_c0 - 1
    border_c1 = hole_c1 + 1
    grid[border_r0 : border_r1 + 1, border_c0] = GRAY
    grid[border_r0 : border_r1 + 1, border_c1] = GRAY
    grid[border_r0, border_c0 : border_c1 + 1] = GRAY
    grid[border_r1, border_c0 : border_c1 + 1] = GRAY


def _choose_counts() -> dict[int, int]:
    red_count, blue_count, green_count = COUNT_PROFILES[int(RNG.integers(0, len(COUNT_PROFILES)))]
    return {3: int(red_count), 4: int(blue_count), 5: int(green_count)}


def _choose_spans(region_count: int, counts: dict[int, int]) -> list[tuple[int, int]]:
    want_three_colors = region_count == 3 and bool(RNG.integers(0, 2))
    candidates_two: list[list[tuple[int, int]]] = []
    candidates_three: list[list[tuple[int, int]]] = []

    for edge_span in EDGE_SPANS:
        for multi_span in MULTI_SPANS:
            if region_count == 2:
                spans = [edge_span, multi_span]
                RNG.shuffle(spans)
                fill_colors = {_fill_color(_region_sum(span, counts)) for span in spans}
                if len(fill_colors) >= 2:
                    candidates_two.append(spans)
                continue

            for extra_span in ALL_SPANS:
                if extra_span in {edge_span, multi_span}:
                    continue
                spans = [edge_span, multi_span, extra_span]
                RNG.shuffle(spans)
                fill_colors = {_fill_color(_region_sum(span, counts)) for span in spans}
                if len(fill_colors) >= 2:
                    candidates_two.append(spans)
                if len(fill_colors) >= 3:
                    candidates_three.append(spans)

    if want_three_colors and candidates_three:
        return candidates_three[int(RNG.integers(0, len(candidates_three)))]
    if candidates_two:
        return candidates_two[int(RNG.integers(0, len(candidates_two)))]
    raise RuntimeError("failed to choose non-degenerate spans")


def _gen_attempt() -> np.ndarray | None:
    region_count = int(RNG.integers(2, 4))
    counts = _choose_counts()
    spans = _choose_spans(region_count, counts)
    inner_heights = [int(RNG.integers(2, 4)) for _ in range(region_count)]
    frame_heights = [inner_h + 2 for inner_h in inner_heights]
    gaps = [int(RNG.integers(1, 3)) for _ in range(region_count - 1)]
    vertical_need = sum(frame_heights) + sum(gaps)
    source_total = sum(counts.values())
    height_low = max(10, vertical_need + 2, source_total + 2)
    height_high = min(16, max(height_low, vertical_need + 5))
    if height_low > height_high:
        return None
    height = int(RNG.integers(height_low, height_high + 1))
    width = int(RNG.integers(11, 16))
    grid = np.full((height, width), BG, dtype=int)

    max_right_border = max(span[1] + 1 for span in spans)
    if max_right_border >= width:
        return None

    top_margin = int(RNG.integers(1, height - vertical_need + 1))
    row = top_margin
    for idx, (span, inner_h, frame_h) in enumerate(zip(spans, inner_heights, frame_heights)):
        _place_frame(grid, row, span, inner_h)
        row += frame_h
        if idx < len(gaps):
            row += gaps[idx]

    colors = np.array(
        [RED] * counts[3] + [BLUE] * counts[4] + [GREEN] * counts[5],
        dtype=int,
    )
    RNG.shuffle(colors)
    rows = RNG.choice(height, size=len(colors), replace=False)
    grid[rows, 0] = colors

    out = main(grid)
    fills = {int(v) for v in np.unique(out) if int(v) in {YELLOW, ORANGE, MAGENTA, CYAN}}
    if len(fills) < 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
