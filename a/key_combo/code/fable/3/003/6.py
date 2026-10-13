from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        comp = [(sr, sc)]
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
                    comp.append((rr, cc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rectangles_and_dots(
    g: np.ndarray,
) -> tuple[list[tuple[int, tuple[int, int, int, int], str, int]], dict[int, int]]:
    rectangles: list[tuple[int, tuple[int, int, int, int], str, int]] = []
    dot_counts: dict[int, int] = defaultdict(int)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components_mask(g == color):
            if len(comp) == 1:
                dot_counts[int(color)] += 1
                continue

            box = _bbox(comp)
            r0, c0, r1, c1 = box
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert len(comp) == height * width, "rectangle components must be solid"

            if height == 3 and width > 3 and width % 2 == 1:
                rectangles.append((int(color), box, "h", int(width)))
            elif width == 3 and height > 3 and height % 2 == 1:
                rectangles.append((int(color), box, "v", int(height)))
            else:
                raise AssertionError("expected every non-dot component to be a 3-wide odd rectangle")

    assert rectangles, "expected at least one rectangle"
    return rectangles, dot_counts


def _paint_rect_with_holes(
    out: np.ndarray,
    color: int,
    box: tuple[int, int, int, int],
    orient: str,
    long_len: int,
    period: int,
) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    if period <= 0:
        return

    if orient == "h":
        mid_r = r0 + 1
        for pos in range(1, long_len, int(period)):
            out[mid_r, c0 + pos] = BG
    else:
        mid_c = c0 + 1
        for pos in range(1, long_len, int(period)):
            out[r0 + pos, mid_c] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch count-spaced holes in each 3-wide rectangle using same-color partner dots as the period.

    Input structure:
    - A black background contains several separated monochrome solid rectangles.
    - Every rectangle has short side 3 and an odd long side, and may be horizontal or vertical.
    - Some rectangle colors also appear as isolated one-pixel partner dots elsewhere on the background.

    Rule:
    - For each rectangle, count the isolated single-pixel components of the same color.
    - If the count is n > 0, canonicalize the rectangle along its long axis and set center-line
      positions 1, 1+n, 1+2n, ... to black.
    - Rectangles with no same-color partner dots are unchanged, and all partner dots are erased.

    Key insight:
    - The scattered same-color pixels are count instructions, not decoration; the familiar
      alternating hole pattern is only the special case n = 2.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids are at most 30x30"

    rectangles, dot_counts = _find_rectangles_and_dots(g)
    out = np.full_like(g, BG)
    for color, box, orient, long_len in rectangles:
        _paint_rect_with_holes(out, color, box, orient, long_len, int(dot_counts.get(color, 0)))
    return out


def _odd_at_least(min_len: int) -> int:
    value = max(5, int(min_len))
    if value % 2 == 0:
        value += 1
    return value


def _shape_for(orient: str, long_len: int) -> tuple[int, int]:
    return (3, int(long_len)) if orient == "h" else (int(long_len), 3)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    rows, cols = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(rows, r0 + h + padding)
    cc1 = min(cols, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_rect(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    orient: str,
    long_len: int,
) -> tuple[int, int, int, int] | None:
    h, w = _shape_for(orient, long_len)
    rows, cols = grid.shape
    origins = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
    random.shuffle(origins)
    for r0, c0 in origins:
        if not _can_place(occupied, r0, c0, h, w, padding=1):
            continue
        grid[r0 : r0 + h, c0 : c0 + w] = int(color)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)
    return None


def _place_dot(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
) -> tuple[int, int] | None:
    rows, cols = grid.shape
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    for r, c in cells:
        if not _can_place(occupied, r, c, 1, 1, padding=1):
            continue
        grid[r, c] = int(color)
        occupied[r, c] = True
        return int(r), int(c)
    return None


def _hole_positions(long_len: int, period: int) -> set[int]:
    if period <= 0:
        return set()
    return {int(pos) for pos in range(1, int(long_len), int(period))}


def _expected_from_specs(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        _paint_rect_with_holes(
            out,
            int(spec["color"]),
            spec["box"],  # type: ignore[arg-type]
            str(spec["orient"]),
            int(spec["long_len"]),
            int(spec["period"]),
        )
    return out


def _valid_generation(inp: np.ndarray, expected: np.ndarray, specs: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    periods = [int(spec["period"]) for spec in specs]
    if 0 not in periods or 2 not in periods or not any(p not in (0, 2) for p in periods):
        return False

    if {str(spec["orient"]) for spec in specs} != {"h", "v"}:
        return False

    partner_specs = [spec for spec in specs if int(spec["period"]) > 0]
    if any(len(_hole_positions(int(spec["long_len"]), int(spec["period"]))) < 2 for spec in partner_specs):
        return False

    same_len_partner_periods: dict[int, set[int]] = defaultdict(set)
    for spec in partner_specs:
        same_len_partner_periods[int(spec["long_len"])].add(int(spec["period"]))
    if not any(len(values) >= 2 for values in same_len_partner_periods.values()):
        return False

    for spec in partner_specs:
        period = int(spec["period"])
        if period != 2 and _hole_positions(int(spec["long_len"]), period) == _hole_positions(int(spec["long_len"]), 2):
            return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_specs(n_rects: int) -> list[dict[str, object]]:
    colors = random.sample(PALETTE, n_rects)
    non_two_period = random.choice([3, 4, 5])
    shared_len = _odd_at_least(2 * non_two_period + 3)

    periods = [2, non_two_period, 0]
    while len(periods) < n_rects:
        periods.append(random.choice([0, 2, 3, 4, 5]))
    random.shuffle(periods)

    orientations = ["h", "v"]
    while len(orientations) < n_rects:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    partner_assigned = 0
    specs: list[dict[str, object]] = []
    for idx, (color, period, orient) in enumerate(zip(colors, periods, orientations)):
        if period > 0 and partner_assigned < 2:
            long_len = shared_len
            partner_assigned += 1
        elif period > 0:
            long_len = random.choice([n for n in (7, 9, 11, 13, 15, 17) if n >= _odd_at_least(2 * period + 3)])
        else:
            long_len = shared_len if idx == 0 or random.random() < 0.35 else random.choice([5, 7, 9, 11, 13])
        specs.append(
            {
                "color": int(color),
                "period": int(period),
                "orient": str(orient),
                "long_len": int(long_len),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    n_rects = random.randint(3, 5)
    specs = _gen_specs(n_rects)

    grid = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    # Place larger rectangles first so the forced same-length counterexamples are easy to fit.
    for spec in sorted(specs, key=lambda s: int(s["long_len"]), reverse=True):
        box = _place_rect(
            grid,
            occupied,
            int(spec["color"]),
            str(spec["orient"]),
            int(spec["long_len"]),
        )
        if box is None:
            return None
        spec["box"] = box

    dot_cells: list[tuple[int, int]] = []
    dot_specs = specs[:]
    random.shuffle(dot_specs)
    for spec in dot_specs:
        color = int(spec["color"])
        for _ in range(int(spec["period"])):
            cell = _place_dot(grid, occupied, color)
            if cell is None:
                return None
            dot_cells.append(cell)

    expected = _expected_from_specs(grid.shape, specs)
    if not _valid_generation(grid, expected, specs):
        return None
    if any(int(expected[r, c]) != BG for r, c in dot_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partner-count hole-punching input")
