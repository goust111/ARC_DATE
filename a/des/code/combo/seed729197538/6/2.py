from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
GRAY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


@dataclass(frozen=True)
class Rectangle:
    r0: int
    c0: int
    r1: int
    c1: int
    color: int


@dataclass(frozen=True)
class PairSpec:
    orientation: str  # "h" or "v"
    a: tuple[int, int]
    b: tuple[int, int]
    color: int

    def segment_cells(self) -> list[tuple[int, int]]:
        if self.orientation == "h":
            r = self.a[0]
            return [(r, c) for c in range(self.a[1] + 1, self.b[1])]
        c = self.a[1]
        return [(r, c) for r in range(self.a[0] + 1, self.b[0])]


def _true_run_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.count_nonzero(starts))


def _detect_path_color(segment: np.ndarray) -> int | None:
    flat = np.asarray(segment, dtype=int).reshape(-1)
    if flat.size == 0 or np.any(flat == GRAY):
        return None
    colored = (flat != BG) & (flat != GRAY)
    if not np.any(colored):
        return None
    if _true_run_count(colored) != 1:
        return None
    colors = np.unique(flat[colored])
    if colors.size != 1:
        return None
    return int(colors[0])


def _collect_pairs(grid: np.ndarray) -> list[PairSpec]:
    gray_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GRAY)]
    assert gray_coords, "expected at least one gray endpoint pair"

    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in gray_coords:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    usage = {coord: 0 for coord in gray_coords}
    pairs: list[PairSpec] = []

    for r, row_cols in rows.items():
        assert len(row_cols) <= 2, "a row cannot contain more than two gray pixels"
        if len(row_cols) == 2:
            c0, c1 = sorted(row_cols)
            pairs.append(PairSpec("h", (r, c0), (r, c1), color=-1))
            usage[(r, c0)] += 1
            usage[(r, c1)] += 1

    for c, col_rows in cols.items():
        assert len(col_rows) <= 2, "a column cannot contain more than two gray pixels"
        if len(col_rows) == 2:
            r0, r1 = sorted(col_rows)
            pairs.append(PairSpec("v", (r0, c), (r1, c), color=-1))
            usage[(r0, c)] += 1
            usage[(r1, c)] += 1

    assert pairs, "expected at least one aligned gray pair"
    assert all(count == 1 for count in usage.values()), "gray pixels must form disjoint row/column pairs"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for pair in _collect_pairs(g):
        if pair.orientation == "h":
            r = pair.a[0]
            c0, c1 = pair.a[1], pair.b[1]
            segment = g[r, c0 + 1 : c1]
            color = _detect_path_color(segment)
            assert color is not None, "horizontal pair must cross exactly one colored rectangle"
            out[r, c0 + 1 : c1] = color
        else:
            c = pair.a[1]
            r0, r1 = pair.a[0], pair.b[0]
            segment = g[r0 + 1 : r1, c]
            color = _detect_path_color(segment)
            assert color is not None, "vertical pair must cross exactly one colored rectangle"
            out[r0 + 1 : r1, c] = color
    return out


GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (12, 23)
RECT_COUNT_CHOICES = (1, 2, 3, 4)
RECT_COUNT_WEIGHTS = (2, 4, 4, 2)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 6)
RECT_BORDER_MARGIN = 1
RECT_GAP = 1

PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 5
PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 400


def _boxes_with_gap(a: Rectangle, b: Rectangle, gap: int) -> bool:
    return (
        a.r1 + gap < b.r0
        or b.r1 + gap < a.r0
        or a.c1 + gap < b.c0
        or b.c1 + gap < a.c0
    )


def _place_rectangles(height: int, width: int) -> tuple[np.ndarray, list[Rectangle]] | None:
    count = random.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(RECT_COLORS, count)
    grid = np.full((height, width), BG, dtype=int)
    rects: list[Rectangle] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            rh = random.randint(*RECT_H_RANGE)
            rw = random.randint(*RECT_W_RANGE)
            if rh > height - 2 * RECT_BORDER_MARGIN or rw > width - 2 * RECT_BORDER_MARGIN:
                continue
            r0 = random.randint(RECT_BORDER_MARGIN, height - rh - RECT_BORDER_MARGIN)
            c0 = random.randint(RECT_BORDER_MARGIN, width - rw - RECT_BORDER_MARGIN)
            rect = Rectangle(r0, c0, r0 + rh - 1, c0 + rw - 1, color)
            if any(not _boxes_with_gap(rect, other, RECT_GAP) for other in rects):
                continue
            rects.append(rect)
            grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color
            placed = True
            break
        if not placed:
            return None

    return grid, rects


def _scan_left_bg(grid: np.ndarray, r: int, c: int) -> list[int]:
    cols = []
    cur = c
    while cur >= 0 and int(grid[r, cur]) == BG:
        cols.append(cur)
        cur -= 1
    return cols


def _scan_right_bg(grid: np.ndarray, r: int, c: int) -> list[int]:
    cols = []
    cur = c
    width = grid.shape[1]
    while cur < width and int(grid[r, cur]) == BG:
        cols.append(cur)
        cur += 1
    return cols


def _scan_up_bg(grid: np.ndarray, r: int, c: int) -> list[int]:
    rows = []
    cur = r
    while cur >= 0 and int(grid[cur, c]) == BG:
        rows.append(cur)
        cur -= 1
    return rows


def _scan_down_bg(grid: np.ndarray, r: int, c: int) -> list[int]:
    rows = []
    cur = r
    height = grid.shape[0]
    while cur < height and int(grid[cur, c]) == BG:
        rows.append(cur)
        cur += 1
    return rows


def _pair_segment(grid: np.ndarray, pair: PairSpec) -> np.ndarray:
    if pair.orientation == "h":
        return grid[pair.a[0], pair.a[1] + 1 : pair.b[1]]
    return grid[pair.a[0] + 1 : pair.b[0], pair.a[1]]


def _pair_is_valid(grid: np.ndarray, pair: PairSpec, line_colors: np.ndarray) -> bool:
    segment = _pair_segment(grid, pair)
    if _detect_path_color(segment) != pair.color:
        return False
    if not np.any(segment == BG):
        return False
    for r, c in pair.segment_cells():
        prev = int(line_colors[r, c])
        if prev == -1:
            continue
        if prev == pair.color and int(grid[r, c]) == pair.color:
            continue
        return False
    return True


def _enumerate_candidates(
    grid: np.ndarray,
    rects: list[Rectangle],
    reserved_rows: set[int],
    reserved_cols: set[int],
    line_colors: np.ndarray,
) -> list[PairSpec]:
    candidates: list[PairSpec] = []

    for rect in rects:
        for r in range(rect.r0, rect.r1 + 1):
            if r in reserved_rows:
                continue
            left = [c for c in _scan_left_bg(grid, r, rect.c0 - 1) if c not in reserved_cols]
            right = [c for c in _scan_right_bg(grid, r, rect.c1 + 1) if c not in reserved_cols]
            for c_left in left:
                for c_right in right:
                    pair = PairSpec("h", (r, c_left), (r, c_right), rect.color)
                    if _pair_is_valid(grid, pair, line_colors):
                        candidates.append(pair)

        for c in range(rect.c0, rect.c1 + 1):
            if c in reserved_cols:
                continue
            up = [r for r in _scan_up_bg(grid, rect.r0 - 1, c) if r not in reserved_rows]
            down = [r for r in _scan_down_bg(grid, rect.r1 + 1, c) if r not in reserved_rows]
            for r_top in up:
                for r_bottom in down:
                    pair = PairSpec("v", (r_top, c), (r_bottom, c), rect.color)
                    if _pair_is_valid(grid, pair, line_colors):
                        candidates.append(pair)

    return candidates


def _apply_pair(input_grid: np.ndarray, pair: PairSpec, line_colors: np.ndarray) -> None:
    ar, ac = pair.a
    br, bc = pair.b
    input_grid[ar, ac] = GRAY
    input_grid[br, bc] = GRAY
    for r, c in pair.segment_cells():
        if line_colors[r, c] == -1:
            line_colors[r, c] = pair.color
        else:
            assert line_colors[r, c] == pair.color and int(input_grid[r, c]) == pair.color


def _capacity_by_orientation(candidates: list[PairSpec]) -> tuple[int, int]:
    horizontal_rows = {pair.a[0] for pair in candidates if pair.orientation == "h"}
    vertical_cols = {pair.a[1] for pair in candidates if pair.orientation == "v"}
    return len(horizontal_rows), len(vertical_cols)


def _choose_mode(h_cap: int, v_cap: int) -> str | None:
    choices: list[str] = []
    weights: list[int] = []
    if h_cap > 0:
        choices.append("h")
        weights.append(3)
    if v_cap > 0:
        choices.append("v")
        weights.append(3)
    if h_cap > 0 and v_cap > 0:
        choices.append("mixed")
        weights.append(4)
    if not choices:
        return None
    return random.choices(choices, weights=weights, k=1)[0]


def _plan_pairs(base_grid: np.ndarray, rects: list[Rectangle]) -> tuple[np.ndarray, list[PairSpec]] | None:
    grid = base_grid.copy()
    line_colors = np.full(grid.shape, -1, dtype=int)
    reserved_rows: set[int] = set()
    reserved_cols: set[int] = set()

    initial = _enumerate_candidates(grid, rects, reserved_rows, reserved_cols, line_colors)
    if not initial:
        return None

    h_cap, v_cap = _capacity_by_orientation(initial)
    mode = _choose_mode(h_cap, v_cap)
    if mode is None:
        return None

    if mode == "h":
        target = random.randint(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, h_cap))
    elif mode == "v":
        target = random.randint(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, v_cap))
    else:
        max_total = min(PAIR_COUNT_MAX, h_cap + v_cap)
        if max_total < 2:
            return None
        target = random.randint(2, max_total)

    pairs: list[PairSpec] = []
    for _ in range(target):
        candidates = _enumerate_candidates(grid, rects, reserved_rows, reserved_cols, line_colors)
        if not candidates:
            return None

        if mode == "h":
            pool = [pair for pair in candidates if pair.orientation == "h"]
        elif mode == "v":
            pool = [pair for pair in candidates if pair.orientation == "v"]
        else:
            existing = {pair.orientation for pair in pairs}
            if "h" not in existing:
                pool = [pair for pair in candidates if pair.orientation == "h"]
            elif "v" not in existing:
                pool = [pair for pair in candidates if pair.orientation == "v"]
            else:
                pool = candidates

        if not pool:
            return None

        pair = random.choice(pool)
        pairs.append(pair)
        _apply_pair(grid, pair, line_colors)
        if pair.orientation == "h":
            reserved_rows.add(pair.a[0])
            reserved_cols.add(pair.a[1])
            reserved_cols.add(pair.b[1])
        else:
            reserved_cols.add(pair.a[1])
            reserved_rows.add(pair.a[0])
            reserved_rows.add(pair.b[0])

    if mode == "mixed" and {pair.orientation for pair in pairs} != {"h", "v"}:
        return None

    return grid, pairs


def _expected_output(input_grid: np.ndarray, pairs: list[PairSpec]) -> np.ndarray:
    out = input_grid.copy()
    for pair in pairs:
        if pair.orientation == "h":
            r = pair.a[0]
            out[r, pair.a[1] + 1 : pair.b[1]] = pair.color
        else:
            c = pair.a[1]
            out[pair.a[0] + 1 : pair.b[0], c] = pair.color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    placed = _place_rectangles(height, width)
    if placed is None:
        return None
    base_grid, rects = placed

    planned = _plan_pairs(base_grid, rects)
    if planned is None:
        return None
    inp, pairs = planned
    expected = _expected_output(inp, pairs)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
