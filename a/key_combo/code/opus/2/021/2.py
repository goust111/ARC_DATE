from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 3
MARK = 4
LEDGE = 5

BAR_COUNT_RANGE = (10, 16)
HEIGHT_RANGE = (15, 20)
WIDTH_RANGE = (17, 22)
EXTRA_LEDGE_PROB = 0.55
GENERATION_ATTEMPTS = 400
SCENE_ATTEMPTS = 600
DISTRACTOR_ATTEMPTS = 12000

Bar = tuple[str, int, int]


def _cells(bar: Bar) -> list[tuple[int, int]]:
    orient, row, col = bar
    if orient == "H":
        return [(row, col), (row, col + 1)]
    return [(row, col), (row + 1, col)]


def _normalize_bar(coords: list[tuple[int, int]]) -> Bar | None:
    coords = sorted(coords)
    if len(coords) != 2:
        return None
    if coords[0][0] == coords[1][0] and coords[0][1] + 1 == coords[1][1]:
        return ("H", coords[0][0], coords[0][1])
    if coords[0][1] == coords[1][1] and coords[0][0] + 1 == coords[1][0]:
        return ("V", coords[0][0], coords[0][1])
    return None


def _find_bars(grid: np.ndarray) -> list[Bar]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    bars: list[Bar] = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) != BAR:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == BAR
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bar = _normalize_bar(component)
            assert bar is not None, "every green component must be a two-pixel bar"
            bars.append(bar)
    return bars


def _place_bar(grid: np.ndarray, bar: Bar, color: int = BAR) -> None:
    for row, col in _cells(bar):
        grid[row, col] = int(color)


def _can_place_bar(grid: np.ndarray, bar: Bar) -> bool:
    height, width = map(int, grid.shape)
    own = set(_cells(bar))
    for row, col in own:
        if not (0 <= row < height and 0 <= col < width):
            return False
        if int(grid[row, col]) != BG:
            return False

    for row, col in own:
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in own and int(grid[nr, nc]) == BAR:
                return False
    return True


def _can_drop(bar: Bar, occupied: np.ndarray) -> bool:
    height, _width = map(int, occupied.shape)
    own = set(_cells(bar))
    for row, col in own:
        nr = row + 1
        if nr >= height:
            return False
        if (nr, col) not in own and bool(occupied[nr, col]):
            return False
    return True


def _drop_bar(bar: Bar, occupied: np.ndarray) -> Bar:
    current = bar
    while _can_drop(current, occupied):
        orient, row, col = current
        current = (orient, row + 1, col)
    return current


def _settle_bars(grid: np.ndarray) -> tuple[np.ndarray, list[Bar]]:
    bars = _find_bars(grid)
    settled_grid = np.array(grid, dtype=int, copy=True)
    settled_grid[settled_grid == BAR] = BG
    occupied = settled_grid != BG

    order = sorted(
        bars,
        key=lambda bar: (
            -max(row for row, _col in _cells(bar)),
            -min(row for row, _col in _cells(bar)),
            min(col for _row, col in _cells(bar)),
            0 if bar[0] == "H" else 1,
        ),
    )

    settled_bars: list[Bar] = []
    for bar in order:
        landed = _drop_bar(bar, occupied)
        settled_bars.append(landed)
        _place_bar(settled_grid, landed, BAR)
        for row, col in _cells(landed):
            occupied[row, col] = True
    return settled_grid, settled_bars


def _bar_anchor_sets(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    horizontals: set[tuple[int, int]] = set()
    verticals: set[tuple[int, int]] = set()
    for orient, row, col in _find_bars(grid):
        if orient == "H":
            horizontals.add((row, col))
        else:
            verticals.add((row, col))
    return horizontals, verticals


def _find_implied_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    horizontals, verticals = _bar_anchor_sets(grid)
    centers: list[tuple[int, int]] = []
    patterns = (
        {"top_v": -3, "bottom_v": 2, "left_h": -3, "right_h": 2},
        {"top_v": -4, "bottom_v": 3, "left_h": -4, "right_h": 3},
    )

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if int(grid[row, col]) != BG:
                continue
            if any(int(grid[nr, nc]) != BG for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))):
                continue
            for pattern in patterns:
                if (
                    (row + pattern["top_v"], col) in verticals
                    and (row + pattern["bottom_v"], col) in verticals
                    and (row, col + pattern["left_h"]) in horizontals
                    and (row, col + pattern["right_h"]) in horizontals
                ):
                    centers.append((row, col))
                    break
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each green two-cell bar onto the gray ledges, then mark every empty plus center that appears after settling.

    Input structure:
    - Black background with isolated green two-pixel bars, each either horizontal or vertical.
    - Gray horizontal ledges are fixed obstacles, and the bottom edge acts as an additional floor.
    - Yellow is reserved for the output and does not appear in the input.

    Rule:
    - Remove the green bars, then settle them one by one from lower starting rows to higher starting rows, treating every bar as a rigid body that only moves straight downward until its next step would hit a ledge, a previously settled bar, or the floor.
    - After settling, scan each background cell whose direct up, down, left, and right neighbors stay background; mark it when four aligned two-cell bars surround it in the implied-plus layout, using either the standard 2-to-3 offset arrangement or the gravity-compatible 3-to-4 offset arrangement.
    - Recolor every detected center yellow and leave the settled green bars and gray ledges unchanged.

    Key insight:
    - The yellow target is not visible in the input scene; it becomes identifiable only after the gravity simulation brings four separate bars into a shared cross-like alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == MARK), "yellow is reserved for output markers"

    settled, _settled_bars = _settle_bars(g)
    out = settled.copy()
    for row, col in _find_implied_centers(settled):
        out[row, col] = MARK
    return out


def _add_ledge(grid: np.ndarray, ledges: list[tuple[int, int, int]], row: int, start: int, length: int) -> bool:
    if np.any(grid[row, start : start + length] != BG):
        return False
    grid[row, start : start + length] = LEDGE
    ledges.append((int(row), int(start), int(length)))
    return True


def _bar_is_supported_by_static(grid: np.ndarray, bar: Bar) -> bool:
    height, _width = map(int, grid.shape)
    orient, row, col = bar
    if orient == "H":
        if row == height - 1:
            return True
        return bool(grid[row + 1, col] == LEDGE or grid[row + 1, col + 1] == LEDGE)
    if row + 1 == height - 1:
        return True
    return bool(grid[row + 2, col] == LEDGE)


def _target_final_bars(center: tuple[int, int]) -> list[Bar]:
    row, col = center
    return [
        ("V", row - 4, col),
        ("V", row + 3, col),
        ("H", row, col - 4),
        ("H", row, col + 3),
    ]


def _generate_final_scene() -> tuple[np.ndarray, list[Bar], tuple[int, int]]:
    for _ in range(SCENE_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        center = (random.randint(5, height - 6), random.randint(7, width - 8))
        row, col = center

        grid = np.zeros((height, width), dtype=int)
        ledges: list[tuple[int, int, int]] = []

        top_len = random.randint(3, 5)
        bottom_len = random.randint(3, 5)
        left_len = random.randint(3, 5)
        right_len = random.randint(3, 5)

        top_start = random.randint(max(0, col - top_len + 1), min(col, width - top_len))
        bottom_start = random.randint(max(0, col - bottom_len + 1), min(col, width - bottom_len))
        left_start = col - 3 - left_len + 1
        right_start = col + 3
        if left_start < 0 or right_start + right_len > width:
            continue

        if not _add_ledge(grid, ledges, row - 2, top_start, top_len):
            continue
        if not _add_ledge(grid, ledges, row + 5, bottom_start, bottom_len):
            continue
        if not _add_ledge(grid, ledges, row + 1, left_start, left_len):
            continue
        if not _add_ledge(grid, ledges, row + 1, right_start, right_len):
            continue

        if random.random() < EXTRA_LEDGE_PROB:
            for _extra in range(80):
                ledge_row = random.randint(3, height - 3)
                if ledge_row in {row - 2, row + 1, row + 5}:
                    continue
                ledge_len = random.randint(3, 8)
                ledge_start = random.randint(0, width - ledge_len)
                if row - 5 <= ledge_row <= row + 5 and not (ledge_start + ledge_len - 1 < col - 6 or ledge_start > col + 6):
                    continue
                if _add_ledge(grid, ledges, ledge_row, ledge_start, ledge_len):
                    break

        bars = _target_final_bars(center)
        ok = True
        for bar in bars:
            if not _can_place_bar(grid, bar):
                ok = False
                break
            _place_bar(grid, bar, BAR)
        if not ok:
            continue
        if _find_implied_centers(grid) != [center]:
            continue

        target_count = random.randint(*BAR_COUNT_RANGE)
        attempts = 0
        while len(bars) < target_count and attempts < DISTRACTOR_ATTEMPTS:
            attempts += 1
            orient = random.choice(("H", "V"))
            use_ledge = bool(ledges) and random.random() < 0.45
            if use_ledge:
                ledge_row, ledge_start, ledge_len = random.choice(ledges)
                if orient == "H":
                    bar_row = ledge_row - 1
                    if bar_row < 0:
                        continue
                    candidates = [
                        c
                        for c in range(max(0, ledge_start - 1), min(width - 1, ledge_start + ledge_len))
                        if ledge_start <= c <= ledge_start + ledge_len - 1 or ledge_start <= c + 1 <= ledge_start + ledge_len - 1
                    ]
                    if not candidates:
                        continue
                    bar = ("H", bar_row, random.choice(candidates))
                else:
                    bar_row = ledge_row - 2
                    if bar_row < 0:
                        continue
                    bar = ("V", bar_row, random.randint(ledge_start, ledge_start + ledge_len - 1))
            else:
                if orient == "H":
                    bar = ("H", height - 1, random.randint(0, width - 2))
                else:
                    bar = ("V", height - 2, random.randint(0, width - 1))

            if any(row - 6 <= r <= row + 6 and col - 6 <= c <= col + 6 for r, c in _cells(bar)):
                continue
            if not _can_place_bar(grid, bar):
                continue
            if not _bar_is_supported_by_static(grid, bar):
                continue

            _place_bar(grid, bar, BAR)
            if _find_implied_centers(grid) != [center]:
                _place_bar(grid, bar, BG)
                continue
            bars.append(bar)

        if len(bars) == target_count:
            return grid, bars, center
    raise RuntimeError("failed to build a valid final settled scene")


def _lift_scene(final_grid: np.ndarray, final_bars: list[Bar]) -> np.ndarray | None:
    height, _width = map(int, final_grid.shape)
    base = final_grid.copy()
    base[base == BAR] = BG
    occupied = base != BG

    candidates: dict[Bar, list[Bar]] = {}
    for bar in final_bars:
        orient, final_row, final_col = bar
        max_start_row = min(height - 4, final_row - 1)
        options: list[Bar] = []
        for row in range(max_start_row + 1):
            candidate = (orient, row, final_col)
            if any(int(base[r, c]) != BG for r, c in _cells(candidate)):
                continue
            if _drop_bar(candidate, occupied) != bar:
                continue
            options.append(candidate)
        random.shuffle(options)
        candidates[bar] = options

    order = sorted(final_bars, key=lambda bar: len(candidates[bar]))
    if any(len(candidates[bar]) == 0 for bar in order):
        return None

    input_grid = base.copy()

    def backtrack(index: int) -> bool:
        if index == len(order):
            return not _find_implied_centers(input_grid)

        for candidate in candidates[order[index]]:
            if not _can_place_bar(input_grid, candidate):
                continue
            _place_bar(input_grid, candidate, BAR)
            if not _find_implied_centers(input_grid) and backtrack(index + 1):
                return True
            _place_bar(input_grid, candidate, BG)
        return False

    if backtrack(0):
        return input_grid
    return None


def _count_ledges(grid: np.ndarray) -> int:
    count = 0
    for row in range(grid.shape[0]):
        col = 0
        while col < grid.shape[1]:
            if int(grid[row, col]) != LEDGE:
                col += 1
                continue
            while col < grid.shape[1] and int(grid[row, col]) == LEDGE:
                col += 1
            count += 1
    return count


def _validate_generated(input_grid: np.ndarray, final_grid: np.ndarray, center: tuple[int, int]) -> bool:
    bars = _find_bars(input_grid)
    if not (BAR_COUNT_RANGE[0] <= len(bars) <= BAR_COUNT_RANGE[1]):
        return False
    ledge_count = _count_ledges(input_grid)
    if not (3 <= ledge_count <= 5):
        return False
    if _find_implied_centers(input_grid):
        return False

    settled, _settled_bars = _settle_bars(input_grid)
    if not np.array_equal(settled, final_grid):
        return False
    if _find_implied_centers(final_grid) != [center]:
        return False

    moved = 0
    input_occ = final_grid.copy()
    input_occ[input_occ == BAR] = BG
    base_occ = input_occ != BG
    for bar in bars:
        if _drop_bar(bar, base_occ) != bar:
            moved += 1
    return moved >= 4


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        final_grid, final_bars, center = _generate_final_scene()
        input_grid = _lift_scene(final_grid, final_bars)
        if input_grid is None:
            continue
        if _validate_generated(input_grid, final_grid, center):
            return input_grid
    raise RuntimeError("failed to generate a valid gravity-implied-plus input")
