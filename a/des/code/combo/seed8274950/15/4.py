from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _line_info(coords: Iterable[tuple[int, int]]) -> dict[str, int | str]:
    pts = [(int(r), int(c)) for r, c in coords]
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})

    if len(rows) == 1:
        row = rows[0]
        start = cols[0]
        end = cols[-1]
        assert cols == list(range(start, end + 1)), "horizontal line must be contiguous"
        return {
            "orientation": "h",
            "fixed": row,
            "start": start,
            "end": end,
            "length": end - start + 1,
        }

    if len(cols) == 1:
        col = cols[0]
        start = rows[0]
        end = rows[-1]
        assert rows == list(range(start, end + 1)), "vertical line must be contiguous"
        return {
            "orientation": "v",
            "fixed": col,
            "start": start,
            "end": end,
            "length": end - start + 1,
        }

    raise AssertionError("component must be a 1-cell-thick straight line")


def _paint_horizontal(out: np.ndarray, row: int, start: int, end: int, gap_size: int, color: int) -> None:
    length = end - start + 1
    if gap_size <= 0:
        out[row, start : end + 1] = color
        return
    if gap_size >= length:
        return

    gap_start = start + (length - gap_size) // 2
    gap_end = gap_start + gap_size - 1
    out[row, start:gap_start] = color
    out[row, gap_end + 1 : end + 1] = color


def _paint_vertical(out: np.ndarray, col: int, start: int, end: int, gap_size: int, color: int) -> None:
    length = end - start + 1
    if gap_size <= 0:
        out[start : end + 1, col] = color
        return
    if gap_size >= length:
        return

    gap_start = start + (length - gap_size) // 2
    gap_end = gap_start + gap_size - 1
    out[start:gap_start, col] = color
    out[gap_end + 1 : end + 1, col] = color


def _apply_pair(out: np.ndarray, line_a: dict[str, int | str], line_b: dict[str, int | str], color: int) -> None:
    assert line_a["orientation"] == line_b["orientation"], "same-color lines must be parallel"
    orientation = str(line_a["orientation"])

    if orientation == "h":
        first, second = sorted((line_a, line_b), key=lambda line: int(line["fixed"]))
        distance = int(second["fixed"]) - int(first["fixed"])
        assert distance > 0, "paired lines must be distinct"
        for step in range(1, distance):
            row = int(first["fixed"]) + step
            _paint_horizontal(
                out,
                row,
                int(first["start"]),
                int(first["end"]),
                step - 1,
                color,
            )
        return

    first, second = sorted((line_a, line_b), key=lambda line: int(line["fixed"]))
    distance = int(second["fixed"]) - int(first["fixed"])
    assert distance > 0, "paired lines must be distinct"
    for step in range(1, distance):
        col = int(first["fixed"]) + step
        _paint_vertical(
            out,
            col,
            int(first["start"]),
            int(first["end"]),
            step - 1,
            color,
        )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the strip between each same-color pair of parallel 1-cell-thick lines.

    For horizontal pairs, the upper line is treated as the first line.
    For vertical pairs, the left line is treated as the first line.
    Interior rows/columns start fully filled, then lose one centered pixel
    per layer so the empty gap expands outward from the middle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        assert len(comps) == 2, "each foreground color must form exactly two lines"
        lines = [_line_info(comp) for comp in comps]
        _apply_pair(out, lines[0], lines[1], color)

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 26
PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 4
LINE_LEN_MIN = 4
LINE_LEN_CAP = 11
DISTANCE_MIN = 2
DISTANCE_CAP = 8
PLACE_TRIES = 200
GEN_TRIES = 300


def _balanced_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("h", "v"))]

    orientations = ["h"] * (count // 2) + ["v"] * (count // 2)
    if count % 2:
        orientations.append(random.choice(("h", "v")))
    random.shuffle(orientations)
    return orientations


def _reserve_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return (
        0 <= r0 <= r1 < mask.shape[0]
        and 0 <= c0 <= c1 < mask.shape[1]
        and not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())
    )


def _place_horizontal(grid: np.ndarray, reserved: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    max_len = min(LINE_LEN_CAP, w - 4)
    if max_len < LINE_LEN_MIN:
        return False

    for _ in range(PLACE_TRIES):
        length = random.randint(LINE_LEN_MIN, max_len)
        max_dist = min(DISTANCE_CAP, length + 1, h - 3)
        if max_dist < DISTANCE_MIN:
            continue
        distance = random.randint(DISTANCE_MIN, max_dist)

        c0 = random.randint(1, w - length - 2)
        r0 = random.randint(1, h - distance - 2)
        c1 = c0 + length - 1
        r1 = r0 + distance

        if not _reserve_clear(reserved, r0 - 1, c0 - 1, r1 + 1, c1 + 1):
            continue

        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return True

    return False


def _place_vertical(grid: np.ndarray, reserved: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    max_len = min(LINE_LEN_CAP, h - 4)
    if max_len < LINE_LEN_MIN:
        return False

    for _ in range(PLACE_TRIES):
        length = random.randint(LINE_LEN_MIN, max_len)
        max_dist = min(DISTANCE_CAP, length + 1, w - 3)
        if max_dist < DISTANCE_MIN:
            continue
        distance = random.randint(DISTANCE_MIN, max_dist)

        r0 = random.randint(1, h - length - 2)
        c0 = random.randint(1, w - distance - 2)
        r1 = r0 + length - 1
        c1 = c0 + distance

        if not _reserve_clear(reserved, r0 - 1, c0 - 1, r1 + 1, c1 + 1):
            continue

        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randint(0, 9)
    grid = np.full((h, w), bg, dtype=int)

    max_pairs = PAIR_COUNT_MAX if h * w >= 220 else 3
    pair_count = random.randint(PAIR_COUNT_MIN, max_pairs)
    colors = random.sample([c for c in range(10) if c != bg], pair_count)
    orientations = _balanced_orientations(pair_count)
    reserved = np.zeros((h, w), dtype=bool)

    for color, orientation in zip(colors, orientations):
        placed = _place_horizontal(grid, reserved, color) if orientation == "h" else _place_vertical(grid, reserved, color)
        if not placed:
            return None

    bg_count = int(np.sum(grid == bg))
    fg_counts = [int(np.sum(grid == color)) for color in colors]
    if not fg_counts or bg_count <= max(fg_counts):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_TRIES} attempts")
