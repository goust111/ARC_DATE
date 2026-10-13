from __future__ import annotations

from collections import Counter
import random

import numpy as np


_BG = 0
_ARROW_COLOR = 5
_DIRECTIONS = ("right", "down", "left", "up")
_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
_GRID_SIDE_RANGE = (18, 24)
_SQUARE_COUNT_RANGE = (3, 5)
_SQUARE_SIZES = (2, 3, 4)
_ARROW_SHAFT_RANGE = (4, 6)
_PLACEMENT_GAP = 1
_MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _describe_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict[str, int | list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    return {
        "color": int(color),
        "coords": coords,
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(len(coords)),
    }


def _extract_components(grid: np.ndarray) -> list[dict[str, int | list[tuple[int, int]]]]:
    bg = _dominant_color(grid)
    comps: list[dict[str, int | list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            comps.append(_describe_component(grid, color, coords))
    return comps


def _is_square_component(grid: np.ndarray, comp: dict[str, int | list[tuple[int, int]]]) -> bool:
    h = int(comp["h"])
    w = int(comp["w"])
    if h != w or h < 2:
        return False
    if int(comp["area"]) != h * w:
        return False
    r0, c0, r1, c1 = (int(comp[k]) for k in ("r0", "c0", "r1", "c1"))
    color = int(comp["color"])
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _arrow_direction(comp: dict[str, int | list[tuple[int, int]]]) -> str:
    coords = list(comp["coords"])
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if width > height:
        row_counts = Counter(r for r, _c in coords)
        max_count = max(row_counts.values())
        shaft_rows = [r for r, count in row_counts.items() if count == max_count]
        shaft_r = min(shaft_rows, key=lambda r: abs(r - (r0 + r1) / 2))
        shaft_cols = [c for r, c in coords if r == shaft_r]
        head_cols = [c for r, c in coords if r != shaft_r]
        assert head_cols, "horizontal arrow must contain off-axis head pixels"
        return "right" if np.mean(head_cols) > np.mean(shaft_cols) else "left"

    col_counts = Counter(c for _r, c in coords)
    max_count = max(col_counts.values())
    shaft_cols = [c for c, count in col_counts.items() if count == max_count]
    shaft_c = min(shaft_cols, key=lambda c: abs(c - (c0 + c1) / 2))
    shaft_rows = [r for r, c in coords if c == shaft_c]
    head_rows = [r for r, c in coords if c != shaft_c]
    assert head_rows, "vertical arrow must contain off-axis head pixels"
    return "down" if np.mean(head_rows) > np.mean(shaft_rows) else "up"


def _square_order_key(comp: dict[str, int | list[tuple[int, int]]], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = (int(comp[k]) for k in ("r0", "c0", "r1", "c1"))
    if direction == "right":
        return c0, r0
    if direction == "down":
        return r0, c0
    if direction == "left":
        return -c1, r0
    return -r1, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the scattered color squares into a 1D stripe ordered by the gray arrow's scan direction.

    Input structure:
    - The background is black and one connected non-background object is a gray arrow:
      a 1-pixel-wide shaft plus a small 3-pixel head pointing left, right, up, or down.
    - Several other connected non-background objects are solid colored square blocks.
    - The blocks do not overlap each other or the arrow.

    Rule:
    - Identify the arrow component and infer whether it points horizontally or vertically,
      then determine which end contains the head.
    - Identify every solid square block and sort them by the scan rule encoded by the arrow:
      left-edge increasing for right, top-edge increasing for down, right-edge decreasing
      for left, and bottom-edge decreasing for up.
    - Replace each square by a contiguous 1D run of its own color whose length equals the
      square's side length, then concatenate those runs without gaps.

    Key insight:
    - The arrow does not move anything directly; it only chooses the ordering axis and
      direction, while each 2D square is reduced to a 1D segment whose length is its side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    components = _extract_components(g)
    squares = [comp for comp in components if _is_square_component(g, comp)]
    arrows = [comp for comp in components if not _is_square_component(g, comp)]
    assert len(arrows) == 1, "expected exactly one arrow component"
    assert 3 <= len(squares) <= 5, "expected 3-5 square components"

    direction = _arrow_direction(arrows[0])
    squares.sort(key=lambda comp: _square_order_key(comp, direction))

    total = sum(int(comp["h"]) for comp in squares)
    if direction in {"left", "right"}:
        out = np.zeros((1, total), dtype=int)
        pos = 0
        for comp in squares:
            side = int(comp["h"])
            out[0, pos : pos + side] = int(comp["color"])
            pos += side
        return out

    out = np.zeros((total, 1), dtype=int)
    pos = 0
    for comp in squares:
        side = int(comp["h"])
        out[pos : pos + side, 0] = int(comp["color"])
        pos += side
    return out


def _arrow_cells(direction: str, shaft_len: int) -> tuple[list[tuple[int, int]], int, int]:
    if direction == "right":
        cells = [(1, c) for c in range(shaft_len)]
        cells += [(0, shaft_len - 1), (2, shaft_len - 1), (1, shaft_len)]
        return cells, 3, shaft_len + 1
    if direction == "left":
        cells = [(1, c + 1) for c in range(shaft_len)]
        cells += [(0, 1), (2, 1), (1, 0)]
        return cells, 3, shaft_len + 1
    if direction == "down":
        cells = [(r, 1) for r in range(shaft_len)]
        cells += [(shaft_len - 1, 0), (shaft_len - 1, 2), (shaft_len, 1)]
        return cells, shaft_len + 1, 3
    cells = [(r + 1, 1) for r in range(shaft_len)]
    cells += [(1, 0), (1, 2), (0, 1)]
    return cells, shaft_len + 1, 3


def _rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _separated(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = _PLACEMENT_GAP) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _ordering_value(r0: int, c0: int, side: int, direction: str) -> int:
    if direction == "right":
        return c0
    if direction == "down":
        return r0
    if direction == "left":
        return c0 + side - 1
    return r0 + side - 1


def _render_arrow(grid: np.ndarray, direction: str, shaft_len: int, r0: int, c0: int) -> None:
    cells, _h, _w = _arrow_cells(direction, shaft_len)
    for dr, dc in cells:
        grid[r0 + dr, c0 + dc] = _ARROW_COLOR


def _expected_output(squares: list[dict[str, int]], direction: str) -> np.ndarray:
    ordered = sorted(
        squares,
        key=lambda sq: _square_order_key(
            {
                "r0": sq["r0"],
                "c0": sq["c0"],
                "r1": sq["r1"],
                "c1": sq["c1"],
            },
            direction,
        ),
    )
    total = sum(int(sq["side"]) for sq in ordered)
    if direction in {"left", "right"}:
        out = np.zeros((1, total), dtype=int)
        pos = 0
        for sq in ordered:
            side = int(sq["side"])
            out[0, pos : pos + side] = int(sq["color"])
            pos += side
        return out
    out = np.zeros((total, 1), dtype=int)
    pos = 0
    for sq in ordered:
        side = int(sq["side"])
        out[pos : pos + side, 0] = int(sq["color"])
        pos += side
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIDE_RANGE)
    w = random.randint(*_GRID_SIDE_RANGE)
    direction = random.choice(_DIRECTIONS)
    shaft_len = random.randint(*_ARROW_SHAFT_RANGE)
    arrow_cells, arrow_h, arrow_w = _arrow_cells(direction, shaft_len)

    if h < arrow_h or w < arrow_w:
        return None

    arrow_r0 = random.randint(0, h - arrow_h)
    arrow_c0 = random.randint(0, w - arrow_w)
    arrow_rect = _rect(arrow_r0, arrow_c0, arrow_h, arrow_w)

    square_count = random.randint(*_SQUARE_COUNT_RANGE)
    colors = random.sample(_COLOR_POOL, square_count)
    sizes = [random.choice(_SQUARE_SIZES) for _ in range(square_count)]

    rects = [arrow_rect]
    used_order_values: set[int] = set()
    placed_squares: list[dict[str, int]] = []

    for color, side in zip(colors, sizes):
        for _ in range(250):
            r0 = random.randint(0, h - side)
            c0 = random.randint(0, w - side)
            rect = _rect(r0, c0, side, side)
            if not _separated(rect, rects):
                continue
            order_value = _ordering_value(r0, c0, side, direction)
            if order_value in used_order_values:
                continue
            rects.append(rect)
            used_order_values.add(order_value)
            placed_squares.append(
                {
                    "color": int(color),
                    "side": int(side),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r0 + side - 1),
                    "c1": int(c0 + side - 1),
                }
            )
            break
        else:
            return None

    grid = np.zeros((h, w), dtype=int)
    _render_arrow(grid, direction, shaft_len, arrow_r0, arrow_c0)
    for sq in placed_squares:
        grid[int(sq["r0"]) : int(sq["r1"]) + 1, int(sq["c0"]) : int(sq["c1"]) + 1] = int(sq["color"])

    expected = _expected_output(placed_squares, direction)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_GENERATION_ATTEMPTS} attempts")
