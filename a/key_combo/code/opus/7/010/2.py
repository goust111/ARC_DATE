from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_010_2",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color


BACKGROUND = Color.GREY
SHIFT_BY_COLOR = {
    Color.RED: (0, -2),
    Color.BLUE: (0, 1),
    Color.ORANGE: (-2, 0),
    Color.MAROON: (2, 0),
}
SNAKE_ORDER = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 1),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
)
ROW_MAJOR_ORDER = tuple((r, c) for r in range(3) for c in range(3))
GENERATOR_COLORS = (Color.BLUE, Color.RED)
MAX_GENERATION_ATTEMPTS = 600
_SEEN_OUTPUTS: set[tuple[int, ...]] = set()


def _foreground_items(grid: np.ndarray) -> list[dict[str, int]]:
    items: list[dict[str, int]] = []
    for r, c in np.argwhere(grid != BACKGROUND):
        color = int(grid[r, c])
        if color not in SHIFT_BY_COLOR:
            raise AssertionError(f"unexpected foreground color: {color}")
        dr, dc = SHIFT_BY_COLOR[color]
        items.append(
            {
                "r": int(r),
                "c": int(c),
                "tr": int(r + dr),
                "tc": int(c + dc),
                "color": color,
            }
        )
    return items


def _paint_sequence(colors: list[int], order: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.full((3, 3), BACKGROUND, dtype=int)
    for index, color in enumerate(colors[:9]):
        r, c = order[index]
        out[r, c] = int(color)
    return out


def _solve_from_items(items: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (item["tc"], item["tr"]))
    return _paint_sequence([item["color"] for item in ordered], SNAKE_ORDER)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each colored singleton by its color-coded move, then snake-compress the colors by shifted column order.

    Input structure:
    - The grid is mostly grey background cells.
    - Up to nine non-grey singleton pixels appear in distinct input columns.
    - Foreground colors may be blue, red, orange, or maroon, and each color encodes one fixed translation.

    Rule:
    - Translate every foreground pixel while keeping its original color: red moves 2 left, blue moves 1 right,
      orange moves 2 up, and maroon moves 2 down.
    - Sort all translated pixels by translated column from left to right, breaking ties by translated row.
    - Write the resulting color sequence into a 3x3 grey output using serpentine order:
      row 0 left-to-right, row 1 right-to-left, row 2 left-to-right.
    - Leave any unused output cells grey.

    Key insight:
    - The colors serve double duty: they identify the output pixels and also alter the ordering key by moving each
      pixel before the column-based compression step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    items = _foreground_items(g)
    assert len(items) <= 9
    return _solve_from_items(items)


def _original_column_shortcut(items: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (item["c"], item["r"]))
    return _paint_sequence([item["color"] for item in ordered], SNAKE_ORDER)


def _row_major_shortcut(items: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (item["tc"], item["tr"]))
    return _paint_sequence([item["color"] for item in ordered], ROW_MAJOR_ORDER)


def _instance_is_semantic(grid: np.ndarray) -> bool:
    items = _foreground_items(grid)
    if not (5 <= len(items) <= 9):
        return False
    if {item["color"] for item in items} != set(GENERATOR_COLORS):
        return False
    if len({item["c"] for item in items}) != len(items):
        return False
    if len({item["tc"] for item in items}) != len(items):
        return False
    if len({item["r"] for item in items}) < 2:
        return False

    height, width = grid.shape
    for item in items:
        if not (0 <= item["tr"] < height and 0 <= item["tc"] < width):
            return False

    expected = _solve_from_items(items)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _original_column_shortcut(items)):
        return False
    if np.array_equal(expected, _row_major_shortcut(items)):
        return False
    return True


def _make_candidate() -> np.ndarray | None:
    count = random.randint(5, 9)
    width = random.randint(max(8, count + 3), 16)
    height = random.randint(8, 16)

    translated_cols = random.sample(range(1, width - 2), count)
    colors = [random.choice(GENERATOR_COLORS) for _ in range(count)]
    if set(colors) != set(GENERATOR_COLORS):
        return None

    rows = [random.randrange(height) for _ in range(count)]
    if len(set(rows)) < 2:
        return None

    items: list[dict[str, int]] = []
    original_cols: set[int] = set()
    for r, tc, color in zip(rows, translated_cols, colors):
        if color == Color.BLUE:
            c = tc - 1
        else:
            c = tc + 2
        if c in original_cols or not (0 <= c < width):
            return None
        original_cols.add(c)
        items.append({"r": int(r), "c": int(c), "tr": int(r), "tc": int(tc), "color": int(color)})

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for item in items:
        grid[item["r"], item["c"]] = item["color"]

    if not _instance_is_semantic(grid):
        return None

    out_sig = tuple(int(x) for x in _solve_from_items(items).ravel())
    if out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a varied valid ARC input for the color-shift snake-compression rule.

    The generator intentionally uses blue and red markers only. Under the puzzle's
    required "no translated-column conflicts" constraint, vertical movers would not
    change the sort key, so excluding them keeps every visible clue semantically active.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate instance")
