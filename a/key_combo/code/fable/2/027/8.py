from __future__ import annotations

import random
from typing import Any

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
RECT_SIZE_CHOICES = tuple(range(7, 16, 2))
MARGIN_RANGE = (1, 6)
EXTRA_PAIR_PROB = 0.45
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_layout(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BACKGROUND if np.any(g == BACKGROUND) else _dominant_color(g)
    fg = np.argwhere(g != bg)
    assert fg.size > 0, "expected one foreground rectangle"

    r0, c0, r1, c1 = _bbox_coords(fg)
    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rect != bg), "foreground object must be a solid rectangle"

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 1 and w % 2 == 1 and h >= 3 and w >= 3

    colors = sorted(int(color) for color in np.unique(rect))
    assert len(colors) == 2, "expected exactly two rectangle colors"

    midr = (r0 + r1) // 2
    midc = (c0 + c1) // 2
    inner_color = int(g[midr, midc])
    assert inner_color in colors
    outer_color = next(color for color in colors if color != inner_color)

    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == inner_color), "inner core must be solid"
    assert g[r0, c0] == outer_color and g[r0, c1] == outer_color
    assert g[r1, c0] == outer_color and g[r1, c1] == outer_color

    left_rows = {r for r in range(r0 + 1, r1) if int(g[r, c0]) == inner_color}
    right_rows = {r for r in range(r0 + 1, r1) if int(g[r, c1]) == inner_color}
    top_cols = {c for c in range(c0 + 1, c1) if int(g[r0, c]) == inner_color}
    bottom_cols = {c for c in range(c0 + 1, c1) if int(g[r1, c]) == inner_color}

    paired_rows = left_rows & right_rows
    paired_cols = top_cols & bottom_cols

    return {
        "bg": bg,
        "bbox": (r0, c0, r1, c1),
        "midr": midr,
        "midc": midc,
        "outer_color": outer_color,
        "inner_color": inner_color,
        "pair_rows": paired_rows,
        "pair_cols": paired_cols,
        "left_single_rows": left_rows - paired_rows,
        "right_single_rows": right_rows - paired_rows,
        "top_single_cols": top_cols - paired_cols,
        "bottom_single_cols": bottom_cols - paired_cols,
    }


def _apply_layout(grid: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, c1 = layout["bbox"]
    midr = int(layout["midr"])
    midc = int(layout["midc"])
    outer = int(layout["outer_color"])
    inner = int(layout["inner_color"])

    for r in layout["pair_rows"]:
        out[r, c0 : c1 + 1] = outer
        out[r, :c0] = inner
        out[r, c1 + 1 :] = inner

    for c in layout["pair_cols"]:
        out[r0 : r1 + 1, c] = outer
        out[:r0, c] = inner
        out[r1 + 1 :, c] = inner

    for r in layout["left_single_rows"]:
        out[r, c0 : midc + 1] = outer
    for r in layout["right_single_rows"]:
        out[r, midc : c1 + 1] = outer
    for c in layout["top_single_cols"]:
        out[r0 : midr + 1, c] = outer
    for c in layout["bottom_single_cols"]:
        out[midr : r1 + 1, c] = outer

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let opposite leaf pairs cut fully through the bicolor rectangle while lone leaves only cut halfway to the center axis.

    Input structure:
    - A black background contains one solid odd-by-odd rectangle made of two colors:
      a one-cell outer border band around a solid inner core.
    - Some non-corner border cells are replaced by single inner-color leaf pixels on the four sides.
    - Leaves may appear as left/right row pairs, top/bottom column pairs, or as unpaired single leaves.

    Rule:
    - A left leaf paired with a right leaf in the same row cuts that entire rectangle row to the outer color,
      then sends inner-color rays outward on both sides until the grid edges.
    - A top leaf paired with a bottom leaf in the same column cuts that entire rectangle column to the outer color,
      then sends inner-color rays outward above and below the rectangle.
    - Any remaining unpaired leaf cuts only from its own side inward to the rectangle's center column or center row,
      inclusive, repainting that half-segment to the outer color and never extending outside.
    - Every other cell stays unchanged.

    Key insight:
    - A leaf does not automatically mean a full cut; only an opposite-side match grants a through-cut with outside rays,
      while a lone leaf can slice only halfway into the rectangle.
    """
    return _apply_layout(np.asarray(grid, dtype=int), _parse_layout(grid))


def _all_leaf_shortcut(grid: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, c1 = layout["bbox"]
    outer = int(layout["outer_color"])
    inner = int(layout["inner_color"])
    horizontal_rows = (
        set(layout["pair_rows"])
        | set(layout["left_single_rows"])
        | set(layout["right_single_rows"])
    )
    vertical_cols = (
        set(layout["pair_cols"])
        | set(layout["top_single_cols"])
        | set(layout["bottom_single_cols"])
    )

    for r in horizontal_rows:
        out[r, c0 : c1 + 1] = outer
        out[r, :c0] = inner
        out[r, c1 + 1 :] = inner

    for c in vertical_cols:
        out[r0 : r1 + 1, c] = outer
        out[:r0, c] = inner
        out[r1 + 1 :, c] = inner

    return out


def _inside_only_pair_shortcut(grid: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, c1 = layout["bbox"]
    midr = int(layout["midr"])
    midc = int(layout["midc"])
    outer = int(layout["outer_color"])

    for r in layout["pair_rows"]:
        out[r, c0 : c1 + 1] = outer
    for c in layout["pair_cols"]:
        out[r0 : r1 + 1, c] = outer

    for r in layout["left_single_rows"]:
        out[r, c0 : midc + 1] = outer
    for r in layout["right_single_rows"]:
        out[r, midc : c1 + 1] = outer
    for c in layout["top_single_cols"]:
        out[r0 : midr + 1, c] = outer
    for c in layout["bottom_single_cols"]:
        out[midr : r1 + 1, c] = outer

    return out


def _sample_distinct(values: list[int], count: int) -> list[int]:
    return random.sample(values, count)


def _sample_attempt() -> tuple[np.ndarray, dict[str, Any]]:
    h = random.choice(RECT_SIZE_CHOICES)
    w = random.choice(RECT_SIZE_CHOICES)
    midr_rel = h // 2
    midc_rel = w // 2

    inner_rows = list(range(1, h - 1))
    inner_cols = list(range(1, w - 1))

    pair_row_count = 1 + int(random.random() < EXTRA_PAIR_PROB)
    pair_col_count = 1 + int(random.random() < EXTRA_PAIR_PROB)

    pair_rows_rel = set(_sample_distinct(inner_rows, pair_row_count))
    pair_cols_rel = set(_sample_distinct(inner_cols, pair_col_count))

    single_row_pool = [r for r in inner_rows if r not in pair_rows_rel and r != midr_rel]
    single_col_pool = [c for c in inner_cols if c not in pair_cols_rel and c != midc_rel]
    assert len(single_row_pool) >= 2 and len(single_col_pool) >= 2

    left_single_rel, right_single_rel = _sample_distinct(single_row_pool, 2)
    top_single_rel, bottom_single_rel = _sample_distinct(single_col_pool, 2)

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    height = h + top_margin + bottom_margin
    width = w + left_margin + right_margin
    assert height <= 30 and width <= 30

    outer_color, inner_color = random.sample(COLOR_POOL, 2)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    grid[r0 : r1 + 1, c0 : c1 + 1] = outer_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner_color

    pair_rows_abs = {r0 + r for r in pair_rows_rel}
    pair_cols_abs = {c0 + c for c in pair_cols_rel}
    left_single_rows_abs = {r0 + left_single_rel}
    right_single_rows_abs = {r0 + right_single_rel}
    top_single_cols_abs = {c0 + top_single_rel}
    bottom_single_cols_abs = {c0 + bottom_single_rel}

    for r in pair_rows_abs:
        grid[r, c0] = inner_color
        grid[r, c1] = inner_color
    for c in pair_cols_abs:
        grid[r0, c] = inner_color
        grid[r1, c] = inner_color

    for r in left_single_rows_abs:
        grid[r, c0] = inner_color
    for r in right_single_rows_abs:
        grid[r, c1] = inner_color
    for c in top_single_cols_abs:
        grid[r0, c] = inner_color
    for c in bottom_single_cols_abs:
        grid[r1, c] = inner_color

    layout = {
        "bg": BACKGROUND,
        "bbox": (r0, c0, r1, c1),
        "midr": (r0 + r1) // 2,
        "midc": (c0 + c1) // 2,
        "outer_color": outer_color,
        "inner_color": inner_color,
        "pair_rows": pair_rows_abs,
        "pair_cols": pair_cols_abs,
        "left_single_rows": left_single_rows_abs,
        "right_single_rows": right_single_rows_abs,
        "top_single_cols": top_single_cols_abs,
        "bottom_single_cols": bottom_single_cols_abs,
    }
    return grid, layout


def _validate_attempt(grid: np.ndarray, layout: dict[str, Any]) -> bool:
    expected = _apply_layout(grid, layout)
    parsed = _parse_layout(grid)

    keys = (
        "bbox",
        "midr",
        "midc",
        "outer_color",
        "inner_color",
        "pair_rows",
        "pair_cols",
        "left_single_rows",
        "right_single_rows",
        "top_single_cols",
        "bottom_single_cols",
    )
    for key in keys:
        if parsed[key] != layout[key]:
            return False

    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False

    if not layout["pair_rows"] or not layout["pair_cols"]:
        return False
    if not layout["left_single_rows"] or not layout["right_single_rows"]:
        return False
    if not layout["top_single_cols"] or not layout["bottom_single_cols"]:
        return False

    midr = int(layout["midr"])
    midc = int(layout["midc"])
    if any(r == midr for r in layout["left_single_rows"] | layout["right_single_rows"]):
        return False
    if any(c == midc for c in layout["top_single_cols"] | layout["bottom_single_cols"]):
        return False

    if np.array_equal(expected, _all_leaf_shortcut(grid, layout)):
        return False
    if np.array_equal(expected, _inside_only_pair_shortcut(grid, layout)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, layout = _sample_attempt()
        if _validate_attempt(grid, layout):
            return grid
    raise RuntimeError("failed to generate a valid half-cut rectangle puzzle")
