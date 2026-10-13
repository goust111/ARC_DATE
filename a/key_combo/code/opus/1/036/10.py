from __future__ import annotations

import itertools
import math

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_CHOICES = (2, 3, 4, 5)
TOP_HEIGHT_RANGE = (8, 11)
BOTTOM_HEIGHT_RANGE = (8, 11)
WIDTH_RANGE = (18, 21)
GENERATION_RETRIES = 500
_RNG = np.random.default_rng()
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _foreground_colors(region: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(region) if int(color) not in (BG, SEP)]


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    row = rows[0]
    assert 0 < row < grid.shape[0] - 1, "separator must split the grid into two non-empty regions"
    return row


def _infer_side_by_color(upper: np.ndarray) -> dict[int, int]:
    side_by_color: dict[int, int] = {}
    for color in _foreground_colors(upper):
        count = int(np.count_nonzero(upper == color))
        side = math.isqrt(count)
        assert side * side == count, "upper-region color counts must be perfect squares"
        side_by_color[color] = side
    assert side_by_color, "expected foreground colors in the upper region"
    sides = list(side_by_color.values())
    assert len(set(sides)) == len(sides), "upper-region square side lengths must be unique"
    return side_by_color


def _infer_lines(lower: np.ndarray, colors: list[int]) -> dict[int, dict[str, object]]:
    h, w = lower.shape
    lines: dict[int, dict[str, object]] = {}
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()

    for color in colors:
        coords = np.argwhere(lower == color)
        assert coords.shape[0] >= 2, "each visible line must expose at least two pixels"
        same_row = bool(np.all(coords[:, 0] == coords[0, 0]))
        same_col = bool(np.all(coords[:, 1] == coords[0, 1]))
        assert same_row ^ same_col, "each visible color must belong to exactly one horizontal or vertical line"

        if same_row:
            row = int(coords[0, 0])
            assert row not in seen_rows, "horizontal lines must use distinct rows"
            seen_rows.add(row)
            cells = frozenset((row, c) for c in range(w))
            kind = "h"
            index = row
        else:
            col = int(coords[0, 1])
            assert col not in seen_cols, "vertical lines must use distinct columns"
            seen_cols.add(col)
            cells = frozenset((r, col) for r in range(h))
            kind = "v"
            index = col

        assert set(map(tuple, coords.tolist())).issubset(cells), "visible pixels must be part of the recovered line"
        lines[color] = {"kind": kind, "index": index, "cells": cells}

    return lines


def _infer_layer_order(lower: np.ndarray, lines: dict[int, dict[str, object]]) -> list[int]:
    colors = list(lines)
    below_edges: set[tuple[int, int]] = set()
    intersects: dict[int, set[int]] = {color: set() for color in colors}

    for r in range(lower.shape[0]):
        for c in range(lower.shape[1]):
            covering = [color for color, spec in lines.items() if (r, c) in spec["cells"]]
            visible = int(lower[r, c])
            assert len(covering) <= 2, "horizontal/vertical inputs should have at most one crossing per cell"

            if not covering:
                assert visible == BG, "background cells may not show a line color"
                continue

            assert visible in covering, "a visible crossing color must belong to one recovered line"
            if len(covering) == 1:
                assert visible == covering[0], "single-line cells must show that line's color"
                continue

            a, b = covering
            intersects[a].add(b)
            intersects[b].add(a)
            for color in covering:
                if color != visible:
                    below_edges.add((color, visible))

    def valid(order: tuple[int, ...]) -> bool:
        positions = {color: idx for idx, color in enumerate(order)}
        if any(positions[below] >= positions[above] for below, above in below_edges):
            return False
        return all(order[i + 1] in intersects[order[i]] for i in range(len(order) - 1))

    valid_orders = [order for order in itertools.permutations(colors) if valid(order)]
    assert len(valid_orders) == 1, "crossings must determine a unique bottom-to-top order"
    return list(valid_orders[0])


def _render_output(side_by_color: dict[int, int], order: list[int]) -> np.ndarray:
    height = max(side_by_color.values())
    width = sum(side_by_color[color] for color in order) + len(order) - 1
    out = np.full((height, width), BG, dtype=int)

    col = 0
    for idx, color in enumerate(order):
        side = side_by_color[color]
        out[height - side : height, col : col + side] = color
        col += side
        if idx != len(order) - 1:
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pack upper-region color counts into squares and place them left-to-right by the lower-region occlusion order.

    Input structure:
    - A black background grid is split by one full-width grey separator row.
    - The upper region contains 3 or 4 non-black, non-grey colors scattered in arbitrary blobs.
      Each color's total pixel count is a distinct perfect square.
    - The lower region uses exactly the same color set, with one full-span one-pixel-thick
      horizontal or vertical line per color; lines cross and visible crossing colors encode layering.

    Rule:
    - In the upper region, count the pixels of each foreground color and convert each count P into a
      filled square of side length sqrt(P).
    - In the lower region, recover each full line from its visible collinear pixels and infer the unique
      bottom-to-top layer order from which color is visible at every crossing.
    - Output only the packed squares: place them from left to right in that bottom-to-top order,
      bottom-aligned, with one black column between neighboring squares.

    Key insight:
    - The upper region supplies sizes while the lower region supplies order; neither region alone is enough.
      The shared colors are the only link between those two independent pieces of information.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row = _find_separator_row(g)
    upper = g[:sep_row]
    lower = g[sep_row + 1 :]

    side_by_color = _infer_side_by_color(upper)
    lower_colors = _foreground_colors(lower)
    assert set(lower_colors) == set(side_by_color), "upper and lower regions must use the same colors"

    lines = _infer_lines(lower, lower_colors)
    order = _infer_layer_order(lower, lines)
    return _render_output(side_by_color, order)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _centroid_order(upper: np.ndarray, colors: list[int]) -> list[int]:
    def key(color: int) -> tuple[float, float]:
        coords = np.argwhere(upper == color)
        return (float(coords[:, 1].mean()), float(coords[:, 0].mean()))

    return sorted(colors, key=key)


def _random_parts(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    parts = max(1, min(parts, total))
    min_part = 2 if total >= 2 * parts else 1
    values = [min_part] * parts
    remaining = total - min_part * parts
    for _ in range(remaining):
        values[int(rng.integers(0, parts))] += 1
    rng.shuffle(values)
    return values


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _random_blob(area: int, rng: np.random.Generator) -> np.ndarray:
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    valid_boxes = [
        (h, w)
        for h in range(1, min(area, 5) + 1)
        for w in range(max(1, math.ceil(area / h)), min(area, 6) + 1)
        if h * w >= area
    ]

    for _ in range(60):
        box_h, box_w = valid_boxes[int(rng.integers(0, len(valid_boxes)))]
        cells = {(int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in _DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(frontier_list[int(rng.integers(0, len(frontier_list)))])

        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if area >= 4 and mask.shape[0] == mask.shape[1] and int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        return mask

    mask = np.zeros((1, area), dtype=bool)
    mask[0, :] = True
    return mask


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        rr, cc = top + int(r), left + int(c)
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    candidates = [
        (top, left)
        for top in range(h - mh + 1)
        for left in range(w - mw + 1)
        if _can_place_mask(occupied, mask, top, left)
    ]
    if not candidates:
        return False

    top, left = candidates[int(rng.integers(0, len(candidates)))]
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = color
    occupied[top + rows, left + cols] = True
    return True


def _build_upper_region(
    top_h: int,
    width: int,
    side_by_color: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    upper = np.full((top_h, width), BG, dtype=int)
    occupied = np.zeros((top_h, width), dtype=bool)

    colors = list(side_by_color)
    rng.shuffle(colors)
    for color in colors:
        area = side_by_color[color] ** 2
        max_parts = 3 if area >= 6 else 2
        part_count = int(rng.integers(2, max_parts + 1))
        parts = _random_parts(area, part_count, rng)
        for part_area in parts:
            mask = _random_blob(part_area, rng)
            if not _place_mask(upper, occupied, mask, color, rng):
                return None
    return upper


def _build_lower_region(
    bottom_h: int,
    width: int,
    order: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, dict[str, int]]] | None:
    kinds = []
    start_kind = "h" if bool(rng.integers(0, 2)) else "v"
    current = start_kind
    for _ in order:
        kinds.append(current)
        current = "v" if current == "h" else "h"

    h_count = sum(kind == "h" for kind in kinds)
    v_count = len(order) - h_count
    if h_count > bottom_h or v_count > width:
        return None

    rows = rng.choice(bottom_h, size=h_count, replace=False).tolist()
    cols = rng.choice(width, size=v_count, replace=False).tolist()
    rng.shuffle(rows)
    rng.shuffle(cols)

    specs: dict[int, dict[str, int]] = {}
    row_idx = 0
    col_idx = 0
    for color, kind in zip(order, kinds):
        if kind == "h":
            specs[color] = {"kind": kind, "index": int(rows[row_idx])}
            row_idx += 1
        else:
            specs[color] = {"kind": kind, "index": int(cols[col_idx])}
            col_idx += 1

    lower = np.full((bottom_h, width), BG, dtype=int)
    for color in order:
        spec = specs[color]
        if spec["kind"] == "h":
            lower[spec["index"], :] = color
        else:
            lower[:, spec["index"]] = color
    return lower, specs


def _semantic_checks(grid: np.ndarray, side_by_color: dict[int, int], order: list[int]) -> bool:
    try:
        sep_row = _find_separator_row(grid)
        upper = grid[:sep_row]
        lower = grid[sep_row + 1 :]

        inferred_sides = _infer_side_by_color(upper)
        if inferred_sides != side_by_color:
            return False
        if any(_component_count(upper == color) < 2 for color in side_by_color):
            return False

        colors = _foreground_colors(lower)
        if set(colors) != set(order):
            return False
        lines = _infer_lines(lower, colors)
        inferred_order = _infer_layer_order(lower, lines)
        if inferred_order != order:
            return False

        size_order = sorted(order, key=lambda color: side_by_color[color])
        if order == size_order or order == size_order[::-1]:
            return False
        if np.array_equal(_render_output(side_by_color, order), _render_output(side_by_color, size_order)):
            return False

        centroid_order = _centroid_order(upper, order)
        if centroid_order == order or centroid_order == order[::-1]:
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    color_count = int(rng.integers(3, 5))
    colors = [int(color) for color in rng.choice(COLOR_POOL, size=color_count, replace=False)]
    sides = [int(side) for side in rng.choice(SIDE_CHOICES, size=color_count, replace=False)]
    side_by_color = {color: side for color, side in zip(colors, sides)}

    order = list(colors)
    rng.shuffle(order)
    size_order = sorted(order, key=lambda color: side_by_color[color])
    if order == size_order or order == size_order[::-1]:
        return None

    top_h = int(rng.integers(*TOP_HEIGHT_RANGE))
    bottom_h = int(rng.integers(*BOTTOM_HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))

    upper = _build_upper_region(top_h, width, side_by_color, rng)
    if upper is None:
        return None

    lower_pack = _build_lower_region(bottom_h, width, order, rng)
    if lower_pack is None:
        return None
    lower, _specs = lower_pack

    grid = np.full((top_h + 1 + bottom_h, width), BG, dtype=int)
    grid[:top_h] = upper
    grid[top_h, :] = SEP
    grid[top_h + 1 :] = lower

    if not _semantic_checks(grid, side_by_color, order):
        return None

    expected = _render_output(side_by_color, order)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
