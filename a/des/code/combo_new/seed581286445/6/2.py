from __future__ import annotations

import random

import numpy as np


DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

HORIZONTAL_DIRECTIONS = ("left", "right")
VERTICAL_DIRECTIONS = ("up", "down")
GENERATE_RETRIES = 300
PLACE_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_dicts(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    mask = grid == int(color)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR_VECTORS.values():
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            comps.append(
                {
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _is_solid_rectangle(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _parse_horizontal_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    coords = comp["coord_set"]
    row_to_cols: dict[int, list[int]] = {}
    for r, c in coords:
        row_to_cols.setdefault(int(r), []).append(int(c))

    row_counts = {r: len(cols) for r, cols in row_to_cols.items()}
    shaft_row = max(row_counts, key=row_counts.get)
    shaft_len = int(row_counts[shaft_row])
    if not (2 <= shaft_len <= 4):
        return None
    if sum(count == shaft_len for count in row_counts.values()) != 1:
        return None

    shaft_cols = sorted(row_to_cols[shaft_row])
    if shaft_cols != list(range(shaft_cols[0], shaft_cols[-1] + 1)):
        return None

    shaft_cells = {(int(shaft_row), int(c)) for c in shaft_cols}
    extras = sorted(set(coords) - shaft_cells)
    if len(extras) not in (1, 2):
        return None

    extra_cols = {c for _r, c in extras}
    if len(extra_cols) != 1:
        return None
    tip_col = next(iter(extra_cols))
    left_col = int(shaft_cols[0])
    right_col = int(shaft_cols[-1])
    if tip_col not in {left_col, right_col}:
        return None
    extra_rows = sorted(r for r, _c in extras)
    if any(abs(int(r) - int(shaft_row)) != 1 for r in extra_rows):
        return None
    if len(extra_rows) != len(set(extra_rows)):
        return None

    return {
        "direction": "left" if tip_col == left_col else "right",
        "axis": "horizontal",
        "line": int(shaft_row),
        "count": int(len(coords)),
        "bbox": comp["bbox"],
        "coords": comp["coords"],
    }


def _parse_vertical_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    coords = comp["coord_set"]
    col_to_rows: dict[int, list[int]] = {}
    for r, c in coords:
        col_to_rows.setdefault(int(c), []).append(int(r))

    col_counts = {c: len(rows) for c, rows in col_to_rows.items()}
    shaft_col = max(col_counts, key=col_counts.get)
    shaft_len = int(col_counts[shaft_col])
    if not (2 <= shaft_len <= 4):
        return None
    if sum(count == shaft_len for count in col_counts.values()) != 1:
        return None

    shaft_rows = sorted(col_to_rows[shaft_col])
    if shaft_rows != list(range(shaft_rows[0], shaft_rows[-1] + 1)):
        return None

    shaft_cells = {(int(r), int(shaft_col)) for r in shaft_rows}
    extras = sorted(set(coords) - shaft_cells)
    if len(extras) not in (1, 2):
        return None

    extra_rows = {r for r, _c in extras}
    if len(extra_rows) != 1:
        return None
    tip_row = next(iter(extra_rows))
    top_row = int(shaft_rows[0])
    bottom_row = int(shaft_rows[-1])
    if tip_row not in {top_row, bottom_row}:
        return None
    extra_cols = sorted(c for _r, c in extras)
    if any(abs(int(c) - int(shaft_col)) != 1 for c in extra_cols):
        return None
    if len(extra_cols) != len(set(extra_cols)):
        return None

    return {
        "direction": "up" if tip_row == top_row else "down",
        "axis": "vertical",
        "line": int(shaft_col),
        "count": int(len(coords)),
        "bbox": comp["bbox"],
        "coords": comp["coords"],
    }


def _parse_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    horizontal = _parse_horizontal_arrow(comp)
    vertical = _parse_vertical_arrow(comp)
    if (horizontal is None) == (vertical is None):
        return None
    return horizontal if horizontal is not None else vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each solid target block in the direction encoded by its paired arrow, using the arrow's pixel count as the travel distance.

    Input structure:
    - The grid has a dominant black background plus one arrow color and one block color.
    - Arrow components are monochrome 4-connected shapes made of a straight shaft and a small head attached at one shaft endpoint.
    - Block components are small monochrome solid rectangles aligned with exactly one arrow's shaft row or shaft column.

    Rule:
    - Identify every arrow's direction from which shaft endpoint carries the head pixels.
    - Count all pixels in that arrow; this count is the movement distance.
    - Pair the arrow with the unique block lying ahead of it on the same shaft row (horizontal arrows) or shaft column (vertical arrows).
    - Keep arrows unchanged, erase each source block, and redraw that block translated by the arrow's direction vector and distance.

    Key insight:
    - The arrow head provides direction while the arrow's total area provides distance, so the same object must be read as both an orientation cue and a hidden numeric instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    height, width = g.shape

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert len(colors) == 2, "expected exactly two foreground colors"

    arrow_color = None
    block_color = None
    arrow_parsed: list[dict[str, object]] = []
    block_comps: list[dict[str, object]] = []

    for color in colors:
        comps = _component_dicts(g, int(color))
        parsed = [_parse_arrow(comp) for comp in comps]
        if comps and all(item is not None for item in parsed):
            arrow_color = int(color)
            arrow_parsed = [item for item in parsed if item is not None]
        if comps and all(_is_solid_rectangle(comp) for comp in comps):
            block_color = int(color)
            block_comps = comps

    assert arrow_color is not None and block_color is not None and arrow_color != block_color, "failed to identify arrows and blocks"
    assert arrow_parsed and block_comps, "expected at least one arrow and one block"

    blocks: list[dict[str, object]] = []
    for comp in block_comps:
        r0, c0, r1, c1 = comp["bbox"]
        blocks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "color": int(block_color),
            }
        )

    arrow_cells = {cell for arrow in arrow_parsed for cell in arrow["coords"]}
    assignments: list[tuple[dict[str, object], int]] = []
    used_blocks: set[int] = set()

    for arrow in arrow_parsed:
        direction = str(arrow["direction"])
        line = int(arrow["line"])
        ar0, ac0, ar1, ac1 = arrow["bbox"]
        candidates: list[int] = []
        for idx, block in enumerate(blocks):
            br0, bc0, br1, bc1 = block["bbox"]
            if direction == "right" and br0 <= line <= br1 and bc0 > ac1:
                candidates.append(int(idx))
            elif direction == "left" and br0 <= line <= br1 and bc1 < ac0:
                candidates.append(int(idx))
            elif direction == "down" and bc0 <= line <= bc1 and br0 > ar1:
                candidates.append(int(idx))
            elif direction == "up" and bc0 <= line <= bc1 and br1 < ar0:
                candidates.append(int(idx))

        assert len(candidates) == 1, "each arrow must match exactly one block"
        block_idx = int(candidates[0])
        assert block_idx not in used_blocks, "a block cannot be paired with multiple arrows"
        used_blocks.add(block_idx)
        assignments.append((arrow, block_idx))

    assert len(used_blocks) == len(blocks) == len(arrow_parsed), "expected one-to-one arrow/block pairing"

    out = g.copy()
    for block_idx in used_blocks:
        for r, c in blocks[block_idx]["coords"]:
            out[int(r), int(c)] = int(bg)

    dest_cells: set[tuple[int, int]] = set()
    for arrow, block_idx in assignments:
        dr, dc = DIR_VECTORS[str(arrow["direction"])]
        distance = int(arrow["count"])
        color = int(blocks[block_idx]["color"])
        for r, c in blocks[block_idx]["coords"]:
            nr = int(r + dr * distance)
            nc = int(c + dc * distance)
            assert 0 <= nr < height and 0 <= nc < width, "translated block leaves the grid"
            assert (nr, nc) not in arrow_cells, "translated block cannot overwrite an arrow"
            dest_cells.add((int(nr), int(nc)))
            out[int(nr), int(nc)] = int(color)

    total_block_area = sum(len(block["coords"]) for block in blocks)
    assert len(dest_cells) == total_block_area, "translated blocks must not overlap"
    return out


def _make_arrow(direction: str, shaft_len: int, tip_size: int) -> dict[str, object]:
    assert direction in DIR_VECTORS
    assert 2 <= shaft_len <= 4
    assert tip_size in (1, 2)
    assert shaft_len > 2 or tip_size == 2, "2+1 arrows are visually ambiguous"

    if tip_size == 1:
        side_offsets = [random.choice([-1, 1])]
    else:
        side_offsets = [-1, 1]

    cells: set[tuple[int, int]] = set()
    if direction == "right":
        cells.update((0, c) for c in range(shaft_len))
        cells.update((dr, shaft_len - 1) for dr in side_offsets)
        axis_row = 0
        axis_col = None
    elif direction == "left":
        cells.update((0, c) for c in range(shaft_len))
        cells.update((dr, 0) for dr in side_offsets)
        axis_row = 0
        axis_col = None
    elif direction == "down":
        cells.update((r, 0) for r in range(shaft_len))
        cells.update((shaft_len - 1, dc) for dc in side_offsets)
        axis_row = None
        axis_col = 0
    else:
        cells.update((r, 0) for r in range(shaft_len))
        cells.update((0, dc) for dc in side_offsets)
        axis_row = None
        axis_col = 0

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm_cells = {(int(r - min_r), int(c - min_c)) for r, c in cells}
    max_r = max(r for r, _c in norm_cells)
    max_c = max(c for _r, c in norm_cells)
    return {
        "cells": sorted(norm_cells),
        "height": int(max_r + 1),
        "width": int(max_c + 1),
        "axis_row": None if axis_row is None else int(axis_row - min_r),
        "axis_col": None if axis_col is None else int(axis_col - min_c),
        "count": int(len(norm_cells)),
    }


def _rectangle_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in range(int(top), int(top + height)) for c in range(int(left), int(left + width))]


def _make_pair(direction: str) -> dict[str, object]:
    shaft_len = random.randint(2, 4)
    tip_size = 2 if shaft_len == 2 else random.randint(1, 2)
    arrow = _make_arrow(direction, shaft_len, tip_size)
    distance = int(arrow["count"])
    block_h = random.randint(1, 2)
    block_w = random.randint(1, 3)
    gap = random.randint(1, 3)

    arrow_cells = [(int(r), int(c)) for r, c in arrow["cells"]]
    if direction == "right":
        axis_row = int(arrow["axis_row"])
        source_top = random.randint(max(0, axis_row - block_h + 1), axis_row)
        source_left = int(arrow["width"] + gap)
        dest_top = source_top
        dest_left = int(source_left + distance)
        shifted_arrow = arrow_cells
    elif direction == "left":
        axis_row = int(arrow["axis_row"])
        source_top = random.randint(max(0, axis_row - block_h + 1), axis_row)
        dest_top = source_top
        dest_left = 0
        source_left = int(dest_left + distance)
        arrow_shift = int(source_left + block_w + gap)
        shifted_arrow = [(int(r), int(c + arrow_shift)) for r, c in arrow_cells]
    elif direction == "down":
        axis_col = int(arrow["axis_col"])
        source_left = random.randint(max(0, axis_col - block_w + 1), axis_col)
        source_top = int(arrow["height"] + gap)
        dest_left = source_left
        dest_top = int(source_top + distance)
        shifted_arrow = arrow_cells
    else:
        axis_col = int(arrow["axis_col"])
        source_left = random.randint(max(0, axis_col - block_w + 1), axis_col)
        dest_left = source_left
        dest_top = 0
        source_top = int(dest_top + distance)
        arrow_shift = int(source_top + block_h + gap)
        shifted_arrow = [(int(r + arrow_shift), int(c)) for r, c in arrow_cells]

    source_cells = _rectangle_cells(int(source_top), int(source_left), int(block_h), int(block_w))
    dest_cells = _rectangle_cells(int(dest_top), int(dest_left), int(block_h), int(block_w))
    all_cells = set(shifted_arrow) | set(source_cells) | set(dest_cells)
    height = max(r for r, _c in all_cells) + 1
    width = max(c for _r, c in all_cells) + 1

    return {
        "direction": direction,
        "arrow_cells": shifted_arrow,
        "source_cells": source_cells,
        "dest_cells": dest_cells,
        "all_cells": sorted(all_cells),
        "height": int(height),
        "width": int(width),
    }


def _sample_directions(count: int) -> list[str]:
    directions = [random.choice(HORIZONTAL_DIRECTIONS), random.choice(VERTICAL_DIRECTIONS)]
    remaining = [d for d in DIR_VECTORS if d not in directions]
    random.shuffle(remaining)
    while len(directions) < count:
        directions.append(remaining.pop())
    random.shuffle(directions)
    return directions


def _place_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    pair: dict[str, object],
    arrow_color: int,
    block_color: int,
) -> bool:
    height, width = grid.shape
    pair_h = int(pair["height"])
    pair_w = int(pair["width"])
    if pair_h > height or pair_w > width:
        return False

    for _ in range(PLACE_RETRIES):
        top = random.randint(0, height - pair_h)
        left = random.randint(0, width - pair_w)
        if any(bool(reserved[top + r, left + c]) for r, c in pair["all_cells"]):
            continue

        for r, c in pair["arrow_cells"]:
            grid[top + r, left + c] = int(arrow_color)
        for r, c in pair["source_cells"]:
            grid[top + r, left + c] = int(block_color)

        for r, c in pair["all_cells"]:
            gr = int(top + r)
            gc = int(left + c)
            r0 = max(0, gr - 1)
            r1 = min(height, gr + 2)
            c0 = max(0, gc - 1)
            c1 = min(width, gc + 2)
            reserved[r0:r1, c0:c1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 30)
    width = random.randint(15, 30)
    grid = np.zeros((height, width), dtype=int)

    arrow_color, block_color = random.sample(list(range(1, 10)), 2)
    pair_count = random.randint(2, 4)
    directions = _sample_directions(pair_count)
    pairs = sorted((_make_pair(direction) for direction in directions), key=lambda item: item["height"] * item["width"], reverse=True)

    reserved = np.zeros_like(grid, dtype=bool)
    for pair in pairs:
        if not _place_pair(grid, reserved, pair, int(arrow_color), int(block_color)):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if out.shape != grid.shape or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
