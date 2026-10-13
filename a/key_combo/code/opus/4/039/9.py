from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, find_components


RANK_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK]
GEN_RETRY_LIMIT = 400


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _region_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color.GREY)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == Color.GREY)]

    row_bands = _runs([r for r in range(grid.shape[0]) if r not in set(grey_rows)])
    col_bands = _runs([c for c in range(grid.shape[1]) if c not in set(grey_cols)])

    boxes = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            boxes.append((r0, c0, r1, c1))
    return boxes


def _orange_count(region: np.ndarray) -> int:
    comps = find_components(
        region,
        bg=Color.BLACK,
        colors=[Color.ORANGE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(comps) == 1, "each region must contain exactly one orange 4-connected sprite"
    return int(comps[0]["area"])


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each divider-bounded region background by sorting the orange sprites by pixel count.

    Input structure:
    - The grid is black except for full grey divider rows and/or columns that partition it into 4 or 5 rectangles.
    - Each rectangle contains exactly one orange 4-connected sprite.
    - Sprites do not touch the grey dividers, and their pixel counts are all distinct.

    Rule:
    - Detect the rectangular regions defined by complete grey divider lines and the outer grid boundary.
    - Count the orange pixels in the single sprite inside each region.
    - Sort regions by sprite size ascending, breaking ties by the region top-left coordinate.
    - Recolor only the black background of the ranked regions: smallest -> blue, then red, green, yellow, and magenta for the fifth region. Keep grey dividers and orange sprites unchanged.

    Key insight:
    - The fill color is determined by comparing sprite pixel counts across regions, not by the regions' spatial positions.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    boxes = _region_boxes(grid)
    assert 4 <= len(boxes) <= 5, "expected 4 or 5 divider-bounded regions"

    region_info = []
    for box in boxes:
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        area = _orange_count(region)
        region_info.append((area, r0, c0, box))

    ordered = sorted(region_info, key=lambda item: (item[0], item[1], item[2]))
    out = grid.copy()
    for rank, (_, _r0, _c0, box) in enumerate(ordered):
        color = int(RANK_COLORS[rank])
        r0, c0, r1, c1 = box
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == Color.BLACK] = color
    return out


def _bounded_partition(total: int, parts: int, low: int, high: int, rng: np.random.Generator) -> list[int] | None:
    values = [low] * parts
    remain = total - low * parts
    if remain < 0 or total > high * parts:
        return None
    while remain > 0:
        candidates = [idx for idx, value in enumerate(values) if value < high]
        if not candidates:
            return None
        idx = int(rng.choice(candidates))
        values[idx] += 1
        remain -= 1
    rng.shuffle(values)
    return values


def _make_layout(rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    layout = rng.choice(["cross4", "strip4h", "strip4v", "strip5h", "strip5v"])
    if layout == "cross4":
        row_parts, col_parts = 2, 2
    elif layout == "strip4h":
        row_parts, col_parts = 1, 4
    elif layout == "strip4v":
        row_parts, col_parts = 4, 1
    elif layout == "strip5h":
        row_parts, col_parts = 1, 5
    else:
        row_parts, col_parts = 5, 1

    max_rows_total = 30 - (row_parts - 1)
    max_cols_total = 30 - (col_parts - 1)

    row_total = int(rng.integers(row_parts * 5, min(row_parts * 8, max_rows_total) + 1))
    col_total = int(rng.integers(col_parts * 5, min(col_parts * 8, max_cols_total) + 1))

    row_sizes = _bounded_partition(row_total, row_parts, 5, 8, rng)
    col_sizes = _bounded_partition(col_total, col_parts, 5, 8, rng)
    if row_sizes is None or col_sizes is None:
        return None
    return row_sizes, col_sizes


def _boxes_from_sizes(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    grid = np.zeros((height, width), dtype=int)

    row_starts = []
    cursor = 0
    for size in row_sizes:
        row_starts.append(cursor)
        cursor += size + 1

    col_starts = []
    cursor = 0
    for size in col_sizes:
        col_starts.append(cursor)
        cursor += size + 1

    for start, size in zip(row_starts[:-1], row_sizes[:-1]):
        grid[start + size, :] = Color.GREY
    for start, size in zip(col_starts[:-1], col_sizes[:-1]):
        grid[:, start + size] = Color.GREY

    boxes = []
    for r_start, r_size in zip(row_starts, row_sizes):
        for c_start, c_size in zip(col_starts, col_sizes):
            boxes.append((r_start, c_start, r_start + r_size - 1, c_start + c_size - 1))
    return grid, boxes


def _random_connected_shape(area: int, max_h: int, max_w: int, rng: np.random.Generator) -> np.ndarray | None:
    if area > max_h * max_w:
        return None
    for _ in range(80):
        cells = {(int(rng.integers(max_h)), int(rng.integers(max_w)))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in cells:
            shape[r - r0, c - c0] = 1
        return shape
    return None


def _assign_sizes(capacities: list[int], rng: np.random.Generator) -> list[int] | None:
    region_count = len(capacities)
    candidates = list(range(3, 11))
    for _ in range(200):
        chosen = sorted(rng.choice(candidates, size=region_count, replace=False).tolist())
        caps_sorted = sorted(capacities)
        if all(size <= cap for size, cap in zip(chosen, caps_sorted)):
            assigned = chosen[:]
            rng.shuffle(assigned)
            if all(size <= cap for size, cap in zip(assigned, capacities)):
                return [int(size) for size in assigned]
    return None


def _position_output(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for rank, box in enumerate(sorted(boxes, key=lambda item: (item[0], item[1]))):
        r0, c0, r1, c1 = box
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == Color.BLACK] = int(RANK_COLORS[rank])
    return out


def _validate_example(grid: np.ndarray) -> bool:
    boxes = _region_boxes(grid)
    if not (4 <= len(boxes) <= 5):
        return False

    counts = []
    for r0, c0, r1, c1 in boxes:
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        comps = find_components(
            region,
            bg=Color.BLACK,
            colors=[Color.ORANGE],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
        if len(comps) != 1:
            return False
        comp = comps[0]
        if comp["area"] < 3 or comp["area"] > 10:
            return False
        coords = comp["coords"]
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == region.shape[0] - 1):
            return False
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == region.shape[1] - 1):
            return False
        counts.append(int(comp["area"]))

    if len(set(counts)) != len(counts):
        return False

    actual = main(grid)
    if np.array_equal(actual, grid):
        return False
    if np.array_equal(actual, _position_output(grid, boxes)):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        layout = _make_layout(rng)
        if layout is None:
            continue
        row_sizes, col_sizes = layout
        grid, boxes = _boxes_from_sizes(row_sizes, col_sizes)

        capacities = []
        for r0, c0, r1, c1 in boxes:
            capacities.append((r1 - r0 - 1) * (c1 - c0 - 1))
        sizes = _assign_sizes(capacities, rng)
        if sizes is None:
            continue

        position_sorted = sorted(range(len(boxes)), key=lambda idx: (boxes[idx][0], boxes[idx][1]))
        size_sorted = sorted(range(len(boxes)), key=lambda idx: sizes[idx])
        if position_sorted == size_sorted:
            continue

        ok = True
        for box, size in zip(boxes, sizes):
            r0, c0, r1, c1 = box
            interior_h = r1 - r0 - 1
            interior_w = c1 - c0 - 1
            shape = _random_connected_shape(size, interior_h, interior_w, rng)
            if shape is None:
                ok = False
                break
            shape_h, shape_w = shape.shape
            top = r0 + 1 + int(rng.integers(interior_h - shape_h + 1))
            left = c0 + 1 + int(rng.integers(interior_w - shape_w + 1))
            grid[top : top + shape_h, left : left + shape_w][shape.astype(bool)] = Color.ORANGE
        if not ok:
            continue

        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
