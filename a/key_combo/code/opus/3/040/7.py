from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    _ARC_UTILS = SourceFileLoader("_arc_utils_shared_040_7", str(_ARC_UTILS_PATH)).load_module()
    Color = _ARC_UTILS.Color
except Exception:
    class Color:
        BLACK = 0
        GREY = 5
        GRAY = 5


BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
VERTICAL_4X4_LAYOUTS = (
    ([6, 2, 3, 2], [7, 8, 4, 7], (1, 2)),
    ([3, 2, 4, 2], [8, 8, 6, 2], (2, 1)),
    ([3, 2, 2, 7], [7, 5, 7, 2], (2, 1)),
    ([8, 3, 2, 5], [6, 8, 6, 6], (1, 1)),
    ([2, 2, 3, 2], [7, 5, 7, 6], (1, 1)),
)
HORIZONTAL_4X4_LAYOUTS = (
    ([4, 7, 8, 4], [5, 2, 2, 2], (2, 1)),
    ([3, 7, 7, 8], [4, 3, 2, 4], (2, 1)),
    ([8, 4, 7, 6], [3, 2, 2, 6], (1, 2)),
    ([5, 6, 7, 7], [2, 2, 4, 4], (2, 2)),
    ([8, 8, 6, 2], [8, 2, 2, 4], (1, 1)),
)


def _full_constant_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(r, int(grid[r, 0])) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]


def _full_constant_cols(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(c, int(grid[0, c])) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]


def _find_divider_color(grid: np.ndarray) -> int:
    row_colors = {color for _idx, color in _full_constant_rows(grid)}
    col_colors = {color for _idx, color in _full_constant_cols(grid)}
    shared = sorted(row_colors & col_colors)
    assert len(shared) == 1, "expected a unique divider color shared by full rows and full columns"
    return int(shared[0])


def _partition_spans(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers) + [size]:
        if start < divider:
            spans.append((start, divider))
        start = divider + 1
    return spans


def _center_of_span(start: int, end: int) -> int:
    return start + (end - start - 1) // 2


def _collect_regions(grid: np.ndarray) -> tuple[int, set[int], set[int], list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    divider = _find_divider_color(grid)
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == divider)]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == divider)]

    row_spans = _partition_spans(grid.shape[0], divider_rows)
    col_spans = _partition_spans(grid.shape[1], divider_cols)
    assert len(row_spans) == len(col_spans) and len(row_spans) in (3, 4), "expected a 3x3 or 4x4 partition grid"

    regions: list[dict] = []
    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            sub = grid[r0:r1, c0:c1]
            mask = (sub != BLACK) & (sub != divider)
            count = int(mask.sum())
            assert count > 0, "every region must contain at least one colored pixel"
            colors = sorted(set(map(int, sub[mask].tolist())))
            assert len(colors) == 1, "each region must contain exactly one non-black, non-divider color"
            regions.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "center_r": _center_of_span(r0, r1),
                    "center_c": _center_of_span(c0, c1),
                    "count": count,
                    "color": colors[0],
                }
            )

    return divider, set(divider_rows), set(divider_cols), row_spans, col_spans, regions


def _plan_connectors(
    regions: list[dict],
    hub: dict,
    divider_rows: set[int],
    divider_cols: set[int],
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    claims: dict[tuple[int, int], int] = {}
    info = {"vertical": 0, "horizontal": 0, "ties": 0, "zero": 0, "positive_cells": 0}

    for idx, region in enumerate(regions):
        if region is hub:
            continue

        dr = abs(region["center_r"] - hub["center_r"])
        dc = abs(region["center_c"] - hub["center_c"])
        if dr == dc:
            info["ties"] += 1

        cells: list[tuple[int, int]] = []
        if dr <= dc:
            step = 1 if hub["center_r"] > region["center_r"] else -1
            for rr in range(region["center_r"], hub["center_r"], step):
                if rr not in divider_rows:
                    cells.append((rr, region["center_c"]))
            if cells:
                info["vertical"] += 1
            else:
                info["zero"] += 1
        else:
            step = 1 if hub["center_c"] > region["center_c"] else -1
            for cc in range(region["center_c"], hub["center_c"], step):
                if cc not in divider_cols:
                    cells.append((region["center_r"], cc))
            if cells:
                info["horizontal"] += 1
            else:
                info["zero"] += 1

        for cell in cells:
            if cell in claims:
                prev_idx = claims[cell]
                assert regions[prev_idx]["color"] == region["color"], "overlapping connectors must share one color"
            else:
                claims[cell] = idx

    info["positive_cells"] = len(claims)
    return claims, info


def _solve(grid: np.ndarray, *, return_info: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    divider, divider_rows, divider_cols, _row_spans, _col_spans, regions = _collect_regions(g)
    max_count = max(region["count"] for region in regions)
    hubs = [region for region in regions if region["count"] == max_count]
    assert len(hubs) == 1, "expected a unique densest region"
    hub = hubs[0]

    row_count = len({region["row_idx"] for region in regions})
    col_count = len({region["col_idx"] for region in regions})
    assert 0 < hub["row_idx"] < row_count - 1, "densest region must not be on the outer partition row"
    assert 0 < hub["col_idx"] < col_count - 1, "densest region must not be on the outer partition column"

    claims, info = _plan_connectors(regions, hub, divider_rows, divider_cols)

    out = np.full_like(g, BLACK)
    out[g == divider] = divider
    out[hub["r0"] : hub["r1"], hub["c0"] : hub["c1"]] = hub["color"]
    out[hub["center_r"], :] = divider
    out[:, hub["center_c"]] = divider

    for cell, region_idx in claims.items():
        rr, cc = cell
        assert out[rr, cc] != divider, "connectors must stop before grey divider/axis cells"
        out[rr, cc] = regions[region_idx]["color"]

    if return_info:
        return out, info
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the densest partition to define a grey cross-axis, then connect every other partition to the nearer axis.

    Input structure:
    - The grid is split by full grey divider rows and columns into a 3x3 or 4x4 array of rectangular partitions.
    - Each partition has black background plus scattered pixels of exactly one non-black, non-grey color.
    - Exactly one non-edge partition contains strictly more colored pixels than every other partition.

    Rule:
    - Count the non-black, non-grey pixels in each partition and find the unique densest one.
    - Flood-fill that winning partition with its color, then draw a full-width grey row and a full-height grey column through its median interior row and column.
    - Erase every other partition's scattered dots and replace each partition by a straight line from its center toward the nearer trunk axis, breaking only at existing grey divider cells and excluding the axis cell itself; ties choose the horizontal trunk axis.

    Key insight:
    - The trunk is not marked in the input: its row and column emerge from the location of the globally densest partition, so a counting decision controls all subsequent geometry.
    """
    return _solve(grid)


def _region_positions(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        spans.append((cursor, cursor + size))
        cursor += size
        if idx != len(sizes) - 1:
            cursor += 1
    return spans


def _paint_dividers(grid: np.ndarray, divider: int, row_sizes: list[int], col_sizes: list[int]) -> None:
    r_cursor = row_sizes[0]
    for size in row_sizes[1:]:
        grid[r_cursor, :] = divider
        r_cursor += 1 + size

    c_cursor = col_sizes[0]
    for size in col_sizes[1:]:
        grid[:, c_cursor] = divider
        c_cursor += 1 + size


def _sample_mode() -> str:
    return random.choices(
        ["tie3", "asym3", "vert4", "horiz4"],
        weights=[0.32, 0.33, 0.18, 0.17],
        k=1,
    )[0]


def _sample_sizes(mode: str) -> tuple[list[int], list[int]]:
    if mode == "tie3":
        side = random.randint(4, 6)
        return [side, side, side], [side, side, side]

    if mode == "asym3":
        while True:
            row_sizes = [random.randint(3, 6) for _ in range(3)]
            col_sizes = [random.randint(3, 6) for _ in range(3)]
            if len(set(row_sizes)) > 1 or len(set(col_sizes)) > 1:
                return row_sizes, col_sizes

    layouts = VERTICAL_4X4_LAYOUTS if mode == "vert4" else HORIZONTAL_4X4_LAYOUTS
    row_sizes, col_sizes, _hub = random.choice(layouts)
    return list(row_sizes), list(col_sizes)


def _hub_position(mode: str) -> tuple[int, int]:
    if mode in ("tie3", "asym3"):
        return 1, 1
    layouts = VERTICAL_4X4_LAYOUTS if mode == "vert4" else HORIZONTAL_4X4_LAYOUTS
    _row_sizes, _col_sizes, hub = random.choice(layouts)
    return hub


def _color_for_region(row_idx: int, col_idx: int, offset: int, mode: str) -> int:
    if mode == "vert4":
        return int(PALETTE[(offset + 3 * col_idx) % len(PALETTE)])
    if mode == "horiz4":
        return int(PALETTE[(offset + 3 * row_idx) % len(PALETTE)])
    return int(PALETTE[(offset + 3 * row_idx + 5 * col_idx) % len(PALETTE)])


def _scatter_region(grid: np.ndarray, region: tuple[int, int, int, int], color: int, count: int) -> None:
    r0, r1, c0, c1 = region
    coords = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    for r, c in random.sample(coords, count):
        grid[r, c] = color


def _make_candidate(mode: str) -> np.ndarray:
    if mode in ("vert4", "horiz4"):
        layouts = VERTICAL_4X4_LAYOUTS if mode == "vert4" else HORIZONTAL_4X4_LAYOUTS
        row_sizes, col_sizes, (hub_row, hub_col) = random.choice(layouts)
        row_sizes = list(row_sizes)
        col_sizes = list(col_sizes)
    else:
        row_sizes, col_sizes = _sample_sizes(mode)
        hub_row, hub_col = _hub_position(mode)

    n = len(row_sizes)

    height = sum(row_sizes) + (n - 1)
    width = sum(col_sizes) + (n - 1)
    assert max(height, width) <= 30

    grid = np.full((height, width), BLACK, dtype=int)
    _paint_dividers(grid, GREY, row_sizes, col_sizes)

    row_spans = _region_positions(row_sizes)
    col_spans = _region_positions(col_sizes)
    areas = {(r, c): row_sizes[r] * col_sizes[c] for r in range(n) for c in range(n)}

    hub_area = areas[(hub_row, hub_col)]
    hub_count = random.randint(max(4, min(6, hub_area)), min(8, hub_area))
    challenger = random.choice([(r, c) for r in range(n) for c in range(n) if (r, c) != (hub_row, hub_col)])

    offset = random.randrange(len(PALETTE))
    for row_idx in range(n):
        for col_idx in range(n):
            color = _color_for_region(row_idx, col_idx, offset, mode)
            area = areas[(row_idx, col_idx)]
            max_other = min(area, hub_count - 1, 7)
            assert max_other >= 1
            if (row_idx, col_idx) == (hub_row, hub_col):
                count = hub_count
            elif (row_idx, col_idx) == challenger:
                count = max_other
            else:
                count = random.randint(1, max_other)

            r0, r1 = row_spans[row_idx]
            c0, c1 = col_spans[col_idx]
            _scatter_region(grid, (r0, r1, c0, c1), color, count)

    return grid


def _valid_candidate(grid: np.ndarray, mode: str) -> bool:
    try:
        _out, info = _solve(grid, return_info=True)
    except AssertionError:
        return False

    if info["positive_cells"] < 4:
        return False

    if mode == "tie3":
        return info["ties"] >= 4 and info["vertical"] >= 4

    if mode == "asym3":
        return info["vertical"] >= 1 and info["horizontal"] >= 1

    if mode == "vert4":
        return info["vertical"] >= 6 and info["horizontal"] == 0

    if mode == "horiz4":
        return info["horizontal"] >= 6 and info["vertical"] == 0

    if info["vertical"] < 1 or info["horizontal"] < 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = _sample_mode()
        grid = _make_candidate(mode)
        if _valid_candidate(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
