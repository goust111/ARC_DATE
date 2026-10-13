from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_040_2",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask


BLACK = 0
GREY = 5
ORANGE = 7
TOP_COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
GEN_MAX_ATTEMPTS = 500


def _shortest_period(values: list[int]) -> list[int]:
    n = len(values)
    for period in range(1, n + 1):
        ok = True
        for idx, value in enumerate(values):
            if value != values[idx % period]:
                ok = False
                break
        if ok:
            return values[:period]
    return values[:]


def _group_consecutive(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    height, width = map(int, grid.shape)
    row_threshold = max(2, width // 2)
    col_threshold = max(2, (height - 1) // 2)

    row_lines = [r for r in range(1, height) if int(np.count_nonzero(grid[r] == GREY)) > row_threshold]
    col_lines = [c for c in range(width) if int(np.count_nonzero(grid[:, c] == GREY)) > col_threshold]

    assert len(row_lines) >= 4 and len(col_lines) >= 4, "expected a grey cell lattice"
    return row_lines, col_lines


def _black_components(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    complete = []
    deficient = []
    for coords in components_mask(grid == BLACK, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        area = int(coords.shape[0])
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        item = {
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "area": area,
            "bbox_area": bbox_area,
        }
        if area == bbox_area:
            complete.append(item)
        else:
            deficient.append(item)
    complete.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    deficient.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return complete, deficient


def _extract_gaps(
    grid: np.ndarray,
    coords: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[str, int, int, int]]:
    height, width = map(int, grid.shape)
    region = np.zeros_like(grid, dtype=bool)
    region[coords[:, 0], coords[:, 1]] = True
    row_line_set = set(row_lines)
    col_line_set = set(col_lines)
    gaps: list[tuple[str, int, int, int]] = []

    for col in col_lines[1:-1]:
        rows = sorted(
            int(r)
            for r in coords[coords[:, 1] == col, 0]
            if int(r) not in row_line_set
        )
        for segment in _group_consecutive(rows):
            if len(segment) not in (1, 2):
                continue
            if all(
                0 < r < height - 1
                and region[r, col]
                and region[r, col - 1]
                and region[r, col + 1]
                for r in segment
            ):
                gaps.append(("v", int(segment[0]), int(col), int(len(segment))))

    for row in row_lines[1:-1]:
        cols = sorted(
            int(c)
            for c in coords[coords[:, 0] == row, 1]
            if int(c) not in col_line_set
        )
        for segment in _group_consecutive(cols):
            if len(segment) not in (1, 2):
                continue
            if all(
                0 < c < width - 1
                and region[row, c]
                and region[row - 1, c]
                and region[row + 1, c]
                for c in segment
            ):
                gaps.append(("h", int(row), int(segment[0]), int(len(segment))))

    gaps.sort(key=lambda gap: (gap[1], gap[2], gap[0]))
    assert gaps, "expected every deficient region to contain at least one detectable gap"
    return gaps


def _paint_gap_pattern(out: np.ndarray, coords: np.ndarray, gap: tuple[str, int, int, int]) -> None:
    orient, anchor_r, anchor_c, _length = gap
    rows = coords[:, 0]
    cols = coords[:, 1]
    if orient == "v":
        mask = ((rows - int(anchor_r)) % 2) == 0
    else:
        mask = ((cols - int(anchor_c)) % 2) == 0
    out[rows[mask], cols[mask]] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor intact lattice cells by the top-row cycle and stripe broken multi-cell regions from their gap directions.

    Input structure:
    - Row 0 is a repeated color cycle and does not belong to the lattice.
    - Below it, grey pixels form a rectangular cell frame with outer borders and internal separators on a black background.
    - A few missing grey wall segments connect neighboring cell interiors into non-rectangular black regions.

    Rule:
    - Find all black 4-connected regions inside the frame and compare each region's area with its bounding-box area.
    - If a region exactly fills its bbox, treat it as one intact cell and fill it with the next color from the top-row cycle in reading order.
    - Otherwise treat it as a broken multi-cell region: detect every wall-gap segment on separator rows or columns, and for each gap paint an orange/black period-2 stripe pattern across the whole region, anchored to that gap and oriented along the gap's separator direction.
    - Keep the top row and all remaining grey frame pixels unchanged.

    Key insight:
    - The same black-region detection step feeds a branch: rectangular regions are complete cells that use the top-row color cycle, while non-rectangular regions expose separator gaps whose orientation determines the orange stripe tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    row_lines, col_lines = _separator_lines(g)
    top_cycle = _shortest_period([int(v) for v in g[0]])

    complete_regions, deficient_regions = _black_components(g)
    for index, region in enumerate(complete_regions):
        color = int(top_cycle[index % len(top_cycle)])
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    for region in deficient_regions:
        gaps = _extract_gaps(g, region["coords"], row_lines, col_lines)
        for gap in gaps:
            _paint_gap_pattern(out, region["coords"], gap)

    return out


def _repeat_cycle(cycle: list[int], width: int) -> np.ndarray:
    return np.array([cycle[c % len(cycle)] for c in range(width)], dtype=int)


def _cell_bbox(
    row_lines: list[int],
    col_lines: list[int],
    cell_r: int,
    cell_c: int,
) -> tuple[int, int, int, int]:
    r0 = row_lines[cell_r] + 1
    r1 = row_lines[cell_r + 1] - 1
    c0 = col_lines[cell_c] + 1
    c1 = col_lines[cell_c + 1] - 1
    return int(r0), int(c0), int(r1), int(c1)


def _component_signature(group: dict) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted((gap["orient"], gap["length"], gap["offset"]) for gap in group["gaps"]))


def _deficient_region_coords(
    row_lines: list[int],
    col_lines: list[int],
    group: dict,
) -> np.ndarray:
    cells: set[tuple[int, int]] = set()
    for cell_r, cell_c in group["cells"]:
        r0, c0, r1, c1 = _cell_bbox(row_lines, col_lines, cell_r, cell_c)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    for gap in group["gaps"]:
        cells.update(gap["coords"])
    coords = np.array(sorted(cells), dtype=int)
    return coords


def _expected_output(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    top_cycle: list[int],
    groups: list[dict],
    cell_rows: int,
    cell_cols: int,
) -> np.ndarray:
    out = grid.copy()
    deficient_cells = {cell for group in groups for cell in group["cells"]}

    complete_bboxes = []
    for cell_r in range(cell_rows):
        for cell_c in range(cell_cols):
            if (cell_r, cell_c) in deficient_cells:
                continue
            complete_bboxes.append(_cell_bbox(row_lines, col_lines, cell_r, cell_c))
    complete_bboxes.sort(key=lambda box: (box[0], box[1]))

    for index, (r0, c0, r1, c1) in enumerate(complete_bboxes):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(top_cycle[index % len(top_cycle)])

    for group in groups:
        coords = _deficient_region_coords(row_lines, col_lines, group)
        for gap in group["gaps"]:
            if gap["orient"] == "v":
                mask = ((coords[:, 0] - int(gap["anchor"][0])) % 2) == 0
            else:
                mask = ((coords[:, 1] - int(gap["anchor"][1])) % 2) == 0
            out[coords[mask, 0], coords[mask, 1]] = ORANGE
    return out


def _candidate_groups(cell_rows: int, cell_cols: int) -> list[dict]:
    templates = [
        {"cells": [(0, 0), (0, 1)], "edges": [((0, 0), (0, 1))]},
        {"cells": [(0, 0), (1, 0)], "edges": [((0, 0), (1, 0))]},
        {"cells": [(0, 0), (0, 1), (1, 0)], "edges": [((0, 0), (0, 1)), ((0, 0), (1, 0))]},
        {"cells": [(0, 0), (0, 1), (1, 1)], "edges": [((0, 0), (0, 1)), ((0, 1), (1, 1))]},
        {"cells": [(0, 0), (1, 0), (1, 1)], "edges": [((0, 0), (1, 0)), ((1, 0), (1, 1))]},
        {"cells": [(0, 1), (1, 0), (1, 1)], "edges": [((0, 1), (1, 1)), ((1, 0), (1, 1))]},
    ]
    candidates = []
    for template in templates:
        max_r = max(r for r, _ in template["cells"])
        max_c = max(c for _, c in template["cells"])
        for base_r in range(cell_rows - max_r):
            for base_c in range(cell_cols - max_c):
                cells = [(base_r + r, base_c + c) for r, c in template["cells"]]
                edges = [
                    ((base_r + a[0], base_c + a[1]), (base_r + b[0], base_c + b[1]))
                    for a, b in template["edges"]
                ]
                candidates.append({"cells": cells, "edges": edges})
    return candidates


def _build_gap_metadata(
    row_lines: list[int],
    col_lines: list[int],
    cell_h: int,
    cell_w: int,
    edge: tuple[tuple[int, int], tuple[int, int]],
    used_parity: dict[str, int],
) -> dict:
    (r0, c0), (r1, c1) = edge
    if r0 == r1:
        orient = "v"
        limit = cell_h
        separator = col_lines[min(c0, c1) + 1]
        axis_base = row_lines[r0] + 1
    else:
        orient = "h"
        limit = cell_w
        separator = row_lines[min(r0, r1) + 1]
        axis_base = col_lines[c0] + 1

    length = random.choice([1, 2]) if limit >= 4 else 1
    offsets = list(range(limit - length + 1))
    parity = used_parity.get(orient)
    if parity is None:
        random.shuffle(offsets)
    else:
        offsets = [offset for offset in offsets if offset % 2 == parity] or offsets
        random.shuffle(offsets)
    offset = int(offsets[0])
    used_parity.setdefault(orient, offset % 2)

    coords = []
    if orient == "v":
        for delta in range(length):
            coords.append((axis_base + offset + delta, separator))
        anchor = (coords[0][0], separator)
    else:
        for delta in range(length):
            coords.append((separator, axis_base + offset + delta))
        anchor = (separator, coords[0][1])

    return {
        "orient": orient,
        "length": int(length),
        "offset": int(offset),
        "coords": tuple(coords),
        "anchor": anchor,
    }


def _render_input(
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    top_cycle: list[int],
    groups: list[dict],
) -> tuple[np.ndarray, list[int], list[int]]:
    height = 1 + (cell_rows + 1) + cell_rows * cell_h
    width = (cell_cols + 1) + cell_cols * cell_w
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = _repeat_cycle(top_cycle, width)

    row_lines = [1 + idx * (cell_h + 1) for idx in range(cell_rows + 1)]
    col_lines = [idx * (cell_w + 1) for idx in range(cell_cols + 1)]

    for row in row_lines:
        grid[row, :] = GREY
    for col in col_lines:
        grid[1:, col] = GREY

    for group in groups:
        used_parity: dict[str, int] = {}
        group["gaps"] = [
            _build_gap_metadata(row_lines, col_lines, cell_h, cell_w, edge, used_parity)
            for edge in group["edges"]
        ]
        for gap in group["gaps"]:
            for r, c in gap["coords"]:
                grid[int(r), int(c)] = BLACK

    return grid, row_lines, col_lines


def _choose_groups(cell_rows: int, cell_cols: int) -> list[dict] | None:
    candidates = _candidate_groups(cell_rows, cell_cols)
    random.shuffle(candidates)
    target_count = random.choice([1, 1, 2])
    groups = []
    used_cells: set[tuple[int, int]] = set()

    for candidate in candidates:
        cell_set = set(candidate["cells"])
        if cell_set & used_cells:
            continue
        groups.append({"cells": list(candidate["cells"]), "edges": list(candidate["edges"])})
        used_cells |= cell_set
        if len(groups) == target_count:
            break

    if len(groups) != target_count:
        return None
    if cell_rows * cell_cols - len(used_cells) < 1:
        return None
    return groups


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    groups: list[dict],
    top_cycle: list[int],
) -> bool:
    if not (np.any(expected == ORANGE) and np.any(expected != grid)):
        return False
    if len(set(top_cycle)) < 3:
        return False

    signatures = [_component_signature(group) for group in groups]
    if len(signatures) != len(set(signatures)):
        return False

    for group in groups:
        coords = _deficient_region_coords(*_separator_lines(grid), group)
        region_out = expected[coords[:, 0], coords[:, 1]]
        orange_count = int(np.count_nonzero(region_out == ORANGE))
        if orange_count == 0 or orange_count == int(coords.shape[0]):
            return False

    complete_regions, deficient_regions = _black_components(grid)
    if not complete_regions or not deficient_regions:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if len(complete_regions) <= len(top_cycle):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    cell_rows = random.choice([3, 4])
    cell_cols = random.choice([3, 4])
    cell_h = random.choice([3, 4, 5])
    cell_w = random.choice([3, 4, 5])

    height = 1 + (cell_rows + 1) + cell_rows * cell_h
    width = (cell_cols + 1) + cell_cols * cell_w
    if max(height, width) > 30:
        return None

    cycle_len = random.choice([3, 4])
    top_cycle = random.sample(TOP_COLOR_POOL, cycle_len)
    groups = _choose_groups(cell_rows, cell_cols)
    if groups is None:
        return None

    grid, row_lines, col_lines = _render_input(cell_rows, cell_cols, cell_h, cell_w, top_cycle, groups)
    expected = _expected_output(grid, row_lines, col_lines, top_cycle, groups, cell_rows, cell_cols)

    if not _validate_generated(grid, expected, groups, top_cycle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 040/2")

