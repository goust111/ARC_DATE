from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import label_mask, neighbors4


GREY = 5
ORANGE = 7
MARKER_COLORS = (1, 2, 3, 4, 6)
SIDE_CHOICES = (2, 3, 4, 5, 6)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATE_RETRIES = 400


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _component_summary(grid: np.ndarray) -> list[dict[str, int | tuple[int, ...]]]:
    bg_lab, n_bg = label_mask(grid == 0, connectivity=4)
    sizes = {comp_id: int((bg_lab == comp_id).sum()) for comp_id in range(1, n_bg + 1)}
    order = sorted(sizes, key=lambda comp_id: (-sizes[comp_id], comp_id))
    exterior_ids = set(order[:2])

    summary: list[dict[str, int | tuple[int, ...]]] = []
    for comp_id in range(1, n_bg + 1):
        if comp_id in exterior_ids:
            continue
        coords = np.argwhere(bg_lab == comp_id)
        marker_colors = set()
        touches_border = False
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            if r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1):
                touches_border = True
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    v = int(grid[nr, nc])
                    if v not in (0, GREY, ORANGE):
                        marker_colors.add(v)
        summary.append(
            {
                "area_without_marker": int(coords.shape[0]),
                "marker_colors": tuple(sorted(marker_colors)),
                "touches_border": int(touches_border),
                "component_id": int(comp_id),
                "size": int(sizes[comp_id]),
            }
        )
    return summary


def _render_output(squares: list[tuple[int, int]]) -> np.ndarray:
    squares = sorted((int(side), int(color)) for side, color in squares)
    height = max(side for side, _color in squares)
    width = sum(side for side, _color in squares) + len(squares) - 1
    out = np.zeros((height, width), dtype=int)
    col = 0
    for side, color in squares:
        out[height - side : height, col : col + side] = int(color)
        col += side + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each enclosed marker-bearing compartment into a solid square whose side length equals the compartment area's square root.

    Input structure:
    - Black cells are background.
    - Grey one-cell-thick contour fragments and orange obstacle pixels partition the background.
    - Every enclosed usable background region has exactly one adjacent marker pixel of its own color; marker colors are distinct non-grey, non-orange colors.
    - The two largest black 4-connected regions are the outside space on the two sides of the main contour drawing.

    Rule:
    - Treat every non-black cell as an obstacle and split the black cells into 4-connected components.
    - Discard the two largest black components as exterior background.
    - For every remaining component, read the unique adjacent marker color and count the component plus that marker cell itself.
    - Convert each counted area A into a solid square of side sqrt(A), then place the squares left-to-right in increasing side order with one black column between them and all bottoms aligned.

    Key insight:
    - The contour geometry only encodes a number: the exact enclosed fillable area after subtracting fixed orange blockers; once that number is decoded, all original geometry is discarded and rebuilt as size-sorted squares.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color_to_area: dict[int, int] = {}
    for comp in _component_summary(g):
        marker_colors = tuple(comp["marker_colors"])
        assert int(comp["touches_border"]) == 0, "interior component touches border"
        assert len(marker_colors) == 1, "each enclosed region must have exactly one marker color"
        marker_color = int(marker_colors[0])
        color_to_area[marker_color] = color_to_area.get(marker_color, 0) + int(comp["area_without_marker"]) + 1

    assert color_to_area, "expected at least one enclosed marker-bearing region"

    squares: list[tuple[int, int]] = []
    for color, area in color_to_area.items():
        side = math.isqrt(int(area))
        assert side * side == int(area), f"region area {area} is not a perfect square"
        squares.append((int(side), int(color)))
    return _render_output(squares)


def _choose_marker(fill_cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = []
    for cell in fill_cells:
        if _connected(set(fill_cells) - {cell}):
            candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _place_oranges(fill_cells: set[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    oranges: set[tuple[int, int]] = set()
    candidates = list(fill_cells)
    random.shuffle(candidates)
    for _ in range(int(count)):
        placed = False
        for cell in candidates:
            if cell in oranges:
                continue
            remaining = set(fill_cells) - (oranges | {cell})
            if _connected(remaining):
                oranges.add(cell)
                placed = True
                break
        if not placed:
            return None
    return oranges


def _build_region_patch(side: int, color: int, with_orange: bool) -> tuple[np.ndarray, int]:
    options = []
    for interior_h in range(side, side + 3):
        for interior_w in range(side, side + 3):
            excess = interior_h * interior_w - side * side
            if excess >= 1:
                options.append((interior_h, interior_w, excess))
    random.shuffle(options)

    for interior_h, interior_w, excess in options:
        if with_orange and excess < 2:
            continue
        orange_count = random.randint(1, min(3, excess - 1)) if with_orange else 0
        notch_count = excess - orange_count

        orientations = []
        if notch_count <= interior_w:
            orientations.extend(("top", "bottom"))
        if notch_count <= interior_h:
            orientations.extend(("left", "right"))
        random.shuffle(orientations)

        for orientation in orientations:
            patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
            patch[0, :] = GREY
            patch[-1, :] = GREY
            patch[:, 0] = GREY
            patch[:, -1] = GREY

            if orientation == "top":
                start = random.randint(1, interior_w - notch_count + 1)
                notch_cells = {(1, c) for c in range(start, start + notch_count)}
            elif orientation == "bottom":
                start = random.randint(1, interior_w - notch_count + 1)
                notch_cells = {(interior_h, c) for c in range(start, start + notch_count)}
            elif orientation == "left":
                start = random.randint(1, interior_h - notch_count + 1)
                notch_cells = {(r, 1) for r in range(start, start + notch_count)}
            else:
                start = random.randint(1, interior_h - notch_count + 1)
                notch_cells = {(r, interior_w) for r in range(start, start + notch_count)}

            for r, c in notch_cells:
                patch[r, c] = GREY

            fill_cells = {
                (r, c)
                for r in range(1, interior_h + 1)
                for c in range(1, interior_w + 1)
                if (r, c) not in notch_cells
            }
            orange_cells = _place_oranges(fill_cells, orange_count)
            if orange_cells is None:
                continue
            marker = _choose_marker(fill_cells - orange_cells)
            if marker is None:
                continue

            for r, c in orange_cells:
                patch[r, c] = ORANGE
            patch[marker] = int(color)
            return patch, int(orange_count)

    raise RuntimeError("failed to build a valid enclosed region patch")


def _assemble_grid(
    regions: list[tuple[int, int, bool, str]]
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    left_regions = [spec for spec in regions if spec[3] == "L"]
    right_regions = [spec for spec in regions if spec[3] == "R"]
    random.shuffle(left_regions)
    random.shuffle(right_regions)

    built_left = [(*_build_region_patch(side, color, with_orange), side, color) for side, color, with_orange, _lane in left_regions]
    built_right = [(*_build_region_patch(side, color, with_orange), side, color) for side, color, with_orange, _lane in right_regions]

    left_gaps = [random.randint(1, 3) for _ in range(max(0, len(built_left) - 1))]
    right_gaps = [random.randint(1, 3) for _ in range(max(0, len(built_right) - 1))]

    left_height = sum(patch.shape[0] for patch, _orange_count, _side, _color in built_left) + sum(left_gaps)
    right_height = sum(patch.shape[0] for patch, _orange_count, _side, _color in built_right) + sum(right_gaps)
    max_left_w = max((patch.shape[1] for patch, _orange_count, _side, _color in built_left), default=0)
    max_right_w = max((patch.shape[1] for patch, _orange_count, _side, _color in built_right), default=0)

    height = max(18, left_height, right_height) + random.randint(2, 4)
    width = max(18, max_left_w + max_right_w + 3 + random.randint(0, 2))
    if height > 30 or width > 30:
        raise RuntimeError("assembled grid would exceed ARC size limits")

    trunk_col = max_left_w + 1
    grid = np.zeros((height, width), dtype=int)
    grid[:, trunk_col] = GREY

    orange_counts: list[int] = []
    metadata: list[tuple[int, int]] = []

    def place_lane(items: list[tuple[np.ndarray, int, int, int]], gaps: list[int], lane: str) -> None:
        total_h = sum(patch.shape[0] for patch, _orange_count, _side, _color in items) + sum(gaps)
        top = random.randint(1, height - total_h - 1)
        row = top
        for idx, (patch, orange_count, side, color) in enumerate(items):
            col = trunk_col - patch.shape[1] if lane == "L" else trunk_col + 1
            coords = np.argwhere(patch != 0)
            for r_raw, c_raw in coords:
                r = row + int(r_raw)
                c = col + int(c_raw)
                if grid[r, c] != 0:
                    raise RuntimeError("region patches overlapped during placement")
                grid[r, c] = int(patch[r_raw, c_raw])
            orange_counts.append(int(orange_count))
            metadata.append((int(side), int(color)))
            row += patch.shape[0] + (gaps[idx] if idx < len(gaps) else 0)

    place_lane(built_left, left_gaps, "L")
    place_lane(built_right, right_gaps, "R")
    return grid, metadata, orange_counts


def _semantic_checks(
    grid: np.ndarray,
    expected_pairs: list[tuple[int, int]],
    orange_counts: list[int],
    want_orange_flags: list[bool],
) -> bool:
    summary = _component_summary(grid)
    if len(summary) != len(expected_pairs):
        return False

    seen_pairs = []
    for comp in summary:
        marker_colors = tuple(comp["marker_colors"])
        if int(comp["touches_border"]) != 0 or len(marker_colors) != 1:
            return False
        area = int(comp["area_without_marker"]) + 1
        side = math.isqrt(area)
        if side * side != area:
            return False
        seen_pairs.append((side, int(marker_colors[0])))

    if sorted(seen_pairs) != sorted((int(side), int(color)) for side, color in expected_pairs):
        return False

    if sorted(orange_counts) == [0] * len(orange_counts):
        return False
    if len(orange_counts) > 1 and all(count > 0 for count in orange_counts):
        return False
    if len(orange_counts) != len(want_orange_flags):
        return False
    if sum(count > 0 for count in orange_counts) != sum(bool(flag) for flag in want_orange_flags):
        return False

    output = main(grid)
    return bool(np.array_equal(output, _render_output(expected_pairs)))


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(3, 5)
    colors = random.sample(MARKER_COLORS, region_count)
    sides = random.sample(SIDE_CHOICES, region_count)

    want_orange = [False] * region_count
    order = list(range(region_count))
    random.shuffle(order)
    want_orange[order[0]] = True
    if region_count > 1:
        want_orange[order[1]] = False
    for idx in order[2:]:
        want_orange[idx] = bool(random.random() < 0.7)

    left_count = region_count // 2
    lanes = ["L"] * left_count + ["R"] * (region_count - left_count)
    random.shuffle(lanes)
    regions = [
        (int(side), int(color), bool(flag), lane)
        for side, color, flag, lane in zip(sides, colors, want_orange, lanes)
    ]

    try:
        grid, expected_pairs, orange_counts = _assemble_grid(regions)
    except RuntimeError:
        return None

    expected_by_color = {color: side for side, color in expected_pairs}
    if len(expected_by_color) != region_count:
        return None
    if len(set(expected_by_color.values())) != region_count:
        return None
    if not _semantic_checks(grid, expected_pairs, orange_counts, want_orange):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
