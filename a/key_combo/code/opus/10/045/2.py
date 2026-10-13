from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = [BLUE, RED, GREEN, YELLOW]
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    grey_mask = grid == GREY
    grey_components = _connected_components(grey_mask)
    assert len(grey_components) == 1, "expected exactly one grey component"
    grey = grey_components[0]

    pieces: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BLACK, GREY):
            continue
        for coords in _connected_components(grid == color):
            pieces.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "distance": _min_manhattan_between(coords, grey),
                }
            )
    return grey, pieces


def _min_manhattan_between(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            best = min(best, abs(r1 - r2) + abs(c1 - c2))
    return best


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _in_bounds(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _touches_mass(coords: list[tuple[int, int]], mass: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in mass:
                return True
    return False


def _landing_step(
    coords: list[tuple[int, int]],
    mass: set[tuple[int, int]],
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> int | None:
    dr, dc = direction
    current = list(coords)
    for step in range(max(shape) + 1):
        current_set = set(current)
        if current_set & mass:
            return None
        if not _in_bounds(current, shape):
            return None
        if _touches_mass(current, mass):
            return step
        current = _translate(current, dr, dc)
    return None


def _sorted_pieces(grey: list[tuple[int, int]], pieces: list[dict]) -> list[dict]:
    return sorted(pieces, key=lambda p: (p["distance"], p["bbox"][0], p["bbox"][1]))


def _simulate(
    grid: np.ndarray,
    *,
    accumulate: bool = True,
    reverse_order: bool = False,
    fill_regions: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    grey, pieces = _extract_objects(g)
    ordered = _sorted_pieces(grey, pieces)
    if reverse_order:
        ordered = list(reversed(ordered))

    out = np.zeros_like(g)
    for r, c in grey:
        out[r, c] = GREY

    grey_mass = set(grey)
    mass = set(grey)
    color_rank = {
        int(piece["color"]): (int(piece["distance"]), int(piece["bbox"][0]), int(piece["bbox"][1]))
        for piece in ordered
    }

    for piece in ordered:
        base_mass = mass if accumulate else grey_mass
        candidates: list[tuple[int, tuple[int, int]]] = []
        for direction in DIRECTIONS:
            step = _landing_step(piece["coords"], base_mass, g.shape, direction)
            if step is not None:
                candidates.append((step, direction))
        assert candidates, "piece must be able to reach the accumulated mass"
        candidates.sort(key=lambda item: item[0])
        min_step = candidates[0][0]
        best_dirs = [d for step, d in candidates if step == min_step]
        # The generator avoids ambiguous equal-length directions; the fixed order keeps
        # the function total on valid but noisier inputs.
        direction = sorted(best_dirs)[0]
        final_coords = _translate(piece["coords"], direction[0] * min_step, direction[1] * min_step)
        for r, c in final_coords:
            out[r, c] = int(piece["color"])
        if accumulate:
            mass |= set(final_coords)
        else:
            mass |= set(final_coords)

    if not fill_regions:
        return out

    regions = _background_regions(out)
    for region in regions:
        if region["touches_border"]:
            continue
        counts: Counter[int] = Counter()
        for cell in region["boundary"]:
            color = int(out[cell])
            if color != GREY:
                counts[color] += 1
        if not counts:
            continue
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        fill_color = min(winners, key=lambda color: color_rank.get(color, (10**9, 10**9, 10**9)))
        for r, c in region["cells"]:
            out[r, c] = fill_color
    return out


def _background_regions(grid: np.ndarray) -> list[dict]:
    comps = _connected_components(grid == BLACK)
    h, w = grid.shape
    regions = []
    for comp in comps:
        cell_set = set(comp)
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        boundary: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and grid[nr, nc] != BLACK:
                    boundary.add((nr, nc))
        regions.append({"cells": comp, "touches_border": touches_border, "boundary": boundary})
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each colored shape into the grey accumulator, then fill emergent enclosed background pockets by boundary-majority color.

    Input structure:
    - A black background contains exactly one grey bar and several small monochrome colored shapes.
    - Each non-grey shape is 4-connected, size 2-4, and shares a row or column projection with the grey bar so it can slide cardinally.
    - The grey bar is the initial accumulator; later stopped shapes become part of the accumulator for subsequent slides.

    Rule:
    - Order the colored shapes by increasing Manhattan distance to the grey bar, breaking ties by the row-major order of each shape's bounding-box top-left corner.
    - For each shape in that order, choose the shortest cardinal translation that makes it 4-neighbor-touch the current accumulated mass, translate it there, and keep its original color.
    - After all slides, find every 4-connected black region that does not touch the grid border; for each such enclosed region, count unique boundary pixels by color, ignore grey, choose the most frequent color, and use the input-distance order as the tiebreaker.

    Key insight:
    - The closed pocket is not present initially; it appears only because early arrivals change where later shapes stop, so both the geometry and the fill color depend on the accumulated sliding order.
    """
    return _simulate(np.asarray(grid, dtype=int), accumulate=True, reverse_order=False, fill_regions=True)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k).copy()


def _build_case() -> np.ndarray | None:
    tie_fill = random.random() < 0.5
    extra_piece = random.random() < 0.8
    equal_side_distance = random.random() < 0.35

    inner_w = random.randint(2, 4)
    bar_len = inner_w + 2
    bar_thickness = random.randint(1, 2)

    if tie_fill:
        side_h = inner_w
    else:
        side_choices = [x for x in range(2, 5) if x != inner_w]
        side_h = random.choice(side_choices)

    d_left = random.randint(1, 3)
    d_right = d_left if equal_side_distance else random.choice([x for x in range(1, 4) if x != d_left])
    d_bottom = random.randint(max(d_left, d_right) + 1, max(d_left, d_right) + 3)
    d_extra = random.randint(1, 3) if extra_piece else 0
    extra_len = random.randint(2, min(3, bar_len)) if extra_piece else 0
    extra_offset = random.randint(0, bar_len - extra_len) if extra_piece else 0

    min_h = max(14, (d_extra + 2) + (bar_thickness + side_h + d_bottom + 2))
    min_w = max(14, bar_len + 4)
    if min_h > 18 or min_w > 18:
        return None
    h = random.randint(min_h, 18)
    w = random.randint(min_w, 18)

    top_min = d_extra + 1 if extra_piece else 1
    top_max = h - (bar_thickness + side_h + d_bottom + 2)
    if top_min > top_max:
        return None
    bar_r0 = random.randint(top_min, top_max)
    bar_c0 = random.randint(2, w - bar_len - 2)

    grid = np.zeros((h, w), dtype=int)
    grid[bar_r0 : bar_r0 + bar_thickness, bar_c0 : bar_c0 + bar_len] = GREY

    colors = random.sample(PALETTE, 4 if extra_piece else 3)
    left_color, right_color, bottom_color = colors[:3]
    extra_color = colors[3] if extra_piece else None

    left_final = [(bar_r0 + bar_thickness + i, bar_c0) for i in range(side_h)]
    right_final = [(bar_r0 + bar_thickness + i, bar_c0 + bar_len - 1) for i in range(side_h)]
    bottom_final = [
        (bar_r0 + bar_thickness + side_h - 1, bar_c0 + 1 + i) for i in range(inner_w)
    ]
    extra_final = (
        [(bar_r0 - 1, bar_c0 + extra_offset + i) for i in range(extra_len)] if extra_piece else []
    )

    left_input = _translate(left_final, d_left, 0)
    right_input = _translate(right_final, d_right, 0)
    bottom_input = _translate(bottom_final, d_bottom, 0)
    extra_input = _translate(extra_final, -d_extra, 0) if extra_piece else []

    pieces = [
        (left_input, left_color),
        (right_input, right_color),
        (bottom_input, bottom_color),
    ]
    if extra_piece:
        pieces.append((extra_input, int(extra_color)))

    occupied: set[tuple[int, int]] = set(map(tuple, np.argwhere(grid != BLACK)))
    for coords, color in pieces:
        if not _in_bounds(coords, grid.shape):
            return None
        coord_set = set(coords)
        if coord_set & occupied:
            return None
        for r, c in coords:
            grid[r, c] = color
        occupied |= coord_set

    rotation = random.randint(0, 3)
    grid = _rotate_grid(grid, rotation)

    if not _validate_case(grid, require_fill_tie=tie_fill):
        return None
    return grid


def _validate_case(grid: np.ndarray, *, require_fill_tie: bool) -> bool:
    try:
        grey, pieces = _extract_objects(grid)
    except AssertionError:
        return False

    if len(pieces) not in (3, 4):
        return False
    colors = {piece["color"] for piece in pieces}
    if len(colors) != len(pieces) or not colors.issubset(set(PALETTE)):
        return False
    for piece in pieces:
        if not (2 <= len(piece["coords"]) <= 4):
            return False
        r0, c0, r1, c1 = piece["bbox"]
        grey_rows = [r for r, _ in grey]
        grey_cols = [c for _, c in grey]
        row_overlap = not (r1 < min(grey_rows) or r0 > max(grey_rows))
        col_overlap = not (c1 < min(grey_cols) or c0 > max(grey_cols))
        if not (row_overlap or col_overlap):
            return False

    if any(not region["touches_border"] for region in _background_regions(grid)):
        return False

    settled = _simulate(grid, accumulate=True, fill_regions=False)
    regions = [region for region in _background_regions(settled) if not region["touches_border"]]
    if not regions:
        return False

    if np.array_equal(_simulate(grid, accumulate=False), main(grid)):
        return False
    if np.array_equal(_simulate(grid, reverse_order=True), main(grid)):
        return False

    grey0, pieces0 = _extract_objects(grid)
    ordered = _sorted_pieces(grey0, pieces0)
    current_mass = set(grey0)
    found_tie = False
    for piece in ordered:
        steps = []
        for direction in DIRECTIONS:
            step = _landing_step(piece["coords"], current_mass, grid.shape, direction)
            if step is not None:
                steps.append((step, direction))
        if not steps:
            return False
        min_step = min(step for step, _ in steps)
        if sum(step == min_step for step, _ in steps) > 1:
            return False
        direction = min((d for step, d in steps if step == min_step))
        current_mass |= set(_translate(piece["coords"], direction[0] * min_step, direction[1] * min_step))

    color_rank = {
        int(piece["color"]): (int(piece["distance"]), int(piece["bbox"][0]), int(piece["bbox"][1]))
        for piece in ordered
    }
    for region in regions:
        counts: Counter[int] = Counter()
        for cell in region["boundary"]:
            color = int(settled[cell])
            if color != GREY:
                counts[color] += 1
        if not counts:
            return False
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        if len(winners) > 1:
            found_tie = True
        chosen = min(winners, key=lambda color: color_rank[color])
        out = main(grid)
        region_colors = {int(out[r, c]) for r, c in region["cells"]}
        if region_colors != {chosen}:
            return False

    return found_tie if require_fill_tie else True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-enclosure ARC input")
