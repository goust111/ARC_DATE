import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
ARM_COLORS = (2, 3, 7, 8, 9)

OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OFFSETS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in OFFSETS4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in OFFSETS8:
        yield r + dr, c + dc


def components_of_color(grid: np.ndarray, color: int, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = OFFSETS4 if connectivity == 4 else OFFSETS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def detect_crosses(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    crosses: list[dict] = []
    for r, c in np.argwhere(grid == YELLOW):
        r = int(r)
        c = int(c)
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        arms = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        arm_colors = [int(grid[rr, cc]) for rr, cc in arms]
        if len(set(arm_colors)) != 1:
            continue
        arm_color = arm_colors[0]
        if arm_color in (BLACK, BLUE, YELLOW):
            continue
        crosses.append(
            {
                "center": (r, c),
                "arm_color": arm_color,
                "cells": {(r, c), *arms},
            }
        )
    return crosses


def detect_defects(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    defects: list[list[tuple[int, int]]] = []
    for comp in components_of_color(grid, BLACK, connectivity=4):
        if len(comp) > 2:
            continue
        comp_set = set(comp)
        enclosed = True
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if not in_bounds(grid.shape, nr, nc):
                    enclosed = False
                    break
                if (nr, nc) in comp_set:
                    continue
                if int(grid[nr, nc]) != BLUE:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            defects.append(comp)
    return defects


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the repeated arm color of yellow-centered plus markers to halo enclosed black defects, then erase the markers.

    Input structure:
    - The grid contains a mostly blue foreground with black border/background regions.
    - Inside the blue region are several yellow-centered plus markers whose four orthogonal arm cells share one non-blue, non-yellow arm color.
    - The blue foreground also contains small enclosed black hole defects of area one or two.

    Rule:
    - Detect every valid yellow-centered plus marker and count its arm color.
    - The unique arm color appearing on more than one marker becomes the halo color.
    - Find every black 4-connected component of area at most two whose orthogonal boundary is entirely blue.
    - Recolor each currently blue cell in the 8-neighborhood of every defect pixel to the halo color, keep defect pixels black, then repaint all marker cells blue.

    Key insight:
    - The plus markers are not output objects: they form a color vote whose repeated arm color is transferred to a different structure, namely the halo drawn around enclosed defects.
    """
    grid = np.asarray(input_grid, dtype=int)
    crosses = detect_crosses(grid)
    color_counts = Counter(cross["arm_color"] for cross in crosses)
    repeated = [color for color, count in color_counts.items() if count > 1]
    assert len(repeated) == 1, "expected exactly one repeated arm color"
    halo_color = repeated[0]

    defects = detect_defects(grid)
    out = grid.copy()
    for comp in defects:
        for r, c in comp:
            for nr, nc in neighbors8(r, c):
                if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BLUE:
                    out[nr, nc] = halo_color

    for cross in crosses:
        for r, c in cross["cells"]:
            out[r, c] = BLUE
    return out


def _paint_notch(grid: np.ndarray, side: str, rng: random.Random, bounds: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bounds
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if side in ("top", "bottom"):
        notch_w = rng.randint(1, max(1, min(4, width // 3)))
        notch_d = rng.randint(1, max(1, min(4, max(1, height // 4))))
        start_c = rng.randint(c0 + 1, c1 - notch_w)
        if side == "top":
            grid[r0 : r0 + notch_d, start_c : start_c + notch_w] = BLACK
        else:
            grid[r1 - notch_d + 1 : r1 + 1, start_c : start_c + notch_w] = BLACK
    else:
        notch_h = rng.randint(1, max(1, min(4, height // 3)))
        notch_d = rng.randint(1, max(1, min(4, max(1, width // 4))))
        start_r = rng.randint(r0 + 1, r1 - notch_h)
        if side == "left":
            grid[start_r : start_r + notch_h, c0 : c0 + notch_d] = BLACK
        else:
            grid[start_r : start_r + notch_h, c1 - notch_d + 1 : c1 + 1] = BLACK


def _fits_all_blue(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLUE for r, c in cells)


def _square_all_blue(grid: np.ndarray, r: int, c: int, radius: int) -> bool:
    for rr in range(r - radius, r + radius + 1):
        for cc in range(c - radius, c + radius + 1):
            if not in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BLUE:
                return False
    return True


def _cross_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _can_place_cross(grid: np.ndarray, center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    r, c = center
    if not _square_all_blue(grid, r, c, radius=2):
        return False
    for rr, cc in centers:
        if max(abs(r - rr), abs(c - cc)) < 5:
            return False
    return True


def _place_cross(grid: np.ndarray, center: tuple[int, int], color: int) -> set[tuple[int, int]]:
    cells = set(_cross_cells(center))
    for r, c in cells:
        grid[r, c] = color if (r, c) != center else YELLOW
    return cells


def _defect_shapes(rng: random.Random) -> list[list[tuple[int, int]]]:
    shapes = [[(0, 0)], [(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    rng.shuffle(shapes)
    return shapes


def _can_place_defect(
    grid: np.ndarray,
    anchor: tuple[int, int],
    shape: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    existing: list[list[tuple[int, int]]],
) -> bool:
    cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape]
    if not _fits_all_blue(grid, cells):
        return False
    comp_set = set(cells)
    for r, c in cells:
        if (r, c) in blocked:
            return False
        for other in existing:
            for rr, cc in other:
                if max(abs(r - rr), abs(c - cc)) < 4:
                    return False
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                return False
            if (nr, nc) in comp_set:
                continue
            if int(grid[nr, nc]) != BLUE:
                return False
    return True


def _paint_defect(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = BLACK


def _expected_halo(
    grid: np.ndarray,
    defects: list[list[tuple[int, int]]],
    halo_color: int,
    cross_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for comp in defects:
        for r, c in comp:
            for nr, nc in neighbors8(r, c):
                if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BLUE:
                    out[nr, nc] = halo_color
    for r, c in cross_cells:
        out[r, c] = BLUE
    return out


def _validate_constructed_grid(
    grid: np.ndarray,
    repeated_color: int,
    expected_cross_count: int,
    expected_cross_cells: set[tuple[int, int]],
    expected_defects: list[list[tuple[int, int]]],
) -> bool:
    parsed_crosses = detect_crosses(grid)
    if len(parsed_crosses) != expected_cross_count:
        return False
    parsed_cells = set()
    counts = Counter()
    for cross in parsed_crosses:
        parsed_cells |= cross["cells"]
        counts[cross["arm_color"]] += 1
    if parsed_cells != expected_cross_cells:
        return False
    repeated = [color for color, count in counts.items() if count > 1]
    if repeated != [repeated_color]:
        return False

    parsed_defects = {frozenset(comp) for comp in detect_defects(grid)}
    expected_defect_sets = {frozenset(comp) for comp in expected_defects}
    if parsed_defects != expected_defect_sets:
        return False

    if not all(len(comp) in (1, 2) for comp in expected_defects):
        return False
    return True


def _validate_output(
    grid: np.ndarray,
    out: np.ndarray,
    repeated_color: int,
    cross_cells: set[tuple[int, int]],
    defects: list[list[tuple[int, int]]],
) -> bool:
    expected = _expected_halo(grid, defects, repeated_color, cross_cells)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False
    if not np.any(out == repeated_color):
        return False
    for r, c in cross_cells:
        if int(out[r, c]) != BLUE:
            return False
    for comp in defects:
        for r, c in comp:
            if int(out[r, c]) != BLACK:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 22)
    w = rng.randint(15, 22)
    grid = np.full((h, w), BLACK, dtype=int)

    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    right = rng.randint(1, 3)
    if h - top - bottom < 11 or w - left - right < 11:
        return None

    r0, c0 = top, left
    r1, c1 = h - bottom - 1, w - right - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    notch_sides = rng.sample(["top", "bottom", "left", "right"], rng.randint(1, 3))
    for side in notch_sides:
        _paint_notch(grid, side, rng, (r0, c0, r1, c1))

    num_crosses = rng.randint(3, 5)
    repeat_color = rng.choice(ARM_COLORS)
    max_repeat = min(3, num_crosses - 1)
    repeat_count = 2 if max_repeat == 2 else rng.choice([2, 3])
    other_colors = rng.sample([c for c in ARM_COLORS if c != repeat_color], num_crosses - repeat_count)
    cross_colors = [repeat_color] * repeat_count + other_colors
    rng.shuffle(cross_colors)

    centers: list[tuple[int, int]] = []
    cross_cells: set[tuple[int, int]] = set()
    candidates = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
    rng.shuffle(candidates)
    for color in cross_colors:
        placed = False
        for center in candidates:
            if not _can_place_cross(grid, center, centers):
                continue
            cells = set(_cross_cells(center))
            if any(cell in cross_cells for cell in cells):
                continue
            centers.append(center)
            cross_cells |= _place_cross(grid, center, color)
            placed = True
            break
        if not placed:
            return None

    blocked = set()
    for r, c in cross_cells:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                blocked.add((r + dr, c + dc))

    num_defects = rng.randint(2, 4)
    defects: list[list[tuple[int, int]]] = []
    anchors = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    rng.shuffle(anchors)
    for idx in range(num_defects):
        placed = False
        shapes = _defect_shapes(rng)
        if idx == 0 and rng.random() < 0.6:
            shapes = [shape for shape in shapes if len(shape) == 2] + [shape for shape in shapes if len(shape) == 1]
        for anchor in anchors:
            for shape in shapes:
                if not _can_place_defect(grid, anchor, shape, blocked, defects):
                    continue
                cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape]
                _paint_defect(grid, cells)
                defects.append(cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if not _validate_constructed_grid(grid, repeat_color, num_crosses, cross_cells, defects):
        return None
    out = main(grid)
    if not _validate_output(grid, out, repeat_color, cross_cells, defects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
