import random
from typing import Iterable

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
DIR_ORDER = ("up", "down", "left", "right")
DIR_INFO = {
    "up": {"missing": (0, 1), "delta": (-1, 0)},
    "down": {"missing": (2, 1), "delta": (1, 0)},
    "left": {"missing": (1, 0), "delta": (0, -1)},
    "right": {"missing": (1, 2), "delta": (0, 1)},
}
RING_CELLS = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
GEN_RETRIES = 300
PLACEMENT_TRIES = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _u_cells(direction: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    missing = DIR_INFO[direction]["missing"]
    filled = sorted(RING_CELLS - {missing})
    return filled, missing


def _find_u_shape(grid: np.ndarray, color: int, bg: int) -> tuple[tuple[int, int], str, tuple[int, int]]:
    rows, cols = grid.shape
    for r0 in range(rows - 2):
        for c0 in range(cols - 2):
            sub = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(sub[1, 1]) != bg:
                continue
            if np.any((sub != bg) & (sub != color)):
                continue
            color_mask = sub == color
            if int(color_mask.sum()) != 7:
                continue
            for direction in DIR_ORDER:
                filled, missing = _u_cells(direction)
                if all(bool(color_mask[rr, cc]) for rr, cc in filled) and not bool(color_mask[missing]):
                    gap = (r0 + missing[0], c0 + missing[1])
                    return (r0, c0), direction, gap
    raise AssertionError(f"no U-shape found for color {color}")


def _find_plug(grid: np.ndarray, color: int, gap: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIR_INFO[direction]["delta"]
    rows, cols = grid.shape
    rr = gap[0] + dr
    cc = gap[1] + dc
    while 0 <= rr < rows and 0 <= cc < cols:
        if int(grid[rr, cc]) == color:
            return rr, cc
        rr += dr
        cc += dc
    raise AssertionError(f"no plug found for color {color}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-colored single-pixel plug straight into its matching U-shaped gap.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly one pair: a 3x3 U-shaped border and one same-colored single pixel.
    - The U-shape is a 3x3 ring missing exactly one side-center cell, and the plug lies on the outward ray of that missing cell.

    Rule:
    - For each color, detect which side-center is missing from the 3x3 border to determine the opening direction.
    - Find the same-colored single pixel on the extension line outside that opening.
    - Remove the plug from its original location and place it into the missing side-center cell, turning the U-shape into a full 3x3 closed border.

    Key insight:
    - The motion direction is encoded by the geometry of the U-shape itself: the missing side-center tells you both where the plug belongs and along which line to retrieve it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        _origin, direction, gap = _find_u_shape(g, color, bg)
        plug = _find_plug(g, color, gap, direction)
        out[plug] = bg
        out[gap] = color

    return out


def _balanced_directions(count: int) -> list[str]:
    dirs: list[str] = []
    while len(dirs) < count:
        shuffled = list(DIR_ORDER)
        random.shuffle(shuffled)
        need = count - len(dirs)
        dirs.extend(shuffled[:need])
    return dirs


def _core_bbox(r0: int, c0: int, direction: str, distance: int) -> tuple[int, int, int, int]:
    if direction == "up":
        return r0 - distance, c0, r0 + 2, c0 + 2
    if direction == "down":
        return r0, c0, r0 + 2 + distance, c0 + 2
    if direction == "left":
        return r0, c0 - distance, r0 + 2, c0 + 2
    return r0, c0, r0 + 2, c0 + 2 + distance


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    rows: int,
    cols: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(rows - 1, r1 + pad), min(cols - 1, c1 + pad)


def _plug_coord(r0: int, c0: int, direction: str, distance: int) -> tuple[int, int]:
    missing = DIR_INFO[direction]["missing"]
    dr, dc = DIR_INFO[direction]["delta"]
    return r0 + missing[0] + dr * distance, c0 + missing[1] + dc * distance


def _place_group(
    grid: np.ndarray,
    out: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    distance: int,
) -> bool:
    rows, cols = grid.shape
    if direction == "up":
        r_min, r_max = distance, rows - 3
        c_min, c_max = 0, cols - 3
    elif direction == "down":
        r_min, r_max = 0, rows - 3 - distance
        c_min, c_max = 0, cols - 3
    elif direction == "left":
        r_min, r_max = 0, rows - 3
        c_min, c_max = distance, cols - 3
    else:
        r_min, r_max = 0, rows - 3
        c_min, c_max = 0, cols - 3 - distance

    if r_min > r_max or c_min > c_max:
        return False

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        bbox = _core_bbox(r0, c0, direction, distance)
        pr0, pc0, pr1, pc1 = _pad_bbox(bbox, rows, cols, pad=1)
        if bool(blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any()):
            continue

        filled, missing = _u_cells(direction)
        for dr, dc in filled:
            grid[r0 + dr, c0 + dc] = color
            out[r0 + dr, c0 + dc] = color

        gap = (r0 + missing[0], c0 + missing[1])
        plug = _plug_coord(r0, c0, direction, distance)
        grid[plug] = color
        out[gap] = color
        blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return True

    return False


def _expected_output_from_specs(
    rows: int,
    cols: int,
    specs: Iterable[tuple[int, str, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    ordered_specs = sorted(specs, key=lambda item: item[2], reverse=True)
    for color, direction, distance in ordered_specs:
        if not _place_group(grid, out, blocked, color, direction, distance):
            return None
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(1, 5)
    colors = random.sample(ALL_COLORS, group_count)
    directions = _balanced_directions(group_count)
    distances = [random.randint(1, 6) for _ in range(group_count)]

    min_side = max(9, 5 + max(distances), 5 + 3 * ((group_count + 1) // 2))
    if min_side > 30:
        return None

    rows = random.randint(min_side, 30)
    cols = random.randint(min_side, 30)
    specs = list(zip(colors, directions, distances))
    built = _expected_output_from_specs(rows, cols, specs)
    if built is None:
        return None

    grid, expected = built
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
