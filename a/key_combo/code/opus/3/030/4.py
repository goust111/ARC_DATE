from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

try:
    from new.solvers.arc_utils import Color, components_mask, in_bounds, paste
except Exception:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fallback", _ARC_UTILS_PATH)
    assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
    _ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
    _ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
    Color = _ARC_UTILS.Color
    components_mask = _ARC_UTILS.components_mask
    in_bounds = _ARC_UTILS.in_bounds
    paste = _ARC_UTILS.paste


GRID_DIM_RANGE = (11, 19)
RADIUS_OPTIONS = (3, 4)
BLUE_LEN_RANGE = (8, 15)
GEN_ATTEMPTS = 300


def _rotate_about(center: tuple[int, int], r: int, c: int, quarter_turns: int) -> tuple[int, int]:
    cr, cc = center
    dr, dc = int(r) - int(cr), int(c) - int(cc)
    for _ in range(int(quarter_turns) % 4):
        dr, dc = dc, -dr
    return int(cr + dr), int(cc + dc)


def _rotate_offset(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    rr, cc = int(dr), int(dc)
    for _ in range(int(quarter_turns) % 4):
        rr, cc = cc, -rr
    return int(rr), int(cc)


def _gray_center(grid: np.ndarray) -> tuple[int, int]:
    grey = grid == Color.GREY
    coords = np.argwhere(grey)
    assert coords.size > 0, "expected a grey guide cross"

    centers = []
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if all(
            in_bounds(grid.shape, r + dr, c + dc) and int(grid[r + dr, c + dc]) == Color.GREY
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            centers.append((r, c))
    assert len(centers) == 1, "expected one grey cross intersection"
    return centers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the gray cross as the rotation center and color the missing 90/180/270 counterparts of the blue shape.

    Input structure:
    - The grid has black background, one horizontal gray bar, and one vertical gray bar crossing at a single center cell.
    - Blue pixels form one irregular 4-connected foreground object near the guide-cross center.
    - The blue object does not overlap the gray guide bars, and the cells to be completed are black.

    Rule:
    - Find the unique intersection cell of the two gray guide bars and use it as the rotation center.
    - For every blue pixel, rotate it 90, 180, and 270 degrees clockwise around that center.
    - When a rotated destination is black in the input, color it red for 90 degrees, green for 180 degrees, or yellow for 270 degrees.
    - If several rotations want the same black cell, keep the smallest clockwise angle: red over green over yellow.

    Key insight:
    - The gray bars are not decorative separators; they pin down the only valid rotational center, and the completion color records how far the source blue pixel had to rotate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center = _gray_center(g)
    out = g.copy()
    blue_coords = np.argwhere(g == Color.BLUE)
    priorities = ((1, Color.RED), (2, Color.GREEN), (3, Color.YELLOW))

    planned: dict[tuple[int, int], int] = {}
    for rr, cc in blue_coords:
        r = int(rr)
        c = int(cc)
        for turns, color in priorities:
            nr, nc = _rotate_about(center, r, c, turns)
            if not in_bounds(g.shape, nr, nc):
                continue
            if int(g[nr, nc]) != Color.BLACK:
                continue
            key = (nr, nc)
            if key not in planned or int(color) < planned[key]:
                planned[key] = int(color)

    for (r, c), color in planned.items():
        out[r, c] = int(color)
    return out


def _square_cycle(radius: int) -> list[tuple[int, int]]:
    r = int(radius)
    cycle: list[tuple[int, int]] = []
    for c in range(-r, r + 1):
        cycle.append((r, c))
    for rr in range(r - 1, -r - 1, -1):
        cycle.append((rr, r))
    for c in range(r - 1, -r - 1, -1):
        cycle.append((-r, c))
    for rr in range(-r + 1, r):
        cycle.append((rr, -r))
    assert len(cycle) == 8 * r
    assert len(set(cycle)) == len(cycle)
    return cycle


def _completion_colors(offsets: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    for dr, dc in offsets:
        for turns, color in ((1, Color.RED), (2, Color.GREEN), (3, Color.YELLOW)):
            target = _rotate_offset(dr, dc, turns)
            if target in offsets:
                continue
            if target not in colors or int(color) < colors[target]:
                colors[target] = int(color)
    return colors


def _good_arc_specs(radius: int) -> list[tuple[int, int]]:
    cycle = _square_cycle(radius)
    good: list[tuple[int, int]] = []
    for length in range(BLUE_LEN_RANGE[0], min(BLUE_LEN_RANGE[1], len(cycle) - 1) + 1):
        for start in range(len(cycle)):
            offsets = {cycle[(start + i) % len(cycle)] for i in range(length)}
            colors = _completion_colors(offsets)
            if set(colors) != set(cycle) - offsets:
                continue
            color_values = set(colors.values())
            if color_values == {Color.RED, Color.GREEN, Color.YELLOW}:
                good.append((start, length))
    assert good, f"expected at least one valid arc for radius={radius}"
    return good


GOOD_ARCS = {radius: _good_arc_specs(radius) for radius in RADIUS_OPTIONS}


def _build_patch(radius: int, arm: int, offsets: set[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=int)
    mid = radius
    patch[mid, mid - arm : mid + arm + 1] = Color.GREY
    patch[mid - arm : mid + arm + 1, mid] = Color.GREY
    for dr, dc in offsets:
        patch[mid + dr, mid + dc] = Color.BLUE
    return patch


def _expected_from_template(
    grid: np.ndarray,
    center: tuple[int, int],
    offsets: set[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    out = grid.copy()
    colors = _completion_colors(offsets)
    cycle = set(_square_cycle(radius))
    assert set(colors) == cycle - offsets
    for (dr, dc), color in colors.items():
        rr = int(center[0] + dr)
        cc = int(center[1] + dc)
        out[rr, cc] = int(color)
    return out


def _touches_grey(grid: np.ndarray) -> bool:
    blue_coords = np.argwhere(grid == Color.BLUE)
    for rr, cc in blue_coords:
        r = int(rr)
        c = int(cc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.GREY:
                return True
    return False


def _validate_generated(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    offsets: set[tuple[int, int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _gray_center(grid) != center:
        return False
    if int(np.count_nonzero(grid == Color.BLUE)) != len(offsets):
        return False
    if not (BLUE_LEN_RANGE[0] <= len(offsets) <= BLUE_LEN_RANGE[1]):
        return False
    if _touches_grey(grid):
        return False

    blue_components = components_mask(grid == Color.BLUE, connectivity=4)
    if len(blue_components) != 1:
        return False

    expected = _expected_from_template(grid, center, offsets, radius)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    colors_present = {int(v) for v in actual.flatten()}
    if not {Color.RED, Color.GREEN, Color.YELLOW}.issubset(colors_present):
        return False

    return True


def _sample_grid_shape(radius: int) -> tuple[int, int]:
    min_dim = max(GRID_DIM_RANGE[0], 2 * radius + 3)
    max_dim = GRID_DIM_RANGE[1]
    return random.randint(min_dim, max_dim), random.randint(min_dim, max_dim)


def _gen_attempt() -> np.ndarray | None:
    radius = random.choice(RADIUS_OPTIONS)
    arm = radius - 2
    start, length = random.choice(GOOD_ARCS[radius])
    cycle = _square_cycle(radius)
    offsets = {cycle[(start + i) % len(cycle)] for i in range(length)}

    height, width = _sample_grid_shape(radius)
    center = (
        random.randint(radius, height - radius - 1),
        random.randint(radius, width - radius - 1),
    )

    grid = np.zeros((height, width), dtype=int)
    patch = _build_patch(radius, arm, offsets)
    paste(grid, patch, center[0] - radius, center[1] - radius)

    if _validate_generated(grid, center, radius, offsets):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
