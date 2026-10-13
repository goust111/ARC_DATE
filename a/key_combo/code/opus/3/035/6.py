from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_035_6", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask


GEN_SIDE_MIN = 5
GEN_SIDE_MAX = 7
GEN_MAX_ATTEMPTS = 200

GEN_SPLIT_TEMPLATES = {
    5: np.array(
        [
            [0, 0, 0, 0, 0],
            [5, 0, 5, 5, 5],
            [5, 5, 0, 0, 0],
            [5, 0, 0, 5, 5],
            [5, 0, 0, 5, 0],
        ],
        dtype=int,
    ),
    6: np.array(
        [
            [0, 5, 5, 5, 5, 5],
            [0, 5, 0, 0, 0, 0],
            [0, 5, 5, 5, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [0, 5, 0, 5, 5, 5],
            [0, 5, 0, 0, 5, 0],
        ],
        dtype=int,
    ),
    7: np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 5, 5],
            [0, 5, 0, 0, 0, 5, 0],
            [0, 5, 0, 0, 0, 5, 0],
            [0, 5, 5, 5, 5, 5, 5],
            [0, 5, 0, 0, 0, 5, 0],
        ],
        dtype=int,
    ),
}


def _pinwheel(grid: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [grid, np.rot90(grid, -1)],
            [np.rot90(grid, 1), np.rot90(grid, 2)],
        ]
    )


def _walkable_components(grid: np.ndarray) -> list[np.ndarray]:
    return components_mask(grid != Color.GREY, connectivity=4)


def _build_merge_template(side: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    grid[:, 1] = Color.GREY
    mid = side // 2
    grid[mid:, side - 2] = Color.GREY
    grid[min(mid + 1, side - 2), side - 2 :] = Color.GREY
    return grid


def _apply_d4(grid: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(grid, int(code) % 4)
    if int(code) >= 4:
        out = np.fliplr(out)
    return out.copy()


def _is_thin_wall_pattern(grid: np.ndarray) -> bool:
    walls = grid == Color.GREY
    if np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:]):
        return False

    side = grid.shape[0]
    for r, c in np.argwhere(walls):
        if (
            (c > 0 and walls[r, c - 1])
            or (c + 1 < side and walls[r, c + 1])
            or (r > 0 and walls[r - 1, c])
            or (r + 1 < side and walls[r + 1, c])
        ):
            continue
        return False
    return True


def _example_stats(grid: np.ndarray) -> tuple[int, int, list[int]]:
    source_components = _walkable_components(grid)
    output_components = _walkable_components(_pinwheel(grid))
    areas = sorted(int(component.shape[0]) for component in output_components)
    return len(source_components), len(output_components), areas


def _valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        return False
    if not set(map(int, np.unique(grid))) <= {Color.BLACK, Color.BLUE, Color.GREY}:
        return False
    if np.count_nonzero(grid == Color.BLUE) != 1:
        return False

    base = grid.copy()
    base[base == Color.BLUE] = Color.BLACK
    if not _is_thin_wall_pattern(base):
        return False

    source_count, output_count, areas = _example_stats(grid)
    if not (3 <= source_count <= 5):
        return False
    if output_count == source_count or len(areas) < 2:
        return False
    return areas[0] < areas[1] and areas[-2] < areas[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the tile into a four-rotation pinwheel and recolor the largest and smallest emergent open regions.

    Input structure:
    - A square black tile contains thin gray wall segments that partition the open cells into several 4-connected regions.
    - Exactly one blue marker pixel lies on an open cell; it only marks orientation and counts as open space for connectivity.
    - All other non-wall cells are black.

    Rule:
    - Build a 2x2 pinwheel with the input unchanged at upper-left, its 90-degree clockwise rotation at upper-right, its 90-degree counterclockwise rotation at lower-left, and its 180-degree rotation at lower-right.
    - In that 2N x 2N canvas, treat gray as walls and every black or blue cell as open space, then find all 4-connected open regions.
    - Recolor the unique largest open region teal and the unique smallest open region orange, then restore every blue marker cell to blue.

    Key insight:
    - The decisive regions are emergent: only after the rotated quadrants meet do seam-aligned wall ends either continue a barrier or leave openings that merge space across quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    side, width = map(int, g.shape)
    assert side == width, "pinwheel assembly requires a square tile"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.GREY}, "unexpected input colors"

    pinwheel = _pinwheel(g)
    components = _walkable_components(pinwheel)
    assert len(components) >= 2, "expected at least two open regions after assembly"

    areas = np.array([int(component.shape[0]) for component in components], dtype=int)
    largest_idx = int(np.argmax(areas))
    smallest_idx = int(np.argmin(areas))
    assert np.count_nonzero(areas == areas[largest_idx]) == 1, "largest region must be unique"
    assert np.count_nonzero(areas == areas[smallest_idx]) == 1, "smallest region must be unique"

    out = pinwheel.copy()
    for region_idx, color in ((largest_idx, Color.TEAL), (smallest_idx, Color.ORANGE)):
        coords = components[region_idx]
        out[coords[:, 0], coords[:, 1]] = int(color)
    out[pinwheel == Color.BLUE] = Color.BLUE
    return out


# --------------- generator (attempt_1) ---------------


def _build_wall_template(side: int, branch: str) -> np.ndarray:
    if branch == "merge":
        return _build_merge_template(side)
    return GEN_SPLIT_TEMPLATES[side].copy()


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        side = int(np.random.randint(GEN_SIDE_MIN, GEN_SIDE_MAX + 1))
        branch = str(np.random.choice(np.array(["merge", "split"])))
        wall_grid = _build_wall_template(side, branch)
        wall_grid = _apply_d4(wall_grid, int(np.random.randint(8)))

        open_components = _walkable_components(wall_grid)
        marker_component = open_components[int(np.random.randint(len(open_components)))]
        marker_row, marker_col = map(int, marker_component[int(np.random.randint(len(marker_component)))])

        grid = wall_grid.copy()
        grid[marker_row, marker_col] = Color.BLUE
        if _valid_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid pinwheel-wall input")
