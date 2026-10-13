from __future__ import annotations

import random

import numpy as np


GRID_SIZE = 14
X_SLICE = (slice(0, 5), slice(0, 5))
TILE_AREA_SLICE = (slice(5, 14), slice(5, 14))
MACRO_SIZE = 3
BG = 0

X_COORDS = (
    (0, 0),
    (0, 4),
    (1, 1),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 0),
    (4, 4),
)

FOLD_MAP = {
    (0, 0): (0, 0),
    (1, 1): (0, 1),
    (0, 4): (0, 2),
    (3, 1): (1, 0),
    (2, 2): (1, 1),
    (1, 3): (1, 2),
    (4, 0): (2, 0),
    (3, 3): (2, 1),
    (4, 4): (2, 2),
}

ANCHOR_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

CORNER_ANCHORS = {(0, 0), (0, 2), (2, 0), (2, 2)}
EDGE_ANCHORS = set(ANCHOR_POSITIONS) - CORNER_ANCHORS
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _rotate_cw(tile: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(tile, -int(turns) % 4)


def _fold_x_to_tile(x_patch: np.ndarray) -> np.ndarray:
    tile = np.zeros((3, 3), dtype=int)
    for src, dst in FOLD_MAP.items():
        tile[dst] = int(x_patch[src])
    return tile


def _macro_origin(mr: int, mc: int) -> tuple[int, int]:
    return 5 + mr * MACRO_SIZE, 5 + mc * MACRO_SIZE


def _anchors_in_macro(grid: np.ndarray, mr: int, mc: int) -> list[tuple[int, int, int]]:
    r0, c0 = _macro_origin(mr, mc)
    anchors = []
    for dr in range(MACRO_SIZE):
        for dc in range(MACRO_SIZE):
            value = int(grid[r0 + dr, c0 + dc])
            if value != BG:
                anchors.append((dr, dc, value))
    return anchors


def _matching_rotations(tile: np.ndarray, anchors: list[tuple[int, int, int]]) -> list[int]:
    matches = []
    for turns in range(4):
        rotated = _rotate_cw(tile, turns)
        if all(int(rotated[dr, dc]) == color for dr, dc, color in anchors):
            matches.append(turns)
    return matches


def _build_output_from_tile(tile: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for mr in range(3):
        for mc in range(3):
            r0, c0 = _macro_origin(mr, mc)
            out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_cw(tile, int(rotations[mr, mc]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the 5x5 X motif into a 3x3 tile, then tile the lower-right area using per-block anchor-constrained rotations.

    Input structure:
    - A 14x14 black background contains a 5x5 X-shaped colored motif in the upper-left corner.
    - The X uses exactly the nine diagonal positions of that 5x5 patch; the other cells there are black.
    - The lower-right 9x9 area is divided into a 3x3 macro-grid of 3x3 blocks and contains non-black anchor pixels inside those blocks.

    Rule:
    - Fold the X into a 3x3 tile with the fixed diagonal-to-edge mapping described in the task statement.
    - For each 3x3 macro block in the lower-right area, test the four rotations of that tile and keep the unique rotation that matches every anchor pixel already present in that block.
    - Paste the selected rotated tile into that macro block, erase the original 5x5 X motif to black, and leave all pixels outside the 9x9 tiling area black.

    Key insight:
    - The X motif determines the tile's contents, while the scattered anchors determine each block's orientation; neither piece alone is sufficient to reconstruct the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    x_patch = g[X_SLICE]
    tile = _fold_x_to_tile(x_patch)

    out = np.zeros_like(g)
    for mr in range(3):
        for mc in range(3):
            anchors = _anchors_in_macro(g, mr, mc)
            assert anchors, "each macro block must contain at least one anchor"
            rotations = _matching_rotations(tile, anchors)
            assert len(rotations) == 1, "anchors must determine a unique rotation"
            r0, c0 = _macro_origin(mr, mc)
            out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_cw(tile, rotations[0])
    return out


def _sample_rotation_grid() -> np.ndarray:
    values = [0, 1, 2, 3]
    values.extend(random.choices(range(4), k=5))
    random.shuffle(values)
    rotations = np.array(values, dtype=int).reshape(3, 3)
    if len(set(map(int, rotations.flat))) < 4:
        return _sample_rotation_grid()
    return rotations


def _sample_anchor_layout() -> np.ndarray:
    while True:
        chosen = np.array(random.choices(ANCHOR_POSITIONS, k=9), dtype=int).reshape(3, 3, 2)
        unique_positions = {tuple(map(int, pos)) for pos in chosen.reshape(-1, 2)}
        if len(unique_positions) < 4:
            continue
        if not any(tuple(map(int, pos)) in CORNER_ANCHORS for pos in chosen.reshape(-1, 2)):
            continue
        if not any(tuple(map(int, pos)) in EDGE_ANCHORS for pos in chosen.reshape(-1, 2)):
            continue
        return chosen


def _make_x_patch() -> np.ndarray:
    colors = random.sample(ALL_COLORS, 9)
    patch = np.zeros((5, 5), dtype=int)
    for color, coord in zip(colors, X_COORDS):
        patch[coord] = int(color)
    return patch


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False

    x_patch = grid[X_SLICE]
    if any(int(x_patch[r, c]) == BG for r, c in X_COORDS):
        return False
    for r in range(5):
        for c in range(5):
            if (r, c) not in X_COORDS and int(x_patch[r, c]) != BG:
                return False

    x_colors = [int(x_patch[r, c]) for r, c in X_COORDS]
    if sorted(x_colors) != list(ALL_COLORS):
        return False

    tile = _fold_x_to_tile(x_patch)
    if len(set(map(int, tile.flat))) != 9:
        return False

    rotations = np.full((3, 3), -1, dtype=int)
    anchor_positions = set()
    for mr in range(3):
        for mc in range(3):
            anchors = _anchors_in_macro(grid, mr, mc)
            if not anchors:
                return False
            if any((dr, dc) == (1, 1) for dr, dc, _ in anchors):
                return False
            for dr, dc, _ in anchors:
                anchor_positions.add((dr, dc))
            matches = _matching_rotations(tile, anchors)
            if len(matches) != 1:
                return False
            rotations[mr, mc] = matches[0]

    if len(set(map(int, rotations.flat))) < 4:
        return False
    if len(anchor_positions) < 4:
        return False
    if not (anchor_positions & CORNER_ANCHORS):
        return False
    if not (anchor_positions & EDGE_ANCHORS):
        return False

    expected = _build_output_from_tile(tile, rotations)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_one() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    x_patch = _make_x_patch()
    tile = _fold_x_to_tile(x_patch)
    rotations = _sample_rotation_grid()
    anchor_layout = _sample_anchor_layout()

    grid[X_SLICE] = x_patch
    for mr in range(3):
        for mc in range(3):
            r0, c0 = _macro_origin(mr, mc)
            dr, dc = map(int, anchor_layout[mr, mc])
            rotated = _rotate_cw(tile, int(rotations[mr, mc]))
            grid[r0 + dr, c0 + dc] = int(rotated[dr, dc])

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
