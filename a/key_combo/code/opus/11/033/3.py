from __future__ import annotations

import random

import numpy as np


MARKER_COLOR = 4
ERASE_COLOR = 0
GRID_SHAPE = (3, 3)
NON_MARKER_COLORS = (1, 2, 3, 6, 7, 8, 9)
NON_CENTER_POSITIONS = tuple(
    (r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)
)
TRANSFORM_BRANCHES = ((1, 1), (0, 1), (1, 0), (0, 0))
MAX_GENERATION_ATTEMPTS = 800


def _array_sig(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return arr.shape, arr.tobytes()


def _apply_slot_transform(tile: np.ndarray, macro_r: int, macro_c: int) -> np.ndarray:
    out = np.asarray(tile, dtype=int)
    if macro_r != 1:
        out = np.flipud(out)
    if macro_c != 1:
        out = np.fliplr(out)
    return out


def _transformed_marker_pos(marker_r: int, marker_c: int, macro_r: int, macro_c: int) -> tuple[int, int]:
    out_r = 2 - marker_r if macro_r != 1 else marker_r
    out_c = 2 - marker_c if macro_c != 1 else marker_c
    return int(out_r), int(out_c)


def _branch_output(tile: np.ndarray, marker_pos: tuple[int, int], macro_pos: tuple[int, int]) -> np.ndarray:
    marker_r, marker_c = marker_pos
    macro_r, macro_c = macro_pos
    out = _apply_slot_transform(tile, macro_r, macro_c).copy()
    erase_r, erase_c = _transformed_marker_pos(marker_r, marker_c, macro_r, macro_c)
    out[erase_r, erase_c] = ERASE_COLOR
    return out


def _marker_position(tile: np.ndarray) -> tuple[int, int]:
    markers = np.argwhere(np.asarray(tile, dtype=int) == MARKER_COLOR)
    assert markers.shape == (1, 2), "expected exactly one yellow marker"
    return int(markers[0, 0]), int(markers[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the mirror-mosaic slot encoded by the yellow marker and erase the marker from that slot.

    Input structure:
    - The input is a single 3x3 tile with every cell filled.
    - Exactly one yellow cell acts as a marker pixel embedded in the tile.
    - The marker's local row and column encode one slot of a conceptual 3x3 mirror mosaic.

    Rule:
    - Conceptually place the input tile in the center of a 3x3 mosaic.
    - Slots above or below the center use the row-flipped tile; slots left or right use the column-flipped tile; corners use both flips.
    - Read the yellow marker's local coordinate `(r, c)` and select mosaic slot `(r, c)`.
    - Output only that 3x3 slot, but replace the transformed yellow marker position inside it with black.

    Key insight:
    - The yellow pixel is both part of the mirrored tile content and a coordinate code that names which mirrored slot to extract.
    """
    tile = np.asarray(grid, dtype=int)
    assert tile.ndim == 2, "expected a 2D grid"
    assert tile.shape == GRID_SHAPE, "expected a 3x3 tile"

    marker_pos = _marker_position(tile)
    return _branch_output(tile, marker_pos, marker_pos)


def _sample_colored_cells(marker_pos: tuple[int, int]) -> np.ndarray:
    palette_size = random.randint(2, 3)
    palette = random.sample(list(NON_MARKER_COLORS), palette_size)

    tile = np.empty(GRID_SHAPE, dtype=int)
    tile[marker_pos] = MARKER_COLOR

    other_positions = [pos for pos in NON_CENTER_POSITIONS + ((1, 1),) if pos != marker_pos]
    colors = list(palette)
    colors.extend(random.choices(palette, k=len(other_positions) - len(palette)))
    random.shuffle(colors)

    for (r, c), color in zip(other_positions, colors):
        tile[r, c] = int(color)
    return tile


def _is_valid_generated_input(tile: np.ndarray) -> bool:
    arr = np.asarray(tile, dtype=int)
    if arr.shape != GRID_SHAPE or np.any(arr == ERASE_COLOR):
        return False

    markers = np.argwhere(arr == MARKER_COLOR)
    if markers.shape != (1, 2):
        return False
    marker_pos = int(markers[0, 0]), int(markers[0, 1])
    if marker_pos == (1, 1):
        return False

    other_colors = {int(v) for v in arr.flat if int(v) != MARKER_COLOR}
    if len(other_colors) not in (2, 3):
        return False

    raw_transforms = [_apply_slot_transform(arr, r, c) for r, c in TRANSFORM_BRANCHES]
    if len({_array_sig(x) for x in raw_transforms}) != len(TRANSFORM_BRANCHES):
        return False

    erased_variants = [_branch_output(arr, marker_pos, rc) for rc in TRANSFORM_BRANCHES]
    if len({_array_sig(x) for x in erased_variants}) != len(TRANSFORM_BRANCHES):
        return False

    target = _branch_output(arr, marker_pos, marker_pos)
    if any(np.array_equal(target, raw) for raw in raw_transforms):
        return False
    if np.count_nonzero(target == ERASE_COLOR) != 1:
        return False
    if np.count_nonzero(target == MARKER_COLOR) != 0:
        return False
    if not np.array_equal(main(arr), target):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        marker_pos = random.choice(NON_CENTER_POSITIONS)
        tile = _sample_colored_cells(marker_pos)
        if _is_valid_generated_input(tile):
            return tile
    raise RuntimeError("failed to generate a valid mirror-slot ARC input")
