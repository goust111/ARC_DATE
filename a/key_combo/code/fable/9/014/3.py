from __future__ import annotations

import numpy as np


TRANSFORM_NAMES = ("identity", "rot90", "rot180", "rot270", "fliplr", "flipud")
INVERSE_TRANSFORM = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "fliplr": "fliplr",
    "flipud": "flipud",
}

OUTER_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

BLOCK_SIZE_CHOICES = (4, 6, 8)
MAX_GENERATION_ATTEMPTS = 400
MAX_REFERENCE_ATTEMPTS = 400
MIN_TRANSFORM_DISTANCE = 4


def _apply_transform(block: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return block.copy()
    if name == "rot90":
        return np.rot90(block, 1).copy()
    if name == "rot180":
        return np.rot90(block, 2).copy()
    if name == "rot270":
        return np.rot90(block, 3).copy()
    if name == "fliplr":
        return np.fliplr(block).copy()
    if name == "flipud":
        return np.flipud(block).copy()
    raise ValueError(f"unknown transform: {name}")


def _transform_coord(coord: tuple[int, int], size: int, name: str) -> tuple[int, int]:
    r, c = map(int, coord)
    n = int(size)
    if name == "identity":
        return r, c
    if name == "rot90":
        return n - 1 - c, r
    if name == "rot180":
        return n - 1 - r, n - 1 - c
    if name == "rot270":
        return c, n - 1 - r
    if name == "fliplr":
        return r, n - 1 - c
    if name == "flipud":
        return n - 1 - r, c
    raise ValueError(f"unknown transform: {name}")


def _split_blocks(grid: np.ndarray) -> list[list[np.ndarray]]:
    h, w = map(int, grid.shape)
    assert h % 3 == 0 and w % 3 == 0, "grid must split evenly into a 3x3 partition"
    block_h = h // 3
    block_w = w // 3
    assert block_h == block_w, "blocks must be square to support quarter turns"
    return [
        [
            grid[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w].copy()
            for c in range(3)
        ]
        for r in range(3)
    ]


def _detect_transform(reference: np.ndarray, block: np.ndarray) -> str:
    ref_colors = set(map(int, np.unique(reference)))
    matches = []
    for name in TRANSFORM_NAMES:
        candidate = _apply_transform(reference, name)
        mismatches = np.argwhere(candidate != block)
        if mismatches.shape[0] != 1:
            continue
        r, c = map(int, mismatches[0])
        if int(block[r, c]) in ref_colors:
            continue
        matches.append(name)
    assert len(matches) == 1, "each outer block must match the reference under exactly one transform"
    return matches[0]


def _recover_marker(reference: np.ndarray, block: np.ndarray) -> tuple[int, int, int]:
    ref_colors = set(map(int, np.unique(reference)))
    transform_name = _detect_transform(reference, block)
    restored = _apply_transform(block, INVERSE_TRANSFORM[transform_name])
    marker_cells = np.argwhere(~np.isin(restored, list(ref_colors)))
    assert marker_cells.shape == (1, 2), "expected exactly one marker pixel after inverse transform"
    r, c = map(int, marker_cells[0])
    color = int(restored[r, c])
    return r, c, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate the inverse-mapped marker pixels from transformed outer blocks onto the center reference block.

    Input structure:
    - The grid is evenly partitioned into a 3x3 array of equal square blocks.
    - The center block is a reference pattern that uses a palette disjoint from the marker colors.
    - Each outer block is one allowed transform of the center block with exactly one cell replaced by a unique marker color.

    Rule:
    - For each outer block, ignore its marker color and determine which of the allowed transforms maps the center block to that block.
    - Apply the inverse transform to restore the block to the center orientation, then read the marker pixel's coordinate in that restored frame.
    - Output a single-block grid whose base is the center block, with all eight recovered marker pixels overlaid at their inverse-mapped coordinates.

    Key insight:
    - The marker positions only become meaningful after undoing each block's local transform, because all eight markers share the center block's hidden coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _split_blocks(g)
    reference = blocks[1][1]
    out = reference.copy()

    recovered = []
    for block_r, block_c in OUTER_POSITIONS:
        recovered.append(_recover_marker(reference, blocks[block_r][block_c]))

    coords = [(r, c) for r, c, _color in recovered]
    colors = [color for _r, _c, color in recovered]
    assert len(set(coords)) == 8, "inverse-mapped marker coordinates must not overlap"
    assert len(set(colors)) == 8, "marker colors must stay unique"

    for r, c, color in recovered:
        out[r, c] = color
    return out


def _reference_is_valid(reference: np.ndarray) -> bool:
    size = int(reference.shape[0])
    if reference.shape != (size, size):
        return False
    if len(np.unique(reference)) != 2:
        return False

    ones_per_row = np.count_nonzero(reference, axis=1)
    ones_per_col = np.count_nonzero(reference, axis=0)
    if np.any(ones_per_row == 0) or np.any(ones_per_row == size):
        return False
    if np.any(ones_per_col == 0) or np.any(ones_per_col == size):
        return False

    transformed = [_apply_transform(reference, name) for name in TRANSFORM_NAMES]
    for i in range(len(transformed)):
        for j in range(i + 1, len(transformed)):
            if np.array_equal(transformed[i], transformed[j]):
                return False
            if int(np.count_nonzero(transformed[i] != transformed[j])) < MIN_TRANSFORM_DISTANCE:
                return False
    return True


def _make_reference_block(size: int, foreground_color: int) -> np.ndarray:
    n = int(size)
    fg = int(foreground_color)
    for _ in range(MAX_REFERENCE_ATTEMPTS):
        density = float(np.random.uniform(0.28, 0.62))
        mask = np.random.random((n, n)) < density
        count = int(np.count_nonzero(mask))
        if count < n + 1 or count > n * n - (n + 1):
            continue
        reference = np.where(mask, fg, 0).astype(int)
        if _reference_is_valid(reference):
            return reference
    raise RuntimeError("failed to generate a non-degenerate reference block")


def _make_transform_assignment() -> list[str]:
    transforms = list(TRANSFORM_NAMES)
    extras = list(np.random.choice(TRANSFORM_NAMES[1:], size=2, replace=True))
    transforms.extend(extras)
    np.random.shuffle(transforms)
    return transforms


def _expected_output(reference: np.ndarray, coords: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = reference.copy()
    for (r, c), color in zip(coords, colors):
        out[int(r), int(c)] = int(color)
    return out


def _naive_overlay(reference: np.ndarray, observed_markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = reference.copy()
    for r, c, color in observed_markers:
        out[int(r), int(c)] = int(color)
    return out


def _build_input(
    reference: np.ndarray,
    transforms: list[str],
    marker_coords: list[tuple[int, int]],
    marker_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    size = int(reference.shape[0])
    grid = np.zeros((3 * size, 3 * size), dtype=int)
    observed_markers = []

    for (block_r, block_c), transform_name, coord, color in zip(
        OUTER_POSITIONS, transforms, marker_coords, marker_colors
    ):
        block = _apply_transform(reference, transform_name)
        obs_r, obs_c = _transform_coord(coord, size, transform_name)
        block[obs_r, obs_c] = int(color)
        grid[
            block_r * size : (block_r + 1) * size,
            block_c * size : (block_c + 1) * size,
        ] = block
        observed_markers.append((obs_r, obs_c, int(color)))

    grid[size : 2 * size, size : 2 * size] = reference
    expected = _expected_output(reference, marker_coords, marker_colors)
    return grid, expected, observed_markers


def _attempt_input() -> np.ndarray | None:
    size = int(np.random.choice(BLOCK_SIZE_CHOICES))
    foreground_color = int(np.random.randint(1, 10))
    reference = _make_reference_block(size, foreground_color)

    transform_names = _make_transform_assignment()
    flat_coords = np.random.choice(size * size, size=8, replace=False)
    marker_coords = [(int(idx // size), int(idx % size)) for idx in flat_coords]
    marker_colors = list(np.random.permutation([c for c in range(1, 10) if c != foreground_color]))

    expected = _expected_output(reference, marker_coords, marker_colors)
    grid, direct_expected, observed_markers = _build_input(
        reference, transform_names, marker_coords, marker_colors
    )
    if not np.array_equal(expected, direct_expected):
        return None

    for transform_name, coord, observed in zip(transform_names, marker_coords, observed_markers):
        if transform_name != "identity" and _transform_coord(coord, size, transform_name) == coord:
            return None
        if int(observed[2]) in set(map(int, np.unique(reference))):
            return None

    if np.array_equal(_naive_overlay(reference, observed_markers), expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    blocks = _split_blocks(grid)
    seen_coords = set()
    seen_colors = set()
    for (block_r, block_c), true_transform, true_coord, true_color in zip(
        OUTER_POSITIONS, transform_names, marker_coords, marker_colors
    ):
        block = blocks[block_r][block_c]
        detected = _detect_transform(reference, block)
        if detected != true_transform:
            return None
        recovered = _recover_marker(reference, block)
        if recovered[:2] != tuple(map(int, true_coord)):
            return None
        if recovered[2] != int(true_color):
            return None
        seen_coords.add(recovered[:2])
        seen_colors.add(recovered[2])

    if len(seen_coords) != 8 or len(seen_colors) != 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
