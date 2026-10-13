from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GRID_SIZE = 16
QUADRANT_SIZE = 8
BLOCK_SIZE = 3
OUTPUT_SIZE = 6
PALETTE = (1, 2, 3, 4, 5, 6, 7)

ROTATION_TURNS = {
    0: 2,  # upper-left: 180
    1: 3,  # upper-right: 90 clockwise
    2: 1,  # lower-left: 90 counterclockwise
    3: 0,  # lower-right: unchanged
}

TWO_COLOR_SPLITS = ((4, 5), (5, 4), (3, 6), (6, 3))
THREE_COLOR_SPLITS = (
    (3, 3, 3),
    (2, 3, 4),
    (2, 2, 5),
    (1, 3, 5),
    (1, 4, 4),
)
MAX_GENERATION_ATTEMPTS = 800

_SEEN_INPUT_SIGNATURES: set[bytes] = set()
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rotate_tile(tile: np.ndarray, quadrant_index: int) -> np.ndarray:
    return np.rot90(tile, k=ROTATION_TURNS[quadrant_index]).copy()


def _rotation_family_distinct(tile: np.ndarray) -> bool:
    family = [np.rot90(tile, k=k) for k in range(4)]
    return len({member.tobytes() for member in family}) == 4


def _extract_quadrant_blocks(region: np.ndarray) -> list[dict]:
    comps = _components(region != BACKGROUND)
    assert len(comps) == 3, "each quadrant must contain exactly three separated blocks"

    blocks: list[dict] = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        crop = _crop_bbox(region, bbox)
        assert crop.shape == (BLOCK_SIZE, BLOCK_SIZE), "each block must crop to 3x3"
        assert len(comp) == BLOCK_SIZE * BLOCK_SIZE, "each block must be fully filled"
        assert np.all(crop != BACKGROUND), "blocks cannot contain background holes"

        colors = tuple(sorted(set(map(int, crop.ravel()))))
        blocks.append(
            {
                "bbox": bbox,
                "crop": crop,
                "colors": colors,
                "color_count": len(colors),
            }
        )

    color_counts = sorted(block["color_count"] for block in blocks)
    assert color_counts == [2, 2, 3], "each quadrant must have exactly one three-color outlier"
    signatures = [tuple(block["colors"]) for block in blocks]
    assert len(set(signatures)) == 3, "the three blocks in a quadrant must have distinct color sets"
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique three-color block from each quadrant, rotate it by quadrant, and pack the four crops into a montage.

    Input structure:
    - The grid is a 16x16 black-background canvas split conceptually into four 8x8 quadrants with no visible divider.
    - Each quadrant contains exactly three separated filled 3x3 blocks, all fully contained inside that quadrant.
    - In every quadrant, exactly one block uses three non-black colors and the other two blocks use exactly two colors each.

    Rule:
    - For each quadrant independently, find the unique 3x3 block whose cells contain exactly three distinct colors.
    - Crop that outlier block to its 3x3 bounding box.
    - Rotate the crop according to quadrant position: upper-left 180 degrees, upper-right 90 degrees clockwise,
      lower-left 90 degrees counterclockwise, lower-right unchanged.
    - Concatenate the four rotated 3x3 crops into a 6x6 output in quadrant order.

    Key insight:
    - Color diversity chooses which local object survives, while quadrant position alone determines how that chosen object is oriented in the final 2x2 montage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 16x16 grid"

    quadrants = [
        g[:QUADRANT_SIZE, :QUADRANT_SIZE],
        g[:QUADRANT_SIZE, QUADRANT_SIZE:],
        g[QUADRANT_SIZE:, :QUADRANT_SIZE],
        g[QUADRANT_SIZE:, QUADRANT_SIZE:],
    ]

    tiles: list[np.ndarray] = []
    for quadrant_index, quadrant in enumerate(quadrants):
        blocks = _extract_quadrant_blocks(quadrant)
        outliers = [block for block in blocks if block["color_count"] == 3]
        assert len(outliers) == 1
        tiles.append(_rotate_tile(outliers[0]["crop"], quadrant_index))

    return np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]]).astype(int)


def _sample_partitioned_colors(rng: np.random.Generator) -> tuple[tuple[int, int, int], tuple[int, int], tuple[int, int]]:
    colors = rng.permutation(np.array(PALETTE, dtype=int)).tolist()
    return tuple(colors[:3]), tuple(colors[3:5]), tuple(colors[5:7])


def _make_pattern(colors: tuple[int, ...], counts: tuple[int, ...], rng: np.random.Generator, *, require_full_rotation_family: bool) -> np.ndarray:
    cells = np.repeat(np.array(colors, dtype=int), np.array(counts, dtype=int))
    while True:
        shuffled = rng.permutation(cells)
        tile = shuffled.reshape(BLOCK_SIZE, BLOCK_SIZE).astype(int)
        if len(set(map(int, tile.ravel()))) != len(colors):
            continue
        if require_full_rotation_family and not _rotation_family_distinct(tile):
            continue
        return tile


def _make_three_color_tile(colors: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    split = THREE_COLOR_SPLITS[int(rng.integers(0, len(THREE_COLOR_SPLITS)))]
    split = tuple(int(x) for x in rng.permutation(np.array(split, dtype=int)))
    return _make_pattern(colors, split, rng, require_full_rotation_family=True)


def _make_two_color_tile(colors: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    split = TWO_COLOR_SPLITS[int(rng.integers(0, len(TWO_COLOR_SPLITS)))]
    return _make_pattern(colors, split, rng, require_full_rotation_family=False)


def _slot_layout(rng: np.random.Generator) -> tuple[list[tuple[int, int]], int]:
    row_low = int(rng.integers(0, 2))
    row_high = int(rng.integers(4, 6))
    col_low = int(rng.integers(0, 2))
    col_high = int(rng.integers(4, 6))
    slots = [
        (row_low, col_low),
        (row_low, col_high),
        (row_high, col_low),
        (row_high, col_high),
    ]
    missing_corner = int(rng.integers(0, 4))
    used_slots = [slot for idx, slot in enumerate(slots) if idx != missing_corner]
    return used_slots, missing_corner


def _paint_block(grid: np.ndarray, tile: np.ndarray, top: int, left: int) -> None:
    grid[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = tile


def _hash_array(arr: np.ndarray) -> bytes:
    return arr.shape[0].to_bytes(2, "little") + arr.shape[1].to_bytes(2, "little") + arr.tobytes()


def _validate_sample(grid: np.ndarray, metadata: list[dict]) -> np.ndarray:
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    output = main(grid)
    assert output.shape == (OUTPUT_SIZE, OUTPUT_SIZE)

    raw_outliers: list[np.ndarray] = []
    rotated_outliers: list[np.ndarray] = []
    outlier_corner_labels: list[int] = []

    for quadrant_index, info in enumerate(metadata):
        r0 = 0 if quadrant_index < 2 else QUADRANT_SIZE
        c0 = 0 if quadrant_index % 2 == 0 else QUADRANT_SIZE
        quadrant = grid[r0 : r0 + QUADRANT_SIZE, c0 : c0 + QUADRANT_SIZE]
        blocks = _extract_quadrant_blocks(quadrant)

        # Match detected blocks back to generated positions to validate the intended outlier.
        generated_tiles = {
            (top - r0, left - c0): tile.copy() for top, left, tile in info["placed_blocks"]
        }
        assert len(generated_tiles) == 3

        matched_outlier = None
        for block in blocks:
            local_pos = (block["bbox"][0], block["bbox"][1])
            assert local_pos in generated_tiles, "detected block location mismatch"
            assert np.array_equal(block["crop"], generated_tiles[local_pos]), "detected block pattern mismatch"
            if block["color_count"] == 3:
                matched_outlier = block

        assert matched_outlier is not None
        expected_local_pos = (info["outlier_abs"][0] - r0, info["outlier_abs"][1] - c0)
        assert (matched_outlier["bbox"][0], matched_outlier["bbox"][1]) == expected_local_pos

        raw = matched_outlier["crop"]
        rotated = _rotate_tile(raw, quadrant_index)
        assert _rotation_family_distinct(raw), "outlier rotation must be visually meaningful"
        for other in blocks:
            if other is matched_outlier:
                continue
            assert not np.array_equal(_rotate_tile(other["crop"], quadrant_index), rotated), (
                "choosing a non-outlier must not accidentally reproduce the true tile"
            )

        raw_outliers.append(raw)
        rotated_outliers.append(rotated)
        outlier_corner_labels.append(info["outlier_corner"])

    assert len(set(tile.tobytes() for tile in raw_outliers)) == 4, "raw outlier patterns must be distinct"
    assert len(set(tile.tobytes() for tile in rotated_outliers)) == 4, "rotated montage tiles must be distinct"
    assert len(set(outlier_corner_labels)) >= 3, "outlier positions should not collapse to one trivial local corner"

    expected = np.block([[rotated_outliers[0], rotated_outliers[1]], [rotated_outliers[2], rotated_outliers[3]]])
    assert np.array_equal(output, expected)
    return output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        metadata: list[dict] = []

        for quadrant_index in range(4):
            quadrant_row = 0 if quadrant_index < 2 else QUADRANT_SIZE
            quadrant_col = 0 if quadrant_index % 2 == 0 else QUADRANT_SIZE
            slots, _missing_corner = _slot_layout(rng)
            outlier_slot_index = int(rng.integers(0, 3))
            outlier_corner = [idx for idx in range(4) if idx != _missing_corner][outlier_slot_index]

            three_colors, pair_a, pair_b = _sample_partitioned_colors(rng)
            outlier_tile = _make_three_color_tile(three_colors, rng)
            filler_a = _make_two_color_tile(pair_a, rng)
            filler_b = _make_two_color_tile(pair_b, rng)
            placed_tiles = [filler_a, filler_b]
            rng.shuffle(placed_tiles)
            placed_tiles.insert(outlier_slot_index, outlier_tile)

            placed_blocks: list[tuple[int, int, np.ndarray]] = []
            for (local_r, local_c), tile in zip(slots, placed_tiles):
                abs_r = quadrant_row + local_r
                abs_c = quadrant_col + local_c
                _paint_block(grid, tile, abs_r, abs_c)
                placed_blocks.append((abs_r, abs_c, tile))

            metadata.append(
                {
                    "placed_blocks": placed_blocks,
                    "outlier_abs": (
                        quadrant_row + slots[outlier_slot_index][0],
                        quadrant_col + slots[outlier_slot_index][1],
                    ),
                    "outlier_corner": outlier_corner,
                }
            )

        try:
            output = _validate_sample(grid, metadata)
        except AssertionError:
            continue

        input_sig = _hash_array(grid)
        output_sig = _hash_array(output)
        if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
            continue

        _SEEN_INPUT_SIGNATURES.add(input_sig)
        _SEEN_OUTPUT_SIGNATURES.add(output_sig)
        return grid

    raise RuntimeError("failed to generate a valid quadrant outlier rotation puzzle input")
