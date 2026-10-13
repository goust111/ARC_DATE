from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GRID_SIZE = 15
BLOCK_SIZE = 5
COARSE_SIZE = 3
TEMPLATE_SIZE = 3

BG = 0
TEMPLATE_COLORS = (1, 2)
MARKER = 3

CENTER_BLOCK = (1, 1)
CENTER_TEMPLATE_SLICE = (slice(1, 4), slice(1, 4))

SIDE_TO_ROTATION = {
    "top": 0,
    "right": 3,   # np.rot90 k is counterclockwise; 3 => clockwise 90
    "bottom": 2,
    "left": 1,    # clockwise 270
}

SIDE_TO_MARKER_POS = {
    "top": (0, 2),
    "right": (2, 4),
    "bottom": (4, 2),
    "left": (2, 0),
}

OUTPUT_SEEN: set[bytes] = set()
INPUT_SEEN: set[bytes] = set()
GEN_ATTEMPTS = 500


def _block_origin(block_r: int, block_c: int) -> tuple[int, int]:
    return block_r * BLOCK_SIZE, block_c * BLOCK_SIZE


def _iter_blocks() -> Iterable[tuple[int, int]]:
    for block_r in range(COARSE_SIZE):
        for block_c in range(COARSE_SIZE):
            yield block_r, block_c


def _extract_block(grid: np.ndarray, block_r: int, block_c: int) -> np.ndarray:
    r0, c0 = _block_origin(block_r, block_c)
    return grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]


def _nonblack_bbox(block: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(block != BG)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(block: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return block[r0 : r1 + 1, c0 : c1 + 1].copy()


def _nearest_side(marker_rc: tuple[int, int]) -> str:
    r, c = map(int, marker_rc)
    distances = {
        "top": r,
        "right": BLOCK_SIZE - 1 - c,
        "bottom": BLOCK_SIZE - 1 - r,
        "left": c,
    }
    best_side = min(distances, key=distances.get)
    best_distance = distances[best_side]
    if list(distances.values()).count(best_distance) != 1:
        raise AssertionError("marker must have a unique nearest edge")
    return best_side


def _find_marker(block: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(block == MARKER)
    if coords.size == 0:
        return None
    if len(coords) != 1:
        raise AssertionError("each marked block must contain exactly one green marker")
    return tuple(map(int, coords[0]))


def _rotations_distinct(tile: np.ndarray) -> bool:
    return len({np.rot90(tile, k).tobytes() for k in range(4)}) == 4


def _extract_template(grid: np.ndarray) -> np.ndarray:
    center_block = _extract_block(grid, *CENTER_BLOCK)
    bbox = _nonblack_bbox(center_block)
    if bbox is None:
        raise AssertionError("center block must contain the template")
    tile = _crop(center_block, bbox)
    if tile.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        raise AssertionError("template bbox must be 3x3")
    if bbox != (1, 1, 3, 3):
        raise AssertionError("template must be centered inside the 5x5 block")
    colors = set(map(int, np.unique(tile)))
    if colors != set(TEMPLATE_COLORS):
        raise AssertionError("template must use exactly blue and red")
    if not _rotations_distinct(tile):
        raise AssertionError("template rotations must all be distinguishable")
    if np.any(center_block == MARKER):
        raise AssertionError("center block may not contain markers")
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each coarse marker block into a rotated copy of the center template.

    Input structure:
    - The input is a 15x15 black grid partitioned into a 3x3 array of 5x5 blocks.
    - The center 5x5 block contains a centered 3x3 blue/red template with a one-cell black margin.
    - Some non-center blocks contain exactly one green marker pixel; all other non-center blocks are black.

    Rule:
    - Extract the 3x3 template from the center block.
    - For each marked 5x5 block, compare the marker's distances to the block's top, right, bottom, and left edges.
    - Stamp the template into the corresponding 3x3 output cell, rotated by the uniquely nearest edge:
      top -> 0 degrees, right -> clockwise 90, bottom -> 180, left -> clockwise 270.
    - Leave the output's center cell and every unmarked cell black.

    Key insight:
    - A single marker pixel encodes two things at once: which coarse block receives the stamp, and which rotation to use, via the nearest block edge.
    """
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIZE, GRID_SIZE):
        raise AssertionError("expected a 15x15 grid")

    template = _extract_template(g)
    out = np.zeros((COARSE_SIZE * TEMPLATE_SIZE, COARSE_SIZE * TEMPLATE_SIZE), dtype=int)

    for block_r, block_c in _iter_blocks():
        if (block_r, block_c) == CENTER_BLOCK:
            continue

        block = _extract_block(g, block_r, block_c)
        marker_rc = _find_marker(block)
        nonzero = set(map(int, np.unique(block))) - {BG}
        if marker_rc is None:
            if nonzero:
                raise AssertionError("non-center blocks must be empty or contain one marker")
            continue

        if nonzero != {MARKER}:
            raise AssertionError("marked blocks may only contain the green marker")

        side = _nearest_side(marker_rc)
        rotation_k = SIDE_TO_ROTATION[side]
        rotated = np.rot90(template, rotation_k)
        r0 = block_r * TEMPLATE_SIZE
        c0 = block_c * TEMPLATE_SIZE
        out[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE] = rotated

    return out


def _make_template() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        tile = np.random.choice(TEMPLATE_COLORS, size=(TEMPLATE_SIZE, TEMPLATE_SIZE)).astype(int)
        if len(set(map(int, np.unique(tile)))) != 2:
            continue
        if _rotations_distinct(tile):
            return tile
    raise RuntimeError("failed to generate a non-symmetric template")


def _pick_marker_layout() -> tuple[list[tuple[int, int]], list[str]]:
    non_center_blocks = [block for block in _iter_blocks() if block != CENTER_BLOCK]
    marker_count = random.choices([2, 3, 4], weights=[1, 3, 4], k=1)[0]
    blocks = random.sample(non_center_blocks, marker_count)

    if marker_count == 4:
        sides = random.sample(list(SIDE_TO_MARKER_POS), 4)
    else:
        sides = random.sample(list(SIDE_TO_MARKER_POS), marker_count)
    random.shuffle(sides)
    return blocks, sides


def _build_input(template: np.ndarray, blocks: list[tuple[int, int]], sides: list[str]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    center = _extract_block(grid, *CENTER_BLOCK)
    center[CENTER_TEMPLATE_SLICE] = template

    for (block_r, block_c), side in zip(blocks, sides):
        block = _extract_block(grid, block_r, block_c)
        mr, mc = SIDE_TO_MARKER_POS[side]
        block[mr, mc] = MARKER
    return grid


def _expected_output(template: np.ndarray, blocks: list[tuple[int, int]], sides: list[str]) -> np.ndarray:
    out = np.zeros((COARSE_SIZE * TEMPLATE_SIZE, COARSE_SIZE * TEMPLATE_SIZE), dtype=int)
    for (block_r, block_c), side in zip(blocks, sides):
        r0 = block_r * TEMPLATE_SIZE
        c0 = block_c * TEMPLATE_SIZE
        out[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE] = np.rot90(template, SIDE_TO_ROTATION[side])
    return out


def _validate_input_semantics(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        raise AssertionError("grid shape must stay fixed at 15x15")

    template = _extract_template(grid)
    marked_blocks: list[tuple[int, int]] = []
    marked_sides: list[str] = []

    for block_r, block_c in _iter_blocks():
        block = _extract_block(grid, block_r, block_c)
        if (block_r, block_c) == CENTER_BLOCK:
            continue

        marker_rc = _find_marker(block)
        if marker_rc is None:
            if np.any(block != BG):
                raise AssertionError("empty non-center blocks must be fully black")
            continue

        if tuple(marker_rc) not in set(SIDE_TO_MARKER_POS.values()):
            raise AssertionError("marker must lie on one of the four canonical edge positions")
        side = _nearest_side(marker_rc)
        dists = {
            "top": marker_rc[0],
            "right": BLOCK_SIZE - 1 - marker_rc[1],
            "bottom": BLOCK_SIZE - 1 - marker_rc[0],
            "left": marker_rc[1],
        }
        if dists[side] > 1:
            raise AssertionError("marker must be within one cell of its nearest edge")
        others = [dist for name, dist in dists.items() if name != side]
        if min(others) < 2:
            raise AssertionError("all non-nearest edges must be at least distance two away")
        marked_blocks.append((block_r, block_c))
        marked_sides.append(side)

    if not (2 <= len(marked_blocks) <= 4):
        raise AssertionError("must place markers in 2 to 4 blocks")
    if len(set(marked_sides)) < 2:
        raise AssertionError("samples must include at least two different rotation directions")

    return template, marked_blocks, marked_sides


def _signature(grid: np.ndarray) -> bytes:
    return grid.shape[0].to_bytes(1, "little") + grid.shape[1].to_bytes(1, "little") + grid.tobytes()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template = _make_template()
        blocks, sides = _pick_marker_layout()
        grid = _build_input(template, blocks, sides)

        parsed_template, parsed_blocks, parsed_sides = _validate_input_semantics(grid)
        expected = _expected_output(parsed_template, parsed_blocks, parsed_sides)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        output_sig = _signature(actual)
        input_sig = _signature(grid)
        if output_sig in OUTPUT_SEEN or input_sig in INPUT_SEEN:
            continue

        OUTPUT_SEEN.add(output_sig)
        INPUT_SEEN.add(input_sig)
        return grid

    raise RuntimeError("failed to generate a fresh valid coarse-marker rotation puzzle")
