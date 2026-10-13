from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

RING_ROWS = 2
RING_COLS = 3
RING_SIZE = 3
RING_GAP = 1
SECTION_GAP = 2
CELL_SIZE = 3
DIVIDER_WIDTH = 1

TOP_H = RING_ROWS * RING_SIZE + (RING_ROWS - 1) * RING_GAP
TOP_W = RING_COLS * RING_SIZE + (RING_COLS - 1) * RING_GAP
BLOCK_H = RING_ROWS * CELL_SIZE + (RING_ROWS - 1) * DIVIDER_WIDTH
BLOCK_W = RING_COLS * CELL_SIZE + (RING_COLS - 1) * DIVIDER_WIDTH
CONSTRUCT_H = TOP_H + SECTION_GAP + BLOCK_H

CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDES = ((0, 1), (1, 2), (2, 1), (1, 0))
RING_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
RING_PALETTE = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL)
SPRITE_PALETTE = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL, MAROON)

MAX_GENERATION_ATTEMPTS = 500
LOCAL_ATTEMPTS = 200


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in bboxes),
        min(box[1] for box in bboxes),
        max(box[2] for box in bboxes),
        max(box[3] for box in bboxes),
    )


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = tuple(
            (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
        )

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _cycle_positions(patch: np.ndarray, positions: tuple[tuple[int, int], ...], step: int) -> None:
    values = [int(patch[r, c]) for r, c in positions]
    n = len(values)
    rotated = [values[(i - step) % n] for i in range(n)]
    for (r, c), value in zip(positions, rotated):
        patch[r, c] = value


def _rotate_ring_patch(patch: np.ndarray, sprite_state: bool) -> np.ndarray:
    out = patch.copy()
    corner_step = 1 if sprite_state else -1
    side_step = -1 if sprite_state else 1
    _cycle_positions(out, CORNERS, corner_step)
    _cycle_positions(out, SIDES, side_step)
    return out


def _find_ring_boxes(region: np.ndarray) -> list[tuple[int, int, int, int]]:
    components = _components_mask(region != BLACK, connectivity=4)
    assert len(components) == 6, "expected exactly six upper-ring components"

    boxes = []
    for coords in components:
        assert int(coords.shape[0]) == 8, "each ring must have eight foreground pixels"
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "each ring must occupy a 3x3 bbox"
        patch = region[r0 : r1 + 1, c0 : c1 + 1]
        assert int(patch[1, 1]) == BLACK
        assert int(np.count_nonzero(patch != BLACK)) == 8
        boxes.append(box)

    boxes.sort(key=lambda box: (box[0], box[1]))
    row_starts = sorted({box[0] for box in boxes})
    col_starts = sorted({box[1] for box in boxes})
    assert len(row_starts) == 2 and len(col_starts) == 3
    assert row_starts[1] - row_starts[0] == RING_SIZE + RING_GAP
    assert col_starts[1] - col_starts[0] == RING_SIZE + RING_GAP
    assert col_starts[2] - col_starts[1] == RING_SIZE + RING_GAP
    expected = {(row_starts[r], col_starts[c]) for r in range(2) for c in range(3)}
    actual = {(box[0], box[1]) for box in boxes}
    assert actual == expected
    return boxes


def _extract_top_region(
    grid: np.ndarray, block_top: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    search_bottom = int(block_top) - SECTION_GAP
    assert search_bottom > 0
    assert np.all(grid[search_bottom:block_top, :] == BLACK), "expected two blank rows between regions"
    search = grid[:search_bottom, :]
    boxes = _find_ring_boxes(search)
    union = _bbox_union(boxes)
    top = _crop_bbox(search, union)
    adjusted = [
        (r0 - union[0], c0 - union[1], r1 - union[0], c1 - union[1])
        for r0, c0, r1, c1 in boxes
    ]
    assert top.shape == (TOP_H, TOP_W)
    return top, adjusted


def _decode_block_instructions(block: np.ndarray) -> np.ndarray:
    assert block.shape == (BLOCK_H, BLOCK_W), "block grid must be 7x11"
    full_grey_rows = [r for r in range(block.shape[0]) if np.all(block[r, :] == GREY)]
    full_grey_cols = [c for c in range(block.shape[1]) if np.all(block[:, c] == GREY)]
    assert full_grey_rows == [CELL_SIZE]
    assert full_grey_cols == [CELL_SIZE, 2 * CELL_SIZE + DIVIDER_WIDTH]

    cells = []
    for br in range(RING_ROWS):
        for bc in range(RING_COLS):
            r0 = br * (CELL_SIZE + DIVIDER_WIDTH)
            c0 = bc * (CELL_SIZE + DIVIDER_WIDTH)
            cell = block[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE].copy()
            assert GREY not in set(map(int, cell.flat))
            cells.append(cell)

    sprite_cells = [cell for cell in cells if len(np.unique(cell)) == 2]
    assert 0 < len(sprite_cells) < len(cells), "need at least one sprite cell and one blank cell"
    sprite = sprite_cells[0]
    for cell in sprite_cells[1:]:
        assert np.array_equal(cell, sprite), "all sprite cells must match"

    sprite_colors = [int(color) for color in np.unique(sprite)]
    assert len(sprite_colors) == 2
    counts = {color: int(np.count_nonzero(sprite == color)) for color in sprite_colors}
    assert len(set(counts.values())) == 2, "sprite colors must have distinct frequencies"
    blank_color = min(counts, key=counts.get)

    directions = np.zeros((RING_ROWS, RING_COLS), dtype=bool)
    for idx, cell in enumerate(cells):
        br, bc = divmod(idx, RING_COLS)
        if np.array_equal(cell, sprite):
            directions[br, bc] = True
        else:
            assert len(np.unique(cell)) == 1
            assert int(cell[0, 0]) == int(blank_color), "blank cells must use the sprite minority color"
    return directions


def _render_rotated_top(top: np.ndarray, directions: np.ndarray) -> np.ndarray:
    out = top.copy()
    for br in range(RING_ROWS):
        for bc in range(RING_COLS):
            r0 = br * (RING_SIZE + RING_GAP)
            c0 = bc * (RING_SIZE + RING_GAP)
            patch = top[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE]
            out[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] = _rotate_ring_patch(
                patch, bool(directions[br, bc])
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each upper 3x3 ring according to the matching sprite-coded lower block.

    Input structure:
    - A black background contains an upper 2x3 array of separated 3x3 rings and, two rows below it, a lower 2x3 block grid split by grey divider lines.
    - Each upper object is a full 3x3 ring of eight colored pixels around a black center.
    - Each lower 3x3 block is either one repeated two-color sprite or one uniform blank block using the sprite's minority color.

    Rule:
    - Infer the repeated mixed-color lower block as the sprite and treat uniform minority-color blocks as blanks.
    - Match lower blocks to upper rings by row-major position.
    - If a block is a sprite, rotate that ring's corners one step clockwise and its side-centers one step counterclockwise.
    - If a block is a blank, reverse both directions, then return only the rotated upper ring array.

    Key insight:
    - The lower block grid is not output content; it is a spatial instruction table whose sprite-vs-blank states encode each ring's rotation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.size > 0, "expected a grey divider grid"
    block_bbox = _bbox_coords(grey_coords)
    block = _crop_bbox(g, block_bbox)
    directions = _decode_block_instructions(block)

    top, ring_boxes = _extract_top_region(g, block_bbox[0])
    out = top.copy()
    for idx, box in enumerate(ring_boxes):
        br, bc = divmod(idx, RING_COLS)
        r0, c0, r1, c1 = box
        patch = top[r0 : r1 + 1, c0 : c1 + 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = _rotate_ring_patch(patch, bool(directions[br, bc]))
    return out


def _has_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _make_sprite() -> tuple[np.ndarray, int]:
    positions = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)]
    majority_color, blank_color = random.sample(SPRITE_PALETTE, 2)
    for _ in range(LOCAL_ATTEMPTS):
        minority_count = random.choice((3, 4))
        minority_cells = random.sample(positions, minority_count)
        mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        for r, c in minority_cells:
            mask[r, c] = True
        if any(mask[r].sum() in {0, CELL_SIZE} for r in range(CELL_SIZE)):
            continue
        if any(mask[:, c].sum() in {0, CELL_SIZE} for c in range(CELL_SIZE)):
            continue
        if _has_symmetry(mask):
            continue
        sprite = np.full((CELL_SIZE, CELL_SIZE), majority_color, dtype=int)
        sprite[mask] = blank_color
        return sprite, int(blank_color)
    raise RuntimeError("failed to sample a non-degenerate sprite")


def _make_direction_mask() -> np.ndarray:
    for _ in range(LOCAL_ATTEMPTS):
        mask = np.array(
            [[bool(random.getrandbits(1)) for _ in range(RING_COLS)] for _ in range(RING_ROWS)],
            dtype=bool,
        )
        if not mask.any() or mask.all():
            continue
        if all(np.all(mask[r, :] == mask[r, 0]) for r in range(RING_ROWS)):
            continue
        if all(mask[0, c] == mask[1, c] for c in range(RING_COLS)):
            continue
        return mask
    raise RuntimeError("failed to sample an irregular instruction mask")


def _make_ring() -> np.ndarray:
    for _ in range(LOCAL_ATTEMPTS):
        unique_count = random.randint(4, len(RING_PALETTE))
        chosen = random.sample(RING_PALETTE, unique_count)
        values = chosen + [random.choice(chosen) for _ in range(len(RING_POSITIONS) - unique_count)]
        random.shuffle(values)
        patch = np.zeros((RING_SIZE, RING_SIZE), dtype=int)
        for (r, c), value in zip(RING_POSITIONS, values):
            patch[r, c] = value
        if len({int(patch[r, c]) for r, c in CORNERS}) < 2:
            continue
        if len({int(patch[r, c]) for r, c in SIDES}) < 2:
            continue
        clockwise = _rotate_ring_patch(patch, True)
        counter = _rotate_ring_patch(patch, False)
        if np.array_equal(clockwise, counter):
            continue
        if np.array_equal(clockwise, patch) or np.array_equal(counter, patch):
            continue
        return patch
    raise RuntimeError("failed to sample a non-degenerate ring")


def _assemble_top(rings: list[np.ndarray]) -> np.ndarray:
    top = np.zeros((TOP_H, TOP_W), dtype=int)
    for idx, patch in enumerate(rings):
        br, bc = divmod(idx, RING_COLS)
        r0 = br * (RING_SIZE + RING_GAP)
        c0 = bc * (RING_SIZE + RING_GAP)
        top[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] = patch
    return top


def _assemble_block(sprite: np.ndarray, blank_color: int, directions: np.ndarray) -> np.ndarray:
    block = np.full((BLOCK_H, BLOCK_W), GREY, dtype=int)
    blank = np.full((CELL_SIZE, CELL_SIZE), int(blank_color), dtype=int)
    for br in range(RING_ROWS):
        for bc in range(RING_COLS):
            r0 = br * (CELL_SIZE + DIVIDER_WIDTH)
            c0 = bc * (CELL_SIZE + DIVIDER_WIDTH)
            block[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = (
                sprite if directions[br, bc] else blank
            )
    return block


def _assemble_input(top: np.ndarray, block: np.ndarray) -> np.ndarray:
    top_margin = random.randint(0, 7)
    left_margin = random.randint(0, 9)
    bottom_margin = random.randint(0, 30 - CONSTRUCT_H - top_margin)
    right_margin = random.randint(0, 30 - TOP_W - left_margin)

    h = top_margin + CONSTRUCT_H + bottom_margin
    w = left_margin + TOP_W + right_margin
    grid = np.zeros((h, w), dtype=int)

    top_r = top_margin
    left_c = left_margin
    block_r = top_r + TOP_H + SECTION_GAP
    grid[top_r : top_r + TOP_H, left_c : left_c + TOP_W] = top
    grid[block_r : block_r + BLOCK_H, left_c : left_c + BLOCK_W] = block
    return grid


def _validate_generated_sample(
    grid: np.ndarray, top: np.ndarray, directions: np.ndarray, sprite: np.ndarray, blank_color: int
) -> bool:
    expected = _render_rotated_top(top, directions)
    if np.array_equal(expected, top):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    for br in range(RING_ROWS):
        for bc in range(RING_COLS):
            r0 = br * (RING_SIZE + RING_GAP)
            c0 = bc * (RING_SIZE + RING_GAP)
            patch = top[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE]
            if np.array_equal(_rotate_ring_patch(patch, True), _rotate_ring_patch(patch, False)):
                return False
            flipped = directions.copy()
            flipped[br, bc] = ~flipped[br, bc]
            if np.array_equal(_render_rotated_top(top, flipped), expected):
                return False

    counts = {int(color): int(np.count_nonzero(sprite == color)) for color in np.unique(sprite)}
    if blank_color != min(counts, key=counts.get):
        return False
    if len(set(counts.values())) != 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sprite, blank_color = _make_sprite()
        directions = _make_direction_mask()
        rings = [_make_ring() for _ in range(RING_ROWS * RING_COLS)]
        top = _assemble_top(rings)
        block = _assemble_block(sprite, blank_color, directions)
        grid = _assemble_input(top, block)
        if _validate_generated_sample(grid, top, directions, sprite, blank_color):
            return grid
    raise RuntimeError("failed to generate a valid instruction-driven ring sample")
