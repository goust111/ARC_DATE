from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8

GRID_H = 11
LEFT_W = 11
SLOT_SIZE = 3
STRIDE = 4
ROW_BANDS = [(0, 3), (4, 7), (8, 11)]
COL_BANDS = [(0, 3), (4, 7), (8, 11)]
INPUT_SLOT_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RIGHT_INPUT_COLORS = [1, 2, 3, 4, 6, 7, 9]
RIGHT_W_RANGE = (5, 10)
RIGHT_OCCUPANCY_RANGE = (0.20, 0.40)
GEN_RETRY_LIMIT = 400
SLOT_POSITIONS = [(r, c) for r in range(3) for c in range(3)]

_RIGHT_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _all_symbol_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        if ((bits >> 4) & 1) == 0:
            continue
        area = bits.bit_count()
        if not (3 <= area <= 6):
            continue
        if area == 9:
            continue
        vals = [(bits >> i) & 1 for i in range(9)]
        masks.append(np.array(vals, dtype=bool).reshape(3, 3))
    return masks


ALL_MASKS = _all_symbol_masks()


def _connector_slots(matching: set[tuple[int, int]]) -> set[tuple[int, int]]:
    connectors: set[tuple[int, int]] = set()
    for slot_r in range(3):
        cols = sorted(slot_c for row, slot_c in matching if row == slot_r)
        for left, right in zip(cols, cols[1:]):
            for slot_c in range(left + 1, right):
                if (slot_r, slot_c) not in matching:
                    connectors.add((slot_r, slot_c))
    for slot_c in range(3):
        rows = sorted(slot_r for slot_r, col in matching if col == slot_c)
        for top, bottom in zip(rows, rows[1:]):
            for slot_r in range(top + 1, bottom):
                if (slot_r, slot_c) not in matching:
                    connectors.add((slot_r, slot_c))
    return connectors


def _matching_layouts() -> dict[int, list[tuple[frozenset[tuple[int, int]], tuple[int, int, int], frozenset[tuple[int, int]]]]]:
    layouts: dict[int, list[tuple[frozenset[tuple[int, int]], tuple[int, int, int], frozenset[tuple[int, int]]]]] = {
        0: [],
        1: [],
        2: [],
    }
    for count in range(3, 7):
        for combo in itertools.combinations(SLOT_POSITIONS, count):
            matching = frozenset(combo)
            row_counts = tuple(sum((row, col) in matching for col in range(3)) for row in range(3))
            if len(set(row_counts)) != 3:
                continue
            if sum(v > 0 for v in row_counts) < 2:
                continue
            connectors = frozenset(_connector_slots(set(matching)))
            if not connectors:
                continue
            winner = int(np.argmax(row_counts))
            if any(row != winner for row, _ in matching):
                layouts[winner].append((matching, row_counts, connectors))
    return layouts


MATCH_LAYOUTS = _matching_layouts()


def _parse_structure(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == GRID_H, f"expected height {GRID_H}, got {h}"

    full_grey_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(full_grey_cols) == 3, f"expected 3 full grey columns, got {full_grey_cols}"
    assert full_grey_cols[1] - full_grey_cols[0] == STRIDE
    assert full_grey_cols[2] - full_grey_cols[1] == STRIDE
    divider_col = int(full_grey_cols[2])
    assert divider_col == LEFT_W, f"expected divider at col {LEFT_W}, got {divider_col}"

    left = g[:, :divider_col]
    grey_rows = [r for r in range(h) if np.all(left[r, :] == GREY)]
    assert grey_rows == [3, 7], f"expected row separators [3, 7], got {grey_rows}"
    return divider_col, grey_rows


def _slot_block(grid: np.ndarray, slot_r: int, slot_c: int) -> np.ndarray:
    r0, r1 = ROW_BANDS[slot_r]
    c0, c1 = COL_BANDS[slot_c]
    return grid[r0:r1, c0:c1]


def _extract_slots(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], dict[str, object]]]:
    divider_col, _ = _parse_structure(grid)
    slots: dict[tuple[int, int], dict[str, object]] = {}
    prototype_slot: tuple[int, int] | None = None

    for slot_r, slot_c in SLOT_POSITIONS:
        block = _slot_block(grid, slot_r, slot_c)
        mask = block != BLACK
        assert np.any(mask), "every slot must contain a symbol"
        colors = {int(v) for v in block[mask]}
        assert len(colors) == 1, "each slot symbol must be monochromatic"
        color = colors.pop()
        slots[(slot_r, slot_c)] = {"mask": mask.copy(), "color": color}
        if color == TEAL:
            assert prototype_slot is None, "expected exactly one teal prototype slot"
            prototype_slot = (slot_r, slot_c)

    assert prototype_slot is not None, "missing teal prototype slot"
    return divider_col, slots


def _matching_from_slots(slots: dict[tuple[int, int], dict[str, object]]) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    prototype_slot = next(pos for pos, info in slots.items() if int(info["color"]) == TEAL)
    prototype_mask = np.asarray(slots[prototype_slot]["mask"], dtype=bool)
    matching = {
        pos for pos, info in slots.items() if np.array_equal(np.asarray(info["mask"], dtype=bool), prototype_mask)
    }
    return prototype_slot, matching


def _row_match_counts(matching: set[tuple[int, int]]) -> tuple[int, int, int]:
    return tuple(sum((row, col) in matching for col in range(3)) for row in range(3))


def _compress_zone(zone: np.ndarray, direction: str) -> np.ndarray:
    mask = np.asarray(zone, dtype=int) != BLACK
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    if direction == "up":
        counts = mask.sum(axis=0)
        for c, count in enumerate(counts):
            out[: int(count), c] = TEAL
    elif direction == "right":
        counts = mask.sum(axis=1)
        for r, count in enumerate(counts):
            out[r, w - int(count) : w] = TEAL
    else:
        assert direction == "down"
        counts = mask.sum(axis=0)
        for c, count in enumerate(counts):
            out[h - int(count) : h, c] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the teal prototype across the left symbol table, use the winning match row to choose a right-side compression direction, then recolor the compressed result teal.

    Input structure:
    - The grid has black background, a 3x3 left symbol table of occupied 3x3 slots, grey internal separators, and one full-height grey divider column before the right region.
    - Exactly one left-table slot is teal; its 3x3 occupancy mask is the prototype, while every slot contains a monochromatic symbol on black.
    - The right region contains scattered non-black pixels of arbitrary colors.

    Rule:
    - Compare every non-prototype slot's 3x3 occupancy mask with the teal prototype mask, ignoring color; exact matches are recolored teal.
    - In each table row and column, recolor every occupied slot strictly between consecutive teal matching slots orange.
    - Count teal matching slots in each table row; the unique row with the largest count selects the right-side compression: top row -> compress each right-region column upward, middle row -> compress each right-region row rightward, bottom row -> compress each right-region column downward.
    - After that compression, recolor every non-black pixel in the right region teal while leaving the left table separators and background unchanged.

    Key insight:
    - The left table is both a visible prototype-matching puzzle and a hidden direction code: only the distribution of matching slots by row determines how the unrelated right-side pixels are geometrically packed.
    """
    g = np.asarray(grid, dtype=int)
    divider_col, slots = _extract_slots(g)
    prototype_slot, matching = _matching_from_slots(slots)
    connectors = _connector_slots(matching)
    counts = _row_match_counts(matching)
    assert len(set(counts)) == 3, f"expected distinct row match counts, got {counts}"
    winner_row = int(np.argmax(counts))

    out = g.copy()
    for slot_r, slot_c in connectors:
        block = _slot_block(out, slot_r, slot_c)
        block[block != BLACK] = ORANGE
    for slot_r, slot_c in matching:
        block = _slot_block(out, slot_r, slot_c)
        block[block != BLACK] = TEAL

    direction = {0: "up", 1: "right", 2: "down"}[winner_row]
    right_zone = g[:, divider_col + 1 :]
    out[:, divider_col + 1 :] = _compress_zone(right_zone, direction)

    assert prototype_slot in matching
    return out


def _hamming_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.count_nonzero(mask_a != mask_b))


def _choose_masks() -> tuple[np.ndarray, list[np.ndarray]]:
    prototype = random.choice(ALL_MASKS)
    candidates = ALL_MASKS[:]
    random.shuffle(candidates)
    chosen: list[np.ndarray] = []
    for mask in candidates:
        if np.array_equal(mask, prototype):
            continue
        if _hamming_distance(mask, prototype) < 2:
            continue
        if any(_hamming_distance(mask, prev) < 2 for prev in chosen):
            continue
        chosen.append(mask)
        if len(chosen) == 6:
            break
    assert len(chosen) == 6
    return prototype.copy(), [m.copy() for m in chosen]


def _paint_slot(grid: np.ndarray, slot_r: int, slot_c: int, mask: np.ndarray, color: int) -> None:
    block = _slot_block(grid, slot_r, slot_c)
    block[mask] = int(color)


def _choose_layout() -> tuple[int, frozenset[tuple[int, int]], tuple[int, int, int], frozenset[tuple[int, int]], tuple[int, int]]:
    winner_row = random.choice([0, 1, 2])
    matching, row_counts, connectors = random.choice(MATCH_LAYOUTS[winner_row])
    prototype_choices = [pos for pos in matching if pos[0] != winner_row]
    prototype_slot = random.choice(prototype_choices)
    return winner_row, matching, row_counts, connectors, prototype_slot


def _rotated_shape(shape: np.ndarray) -> np.ndarray:
    return np.rot90(shape, k=random.randint(0, 3))


def _can_place_shape(occupied: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    sh, sw = shape.shape
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False
    sub = occupied[r0 : r0 + sh, c0 : c0 + sw]
    if np.any(sub & shape):
        return False
    return True


def _make_right_zone(width: int) -> np.ndarray | None:
    zone = np.zeros((GRID_H, width), dtype=int)
    occupied = np.zeros_like(zone, dtype=bool)
    area = GRID_H * width
    min_pixels = max(11, int(np.ceil(area * RIGHT_OCCUPANCY_RANGE[0])))
    max_pixels = int(np.floor(area * RIGHT_OCCUPANCY_RANGE[1]))
    target = random.randint(min_pixels, max_pixels)
    palette = random.sample(RIGHT_INPUT_COLORS, k=random.randint(2, min(4, len(RIGHT_INPUT_COLORS))))

    total = 0
    component_count = random.randint(4, 8)
    for comp_i in range(component_count):
        remaining_components = component_count - comp_i
        remaining_area = target - total
        if remaining_area <= 0:
            break
        random_shapes = _RIGHT_SHAPES[:]
        random.shuffle(random_shapes)
        placed = False
        for base_shape in random_shapes:
            shape = _rotated_shape(base_shape)
            shape_area = int(shape.sum())
            if total + shape_area > max_pixels:
                continue
            if remaining_area > shape_area * remaining_components + 8:
                continue
            for _ in range(20):
                r0 = random.randint(0, GRID_H - shape.shape[0])
                c0 = random.randint(0, width - shape.shape[1])
                if not _can_place_shape(occupied, shape, r0, c0):
                    continue
                color = random.choice(palette)
                zone[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = color
                occupied[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] |= shape
                total += shape_area
                placed = True
                break
            if placed:
                break
        if not placed and total >= min_pixels:
            break
    if not (min_pixels <= total <= max_pixels):
        return None

    mask = zone != BLACK
    if len({int(v) for v in zone[mask]}) < 2:
        return None
    if np.max(mask.sum(axis=0)) < 2:
        return None
    if np.max(mask.sum(axis=1)) < 2:
        return None

    up = _compress_zone(zone, "up")
    right = _compress_zone(zone, "right")
    down = _compress_zone(zone, "down")
    if np.array_equal(up, right) or np.array_equal(up, down) or np.array_equal(right, down):
        return None
    if np.array_equal(mask.astype(int) * TEAL, up):
        return None
    if np.array_equal(mask.astype(int) * TEAL, right):
        return None
    if np.array_equal(mask.astype(int) * TEAL, down):
        return None
    return zone


def _render_input(
    right_zone: np.ndarray,
    matching: frozenset[tuple[int, int]],
    prototype_slot: tuple[int, int],
    prototype_mask: np.ndarray,
    other_masks: list[np.ndarray],
) -> np.ndarray:
    width = LEFT_W + 1 + right_zone.shape[1]
    grid = np.zeros((GRID_H, width), dtype=int)
    grid[:, 3] = GREY
    grid[:, 7] = GREY
    grid[3, :LEFT_W] = GREY
    grid[7, :LEFT_W] = GREY
    grid[:, LEFT_W] = GREY

    slots = SLOT_POSITIONS[:]
    remaining_masks = other_masks[:]
    random.shuffle(remaining_masks)

    color_assignment = {}
    other_colors = [c for c in INPUT_SLOT_COLORS if c != TEAL]
    random.shuffle(other_colors)
    for pos, color in zip([p for p in slots if p != prototype_slot], other_colors):
        color_assignment[pos] = color
    color_assignment[prototype_slot] = TEAL

    for pos in slots:
        if pos in matching:
            mask = prototype_mask
        else:
            mask = remaining_masks.pop()
        _paint_slot(grid, pos[0], pos[1], mask, color_assignment[pos])

    grid[:, LEFT_W + 1 :] = right_zone
    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        divider_col, slots = _extract_slots(inp)
        prototype_slot, matching = _matching_from_slots(slots)
        connectors = _connector_slots(matching)
        counts = _row_match_counts(matching)
        if len(set(counts)) != 3:
            return False
        winner = int(np.argmax(counts))
        if prototype_slot[0] == winner:
            return False
        if len(connectors) < 1:
            return False
        if sum(pos in matching for pos in SLOT_POSITIONS) < 3:
            return False
        if sum(count > 0 for count in counts) < 2:
            return False

        right_zone = inp[:, divider_col + 1 :]
        area = right_zone.size
        occ = int(np.count_nonzero(right_zone))
        if not (int(np.ceil(area * RIGHT_OCCUPANCY_RANGE[0])) <= occ <= int(np.floor(area * RIGHT_OCCUPANCY_RANGE[1]))):
            return False

        up = _compress_zone(right_zone, "up")
        right = _compress_zone(right_zone, "right")
        down = _compress_zone(right_zone, "down")
        if np.array_equal(up, right) or np.array_equal(up, down) or np.array_equal(right, down):
            return False

        outputs = {0: up, 1: right, 2: down}
        inp_mask_teal = (right_zone != BLACK).astype(int) * TEAL
        if np.array_equal(outputs[winner], inp_mask_teal):
            return False

        out = _expected_output(inp)
        if not np.array_equal(out[:, divider_col + 1 :], outputs[winner]):
            return False

        for slot_r, slot_c in matching:
            block = _slot_block(out, slot_r, slot_c)
            if np.any(block[block != BLACK] != TEAL):
                return False
        for slot_r, slot_c in connectors:
            if (slot_r, slot_c) in matching:
                continue
            block = _slot_block(out, slot_r, slot_c)
            if np.any(block[block != BLACK] != ORANGE):
                return False

        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    right_width = random.randint(*RIGHT_W_RANGE)
    right_zone = _make_right_zone(right_width)
    if right_zone is None:
        return None

    winner, matching, _counts, _connectors, prototype_slot = _choose_layout()
    if prototype_slot[0] == winner:
        return None

    prototype_mask, other_masks = _choose_masks()
    inp = _render_input(right_zone, matching, prototype_slot, prototype_mask, other_masks)
    if _semantic_checks(inp):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
