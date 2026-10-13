from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import itertools
import random


def _slot_bands(length: int) -> list[tuple[int, int]]:
    assert length % 4 == 1, "expected 3-cell slots separated by 1-cell grid lines"
    return [(start + 1, start + 4) for start in range(0, length - 1, 4)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor all copies of the teal prototype symbol and the straight symbol-slot runs between aligned copies.

    Input structure
    - The grid is a table of 3x3 symbol slots separated and bordered by one-pixel background lines.
    - Each occupied slot contains a single-color symbol mask.
    - Exactly one slot is teal; its 3x3 mask is the prototype.

    Rule
    - Every slot whose symbol mask matches the teal prototype is recolored teal.
    - In each table row or column, slots strictly between consecutive matching prototype slots are recolored orange.
    - The symbol masks and all background separator pixels are preserved.

    Key insight
    - The visible colors mark roles, but the matching relation is the 3x3 shape mask; orange marks the row/column intervals connecting repeated prototype masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    row_bands = _slot_bands(g.shape[0])
    col_bands = _slot_bands(g.shape[1])

    assert np.all(g[::4, :] == bg), "row separators must be background"
    assert np.all(g[:, ::4] == bg), "column separators must be background"

    slots: list[list[tuple[int, np.ndarray]]] = []
    prototype: tuple[int, int, np.ndarray] | None = None
    for slot_r, (r0, r1) in enumerate(row_bands):
        row: list[tuple[int, np.ndarray]] = []
        for slot_c, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            mask = block != bg
            assert np.any(mask), "expected every slot to contain a symbol"
            colors = set(map(int, block[mask]))
            assert len(colors) == 1, "each symbol slot must be monochromatic"
            color = colors.pop()
            row.append((color, mask))
            if color == Color.TEAL:
                assert prototype is None, "expected exactly one teal prototype slot"
                prototype = (slot_r, slot_c, mask.copy())
        slots.append(row)

    assert prototype is not None, "missing teal prototype slot"
    prototype_r, prototype_c, prototype_mask = prototype

    matching: set[tuple[int, int]] = set()
    for slot_r, row in enumerate(slots):
        for slot_c, (_color, mask) in enumerate(row):
            if np.array_equal(mask, prototype_mask):
                matching.add((slot_r, slot_c))

    assert (prototype_r, prototype_c) in matching
    assert len(matching) >= 2, "prototype mask must occur in multiple slots"

    connector_slots: set[tuple[int, int]] = set()
    for slot_r in range(len(row_bands)):
        matching_cols = sorted(slot_c for row, slot_c in matching if row == slot_r)
        for left, right in zip(matching_cols, matching_cols[1:]):
            for slot_c in range(left + 1, right):
                if (slot_r, slot_c) not in matching:
                    connector_slots.add((slot_r, slot_c))

    for slot_c in range(len(col_bands)):
        matching_rows = sorted(slot_r for slot_r, col in matching if col == slot_c)
        for top, bottom in zip(matching_rows, matching_rows[1:]):
            for slot_r in range(top + 1, bottom):
                if (slot_r, slot_c) not in matching:
                    connector_slots.add((slot_r, slot_c))

    out = g.copy()
    for slot_r, slot_c in connector_slots:
        r0, r1 = row_bands[slot_r]
        c0, c1 = col_bands[slot_c]
        block = out[r0:r1, c0:c1]
        block[block != bg] = Color.ORANGE

    for slot_r, slot_c in matching:
        r0, r1 = row_bands[slot_r]
        c0, c1 = col_bands[slot_c]
        block = out[r0:r1, c0:c1]
        block[block != bg] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Slot geometry and full canvas.
_GEN_SLOT_ROWS, _GEN_SLOT_COLS = 4, 6
_GEN_SLOT_SIZE, _GEN_SLOT_STRIDE, _GEN_SLOT_OFFSET = 3, 4, 1
_GEN_GRID_SHAPE = (17, 25)
_GEN_SLOT_POSITIONS = [(r, c) for r in range(_GEN_SLOT_ROWS) for c in range(_GEN_SLOT_COLS)]

# Shape count and mask area ranges.
_GEN_MASK_AREA_MIN, _GEN_MASK_AREA_MAX = 1, _GEN_SLOT_SIZE * _GEN_SLOT_SIZE
_GEN_PROTOTYPE_AREA_MIN, _GEN_PROTOTYPE_AREA_MAX = 4, 8
_GEN_SHAPE_COUNT_MIN, _GEN_SHAPE_COUNT_MAX = 4, 6

# Matching-copy and connector constraints.
_GEN_COPY_COUNT_MIN, _GEN_COPY_COUNT_MAX = 3, 6
_GEN_CONNECTOR_COUNT_MIN, _GEN_CONNECTOR_COUNT_MAX = 3, 10

# Generation retry budget.
_GEN_RETRY_LIMIT = 300

_GEN_BITS_BY_AREA: dict[int, list[int]] | None = None
_GEN_MATCHING_GROUPS: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] | None = None


def _gen_bits_by_area() -> dict[int, list[int]]:
    global _GEN_BITS_BY_AREA
    if _GEN_BITS_BY_AREA is None:
        groups = {area: [] for area in range(_GEN_MASK_AREA_MIN, _GEN_MASK_AREA_MAX + 1)}
        for bits in range(1, 1 << _GEN_MASK_AREA_MAX):
            groups[bits.bit_count()].append(bits)
        _GEN_BITS_BY_AREA = groups
    return _GEN_BITS_BY_AREA


def _gen_mask(bits: int) -> np.ndarray:
    vals = [(bits >> i) & 1 for i in range(_GEN_MASK_AREA_MAX)]
    return np.array(vals, dtype=bool).reshape(_GEN_SLOT_SIZE, _GEN_SLOT_SIZE)


def _gen_connector_slots(matching: set[tuple[int, int]]) -> set[tuple[int, int]]:
    connectors: set[tuple[int, int]] = set()
    for slot_r in range(_GEN_SLOT_ROWS):
        cols = sorted(slot_c for row, slot_c in matching if row == slot_r)
        for left, right in zip(cols, cols[1:]):
            for slot_c in range(left + 1, right):
                if (slot_r, slot_c) not in matching:
                    connectors.add((slot_r, slot_c))

    for slot_c in range(_GEN_SLOT_COLS):
        rows = sorted(slot_r for slot_r, col in matching if col == slot_c)
        for top, bottom in zip(rows, rows[1:]):
            for slot_r in range(top + 1, bottom):
                if (slot_r, slot_c) not in matching:
                    connectors.add((slot_r, slot_c))
    return connectors


def _gen_matching_groups() -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    global _GEN_MATCHING_GROUPS
    if _GEN_MATCHING_GROUPS is None:
        groups: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}
        for count in range(_GEN_COPY_COUNT_MIN, _GEN_COPY_COUNT_MAX + 1):
            for combo in itertools.combinations(_GEN_SLOT_POSITIONS, count):
                connector_count = len(_gen_connector_slots(set(combo)))
                if _GEN_CONNECTOR_COUNT_MIN <= connector_count <= _GEN_CONNECTOR_COUNT_MAX:
                    groups.setdefault((count, connector_count), []).append(combo)
        _GEN_MATCHING_GROUPS = groups
    return _GEN_MATCHING_GROUPS


def _gen_paint_slot(grid: np.ndarray, slot_r: int, slot_c: int, mask: np.ndarray, color: int) -> None:
    r0 = slot_r * _GEN_SLOT_STRIDE + _GEN_SLOT_OFFSET
    c0 = slot_c * _GEN_SLOT_STRIDE + _GEN_SLOT_OFFSET
    grid[r0 : r0 + _GEN_SLOT_SIZE, c0 : c0 + _GEN_SLOT_SIZE][mask] = color


def _gen_choose_masks() -> tuple[np.ndarray, list[np.ndarray]]:
    bits_by_area = _gen_bits_by_area()
    prototype_area = random.randint(_GEN_PROTOTYPE_AREA_MIN, _GEN_PROTOTYPE_AREA_MAX)
    prototype_bits = random.choice(bits_by_area[prototype_area])
    shape_count = random.randint(_GEN_SHAPE_COUNT_MIN, _GEN_SHAPE_COUNT_MAX)
    available = {
        area: [bits for bits in bits_by_area[area] if bits != prototype_bits]
        for area in range(_GEN_MASK_AREA_MIN, _GEN_MASK_AREA_MAX + 1)
    }
    rest_bits = []
    for _ in range(shape_count - 1):
        area = random.randint(_GEN_MASK_AREA_MIN, _GEN_MASK_AREA_MAX)
        if not available[area]:
            area = random.choice([a for a, bits in available.items() if bits])
        bit = random.choice(available[area])
        available[area].remove(bit)
        rest_bits.append(bit)
    return _gen_mask(prototype_bits), [_gen_mask(bits) for bits in rest_bits]


def _gen_choose_matching() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    groups = _gen_matching_groups()
    copy_count = random.randint(_GEN_COPY_COUNT_MIN, _GEN_COPY_COUNT_MAX)
    connector_counts = sorted(conn for count, conn in groups if count == copy_count)
    connector_count = random.choice(connector_counts)
    matching = set(random.choice(groups[(copy_count, connector_count)]))
    return matching, _gen_connector_slots(matching)


def _gen_assign_nonmatching_masks(nonmatching: list[tuple[int, int]], other_masks: list[np.ndarray]):
    mask_indices = list(range(len(other_masks)))
    assignments = mask_indices[:]
    assignments.extend(random.randrange(len(other_masks)) for _ in range(len(nonmatching) - len(assignments)))
    random.shuffle(assignments)
    return dict(zip(nonmatching, assignments))


def _gen_render_pair(prototype_mask, other_masks, matching, connectors, nonmatch_masks):
    output = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    input_grid = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    prototype_input_slot = random.choice(tuple(matching))

    for slot_r, slot_c in _GEN_SLOT_POSITIONS:
        pos = (slot_r, slot_c)
        if pos in matching:
            mask = prototype_mask
            out_color = Color.TEAL
            in_color = Color.TEAL if pos == prototype_input_slot else Color.BLUE
        else:
            mask = other_masks[nonmatch_masks[pos]]
            out_color = Color.ORANGE if pos in connectors else Color.BLUE
            in_color = Color.BLUE

        _gen_paint_slot(output, slot_r, slot_c, mask, out_color)
        _gen_paint_slot(input_grid, slot_r, slot_c, mask, in_color)
    return input_grid, output


def _gen_validated_input(input_grid: np.ndarray, output: np.ndarray) -> np.ndarray | None:
    try:
        if np.array_equal(main(input_grid), output):
            return input_grid
    except Exception:
        return None
    return None


def _gen_attempt() -> np.ndarray | None:
    prototype_mask, other_masks = _gen_choose_masks()
    matching, connectors = _gen_choose_matching()
    nonmatching = [pos for pos in _GEN_SLOT_POSITIONS if pos not in matching]
    nonmatch_masks = _gen_assign_nonmatching_masks(nonmatching, other_masks)
    input_grid, output = _gen_render_pair(prototype_mask, other_masks, matching, connectors, nonmatch_masks)
    return _gen_validated_input(input_grid, output)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d94c3b52 input")
