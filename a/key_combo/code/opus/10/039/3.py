from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8

LEFT = 0
RIGHT = 1

PANEL_COLORS = {LEFT: BLUE, RIGHT: RED}
OPPOSITE_PANEL_COLORS = {LEFT: RED, RIGHT: BLUE}
SYMBOL_COLORS = (GREEN, YELLOW, MAGENTA, ORANGE)

TABLE_H = 11
TABLE_W = 23
SLOT_SIZE = 3
ROW_SEPARATORS = (3, 7)
COL_SEPARATORS = (3, 7, 11, 15, 19)
MAX_GENERATE_ATTEMPTS = 400


@dataclass(frozen=True)
class Slot:
    panel: int
    panel_color: int
    color: int | None
    mask: np.ndarray


@dataclass(frozen=True)
class LineSpec:
    midpoint: tuple[int, int]
    endpoints: tuple[tuple[int, int], tuple[int, int]]


def _bbox_nonblack(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected a non-empty puzzle canvas"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    prev = 0
    for sep in separators:
        bands.append((prev, sep))
        prev = sep + 1
    bands.append((prev, length))
    return bands


def _global_pos(panel: int, row: int, col: int) -> tuple[int, int]:
    return row, col + (0 if panel == LEFT else 3)


def _panel_of(slot_pos: tuple[int, int]) -> int:
    return LEFT if slot_pos[1] < 3 else RIGHT


def _line_specs(panel: int) -> list[LineSpec]:
    lines = []
    for row in range(3):
        lines.append(
            LineSpec(
                midpoint=_global_pos(panel, row, 1),
                endpoints=(_global_pos(panel, row, 0), _global_pos(panel, row, 2)),
            )
        )
    for col in range(3):
        lines.append(
            LineSpec(
                midpoint=_global_pos(panel, 1, col),
                endpoints=(_global_pos(panel, 0, col), _global_pos(panel, 2, col)),
            )
        )
    return lines


def _parse_structure(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], Slot], tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _bbox_nonblack(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert crop.shape == (TABLE_H, TABLE_W), f"expected {TABLE_H}x{TABLE_W} non-black table, got {crop.shape}"

    row_seps = [r for r in range(crop.shape[0]) if np.all(crop[r, :] == GRAY)]
    col_seps = [c for c in range(crop.shape[1]) if np.all(crop[:, c] == GRAY)]
    assert row_seps == list(ROW_SEPARATORS), f"unexpected row separators: {row_seps}"
    assert col_seps == list(COL_SEPARATORS), f"unexpected column separators: {col_seps}"

    row_bands = _bands_from_separators(crop.shape[0], row_seps)
    col_bands = _bands_from_separators(crop.shape[1], col_seps)
    assert len(row_bands) == 3 and all(b1 - b0 == SLOT_SIZE for b0, b1 in row_bands)
    assert len(col_bands) == 6 and all(b1 - b0 == SLOT_SIZE for b0, b1 in col_bands)

    slots: dict[tuple[int, int], Slot] = {}
    prototype_pos: tuple[int, int] | None = None
    prototype_mask: np.ndarray | None = None

    for sr, (rb0, rb1) in enumerate(row_bands):
        for sc, (cb0, cb1) in enumerate(col_bands):
            panel = LEFT if sc < 3 else RIGHT
            panel_color = PANEL_COLORS[panel]
            block = crop[rb0:rb1, cb0:cb1]
            allowed = set(np.unique(block).tolist())
            assert allowed <= {panel_color, CYAN, GREEN, YELLOW, MAGENTA, ORANGE}, "invalid slot colors"
            mask = block != panel_color
            color: int | None
            if not mask.any():
                color = None
            else:
                colors = set(np.unique(block[mask]).tolist())
                assert len(colors) == 1, "each symbol slot must be monochromatic"
                color = int(next(iter(colors)))
            slots[(sr, sc)] = Slot(panel=panel, panel_color=panel_color, color=color, mask=mask.copy())
            if color == CYAN:
                assert prototype_pos is None, "expected exactly one cyan prototype"
                prototype_pos = (sr, sc)
                prototype_mask = mask.copy()

    assert prototype_pos is not None and prototype_mask is not None, "missing cyan prototype"
    assert prototype_mask.any(), "prototype must occupy at least one pixel"
    return crop, (r0, c0, r1, c1), row_bands, col_bands, slots, prototype_pos, prototype_mask


def _matching_slots(
    slots: dict[tuple[int, int], Slot],
    prototype_pos: tuple[int, int],
    prototype_mask: np.ndarray,
) -> set[tuple[int, int]]:
    matches = set()
    for pos, slot in slots.items():
        if pos == prototype_pos or slot.color is None:
            continue
        if np.array_equal(slot.mask, prototype_mask):
            matches.add(pos)
    return matches


def _connector_slots(
    matches: set[tuple[int, int]],
    prototype_pos: tuple[int, int],
) -> set[tuple[int, int]]:
    connectors = set()
    for panel in (LEFT, RIGHT):
        offset = 0 if panel == LEFT else 3
        local_matches = {(r, c - offset) for r, c in matches if _panel_of((r, c)) == panel}

        for row in range(3):
            cols = sorted(c for r, c in local_matches if r == row)
            for left, right in zip(cols, cols[1:]):
                for col in range(left + 1, right):
                    pos = (row, col + offset)
                    if pos not in matches and pos != prototype_pos:
                        connectors.add(pos)

        for col in range(3):
            rows = sorted(r for r, c in local_matches if c == col)
            for top, bottom in zip(rows, rows[1:]):
                for row in range(top + 1, bottom):
                    pos = (row, col + offset)
                    if pos not in matches and pos != prototype_pos:
                        connectors.add(pos)

    return connectors


def _apply_transform(
    crop: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    slots: dict[tuple[int, int], Slot],
    matches: set[tuple[int, int]],
    connectors: set[tuple[int, int]],
) -> np.ndarray:
    out = crop.copy()

    for sr, sc in connectors:
        slot = slots[(sr, sc)]
        rb0, rb1 = row_bands[sr]
        cb0, cb1 = col_bands[sc]
        block = out[rb0:rb1, cb0:cb1]
        block[slot.mask] = slot.panel_color

    for sr, sc in matches:
        slot = slots[(sr, sc)]
        rb0, rb1 = row_bands[sr]
        cb0, cb1 = col_bands[sc]
        block = out[rb0:rb1, cb0:cb1]
        block[slot.mask] = OPPOSITE_PANEL_COLORS[slot.panel]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor prototype-matching symbols by opposite-panel color and recolor in-between symbols by their own panel color.

    Input structure:
    - A black background surrounds one 3x6 table of 3x3 symbol slots separated by single-pixel gray lines.
    - The middle gray separator splits the table into a left blue panel and a right red panel, each with 3x3 slots.
    - Exactly one occupied slot is cyan; its 3x3 occupancy mask is the prototype, while other occupied slots use green, yellow, magenta, or orange.

    Rule:
    - Read the cyan slot's 3x3 binary mask, treating colored pixels as foreground and the panel base color as empty.
    - Every non-cyan slot whose 3x3 mask exactly matches the prototype is recolored with the opposite panel's base color.
    - Within each panel, any non-matching slot strictly between consecutive matching slots in the same table row or column is recolored with that panel's own base color.
    - The cyan prototype itself, the gray separators, the panel backgrounds, and every non-targeted slot remain unchanged.

    Key insight:
    - Shape matching ignores symbol color, but the recolor depends on panel membership: the same matched mask turns red on the blue side and blue on the red side.
    """
    g = np.asarray(grid, dtype=int)
    crop, bbox, row_bands, col_bands, slots, prototype_pos, prototype_mask = _parse_structure(g)
    matches = _matching_slots(slots, prototype_pos, prototype_mask)
    connectors = _connector_slots(matches, prototype_pos)

    transformed = _apply_transform(crop, row_bands, col_bands, slots, matches, connectors)
    out = g.copy()
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _mask_from_bits(bits: int) -> np.ndarray:
    flat = [(bits >> idx) & 1 for idx in range(9)]
    return np.array(flat, dtype=bool).reshape(3, 3)


MASK_BITS_BY_AREA: dict[int, tuple[int, ...]] = {}
for _bits in range(1, 1 << 9):
    MASK_BITS_BY_AREA.setdefault(int(_mask_from_bits(_bits).sum()), []).append(_bits)
MASK_BITS_BY_AREA = {area: tuple(bits_list) for area, bits_list in MASK_BITS_BY_AREA.items()}


def _random_mask_bits(area_lo: int, area_hi: int, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    areas = [area for area in range(area_lo, area_hi + 1) if area in MASK_BITS_BY_AREA]
    candidates = [bits for area in areas for bits in MASK_BITS_BY_AREA[area] if bits not in exclude]
    assert candidates, "no mask candidates left"
    return random.choice(candidates)


def _build_canvas(top: int, bottom: int, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((TABLE_H + top + bottom, TABLE_W + left + right), BLACK, dtype=int)
    crop = grid[top : top + TABLE_H, left : left + TABLE_W]

    crop[:, :11] = BLUE
    crop[:, 12:] = RED
    crop[:, 11] = GRAY
    crop[list(ROW_SEPARATORS), :] = GRAY
    crop[:, list(COL_SEPARATORS)] = GRAY
    return grid, crop


def _paint_slot(crop: np.ndarray, pos: tuple[int, int], mask: np.ndarray, color: int) -> None:
    sr, sc = pos
    r0 = sr * 4
    c0 = sc * 4
    block = crop[r0 : r0 + 3, c0 : c0 + 3]
    block[mask] = color


def _slot_positions(panel: int) -> list[tuple[int, int]]:
    return [_global_pos(panel, r, c) for r in range(3) for c in range(3)]


def _build_plan() -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], bool]:
    matches: set[tuple[int, int]] = set()
    forced_nonmatch: set[tuple[int, int]] = set()

    visible_panel = random.choice((LEFT, RIGHT))
    visible_line = random.choice(_line_specs(visible_panel))
    matches.update(visible_line.endpoints)
    forced_nonmatch.add(visible_line.midpoint)

    prototype_panel = random.choice((LEFT, RIGHT))
    prototype_between = False
    prototype_pos: tuple[int, int] | None = None

    if random.random() < 0.35:
        candidates = [
            line
            for line in _line_specs(prototype_panel)
            if line.midpoint not in matches
            and line.midpoint not in forced_nonmatch
            and all(endpoint not in forced_nonmatch for endpoint in line.endpoints)
        ]
        if candidates:
            chosen = random.choice(candidates)
            matches.update(chosen.endpoints)
            prototype_pos = chosen.midpoint
            prototype_between = True

    other_panel = RIGHT if visible_panel == LEFT else LEFT
    if not any(_panel_of(pos) == other_panel for pos in matches):
        candidates = [
            pos
            for pos in _slot_positions(other_panel)
            if pos not in matches and pos not in forced_nonmatch and pos != prototype_pos
        ]
        if not candidates:
            raise AssertionError("could not place the required opposite-panel match")
        matches.add(random.choice(candidates))

    if prototype_pos is None:
        candidates = [
            pos
            for pos in _slot_positions(prototype_panel)
            if pos not in matches and pos not in forced_nonmatch
        ]
        if not candidates:
            raise AssertionError("could not place prototype")
        prototype_pos = random.choice(candidates)

    extra_candidates = [
        pos
        for pos in (_slot_positions(LEFT) + _slot_positions(RIGHT))
        if pos not in matches and pos not in forced_nonmatch and pos != prototype_pos
    ]
    random.shuffle(extra_candidates)
    for pos in extra_candidates:
        if random.random() < 0.22:
            matches.add(pos)

    if not any(_panel_of(pos) == LEFT for pos in matches):
        free_left = [pos for pos in _slot_positions(LEFT) if pos not in matches and pos not in forced_nonmatch and pos != prototype_pos]
        if not free_left:
            raise AssertionError("missing left-panel match")
        matches.add(random.choice(free_left))
    if not any(_panel_of(pos) == RIGHT for pos in matches):
        free_right = [pos for pos in _slot_positions(RIGHT) if pos not in matches and pos not in forced_nonmatch and pos != prototype_pos]
        if not free_right:
            raise AssertionError("missing right-panel match")
        matches.add(random.choice(free_right))

    return prototype_pos, matches, forced_nonmatch, prototype_between


def _expected_from_plan(
    grid: np.ndarray,
    slot_masks: dict[tuple[int, int], np.ndarray],
    prototype_pos: tuple[int, int],
    matches: set[tuple[int, int]],
) -> np.ndarray:
    crop, bbox, row_bands, col_bands, slots, parsed_prototype_pos, prototype_mask = _parse_structure(grid)
    assert parsed_prototype_pos == prototype_pos
    assert matches == _matching_slots(slots, prototype_pos, prototype_mask)
    connectors = _connector_slots(matches, prototype_pos)
    expected_crop = crop.copy()

    for pos, mask in slot_masks.items():
        sr, sc = pos
        rb0, rb1 = row_bands[sr]
        cb0, cb1 = col_bands[sc]
        block = expected_crop[rb0:rb1, cb0:cb1]
        panel = _panel_of(pos)
        if pos in connectors and pos != prototype_pos:
            block[mask] = PANEL_COLORS[panel]
        if pos in matches:
            block[mask] = OPPOSITE_PANEL_COLORS[panel]

    expected = grid.copy()
    r0, c0, r1, c1 = bbox
    expected[r0 : r1 + 1, c0 : c1 + 1] = expected_crop
    return expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        try:
            top = random.randint(0, 3)
            bottom = random.randint(0, 3)
            left = random.randint(0, 3)
            right = random.randint(0, 3)
            if top == bottom == left == right == 0:
                top = 1

            prototype_bits = _random_mask_bits(3, 6)
            prototype_mask = _mask_from_bits(prototype_bits)
            other_mask_bits = set()
            while len(other_mask_bits) < random.randint(5, 8):
                other_mask_bits.add(_random_mask_bits(1, 8, exclude={prototype_bits} | other_mask_bits))

            prototype_pos, matches, forced_nonmatch, prototype_between = _build_plan()

            grid, crop = _build_canvas(top, bottom, left, right)
            slot_masks: dict[tuple[int, int], np.ndarray] = {}

            remaining = [
                pos
                for pos in (_slot_positions(LEFT) + _slot_positions(RIGHT))
                if pos not in matches and pos not in forced_nonmatch and pos != prototype_pos
            ]
            if not remaining:
                continue
            empty_positions: set[tuple[int, int]] = set()

            slot_masks[prototype_pos] = prototype_mask.copy()
            _paint_slot(crop, prototype_pos, prototype_mask, CYAN)

            for pos in matches:
                color = random.choice(SYMBOL_COLORS)
                slot_masks[pos] = prototype_mask.copy()
                _paint_slot(crop, pos, prototype_mask, color)

            for pos in forced_nonmatch:
                bits = random.choice(tuple(other_mask_bits))
                mask = _mask_from_bits(bits)
                color = random.choice(SYMBOL_COLORS)
                slot_masks[pos] = mask.copy()
                _paint_slot(crop, pos, mask, color)

            random.shuffle(remaining)
            for pos in remaining:
                if random.random() < 0.33:
                    empty_positions.add(pos)
                    slot_masks[pos] = np.zeros((3, 3), dtype=bool)
                    continue
                bits = random.choice(tuple(other_mask_bits))
                mask = _mask_from_bits(bits)
                color = random.choice(SYMBOL_COLORS)
                slot_masks[pos] = mask.copy()
                _paint_slot(crop, pos, mask, color)

            if not empty_positions:
                chosen_empty = random.choice(remaining)
                sr, sc = chosen_empty
                r0 = sr * 4
                c0 = sc * 4
                block = crop[r0 : r0 + 3, c0 : c0 + 3]
                block[:] = PANEL_COLORS[_panel_of(chosen_empty)]
                slot_masks[chosen_empty] = np.zeros((3, 3), dtype=bool)
                empty_positions.add(chosen_empty)

            expected = _expected_from_plan(grid, slot_masks, prototype_pos, matches)
            actual = main(grid)
            if not np.array_equal(actual, expected):
                continue

            parsed_crop, _bbox, _row_bands, _col_bands, slots, parsed_proto_pos, parsed_proto_mask = _parse_structure(grid)
            parsed_matches = _matching_slots(slots, parsed_proto_pos, parsed_proto_mask)
            parsed_connectors = _connector_slots(parsed_matches, parsed_proto_pos)

            if parsed_proto_pos != prototype_pos:
                continue
            if parsed_matches != matches:
                continue
            if sum(1 for pos in parsed_matches if _panel_of(pos) == LEFT) < 1:
                continue
            if sum(1 for pos in parsed_matches if _panel_of(pos) == RIGHT) < 1:
                continue
            visible_connectors = [pos for pos in parsed_connectors if slot_masks[pos].any()]
            if not visible_connectors:
                continue
            if prototype_between:
                between_kept = False
                for line in _line_specs(_panel_of(prototype_pos)):
                    if line.midpoint == prototype_pos and all(endpoint in matches for endpoint in line.endpoints):
                        sr, sc = prototype_pos
                        r0 = sr * 4
                        c0 = sc * 4
                        if np.array_equal(grid[r0 + top : r0 + top + 3, c0 + left : c0 + left + 3], actual[r0 + top : r0 + top + 3, c0 + left : c0 + left + 3]):
                            between_kept = True
                        break
                if not between_kept:
                    continue

            if np.array_equal(grid, actual):
                continue

            left_recolored = False
            right_recolored = False
            for pos in matches:
                sr, sc = pos
                r0 = top + sr * 4
                c0 = left + sc * 4
                block = actual[r0 : r0 + 3, c0 : c0 + 3]
                mask = slot_masks[pos]
                if _panel_of(pos) == LEFT and np.all(block[mask] == RED):
                    left_recolored = True
                if _panel_of(pos) == RIGHT and np.all(block[mask] == BLUE):
                    right_recolored = True
            if not left_recolored:
                continue
            if not right_recolored:
                continue

            _ = parsed_crop  # keep the structure parse exercised in generation-time validation
            return grid
        except AssertionError:
            continue

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
