from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
CONTROL_COLORS = (1, 2, 3, 4)
SLOT_SIZE = 3
SLOT_STRIDE = 4
GEN_MAX_ATTEMPTS = 400

CORNER_TO_ROT_K = {
    (0, 0): 0,
    (0, 2): -1,
    (2, 2): 2,
    (2, 0): 1,
}
CORNER_POSITIONS = tuple(CORNER_TO_ROT_K)
NON_CORNER_POSITIONS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
VALID_SLOT_DIMS = ((2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_components(mask)) == 1


def _build_template_catalog() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    cells = list(range(SLOT_SIZE * SLOT_SIZE))
    for area in range(3, 7):
        for combo in itertools.combinations(cells, area):
            mask = np.zeros((SLOT_SIZE, SLOT_SIZE), dtype=bool)
            for idx in combo:
                mask[idx // SLOT_SIZE, idx % SLOT_SIZE] = True
            if not _is_connected(mask):
                continue
            rr, cc = np.where(mask)
            if rr.min() != 0 or rr.max() != SLOT_SIZE - 1 or cc.min() != 0 or cc.max() != SLOT_SIZE - 1:
                continue
            rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
            if len(rotations) != 4:
                continue
            masks.append(mask)
    return tuple(masks)


TEMPLATE_CATALOG = _build_template_catalog()


def _parse(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], np.ndarray]]:
    comps = _components(grid != BG)
    template_candidates: list[np.ndarray] = []
    slot_boxes: dict[tuple[int, int], np.ndarray] = {}

    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if (h, w) != (SLOT_SIZE, SLOT_SIZE):
            continue
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(coords) < SLOT_SIZE * SLOT_SIZE:
            assert set(map(int, crop.ravel())).issubset({BG, GREY}), "template must use only black and grey"
            template_candidates.append(crop.copy())
        else:
            slot_boxes[(r0, c0)] = crop.copy()

    assert len(template_candidates) == 1, "expected one 3x3 grey template"
    assert slot_boxes, "expected a lower slot array"

    row_starts = sorted({r for r, _c in slot_boxes})
    col_starts = sorted({c for _r, c in slot_boxes})
    assert all(b - a == SLOT_STRIDE for a, b in zip(row_starts, row_starts[1:])), "slot rows must be evenly spaced"
    assert all(b - a == SLOT_STRIDE for a, b in zip(col_starts, col_starts[1:])), "slot cols must be evenly spaced"
    assert len(slot_boxes) == len(row_starts) * len(col_starts), "slots must form a full rectangle"

    template = template_candidates[0] == GREY
    assert np.any(template), "template must contain grey pixels"
    return template, row_starts, col_starts, slot_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the top grey 3x3 template inside each lower slot according to a corner control pixel, recolor it, and stamp it into the output array.

    Input structure:
    - The background is black.
    - The top area contains one grey 3x3 template mask whose filled cells form a connected, non-rotationally-symmetric shape.
    - The lower area is a full rectangular lattice of 3x3 grey slots separated by one black row and one black column.
    - A slot may contain one non-grey non-black control pixel replacing one grey cell.

    Rule:
    - Read the top 3x3 grey pattern as the template.
    - For each lower 3x3 slot, ignore empty slots and also ignore slots whose control pixel is on an edge midpoint or the center.
    - If a slot's control pixel is in a corner, rotate the template by that corner's slot-local orientation:
      top-left = 0 degrees, top-right = 90 degrees clockwise, bottom-right = 180 degrees, bottom-left = 270 degrees clockwise.
    - Recolor the rotated grey template with the control pixel's color, keep black as black, and write the result into the matching output slot.
    - Return only the lower slot-array region, preserving its slot spacing.

    Key insight:
    - The template shape, the rotation angle, and the output color come from three different cues: the top template, the control pixel's slot-local corner, and the control pixel's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template, row_starts, col_starts, slot_boxes = _parse(g)
    out = np.zeros((len(row_starts) * SLOT_STRIDE - 1, len(col_starts) * SLOT_STRIDE - 1), dtype=int)

    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            tile = slot_boxes[(r0, c0)]
            control_coords = np.argwhere((tile != BG) & (tile != GREY))
            assert len(control_coords) <= 1, "each slot may contain at most one control pixel"
            if len(control_coords) == 0:
                continue

            pr, pc = map(int, control_coords[0])
            if (pr, pc) not in CORNER_TO_ROT_K:
                continue

            color = int(tile[pr, pc])
            assert color in CONTROL_COLORS, "control color must be one of the allowed visible colors"
            stamp = np.rot90(template, CORNER_TO_ROT_K[(pr, pc)])
            rr = br * SLOT_STRIDE
            cc = bc * SLOT_STRIDE
            out[rr : rr + SLOT_SIZE, cc : cc + SLOT_SIZE] = np.where(stamp, color, BG)

    return out


def _oracle_output(
    template: np.ndarray, rows: int, cols: int, slot_specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]]
) -> np.ndarray:
    out = np.zeros((rows * SLOT_STRIDE - 1, cols * SLOT_STRIDE - 1), dtype=int)
    for sr in range(rows):
        for sc in range(cols):
            kind, pos, color = slot_specs[(sr, sc)]
            if kind != "corner":
                continue
            assert pos is not None and color is not None
            stamp = np.rot90(template, CORNER_TO_ROT_K[pos])
            rr = sr * SLOT_STRIDE
            cc = sc * SLOT_STRIDE
            out[rr : rr + SLOT_SIZE, cc : cc + SLOT_SIZE] = np.where(stamp, color, BG)
    return out


def _wrong_no_rotation(
    template: np.ndarray, rows: int, cols: int, slot_specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]]
) -> np.ndarray:
    out = np.zeros((rows * SLOT_STRIDE - 1, cols * SLOT_STRIDE - 1), dtype=int)
    for sr in range(rows):
        for sc in range(cols):
            kind, _pos, color = slot_specs[(sr, sc)]
            if kind != "corner":
                continue
            rr = sr * SLOT_STRIDE
            cc = sc * SLOT_STRIDE
            out[rr : rr + SLOT_SIZE, cc : cc + SLOT_SIZE] = np.where(template, int(color), BG)
    return out


def _wrong_any_control(
    template: np.ndarray, rows: int, cols: int, slot_specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]]
) -> np.ndarray:
    out = np.zeros((rows * SLOT_STRIDE - 1, cols * SLOT_STRIDE - 1), dtype=int)
    for sr in range(rows):
        for sc in range(cols):
            kind, _pos, color = slot_specs[(sr, sc)]
            if kind == "empty":
                continue
            rr = sr * SLOT_STRIDE
            cc = sc * SLOT_STRIDE
            out[rr : rr + SLOT_SIZE, cc : cc + SLOT_SIZE] = np.where(template, int(color), BG)
    return out


def _build_slot_specs(rows: int, cols: int) -> dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]]:
    total = rows * cols
    specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]] = {}
    slots = [(i // cols, i % cols) for i in random.sample(range(total), total)]

    mandatory_corner_slots = slots[:5]
    non_corner_slot = slots[5]
    empty_slot = slots[6]
    rest = slots[7:]

    orientations = list(CORNER_POSITIONS)
    random.shuffle(orientations)
    repeated_color = random.choice(CONTROL_COLORS)
    remaining_colors = [c for c in CONTROL_COLORS if c != repeated_color]
    random.shuffle(remaining_colors)
    first_four_colors = [repeated_color, remaining_colors[0], repeated_color, remaining_colors[1]]

    corner_color_by_orientation: dict[tuple[int, int], int] = {}
    for slot, orientation, color in zip(mandatory_corner_slots[:4], orientations, first_four_colors):
        specs[slot] = ("corner", orientation, color)
        corner_color_by_orientation[orientation] = color

    duplicate_orientation = random.choice(orientations)
    duplicate_color_choices = [c for c in CONTROL_COLORS if c != corner_color_by_orientation[duplicate_orientation]]
    specs[mandatory_corner_slots[4]] = ("corner", duplicate_orientation, random.choice(duplicate_color_choices))
    specs[non_corner_slot] = ("noncorner", random.choice(NON_CORNER_POSITIONS), random.choice(CONTROL_COLORS))
    specs[empty_slot] = ("empty", None, None)

    for slot in rest:
        roll = random.random()
        if roll < 0.25:
            specs[slot] = ("empty", None, None)
        elif roll < 0.50:
            specs[slot] = ("noncorner", random.choice(NON_CORNER_POSITIONS), random.choice(CONTROL_COLORS))
        else:
            specs[slot] = ("corner", random.choice(CORNER_POSITIONS), random.choice(CONTROL_COLORS))

    for sr in range(rows):
        for sc in range(cols):
            specs.setdefault((sr, sc), ("empty", None, None))
    return specs


def _render_input(
    template: np.ndarray, rows: int, cols: int, slot_specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]]
) -> np.ndarray:
    array_h = rows * SLOT_STRIDE - 1
    array_w = cols * SLOT_STRIDE - 1
    template_r0 = random.randint(1, 3)
    template_c0 = random.randint(0, 2)
    gap = random.randint(2, 4)
    slot_r0 = template_r0 + SLOT_SIZE + gap
    slot_c0 = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)

    height = slot_r0 + array_h + bottom_margin
    width = max(template_c0 + SLOT_SIZE, slot_c0 + array_w) + right_margin
    assert max(height, width) <= 30, "grid must stay within ARC limits"

    grid = np.zeros((height, width), dtype=int)
    grid[template_r0 : template_r0 + SLOT_SIZE, template_c0 : template_c0 + SLOT_SIZE] = np.where(template, GREY, BG)

    for sr in range(rows):
        for sc in range(cols):
            r0 = slot_r0 + sr * SLOT_STRIDE
            c0 = slot_c0 + sc * SLOT_STRIDE
            grid[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = GREY
            kind, pos, color = slot_specs[(sr, sc)]
            if kind == "empty":
                continue
            assert pos is not None and color is not None
            grid[r0 + pos[0], c0 + pos[1]] = int(color)
    return grid


def _semantic_checks(
    template: np.ndarray,
    rows: int,
    cols: int,
    slot_specs: dict[tuple[int, int], tuple[str, tuple[int, int] | None, int | None]],
    inp: np.ndarray,
    expected: np.ndarray,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    kinds = [kind for kind, _pos, _color in slot_specs.values()]
    if "corner" not in kinds or "noncorner" not in kinds or "empty" not in kinds:
        return False

    corner_specs = [(pos, color) for kind, pos, color in slot_specs.values() if kind == "corner"]
    if {pos for pos, _color in corner_specs} != set(CORNER_POSITIONS):
        return False

    colors_by_orientation: dict[tuple[int, int], set[int]] = {}
    orientations_by_color: dict[int, set[tuple[int, int]]] = {}
    for pos, color in corner_specs:
        assert pos is not None and color is not None
        colors_by_orientation.setdefault(pos, set()).add(int(color))
        orientations_by_color.setdefault(int(color), set()).add(pos)

    if not any(len(v) >= 2 for v in colors_by_orientation.values()):
        return False
    if not any(len(v) >= 2 for v in orientations_by_color.values()):
        return False
    if np.array_equal(expected, _wrong_no_rotation(template, rows, cols, slot_specs)):
        return False
    if np.array_equal(expected, _wrong_any_control(template, rows, cols, slot_specs)):
        return False
    if not np.any(expected != 0):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(TEMPLATE_CATALOG)
    rows, cols = random.choice(VALID_SLOT_DIMS)
    slot_specs = _build_slot_specs(rows, cols)
    expected = _oracle_output(template, rows, cols, slot_specs)
    inp = _render_input(template, rows, cols, slot_specs)

    if not _semantic_checks(template, rows, cols, slot_specs, inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid slot-rotation-stamp input")
