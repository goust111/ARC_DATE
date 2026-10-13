from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
TEAL = 8
COUNTER_COLORS = (1, 2, 3, 4, 7, 8, 9)
GENERATION_RETRY_LIMIT = 500


def _parse_slots(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Return divider columns and slot column intervals [c0, c1)."""
    h, w = grid.shape
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert divider_cols, "expected at least one full grey divider column"

    bounds = [-1, *divider_cols, w]
    slots: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        c0 = bounds[i] + 1
        c1 = bounds[i + 1]
        assert c0 < c1, "divider columns must separate non-empty slots"
        slots.append((c0, c1))

    widths = [c1 - c0 for c0, c1 in slots]
    assert len(set(widths)) == 1, "all slots must have the same width"
    assert widths[0] == h, "slots must be square"
    return divider_cols, slots


def _build_template_from_slot(slot: np.ndarray) -> np.ndarray:
    """Decode the horizontal teal-bar template from the first slot."""
    h, w = slot.shape
    assert h == w, "template slot must be square"

    markers = np.argwhere(slot == PINK)
    assert 1 < len(markers) <= h, "expected multiple marker rows"
    marker_rows = {int(r) for r, _c in markers}
    assert len(marker_rows) == len(markers), "each marker must lie on a distinct row"

    out = np.zeros_like(slot)
    for r, c in markers:
        r = int(r)
        c = int(c)
        row = slot[r]
        assert np.count_nonzero(row == PINK) == 1, "each marker row needs exactly one pink pixel"
        left_count = int(np.count_nonzero((row[:c] != BLACK) & (row[:c] != PINK)))
        right_count = int(np.count_nonzero((row[c + 1 :] != BLACK) & (row[c + 1 :] != PINK)))
        start = c - left_count
        end = c + right_count
        assert 0 <= start <= c <= end < w
        out[r, start : end + 1] = TEAL
    return out


def _render_slots(template: np.ndarray, shape: tuple[int, int], divider_cols: list[int], slots: list[tuple[int, int]]) -> np.ndarray:
    """Place the template and its successive clockwise rotations into all slots."""
    out = np.zeros(shape, dtype=int)
    for c in divider_cols:
        out[:, c] = GREY
    for i, (c0, c1) in enumerate(slots):
        out[:, c0:c1] = np.rot90(template, k=-i)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a teal horizontal-bar template from the first slot and rotate it through the later slots.

    Input structure:
    - The grid is split by full grey vertical divider columns into four equal square slots.
    - Only the first slot contains encoding pixels: black background, 2-3 pink control pixels on distinct rows, and scattered non-black non-pink counter pixels on those rows.
    - The remaining slots are empty black squares.

    Rule:
    - In the first slot, start from a blank black template of the same size.
    - For each pink control pixel at column c, count the non-black non-pink pixels strictly to its left and strictly to its right on the same row; draw a teal bar from c-left_count through c+right_count on that row.
    - Put that decoded template in the first slot, then fill the next slots with its 90, 180, and 270 degree clockwise rotations while keeping the grey divider columns unchanged.

    Key insight:
    - The visible pixels in the first slot are only an encoding: they disappear after their left/right counts reconstruct the hidden template that is then propagated by successive rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_cols, slots = _parse_slots(g)
    template = _build_template_from_slot(g[:, slots[0][0] : slots[0][1]])
    return _render_slots(template, g.shape, divider_cols, slots)


def _rotation_count(template: np.ndarray) -> int:
    return len({np.rot90(template, k=-k).tobytes() for k in range(4)})


def _side_positions(size: int, c: int, count: int, side: str) -> tuple[int, ...]:
    if count == 0:
        return ()

    if side == "left":
        cols = list(range(c))
        near = tuple(cols[-count:])
        bar_edge = c - count
    else:
        cols = list(range(c + 1, size))
        near = tuple(cols[:count])
        bar_edge = c + count

    combos = list(itertools.combinations(cols, count))

    def score(combo: tuple[int, ...]) -> tuple[int, int, int, int]:
        consecutive = int(all(b - a == 1 for a, b in zip(combo, combo[1:])))
        if side == "left":
            gap_to_marker = int(combo[-1] != c - 1)
            anti_span = int(combo[0] != bar_edge)
        else:
            gap_to_marker = int(combo[0] != c + 1)
            anti_span = int(combo[-1] != bar_edge)
        return (
            int(combo != near),
            anti_span,
            1 - consecutive,
            gap_to_marker,
        )

    best = max(score(combo) for combo in combos)
    preferred = [combo for combo in combos if score(combo) == best]
    return random.choice(preferred)


def _row_options(size: int) -> list[tuple[int, int, int]]:
    half = size // 2
    options: list[tuple[int, int, int]] = []
    for c in range(size):
        for left_count in range(min(c, half) + 1):
            for right_count in range(min(size - 1 - c, half) + 1):
                if left_count + right_count == 0:
                    continue
                options.append((c, left_count, right_count))
    return options


def _make_template(slot_size: int, specs: list[dict]) -> np.ndarray:
    template = np.zeros((slot_size, slot_size), dtype=int)
    for spec in specs:
        start = spec["col"] - spec["left_count"]
        end = spec["col"] + spec["right_count"]
        template[spec["row"], start : end + 1] = TEAL
    return template


def _color_stream():
    bag: list[int] = []
    while True:
        if not bag:
            bag = list(COUNTER_COLORS)
            random.shuffle(bag)
        yield bag.pop()


def _sample_specs(slot_size: int) -> list[dict]:
    rows = sorted(random.sample(range(slot_size), random.choice([2, 3])))
    options = _row_options(slot_size)
    specs: list[dict] = []
    for row in rows:
        col, left_count, right_count = random.choice(options)
        left_cols = _side_positions(slot_size, col, left_count, "left")
        right_cols = _side_positions(slot_size, col, right_count, "right")
        specs.append(
            {
                "row": int(row),
                "col": int(col),
                "left_count": int(left_count),
                "right_count": int(right_count),
                "left_cols": tuple(int(x) for x in left_cols),
                "right_cols": tuple(int(x) for x in right_cols),
            }
        )
    return specs


def _shortcut_flags(specs: list[dict]) -> tuple[bool, bool, bool]:
    has_asymmetry = False
    has_one_sided = False
    breaks_visible_span = False
    for spec in specs:
        col = spec["col"]
        left_count = spec["left_count"]
        right_count = spec["right_count"]
        visible = sorted((*spec["left_cols"], col, *spec["right_cols"]))
        start = col - left_count
        end = col + right_count
        has_asymmetry |= left_count != right_count
        has_one_sided |= (left_count == 0) ^ (right_count == 0)
        breaks_visible_span |= min(visible) != start or max(visible) != end
    return has_asymmetry, has_one_sided, breaks_visible_span


def _add_distractors(slot: np.ndarray, marker_rows: set[int], colors) -> None:
    free_rows = [r for r in range(slot.shape[0]) if r not in marker_rows]
    if not free_rows:
        return
    distractor_row_count = random.randint(0, min(2, len(free_rows)))
    for row in random.sample(free_rows, distractor_row_count):
        count = random.randint(1, max(1, min(2, slot.shape[1] - 1)))
        for col in random.sample(range(slot.shape[1]), count):
            slot[row, col] = next(colors)


def _build_input_from_specs(slot_size: int, specs: list[dict]) -> np.ndarray:
    width = slot_size * 4 + 3
    grid = np.zeros((slot_size, width), dtype=int)
    divider_cols = [slot_size, 2 * slot_size + 1, 3 * slot_size + 2]
    for c in divider_cols:
        grid[:, c] = GREY

    first_slot = grid[:, :slot_size]
    colors = _color_stream()
    for spec in specs:
        row = spec["row"]
        first_slot[row, spec["col"]] = PINK
        for col in spec["left_cols"]:
            first_slot[row, col] = next(colors)
        for col in spec["right_cols"]:
            first_slot[row, col] = next(colors)

    _add_distractors(first_slot, {spec["row"] for spec in specs}, colors)
    return grid


def _explicit_output(slot_size: int, specs: list[dict]) -> np.ndarray:
    template = _make_template(slot_size, specs)
    width = slot_size * 4 + 3
    divider_cols = [slot_size, 2 * slot_size + 1, 3 * slot_size + 2]
    slots = []
    start = 0
    for divider in divider_cols + [width]:
        slots.append((start, divider))
        start = divider + 1
    return _render_slots(template, (slot_size, width), divider_cols, slots)


def _validate_semantics(inp: np.ndarray, specs: list[dict]) -> bool:
    slot_size = inp.shape[0]
    divider_cols, slots = _parse_slots(inp)
    if len(slots) != 4:
        return False
    if divider_cols != [slot_size, 2 * slot_size + 1, 3 * slot_size + 2]:
        return False
    for c0, c1 in slots[1:]:
        if np.any(inp[:, c0:c1] != BLACK):
            return False

    first_slot = inp[:, slots[0][0] : slots[0][1]]
    for spec in specs:
        row = spec["row"]
        col = spec["col"]
        if first_slot[row, col] != PINK:
            return False
        if np.count_nonzero(first_slot[row] == PINK) != 1:
            return False
        left_count = int(np.count_nonzero((first_slot[row, :col] != BLACK) & (first_slot[row, :col] != PINK)))
        right_count = int(np.count_nonzero((first_slot[row, col + 1 :] != BLACK) & (first_slot[row, col + 1 :] != PINK)))
        if left_count != spec["left_count"] or right_count != spec["right_count"]:
            return False
        if left_count > slot_size // 2 or right_count > slot_size // 2:
            return False

    template = _build_template_from_slot(first_slot)
    if _rotation_count(template) != 4:
        return False

    has_asymmetry, want_one_sided, breaks_visible_span = _shortcut_flags(specs)
    if not has_asymmetry or not breaks_visible_span:
        return False
    if want_one_sided and sum((spec["left_count"] == 0) ^ (spec["right_count"] == 0) for spec in specs) == 0:
        return False

    expected = _explicit_output(slot_size, specs)
    if not np.array_equal(main(inp), expected):
        return False

    for i, (c0, c1) in enumerate(slots):
        if not np.array_equal(expected[:, c0:c1], np.rot90(template, k=-i)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    slot_size = random.choice([5, 6])
    require_one_sided = random.random() < 0.6
    for _ in range(80):
        specs = _sample_specs(slot_size)
        has_asymmetry, has_one_sided, breaks_visible_span = _shortcut_flags(specs)
        if not has_asymmetry or not breaks_visible_span:
            continue
        if require_one_sided and not has_one_sided:
            continue

        template = _make_template(slot_size, specs)
        if _rotation_count(template) != 4:
            continue

        inp = _build_input_from_specs(slot_size, specs)
        if _validate_semantics(inp, specs):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid for puzzle 7")
