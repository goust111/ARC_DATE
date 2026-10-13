import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
CONTROL_COLORS = (RED, BLUE, GREEN, YELLOW)

Square = tuple[int, int, int]  # (side, row, col)

GEN_ATTEMPTS = 400
SLOT_GAP = 2
SLOT_ROWS = 2
SLOT_COLS = 3
SLOT_SIDE_RANGE = (4, 7)


def _row_major(square: Square) -> tuple[int, int]:
    return square[1], square[2]


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _square_sum(prefix: np.ndarray, r: int, c: int, side: int) -> int:
    r1 = r + side
    c1 = c + side
    return int(prefix[r1, c1] - prefix[r, c1] - prefix[r1, c] + prefix[r, c])


def _enumerate_black_squares(main_grid: np.ndarray) -> list[Square]:
    h, w = main_grid.shape
    black = main_grid == BLACK
    prefix = _prefix_sum(black)
    candidates: list[Square] = []
    for side in range(2, min(h, w) + 1):
        area = side * side
        for r in range(h - side + 1):
            for c in range(w - side + 1):
                if _square_sum(prefix, r, c, side) == area:
                    candidates.append((side, r, c))
    candidates.sort(key=lambda square: (-square[0], square[1], square[2]))
    return candidates


def _select_non_overlapping(candidates: list[Square], shape: tuple[int, int]) -> list[Square]:
    occupied = np.zeros(shape, dtype=bool)
    selected: list[Square] = []
    for side, r, c in candidates:
        if occupied[r : r + side, c : c + side].any():
            continue
        occupied[r : r + side, c : c + side] = True
        selected.append((side, r, c))
    return selected


def _parse_control_row(control_row: np.ndarray) -> list[int]:
    sequence = [int(control_row[c]) for c in range(control_row.shape[0]) if int(control_row[c]) != BLACK]
    if not sequence:
        raise ValueError("control row must contain at least one non-black marker")
    return sequence


def _solve(input_grid: np.ndarray) -> tuple[np.ndarray, list[Square], list[Square], list[int]]:
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2 or grid.shape[0] < 2:
        raise ValueError("expected a 2D grid with a control row and a main region")

    color_sequence = _parse_control_row(grid[0])
    main_grid = grid[1:].copy()
    candidates = _enumerate_black_squares(main_grid)
    if not candidates:
        raise ValueError("main region must contain at least one all-black square of side >= 2")
    selected = _select_non_overlapping(candidates, main_grid.shape)

    output = main_grid.copy()
    for idx, (side, r, c) in enumerate(selected):
        output[r : r + side, c : c + side] = color_sequence[idx % len(color_sequence)]
    return output, candidates, selected, color_sequence


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill greedily selected black squares with colors read from the control row, cycling when needed.

    Input structure:
    - Row 0 is a black control row containing 2 to 4 colored marker pixels; their left-to-right order is the color sequence.
    - Rows 1 onward form the main region, where black cells are free space and grey cells are obstacles.
    - The main region contains many all-black square candidates of side at least 2, including overlapping ones.

    Rule:
    - Read the non-black control-row pixels from left to right to obtain a color sequence.
    - Enumerate every all-black square in the main region, sort them by larger side first and then by top-left row and column.
    - Greedily keep a square if it does not overlap any previously kept square.
    - Remove the control row and fill the kept squares in that order, cycling through the control-row colors.

    Key insight:
    - The control row gives an ordered color list, but the list is consumed only after a separate largest-first non-overlap selection over all blank squares.
    """
    output, _candidates, _selected, _sequence = _solve(input_grid)
    return output


def _bbox_overlap(a: Square, b: Square) -> int:
    sa, ra, ca = a
    sb, rb, cb = b
    r0 = max(ra, rb)
    c0 = max(ca, cb)
    r1 = min(ra + sa, rb + sb)
    c1 = min(ca + sa, cb + sb)
    if r0 >= r1 or c0 >= c1:
        return 0
    return (r1 - r0) * (c1 - c0)


def _contains(outer: Square, inner: Square) -> bool:
    so, ro, co = outer
    si, ri, ci = inner
    return ro <= ri and co <= ci and ri + si <= ro + so and ci + si <= co + so


def _strict_partial_overlap(a: Square, b: Square) -> bool:
    return _bbox_overlap(a, b) > 0 and not _contains(a, b) and not _contains(b, a)


def _expand_mask8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _slot_bboxes(row_heights: list[int], col_widths: list[int]) -> list[tuple[int, int, int, int]]:
    slots: list[tuple[int, int, int, int]] = []
    row_start = 0
    for rh in row_heights:
        col_start = 0
        for cw in col_widths:
            slots.append((row_start, col_start, rh, cw))
            col_start += cw + SLOT_GAP
        row_start += rh + SLOT_GAP
    return slots


def _choose_base_sides(slot_max_sides: list[int], count: int) -> list[int] | None:
    if count < 3:
        return None
    indices = list(range(count))
    duplicate_pairs = []
    for i in range(count):
        for j in range(i + 1, count):
            hi = min(slot_max_sides[i], slot_max_sides[j])
            if hi >= 2:
                duplicate_pairs.append((i, j, hi))
    if not duplicate_pairs:
        return None

    for _ in range(80):
        sides = [0] * count
        i, j, hi = random.choice(duplicate_pairs)
        dup_side = random.randint(2, hi)
        sides[i] = dup_side
        sides[j] = dup_side

        remaining = [k for k in indices if k not in (i, j)]
        if not remaining:
            return None
        random.shuffle(remaining)

        changed = False
        for k in remaining:
            options = [side for side in range(2, slot_max_sides[k] + 1) if side != dup_side]
            if options:
                sides[k] = random.choice(options)
                changed = True
                break
        if not changed:
            continue

        for k in remaining:
            if sides[k] == 0:
                sides[k] = random.randint(2, slot_max_sides[k])

        counts = Counter(sides)
        if len(counts) < 2:
            continue
        if max(counts.values()) < 2:
            continue
        return sides
    return None


def _place_base_square(slot: tuple[int, int, int, int], side: int) -> Square:
    r0, c0, h, w = slot
    r = random.randint(r0, r0 + h - side)
    c = random.randint(c0, c0 + w - side)
    return side, r, c


def _candidate_overlap_square(anchor: Square, slot: tuple[int, int, int, int]) -> Square | None:
    base_side, base_r, base_c = anchor
    slot_r, slot_c, slot_h, slot_w = slot
    max_side = min(slot_h, slot_w)
    candidates: list[Square] = []
    for side in range(2, max_side + 1):
        for r in range(slot_r, slot_r + slot_h - side + 1):
            for c in range(slot_c, slot_c + slot_w - side + 1):
                square = (side, r, c)
                if square == anchor:
                    continue
                if _strict_partial_overlap(square, anchor):
                    candidates.append(square)
    if not candidates:
        return None
    candidates.sort(
        key=lambda square: (
            abs(square[0] - base_side),
            abs(square[1] - base_r) + abs(square[2] - base_c),
            square[1],
            square[2],
        )
    )
    top_band = candidates[: min(12, len(candidates))]
    return random.choice(top_band)


def _paint_designed_region(
    shape: tuple[int, int], designed_squares: list[Square], parity: int
) -> np.ndarray:
    h, w = shape
    protected = np.zeros(shape, dtype=bool)
    for side, r, c in designed_squares:
        protected[r : r + side, c : c + side] = True

    moat = _expand_mask8(protected) & ~protected
    main_grid = np.zeros(shape, dtype=int)
    main_grid[moat] = GREY

    for r in range(h):
        for c in range(w):
            if protected[r, c] or moat[r, c]:
                continue
            if (r + c + parity) % 2 == 0:
                main_grid[r, c] = GREY
    return main_grid


def _has_size_order_conflict(selected: list[Square]) -> bool:
    for i, first in enumerate(selected):
        for second in selected[i + 1 :]:
            if first[0] > second[0] and _row_major(first) > _row_major(second):
                return True
            if second[0] > first[0] and _row_major(second) > _row_major(first):
                return True
    return False


def _has_partial_overlap_witness(candidates: list[Square], selected: list[Square]) -> bool:
    selected_set = set(selected)
    for candidate in candidates:
        if candidate in selected_set:
            continue
        for keep in selected:
            if _strict_partial_overlap(candidate, keep):
                return True
    return False


def _generate_attempt() -> np.ndarray | None:
    row_heights = [random.randint(*SLOT_SIDE_RANGE) for _ in range(SLOT_ROWS)]
    col_widths = [random.randint(*SLOT_SIDE_RANGE) for _ in range(SLOT_COLS)]
    slots = _slot_bboxes(row_heights, col_widths)
    shape = (
        sum(row_heights) + SLOT_GAP * (SLOT_ROWS - 1),
        sum(col_widths) + SLOT_GAP * (SLOT_COLS - 1),
    )

    slot_indices = list(range(len(slots)))
    occupied_count = random.choice((5, 6))
    occupied_slots = random.sample(slot_indices, occupied_count)
    slot_max_sides = [min(slots[idx][2], slots[idx][3]) for idx in occupied_slots]
    base_sides = _choose_base_sides(slot_max_sides, occupied_count)
    if base_sides is None:
        return None

    base_squares = [
        _place_base_square(slots[idx], side)
        for idx, side in zip(occupied_slots, base_sides)
    ]

    overlap_slot_candidates = [
        (slot_idx, square)
        for slot_idx, square in zip(occupied_slots, base_squares)
        if min(slots[slot_idx][2], slots[slot_idx][3]) >= 3 and square[0] >= 3
    ]
    if not overlap_slot_candidates:
        return None

    overlap_slot_idx, anchor_square = random.choice(overlap_slot_candidates)
    overlap_square = _candidate_overlap_square(anchor_square, slots[overlap_slot_idx])
    if overlap_square is None:
        return None

    designed_squares = base_squares + [overlap_square]
    if random.random() < 0.45:
        second_candidates = [
            (slot_idx, square)
            for slot_idx, square in overlap_slot_candidates
            if slot_idx != overlap_slot_idx
        ]
        if second_candidates:
            slot_idx, anchor_square = random.choice(second_candidates)
            second_overlap = _candidate_overlap_square(anchor_square, slots[slot_idx])
            if second_overlap is not None:
                designed_squares.append(second_overlap)

    main_grid = _paint_designed_region(shape, designed_squares, parity=random.randint(0, 1))

    marker_count = random.randint(2, 4)
    control_colors = random.sample(list(CONTROL_COLORS), marker_count)
    control_positions = sorted(random.sample(range(shape[1]), marker_count))
    control_row = np.zeros(shape[1], dtype=int)
    for pos, color in zip(control_positions, control_colors):
        control_row[pos] = color

    input_grid = np.vstack([control_row, main_grid])
    output_grid, candidates, selected, sequence = _solve(input_grid)
    selected_sizes = [square[0] for square in selected]

    if len(selected) < 3 or len(selected) <= len(sequence):
        return None
    if len(set(selected_sizes)) < 2:
        return None
    if max(Counter(selected_sizes).values()) < 2:
        return None
    if not _has_size_order_conflict(selected):
        return None
    if not _has_partial_overlap_witness(candidates, selected):
        return None
    if not any(_strict_partial_overlap(a, b) for i, a in enumerate(designed_squares) for b in designed_squares[i + 1 :]):
        return None
    if output_grid.shape != shape:
        return None
    if max(input_grid.shape) > 30 or max(output_grid.shape) > 30:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
