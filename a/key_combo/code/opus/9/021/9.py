from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 6, 7, 8, 9]
BAR_INTERIOR_HEIGHT = 3
BAR_INTERIOR_WIDTH = 5
FRAME_HEIGHT = BAR_INTERIOR_HEIGHT + 2
FRAME_WIDTH = BAR_INTERIOR_WIDTH + 2
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_grey_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey bar-chart frame"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    assert r1 - r0 + 1 == FRAME_HEIGHT
    assert c1 - c0 + 1 == FRAME_WIDTH

    frame_mask = np.zeros_like(grid, dtype=bool)
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == GREY, frame_mask), "grey pixels must form one thin rectangle"
    return r0, c0, r1, c1


def _decode_thresholds(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int]
) -> tuple[dict[int, int], list[int]]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape == (BAR_INTERIOR_HEIGHT, BAR_INTERIOR_WIDTH)

    thresholds: dict[int, int] = {}
    order: list[int] = []
    for row in inner:
        nz = np.flatnonzero(row != BG)
        assert len(nz) > 0, "each chart row must contain one colored bar"
        length = int(nz[-1] + 1)
        color = int(row[0])
        assert color not in (BG, YELLOW, GREY)
        assert np.all(row[:length] == color)
        assert np.all(row[length:] == BG)
        thresholds[color] = length
        order.append(color)

    assert len(order) == 3 and len(set(order)) == 3
    return thresholds, order


def _first_appearance_order(grid: np.ndarray, top_limit: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for r in range(top_limit):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BG, YELLOW, GREY) or color in seen:
                continue
            seen.add(color)
            order.append(color)
    return order


def _count_blocks(
    grid: np.ndarray, top_limit: int
) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    counts: dict[int, int] = {}
    blocks: dict[int, list[tuple[int, int]]] = {}
    for r in range(top_limit - 1):
        for c in range(grid.shape[1] - 1):
            color = int(grid[r, c])
            if color in (BG, YELLOW, GREY):
                continue
            if (
                color == int(grid[r + 1, c])
                and color == int(grid[r, c + 1])
                and color == int(grid[r + 1, c + 1])
            ):
                counts[color] = counts.get(color, 0) + 1
                blocks.setdefault(color, []).append((r, c))
    return counts, blocks


def _ring_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r + 1, c - 1),
        (r, c + 2),
        (r + 1, c + 2),
        (r + 2, c),
        (r + 2, c + 1),
    ]


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame = _find_grey_frame(g)
    thresholds, _ = _decode_thresholds(g, frame)
    counts, blocks = _count_blocks(g, frame[0])

    out = g.copy()
    for color, block_list in blocks.items():
        if counts.get(color, 0) <= thresholds.get(color, -1):
            continue
        for r, c in block_list:
            for rr, cc in _ring_cells(r, c):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    out[rr, cc] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the lower bar chart and yellow-outline only the colors whose 2x2-block counts exceed their thresholds.

    Input structure:
    - The grid has a black background, an upper object field, and a lower grey rectangular frame containing a 3-row bar chart.
    - The upper field contains solid 2x2 blocks in three foreground colors plus isolated single-pixel distractors in those same colors.
    - Each chart row is a left-aligned colored bar whose length encodes that color's threshold.

    Rule:
    - Locate the grey frame and read each interior row as one colored bar; the bar length is that color's threshold.
    - In the area above the frame, count every fully monochrome 2x2 window for each foreground color, ignoring singleton distractors.
    - For each color with count greater than its threshold, paint yellow onto all 4-neighbor cells surrounding each of that color's 2x2 blocks; otherwise leave that color unchanged. Keep the bar chart and all other pixels as they are unless a qualifying outline overwrites them.

    Key insight:
    - The lower chart is not the answer image: it is an input-side codebook of per-color thresholds that must be combined with an independent count of upper-area 2x2 blocks before any outlining decision can be made.
    """
    return _oracle_output(grid)


def _choose_relations(colors: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    counts: dict[int, int] = {}
    thresholds: dict[int, int] = {}
    relations = [">", "=", "<"]
    random.shuffle(relations)

    for color, relation in zip(colors, relations):
        if relation == ">":
            count = random.randint(2, 5)
            threshold = random.randint(1, count - 1)
        elif relation == "=":
            count = random.randint(1, 5)
            threshold = count
        else:
            count = random.randint(1, 4)
            threshold = random.randint(count + 1, 5)
        counts[color] = count
        thresholds[color] = threshold
    return counts, thresholds


def _slot_layout(total_blocks: int) -> tuple[int, int, int, int]:
    choices: list[tuple[int, int]] = []
    for slot_rows in range(2, 5):
        for slot_cols in range(3, 7):
            if slot_rows * slot_cols >= total_blocks:
                choices.append((slot_rows, slot_cols))
    assert choices
    slot_rows, slot_cols = random.choice(choices)
    upper_h = random.randint(max(10, 4 * slot_rows), min(18, 4 * slot_rows + 3))
    width = random.randint(max(14, 4 * slot_cols), min(24, 4 * slot_cols + 3))
    return slot_rows, slot_cols, upper_h, width


def _place_blocks(
    grid: np.ndarray,
    colors: list[int],
    counts: dict[int, int],
    upper_h: int,
    slot_rows: int,
    slot_cols: int,
) -> tuple[dict[int, list[tuple[int, int]]], set[tuple[int, int]]]:
    slots = [(1 + 4 * sr, 1 + 4 * sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    random.shuffle(slots)

    assignment = []
    for color in colors:
        assignment.extend([color] * counts[color])
    random.shuffle(assignment)

    blocks: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}
    protected: set[tuple[int, int]] = set()
    for color, (r, c) in zip(assignment, slots):
        grid[r : r + 2, c : c + 2] = color
        blocks[color].append((r, c))
        for rr in range(r - 1, r + 3):
            for cc in range(c - 1, c + 3):
                if 0 <= rr < upper_h and 0 <= cc < grid.shape[1]:
                    protected.add((rr, cc))
    return blocks, protected


def _place_distractors(
    grid: np.ndarray,
    colors: list[int],
    upper_h: int,
    protected: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid[:upper_h] != BG)}
    candidate_cells = [(r, c) for r in range(upper_h) for c in range(w) if (r, c) not in protected]
    random.shuffle(candidate_cells)

    total = random.randint(1, 5)
    if random.random() < 0.45:
        earliest = min(candidate_cells, default=None)
        if earliest is not None:
            total = max(total, 1)
            color = random.choice(colors)
            r, c = earliest
            if all(
                not (0 <= nr < upper_h and 0 <= nc < w and (nr, nc) in occupied)
                for nr in range(r - 1, r + 2)
                for nc in range(c - 1, c + 2)
            ):
                grid[r, c] = color
                occupied.add((r, c))
                total -= 1

    for _ in range(total):
        placed = False
        random.shuffle(candidate_cells)
        for r, c in candidate_cells:
            if grid[r, c] != BG:
                continue
            isolated = True
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < upper_h and 0 <= nc < w and (nr, nc) in occupied:
                        isolated = False
                        break
                if not isolated:
                    break
            if not isolated:
                continue
            grid[r, c] = random.choice(colors)
            occupied.add((r, c))
            placed = True
            break
        if not placed:
            return


def _build_chart(
    grid: np.ndarray,
    colors: list[int],
    thresholds: dict[int, int],
    upper_h: int,
    frame_top: int,
    frame_left: int,
) -> None:
    assert frame_top >= upper_h + 2
    assert frame_top + FRAME_HEIGHT <= grid.shape[0]
    assert 0 <= frame_left <= grid.shape[1] - FRAME_WIDTH

    grid[frame_top, frame_left : frame_left + FRAME_WIDTH] = GREY
    grid[frame_top + FRAME_HEIGHT - 1, frame_left : frame_left + FRAME_WIDTH] = GREY
    grid[frame_top : frame_top + FRAME_HEIGHT, frame_left] = GREY
    grid[frame_top : frame_top + FRAME_HEIGHT, frame_left + FRAME_WIDTH - 1] = GREY

    order = _first_appearance_order(grid, upper_h)
    assert set(order) == set(colors) and len(order) == 3
    for idx, color in enumerate(order):
        length = thresholds[color]
        row = frame_top + 1 + idx
        col0 = frame_left + 1
        grid[row, col0 : col0 + length] = color


def _summarize_relations(grid: np.ndarray) -> set[str]:
    frame = _find_grey_frame(grid)
    thresholds, order = _decode_thresholds(grid, frame)
    counts, blocks = _count_blocks(grid, frame[0])
    assert len(order) == 3
    assert set(order) == set(thresholds) == set(counts) == set(blocks)

    relations = set()
    for color in order:
        count = counts[color]
        threshold = thresholds[color]
        if count > threshold:
            relations.add(">")
        elif count == threshold:
            relations.add("=")
        else:
            relations.add("<")
    return relations


def _validate_instance(grid: np.ndarray) -> None:
    frame = _find_grey_frame(grid)
    thresholds, order = _decode_thresholds(grid, frame)
    assert order == _first_appearance_order(grid, frame[0])

    counts, blocks = _count_blocks(grid, frame[0])
    assert set(order) == set(thresholds) == set(counts) == set(blocks)
    assert _summarize_relations(grid) == {">", "=", "<"}

    block_cells = set()
    ring_cells = set()
    for color, block_list in blocks.items():
        for r, c in block_list:
            for rr in (r, r + 1):
                for cc in (c, c + 1):
                    block_cells.add((rr, cc))
            for rr, cc in _ring_cells(r, c):
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    ring_cells.add((rr, cc))

        for r, c in block_list:
            for rr, cc in _ring_cells(r, c):
                assert not (frame[0] <= rr <= frame[2] and frame[1] <= cc <= frame[3])

    upper = grid[: frame[0]]
    for r, c in np.argwhere(upper != BG):
        rr, cc = int(r), int(c)
        color = int(upper[rr, cc])
        if (rr, cc) in block_cells:
            continue
        assert color in thresholds
        assert (rr, cc) not in ring_cells
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < upper.shape[0] and 0 <= nc < upper.shape[1]:
                assert (nr, nc) not in block_cells

    expected = _oracle_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    changed = {(int(r), int(c)) for r, c in np.argwhere(actual != grid)}
    expected_changed = set()
    for color, block_list in blocks.items():
        if counts[color] <= thresholds[color]:
            continue
        for r, c in block_list:
            for rr, cc in _ring_cells(r, c):
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    expected_changed.add((rr, cc))
    assert changed == expected_changed


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(NON_SPECIAL_COLORS, 3)
    counts, thresholds = _choose_relations(colors)
    total_blocks = sum(counts.values())
    slot_rows, slot_cols, upper_h, width = _slot_layout(total_blocks)

    gap = random.randint(2, 3)
    bottom_margin = random.randint(0, 2)
    frame_top = upper_h + gap
    height = frame_top + FRAME_HEIGHT + bottom_margin
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    _, protected = _place_blocks(grid, colors, counts, upper_h, slot_rows, slot_cols)
    _place_distractors(grid, colors, upper_h, protected)
    frame_left = random.randint(0, width - FRAME_WIDTH)

    try:
        _build_chart(grid, colors, thresholds, upper_h, frame_top, frame_left)
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
