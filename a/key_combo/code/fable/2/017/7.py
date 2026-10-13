from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
TOTAL_CUP_RANGE = (2, 5)
CHAIN_LENGTH_CAP = 4

CUP_HEIGHT_RANGE = (4, 7)
CUP_WIDTH_RANGE = (5, 9)
MIN_BLANK_ROWS = 1
EDGE_MARGIN = 1

MAX_CHAIN_ATTEMPTS = 100
MAX_DISTRACTOR_ATTEMPTS = 120
MAX_GENERATE_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_ball(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    singletons = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
    assert len(singletons) == 1, "expected exactly one singleton ball color"
    ball_color = singletons[0]
    ball_pos = np.argwhere(grid == ball_color)
    assert ball_pos.shape == (1, 2)
    return int(ball_pos[0, 0]), int(ball_pos[0, 1]), ball_color


def _extract_cup(grid: np.ndarray, color: int, bg: int) -> dict[str, int] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 5:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[:, 0] = True
    border[:, -1] = True
    border[-1, :] = True
    actual = patch == color

    if np.any(actual & ~border):
        return None
    if np.any(patch[~border] != bg):
        return None

    missing = np.argwhere(border & ~actual)
    if missing.shape[0] == 0:
        gap_col = None
    elif missing.shape[0] == 1:
        mr, mc = map(int, missing[0])
        if mr != height - 1 or mc in (0, width - 1):
            return None
        gap_col = c0 + mc
    else:
        return None

    return {
        "color": int(color),
        "top": r0,
        "left": c0,
        "bottom": r1,
        "right": c1,
        "gap_col": -1 if gap_col is None else int(gap_col),
    }


def _extract_cups(grid: np.ndarray, bg: int, ball_color: int) -> list[dict[str, int]]:
    cups = []
    for color in sorted(set(map(int, grid.flatten())) - {bg, ball_color}):
        cup = _extract_cup(grid, int(color), bg)
        assert cup is not None, f"color {color} is not a valid cup"
        cups.append(cup)
    assert cups, "expected at least one cup"
    return cups


def _first_cup_below(cups: list[dict[str, int]], row: int, col: int) -> dict[str, int] | None:
    candidates = [
        cup
        for cup in cups
        if cup["top"] > row and cup["left"] < col < cup["right"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda cup: (cup["top"], cup["left"], cup["right"]))


def _final_ball_position(
    cups: list[dict[str, int]], start_row: int, start_col: int, height: int
) -> tuple[int, int]:
    row = int(start_row)
    col = int(start_col)
    for _ in range(len(cups) + 1):
        cup = _first_cup_below(cups, row, col)
        if cup is None:
            return height - 1, col
        if cup["gap_col"] < 0:
            return cup["bottom"] - 1, col
        row = cup["bottom"]
        col = cup["gap_col"]
    raise AssertionError("cup chain should be acyclic in row order")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the ball through a vertical relay of leaky cups and keep only its final resting place.

    Input structure:
    - The background is black.
    - Several monochrome U-shaped cups are scattered at different heights; each cup opens upward.
    - Exactly one single-pixel ball uses a color absent from all cups.

    Rule:
    - Erase the ball from its starting cell.
    - Repeatedly let it fall straight down into the first cup whose mouth spans its column.
    - Inside a cup, stop on the inner floor when the bottom is intact; otherwise slide horizontally to the unique bottom gap, leak out there, and continue falling from that gap column.
    - If no further cup is reached, the ball stops on the bottom row.

    Key insight:
    - The visible cups define an invisible pipe network: only the nearest cup under the current fall column matters, and each bottom gap rewires the next vertical descent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    ball_row, ball_col, ball_color = _find_ball(g, bg)
    cups = _extract_cups(g, bg, ball_color)
    final_row, final_col = _final_ball_position(cups, ball_row, ball_col, g.shape[0])

    out = g.copy()
    out[ball_row, ball_col] = bg
    out[final_row, final_col] = ball_color
    return out


def _box_clear(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return False
    return True


def _min_rows_after(remaining_cups: int) -> int:
    return remaining_cups * (CUP_HEIGHT_RANGE[0] + MIN_BLANK_ROWS) + 1


def _make_cup(color: int, top: int, left: int, height: int, width: int, gap_col: int | None) -> dict[str, int]:
    return {
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "bottom": int(top + height - 1),
        "right": int(left + width - 1),
        "gap_col": -1 if gap_col is None else int(gap_col),
    }


def _sample_chain(
    height: int, width: int, colors: list[int], chain_len: int, terminal_mode: str
) -> tuple[list[dict[str, int]], list[int], tuple[int, int]] | None:
    cups: list[dict[str, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    current_col = random.randint(EDGE_MARGIN + 2, width - EDGE_MARGIN - 3)
    path_cols = [current_col]

    for idx in range(chain_len):
        leaky = idx < chain_len - 1 or terminal_mode == "floor"
        remaining = chain_len - idx - 1
        placed = False

        for _ in range(MAX_CHAIN_ATTEMPTS):
            cup_h = random.randint(*CUP_HEIGHT_RANGE)
            cup_w = random.randint(*CUP_WIDTH_RANGE)

            left_lo = max(EDGE_MARGIN, current_col - (cup_w - 2))
            left_hi = min(current_col - 1, width - cup_w - EDGE_MARGIN)
            if left_lo > left_hi:
                continue

            top_lo = 2 if idx == 0 else cups[-1]["bottom"] + MIN_BLANK_ROWS + 1
            top_hi = height - cup_h - _min_rows_after(remaining)
            if top_lo > top_hi:
                continue

            left = random.randint(left_lo, left_hi)
            top = random.randint(top_lo, top_hi)
            box = (top, left, top + cup_h - 1, left + cup_w - 1)
            if not _box_clear(box, boxes):
                continue

            gap_col = None
            if leaky:
                choices = list(range(left + 1, left + cup_w - 1))
                if len(choices) > 1 and current_col in choices and random.random() < 0.8:
                    alt = [col for col in choices if col != current_col]
                    gap_col = random.choice(alt if alt else choices)
                else:
                    gap_col = random.choice(choices)

            cup = _make_cup(colors[idx], top, left, cup_h, cup_w, gap_col)
            cups.append(cup)
            boxes.append(box)
            if gap_col is not None:
                current_col = gap_col
                path_cols.append(current_col)
            placed = True
            break

        if not placed:
            return None

    ball_row = random.randint(0, cups[0]["top"] - 2)
    return cups, path_cols, (ball_row, path_cols[0])


def _sample_distractors(
    height: int,
    width: int,
    colors: list[int],
    used_boxes: list[tuple[int, int, int, int]],
    path_cols: list[int],
) -> list[dict[str, int]] | None:
    distractors: list[dict[str, int]] = []

    for color in colors:
        placed = False
        for _ in range(MAX_DISTRACTOR_ATTEMPTS):
            cup_h = random.randint(*CUP_HEIGHT_RANGE)
            cup_w = random.randint(*CUP_WIDTH_RANGE)
            left_hi = width - cup_w - EDGE_MARGIN
            top_hi = height - cup_h - 1
            if left_hi < EDGE_MARGIN or top_hi < 1:
                continue

            left = random.randint(EDGE_MARGIN, left_hi)
            top = random.randint(1, top_hi)
            right = left + cup_w - 1
            if any(left <= col <= right for col in path_cols):
                continue

            box = (top, left, top + cup_h - 1, right)
            if not _box_clear(box, used_boxes):
                continue

            gap_col = None
            if random.random() < 0.5:
                gap_col = random.choice(list(range(left + 1, right)))
            distractors.append(_make_cup(color, top, left, cup_h, cup_w, gap_col))
            used_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return distractors


def _draw_cup(grid: np.ndarray, cup: dict[str, int]) -> None:
    top = cup["top"]
    left = cup["left"]
    bottom = cup["bottom"]
    right = cup["right"]
    color = cup["color"]
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[bottom, left : right + 1] = color
    if cup["gap_col"] >= 0:
        grid[bottom, cup["gap_col"]] = BACKGROUND


def _render_input(
    height: int,
    width: int,
    cups: list[dict[str, int]],
    ball: tuple[int, int],
    ball_color: int,
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for cup in cups:
        _draw_cup(grid, cup)
    grid[ball[0], ball[1]] = int(ball_color)
    return grid


def _cup_boxes(cups: list[dict[str, int]]) -> list[tuple[int, int, int, int]]:
    return [(cup["top"], cup["left"], cup["bottom"], cup["right"]) for cup in cups]


def _spec_output(
    height: int,
    width: int,
    cups: list[dict[str, int]],
    ball: tuple[int, int],
    ball_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    grid = _render_input(height, width, cups, ball, ball_color)
    final_row, final_col = _final_ball_position(cups, ball[0], ball[1], height)
    out = grid.copy()
    out[ball[0], ball[1]] = BACKGROUND
    out[final_row, final_col] = int(ball_color)
    return out, (final_row, final_col)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    total_cups = random.randint(*TOTAL_CUP_RANGE)
    max_chain = min(CHAIN_LENGTH_CAP, total_cups, (height - 2) // 5)
    if max_chain < 1:
        return None

    chain_len = random.randint(1, max_chain)
    terminal_mode = random.choice(["intact", "floor"])
    colors = random.sample(PALETTE, total_cups + 1)
    ball_color = colors[-1]

    chain = _sample_chain(height, width, colors[:chain_len], chain_len, terminal_mode)
    if chain is None:
        return None
    chain_cups, path_cols, ball = chain
    boxes = _cup_boxes(chain_cups)

    distractor_colors = colors[chain_len:total_cups]
    distractors = _sample_distractors(height, width, distractor_colors, boxes, path_cols)
    if distractors is None:
        return None

    cups = chain_cups + distractors
    inp = _render_input(height, width, cups, ball, ball_color)
    expected, final_pos = _spec_output(height, width, cups, ball, ball_color)

    if final_pos == ball:
        return None
    if terminal_mode == "floor" and final_pos[0] != height - 1:
        return None
    if terminal_mode == "intact" and final_pos[0] == height - 1:
        return None
    if len(path_cols) > 1 and all(col == path_cols[0] for col in path_cols[1:]) and random.random() < 0.7:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid funnel-chain input")
