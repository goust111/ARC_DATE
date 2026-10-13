from __future__ import annotations

import random

import numpy as np


def _checkerboard(height: int, width: int, color_even: int, color_odd: int, phase: int) -> np.ndarray:
    parity = (np.add.outer(np.arange(height), np.arange(width)) + int(phase)) % 2
    return np.where(parity == 0, int(color_even), int(color_odd)).astype(int)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _candidate_output(
    board_area: np.ndarray,
    restored_board: np.ndarray,
    record_bg: int,
    board_colors: set[int],
) -> np.ndarray | None:
    mismatch = board_area != restored_board
    record_row = np.full(board_area.shape[1], int(record_bg), dtype=int)

    for comp in _components(mismatch):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(board_area[comp[0][0], comp[0][1]])

        if color == int(record_bg) or color in board_colors:
            return None
        if any(int(board_area[r, c]) != color for r, c in comp):
            return None
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        if record_row[c0] != int(record_bg):
            return None

        record_row[c0] = color

    out = np.empty((board_area.shape[0] + 1, board_area.shape[1]), dtype=int)
    out[:-1] = restored_board
    out[-1] = record_row
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove rectangular occluders from a checkerboard and drop their colors onto the bottom record row.

    Input structure:
    - The last row is a uniform black record row, while every row above it belongs to a global two-color checkerboard.
    - Several non-overlapping solid-color rectangles sit on top of the checkerboard; each rectangle uses a color different from the checkerboard colors and from black.
    - Exposed checkerboard cells keep one of the two alternating colors, so each board color only appears on one parity class of positions.

    Rule:
    - Infer the two checkerboard colors and their global phase from the exposed cells.
    - Replace every occluded cell by the checkerboard color dictated by its row+column parity.
    - For each removed rectangle, place one pixel of that rectangle's color in the last row at the rectangle's leftmost column.

    Key insight:
    - The true board colors are the pair whose visible cells already respect one fixed parity assignment; every other color must therefore belong to a solid rectangular occluder that can be erased and logged on the bottom edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    h, w = g.shape
    record_bg = int(g[-1, 0])
    assert np.all(g[-1] == record_bg)
    board_area = g[:-1]

    colors = [int(color) for color in np.unique(board_area) if int(color) != record_bg]
    assert len(colors) >= 2

    best_output: np.ndarray | None = None
    best_support = -1

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            for phase in (0, 1):
                restored_board = _checkerboard(h - 1, w, color_a, color_b, phase)
                if np.any(restored_board[board_area == color_a] != color_a):
                    continue
                if np.any(restored_board[board_area == color_b] != color_b):
                    continue

                candidate = _candidate_output(
                    board_area,
                    restored_board,
                    record_bg,
                    {int(color_a), int(color_b)},
                )
                if candidate is None:
                    continue

                support = int(np.count_nonzero((board_area == color_a) | (board_area == color_b)))
                if support > best_support:
                    best_output = candidate
                    best_support = support
                elif support == best_support and best_output is not None and not np.array_equal(candidate, best_output):
                    raise AssertionError("ambiguous checkerboard interpretation")

    assert best_output is not None, "failed to infer checkerboard colors and occluders"
    return best_output


BACKGROUND = 0
PALETTE = tuple(range(10))
GRID_HEIGHT_RANGE = (8, 24)
GRID_WIDTH_RANGE = (8, 24)
OCCLUDER_COUNT_RANGE = (1, 4)
OCCLUDER_COLOR_LIMIT = 7
VISIBLE_COLOR_MIN = 4
TOTAL_OCCLUSION_RATIO_CAP = 0.48
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _touches_or_overlaps(candidate: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in others:
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return True
    return False


def _sample_rect_size(board_h: int, width: int) -> tuple[int, int]:
    shape_kind = random.choice(("wide", "tall", "block"))
    if shape_kind == "wide":
        rect_h = random.randint(1, min(4, board_h))
        rect_w = random.randint(2, min(8, width))
    elif shape_kind == "tall":
        rect_h = random.randint(2, min(8, board_h))
        rect_w = random.randint(1, min(4, width))
    else:
        rect_h = random.randint(2, min(6, board_h))
        rect_w = random.randint(2, min(6, width))

    if rect_h * rect_w < 2:
        rect_h = 2
    rect_h = min(rect_h, board_h)
    rect_w = min(rect_w, width)
    return int(rect_h), int(rect_w)


def _place_rectangles(board_h: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    used_left_cols: set[int] = set()

    for _ in range(PLACEMENT_ATTEMPTS):
        if len(placed) == count:
            return placed

        rect_h, rect_w = _sample_rect_size(board_h, width)
        if rect_h > board_h or rect_w > width:
            continue

        r0 = random.randint(0, board_h - rect_h)
        c0 = random.randint(0, width - rect_w)
        if c0 in used_left_cols:
            continue

        candidate = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _touches_or_overlaps(candidate, placed):
            continue

        placed.append(candidate)
        used_left_cols.add(c0)

    return None


def _render_case(
    board_h: int,
    width: int,
    board_colors: tuple[int, int],
    phase: int,
    rectangles: list[tuple[int, int, int, int]],
    occluder_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    color_even, color_odd = board_colors
    board = _checkerboard(board_h, width, color_even, color_odd, phase)

    inp = np.full((board_h + 1, width), BACKGROUND, dtype=int)
    out = np.full((board_h + 1, width), BACKGROUND, dtype=int)
    inp[:-1] = board
    out[:-1] = board

    for (r0, c0, r1, c1), color in zip(rectangles, occluder_colors, strict=False):
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        out[-1, c0] = int(color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    board_h = height - 1

    board_colors = tuple(random.sample([color for color in PALETTE if color != BACKGROUND], 2))
    phase = random.randint(0, 1)

    available_occluder_colors = [color for color in PALETTE if color not in board_colors and color != BACKGROUND]
    max_count = min(OCCLUDER_COUNT_RANGE[1], len(available_occluder_colors), OCCLUDER_COLOR_LIMIT, width)
    count = random.randint(OCCLUDER_COUNT_RANGE[0], max_count)
    rectangles = _place_rectangles(board_h, width, count)
    if rectangles is None:
        return None

    occluded_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rectangles)
    if occluded_area >= int(board_h * width * TOTAL_OCCLUSION_RATIO_CAP):
        return None

    occluder_colors = random.sample(available_occluder_colors, count)
    inp, out = _render_case(board_h, width, board_colors, phase, rectangles, occluder_colors)

    if int(np.count_nonzero(inp[:-1] == board_colors[0])) < VISIBLE_COLOR_MIN:
        return None
    if int(np.count_nonzero(inp[:-1] == board_colors[1])) < VISIBLE_COLOR_MIN:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid checkerboard-occluder input")
