from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
BOARD_COLOR = 1
MOLD_COLOR = 2

MAX_ARC_SIZE = 30
GENERATION_RETRIES = 400

WIDTH_RANGE = (8, 24)
BOARD_HEIGHT_RANGE = (3, 8)
MOLD_HEIGHT_RANGE = (1, 6)
TOP_PADDING_RANGE = (0, 2)
GAP_RANGE = (1, 4)
BOTTOM_EXTRA_RANGE = (0, 5)

INTERNAL_CUT_PROBABILITY = 0.5


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _find_board(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if c0 == 0 and c1 == w - 1 and np.all(grid[r0 : r1 + 1, :] == int(color)):
            candidates.append((int(color), int(r0), int(r1)))
    assert len(candidates) == 1, "expected one full-width solid board"
    return candidates[0]


def _find_mold_and_bg(grid: np.ndarray, board_color: int, board_top: int) -> tuple[int, int]:
    mold_candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == board_color:
            continue
        coords = np.argwhere(grid == color)
        if coords.size and int(coords[:, 0].max()) < board_top:
            mold_candidates.append(color)
    assert len(mold_candidates) == 1, "expected one mold color above the board"
    mold_color = mold_candidates[0]
    bg_candidates = [int(color) for color in np.unique(grid) if int(color) not in {board_color, mold_color}]
    assert len(bg_candidates) == 1, "expected one background color"
    return mold_color, bg_candidates[0]


def _drop_components(components: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    fallen = np.zeros((h, w), dtype=bool)
    ordered = sorted(
        components,
        key=lambda coords: (
            -int(coords[:, 0].max()),
            -int(coords[:, 0].min()),
            int(coords[:, 1].min()),
        ),
    )
    for coords in ordered:
        shifted = coords.copy()
        while True:
            trial = shifted + np.array([1, 0], dtype=int)
            if int(trial[:, 0].max()) >= h:
                break
            if np.any(fallen[trial[:, 0], trial[:, 1]]):
                break
            shifted = trial
        fallen[shifted[:, 0], shifted[:, 1]] = True
    return fallen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a solid board with the mold's projected columns, then show the detached pieces dropped to the bottom.

    Input structure:
    - The grid has three colors: background, a red mold shape near the top, and a blue full-width solid rectangle lower down.
    - The mold is one 4-connected shape suspended above the board with at least one blank row between them.
    - The board occupies consecutive rows and every column of the grid; the space below it is background.

    Rule:
    - Project every mold pixel straight downward through the board: any board cell in the same column is removed to background.
    - Find the remaining 4-connected board fragments after that column cutting.
    - Keep the holed board visible in its original place, and also copy each fragment downward under gravity until it hits the bottom or a previously fallen fragment.

    Key insight:
    - The mold matters only through the set of columns it occupies; those projected cut columns split the board into independent fragments that are then dropped as rigid pieces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    board_color, board_top, board_bottom = _find_board(g)
    mold_color, bg_color = _find_mold_and_bg(g, board_color, board_top)

    mold_mask = g == mold_color
    assert len(_components4(mold_mask)) == 1, "expected one connected mold"
    cut_columns = np.any(mold_mask, axis=0)
    assert np.any(cut_columns), "expected at least one cut column"

    out = g.copy()
    out[board_top : board_bottom + 1, cut_columns] = bg_color

    board_mask = out == board_color
    fragments = _components4(board_mask)
    fallen_mask = _drop_components(fragments, (h, w))
    out[fallen_mask] = board_color
    return out


def _sample_geometry() -> tuple[int, int, int, int, int, int] | None:
    for _ in range(100):
        width = random.randint(*WIDTH_RANGE)
        board_h = random.randint(*BOARD_HEIGHT_RANGE)
        top_pad = random.randint(*TOP_PADDING_RANGE)
        gap = random.randint(*GAP_RANGE)
        max_mold_h = min(MOLD_HEIGHT_RANGE[1], MAX_ARC_SIZE - (top_pad + gap + 2 * board_h))
        if max_mold_h < MOLD_HEIGHT_RANGE[0]:
            continue
        mold_h = random.randint(MOLD_HEIGHT_RANGE[0], max_mold_h)
        remaining = MAX_ARC_SIZE - (top_pad + mold_h + gap + board_h)
        if remaining < board_h:
            continue
        bottom_blank = random.randint(board_h, min(remaining, board_h + BOTTOM_EXTRA_RANGE[1]))
        total_h = top_pad + mold_h + gap + board_h + bottom_blank
        if total_h <= MAX_ARC_SIZE:
            board_top = top_pad + mold_h + gap
            return total_h, width, top_pad, mold_h, board_top, board_h
    return None


def _sample_cut_interval(width: int) -> tuple[int, int]:
    if width >= 3 and random.random() < INTERNAL_CUT_PROBABILITY:
        start = random.randint(1, width - 2)
        end = random.randint(start, width - 2)
        return start, end
    if random.random() < 0.5:
        return 0, random.randint(0, width - 2)
    return random.randint(1, width - 1), width - 1


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in zip(*np.where(mask)):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _make_mold_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    row = random.randint(0, height - 1)
    mask[row, 0] = True
    for col in range(1, width):
        next_row = min(height - 1, max(0, row + random.choice((-1, 0, 1))))
        mask[row, col] = True
        lo, hi = sorted((row, next_row))
        mask[lo : hi + 1, col] = True
        row = next_row

    current_area = int(mask.sum())
    full_area = height * width
    max_area = full_area if current_area == full_area else full_area - 1
    if current_area < max_area:
        target_area = random.randint(current_area, max_area)
        while int(mask.sum()) < target_area:
            choices = _frontier(mask)
            if not choices:
                break
            rr, cc = random.choice(choices)
            mask[rr, cc] = True

    assert np.all(mask.any(axis=0)), "mold must cover every cut column"
    assert len(_components4(mask)) == 1, "mold must stay connected"
    return mask


def _render_expected(
    height: int,
    width: int,
    top_pad: int,
    mold_mask: np.ndarray,
    cut_start: int,
    cut_end: int,
    board_top: int,
    board_h: int,
) -> np.ndarray:
    out = np.full((height, width), BG_COLOR, dtype=int)
    out[board_top : board_top + board_h, :] = BOARD_COLOR
    out[board_top : board_top + board_h, cut_start : cut_end + 1] = BG_COLOR
    out[top_pad : top_pad + mold_mask.shape[0], cut_start : cut_end + 1][mold_mask] = MOLD_COLOR

    left_w = cut_start
    right_start = cut_end + 1
    if left_w > 0:
        out[height - board_h : height, :left_w] = BOARD_COLOR
    if right_start < width:
        out[height - board_h : height, right_start:width] = BOARD_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    height, width, top_pad, mold_h, board_top, board_h = geometry
    cut_start, cut_end = _sample_cut_interval(width)
    mold_mask = _make_mold_mask(mold_h, cut_end - cut_start + 1)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[board_top : board_top + board_h, :] = BOARD_COLOR
    grid[top_pad : top_pad + mold_h, cut_start : cut_end + 1][mold_mask] = MOLD_COLOR

    expected = _render_expected(height, width, top_pad, mold_mask, cut_start, cut_end, board_top, board_h)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")
