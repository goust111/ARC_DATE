from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 30
MAX_FRAME_COUNT = 4
MIN_FRAME_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
OUTER_MARGIN = 1
INNER_LIMIT = 7


def _neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def _collect_component(mask: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> list[tuple[int, int]]:
    height, width = mask.shape
    queue: deque[tuple[int, int]] = deque([start])
    visited[start] = True
    cells: list[tuple[int, int]] = []

    while queue:
        row, col = queue.popleft()
        cells.append((row, col))
        for next_row, next_col in _neighbors4(row, col):
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if visited[next_row, next_col] or not mask[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col))

    return cells


def _frame_from_component(cells: list[tuple[int, int]], color: int) -> dict[str, int] | None:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    outer_h = bottom - top + 1
    outer_w = right - left + 1
    if outer_h < 3 or outer_w < 3:
        return None

    border_len = 2 * (outer_h + outer_w) - 4
    if len(cells) != border_len:
        return None

    local = np.zeros((outer_h, outer_w), dtype=bool)
    for row, col in cells:
        local[row - top, col - left] = True

    expected = np.zeros_like(local)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(local, expected):
        return None

    return {
        "color": int(color),
        "top": int(top),
        "bottom": int(bottom),
        "left": int(left),
        "right": int(right),
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    grid = np.asarray(grid, dtype=int)
    frames: list[dict[str, int]] = []
    for color in sorted(int(value) for value in np.unique(grid) if int(value) not in (BG, YELLOW)):
        mask = grid == color
        visited = np.zeros_like(mask, dtype=bool)
        starts = np.argwhere(mask)
        for row, col in starts:
            row = int(row)
            col = int(col)
            if visited[row, col]:
                continue
            component = _collect_component(mask, (row, col), visited)
            frame = _frame_from_component(component, color)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda frame: (frame["top"], frame["left"], frame["bottom"], frame["right"], frame["color"]))
    return frames


def _spiral_cells(top: int, bottom: int, left: int, right: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            cells.append((top, col))
        for row in range(top + 1, bottom + 1):
            cells.append((row, right))
        if top < bottom:
            for col in range(right - 1, left - 1, -1):
                cells.append((bottom, col))
        if left < right:
            for row in range(bottom - 1, top, -1):
                cells.append((row, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return cells


def _first_layer_length(inner_h: int, inner_w: int) -> int:
    if inner_h == 1 or inner_w == 1:
        return inner_h * inner_w
    return 2 * (inner_h + inner_w) - 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange the yellow markers inside each colored rectangular frame into a clockwise spiral.

    Input structure:
    - The background is black.
    - The grid contains one-pixel-thick colored rectangular frames.
    - Each frame encloses a rectangular interior containing scattered yellow marker pixels.

    Rule:
    - For each colored frame, count the yellow pixels strictly inside the frame.
    - Clear those interior yellow pixels back to the black background.
    - Starting from the interior's top-left cell, place the same number of yellow pixels along the clockwise spiral path hugging the frame's inner edge, continuing inward as needed.
    - Keep every colored frame unchanged.

    Key insight:
    - The original marker positions are irrelevant; only the count of yellow pixels inside each frame matters, and the output order is the interior spiral traversal.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    out = grid.copy()
    for frame in _find_frames(grid):
        top = frame["top"] + 1
        bottom = frame["bottom"] - 1
        left = frame["left"] + 1
        right = frame["right"] - 1
        spiral = _spiral_cells(top, bottom, left, right)
        count = int(np.count_nonzero(grid[top : bottom + 1, left : right + 1] == YELLOW))
        interior = out[top : bottom + 1, left : right + 1]
        interior[interior == YELLOW] = BG
        for row, col in spiral[:count]:
            out[row, col] = YELLOW

    return out


def _sample_outer_branch_dims() -> tuple[int, int]:
    while True:
        inner_h = random.randint(1, INNER_LIMIT)
        inner_w = random.randint(1, INNER_LIMIT)
        if inner_h * inner_w >= 2:
            return inner_h, inner_w


def _sample_deep_branch_dims() -> tuple[int, int]:
    candidates = [
        (inner_h, inner_w)
        for inner_h in range(3, INNER_LIMIT + 1)
        for inner_w in range(3, INNER_LIMIT + 1)
        if inner_h * inner_w - _first_layer_length(inner_h, inner_w) >= 2
    ]
    return random.choice(candidates)


def _sample_frame_specs(frame_count: int) -> list[dict[str, int]]:
    colors = random.sample(FRAME_COLORS, frame_count)
    specs: list[dict[str, int]] = []

    for color in colors:
        deep_branch = random.random() < 0.5
        if deep_branch:
            inner_h, inner_w = _sample_deep_branch_dims()
        else:
            inner_h, inner_w = _sample_outer_branch_dims()

        area = inner_h * inner_w
        first_layer = _first_layer_length(inner_h, inner_w)
        if deep_branch:
            yellow_count = random.randint(first_layer + 1, area - 1)
        else:
            yellow_count = random.randint(1, min(first_layer, area - 1))

        specs.append(
            {
                "color": int(color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "yellow_count": int(yellow_count),
            }
        )

    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _placement_candidates(occupied: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    height, width = occupied.shape
    rows = range(OUTER_MARGIN, height - outer_h - OUTER_MARGIN + 1)
    cols = range(OUTER_MARGIN, width - outer_w - OUTER_MARGIN + 1)
    candidates: list[tuple[int, int]] = []

    for top in rows:
        for left in cols:
            row0 = max(0, top - MIN_FRAME_GAP)
            row1 = min(height, top + outer_h + MIN_FRAME_GAP)
            col0 = max(0, left - MIN_FRAME_GAP)
            col1 = min(width, left + outer_w + MIN_FRAME_GAP)
            if occupied[row0:row1, col0:col1].any():
                continue
            candidates.append((top, left))

    return candidates


def _paint_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, color: int) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _scatter_positions(cells: list[tuple[int, int]], count: int, target_prefix: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    target_set = set(target_prefix[:count])
    for _ in range(20):
        picked = random.sample(cells, count)
        if set(picked) != target_set:
            return picked
    return None


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)

    max_frames = max(1, min(MAX_FRAME_COUNT, (height * width) // 70 + 1))
    frame_count = random.randint(1, max_frames)
    specs = _sample_frame_specs(frame_count)

    occupied = np.zeros((height, width), dtype=bool)
    placed_specs: list[dict[str, int]] = []

    for spec in specs:
        candidates = _placement_candidates(occupied, spec["outer_h"], spec["outer_w"])
        if not candidates:
            return None

        top, left = random.choice(candidates)
        spec = spec.copy()
        spec["top"] = int(top)
        spec["left"] = int(left)
        spec["bottom"] = int(top + spec["outer_h"] - 1)
        spec["right"] = int(left + spec["outer_w"] - 1)
        placed_specs.append(spec)

        row0 = max(0, top - MIN_FRAME_GAP)
        row1 = min(height, top + spec["outer_h"] + MIN_FRAME_GAP)
        col0 = max(0, left - MIN_FRAME_GAP)
        col1 = min(width, left + spec["outer_w"] + MIN_FRAME_GAP)
        occupied[row0:row1, col0:col1] = True

    base = np.full((height, width), BG, dtype=int)
    inp = base.copy()
    expected = base.copy()

    for spec in placed_specs:
        _paint_frame(base, spec["top"], spec["left"], spec["outer_h"], spec["outer_w"], spec["color"])
        _paint_frame(inp, spec["top"], spec["left"], spec["outer_h"], spec["outer_w"], spec["color"])
        _paint_frame(expected, spec["top"], spec["left"], spec["outer_h"], spec["outer_w"], spec["color"])

        top = spec["top"] + 1
        bottom = spec["bottom"] - 1
        left = spec["left"] + 1
        right = spec["right"] - 1
        spiral = _spiral_cells(top, bottom, left, right)
        interior_cells = [(row, col) for row in range(top, bottom + 1) for col in range(left, right + 1)]
        scattered = _scatter_positions(interior_cells, spec["yellow_count"], spiral)
        if scattered is None:
            return None

        for row, col in scattered:
            inp[row, col] = YELLOW
        for row, col in spiral[: spec["yellow_count"]]:
            expected[row, col] = YELLOW

    if np.array_equal(inp, expected):
        return None
    if len(_find_frames(inp)) != frame_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _build_attempt()
        if attempt is not None:
            inp, _expected = attempt
            return inp
    raise RuntimeError("failed to generate a valid spiral-frame ARC input")
