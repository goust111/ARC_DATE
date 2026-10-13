from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PUSHER_COLOR = 2
PILLAR_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _pillar_height(grid: np.ndarray, col: int) -> tuple[int, int]:
    color = int(grid[-1, col])
    row = grid.shape[0] - 1
    while row >= 0 and int(grid[row, col]) == color:
        row -= 1
    return color, grid.shape[0] - 1 - row


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    bottom = g[-1]
    pusher_cols = np.flatnonzero(bottom == PUSHER_COLOR)
    assert len(pusher_cols) == 1, "expected exactly one red pusher on the bottom row"
    pusher_col = int(pusher_cols[0])

    foot_cols = [int(c) for c in np.flatnonzero(bottom != bg) if int(c) != pusher_col]
    assert foot_cols, "expected at least one pillar"

    if pusher_col < min(foot_cols):
        direction = 1
        ordered_cols = sorted(foot_cols)
    elif pusher_col > max(foot_cols):
        direction = -1
        ordered_cols = sorted(foot_cols, reverse=True)
    else:
        raise AssertionError("pusher must sit outside the pillar row")

    pillars = []
    for col in ordered_cols:
        color, height = _pillar_height(g, col)
        assert color != bg, "pillar foot cannot be background"
        assert color != PUSHER_COLOR, "pillar colors must stay distinct from the red pusher"
        pillars.append((col, color, height))
    return bg, pusher_col, direction, pillars


def _simulate_chain(pillars: list[tuple[int, int, int]]) -> list[bool]:
    toppled = [False] * len(pillars)
    toppled[0] = True
    for i in range(len(pillars) - 1):
        if not toppled[i]:
            break
        col, _color, height = pillars[i]
        next_col, _next_color, _next_height = pillars[i + 1]
        gap = abs(next_col - col)
        if height >= gap:
            toppled[i + 1] = True
        else:
            break
    return toppled


def _render_from_specs(
    shape: tuple[int, int],
    bg: int,
    pusher_col: int,
    direction: int,
    cols: list[int],
    heights: list[int],
    colors: list[int],
    toppled_count: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[-1, pusher_col] = PUSHER_COLOR

    for i, (col, height, color) in enumerate(zip(cols, heights, colors)):
        if i < toppled_count:
            body_len = height - 1
            if i + 1 < len(cols):
                gap = abs(cols[i + 1] - col)
                body_len = min(body_len, gap - 1)
            if direction == 1:
                out[-1, col : col + body_len + 1] = color
            else:
                out[-1, col - body_len : col + 1] = color
        else:
            out[shape[0] - height :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Topple the bottom-row pillars away from the red pusher until one fallen pillar can no longer reach the next foot.

    Input structure:
    - A black background contains several separated one-cell-wide vertical pillars whose feet all sit on the bottom row.
    - Each pillar is monochrome and the red pusher is a single bottom-row pixel adjacent to the outermost pillar on one side.
    - The pillars are ordered outward from the pusher, with empty horizontal gaps between consecutive feet.

    Rule:
    - The pusher knocks down the adjacent outermost pillar, and every toppled pillar falls away from the pusher while keeping its foot fixed.
    - A fallen pillar paints a same-color horizontal run on the bottom row, with length equal to its height and truncated just before the next pillar foot.
    - If a pillar's height reaches the next foot's column distance, that next pillar also falls; otherwise the chain stops and all later pillars remain upright.

    Key insight:
    - The chain reaction depends only on a local comparison at each step: a pillar's standing height is exactly the reach that can trigger the next pillar.
    """
    bg, pusher_col, direction, pillars = _parse_scene(grid)
    cols = [col for col, _color, _height in pillars]
    colors = [color for _col, color, _height in pillars]
    heights = [height for _col, _color, height in pillars]
    toppled = _simulate_chain(pillars)
    toppled_count = sum(toppled)
    return _render_from_specs(
        np.asarray(grid).shape,
        bg,
        pusher_col,
        direction,
        cols,
        heights,
        colors,
        toppled_count,
    )


# --------------- generator (o5x) ---------------

GEN_RETRY_LIMIT = 400
GEN_PILLAR_COUNT_RANGE = (3, 6)
GEN_HEIGHT_RANGE = (2, 8)
GEN_TOP_PAD_RANGE = (0, 3)
GEN_SIDE_PAD_RANGE = (0, 2)
GEN_MIN_GAP = 2
GEN_FAIL_EXTRA_GAP = 4
GEN_FREE_GAP_MAX = 8
GEN_FULL_CHAIN_PROB = 0.4


def _mirror_h(grid: np.ndarray) -> np.ndarray:
    return grid[:, ::-1].copy()


def _gen_reaching_gap(height: int) -> int | None:
    if height < GEN_MIN_GAP:
        return None
    if random.random() < 0.35:
        return height
    return random.randint(GEN_MIN_GAP, height)


def _gen_failing_gap(height: int) -> int | None:
    lo = height + 1
    hi = min(height + GEN_FAIL_EXTRA_GAP, 10)
    if lo > hi:
        return None
    if random.random() < 0.35:
        return lo
    return random.randint(lo, hi)


def _build_input(
    shape: tuple[int, int],
    pusher_col: int,
    cols: list[int],
    heights: list[int],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[-1, pusher_col] = PUSHER_COLOR
    for col, height, color in zip(cols, heights, colors):
        grid[shape[0] - height :, col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    pillar_count = random.randint(*GEN_PILLAR_COUNT_RANGE)
    heights = [random.randint(*GEN_HEIGHT_RANGE) for _ in range(pillar_count)]
    colors = random.sample(PILLAR_COLORS, pillar_count)

    if random.random() < GEN_FULL_CHAIN_PROB:
        toppled_count = pillar_count
    else:
        toppled_count = random.randint(1, pillar_count - 1)

    gaps = []
    for i in range(pillar_count - 1):
        if i < toppled_count - 1:
            gap = _gen_reaching_gap(heights[i])
        elif i == toppled_count - 1 and toppled_count < pillar_count:
            gap = _gen_failing_gap(heights[i])
        else:
            gap = random.randint(GEN_MIN_GAP, GEN_FREE_GAP_MAX)
        if gap is None:
            return None
        gaps.append(gap)

    left_pad = random.randint(*GEN_SIDE_PAD_RANGE)
    right_pad = random.randint(*GEN_SIDE_PAD_RANGE)
    top_pad = random.randint(*GEN_TOP_PAD_RANGE)

    pusher_col = left_pad
    cols = [pusher_col + 1]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    if toppled_count == pillar_count:
        farthest = cols[-1] + heights[-1] - 1
    else:
        fail_i = toppled_count - 1
        farthest = max(cols[-1], cols[fail_i] + min(heights[fail_i] - 1, gaps[fail_i] - 1))

    width = farthest + 1 + right_pad
    height = max(heights) + top_pad
    if width > 30 or height > 30:
        return None

    inp = _build_input((height, width), pusher_col, cols, heights, colors)
    expected = _render_from_specs(
        inp.shape,
        BACKGROUND,
        pusher_col,
        1,
        cols,
        heights,
        colors,
        toppled_count,
    )

    if random.random() < 0.5:
        inp = _mirror_h(inp)
        expected = _mirror_h(expected)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-reach input")
