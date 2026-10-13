from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 9
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    if np.any(np.asarray(grid, dtype=int) == BG):
        return BG
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _analyze_vertical_strip(strip: np.ndarray, bg: int) -> list[int] | None:
    h, w = strip.shape
    if w != 2:
        return None

    for marker_col in (0, 1):
        color_col = 1 - marker_col
        marker_values = [int(v) for v in np.unique(strip[:, marker_col]) if int(v) != bg]
        for marker in marker_values:
            active = (
                (strip[:, marker_col] == marker)
                & (strip[:, color_col] != bg)
                & (strip[:, color_col] != marker)
            )
            idx = np.flatnonzero(active)
            if idx.size < 4:
                continue
            if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
                continue
            inactive = np.ones(h, dtype=bool)
            inactive[idx] = False
            if not np.all(strip[inactive] == bg):
                continue
            order = [int(v) for v in strip[idx, color_col]]
            if len(set(order)) != len(order):
                continue
            return order
    return None


def _analyze_horizontal_strip(strip: np.ndarray, bg: int) -> list[int] | None:
    h, w = strip.shape
    if h != 2:
        return None

    for marker_row in (0, 1):
        color_row = 1 - marker_row
        marker_values = [int(v) for v in np.unique(strip[marker_row, :]) if int(v) != bg]
        for marker in marker_values:
            active = (
                (strip[marker_row, :] == marker)
                & (strip[color_row, :] != bg)
                & (strip[color_row, :] != marker)
            )
            idx = np.flatnonzero(active)
            if idx.size < 4:
                continue
            if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
                continue
            inactive = np.ones(w, dtype=bool)
            inactive[idx] = False
            if not np.all(strip[:, inactive] == bg):
                continue
            order = [int(v) for v in strip[color_row, idx]]
            if len(set(order)) != len(order):
                continue
            return order
    return None


def _extract_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    h, w = g.shape
    candidates: list[tuple[str, list[int]]] = []

    if w >= 2:
        order = _analyze_vertical_strip(g[:, :2], bg)
        if order is not None:
            candidates.append(("left", order))
        order = _analyze_vertical_strip(g[:, w - 2 :], bg)
        if order is not None:
            candidates.append(("right", order))
    if h >= 2:
        order = _analyze_horizontal_strip(g[:2, :], bg)
        if order is not None:
            candidates.append(("top", order))
        order = _analyze_horizontal_strip(g[h - 2 :, :], bg)
        if order is not None:
            candidates.append(("bottom", order))

    assert len(candidates) == 1, "expected exactly one edge legend strip"
    side, order = candidates[0]

    if side == "left":
        body = g[:, 2:]
    elif side == "right":
        body = g[:, :-2]
    elif side == "top":
        body = g[2:, :]
    else:
        body = g[:-2, :]
    return body, order, bg


def _solve_body(body: np.ndarray, order: list[int], bg: int) -> np.ndarray:
    rank = {int(color): idx for idx, color in enumerate(order)}
    out = body.copy()
    h, w = body.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            left = int(body[r, c - 1])
            right = int(body[r, c + 1])
            up = int(body[r - 1, c])
            down = int(body[r + 1, c])
            center = int(body[r, c])

            if left == right != bg and up == down != bg and left != up and center in (left, up):
                if left in rank and up in rank:
                    out[r, c] = left if rank[left] < rank[up] else up

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-render overlapping orthogonal paths so each crossing shows the color with higher legend priority.

    Input structure:
    - The background is black.
    - Several one-pixel-wide straight paths run horizontally or vertically, each in its own color.
    - Horizontal and vertical paths may cross; at each crossing the visible pixel shows only the color of the path currently on top.
    - A two-cell-thick legend strip sits on one outer edge: one lane lists colors in priority order, and the adjacent lane is a uniform marker color.

    Rule:
    - Remove the legend strip from the output.
    - Read the legend order from its first listed color to its last listed color; earlier means higher priority.
    - For every cell whose left-right neighbors form one colored path and whose up-down neighbors form a different colored path, treat that cell as a crossing.
    - Recolor the crossing to the higher-priority of the two path colors and leave all non-crossing cells unchanged.

    Key insight:
    - The crossing pixel does not describe the permanent identity of the intersection by itself; the edge legend orders the path colors, and that ordering decides which color must be visible on top.
    """
    g = np.asarray(grid, dtype=int)
    body, order, bg = _extract_legend(g)
    return _solve_body(body, order, bg)


def _max_spaced_count(size: int) -> int:
    return max(0, (size - 1) // 2)


def _sample_spaced_positions(size: int, count: int, rng: random.Random) -> list[int]:
    choices = list(range(1, size - 1))
    for _ in range(200):
        picked = sorted(rng.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    raise RuntimeError("failed to sample non-adjacent path positions")


def _place_legend(body: np.ndarray, order: list[int], side: str, rng: random.Random) -> np.ndarray:
    h, w = body.shape
    n = len(order)

    if side == "left":
        grid = np.full((h, w + 2), BG, dtype=int)
        grid[:, 2:] = body
        start = rng.randint(0, h - n)
        for i, color in enumerate(order):
            grid[start + i, 0] = color
            grid[start + i, 1] = MARKER
        return grid

    if side == "right":
        grid = np.full((h, w + 2), BG, dtype=int)
        grid[:, :w] = body
        start = rng.randint(0, h - n)
        for i, color in enumerate(order):
            grid[start + i, w] = color
            grid[start + i, w + 1] = MARKER
        return grid

    if side == "top":
        grid = np.full((h + 2, w), BG, dtype=int)
        grid[2:, :] = body
        start = rng.randint(0, w - n)
        for i, color in enumerate(order):
            grid[0, start + i] = color
            grid[1, start + i] = MARKER
        return grid

    grid = np.full((h + 2, w), BG, dtype=int)
    grid[:h, :] = body
    start = rng.randint(0, w - n)
    for i, color in enumerate(order):
        grid[h, start + i] = color
        grid[h + 1, start + i] = MARKER
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    body_h = rng.randint(10, 18)
    body_w = rng.randint(10, 18)

    max_h = min(4, _max_spaced_count(body_h))
    max_v = min(4, _max_spaced_count(body_w))
    if max_h < 2 or max_v < 2:
        return None

    num_h = rng.randint(2, max_h)
    num_v = rng.randint(2, min(max_v, len(PATH_COLORS) - num_h))
    used_colors = list(PATH_COLORS)
    rng.shuffle(used_colors)
    horiz_colors = used_colors[:num_h]
    vert_colors = used_colors[num_h : num_h + num_v]
    row_positions = _sample_spaced_positions(body_h, num_h, rng)
    col_positions = _sample_spaced_positions(body_w, num_v, rng)
    horiz_rows = dict(zip(horiz_colors, row_positions))
    vert_cols = dict(zip(vert_colors, col_positions))
    order = horiz_colors + vert_colors
    rng.shuffle(order)
    rank = {color: idx for idx, color in enumerate(order)}

    body = np.full((body_h, body_w), BG, dtype=int)
    for color, row in horiz_rows.items():
        body[row, :] = color
    for color, col in vert_cols.items():
        body[:, col] = color

    expected = body.copy()
    crossings: list[dict[str, int | bool]] = []
    for h_color, row in horiz_rows.items():
        for v_color, col in vert_cols.items():
            correct = h_color if rank[h_color] < rank[v_color] else v_color
            wrong = v_color if correct == h_color else h_color
            is_correct = rng.random() < 0.5
            crossings.append(
                {
                    "row": row,
                    "col": col,
                    "correct": correct,
                    "wrong": wrong,
                    "is_correct": is_correct,
                }
            )

    if all(item["is_correct"] for item in crossings):
        crossings[rng.randrange(len(crossings))]["is_correct"] = False
    if not any(item["is_correct"] for item in crossings):
        crossings[rng.randrange(len(crossings))]["is_correct"] = True

    for item in crossings:
        row = int(item["row"])
        col = int(item["col"])
        correct = int(item["correct"])
        wrong = int(item["wrong"])
        top = correct if bool(item["is_correct"]) else wrong
        body[row, col] = top
        expected[row, col] = correct

    if np.array_equal(body, expected):
        return None

    side = rng.choice(("left", "right", "top", "bottom"))
    if side in ("left", "right") and len(order) > body_h:
        return None
    if side in ("top", "bottom") and len(order) > body_w:
        return None

    grid = _place_legend(body, order, side, rng)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
