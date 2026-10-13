from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _infer_horizontal(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, str]], list[int]]:
    h, w = grid.shape
    assert w >= 3, "horizontal stripes need room for one marker and visible body"

    rows = [int(r) for r in range(h) if bool(np.all(grid[r] != bg))]
    stripes: list[tuple[int, int, str]] = []

    for r in rows:
        row = grid[r]
        left_marker = bool(row[0] != row[1] and np.all(row[1:] == row[1]))
        right_marker = bool(row[-1] != row[-2] and np.all(row[:-1] == row[-2]))
        assert left_marker ^ right_marker, "horizontal stripe must have exactly one endpoint marker"

        if left_marker:
            stripes.append((int(r), int(row[1]), "left"))
        else:
            stripes.append((int(r), int(row[-2]), "right"))

    return stripes, rows


def _infer_vertical(
    grid: np.ndarray, bg: int, horizontal_rows: list[int]
) -> list[tuple[int, int, str]]:
    h, w = grid.shape
    row_mask = np.zeros(h, dtype=bool)
    if horizontal_rows:
        row_mask[np.array(horizontal_rows, dtype=int)] = True

    visible_rows = np.flatnonzero(~row_mask)
    assert len(visible_rows) >= 3, "need at least three non-horizontal rows to read vertical markers"

    cols = [int(c) for c in range(w) if bool(np.all(grid[visible_rows, c] != bg))]
    stripes: list[tuple[int, int, str]] = []

    for c in cols:
        vis = grid[visible_rows, c]
        top_marker = bool(vis[0] != vis[1] and np.all(vis[1:] == vis[1]))
        bottom_marker = bool(vis[-1] != vis[-2] and np.all(vis[:-1] == vis[-2]))
        assert top_marker ^ bottom_marker, "vertical stripe must have exactly one endpoint marker"

        if top_marker:
            stripes.append((int(c), int(vis[1]), "top"))
        else:
            stripes.append((int(c), int(vis[-2]), "bottom"))

    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push full-span horizontal and vertical stripes toward the edge indicated by
    their endpoint marker, then erase the markers.

    Input structure:
    - Background fills the grid corners.
    - Horizontal stripes occupy full rows; vertical stripes occupy full columns.
    - Each stripe has one endpoint cell replaced by a distinct marker color.
    - Horizontal stripes are the visible top layer at row/column crossings.

    Rule:
    - A left-end marker means the horizontal body shifts left, leaving its gap on
      the right edge; a right-end marker does the opposite.
    - A top-end marker means the vertical body shifts up, leaving its gap on the
      bottom edge; a bottom-end marker does the opposite.
    - Vertical stripes are painted first and horizontal stripes second, so
      horizontal stripes cover vertical stripes at intersections.
    - Marker colors never appear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    bg = _get_bg(g)

    horizontal, horizontal_rows = _infer_horizontal(g, bg)
    vertical = _infer_vertical(g, bg, horizontal_rows)

    out = np.full((h, w), int(bg), dtype=int)

    for c, color, side in vertical:
        if side == "top":
            out[: h - 1, c] = int(color)
        else:
            out[1:, c] = int(color)

    for r, color, side in horizontal:
        if side == "left":
            out[r, : w - 1] = int(color)
        else:
            out[r, 1:] = int(color)

    return out


# --------------- generator ---------------

_PALETTE = list(range(10))
_GRID_H_RANGE = (8, 16)
_GRID_W_RANGE = (8, 16)
_H_COUNT_RANGE = (1, 4)
_V_COUNT_RANGE = (1, 4)
_GEN_RETRY_LIMIT = 400
_SPACING_RETRY_LIMIT = 80


def _pick_spaced_positions(length: int, count: int) -> list[int] | None:
    positions = list(range(1, length - 1))
    if count == 0:
        return []
    if count > len(positions):
        return None

    for _ in range(_SPACING_RETRY_LIMIT):
        chosen = sorted(random.sample(positions, count))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _render_input(
    h: int,
    w: int,
    bg: int,
    horizontal: list[tuple[int, int, str, int]],
    vertical: list[tuple[int, int, str, int]],
) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)

    for c, color, side, marker in vertical:
        if side == "top":
            grid[1:, c] = int(color)
            grid[0, c] = int(marker)
        else:
            grid[: h - 1, c] = int(color)
            grid[h - 1, c] = int(marker)

    for r, color, side, marker in horizontal:
        if side == "left":
            grid[r, 1:] = int(color)
            grid[r, 0] = int(marker)
        else:
            grid[r, : w - 1] = int(color)
            grid[r, w - 1] = int(marker)

    return grid


def _render_output(
    h: int,
    w: int,
    bg: int,
    horizontal: list[tuple[int, int, str, int]],
    vertical: list[tuple[int, int, str, int]],
) -> np.ndarray:
    out = np.full((h, w), int(bg), dtype=int)

    for c, color, side, _marker in vertical:
        if side == "top":
            out[: h - 1, c] = int(color)
        else:
            out[1:, c] = int(color)

    for r, color, side, _marker in horizontal:
        if side == "left":
            out[r, : w - 1] = int(color)
        else:
            out[r, 1:] = int(color)

    return out


def _random_marker(stripe_color: int, bg: int) -> int:
    options = [c for c in _PALETTE if c not in (bg, stripe_color)]
    return int(random.choice(options))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)

    max_h = min(_H_COUNT_RANGE[1], (h - 1) // 2)
    max_v = min(_V_COUNT_RANGE[1], (w - 1) // 2)
    if max_h < _H_COUNT_RANGE[0] or max_v < _V_COUNT_RANGE[0]:
        return None

    n_h = random.randint(_H_COUNT_RANGE[0], max_h)
    n_v = random.randint(_V_COUNT_RANGE[0], max_v)

    if 1 + n_h + n_v > len(_PALETTE):
        return None

    row_positions = _pick_spaced_positions(h, n_h)
    col_positions = _pick_spaced_positions(w, n_v)
    if row_positions is None or col_positions is None:
        return None

    colors = random.sample(_PALETTE, 1 + n_h + n_v)
    bg = int(colors[0])
    h_colors = [int(c) for c in colors[1 : 1 + n_h]]
    v_colors = [int(c) for c in colors[1 + n_h :]]

    horizontal: list[tuple[int, int, str, int]] = []
    for r, color in zip(row_positions, h_colors):
        side = random.choice(["left", "right"])
        marker = _random_marker(int(color), int(bg))
        horizontal.append((int(r), int(color), side, int(marker)))

    vertical: list[tuple[int, int, str, int]] = []
    for c, color in zip(col_positions, v_colors):
        side = random.choice(["top", "bottom"])
        marker = _random_marker(int(color), int(bg))
        vertical.append((int(c), int(color), side, int(marker)))

    inp = _render_input(h, w, bg, horizontal, vertical)
    expected = _render_output(h, w, bg, horizontal, vertical)

    try:
        got = main(inp)
    except Exception:
        return None

    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pushed-stripe input")
