from __future__ import annotations

import random

import numpy as np


def _nonzero_colors(arr: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(arr) if int(v) != 0)


def _find_divider_column(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        if c != w - c - 1:
            continue
        colors = _nonzero_colors(grid[:, c])
        if len(colors) == 1:
            candidates.append((c, colors[0]))
    assert len(candidates) == 1, "expected one full-height divider column splitting equal-width panels"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Merge the two side panels row by row, with the divider color selecting union and its gaps selecting XOR.

    Input structure:
    - The grid is split by one full-height divider column into equal-width left
      and right panels.
    - The left panel uses only black and one foreground color A; the right panel
      uses only black and one different foreground color B.
    - The divider column uses one foreground color C on some rows and black gaps
      on the remaining rows, with A, B, and C all distinct.

    Rule:
    - Remove the divider column and align the two side panels cell by cell.
    - On rows where the divider cell is color C, output the row-wise union:
      any occupied cell from either panel becomes color C, otherwise black.
    - On rows where the divider cell is black, output the row-wise XOR: cells
      occupied only on the left become A, cells occupied only on the right
      become B, and cells occupied on both sides or neither side become black.

    Key insight:
    - The divider is not just a separator; each of its rows is a visible switch
      telling that row whether to use a recolored union or a color-preserving XOR.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_c, divider_color = _find_divider_column(g)
    left = g[:, :divider_c]
    right = g[:, divider_c + 1 :]
    assert left.shape == right.shape and left.size > 0

    left_colors = _nonzero_colors(left)
    right_colors = _nonzero_colors(right)
    divider_values = set(int(v) for v in np.unique(g[:, divider_c]))
    assert len(left_colors) == 1, "left panel must use exactly one foreground color"
    assert len(right_colors) == 1, "right panel must use exactly one foreground color"
    assert divider_values <= {0, divider_color}, "divider rows must be color or gap only"

    left_color = left_colors[0]
    right_color = right_colors[0]
    assert len({left_color, right_color, divider_color}) == 3

    left_mask = left != 0
    right_mask = right != 0

    out = np.zeros_like(left)
    for r in range(g.shape[0]):
        if int(g[r, divider_c]) == divider_color:
            out[r, left_mask[r] | right_mask[r]] = divider_color
        else:
            left_only = left_mask[r] & ~right_mask[r]
            right_only = right_mask[r] & ~left_mask[r]
            out[r, left_only] = left_color
            out[r, right_only] = right_color
    return out


def _paint_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _seed_panel_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    n_rects = random.randint(2, 4)
    for _ in range(n_rects):
        rect_h = random.randint(2, min(height, max(2, height // 2 + 1)))
        rect_w = random.randint(2, min(width, max(2, width - 1)))
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        _paint_rect(mask, r0, c0, rect_h, rect_w)

    if not mask.any():
        r = random.randrange(height)
        c = random.randrange(width)
        mask[r, c] = True

    if random.random() < 0.7:
        c = random.randrange(width)
        r0 = random.randrange(height)
        r1 = random.randrange(r0, height)
        mask[r0:r1, c] = True

    return mask


def _ensure_union_row(left_row: np.ndarray, right_row: np.ndarray) -> None:
    width = left_row.shape[0]
    a, b = random.sample(range(width), 2)
    left_row[a] = True
    right_row[a] = False
    left_row[b] = False
    right_row[b] = True
    if width >= 3 and random.random() < 0.75:
        extra = random.choice([c for c in range(width) if c not in {a, b}])
        left_row[extra] = True
        right_row[extra] = True


def _ensure_xor_row(left_row: np.ndarray, right_row: np.ndarray) -> None:
    width = left_row.shape[0]
    a, b, c = random.sample(range(width), 3)
    left_row[a] = True
    right_row[a] = False
    left_row[b] = False
    right_row[b] = True
    left_row[c] = True
    right_row[c] = True


def _gen_attempt() -> np.ndarray:
    height = random.randint(6, 15)
    panel_width = random.randint(4, 12)

    left_mask = _seed_panel_mask(height, panel_width)
    right_mask = _seed_panel_mask(height, panel_width)

    gap_rows = np.zeros(height, dtype=bool)
    gap_count = random.randint(2, height - 2)
    gap_rows[random.sample(range(height), gap_count)] = True

    for r in range(height):
        if gap_rows[r]:
            _ensure_xor_row(left_mask[r], right_mask[r])
        else:
            _ensure_union_row(left_mask[r], right_mask[r])

    if not np.any(left_mask & right_mask):
        r = random.randrange(height)
        c = random.randrange(panel_width)
        left_mask[r, c] = True
        right_mask[r, c] = True

    left_color, right_color, divider_color = random.sample(range(1, 10), 3)
    grid = np.zeros((height, panel_width * 2 + 1), dtype=int)
    grid[:, :panel_width][left_mask] = left_color
    grid[:, panel_width + 1 :][right_mask] = right_color
    grid[~gap_rows, panel_width] = divider_color

    assert max(grid.shape) <= 30
    assert np.any(grid[:, panel_width] == 0)
    assert np.any(grid[:, panel_width] == divider_color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        out = main(grid)
        if (
            isinstance(grid, np.ndarray)
            and isinstance(out, np.ndarray)
            and grid.ndim == 2
            and out.ndim == 2
            and grid.shape[1] == out.shape[1] * 2 + 1
        ):
            return grid
    raise RuntimeError("failed to generate a valid divider-switch puzzle input")
