import random

import numpy as np


def _most_common(values):
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _side_has_enough_frame(side, frame_color):
    need = 2 if len(side) >= 4 else 1
    return int(np.count_nonzero(side == frame_color)) >= need


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each rectangular frame by using its number of border gaps as a palette index.

    Input structure:
    The grid has a most-common background color, several one-pixel rectangular
    frames drawn in one shared frame color, and a bottom palette row containing
    separated one-cell color swatches. Frame interiors are background. Some
    border cells of each frame are missing and remain background-colored gaps.
    Rule:
    Read the bottom-row swatches from left to right. For every rectangular
    frame above that row, count the background-colored cells on its expected
    perimeter. If the count is n, fill the rectangle interior with the nth
    palette color while leaving the frame and its gaps unchanged. Clear the
    palette row to background in the output.
    Key insight:
    The border gaps are not damage; their count is an index into the visible
    palette row.
    """
    g = np.array(grid, copy=True)
    rows, cols = g.shape
    bg = _most_common(g)
    out = g.copy()

    palette = [int(x) for x in g[-1] if int(x) != bg]
    out[-1, :] = bg
    if rows < 4 or cols < 3 or not palette:
        return out

    body = g[:-1, :]
    body_non_bg = body[body != bg]
    if body_non_bg.size == 0:
        return out
    frame_color = _most_common(body_non_bg)
    max_gaps = len(palette)

    candidates = []
    for r1 in range(rows - 3):
        for r2 in range(r1 + 2, rows - 1):
            for c1 in range(cols - 2):
                for c2 in range(c1 + 2, cols):
                    interior = g[r1 + 1 : r2, c1 + 1 : c2]
                    if np.any(interior != bg):
                        continue

                    top = g[r1, c1 : c2 + 1]
                    bottom = g[r2, c1 : c2 + 1]
                    left = g[r1 + 1 : r2, c1]
                    right = g[r1 + 1 : r2, c2]
                    perimeter = np.concatenate((top, bottom, left, right))
                    ok_values = (perimeter == bg) | (perimeter == frame_color)
                    if not np.all(ok_values):
                        continue

                    gaps = int(np.count_nonzero(perimeter == bg))
                    if gaps < 1 or gaps > max_gaps:
                        continue
                    if not (
                        _side_has_enough_frame(top, frame_color)
                        and _side_has_enough_frame(bottom, frame_color)
                        and _side_has_enough_frame(g[r1:r2 + 1, c1], frame_color)
                        and _side_has_enough_frame(g[r1:r2 + 1, c2], frame_color)
                    ):
                        continue

                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    candidates.append((area, r1, r2, c1, c2, gaps))

    used = np.zeros(g.shape, dtype=bool)
    for _, r1, r2, c1, c2, gaps in sorted(candidates, reverse=True):
        if np.any(used[r1 : r2 + 1, c1 : c2 + 1]):
            continue
        out[r1 + 1 : r2, c1 + 1 : c2] = palette[gaps - 1]
        used[r1 : r2 + 1, c1 : c2 + 1] = True

    return out


def _palette_positions(width, count):
    required = 2 * count - 1
    pos = random.randint(0, width - required)
    positions = []
    for i in range(count):
        positions.append(pos)
        left = count - i - 1
        if left:
            min_next = pos + 2
            spare = width - (min_next + (2 * left - 1))
            pos = min_next + random.randint(0, min(2, max(0, spare)))
    return positions


def _perimeter_cells(r1, r2, c1, c2):
    cells = []
    for c in range(c1 + 1, c2):
        cells.append((r1, c))
        cells.append((r2, c))
    for r in range(r1 + 1, r2):
        cells.append((r, c1))
        cells.append((r, c2))
    return cells


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=int)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid
