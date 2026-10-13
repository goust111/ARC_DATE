from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_SPACING = 3
MIN_PATHS = 3
MAX_PATHS = 6
MAX_PER_AXIS = 3
MIN_BOARD_SIZE = 9
MAX_BOARD_SIZE = 18
GEN_ATTEMPTS = 400
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_layout(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    legend: list[tuple[int, int]] | None = None
    board: list[tuple[int, int]] | None = None

    for comp in _components(g != BLACK):
        colors = {int(g[r, c]) for r, c in comp}
        r0, c0, r1, c1 = _bbox(comp)
        if GREY in colors:
            assert board is None, "expected a single path network"
            board = comp
            continue

        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner and c0 == c1:
            assert legend is None, "expected a single legend component"
            assert len(comp) == r1 - r0 + 1, "legend must be a solid vertical strip"
            legend = comp

    assert legend is not None, "legend component not found"
    assert board is not None, "path network not found"

    lr0, lc0, lr1, _ = _bbox(legend)
    priority = [int(g[r, lc0]) for r in range(lr0, lr1 + 1)]
    assert len(priority) == len(set(priority)), "legend colors must be unique"

    return priority, _bbox(board)


def _overpass(board: np.ndarray, priority: list[int]) -> np.ndarray:
    g = np.asarray(board, dtype=int)
    out = g.copy()
    rank = {color: i for i, color in enumerate(priority)}

    for r, c in np.argwhere(g == GREY):
        neighbors: list[tuple[int, int, int]] = []
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                color = int(g[nr, nc])
                if color not in (BLACK, GREY):
                    neighbors.append((nr, nc, color))

        colors = sorted({color for _, _, color in neighbors}, key=lambda color: rank[color])
        assert len(colors) == 2, "each crossing must involve exactly two path colors"

        high = colors[0]
        low = colors[1]
        high_cells = [(nr, nc) for nr, nc, color in neighbors if color == high]
        low_cells = [(nr, nc) for nr, nc, color in neighbors if color == low]
        assert len(high_cells) == 2 and len(low_cells) == 2, "crossings must have two cells per path"

        out[r, c] = high
        for nr, nc in low_cells:
            out[nr, nc] = BLACK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    priority, (r0, c0, r1, c1) = _extract_layout(g)
    board = g[r0 : r1 + 1, c0 : c1 + 1]
    return _overpass(board, priority)


def _sample_spaced_positions(size: int, count: int) -> list[int]:
    compressed_limit = (size - 2) - (count - 1) * (MIN_SPACING - 1)
    assert compressed_limit >= count
    anchors = sorted(random.sample(range(1, compressed_limit + 1), count))
    return [anchor + i * (MIN_SPACING - 1) for i, anchor in enumerate(anchors)]


def _horizontal_rank_subset(total: int, count: int) -> tuple[int, ...]:
    choices = [
        subset
        for subset in itertools.combinations(range(total), count)
        if subset != tuple(range(count)) and subset != tuple(range(total - count, total))
    ]
    assert choices
    return random.choice(choices)


def _build_board(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    horizontal_colors: list[int],
    vertical_colors: list[int],
) -> np.ndarray:
    board = np.full((height, width), BLACK, dtype=int)
    for row, color in zip(rows, horizontal_colors):
        board[row, :] = int(color)
    for col, color in zip(cols, vertical_colors):
        board[:, col] = int(color)
    for row in rows:
        for col in cols:
            board[row, col] = GREY
    return board


def _embed_with_legend(board: np.ndarray, priority: list[int], corner: str) -> np.ndarray:
    bh, bw = board.shape
    legend_h = len(priority)
    full_h = bh + legend_h + 1
    grid = np.full((full_h, bw), BLACK, dtype=int)

    if corner in ("top_left", "top_right"):
        board_r0 = legend_h + 1
        legend_rows = range(legend_h)
    else:
        board_r0 = 0
        legend_rows = range(bh + 1, bh + 1 + legend_h)

    grid[board_r0 : board_r0 + bh, :] = board
    legend_col = 0 if corner.endswith("left") else bw - 1
    for row, color in zip(legend_rows, priority):
        grid[row, legend_col] = int(color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    total_paths = random.randint(MIN_PATHS, MAX_PATHS)
    valid_horizontal_counts = [
        count
        for count in range(1, total_paths)
        if count <= MAX_PER_AXIS and total_paths - count <= MAX_PER_AXIS
    ]
    horizontal_count = random.choice(valid_horizontal_counts)
    vertical_count = total_paths - horizontal_count

    height_min = max(MIN_BOARD_SIZE, horizontal_count * MIN_SPACING + 1)
    width_min = max(MIN_BOARD_SIZE, vertical_count * MIN_SPACING + 1)
    height = random.randint(height_min, MAX_BOARD_SIZE)
    width = random.randint(width_min, MAX_BOARD_SIZE)

    rows = _sample_spaced_positions(height, horizontal_count)
    cols = _sample_spaced_positions(width, vertical_count)

    palette = random.sample(PATH_COLORS, total_paths)
    priority = random.sample(palette, len(palette))
    horizontal_ranks = set(_horizontal_rank_subset(total_paths, horizontal_count))

    horizontal_colors = [priority[i] for i in range(total_paths) if i in horizontal_ranks]
    vertical_colors = [priority[i] for i in range(total_paths) if i not in horizontal_ranks]
    random.shuffle(horizontal_colors)
    random.shuffle(vertical_colors)

    board = _build_board(height, width, rows, cols, horizontal_colors, vertical_colors)
    corner = random.choice(LEGEND_CORNERS)
    grid = _embed_with_legend(board, priority, corner)

    detected_priority, board_bbox = _extract_layout(grid)
    br0, bc0, br1, bc1 = board_bbox
    if detected_priority != priority:
        return None
    if not np.array_equal(grid[br0 : br1 + 1, bc0 : bc1 + 1], board):
        return None

    out = main(grid)
    if out.shape != board.shape:
        return None
    if np.count_nonzero(board == GREY) == 0:
        return None
    if np.count_nonzero((board != BLACK) & (out == BLACK)) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
