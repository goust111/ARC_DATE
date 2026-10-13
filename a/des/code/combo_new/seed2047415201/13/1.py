from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
LEGEND_SIZE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _grey_components(g: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(g == GREY):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        cells: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and g[rr, cc] == GREY:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        comps.append(np.array(cells, dtype=int))

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by the legend color for its larger bounding-box span.

    Input structure:
    - The background is Black.
    - Several Grey 4-connected foreground objects are scattered away from the right edge.
    - The rightmost column contains a vertical five-cell color legend, ordered from size 1 to size 5.

    Rule:
    - For each Grey object, compute the row span and column span of its bounding box.
    - Let M be the larger of those two spans.
    - Replace every Grey pixel of that object with the M-th color in the right-edge legend.
    - Leave the background and the legend cells unchanged.

    Key insight: The object's area and exact shape do not matter; only the larger side of its bounding box selects the legend color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_rows = np.flatnonzero(g[:, -1] != BLACK)
    assert len(legend_rows) == LEGEND_SIZE, "expected a five-cell legend in the rightmost column"
    legend_rows = np.sort(legend_rows)
    legend_colors = [int(g[int(r), -1]) for r in legend_rows]
    assert len(set(legend_colors)) == LEGEND_SIZE, "legend colors must be distinct"
    size_to_color = {i + 1: color for i, color in enumerate(legend_colors)}

    for comp in _grey_components(g):
        rows = comp[:, 0]
        cols = comp[:, 1]
        row_span = int(rows.max() - rows.min() + 1)
        col_span = int(cols.max() - cols.min() + 1)
        span = max(row_span, col_span)
        assert span in size_to_color, "grey object span must be represented by the legend"
        out[rows, cols] = size_to_color[span]

    return out


def _random_shape(row_span: int, col_span: int) -> np.ndarray:
    if row_span == 1 and col_span == 1:
        return np.ones((1, 1), dtype=bool)

    start = random.choice(
        [
            (0, 0, row_span - 1, col_span - 1),
            (0, col_span - 1, row_span - 1, 0),
            (row_span - 1, 0, 0, col_span - 1),
            (row_span - 1, col_span - 1, 0, 0),
        ]
    )
    r, c, tr, tc = start
    mask = np.zeros((row_span, col_span), dtype=bool)
    mask[r, c] = True

    steps: list[tuple[int, int]] = []
    steps.extend([(1 if tr > r else -1, 0)] * abs(tr - r))
    steps.extend([(0, 1 if tc > c else -1)] * abs(tc - c))
    random.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True

    target_fill = random.randint(int(mask.sum()), row_span * col_span)
    while int(mask.sum()) < target_fill:
        frontier: list[tuple[int, int]] = []
        for rr in range(row_span):
            for cc in range(col_span):
                if mask[rr, cc]:
                    continue
                if (
                    (rr > 0 and mask[rr - 1, cc])
                    or (rr + 1 < row_span and mask[rr + 1, cc])
                    or (cc > 0 and mask[rr, cc - 1])
                    or (cc + 1 < col_span and mask[rr, cc + 1])
                ):
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    return mask


def _span_pair(max_span: int) -> tuple[int, int]:
    pairs = [(r, c) for r in range(1, max_span + 1) for c in range(1, max_span + 1) if max(r, c) == max_span]
    return random.choice(pairs)


def _can_place(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    return not bool(np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask))


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(mask):
        r = int(r0 + dr)
        c = int(c0 + dc)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_shape(g: np.ndarray, blocked: np.ndarray, mask: np.ndarray) -> bool:
    h, w = mask.shape
    grid_h, grid_w = g.shape
    usable_w = grid_w - 2
    if h > grid_h or w > usable_w:
        return False

    for _ in range(500):
        r0 = random.randint(0, grid_h - h)
        c0 = random.randint(0, usable_w - w)
        if not _can_place(blocked, mask, r0, c0):
            continue
        rr, cc = np.nonzero(mask)
        g[r0 + rr, c0 + cc] = GREY
        _mark_blocked(blocked, mask, r0, c0)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 28)
    width = random.randint(14, 28)
    grid = np.full((height, width), BLACK, dtype=int)

    legend_start = random.randint(0, height - LEGEND_SIZE)
    legend_colors = random.sample(PALETTE, LEGEND_SIZE)
    for i, color in enumerate(legend_colors):
        grid[legend_start + i, width - 1] = color

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:, width - 2 :] = True

    spans = list(range(1, LEGEND_SIZE + 1))
    spans.extend(random.choices(spans, k=random.randint(0, 4)))
    random.shuffle(spans)

    for max_span in spans:
        row_span, col_span = _span_pair(max_span)
        shape = _random_shape(row_span, col_span)
        if not _place_shape(grid, blocked, shape):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
