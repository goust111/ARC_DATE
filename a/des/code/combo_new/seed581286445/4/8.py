from __future__ import annotations

import random

import numpy as np


BG = 0
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
LEGEND_LEN_RANGE = (3, 5)
OBJECT_COUNT_RANGE = (2, 6)
OBJECT_MAX_WIDTH = 7
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(comp)
    return comps


def _legend_cells(corner: str, h: int, w: int, k: int) -> list[tuple[int, int]]:
    if corner == "tl":
        rows = range(0, k)
        col = 0
    elif corner == "tr":
        rows = range(0, k)
        col = w - 1
    elif corner == "bl":
        rows = range(h - k, h)
        col = 0
    elif corner == "br":
        rows = range(h - k, h)
        col = w - 1
    else:
        raise ValueError(corner)
    return [(int(r), int(col)) for r in rows]


def _extract_corner_legend(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[list[tuple[int, int]], list[int]]] = []

    def collect(start_r: int, col: int, step: int) -> None:
        if int(grid[start_r, col]) == bg:
            return

        inside_col = col + 1 if col == 0 else col - 1
        cells: list[tuple[int, int]] = []
        colors: list[int] = []
        seen_colors: set[int] = set()
        r = start_r
        while 0 <= r < h and int(grid[r, col]) != bg:
            color = int(grid[r, col])
            if color in seen_colors:
                return
            if 0 <= inside_col < w and int(grid[r, inside_col]) != bg:
                return
            cells.append((int(r), int(col)))
            colors.append(color)
            seen_colors.add(color)
            r += step

        if len(cells) < 3:
            return
        if step == 1 and 0 <= r < h and np.any(grid[r:, col] != bg):
            return
        if step == -1 and 0 <= r < h and np.any(grid[: r + 1, col] != bg):
            return

        if step == -1:
            cells.reverse()
            colors.reverse()
        candidates.append((cells, colors))

    collect(0, 0, 1)
    collect(0, w - 1, 1)
    collect(h - 1, 0, -1)
    collect(h - 1, w - 1, -1)

    assert candidates, "expected a vertical legend on one corner"
    cells, colors = max(candidates, key=lambda item: len(item[0]))
    return cells, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome object by row using the color legend hidden in a grid corner.

    Input structure:
    - Background cells are black, and one corner contains a 1-pixel-wide vertical legend bar.
    - The legend bar is a contiguous stack of K distinct colors read from top to bottom.
    - Away from the corner, several disjoint 4-connected monochrome objects appear on the background.
    - Every foreground object spans exactly K rows in its bounding box.

    Rule:
    - Read the corner legend from top to bottom to obtain the ordered color sequence.
    - Remove the legend itself back to the background color.
    - For each remaining 4-connected monochrome object, find the top row of its bounding box.
    - Recolor every pixel in the object's i-th row layer with the i-th legend color.

    Key insight:
    - The small corner strip is not another object; it is a row-by-row color key whose vertical order
      directly maps onto each object's internal top-to-bottom layers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_pts, legend_colors = _extract_corner_legend(g, bg)
    k = len(legend_colors)

    out = g.copy()
    for r, c in legend_pts:
        out[r, c] = bg

    fg_mask = g != bg
    for r, c in legend_pts:
        fg_mask[r, c] = False

    for comp in _components(fg_mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        top = min(r for r, _ in comp)
        bottom = max(r for r, _ in comp)
        if bottom - top + 1 != k:
            continue
        for r, c in comp:
            out[r, c] = int(legend_colors[r - top])

    return out


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _make_object_mask(k: int, max_width: int) -> np.ndarray:
    max_width = max(3, min(max_width, OBJECT_MAX_WIDTH))
    for _ in range(80):
        width = random.randint(3, max_width)
        segments: list[tuple[int, int]] = []

        length = random.randint(1, min(4, width))
        start = random.randint(0, width - length)
        segments.append((start, start + length - 1))

        for _row in range(1, k):
            prev_start, prev_end = segments[-1]
            anchor = random.randint(prev_start, prev_end)
            length = random.randint(1, min(4, width))
            lo = max(0, anchor - length + 1)
            hi = min(anchor, width - length)
            start = random.randint(lo, hi)
            segments.append((start, start + length - 1))

        if len(set(segments)) == 1 and k > 1:
            row = random.randint(1, k - 1)
            prev_start, prev_end = segments[row - 1]
            anchor = random.randint(prev_start, prev_end)
            options: list[tuple[int, int]] = []
            for length in range(1, min(4, width) + 1):
                lo = max(0, anchor - length + 1)
                hi = min(anchor, width - length)
                for start in range(lo, hi + 1):
                    seg = (start, start + length - 1)
                    if seg != segments[row]:
                        options.append(seg)
            if options:
                segments[row] = random.choice(options)

        mask = np.zeros((k, width), dtype=bool)
        for r, (start, end) in enumerate(segments):
            mask[r, start : end + 1] = True

        cols = np.where(mask.any(axis=0))[0]
        if cols.size == 0:
            continue
        mask = mask[:, cols[0] : cols[-1] + 1]
        if random.random() < 0.5:
            mask = mask[:, ::-1]
        if mask.shape[1] >= 2 and int(mask.sum()) >= k + 1:
            return mask

    return np.ones((k, 2), dtype=bool)


def _expected_output(
    grid: np.ndarray,
    legend_pts: list[tuple[int, int]],
    legend_colors: list[int],
    objects: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in legend_pts:
        out[r, c] = BG
    for top, left, mask in objects:
        for r, c in np.argwhere(mask):
            out[top + int(r), left + int(c)] = int(legend_colors[int(r)])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    k = random.randint(*LEGEND_LEN_RANGE)
    if h <= k + 2 or w <= 6:
        return None

    grid = np.full((h, w), BG, dtype=int)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_colors = random.sample(list(range(1, 10)), k)
    object_color = random.choice([c for c in range(1, 10) if c not in legend_colors])
    legend_pts = _legend_cells(corner, h, w, k)

    for (r, c), color in zip(legend_pts, legend_colors, strict=False):
        grid[r, c] = int(color)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    _mark_blocked(blocked, legend_pts)

    max_objects = min(
        OBJECT_COUNT_RANGE[1],
        max(OBJECT_COUNT_RANGE[0], (h * w) // 90 + 2),
    )
    n_objects = random.randint(OBJECT_COUNT_RANGE[0], max_objects)

    objects: list[tuple[int, int, np.ndarray]] = []
    for _ in range(n_objects):
        placed = False
        for _ in range(120):
            mask = _make_object_mask(k, min(OBJECT_MAX_WIDTH, w - 2))
            mh, mw = mask.shape
            if h - mh - 1 < 1 or w - mw - 1 < 1:
                continue
            top = random.randint(1, h - mh - 1)
            left = random.randint(1, w - mw - 1)
            cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = object_color
            _mark_blocked(blocked, cells)
            objects.append((top, left, mask))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, legend_pts, legend_colors, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
