from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
MAGENTA = 6
PALETTE_COLORS = (1, 2, 3, 4, 7, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    candidates: list[tuple[int, int, int, int, list[int]]] = []
    for comp in _components(grid == GREY):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 3 or height < 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == GREY):
            continue
        if not np.all(patch[-1, :] == GREY):
            continue
        if not np.all(patch[:, 0] == GREY):
            continue
        if not np.all(patch[:, -1] == GREY):
            continue
        inner = patch[1:-1, 1]
        if inner.size == 0:
            continue
        if np.any(inner == bg) or np.any(inner == GREY) or np.any(inner == MAGENTA):
            continue
        colors = [int(v) for v in inner]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((r1, c1, r0, c0, colors))

    if not candidates:
        raise AssertionError("main: failed to locate the grey-framed vertical legend")

    _, _, r0, c0, colors = max(candidates, key=lambda item: (item[0], item[1], len(item[4])))
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + len(colors) + 2, c0 : c0 + 3] = True
    return colors, mask


def _segment_orientation(grid: np.ndarray, comp: np.ndarray) -> tuple[str, int]:
    rows = sorted(int(v) for v in np.unique(comp[:, 0]))
    cols = sorted(int(v) for v in np.unique(comp[:, 1]))
    h, w = grid.shape

    if len(comp) == 1:
        r, c = map(int, comp[0])
        horizontal = c - 1 >= 0 and c + 1 < w and grid[r, c - 1] == MAGENTA and grid[r, c + 1] == MAGENTA
        vertical = r - 1 >= 0 and r + 1 < h and grid[r - 1, c] == MAGENTA and grid[r + 1, c] == MAGENTA
        if horizontal ^ vertical:
            return ("H", r) if horizontal else ("V", c)
        raise AssertionError("main: single-pixel segment must have exactly one valid arrow axis")

    if len(rows) == 1:
        r = rows[0]
        c0, c1 = cols[0], cols[-1]
        if cols != list(range(c0, c1 + 1)):
            raise AssertionError("main: horizontal segment must be contiguous")
        if c0 - 1 < 0 or c1 + 1 >= w:
            raise AssertionError("main: horizontal segment endpoints must have room for arrows")
        if grid[r, c0 - 1] != MAGENTA or grid[r, c1 + 1] != MAGENTA:
            raise AssertionError("main: horizontal segment endpoints must have magenta arrows")
        return "H", r

    if len(cols) == 1:
        c = cols[0]
        r0, r1 = rows[0], rows[-1]
        if rows != list(range(r0, r1 + 1)):
            raise AssertionError("main: vertical segment must be contiguous")
        if r0 - 1 < 0 or r1 + 1 >= h:
            raise AssertionError("main: vertical segment endpoints must have room for arrows")
        if grid[r0 - 1, c] != MAGENTA or grid[r1 + 1, c] != MAGENTA:
            raise AssertionError("main: vertical segment endpoints must have magenta arrows")
        return "V", c

    raise AssertionError("main: every non-legend colored component must be a straight segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend scattered colored segments into a woven grid using the legend's top-to-bottom layering order.

    Input structure:
    - The background is black.
    - A small grey-framed vertical legend sits near the lower-right corner; its interior contains one colored pixel per row.
    - Elsewhere, there are isolated single-cell-wide horizontal and vertical colored segments, and each segment has one adjacent magenta pixel beyond each endpoint along its extension axis.

    Rule:
    - Read the legend interior from top to bottom to obtain the color priority, with higher rows meaning higher visual layers.
    - For every segment, use the two magenta endpoint markers to confirm its axis, then extend the segment from both ends all the way to the grid boundary using the segment's own color.
    - Remove all magenta markers, and whenever extended lines overlap, keep only the color whose legend entry is highest. Restore the legend unchanged.

    Key insight:
    - The magenta pixels are directional endpoint cues, while the legend order controls occlusion at every crossing rather than acting as a color-substitution table.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_colors, legend_mask = _legend_info(g, bg)
    priority = {color: idx for idx, color in enumerate(legend_colors)}

    out = np.full(g.shape, bg, dtype=int)
    top_rank = np.full(g.shape, len(legend_colors) + 1, dtype=int)
    found_segment = False

    for color in legend_colors:
        for comp in _components((g == color) & ~legend_mask):
            found_segment = True
            orient, axis_value = _segment_orientation(g, comp)
            rank = priority[color]
            if orient == "H":
                r = axis_value
                for c in range(g.shape[1]):
                    if legend_mask[r, c]:
                        continue
                    if rank < top_rank[r, c]:
                        top_rank[r, c] = rank
                        out[r, c] = color
            else:
                c = axis_value
                for r in range(g.shape[0]):
                    if legend_mask[r, c]:
                        continue
                    if rank < top_rank[r, c]:
                        top_rank[r, c] = rank
                        out[r, c] = color

    if not found_segment:
        raise AssertionError("main: expected at least one colored segment outside the legend")

    out[legend_mask] = g[legend_mask]
    return out


def _paint_legend(grid: np.ndarray, colors: list[int], r0: int, c0: int) -> None:
    height = len(colors) + 2
    grid[r0 : r0 + height, c0 : c0 + 3] = BG
    grid[r0, c0 : c0 + 3] = GREY
    grid[r0 + height - 1, c0 : c0 + 3] = GREY
    grid[r0 : r0 + height, c0] = GREY
    grid[r0 : r0 + height, c0 + 2] = GREY
    for idx, color in enumerate(colors):
        grid[r0 + 1 + idx, c0 + 1] = color


def _reserve_box(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _cells_clear(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _block_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _segment_cells(spec: dict[str, int]) -> list[tuple[int, int]]:
    if spec["orientation"] == "H":
        r = spec["row"]
        c0 = spec["start"]
        c1 = spec["end"]
        return [(r, c) for c in range(c0, c1 + 1)] + [(r, c0 - 1), (r, c1 + 1)]
    c = spec["col"]
    r0 = spec["start"]
    r1 = spec["end"]
    return [(r, c) for r in range(r0, r1 + 1)] + [(r0 - 1, c), (r1 + 1, c)]


def _place_horizontal(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    max_col: int,
    used_rows: set[int],
) -> dict[str, int] | None:
    h, _ = grid.shape
    if max_col < 5:
        return None
    max_len = min(7, max_col - 3)
    if max_len < 2:
        return None
    for _ in range(PLACE_RETRIES):
        row = rng.randint(2, h - 3)
        if row in used_rows:
            continue
        length = rng.randint(2, max_len)
        start = rng.randint(2, max_col - length)
        end = start + length - 1
        spec = {"orientation": "H", "row": row, "start": start, "end": end, "color": color}
        cells = _segment_cells(spec)
        if not _cells_clear(blocked, cells):
            continue
        for r, c in cells[:-2]:
            grid[r, c] = color
        for r, c in cells[-2:]:
            grid[r, c] = MAGENTA
        _block_cells(blocked, cells)
        used_rows.add(row)
        return spec
    return None


def _place_vertical(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    used_cols: set[int],
) -> dict[str, int] | None:
    h, w = grid.shape
    if w < 7 or h < 7:
        return None
    max_len = min(7, h - 5)
    if max_len < 2:
        return None
    for _ in range(PLACE_RETRIES):
        col = rng.randint(2, w - 3)
        if col in used_cols:
            continue
        length = rng.randint(2, max_len)
        start = rng.randint(2, h - length - 2)
        end = start + length - 1
        spec = {"orientation": "V", "col": col, "start": start, "end": end, "color": color}
        cells = _segment_cells(spec)
        if not _cells_clear(blocked, cells):
            continue
        for r, c in cells[:-2]:
            grid[r, c] = color
        for r, c in cells[-2:]:
            grid[r, c] = MAGENTA
        _block_cells(blocked, cells)
        used_cols.add(col)
        return spec
    return None


def _expected_output(
    grid: np.ndarray,
    legend_colors: list[int],
    legend_bbox: tuple[int, int, int, int],
    specs: list[dict[str, int]],
) -> np.ndarray:
    bg = _background_color(grid)
    out = np.full_like(grid, bg)
    top_rank = np.full(grid.shape, len(legend_colors) + 1, dtype=int)
    priority = {color: idx for idx, color in enumerate(legend_colors)}
    r0, c0, r1, c1 = legend_bbox
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for spec in specs:
        color = spec["color"]
        rank = priority[color]
        if spec["orientation"] == "H":
            row = spec["row"]
            for col in range(grid.shape[1]):
                if legend_mask[row, col]:
                    continue
                if rank < top_rank[row, col]:
                    top_rank[row, col] = rank
                    out[row, col] = color
        else:
            col = spec["col"]
            for row in range(grid.shape[0]):
                if legend_mask[row, col]:
                    continue
                if rank < top_rank[row, col]:
                    top_rank[row, col] = rank
                    out[row, col] = color

    out[legend_mask] = grid[legend_mask]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    total_h = rng.randint(1, 3)
    total_v = rng.randint(1, 3)
    total_segments = total_h + total_v
    n_colors = rng.randint(2, min(4, total_segments, len(PALETTE_COLORS)))
    legend_colors = rng.sample(PALETTE_COLORS, n_colors)

    legend_height = n_colors + 2
    bottom_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    if legend_height + bottom_margin > h:
        return None

    legend_r0 = h - legend_height - bottom_margin
    legend_c0 = w - 3 - right_margin
    if legend_r0 < 0 or legend_c0 < 3:
        return None

    max_h_col = legend_c0 - 2
    if max_h_col < 5:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_legend(grid, legend_colors, legend_r0, legend_c0)

    blocked = np.zeros((h, w), dtype=bool)
    _reserve_box(blocked, legend_r0, legend_c0, legend_r0 + legend_height - 1, legend_c0 + 2, pad=1)

    specs: list[dict[str, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    horiz_colors = [legend_colors[0]]
    vert_colors = [legend_colors[-1]]
    spare = list(legend_colors[1:-1])
    while len(spare) < total_segments - 2:
        spare.append(rng.choice(legend_colors))
    rng.shuffle(spare)
    while len(horiz_colors) < total_h and spare:
        horiz_colors.append(spare.pop())
    while len(vert_colors) < total_v and spare:
        vert_colors.append(spare.pop())
    while len(horiz_colors) < total_h:
        horiz_colors.append(rng.choice(legend_colors))
    while len(vert_colors) < total_v:
        vert_colors.append(rng.choice(legend_colors))
    rng.shuffle(horiz_colors)
    rng.shuffle(vert_colors)
    if horiz_colors[0] == vert_colors[0]:
        vert_colors[0] = legend_colors[-1]

    for color in horiz_colors:
        spec = _place_horizontal(rng, grid, blocked, color, max_h_col, used_rows)
        if spec is None:
            return None
        specs.append(spec)

    for color in vert_colors:
        spec = _place_vertical(rng, grid, blocked, color, used_cols)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_output(
        grid,
        legend_colors,
        (legend_r0, legend_c0, legend_r0 + legend_height - 1, legend_c0 + 2),
        specs,
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, actual):
        return None
    if len({spec["color"] for spec in specs}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid woven-line sample")
