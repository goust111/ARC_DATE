from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
STYLES = ("solid", "dashed", "double")

LEGEND_HEIGHT = 5
LEGEND_WIDTHS = (5, 7, 9)
MIN_INTERIOR = 4
MAX_INTERIOR = 12
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != int(bg):
            candidates.append((int(c), int(col[0])))
    assert len(candidates) == 1, "expected a unique full-height separator column"
    return candidates[0]


def _legend_style(panel: np.ndarray, color: int, bg: int) -> str:
    box = _bbox(panel == int(color))
    assert box is not None, "legend color must appear in the left panel"
    r0, c0, r1, c1 = box
    block = panel[r0 : r1 + 1, c0 : c1 + 1]
    holes = block == int(bg)
    row_counts = holes.sum(axis=1)
    active_rows = [r for r, count in enumerate(row_counts.tolist()) if count > 0]

    if len(active_rows) == 2 and all(int(row_counts[r]) == block.shape[1] for r in active_rows):
        return "double"

    assert len(active_rows) == 1, "single-track legend must have one active hole row"
    pattern = holes[active_rows[0]]
    if np.all(pattern):
        return "solid"

    alt0 = np.zeros(block.shape[1], dtype=bool)
    alt0[::2] = True
    alt1 = np.zeros(block.shape[1], dtype=bool)
    alt1[1::2] = True
    assert np.array_equal(pattern, alt0) or np.array_equal(pattern, alt1), "unknown legend pattern"
    return "dashed"


def _endpoint_pair(panel: np.ndarray, color: int) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(panel == int(color))]
    assert len(coords) == 2, "each work-panel color must have exactly two endpoints"
    a, b = coords
    if a[0] == b[0]:
        return (a, b) if a[1] <= b[1] else (b, a)
    assert a[1] == b[1], "endpoints must align horizontally or vertically"
    return (a, b) if a[0] <= b[0] else (b, a)


def _draw_connection(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: int,
    style: str,
    bg: int,
) -> None:
    if start[0] == end[0]:
        r = int(start[0])
        c0, c1 = int(start[1]), int(end[1])
        assert c0 < c1
        if style == "solid":
            out[r, c0 + 1 : c1] = int(color)
            return
        if style == "dashed":
            for step, c in enumerate(range(c0 + 1, c1), start=1):
                if step % 2 == 1:
                    out[r, c] = int(color)
            return
        assert style == "double"
        assert 0 < r < out.shape[0] - 1, "double horizontal line needs one row of clearance"
        if c1 - c0 > 1:
            out[r - 1, c0 + 1 : c1] = int(color)
            out[r + 1, c0 + 1 : c1] = int(color)
        return

    c = int(start[1])
    r0, r1 = int(start[0]), int(end[0])
    assert r0 < r1 and c == int(end[1])
    if style == "solid":
        out[r0 + 1 : r1, c] = int(color)
        return
    if style == "dashed":
        for step, r in enumerate(range(r0 + 1, r1), start=1):
            if step % 2 == 1:
                out[r, c] = int(color)
        return
    assert style == "double"
    assert 0 < c < out.shape[1] - 1, "double vertical line needs one column of clearance"
    if r1 - r0 > 1:
        out[r0 + 1 : r1, c - 1] = int(color)
        out[r0 + 1 : r1, c + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Use each legend block's internal black path to connect the matching endpoint pair in the work panel.

    Input structure:
    - The background is black.
    - A single full-height grey column separates a left legend panel from a right work panel.
    - The legend contains 2-3 colored rectangles; the black holes inside each rectangle form a solid line, an alternating dash pattern, or a double rail.
    - The work panel contains one same-color endpoint pair for every legend color, and each pair lies on one row or one column.

    Rule:
    - Detect the separator, then read each legend color's bounding box in the left panel.
    - Interpret the black negative-space path inside that colored box as the line style for the same color.
    - In the right panel, connect the matching endpoint pair with that style, rotating the template if the pair is vertical, and never recolor the endpoints themselves.
    - Keep the legend panel and the grey separator unchanged.

    Key insight:
    - The informative template is not the colored rectangle itself but the black path carved out of it, so the solve step is color matching plus "negative-space" style transfer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col, sep_color = _find_separator_col(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    colors = sorted(int(c) for c in np.unique(left) if int(c) not in {int(bg), int(sep_color)})
    assert 2 <= len(colors) <= 3, "expected 2-3 legend colors"

    out = g.copy()
    for color in colors:
        style = _legend_style(left, color, bg)
        start_local, end_local = _endpoint_pair(right, color)
        start = (int(start_local[0]), int(start_local[1] + sep_col + 1))
        end = (int(end_local[0]), int(end_local[1] + sep_col + 1))
        _draw_connection(out, start, end, color=color, style=style, bg=bg)
    return out


def _paint_legend_block(grid: np.ndarray, top: int, left: int, width: int, color: int, style: str) -> None:
    grid[top : top + LEGEND_HEIGHT, left : left + width] = int(color)
    center = top + LEGEND_HEIGHT // 2
    if style == "solid":
        grid[center, left : left + width] = BG
        return
    if style == "dashed":
        grid[center, left : left + width : 2] = BG
        return
    assert style == "double"
    grid[center - 1, left : left + width] = BG
    grid[center + 1, left : left + width] = BG


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0] - 1, int(r1) + pad)
    cc1 = min(mask.shape[1] - 1, int(c1) + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_positions(height: int, left_width: int, widths: list[int]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, left_width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(widths)
    order = list(range(len(widths)))
    random.shuffle(order)

    for idx in order:
        width = widths[idx]
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - LEGEND_HEIGHT):
            for c in range(1, left_width - width):
                if occupied[r - 1 : r + LEGEND_HEIGHT + 1, c - 1 : c + width + 1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        _reserve(occupied, top, left, top + LEGEND_HEIGHT - 1, left + width - 1, pad=1)

    assert all(p is not None for p in placements)
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _pair_cells(
    style: str,
    orientation: str,
    a0: int,
    a1: int,
    fixed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if orientation == "H":
        endpoints = [(fixed, a0), (fixed, a1)]
        if style == "solid":
            line = [(fixed, c) for c in range(a0 + 1, a1)]
        elif style == "dashed":
            line = [(fixed, c) for step, c in enumerate(range(a0 + 1, a1), start=1) if step % 2 == 1]
        else:
            line = [(fixed - 1, c) for c in range(a0 + 1, a1)] + [(fixed + 1, c) for c in range(a0 + 1, a1)]
        return endpoints, line

    endpoints = [(a0, fixed), (a1, fixed)]
    if style == "solid":
        line = [(r, fixed) for r in range(a0 + 1, a1)]
    elif style == "dashed":
        line = [(r, fixed) for step, r in enumerate(range(a0 + 1, a1), start=1) if step % 2 == 1]
    else:
        line = [(r, fixed - 1) for r in range(a0 + 1, a1)] + [(r, fixed + 1) for r in range(a0 + 1, a1)]
    return endpoints, line


def _pair_candidates(height: int, width: int, style: str, orientation: str, reserved: np.ndarray) -> list[tuple]:
    candidates = []
    max_interior = min(MAX_INTERIOR, width - 2 if orientation == "H" else height - 2)
    if max_interior < MIN_INTERIOR:
        return candidates

    if orientation == "H":
        row_range = range(1, height - 1) if style == "double" else range(height)
        for r in row_range:
            for c0 in range(width):
                for c1 in range(c0 + MIN_INTERIOR + 1, min(width, c0 + max_interior + 2)):
                    endpoints, line = _pair_cells(style, orientation, c0, c1, r)
                    cells = endpoints + line
                    rs = [cell[0] for cell in cells]
                    cs = [cell[1] for cell in cells]
                    r0, r1 = max(0, min(rs) - 1), min(height - 1, max(rs) + 1)
                    c0b, c1b = max(0, min(cs) - 1), min(width - 1, max(cs) + 1)
                    if reserved[r0 : r1 + 1, c0b : c1b + 1].any():
                        continue
                    candidates.append((endpoints, line, (r0, c0b, r1, c1b)))
        return candidates

    col_range = range(1, width - 1) if style == "double" else range(width)
    for c in col_range:
        for r0 in range(height):
            for r1 in range(r0 + MIN_INTERIOR + 1, min(height, r0 + max_interior + 2)):
                endpoints, line = _pair_cells(style, orientation, r0, r1, c)
                cells = endpoints + line
                rs = [cell[0] for cell in cells]
                cs = [cell[1] for cell in cells]
                rr0, rr1 = max(0, min(rs) - 1), min(height - 1, max(rs) + 1)
                cc0, cc1 = max(0, min(cs) - 1), min(width - 1, max(cs) + 1)
                if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((endpoints, line, (rr0, cc0, rr1, cc1)))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice((2, 3))
    colors = random.sample(COLOR_POOL, n_colors)
    styles = random.sample(STYLES, n_colors)
    widths = [random.choice(LEGEND_WIDTHS) for _ in range(n_colors)]

    min_height = n_colors * LEGEND_HEIGHT + (n_colors + 1)
    height = random.randint(min_height, min(24, min_height + 6))
    left_width = max(widths) + random.randint(3, 5)
    max_work_width = 30 - left_width - 1
    if max_work_width < 10:
        return None
    work_width = random.randint(10, min(18, max_work_width))
    width = left_width + 1 + work_width

    legend_pos = _legend_positions(height, left_width, widths)
    if legend_pos is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    inp[:, left_width] = SEPARATOR

    for (top, left), block_width, color, style in zip(legend_pos, widths, colors, styles, strict=False):
        _paint_legend_block(inp, top, left, block_width, color, style)

    out = inp.copy()
    reserved = np.zeros((height, work_width), dtype=bool)
    pair_specs = list(zip(colors, styles, strict=False))
    random.shuffle(pair_specs)

    for color, style in pair_specs:
        orientation = random.choice(("H", "V"))
        candidates = _pair_candidates(height, work_width, style, orientation, reserved)
        if not candidates:
            orientation = "V" if orientation == "H" else "H"
            candidates = _pair_candidates(height, work_width, style, orientation, reserved)
        if not candidates:
            return None

        endpoints, _line, (r0, c0, r1, c1) = random.choice(candidates)
        for r, c in endpoints:
            inp[r, left_width + 1 + c] = int(color)
            out[r, left_width + 1 + c] = int(color)
        start = (int(endpoints[0][0]), int(left_width + 1 + endpoints[0][1]))
        end = (int(endpoints[1][0]), int(left_width + 1 + endpoints[1][1]))
        _draw_connection(out, start, end, color=int(color), style=style, bg=BG)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
