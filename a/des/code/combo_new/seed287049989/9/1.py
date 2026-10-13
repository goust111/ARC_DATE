import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _legend_mapping(grid: np.ndarray) -> tuple[int, dict[int, int]]:
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        cols = np.flatnonzero(row != BLACK)
        if cols.size == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1])
        span = row[c0 : c1 + 1]
        if np.any(span == BLACK):
            continue

        runs: list[tuple[int, int]] = []
        c = c0
        while c <= c1:
            color = int(row[c])
            d = c
            while d <= c1 and int(row[d]) == color:
                d += 1
            runs.append((color, d - c))
            c = d

        if len(runs) < 2:
            continue
        if any(color == GREY or width < 1 or width > 4 for color, width in runs):
            continue
        if len({color for color, _ in runs}) != len(runs):
            continue
        return r, {color: width for color, width in runs}

    raise AssertionError("expected one horizontal legend strip near the top")


def _rect_frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for pts in _components_mask(grid == GREY):
        rs = pts[:, 0]
        cs = pts[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))

        if set(map(tuple, pts)) == border:
            boxes.append((r0, c0, r1, c1))

    boxes.sort()
    assert boxes, "expected at least one grey rectangular frame"
    return boxes


def _centered_shrunk_bbox(r0: int, c0: int, r1: int, c1: int, shrink: int) -> tuple[int, int, int, int]:
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    inner_h = max(1, outer_h - 2 * shrink)
    inner_w = max(1, outer_w - 2 * shrink)
    nr0 = (r0 + r1 - inner_h + 1) // 2
    nc0 = (c0 + c1 - inner_w + 1) // 2
    nr1 = nr0 + inner_h - 1
    nc1 = nc0 + inner_w - 1
    return nr0, nc0, nr1, nc1


def _draw_box_or_fill(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h >= 3 and w >= 3:
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    else:
        out[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each grey rectangular frame inward by the width encoded by its marker color's legend segment.

    Input structure:
    - The background is black.
    - Near the top sits one horizontal legend strip made of adjacent colored runs, where each run has a distinct width from 1 to 4.
    - Lower in the grid are several grey hollow rectangles drawn as 1-cell-thick frames.
    - Inside each grey frame are one or more scattered marker pixels, all with the same non-grey color that also appears in the legend strip.

    Rule:
    - Read the top legend strip as a color-to-width lookup table, using each color run's width as that color's shrink amount.
    - For every grey frame, identify its interior marker color C and delete those marker pixels from the output.
    - Centered on the original frame, draw a new C-colored rectangle after shrinking the frame inward by C's legend width.
    - If the shrunken rectangle is still at least 3 cells tall and 3 cells wide, draw only its border; otherwise fill the entire shrunken rectangle solid.

    Key insight:
    The legend strip is not decorative: each color's visible run length is the hidden geometric parameter that determines how far the matching frame must contract inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    legend_row, mapping = _legend_mapping(g)
    boxes = _rect_frame_boxes(g)

    out = g.copy()
    for r0, c0, r1, c1 in boxes:
        assert r0 > legend_row, "frames should lie below the legend strip"
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in np.unique(interior) if int(v) not in (BLACK, GREY)]
        assert len(colors) == 1, "each frame interior must contain markers of exactly one color"
        color = colors[0]
        assert color in mapping, "marker color must appear in the legend"

        marker_mask = (out[r0 + 1 : r1, c0 + 1 : c1] != BLACK) & (out[r0 + 1 : r1, c0 + 1 : c1] != GREY)
        out[r0 + 1 : r1, c0 + 1 : c1][marker_mask] = BLACK

        shrunk = _centered_shrunk_bbox(r0, c0, r1, c1, mapping[color])
        _draw_box_or_fill(out, shrunk, color)

    return out


def _choose_box_size(rng: random.Random, shrink: int, solid: bool) -> tuple[int, int]:
    if solid:
        if rng.random() < 0.5:
            inner_h = rng.choice([1, 2])
            inner_w = rng.randint(1, 5)
        else:
            inner_h = rng.randint(1, 5)
            inner_w = rng.choice([1, 2])
    else:
        inner_h = rng.randint(3, 5)
        inner_w = rng.randint(3, 6)
    return inner_h + 2 * shrink, inner_w + 2 * shrink


def _marker_positions(rng: random.Random, box_h: int, box_w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(1, box_h - 1) for c in range(1, box_w - 1)]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 2 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return cells[:count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_segments = rng.randint(2, 4)
    widths = rng.sample([1, 2, 3, 4], n_segments)
    colors = rng.sample(list(MARKER_COLORS), n_segments)
    legend = list(zip(colors, widths))
    rng.shuffle(legend)
    mapping = {color: width for color, width in legend}
    legend_len = sum(width for _, width in legend)

    n_boxes = rng.randint(2, 4)
    branches = [False, True]  # False -> hollow inner frame, True -> solid fill
    while len(branches) < n_boxes:
        branches.append(rng.random() < 0.5)
    rng.shuffle(branches)

    box_colors = rng.sample(colors, min(len(colors), n_boxes))
    while len(box_colors) < n_boxes:
        box_colors.append(rng.choice(colors))
    rng.shuffle(box_colors)

    specs = []
    for color, solid in zip(box_colors, branches):
        box_h, box_w = _choose_box_size(rng, mapping[color], solid)
        if box_h > 16 or box_w > 16:
            return None
        interior_area = max(1, (box_h - 2) * (box_w - 2))
        marker_count = rng.randint(1, min(3, interior_area))
        specs.append(
            {
                "color": color,
                "solid": solid,
                "h": box_h,
                "w": box_w,
                "markers": _marker_positions(rng, box_h, box_w, marker_count),
            }
        )

    rng.shuffle(specs)
    if n_boxes >= 3:
        rows = [specs[::2], specs[1::2]]
    elif n_boxes == 2 and rng.random() < 0.5:
        rows = [[specs[0]], [specs[1]]]
    else:
        rows = [specs]
    rows = [row for row in rows if row]

    row_gaps = [[rng.randint(2, 4) for _ in range(len(row) - 1)] for row in rows]
    row_widths = [
        sum(spec["w"] for spec in row) + sum(gaps)
        for row, gaps in zip(rows, row_gaps)
    ]
    row_heights = [max(spec["h"] for spec in row) for row in rows]

    legend_row = rng.randint(0, 1)
    gap_after_legend = rng.randint(2, 3)
    vertical_gap = rng.randint(2, 4) if len(rows) == 2 else 0
    needed_w = max(legend_len + 2, max(width + 2 for width in row_widths))
    needed_h = legend_row + 1 + gap_after_legend + sum(row_heights) + vertical_gap * (len(rows) - 1) + 1
    if needed_w > 30 or needed_h > 30:
        return None

    grid_w = rng.randint(needed_w, min(30, needed_w + 4))
    grid_h = rng.randint(needed_h, min(30, needed_h + 4))

    grid = np.zeros((grid_h, grid_w), dtype=int)

    legend_c0 = rng.randint(1, grid_w - legend_len - 1)
    c = legend_c0
    for color, width in legend:
        grid[legend_row, c : c + width] = color
        c += width

    extra_down = grid_h - needed_h
    current_r = legend_row + 1 + gap_after_legend + rng.randint(0, extra_down)
    for row, gaps, row_h, row_w in zip(rows, row_gaps, row_heights, row_widths):
        start_c = rng.randint(1, grid_w - row_w - 1)
        c = start_c
        for idx, spec in enumerate(row):
            r0 = current_r + rng.randint(0, row_h - spec["h"])
            c0 = c
            r1 = r0 + spec["h"] - 1
            c1 = c0 + spec["w"] - 1

            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            for mr, mc in spec["markers"]:
                grid[r0 + mr, c0 + mc] = spec["color"]

            if idx < len(gaps):
                c += spec["w"] + gaps[idx]
        current_r += row_h + vertical_gap

    if not any(spec["solid"] for spec in specs) or not any(not spec["solid"] for spec in specs):
        return None

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
