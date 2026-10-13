from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bottom_legend(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    entries: list[tuple[int, int]] = []
    c = 0
    while c < row.size:
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        while c < row.size and int(row[c]) != bg:
            c += 1
        if c - start == 1:
            entries.append((int(start), int(row[start])))
    assert entries, "expected a bottom-row legend made of isolated color cells"
    colors = [color for _, color in entries]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return entries


def _is_frame_component(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    cell_set = set(cells)
    if len(cell_set) != 2 * (height + width) - 4:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in cell_set:
                return False
    return True


def _frame_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    comps = _components(grid == color)
    return [comp for comp in comps if _is_frame_component(comp)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangular outline using the legend color indexed by its interior dot count.

    Input structure:
    - The background is the most common color.
    - The bottom row contains a horizontal legend of distinct single-cell colors, ordered left to right and separated by background cells.
    - Above the legend are one or more one-cell-thick closed rectangular outlines drawn in a shared border color.
    - Each rectangle encloses background, plus one or more disconnected border-color dot markers inside it.

    Rule:
    - Read the bottom-row legend left to right so the first legend cell corresponds to count 1, the second to count 2, and so on.
    - For each rectangular outline, count the number of disconnected border-color dot markers strictly inside the rectangle.
    - Recolor every border pixel of that rectangle with the legend color at the matching count.
    - Erase all interior dot markers back to the background, while leaving the legend unchanged.

    Key insight:
    - The interior dots are an index, not content to preserve: each rectangle must be solved by count -> legend lookup -> border recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    legend = _bottom_legend(g[-1], bg)
    legend_colors = [color for _, color in legend]
    work = g[:-1, :]

    frame_color = None
    frames: list[list[tuple[int, int]]] = []
    for color in map(int, np.unique(work)):
        if color == bg:
            continue
        color_frames = _frame_components(work, color)
        if len(color_frames) > len(frames):
            frame_color = color
            frames = color_frames

    assert frame_color is not None and frames, "expected at least one rectangular frame color"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = _bbox(frame)
        interior_mask = g[r0 + 1 : r1, c0 + 1 : c1] == frame_color
        marker_count = len(_components(interior_mask))
        assert 1 <= marker_count <= len(legend_colors), "marker count must index the legend"
        target_color = legend_colors[marker_count - 1]

        for r, c in frame:
            out[r, c] = target_color
        out[r0 + 1 : r1, c0 + 1 : c1][interior_mask] = bg

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 30)
LEGEND_LEN_RANGE = (2, 6)
RECT_COUNT_RANGE = (1, 4)
RECT_INNER_HEIGHT_RANGE = (3, 8)
RECT_INNER_WIDTH_RANGE = (3, 9)
LEGEND_GAP_CHOICES = (1, 2)
GEN_RETRIES = 500


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _frame_cells(r0: int, c0: int, height: int, width: int) -> list[tuple[int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _marker_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    for _ in range(80):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _sample_rect_shape(count: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    for _ in range(100):
        inner_h = random.randint(*RECT_INNER_HEIGHT_RANGE)
        inner_w = random.randint(*RECT_INNER_WIDTH_RANGE)
        if ((inner_h - 1) // 2) * ((inner_w - 1) // 2) < count:
            continue
        markers = _marker_positions(inner_h, inner_w, count)
        if markers is None:
            continue
        return inner_h + 2, inner_w + 2, markers
    return None


def _legend_columns(width: int, length: int) -> list[int] | None:
    for _ in range(80):
        gaps = [random.choice(LEGEND_GAP_CHOICES) for _ in range(length - 1)]
        span = length + sum(gaps)
        if span > width:
            continue
        start = random.randint(0, width - span)
        cols = [start]
        cursor = start
        for gap in gaps:
            cursor += 1 + gap
            cols.append(cursor)
        return cols
    return None


def _expected_output(
    grid: np.ndarray,
    bg: int,
    legend_colors: list[int],
    rects: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        k = int(rect["count"])
        target = legend_colors[k - 1]
        for r, c in rect["frame"]:
            out[r, c] = target
        for r, c in rect["markers"]:
            out[r, c] = bg
    return out


def _attempt_grid() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_legend = min(LEGEND_LEN_RANGE[1], (width + 1) // 2)
    if max_legend < LEGEND_LEN_RANGE[0]:
        return None
    legend_len = random.randint(LEGEND_LEN_RANGE[0], max_legend)

    colors = list(range(10))
    bg = random.choice(colors)
    remaining = [color for color in colors if color != bg]
    border_color = random.choice(remaining)
    remaining = [color for color in remaining if color != border_color]
    if len(remaining) < legend_len:
        return None
    legend_colors = random.sample(remaining, legend_len)

    legend_cols = _legend_columns(width, legend_len)
    if legend_cols is None:
        return None

    rect_limit = min(RECT_COUNT_RANGE[1], legend_len + 1)
    rect_count = random.randint(RECT_COUNT_RANGE[0], rect_limit)
    counts = [random.randint(1, legend_len) for _ in range(rect_count)]
    random.shuffle(counts)

    rect_specs: list[tuple[int, int, list[tuple[int, int]], int]] = []
    for count in counts:
        spec = _sample_rect_shape(count)
        if spec is None:
            return None
        outer_h, outer_w, markers = spec
        rect_specs.append((outer_h, outer_w, markers, count))
    rect_specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    rects: list[dict[str, object]] = []
    padded_boxes: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w, marker_offsets, count in rect_specs:
        if height - outer_h - 2 < 1 or width - outer_w - 1 < 1:
            return None

        placed = False
        for _ in range(120):
            r0 = random.randint(1, height - outer_h - 2)
            c0 = random.randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pad_box = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
            if any(_boxes_overlap(pad_box, other) for other in padded_boxes):
                continue

            frame = _frame_cells(r0, c0, outer_h, outer_w)
            markers = [(r0 + 1 + dr, c0 + 1 + dc) for dr, dc in marker_offsets]
            rects.append(
                {
                    "count": count,
                    "frame": frame,
                    "markers": markers,
                }
            )
            padded_boxes.append(pad_box)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), bg, dtype=int)
    for col, color in zip(legend_cols, legend_colors, strict=False):
        grid[height - 1, col] = color

    for rect in rects:
        for r, c in rect["frame"]:
            grid[r, c] = border_color
        for r, c in rect["markers"]:
            grid[r, c] = border_color

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_output(grid, bg, legend_colors, rects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
