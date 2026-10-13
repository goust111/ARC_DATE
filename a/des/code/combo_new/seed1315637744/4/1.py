from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BG, GRAY))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _dominant_non_bg_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g, return_counts=True)
    candidates = [(int(v), int(ct)) for v, ct in zip(vals, counts, strict=False) if int(v) != int(bg)]
    assert candidates, "expected at least one non-background color"
    return max(candidates, key=lambda item: item[1])[0]


def _legend_colors(g: np.ndarray, bg: int, line_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    row = g[-1]
    mask = (row != int(bg)) & (row != int(line_color))
    runs: list[tuple[int, int]] = []
    c = 0
    while c < row.size:
        if not bool(mask[c]):
            c += 1
            continue
        start = c
        while c < row.size and bool(mask[c]):
            c += 1
        runs.append((start, c))
    assert runs, "expected a bottom legend strip"
    start, end = max(runs, key=lambda seg: seg[1] - seg[0])
    colors = row[start:end].copy()
    assert colors.size >= 1
    assert len(set(map(int, colors.tolist()))) == int(colors.size), "legend colors must be distinct"
    return colors.astype(int), (int(start), int(end))


def _region_boxes(scene: np.ndarray, bg: int, line_color: int) -> list[tuple[int, int, int, int]]:
    mask = scene != int(line_color)
    h, w = map(int, scene.shape)
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        if _touches_edge(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if not bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            continue
        if not bool(np.all(scene[r0 - 1, c0 : c1 + 1] == int(line_color))):
            continue
        if not bool(np.all(scene[r1 + 1, c0 : c1 + 1] == int(line_color))):
            continue
        if not bool(np.all(scene[r0 : r1 + 1, c0 - 1] == int(line_color))):
            continue
        if not bool(np.all(scene[r0 : r1 + 1, c1 + 1] == int(line_color))):
            continue
        non_bg = set(map(int, np.unique(scene[r0 : r1 + 1, c0 : c1 + 1])))
        non_bg.discard(int(bg))
        assert non_bg, "each enclosed region must contain colored marker pixels"
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    assert boxes, "expected enclosed rectangular regions"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-bounded rectangle with the legend color indexed by the number of interior color kinds.

    Input structure:
    - The background is black.
    - Gray horizontal and vertical line segments form a rectangular partition with one or more enclosed rectangular regions.
    - Each enclosed region contains scattered single-cell colored markers on black cells; the markers are neither black nor gray.
    - The bottom row contains a contiguous legend strip of distinct single-cell colors, read left to right as counts 1, 2, 3, ...

    Rule:
    - Read the bottom legend strip and map its first color to count 1, second to count 2, and so on.
    - For every enclosed rectangular region bounded by gray lines, count how many distinct non-black marker colors appear inside it.
    - Recolor every non-gray cell in that region, including black background cells and the marker pixels, to the legend color for that count.
    - Keep all gray boundary lines unchanged and erase the bottom legend strip to black in the output.

    Key insight:
    - The colored pixels inside a region are only labels to be counted; the actual fill color comes from the bottom legend position indexed by the number of distinct interior colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a 2D grid with a bottom legend row"

    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    line_color = _dominant_non_bg_color(g, bg)
    legend, _legend_span = _legend_colors(g, bg, line_color)

    scene = g[:-1]
    boxes = _region_boxes(scene, bg, line_color)

    out = g.copy()
    out[-1, :] = int(bg)
    for r0, c0, r1, c1 in boxes:
        patch = scene[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(patch)))
        colors.discard(int(bg))
        count = len(colors)
        assert 1 <= count <= int(legend.size), "region color count must be covered by the legend"
        out[r0 : r1 + 1, c0 : c1 + 1] = int(legend[count - 1])

    return out


def _sample_sizes(n: int, lo: int, hi: int) -> list[int]:
    return [random.randint(lo, hi) for _ in range(n)]


def _count_assignment(region_count: int, legend_len: int) -> list[int]:
    counts = list(range(1, legend_len + 1))
    counts.extend(random.randint(1, legend_len) for _ in range(region_count - legend_len))
    random.shuffle(counts)
    return counts


def _frame_geometry(
    row_sizes: list[int],
    col_sizes: list[int],
    top: int,
    left: int,
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]]:
    h_lines = [int(top)]
    for size in row_sizes:
        h_lines.append(h_lines[-1] + int(size) + 1)
    v_lines = [int(left)]
    for size in col_sizes:
        v_lines.append(v_lines[-1] + int(size) + 1)

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(row_sizes)):
        for j in range(len(col_sizes)):
            boxes.append((h_lines[i] + 1, v_lines[j] + 1, h_lines[i + 1] - 1, v_lines[j + 1] - 1))
    return h_lines, v_lines, boxes


def _draw_partition(grid: np.ndarray, h_lines: list[int], v_lines: list[int]) -> None:
    top = h_lines[0]
    bottom = h_lines[-1]
    left = v_lines[0]
    right = v_lines[-1]
    for r in h_lines:
        grid[int(r), left : right + 1] = GRAY
    for c in v_lines:
        grid[top : bottom + 1, int(c)] = GRAY


def _place_markers(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    count: int,
) -> None:
    r0, c0, r1, c1 = map(int, box)
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    chosen_cells = random.sample(cells, count)
    colors = random.sample(COLOR_POOL, count)
    for (r, c), color in zip(chosen_cells, colors, strict=False):
        grid[r, c] = int(color)


def _expected_output(
    inp: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    counts: list[int],
    legend_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    out[-1, :] = BG
    for box, count in zip(boxes, counts, strict=False):
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = int(legend_colors[count - 1])
    return out


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(2, 3)
    cols = random.randint(2, 3)
    row_sizes = _sample_sizes(rows, 3, 6)
    col_sizes = _sample_sizes(cols, 3, 6)
    content_h = sum(row_sizes) + rows + 1
    content_w = sum(col_sizes) + cols + 1

    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gap_above_legend = random.randint(0, 2)
    height = top_margin + content_h + gap_above_legend + 1
    width = left_margin + content_w + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    h_lines, v_lines, boxes = _frame_geometry(row_sizes, col_sizes, top_margin, left_margin)
    _draw_partition(grid, h_lines, v_lines)

    region_count = len(boxes)
    legend_len = random.randint(2, min(5, region_count))
    legend_colors = random.sample(COLOR_POOL, legend_len)
    counts = _count_assignment(region_count, legend_len)
    for box, count in zip(boxes, counts, strict=False):
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        if count > area:
            return None
        _place_markers(grid, box, count)

    legend_start = random.randint(0, width - legend_len)
    grid[-1, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)

    expected = _expected_output(grid, boxes, counts, legend_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
