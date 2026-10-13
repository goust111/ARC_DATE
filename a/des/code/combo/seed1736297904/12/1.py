import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(10))
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    non_bg = [(int(cnt), int(val)) for val, cnt in zip(vals.tolist(), cnts.tolist(), strict=False) if int(val) != BG]
    assert non_bg, "expected a non-background line color"
    return max(non_bg)[1]


def _edge_runs(edge: np.ndarray, forbidden: set[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    idx = 0
    while idx < len(edge):
        color = int(edge[idx])
        if color in forbidden:
            idx += 1
            continue
        end = idx + 1
        while end < len(edge) and int(edge[end]) == color:
            end += 1
        runs.append((idx, end, color))
        idx = end
    return runs


def _legend(grid: np.ndarray, line_color: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    forbidden = {BG, line_color}
    candidates: list[tuple[tuple[int, int, int], str, list[tuple[int, int, int]]]] = []
    for side, edge in (
        ("top", grid[0, :]),
        ("bottom", grid[h - 1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, w - 1]),
    ):
        runs = _edge_runs(edge, forbidden)
        if len(runs) < 2:
            continue
        colors = [color for _, _, color in runs]
        if len(colors) != len(set(colors)):
            continue
        total = sum(end - start for start, end, _ in runs)
        candidates.append(((len(runs), total, len(set(colors))), side, runs))

    assert candidates, "expected legend bars on one grid edge"
    _, side, runs = max(candidates, key=lambda item: item[0])

    mask = np.zeros_like(grid, dtype=bool)
    colors = [color for _, _, color in runs]
    if side == "top":
        for start, end, _ in runs:
            mask[0, start:end] = True
    elif side == "bottom":
        for start, end, _ in runs:
            mask[h - 1, start:end] = True
    elif side == "left":
        for start, end, _ in runs:
            mask[start:end, 0] = True
    else:
        for start, end, _ in runs:
            mask[start:end, w - 1] = True
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle with the color of legend bar N, where N is the
    number of distinct marker colors inside that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_color = _line_color(g)
    legend_colors, legend_mask = _legend(g, line_color)

    region_mask = (g != line_color) & (~legend_mask)
    out = g.copy()
    h, w = g.shape

    for comp in _components(region_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        marker_colors = {
            int(g[r, c])
            for r, c in comp
            if int(g[r, c]) not in (BG, line_color)
        }
        count = len(marker_colors)
        if count == 0:
            continue
        assert count <= len(legend_colors), "region marker-color count exceeds legend length"
        fill_color = legend_colors[count - 1]
        for r, c in comp:
            if int(out[r, c]) == BG:
                out[r, c] = fill_color

    return out


def _sample_sizes(n: int, low: int, high: int) -> list[int]:
    return [random.randint(low, high) for _ in range(n)]


def _bounded_composition(total: int, parts: int, max_each: int) -> list[int] | None:
    if total < parts or total > parts * max_each:
        return None
    out: list[int] = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remaining - left * max_each)
        hi = min(max_each, remaining - left)
        if lo > hi:
            return None
        val = random.randint(lo, hi)
        out.append(val)
        remaining -= val
    random.shuffle(out)
    return out


def _sample_legend_segments(side_len: int, count: int) -> list[tuple[int, int]] | None:
    max_bar_total = side_len - (count + 1)
    if max_bar_total < count:
        return None
    total = random.randint(count, min(max_bar_total, 4 * count))
    lengths = _bounded_composition(total, count, 4)
    if lengths is None:
        return None

    extra = side_len - (sum(lengths) + count + 1)
    gap_extras = [0] * (count + 1)
    for _ in range(extra):
        gap_extras[random.randrange(count + 1)] += 1
    gaps = [1 + x for x in gap_extras]

    cursor = gaps[0]
    segs: list[tuple[int, int]] = []
    for idx, length in enumerate(lengths):
        segs.append((cursor, cursor + length))
        cursor += length + gaps[idx + 1]
    return segs


def _sample_spaced_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) != 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _sample_region_counts(region_count: int, legend_count: int) -> list[int] | None:
    require_zero = random.random() < 0.65
    require_max = random.random() < 0.45
    for _ in range(200):
        counts = [random.randint(0, legend_count) for _ in range(region_count)]
        if max(counts) == 0:
            continue
        if max(counts) < 2:
            continue
        if len(set(counts)) < 2:
            continue
        if require_zero and 0 not in counts:
            continue
        if require_max and legend_count not in counts:
            continue
        return counts
    return None


def _legend_side_spaces(side: str) -> tuple[int, int, int, int]:
    top = random.randint(2, 3) if side == "top" else random.randint(1, 2)
    bottom = random.randint(2, 3) if side == "bottom" else random.randint(1, 2)
    left = random.randint(2, 3) if side == "left" else random.randint(1, 2)
    right = random.randint(2, 3) if side == "right" else random.randint(1, 2)
    return top, bottom, left, right


def _draw_frame(
    grid: np.ndarray,
    line_color: int,
    top: int,
    left: int,
    row_heights: list[int],
    col_widths: list[int],
) -> list[tuple[int, int, int, int]]:
    frame_h = sum(row_heights) + len(row_heights) + 1
    frame_w = sum(col_widths) + len(col_widths) + 1
    bottom = top + frame_h - 1
    right = left + frame_w - 1

    row_lines = [top]
    cursor = top + 1
    for height in row_heights:
        cursor += height
        row_lines.append(cursor)
        cursor += 1

    col_lines = [left]
    cursor = left + 1
    for width in col_widths:
        cursor += width
        col_lines.append(cursor)
        cursor += 1

    for rr in row_lines:
        grid[rr, left : right + 1] = line_color
    for cc in col_lines:
        grid[top : bottom + 1, cc] = line_color

    regions: list[tuple[int, int, int, int]] = []
    for r_idx in range(len(row_heights)):
        for c_idx in range(len(col_widths)):
            r0 = row_lines[r_idx] + 1
            r1 = row_lines[r_idx + 1] - 1
            c0 = col_lines[c_idx] + 1
            c1 = col_lines[c_idx + 1] - 1
            regions.append((r0, r1, c0, c1))
    return regions


def _draw_legend(
    grid: np.ndarray,
    side: str,
    colors: list[int],
    segments: list[tuple[int, int]],
) -> None:
    h, w = grid.shape
    if side == "top":
        for (start, end), color in zip(segments, colors, strict=False):
            grid[0, start:end] = color
    elif side == "bottom":
        for (start, end), color in zip(segments, colors, strict=False):
            grid[h - 1, start:end] = color
    elif side == "left":
        for (start, end), color in zip(segments, colors, strict=False):
            grid[start:end, 0] = color
    else:
        for (start, end), color in zip(segments, colors, strict=False):
            grid[start:end, w - 1] = color


def _populate_region(
    grid: np.ndarray,
    expected: np.ndarray,
    region: tuple[int, int, int, int],
    distinct_count: int,
    legend_colors: list[int],
    marker_palette: list[int],
) -> bool:
    r0, r1, c0, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if distinct_count == 0:
        return True

    max_positions = (height * width + 1) // 2
    extra_cap = min(3, max_positions - distinct_count)
    if extra_cap < 0:
        return False
    extra = random.randint(0, extra_cap) if extra_cap > 0 and random.random() < 0.7 else 0
    total_markers = distinct_count + extra
    positions = _sample_spaced_positions(height, width, total_markers)
    if positions is None:
        return False

    marker_colors = random.sample(marker_palette, distinct_count)
    assignments = marker_colors[:]
    for _ in range(extra):
        assignments.append(random.choice(marker_colors))
    random.shuffle(assignments)

    for (dr, dc), color in zip(positions, assignments, strict=False):
        grid[r0 + dr, c0 + dc] = color
        expected[r0 + dr, c0 + dc] = color

    fill_color = legend_colors[distinct_count - 1]
    window = expected[r0 : r1 + 1, c0 : c1 + 1]
    window[window == BG] = fill_color
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(("top", "bottom", "left", "right"))
    legend_count = random.randint(2, 5)
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    row_heights = _sample_sizes(rows, 4, 7)
    col_widths = _sample_sizes(cols, 4, 7)

    line_color = random.choice([c for c in COLORS if c != BG])
    legend_palette = [c for c in COLORS if c not in (BG, line_color)]
    legend_colors = random.sample(legend_palette, legend_count)
    marker_palette = [c for c in COLORS if c not in (BG, line_color)]

    top_space, bottom_space, left_space, right_space = _legend_side_spaces(side)
    frame_h = sum(row_heights) + rows + 1
    frame_w = sum(col_widths) + cols + 1
    height = top_space + frame_h + bottom_space
    width = left_space + frame_w + right_space

    side_len = width if side in ("top", "bottom") else height
    segments = _sample_legend_segments(side_len, legend_count)
    if segments is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected = grid.copy()
    regions = _draw_frame(grid, line_color, top_space, left_space, row_heights, col_widths)
    expected[:] = grid
    _draw_legend(grid, side, legend_colors, segments)
    _draw_legend(expected, side, legend_colors, segments)

    region_counts = _sample_region_counts(len(regions), legend_count)
    if region_counts is None:
        return None

    has_duplicate = False
    for region, distinct_count in zip(regions, region_counts, strict=False):
        if distinct_count == 0:
            continue
        ok = _populate_region(grid, expected, region, distinct_count, legend_colors, marker_palette)
        if not ok:
            return None
        r0, r1, c0, c1 = region
        in_region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(in_region) > distinct_count:
            has_duplicate = True

    if not has_duplicate and random.random() < 0.7:
        return None
    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _gen_attempt()
        if pair is None:
            continue
        grid, expected = pair
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
