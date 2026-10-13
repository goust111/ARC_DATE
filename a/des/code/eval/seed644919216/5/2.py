import random

import numpy as np


GREY = 5
MAROON = 9
DIR_VECS = {
    "E": (0, 1),
    "S": (1, 0),
    "SE": (1, 1),
    "SW": (1, -1),
}
DIR_NAMES = tuple(DIR_VECS.keys())
NON_SPECIAL = [0, 1, 2, 3, 4, 6, 7, 8]
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert cols.size == 1, "expected exactly one full-height grey separator"
    return int(cols[0])


def _find_legend(grid: np.ndarray, bg: int, sep_col: int) -> tuple[int, list[int], np.ndarray]:
    bottom = np.flatnonzero((grid[-1, :sep_col] != bg) & (grid[-1, :sep_col] != GREY))
    assert bottom.size >= 1, "expected a legend in the bottom-left"
    legend_c = int(bottom[0])
    colors: list[int] = []
    mask = np.zeros_like(grid, dtype=bool)
    r = grid.shape[0] - 1
    while r >= 0 and int(grid[r, legend_c]) not in {bg, GREY}:
        col = int(grid[r, legend_c])
        colors.append(col)
        mask[r, legend_c] = True
        r -= 1
    assert colors, "legend must contain at least one color"
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return legend_c, colors, mask


def _decode_segment(points: list[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    pts = sorted(points)
    assert 2 <= len(pts) <= 5, "segments must have length 2..5"

    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    step_count = len(pts)

    if len(set(rows)) == 1:
        ordered = sorted(pts, key=lambda p: p[1])
        assert all(ordered[i + 1][1] - ordered[i][1] == 1 for i in range(step_count - 1))
        return (0, 1), step_count

    if len(set(cols)) == 1:
        ordered = sorted(pts, key=lambda p: p[0])
        assert all(ordered[i + 1][0] - ordered[i][0] == 1 for i in range(step_count - 1))
        return (1, 0), step_count

    diffs = {r - c for r, c in pts}
    if len(diffs) == 1:
        ordered = sorted(pts, key=lambda p: p[0])
        assert all(
            ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == 1
            for i in range(step_count - 1)
        )
        return (1, 1), step_count

    sums = {r + c for r, c in pts}
    if len(sums) == 1:
        ordered = sorted(pts, key=lambda p: p[0])
        assert all(
            ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == -1
            for i in range(step_count - 1)
        )
        return (1, -1), step_count

    raise AssertionError("segment must be horizontal, vertical, or 45-degree diagonal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort colored line fragments by a bottom-left legend, then replay them as a path.

    Horizontal fragments mean move east, vertical fragments mean move south,
    and diagonals mean move southeast or southwest depending on the visible slope.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    sep_col = _find_separator(g)
    _legend_c, order, legend_mask = _find_legend(g, bg, sep_col)

    start_pts = np.argwhere(g == MAROON)
    assert start_pts.shape[0] == 1, "expected exactly one maroon start pixel"
    sr, sc = map(int, start_pts[0])
    assert sc > sep_col, "start pixel must be on the right side"

    segments: dict[int, list[tuple[int, int]]] = {}
    for color in order:
        pts = np.argwhere((g == int(color)) & (~legend_mask))
        assert pts.shape[0] >= 2, "every legend color must have one matching segment"
        segments[int(color)] = [tuple(map(int, p)) for p in pts]

    out = np.full((h, w), bg, dtype=int)
    out[:, sep_col] = GREY
    out[sr, sc] = MAROON

    r, c = sr, sc
    for color in order:
        (dr, dc), steps = _decode_segment(segments[int(color)])
        for _ in range(steps):
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "decoded path must stay inside the canvas"
            assert c != sep_col, "decoded path may not overwrite the separator"
            out[r, c] = int(color)

    return out


def _segment_cells(direction: str, length: int) -> list[tuple[int, int]]:
    if direction == "E":
        return [(0, i) for i in range(length)]
    if direction == "S":
        return [(i, 0) for i in range(length)]
    if direction == "SE":
        return [(i, i) for i in range(length)]
    if direction == "SW":
        return [(i, length - 1 - i) for i in range(length)]
    raise ValueError(direction)


def _trace_path(specs: list[tuple[str, int]]) -> list[tuple[int, int]]:
    r = 0
    c = 0
    cells = [(0, 0)]
    for direction, length in specs:
        dr, dc = DIR_VECS[direction]
        for _ in range(length):
            r += dr
            c += dc
            cells.append((r, c))
    return cells


def _sample_path(height: int, width: int, count: int) -> tuple[list[tuple[str, int]], list[tuple[int, int]], int, int] | None:
    for _ in range(200):
        specs = [(random.choice(DIR_NAMES), random.randint(2, 5)) for _ in range(count)]
        if len(set(direction for direction, _ in specs)) == 1 and random.random() < 0.7:
            continue

        cells = _trace_path(specs)
        if len(cells) != len(set(cells)):
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if max_r - min_r + 1 > height:
            continue
        if max_c - min_c + 1 > width:
            continue
        return specs, cells, min_c, max_c
    return None


def _place_segments(height: int, left_w: int, legend_len: int, specs: list[tuple[int, str, int]]) -> dict[int, list[tuple[int, int]]] | None:
    occupied = np.zeros((height, left_w), dtype=bool)

    legend_top = height - legend_len
    occupied[max(0, legend_top - 1) : height, : min(left_w, 2)] = True

    placements: dict[int, list[tuple[int, int]]] = {}
    order = list(specs)
    random.shuffle(order)
    order.sort(key=lambda item: (-item[2], item[1] not in {"SE", "SW"}))

    for color, direction, length in order:
        rel = _segment_cells(direction, length)
        max_r = max(r for r, _ in rel)
        max_c = max(c for _, c in rel)
        candidates: list[tuple[int, int]] = []
        for base_r in range(0, height - max_r):
            for base_c in range(0, left_w - max_c):
                cells = [(base_r + r, base_c + c) for r, c in rel]
                if any(occupied[r, c] for r, c in cells):
                    continue
                candidates.append((base_r, base_c))

        if not candidates:
            return None

        base_r, base_c = random.choice(candidates)
        abs_cells = [(base_r + r, base_c + c) for r, c in rel]
        placements[int(color)] = abs_cells

        rr0 = max(0, base_r - 1)
        rr1 = min(height, base_r + max_r + 2)
        cc0 = max(0, base_c - 1)
        cc1 = min(left_w, base_c + max_c + 2)
        occupied[rr0:rr1, cc0:cc1] = True

    return placements


def _build_grid() -> np.ndarray | None:
    height = random.randint(16, 28)
    left_w = random.randint(9, 13)
    right_w = random.randint(8, 14)
    seg_count = random.randint(3, 6)
    if seg_count > height - 2:
        return None

    palette = list(NON_SPECIAL)
    bg = random.choice(palette)
    seg_palette = [c for c in palette if c != bg]
    if len(seg_palette) < seg_count:
        return None

    order_colors = random.sample(seg_palette, seg_count)
    path_sample = _sample_path(height, right_w, seg_count)
    if path_sample is None:
        return None
    specs, cells, min_c, max_c = path_sample

    width = left_w + 1 + right_w
    sep_col = left_w
    start_r_max = height - 1 - max(r for r, _ in cells)
    if start_r_max < 0:
        return None
    start_c_lo = sep_col + 1 - min_c
    start_c_hi = width - 1 - max_c
    if start_c_lo > start_c_hi:
        return None

    start_r = random.randint(0, start_r_max)
    start_c = random.randint(start_c_lo, start_c_hi)

    seg_specs = [(color, direction, length) for color, (direction, length) in zip(order_colors, specs)]
    placements = _place_segments(height, left_w, seg_count, seg_specs)
    if placements is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[:, sep_col] = GREY

    for i, color in enumerate(order_colors):
        grid[height - 1 - i, 0] = int(color)

    for color, cells_abs in placements.items():
        for r, c in cells_abs:
            grid[r, c] = int(color)

    grid[start_r, start_c] = MAROON

    out = main(grid)
    assert out.shape == grid.shape
    assert np.all(out[:, sep_col] == GREY)
    assert np.all(out[:, :sep_col][grid[:, :sep_col] == GREY] == GREY)
    assert np.all(out[:, :sep_col][grid[:, :sep_col] != GREY] == bg)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
