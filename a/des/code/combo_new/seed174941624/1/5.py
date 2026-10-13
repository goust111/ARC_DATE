import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

LEGEND_POOL = (BLUE, GREEN, ORANGE, CYAN, MAGENTA, MAROON)
TOPOLOGY_MODES = ("single_cross", "ladder", "mixed")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[int, list[int], np.ndarray]:
    legend_mask = (grid != bg) & (grid != GREY) & (grid != RED)
    coords = np.argwhere(legend_mask)
    assert coords.size > 0, "legend must contain non-path colors"
    cols = sorted({int(c) for _r, c in coords})
    assert len(cols) == 1, f"legend must occupy one column, got {cols}"
    col = cols[0]
    rows = sorted(int(r) for r, _c in coords)
    expected_rows = list(range(rows[0], rows[0] + len(rows)))
    assert rows == expected_rows, "legend cells must form one contiguous vertical stack"
    colors = [int(grid[r, col]) for r in rows]
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    return col, colors, legend_mask


def _intersection_mask(path_mask: np.ndarray) -> np.ndarray:
    h, w = path_mask.shape
    intersections = np.zeros_like(path_mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not path_mask[r, c]:
                continue
            has_horizontal = (c > 0 and path_mask[r, c - 1]) or (c + 1 < w and path_mask[r, c + 1])
            has_vertical = (r > 0 and path_mask[r - 1, c]) or (r + 1 < h and path_mask[r + 1, c])
            if has_horizontal and has_vertical:
                intersections[r, c] = True
    return intersections


def _segment_touch_count(
    component: list[tuple[int, int]], intersections: np.ndarray
) -> int:
    h, w = intersections.shape
    touching: set[tuple[int, int]] = set()
    for r, c in component:
        for nr, nc in _neighbors4(r, c, h, w):
            if intersections[nr, nc]:
                touching.add((nr, nc))
    return len(touching)


def main(grid: np.ndarray) -> np.ndarray:
    """Split the grey path network at intersections and recolor each segment by its red-marker count.

    Input structure:
    - The background is the dominant black color.
    - Grey and red pixels together form one-pixel-wide horizontal and vertical path lines.
    - Red pixels are sparse markers that replace some grey path cells.
    - A vertical legend stack sits in the upper-right area; its top-to-bottom order gives the colors for counts 0, 1, 2, ...

    Rule:
    - Detect path intersections: any path cell with at least one horizontal path neighbor and at least one vertical path neighbor.
    - Remove those intersection pixels as split points, so the remaining path cells break into independent straight segments.
    - Count the red pixels on each segment, call that count k, and recolor the whole segment to the k-th legend color.
    - Recolor every intersection pixel to yellow and leave the legend unchanged.

    Key insight:
    - Intersections are boundaries, not members of the counted segments, so the recoloring depends on counts after the path is cut apart at every crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _legend_col, legend, _legend_mask = _extract_legend(g, bg)

    path_mask = (g == GREY) | (g == RED)
    assert np.any(path_mask), "expected at least one path pixel"

    intersections = _intersection_mask(path_mask)
    segment_components = _components(path_mask & ~intersections)

    out = g.copy()
    for comp in segment_components:
        red_count = sum(1 for r, c in comp if int(g[r, c]) == RED)
        assert 0 <= red_count < len(legend), f"segment marker count {red_count} exceeds legend"
        segment_color = int(legend[red_count])
        for r, c in comp:
            out[r, c] = segment_color

    out[intersections] = YELLOW
    return out


def _draw_horizontal(mask: np.ndarray, row: int, c0: int, c1: int) -> None:
    mask[row, c0 : c1 + 1] = True


def _draw_vertical(mask: np.ndarray, col: int, r0: int, r1: int) -> None:
    mask[r0 : r1 + 1, col] = True


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    chosen: list[int] = []
    for v in values:
        if all(abs(v - other) >= gap for other in chosen):
            chosen.append(v)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _valid_straight_path(path_mask: np.ndarray) -> bool:
    h, w = path_mask.shape
    if not np.any(path_mask):
        return False
    intersections = _intersection_mask(path_mask)
    if not np.any(intersections):
        return False
    for r in range(h):
        for c in range(w):
            if not path_mask[r, c]:
                continue
            left = c > 0 and path_mask[r, c - 1]
            right = c + 1 < w and path_mask[r, c + 1]
            up = r > 0 and path_mask[r - 1, c]
            down = r + 1 < h and path_mask[r + 1, c]
            degree = int(left) + int(right) + int(up) + int(down)
            has_horizontal = left or right
            has_vertical = up or down
            if has_horizontal and has_vertical:
                if degree != 4:
                    return False
            else:
                if degree not in (1, 2):
                    return False
                if degree == 2 and not ((left and right) or (up and down)):
                    return False
    return True


def _segment_info(path_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    intersections = _intersection_mask(path_mask)
    comps = _components(path_mask & ~intersections)
    info: list[dict[str, object]] = []
    for comp in comps:
        touch_count = _segment_touch_count(comp, intersections)
        if touch_count == 0:
            seg_type = "end_end"
        elif touch_count == 1:
            seg_type = "cross_end"
        elif touch_count == 2:
            seg_type = "cross_cross"
        else:
            raise AssertionError(f"unexpected segment touching {touch_count} intersections")
        info.append({"coords": comp, "touch_count": touch_count, "type": seg_type})
    return intersections, info


def _build_single_cross(h: int, w: int) -> np.ndarray | None:
    max_col = w - 3
    path = np.zeros((h, w), dtype=bool)
    if h < 12 or max_col < 9:
        return None

    row = random.randint(4, h - 5)
    col = random.randint(4, max_col - 3)
    c0 = random.randint(1, col - 3)
    c1 = random.randint(col + 3, max_col)
    r0 = random.randint(1, row - 3)
    r1 = random.randint(row + 3, h - 2)

    _draw_horizontal(path, row, c0, c1)
    _draw_vertical(path, col, r0, r1)
    return path


def _build_horizontal_ladder(h: int, w: int, with_isolated: bool) -> np.ndarray | None:
    max_col = w - 3
    path = np.zeros((h, w), dtype=bool)
    cross_count = random.choice([2, 3])
    if h < 12 or max_col < 12:
        return None

    row = random.randint(4, h - 5)
    col_lo = 4
    col_hi = max_col - 4
    cols = _sample_spaced_positions(col_lo, col_hi, cross_count, 3)
    if cols is None:
        return None

    c0 = random.randint(1, cols[0] - 3)
    c1 = random.randint(cols[-1] + 3, max_col)
    _draw_horizontal(path, row, c0, c1)

    for col in cols:
        r0 = random.randint(1, row - 3)
        r1 = random.randint(row + 3, h - 2)
        _draw_vertical(path, col, r0, r1)

    if with_isolated:
        candidates: list[tuple[int, int]] = []
        left_end = cols[0] - 2
        right_start = cols[-1] + 2
        if left_end >= 2:
            candidates.append((1, left_end))
        if max_col - right_start + 1 >= 2:
            candidates.append((right_start, max_col))
        if not candidates:
            return None
        iso_start, iso_end = random.choice(candidates)
        row_choices = [r for r in range(1, h - 1) if abs(r - row) >= 2]
        if not row_choices:
            return None
        iso_row = random.choice(row_choices)
        _draw_horizontal(path, iso_row, iso_start, iso_end)

    return path


def _build_vertical_ladder(h: int, w: int, with_isolated: bool) -> np.ndarray | None:
    max_col = w - 3
    path = np.zeros((h, w), dtype=bool)
    cross_count = random.choice([2, 3])
    if h < 12 or max_col < 12:
        return None

    col = random.randint(4, max_col - 4)
    row_lo = 4
    row_hi = h - 5
    rows = _sample_spaced_positions(row_lo, row_hi, cross_count, 3)
    if rows is None:
        return None

    r0 = random.randint(1, rows[0] - 3)
    r1 = random.randint(rows[-1] + 3, h - 2)
    _draw_vertical(path, col, r0, r1)

    for row in rows:
        c0 = random.randint(1, col - 3)
        c1 = random.randint(col + 3, max_col)
        _draw_horizontal(path, row, c0, c1)

    if with_isolated:
        candidates: list[tuple[int, int]] = []
        top_end = rows[0] - 2
        bottom_start = rows[-1] + 2
        if top_end >= 2:
            candidates.append((1, top_end))
        if h - 2 - bottom_start + 1 >= 2:
            candidates.append((bottom_start, h - 2))
        if not candidates:
            return None
        iso_r0, iso_r1 = random.choice(candidates)
        col_choices = [c for c in range(1, max_col + 1) if abs(c - col) >= 2]
        if not col_choices:
            return None
        iso_col = random.choice(col_choices)
        _draw_vertical(path, iso_col, iso_r0, iso_r1)

    return path


def _build_topology(mode: str, h: int, w: int) -> np.ndarray | None:
    if mode == "single_cross":
        return _build_single_cross(h, w)
    if mode == "ladder":
        if random.random() < 0.5:
            return _build_horizontal_ladder(h, w, with_isolated=False)
        return _build_vertical_ladder(h, w, with_isolated=False)
    if mode == "mixed":
        if random.random() < 0.5:
            return _build_horizontal_ladder(h, w, with_isolated=True)
        return _build_vertical_ladder(h, w, with_isolated=True)
    raise ValueError(f"unknown mode: {mode}")


def _mode_matches(mode: str, segment_info: list[dict[str, object]], intersections: np.ndarray) -> bool:
    types = {str(item["type"]) for item in segment_info}
    intersection_count = int(np.count_nonzero(intersections))
    if mode == "single_cross":
        return intersection_count == 1 and types == {"cross_end"} and len(segment_info) == 4
    if mode == "ladder":
        return intersection_count >= 2 and "cross_end" in types and "cross_cross" in types and "end_end" not in types
    if mode == "mixed":
        return intersection_count >= 2 and types == {"end_end", "cross_end", "cross_cross"}
    return False


def _choose_red_counts(segment_info: list[dict[str, object]], legend_len: int) -> list[int]:
    maxima = [min(legend_len - 1, len(item["coords"])) for item in segment_info]
    counts = [random.randint(0, m) for m in maxima]

    positive_candidates = [i for i, m in enumerate(maxima) if m >= 1]
    if positive_candidates and all(count == 0 for count in counts):
        idx = random.choice(positive_candidates)
        counts[idx] = random.randint(1, maxima[idx])

    larger_candidates = [i for i, m in enumerate(maxima) if m >= 2]
    if legend_len >= 3 and larger_candidates and max(counts) < 2:
        idx = random.choice(larger_candidates)
        counts[idx] = random.randint(2, maxima[idx])

    if len(counts) >= 2 and len(set(counts)) == 1:
        for idx, m in enumerate(maxima):
            if counts[idx] < m:
                counts[idx] += 1
                break
        else:
            for idx in range(len(counts)):
                if counts[idx] > 0:
                    counts[idx] -= 1
                    break

    return counts


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(TOPOLOGY_MODES)
    h = random.randint(15, 28)
    w = random.randint(16, 30)
    legend_len = random.choice([3, 4])
    if h <= legend_len + 3 or w < 16:
        return None

    path_mask = _build_topology(mode, h, w)
    if path_mask is None or not _valid_straight_path(path_mask):
        return None

    intersections, segment_info = _segment_info(path_mask)
    if not segment_info or not _mode_matches(mode, segment_info, intersections):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    legend_colors = random.sample(list(LEGEND_POOL), legend_len)
    legend_col = w - 1
    grid[:legend_len, legend_col] = legend_colors
    grid[path_mask] = GREY

    red_counts = _choose_red_counts(segment_info, legend_len)
    for item, red_count in zip(segment_info, red_counts):
        coords = list(item["coords"])
        if red_count == 0:
            continue
        chosen = random.sample(coords, red_count)
        for r, c in chosen:
            grid[r, c] = RED

    if grid[:, legend_col - 1].any():
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == YELLOW):
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
