import random

import numpy as np


BG = 0
MIN_LEGEND_LEN = 3
MAX_LEGEND_LEN = 5
MIN_FRAMES = 2
MAX_FRAMES = 4
MAX_TOTAL_PAD = 10
MAX_OUTER_DIM = 24
MAX_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for row in (0, h - 1):
        adjacent_row = 1 if row == 0 else h - 2
        row_mask = grid[row] != bg
        for start, end in _runs(row_mask):
            if h > 1 and not np.all(grid[adjacent_row, start : end + 1] == bg):
                continue
            candidates.append((end - start + 1, row, start, end))
    assert candidates, "legend strip not found on top/bottom edge"

    _, row, start, end = max(candidates, key=lambda item: item[0])
    legend = [int(v) for v in grid[row, start : end + 1]]
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[row, start : end + 1] = True
    return legend, legend_mask


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_perimeter(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _find_frames(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> list[tuple[tuple[int, int, int, int], int]]:
    fg_mask = (grid != bg) & (~legend_mask)
    frames: list[tuple[tuple[int, int, int, int], int]] = []
    for comp in _components(fg_mask):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        color = int(grid[comp[0]])
        if any(int(grid[r, c]) != color for r, c in comp):
            continue
        expected = _frame_perimeter(bbox)
        if len(comp) != len(expected):
            continue
        if set(comp) != expected:
            continue
        frames.append((bbox, color))

    frames.sort(
        key=lambda item: (
            (item[0][2] - item[0][0] + 1) * (item[0][3] - item[0][1] + 1),
            -item[0][0],
            -item[0][1],
        ),
        reverse=True,
    )
    assert frames, "no rectangular frames found"
    return frames


def _nested_gap(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> int:
    top = inner[0] - outer[0] - 1
    left = inner[1] - outer[1] - 1
    bottom = outer[2] - inner[2] - 1
    right = outer[3] - inner[3] - 1
    gap = min(top, left, bottom, right)
    assert gap >= 1, "adjacent frames must have at least one background cell between them"
    return gap


def _fill_rings(
    grid: np.ndarray,
    bg: int,
    legend: list[int],
    frames: list[tuple[tuple[int, int, int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    for idx in range(len(frames) - 1):
        outer = frames[idx][0]
        inner = frames[idx + 1][0]
        gap = _nested_gap(outer, inner)
        assert gap <= len(legend), "legend too short for detected gap"
        fill_color = int(legend[gap - 1])

        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        window = out[or0 + 1 : or1, oc0 + 1 : oc1]
        ring_mask = np.ones(window.shape, dtype=bool)
        ring_mask[ir0 - or0 - 1 : ir1 - or0, ic0 - oc0 - 1 : ic1 - oc0] = False
        window[ring_mask & (window == bg)] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the background ring between each adjacent pair of nested frames.

    The legend strip on the top or bottom edge lists fill colors from left to
    right. A ring whose frame-to-frame gap is G receives the legend color at
    index G - 1.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    legend, legend_mask = _find_legend(g, bg)
    frames = _find_frames(g, bg, legend_mask)

    for (outer_bbox, _), (inner_bbox, _) in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer_bbox
        ir0, ic0, ir1, ic1 = inner_bbox
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "frames must be nested"

    return _fill_rings(g, bg, legend, frames)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _choose_gaps(rng: random.Random, legend_len: int, pair_count: int) -> list[int]:
    for _ in range(200):
        gaps = [rng.randint(1, legend_len) for _ in range(pair_count)]
        if sum(g + 1 for g in gaps) > MAX_TOTAL_PAD:
            continue
        if pair_count > 1 and rng.random() < 0.6 and len(set(gaps)) == 1:
            continue
        return gaps
    return [1] * pair_count


def _build_bboxes(
    outer_bbox: tuple[int, int, int, int], gaps: list[int]
) -> list[tuple[int, int, int, int]]:
    frames = [outer_bbox]
    r0, c0, r1, c1 = outer_bbox
    for gap in gaps:
        step = gap + 1
        r0 += step
        c0 += step
        r1 -= step
        c1 -= step
        frames.append((r0, c0, r1, c1))
    return frames


def _expected_from_structure(
    grid: np.ndarray, legend: list[int], bboxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    tagged_frames = [(bbox, -1) for bbox in bboxes]
    return _fill_rings(grid, BG, legend, tagged_frames)


def _generate_case(
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    legend_len = rng.randint(MIN_LEGEND_LEN, MAX_LEGEND_LEN)
    frame_count = rng.randint(MIN_FRAMES, MAX_FRAMES)
    pair_count = frame_count - 1
    gaps = _choose_gaps(rng, legend_len, pair_count)

    total_pad = sum(g + 1 for g in gaps)
    max_inner_h = MAX_OUTER_DIM - 2 * total_pad
    max_inner_w = MAX_OUTER_DIM - 2 * total_pad
    if max_inner_h < 3 or max_inner_w < 3:
        raise ValueError("gaps too large for configured canvas")

    inner_h = rng.randint(3, min(8, max_inner_h))
    inner_w = rng.randint(3, min(8, max_inner_w))
    outer_h = inner_h + 2 * total_pad
    outer_w = inner_w + 2 * total_pad

    legend_side = rng.choice(("top", "bottom"))
    top_clear = 2 if legend_side == "top" else 1
    bottom_clear = 2 if legend_side == "bottom" else 1

    min_h = outer_h + top_clear + bottom_clear
    min_w = max(outer_w + 2, legend_len + 2)
    grid_h = min_h + rng.randint(0, 6)
    grid_w = min_w + rng.randint(0, 6)

    outer_r0 = rng.randint(top_clear, grid_h - bottom_clear - outer_h)
    outer_c0 = rng.randint(1, grid_w - outer_w - 1)
    outer_bbox = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    bboxes = _build_bboxes(outer_bbox, gaps)

    total_colors = legend_len + frame_count
    assert total_colors <= 9
    palette = list(range(1, 10))
    rng.shuffle(palette)
    legend = palette[:legend_len]
    frame_colors = palette[legend_len : legend_len + frame_count]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    legend_row = 0 if legend_side == "top" else grid_h - 1
    legend_start = rng.randint(0, grid_w - legend_len)
    grid[legend_row, legend_start : legend_start + legend_len] = np.array(legend, dtype=int)

    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    expected = _expected_from_structure(grid, legend, bboxes)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        try:
            grid, _expected = _generate_case(rng)
        except ValueError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
