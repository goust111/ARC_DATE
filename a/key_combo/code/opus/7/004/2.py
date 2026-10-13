import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LEGEND_COLORS = (BLUE, RED, GREEN, YELLOW)


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = components_mask(grid == GREY, connectivity=4)
    frames = []
    for coords in comps:
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame too small"
        actual = {tuple(map(int, p)) for p in coords}
        expected = _border_cells(bbox)
        assert actual == expected, "grey component must be a one-pixel hollow rectangle"
        frames.append(bbox)
    assert 3 <= len(frames) <= 4, "expected 3 or 4 nested frames"
    frames.sort(key=lambda b: -((b[2] - b[0] + 1) * (b[3] - b[1] + 1)))
    for outer, inner in zip(frames, frames[1:]):
        assert outer[0] < inner[0] and outer[1] < inner[1]
        assert inner[2] < outer[2] and inner[3] < outer[3]
    return frames


def _extract_legend(grid: np.ndarray, depth_count: int) -> dict[int, int]:
    comps = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for coords in components_mask(grid == color, connectivity=4):
            bbox = bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            assert c0 == c1, "legend entries must be single-column bars"
            assert len(coords) == (r1 - r0 + 1), "legend entries must be solid bars"
            comps.append((bbox, color))
    assert len(comps) == depth_count, "legend count must match frame depth count"
    depth_to_color = {}
    for bbox, color in comps:
        height = bbox[2] - bbox[0] + 1
        assert 1 <= height <= depth_count, "legend height out of range"
        assert height not in depth_to_color, "legend heights must be unique"
        depth_to_color[height] = color
    assert set(depth_to_color) == set(range(1, depth_count + 1)), "legend heights must encode all depths"
    return depth_to_color


def _depth_region_mask(shape: tuple[int, int], frames: list[tuple[int, int, int, int]], depth: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = frames[depth - 1]
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if depth < len(frames):
        ir0, ic0, ir1, ic1 = frames[depth]
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return mask


def _expected_output_from_structure(grid: np.ndarray, frames: list[tuple[int, int, int, int]], depth_to_color: dict[int, int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for depth in range(1, len(frames) + 1):
        region = _depth_region_mask(grid.shape, frames, depth)
        out[region & (grid == BLACK)] = depth_to_color[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular ring by matching its nesting depth to a legend bar height.

    Input structure:
    - The background is black.
    - Three or four disjoint grey one-pixel hollow rectangles are strictly nested.
    - A legend near the upper-left consists of single-column colored bars separated by black gaps; each bar has a unique height.

    Rule:
    - Sort the grey hollow rectangles by bounding-box area from largest to smallest.
    - Treat the black region inside rectangle depth d but outside the next inner rectangle as depth d; the deepest region is the black interior of the innermost rectangle.
    - Read the legend as height-to-color pairs and recolor every black pixel in depth d with the legend color whose bar height is d.
    - Keep all grey frame pixels and all legend pixels unchanged.

    Key insight:
    - The legend heights are not physical region sizes; they encode nesting depth, so the solver must infer containment first and only then translate depth into color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = get_bg(g)
    assert bg == BLACK, "expected black background"

    frames = _extract_frames(g)
    depth_to_color = _extract_legend(g, len(frames))
    return _expected_output_from_structure(g, frames, depth_to_color)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _random_height_order(n: int) -> list[int]:
    order = list(range(1, n + 1))
    while True:
        random.shuffle(order)
        if order != list(range(1, n + 1)) and order != list(range(n, 0, -1)):
            return order


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], dict[int, int]]:
    n_frames = random.choice((3, 4))
    gaps = [random.choice((2, 3)) for _ in range(n_frames - 1)]
    total_inset = sum(gap + 1 for gap in gaps)

    max_outer_w = 27
    max_outer_h = 29
    min_inner = 3
    max_inner_h = max(min_inner, min(8, max_outer_h - 2 * total_inset))
    max_inner_w = max(min_inner, min(8, max_outer_w - 2 * total_inset))
    inner_h = random.randint(min_inner, max_inner_h)
    inner_w = random.randint(min_inner, max_inner_w)
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset

    left_margin = random.randint(3, 30 - outer_w)
    right_margin = random.randint(0, 30 - outer_w - left_margin)
    top_margin = random.randint(0, 30 - outer_h)
    bottom_margin = random.randint(0, 30 - outer_h - top_margin)

    legend_heights = _random_height_order(n_frames)
    legend_top = random.randint(0, 1)
    legend_height = sum(legend_heights) + (n_frames - 1)

    h = max(outer_h + top_margin + bottom_margin, legend_top + legend_height)
    w = outer_w + left_margin + right_margin
    assert 1 <= h <= 30 and 1 <= w <= 30

    grid = np.zeros((h, w), dtype=int)

    available_colors = list(LEGEND_COLORS)
    random.shuffle(available_colors)
    chosen_colors = available_colors[:n_frames]
    depth_to_color = {
        depth: color
        for depth, color in zip(range(1, n_frames + 1), chosen_colors)
    }

    row = legend_top
    used_legend_cells = set()
    for depth_height in legend_heights:
        color = depth_to_color[depth_height]
        grid[row : row + depth_height, 0] = color
        for rr in range(row, row + depth_height):
            used_legend_cells.add((rr, 0))
        row += depth_height + 1

    frames = []
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    frames.append((r0, c0, r1, c1))
    for gap in gaps:
        inset = gap + 1
        r0 += inset
        c0 += inset
        r1 -= inset
        c1 -= inset
        frames.append((r0, c0, r1, c1))

    for bbox in frames:
        _draw_frame(grid, bbox)

    expected = _expected_output_from_structure(grid, frames, depth_to_color)

    for depth in range(1, n_frames + 1):
        region = _depth_region_mask(grid.shape, frames, depth)
        assert np.count_nonzero(region & (grid == BLACK)) > 0, "every depth region must matter"
        assert np.all(expected[region & (grid == BLACK)] == depth_to_color[depth])

    assert not any(cell in used_legend_cells for bbox in frames for cell in _border_cells(bbox))
    return grid, expected, frames, depth_to_color


def _validate_instance(grid: np.ndarray, expected: np.ndarray, frames: list[tuple[int, int, int, int]], depth_to_color: dict[int, int]) -> bool:
    try:
        parsed_frames = _extract_frames(grid)
        parsed_legend = _extract_legend(grid, len(frames))
    except AssertionError:
        return False

    if parsed_frames != frames:
        return False
    if parsed_legend != depth_to_color:
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    if np.any(out[grid == GREY] != GREY):
        return False
    if np.any(out[:, 0][grid[:, 0] != BLACK] != grid[:, 0][grid[:, 0] != BLACK]):
        return False

    for depth in range(1, len(frames) + 1):
        region = _depth_region_mask(grid.shape, frames, depth)
        black_region = region & (grid == BLACK)
        if not np.any(black_region):
            return False
        if len(np.unique(out[black_region])) != 1:
            return False
        if int(out[black_region][0]) != depth_to_color[depth]:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid, expected, frames, depth_to_color = _build_instance()
        if _validate_instance(grid, expected, frames, depth_to_color):
            return grid
    raise RuntimeError("failed to generate a valid nested-frame legend instance")
