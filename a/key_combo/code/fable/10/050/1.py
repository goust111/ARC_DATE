from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, upper in edges:
        assert lower != upper
        if upper not in above[lower]:
            above[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        ready = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(ready) == 1, "overlap constraints must determine one total order"
        color = ready[0]
        order.append(color)
        placed.add(color)
        for upper in above[color]:
            indegree[upper] -= 1
    return order


def _segment_open_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac + step, bc, step)]
    assert ac == bc, "consecutive centers must align by row or by column"
    step = 1 if br > ar else -1
    return [(r, ac) for r in range(ar + step, br, step)]


def _recover_scene(
    grid: np.ndarray,
) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected at least two colored frames"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        bbox = _bbox(visible)
        assert bbox is not None, "every frame color must remain visible"
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3
        assert height % 2 == 1 and width % 2 == 1, "frame centers must be grid cells"
        mask = _frame_mask(g.shape, bbox)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        bboxes[color] = bbox
        masks[color] = mask

    edges: set[tuple[int, int]] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color in colors if masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "every overlap cell must show one covering frame"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    assert edges, "need overlap evidence to recover the layer order"
    order = _topological_order(colors, edges)

    centers = [_center_of_bbox(bboxes[color]) for color in order]
    center_set = set(centers)
    assert len(center_set) == len(centers), "frame centers must be distinct"
    for i in range(len(centers) - 1):
        seg_cells = _segment_open_cells(centers[i], centers[i + 1])
        for cell in seg_cells:
            assert cell not in center_set - {centers[i], centers[i + 1]}, "segment crosses a third center"

    return bg, colors, bboxes, masks, order


def _draw_itinerary(
    base: np.ndarray,
    bg: int,
    bboxes: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    centers = {color: _center_of_bbox(bboxes[color]) for color in order}

    for lower, upper in zip(order, order[1:]):
        for r, c in _segment_open_cells(centers[lower], centers[upper]):
            if int(out[r, c]) == bg:
                out[r, c] = int(upper)

    for color in order:
        r, c = centers[color]
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the recovered frame stacking order as a travel itinerary and draw the trip through frame centers.

    Input structure:
    - Black background with several one-cell-thick hollow rectangular frames, one color per frame.
    - Frames overlap, and every overlap cell shows only the color of the frame that is on the higher layer.
    - Frame size and position do not encode the layer order; each consecutive pair in the true order has aligned centers.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - Use every overlap cell to infer which frame is above which, combine those constraints into the unique bottom-to-top order, and keep the visible input scene unchanged.
    - Mark every frame center with that frame's color, then connect consecutive centers with straight horizontal or vertical segments colored by the destination frame, painting only background cells.

    Key insight:
    - Occlusion is not damage to repair; it is the schedule. Recover the hidden ordering from visibility, then use that order as the path to draw.
    """
    g = np.asarray(grid, dtype=int)
    bg, _colors, bboxes, _masks, order = _recover_scene(g)
    return _draw_itinerary(g, bg, bboxes, order)


GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
FRAME_COUNT_CHOICES = (4, 5)
FRAME_SPANS = (5, 7, 9, 11, 13)
MIN_CENTER_MARGIN = 3
CENTER_GAP_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 500
SEARCH_ATTEMPTS = 5000
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _axis_moves(n_frames: int, start_axis: str) -> list[str]:
    axis = start_axis
    moves: list[str] = []
    for _ in range(n_frames - 1):
        moves.append(axis)
        axis = "v" if axis == "h" else "h"
    return moves


def _sample_axis_values(limit: int, count: int) -> list[int] | None:
    lo_gap, hi_gap = CENTER_GAP_RANGE
    for _ in range(200):
        gaps = [random.randint(lo_gap, hi_gap) for _ in range(count - 1)]
        span = sum(gaps)
        start_lo = MIN_CENTER_MARGIN
        start_hi = limit - 1 - MIN_CENTER_MARGIN - span
        if start_hi < start_lo:
            continue
        start = random.randint(start_lo, start_hi)
        values = [start]
        for gap in gaps:
            values.append(values[-1] + gap)
        if random.random() < 0.5:
            values.reverse()
        return values
    return None


def _sample_centers(shape: tuple[int, int], n_frames: int) -> list[tuple[int, int]] | None:
    h, w = shape
    start_axis = random.choice(("h", "v"))
    moves = _axis_moves(n_frames, start_axis)
    n_h = sum(move == "h" for move in moves)
    n_v = len(moves) - n_h
    rows = _sample_axis_values(h, n_v + 1)
    cols = _sample_axis_values(w, n_h + 1)
    if rows is None or cols is None:
        return None

    ri = 0
    ci = 0
    centers = [(rows[ri], cols[ci])]
    for move in moves:
        if move == "h":
            ci += 1
        else:
            ri += 1
        centers.append((rows[ri], cols[ci]))
    return centers


def _size_options(shape: tuple[int, int], center: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    max_h = 2 * min(r, h - 1 - r) + 1
    max_w = 2 * min(c, w - 1 - c) + 1
    heights = [span for span in FRAME_SPANS if span <= max_h]
    widths = [span for span in FRAME_SPANS if span <= max_w]
    return [(hh, ww) for hh in heights for ww in widths]


def _bbox_from_center(center: tuple[int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    hh, ww = size
    hr = hh // 2
    wc = ww // 2
    return (r - hr, c - wc, r + hr, c + wc)


def _render_input(
    shape: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    order: list[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    grid = np.full(shape, BACKGROUND, dtype=int)
    masks = [_frame_mask(shape, bbox) for bbox in bboxes]
    for color, mask in zip(order, masks):
        grid[mask] = int(color)
    return grid, masks


def _visible_sides_ok(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        bool(np.any(grid[r0, c0 : c1 + 1] == color))
        and bool(np.any(grid[r1, c0 : c1 + 1] == color))
        and bool(np.any(grid[r0 : r1 + 1, c0] == color))
        and bool(np.any(grid[r0 : r1 + 1, c1] == color))
    )


def _pairwise_overlap_pairs(masks: list[np.ndarray]) -> dict[tuple[int, int], int]:
    overlaps: dict[tuple[int, int], int] = {}
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            count = int(np.logical_and(masks[i], masks[j]).sum())
            if count:
                overlaps[(i, j)] = count
    return overlaps


def _search_bboxes(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    extra_overlap_mode: int,
) -> list[tuple[int, int, int, int]] | None:
    options = [_size_options(shape, center) for center in centers]
    if any(not opt for opt in options):
        return None

    consecutive = {(i, i + 1) for i in range(len(centers) - 1)}
    for _ in range(SEARCH_ATTEMPTS):
        sizes = [random.choice(opt) for opt in options]
        bboxes = [_bbox_from_center(center, size) for center, size in zip(centers, sizes)]
        masks = [_frame_mask(shape, bbox) for bbox in bboxes]

        coverage = np.zeros(shape, dtype=int)
        for mask in masks:
            coverage += mask.astype(int)
        if int(coverage.max()) > 2:
            continue

        overlaps = _pairwise_overlap_pairs(masks)
        if not all(pair in overlaps for pair in consecutive):
            continue
        if any(count not in {2, 4} for count in overlaps.values()):
            continue

        extra_pairs = set(overlaps) - consecutive
        if len(extra_pairs) > 1:
            continue
        if extra_overlap_mode == 0 and extra_pairs:
            continue
        if extra_overlap_mode == 1 and len(extra_pairs) != 1:
            continue

        return bboxes
    return None


def _segment_bg_budget(grid: np.ndarray, centers: list[tuple[int, int]]) -> list[int]:
    counts: list[int] = []
    for a, b in zip(centers, centers[1:]):
        cells = _segment_open_cells(a, b)
        counts.append(sum(int(grid[r, c]) == BACKGROUND for r, c in cells))
    return counts


def _expected_output(
    inp: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    bbox_map = {color: bbox for color, bbox in zip(order, bboxes)}
    return _draw_itinerary(inp, BACKGROUND, bbox_map, order)


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    order: list[int],
    masks: list[np.ndarray],
) -> bool:
    try:
        recovered_bg, _colors, recovered_bboxes, recovered_masks, recovered_order = _recover_scene(inp)
    except Exception:
        return False

    if recovered_bg != BACKGROUND or recovered_order != order:
        return False
    for color, bbox, mask in zip(order, bboxes, masks):
        if recovered_bboxes[color] != bbox:
            return False
        if not np.array_equal(recovered_masks[color], mask):
            return False
        if _bbox(inp == color) != bbox:
            return False
        if not _visible_sides_ok(inp, color, bbox):
            return False
        if int((inp == color).sum()) < 6:
            return False

    centers = [_center_of_bbox(bbox) for bbox in bboxes]
    bg_counts = _segment_bg_budget(inp, centers)
    if min(bg_counts) < 2 or sum(bg_counts) < len(bg_counts) * 3:
        return False

    expected = _expected_output(inp, bboxes, order)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, inp):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    order = random.sample(PALETTE, n_frames)

    centers = _sample_centers((h, w), n_frames)
    if centers is None:
        return None

    extra_overlap_mode = 1 if n_frames == 5 and random.random() < 0.4 else 0
    bboxes = _search_bboxes((h, w), centers, extra_overlap_mode)
    if bboxes is None:
        return None

    inp, masks = _render_input((h, w), bboxes, order)
    out = main(inp)
    if not _semantic_checks(inp, out, bboxes, order, masks):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
