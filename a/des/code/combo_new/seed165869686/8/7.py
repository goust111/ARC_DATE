from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Optional

import numpy as np


BBox = tuple[int, int, int, int]
GREY = 5
PALETTE = tuple(range(10))
MAX_DEPTH = 4
MAX_GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert mask.ndim == 2
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def frame_bbox(coords: np.ndarray) -> Optional[BBox]:
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords}
    for rr, cc in coord_set:
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None

    for rr in range(r0, r1 + 1):
        if (rr, c0) not in coord_set or (rr, c1) not in coord_set:
            return None
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coord_set or (r1, cc) not in coord_set:
            return None
    return r0, c0, r1, c1


def contains_bbox(outer: BBox, inner: BBox) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _legend_candidate_score(bbox: BBox, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def find_legend(grid: np.ndarray, bg: int) -> tuple[BBox, list[int]]:
    candidates: list[tuple[int, BBox, list[int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in components_mask(grid == color, connectivity=4):
            bbox = frame_bbox(coords)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if w != 3:
                continue
            depth = h - 2
            if not (1 <= depth <= MAX_DEPTH):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.shape != (depth, 1):
                continue
            colors = [int(v) for v in inner[:, 0]]
            if any(v == bg or v == color for v in colors):
                continue
            if len(set(colors)) != depth:
                continue
            score = _legend_candidate_score(bbox, grid.shape)
            candidates.append((score, bbox, colors))

    assert candidates, "expected a legend strip"
    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "legend must be unique"
    _, bbox, colors = best[0]
    return bbox, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular pool border according to its containment depth and the corner legend.

    Input structure:
    - The dominant color is the background.
    - Away from one corner legend, the grid contains one or more disconnected 1-pixel-thick rectangular borders, all drawn in the same source color and separated by background gaps.
    - Some borders may contain smaller borders with exactly one background cell gap on every side, creating nesting depths from 1 up to 4.
    - A small vertical 1-column color strip sits inside a 1-pixel-thick rectangular legend frame near a corner; the strip lists one color per depth from top to bottom.

    Rule:
    - Detect the legend strip and read its colors from top to bottom as C1, C2, C3, ...
    - Remove the entire legend frame and strip, replacing it with background.
    - Among the remaining rectangular borders, compute each border's depth by counting how many larger border bboxes strictly contain it; the outermost border has depth 1.
    - Recolor every border of depth k with Ck and leave all background gaps unchanged.

    Key insight:
    - The borders all start with the same color, so the only way to choose the output color is to infer geometric containment depth first and then use the corner strip as the visible depth-to-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    legend_bbox, legend_colors = find_legend(g, bg)
    lr0, lc0, lr1, lc1 = legend_bbox

    out = g.copy()
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    frames: list[BBox] = []
    frame_cells: list[np.ndarray] = []
    for color in map(int, np.unique(out)):
        if color == bg:
            continue
        for coords in components_mask(out == color, connectivity=4):
            bbox = frame_bbox(coords)
            if bbox is None:
                continue
            frames.append(bbox)
            frame_cells.append(coords)

    assert frames, "expected at least one pool border"

    for bbox, coords in zip(frames, frame_cells):
        depth = 1 + sum(contains_bbox(other, bbox) for other in frames)
        assert 1 <= depth <= len(legend_colors), "frame depth must fit the legend"
        out[coords[:, 0], coords[:, 1]] = legend_colors[depth - 1]

    return out


def _paint_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _nested_bboxes(outer: BBox, depth: int) -> list[BBox]:
    r0, c0, r1, c1 = outer
    return [(r0 + 2 * i, c0 + 2 * i, r1 - 2 * i, c1 - 2 * i) for i in range(depth)]


def _rectangles_overlap(a: BBox, b: BBox, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _legend_bbox_for_corner(height: int, width: int, depth: int, corner: str, inset: int) -> BBox:
    legend_h = depth + 2
    legend_w = 3
    if corner == "tl":
        r0 = inset
        c0 = inset
    elif corner == "tr":
        r0 = inset
        c0 = width - legend_w - inset
    elif corner == "bl":
        r0 = height - legend_h - inset
        c0 = inset
    else:
        assert corner == "br"
        r0 = height - legend_h - inset
        c0 = width - legend_w - inset
    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _choose_stack_count(max_depth: int) -> int:
    if max_depth == 4:
        return 1
    if max_depth == 3:
        return random.choice([1, 2])
    return random.choice([1, 2, 3])


def _sample_outer_size(depth: int) -> tuple[int, int]:
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 8)
    return inner_h + 4 * (depth - 1), inner_w + 4 * (depth - 1)


def _pick_colors(max_depth: int) -> tuple[int, int, list[int]]:
    bg = random.choice([c for c in PALETTE if c != GREY])
    remaining = [c for c in PALETTE if c != bg and c != GREY]
    source = random.choice(remaining)
    legend_pool = [c for c in remaining if c != source]
    legend_colors = random.sample(legend_pool, max_depth)
    return bg, source, legend_colors


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    max_depth = random.randint(1, MAX_DEPTH)
    stack_count = _choose_stack_count(max_depth)
    stack_depths = [max_depth]
    while len(stack_depths) < stack_count:
        stack_depths.append(random.randint(1, max_depth))
    random.shuffle(stack_depths)

    bg, source, legend_colors = _pick_colors(max_depth)
    height = random.randint(max(14, 8 + 3 * max_depth), 30)
    width = random.randint(max(14, 8 + 3 * max_depth), 30)
    grid = np.full((height, width), bg, dtype=int)

    corner = random.choice(["tl", "tr", "bl", "br"])
    inset = random.randint(0, 1)
    legend_bbox = _legend_bbox_for_corner(height, width, max_depth, corner, inset)
    lr0, lc0, lr1, lc1 = legend_bbox

    placed: list[BBox] = [legend_bbox]
    stack_boxes: list[tuple[BBox, int]] = []
    for depth in stack_depths:
        outer_h, outer_w = _sample_outer_size(depth)
        if outer_h > height or outer_w > width:
            raise RuntimeError("sampled stack too large")

        bbox = None
        for _ in range(200):
            r0 = random.randint(0, height - outer_h)
            c0 = random.randint(0, width - outer_w)
            candidate = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_rectangles_overlap(candidate, other, pad=2) for other in placed):
                continue
            bbox = candidate
            break
        if bbox is None:
            raise RuntimeError("failed to place stack")
        placed.append(bbox)
        stack_boxes.append((bbox, depth))

    for bbox, depth in stack_boxes:
        for layer_bbox in _nested_bboxes(bbox, depth):
            _paint_frame(grid, layer_bbox, source)

    grid[lr0, lc0 : lc1 + 1] = GREY
    grid[lr1, lc0 : lc1 + 1] = GREY
    grid[lr0 : lr1 + 1, lc0] = GREY
    grid[lr0 : lr1 + 1, lc1] = GREY
    for idx, color in enumerate(legend_colors):
        grid[lr0 + 1 + idx, lc0 + 1] = color

    if get_bg(grid) != bg:
        raise RuntimeError("background lost majority")

    expected = np.full_like(grid, bg)
    for bbox, depth in stack_boxes:
        for layer_index, layer_bbox in enumerate(_nested_bboxes(bbox, depth), start=1):
            _paint_frame(expected, layer_bbox, legend_colors[layer_index - 1])
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, expected = _build_instance()
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-pool input")
