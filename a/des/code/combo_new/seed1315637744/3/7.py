import random
from typing import Optional

import numpy as np


BG = 0
PATH_GREY = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_SEGMENT_STEPS = 4
MAX_SEGMENT_STEPS = 7
ENTRY_OFFSET_RANGE = (2, 4)
EXIT_CLEARANCE_RANGE = (2, 4)
RECT_COUNT_RANGE = (2, 4)
RECT_SPAN_MAX = 7
RECT_THICKNESS_MAX = 6
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _path_order(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    assert coords, "expected a grey path"

    coord_set = set(coords)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        nbrs = [cell for cell in _neighbors4(r, c, mask.shape) if cell in coord_set]
        assert 1 <= len(nbrs) <= 2, "path must be a non-branching 4-connected chain"
        adjacency[(r, c)] = nbrs
        if len(nbrs) == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))

    ordered = [start]
    prev: Optional[tuple[int, int]] = None
    cur = start
    while True:
        nxt = [cell for cell in adjacency[cur] if cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must stay unambiguous"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "grey path must be fully connected"
    return ordered


def _rectangle_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    rects: list[tuple[int, int, int, int, int]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (BG, PATH_GREY):
            continue
        pts = np.argwhere(grid == color)
        assert len(pts) > 0
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        ok = (patch == color) | (patch == PATH_GREY)
        assert bool(ok.all()), "each non-grey color must be one solid rectangle occluded only by the path"
        rects.append((r0, c0, r1, c1, color))
    assert rects, "expected at least one rectangle"
    return rects


def _rectangle_owner(shape: tuple[int, int], rects: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for r0, c0, r1, c1, color in rects:
        patch = owner[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool((patch != -1).any()), "rectangles must not overlap"
        patch[:] = int(color)
    return owner


def _recolor_path(grid: np.ndarray, ordered: list[tuple[int, int]], owner: np.ndarray) -> np.ndarray:
    out = grid.copy()
    active_color: Optional[int] = None
    for idx, (r, c) in enumerate(ordered):
        rect_color = int(owner[r, c]) if owner[r, c] != -1 else None
        out[r, c] = PATH_GREY if rect_color is not None or active_color is None else int(active_color)

        next_rect_color = None
        if idx + 1 < len(ordered):
            nr, nc = ordered[idx + 1]
            if owner[nr, nc] != -1:
                next_rect_color = int(owner[nr, nc])
        if rect_color is not None and next_rect_color != rect_color:
            active_color = rect_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each outside-path segment with the color of the most recently crossed rectangle.

    Input structure:
    - The background is black.
    - Several non-overlapping solid rectangles with distinct colors sit in the grid interior.
    - A one-pixel-wide grey orthogonal path forms one 4-connected non-branching chain; one endpoint is on the top or left border and the other endpoint is on the bottom or right border.
    - Wherever the path crosses a rectangle, those covered rectangle cells appear grey.

    Rule:
    - Trace the grey path from the upper/left border endpoint to the opposite endpoint.
    - Keep the path grey before it reaches the first rectangle.
    - Keep the path grey while it is inside a rectangle.
    - As soon as the path leaves a rectangle, recolor later path cells outside rectangles with that rectangle's color until the path enters the next rectangle.
    - After the last rectangle, color the remaining tail with the last rectangle's color, while leaving the rectangles themselves unchanged apart from their grey path cutout.

    Key insight:
    - The path color depends on the previous rectangle in traversal order, not the region the path is currently occupying, so the crucial step is recovering the ordered path and detecting exit moments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_mask = g == PATH_GREY
    ordered = _path_order(grey_mask)
    rects = _rectangle_bboxes(g)
    owner = _rectangle_owner(g.shape, rects)
    return _recolor_path(g, ordered, owner)


def _segment_orientations(start_side: str, n_segments: int) -> list[str]:
    first = "V" if start_side == "top" else "H"
    second = "H" if first == "V" else "V"
    return [first if i % 2 == 0 else second for i in range(n_segments)]


def _build_path(
    start_side: str,
    steps: list[int],
    start_offset: int,
    side_clearance: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict[str, object]], str]:
    orientations = _segment_orientations(start_side, len(steps))
    end_side = "bottom" if orientations[-1] == "V" else "right"

    total_vertical = sum(step for step, orient in zip(steps, orientations) if orient == "V")
    total_horizontal = sum(step for step, orient in zip(steps, orientations) if orient == "H")

    if start_side == "top":
        h = total_vertical + 1 + (0 if end_side == "bottom" else side_clearance)
        w = start_offset + total_horizontal + 1 + (0 if end_side == "right" else side_clearance)
        cur = (0, start_offset)
    else:
        h = start_offset + total_vertical + 1 + (0 if end_side == "bottom" else side_clearance)
        w = total_horizontal + 1 + (0 if end_side == "right" else side_clearance)
        cur = (start_offset, 0)

    grid = np.full((h, w), BG, dtype=int)
    path = [cur]
    segments: list[dict[str, object]] = []
    for orient, step in zip(orientations, steps):
        dr, dc = (1, 0) if orient == "V" else (0, 1)
        seg_cells = [cur]
        for _ in range(step):
            cur = (cur[0] + dr, cur[1] + dc)
            seg_cells.append(cur)
            path.append(cur)
        segments.append({"orient": orient, "cells": seg_cells})
    return grid, path, segments, end_side


def _bbox_touches(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _path_cells_in_bbox(path: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r, c in path if r0 <= r <= r1 and c0 <= c <= c1}


def _segment_candidates(
    segment: dict[str, object],
    shape: tuple[int, int],
    path: list[tuple[int, int]],
) -> list[dict[str, object]]:
    h, w = shape
    cells = segment["cells"]
    assert isinstance(cells, list)
    orient = segment["orient"]
    assert orient in ("H", "V")
    cell_list = list(cells)
    total = len(cell_list)
    max_span = min(RECT_SPAN_MAX, total - 2)
    candidates: list[dict[str, object]] = []

    for span in range(3, max_span + 1):
        for start_idx in range(1, total - span):
            sub = cell_list[start_idx : start_idx + span]
            if orient == "H":
                row = sub[0][0]
                c0 = sub[0][1]
                c1 = sub[-1][1]
                for rect_h in range(3, min(RECT_THICKNESS_MAX, h - 2) + 1):
                    top_lo = max(1, row - (rect_h - 2))
                    top_hi = min(row - 1, h - rect_h - 1)
                    for top in range(top_lo, top_hi + 1):
                        bbox = (top, c0, top + rect_h - 1, c1)
                        inside = _path_cells_in_bbox(path, bbox)
                        if inside == set(sub):
                            candidates.append({"bbox": bbox, "path_cells": tuple(sub)})
            else:
                col = sub[0][1]
                r0 = sub[0][0]
                r1 = sub[-1][0]
                for rect_w in range(3, min(RECT_THICKNESS_MAX, w - 2) + 1):
                    left_lo = max(1, col - (rect_w - 2))
                    left_hi = min(col - 1, w - rect_w - 1)
                    for left in range(left_lo, left_hi + 1):
                        bbox = (r0, left, r1, left + rect_w - 1)
                        inside = _path_cells_in_bbox(path, bbox)
                        if inside == set(sub):
                            candidates.append({"bbox": bbox, "path_cells": tuple(sub)})
    return candidates


def _choose_rectangles(
    rng: random.Random,
    segments: list[dict[str, object]],
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    rect_count: int,
) -> Optional[list[tuple[int, int, int, int]]]:
    segment_ids = list(range(len(segments)))
    rng.shuffle(segment_ids)
    chosen_segments = segment_ids[:rect_count]

    candidate_map = {idx: _segment_candidates(segments[idx], shape, path) for idx in chosen_segments}
    if any(not candidate_map[idx] for idx in chosen_segments):
        return None

    order = chosen_segments[:]
    rng.shuffle(order)
    placed: list[tuple[int, int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        seg_id = order[pos]
        choices = candidate_map[seg_id][:]
        rng.shuffle(choices)
        for cand in choices:
            bbox = cand["bbox"]
            if any(_bbox_touches(bbox, other, pad=1) for other in placed):
                continue
            placed.append(bbox)
            if backtrack(pos + 1):
                return True
            placed.pop()
        return False

    if not backtrack(0):
        return None
    return placed


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    start_side = rng.choice(["top", "left"])
    n_segments = rng.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    steps = [rng.randint(MIN_SEGMENT_STEPS, MAX_SEGMENT_STEPS) for _ in range(n_segments)]
    start_offset = rng.randint(*ENTRY_OFFSET_RANGE)
    side_clearance = rng.randint(*EXIT_CLEARANCE_RANGE)

    base, path, segments, _ = _build_path(start_side, steps, start_offset, side_clearance)
    h, w = base.shape
    if max(h, w) > 30:
        return None

    rect_count = rng.randint(RECT_COUNT_RANGE[0], min(RECT_COUNT_RANGE[1], n_segments))
    rects = _choose_rectangles(rng, segments, base.shape, path, rect_count)
    if rects is None:
        return None

    colors = rng.sample(RECT_COLORS, rect_count)
    inp = base.copy()
    for bbox, color in zip(rects, colors):
        r0, c0, r1, c1 = bbox
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    for r, c in path:
        inp[r, c] = PATH_GREY

    try:
        out = main(inp)
    except AssertionError:
        return None

    # Keep only scenes where the lag effect is visible more than once.
    outside_changes = int(np.sum((inp == PATH_GREY) & (out != PATH_GREY)))
    if outside_changes < 3:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
