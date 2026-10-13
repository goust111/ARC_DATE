from __future__ import annotations

import random

import numpy as np


BG = 0
RECT_COUNT_RANGE = (3, 5)
CROP_START_RANGE = (2, 6)
OUTPUT_H_RANGE = (6, 11)
OUTPUT_W_RANGE = (6, 11)
EDGE_TICK_LENGTHS = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _find_crop_window(g: np.ndarray) -> tuple[int, int, int, int]:
    top_colors = [int(c) for c in np.unique(g[0, :]) if int(c) != BG]
    assert len(top_colors) == 1, "expected exactly one top-edge marker color"
    top_color = top_colors[0]
    cols = np.flatnonzero(g[0, :] == top_color)
    assert len(cols) == 2, "expected exactly two top-edge ticks"

    left_colors = [int(c) for c in np.unique(g[:, 0]) if int(c) != BG]
    assert len(left_colors) == 1, "expected exactly one left-edge marker color"
    left_color = left_colors[0]
    rows = np.flatnonzero(g[:, 0] == left_color)
    assert len(rows) == 2, "expected exactly two left-edge ticks"

    return int(rows[0]), int(rows[1]), int(cols[0]), int(cols[1])


def _find_legend(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = g.shape
    for col in range(width - 1, max(-1, width - 4), -1):
        rows = np.flatnonzero(g[:, col] != BG)
        if len(rows) < 2:
            continue
        if rows[-1] != height - 1 or rows[0] < height // 2:
            continue
        if not np.all(np.diff(rows) == 1):
            continue
        if col < width - 1 and np.any(g[:, col + 1 :] != BG):
            continue

        colors = [int(g[row, col]) for row in rows]
        if len(colors) != len(set(colors)):
            continue
        return int(col), [int(row) for row in rows], colors

    raise AssertionError("legend not found near the lower-right corner")


def _compose_crop(
    rectangles: dict[int, tuple[int, int, int, int]],
    order: list[int],
    crop_r0: int,
    crop_r1: int,
    crop_c0: int,
    crop_c1: int,
) -> np.ndarray:
    out = np.full((crop_r1 - crop_r0 + 1, crop_c1 - crop_c0 + 1), BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rectangles[color]
        rr0 = max(r0, crop_r0)
        rr1 = min(r1, crop_r1)
        cc0 = max(c0, crop_c0)
        cc1 = min(c1, crop_c1)
        if rr0 <= rr1 and cc0 <= cc1:
            out[rr0 - crop_r0 : rr1 - crop_r0 + 1, cc0 - crop_c0 : cc1 - crop_c0 + 1] = color
    return out


def _render_rectangles(
    shape: tuple[int, int],
    rectangles: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rectangles[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _visible_bbox_matches(
    scene: np.ndarray, rectangles: dict[int, tuple[int, int, int, int]]
) -> bool:
    for color, bbox in rectangles.items():
        pts = np.argwhere(scene == color)
        if len(pts) == 0:
            return False
        mins = tuple(map(int, pts.min(axis=0)))
        maxs = tuple(map(int, pts.max(axis=0)))
        if mins != bbox[:2] or maxs != bbox[2:]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the edge-marked crop window, reconstruct each overlapping rectangle from its visible footprint, and re-composite the crop by the lower-right depth legend.

    Input structure:
    - The background is black.
    - The top edge has one same-color tick pair marking the crop columns, and the left edge has another same-color tick pair marking the crop rows.
    - Inside the scene are several solid colored rectangles whose visible fragments may overlap, plus a single-pixel vertical legend near the lower-right corner listing the rectangle colors.

    Rule:
    - Read the two top-edge tick columns as the horizontal crop limits and the two left-edge tick rows as the vertical crop limits.
    - Read the lower-right vertical legend from top to bottom as the far-to-near layer order.
    - For each legend color, recover its full rectangle by taking the bounding box of all visible pixels of that color in the scene, then clip those rectangles to the crop window and paint them in legend order so nearer layers overwrite farther ones.

    Key insight:
    - The edge ticks do not describe content; they describe where to crop, while the lower-right color sequence separately describes how to resolve overlaps after each rectangle is reconstructed from its visible extrema.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    crop_r0, crop_r1, crop_c0, crop_c1 = _find_crop_window(g)
    legend_c, legend_rows, depth_order = _find_legend(g)

    scene = g.copy()
    scene[legend_rows, legend_c] = BG

    rectangles = {}
    for color in depth_order:
        pts = np.argwhere(scene == color)
        assert len(pts) > 0, "each legend color must appear in the rectangle scene"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rectangles[int(color)] = (int(r0), int(c0), int(r1), int(c1))

    return _compose_crop(rectangles, depth_order, crop_r0, crop_r1, crop_c0, crop_c1)


def _gen_nested_rectangles(
    height: int,
    width: int,
    crop_r0: int,
    crop_r1: int,
    crop_c0: int,
    crop_c1: int,
    rect_colors: list[int],
) -> dict[int, tuple[int, int, int, int]] | None:
    rect_count = len(rect_colors)
    margin = rect_count - 1

    top_candidates = list(
        range(max(2, crop_r0 - 2), min(crop_r0 + 2, crop_r1 - 2 * margin) + 1)
    )
    left_candidates = list(
        range(max(2, crop_c0 - 2), min(crop_c0 + 2, crop_c1 - 2 * margin) + 1)
    )
    if not top_candidates or not left_candidates:
        return None

    top = random.choice(top_candidates)
    left = random.choice(left_candidates)
    bottom_min = max(crop_r1 - 1, top + 2 * margin + 1)
    bottom_max = min(height - 2, crop_r1 + 2)
    right_min = max(crop_c1 - 1, left + 2 * margin + 1)
    right_max = min(width - 2, crop_c1 + 2)
    if bottom_min > bottom_max or right_min > right_max:
        return None

    bottom = random.randint(bottom_min, bottom_max)
    right = random.randint(right_min, right_max)
    if top <= crop_r0 and left <= crop_c0 and bottom >= crop_r1 and right >= crop_c1:
        adjustments = []
        if crop_r0 + 1 < bottom - 2 * margin:
            adjustments.append((crop_r0 + 1, left, bottom, right))
        if crop_c0 + 1 < right - 2 * margin:
            adjustments.append((top, crop_c0 + 1, bottom, right))
        if top + 2 * margin < crop_r1 - 1:
            adjustments.append((top, left, crop_r1 - 1, right))
        if left + 2 * margin < crop_c1 - 1:
            adjustments.append((top, left, bottom, crop_c1 - 1))
        if not adjustments:
            return None
        top, left, bottom, right = random.choice(adjustments)

    boxes = [(top, left, bottom, right)]
    for _ in range(1, rect_count):
        prev_top, prev_left, prev_bottom, prev_right = boxes[-1]
        choices = []
        for dt in (1, 2):
            for dl in (1, 2):
                for db in (1, 2):
                    for dr in (1, 2):
                        nxt = (
                            prev_top + dt,
                            prev_left + dl,
                            prev_bottom - db,
                            prev_right - dr,
                        )
                        nt, nl, nb, nr = nxt
                        if nb - nt + 1 < 2 or nr - nl + 1 < 2:
                            continue
                        choices.append(nxt)
        if not choices:
            return None
        boxes.append(random.choice(choices))

    return {color: box for color, box in zip(rect_colors, boxes)}


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    crop_r0 = random.randint(*CROP_START_RANGE)
    crop_c0 = random.randint(*CROP_START_RANGE)

    min_out_h = max(OUTPUT_H_RANGE[0], 2 * rect_count)
    min_out_w = max(OUTPUT_W_RANGE[0], 2 * rect_count)
    max_out_h = min(OUTPUT_H_RANGE[1], 30 - crop_r0 - rect_count - 2)
    max_out_w = min(OUTPUT_W_RANGE[1], 30 - crop_c0 - 2)
    if max_out_h < min_out_h or max_out_w < min_out_w:
        return None

    out_h = random.randint(min_out_h, max_out_h)
    out_w = random.randint(min_out_w, max_out_w)
    crop_r1 = crop_r0 + out_h - 1
    crop_c1 = crop_c0 + out_w - 1

    min_height = crop_r1 + rect_count + 2
    min_width = crop_c1 + 2
    if min_height > 30 or min_width > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 4))
    width = random.randint(min_width, min(30, min_width + 4))

    palette = random.sample(range(1, 10), rect_count + 2)
    top_marker_color = palette[0]
    left_marker_color = palette[1]
    rect_colors = palette[2:]

    rectangles = _gen_nested_rectangles(
        height, width, crop_r0, crop_r1, crop_c0, crop_c1, rect_colors
    )
    if rectangles is None:
        return None

    input_order = rect_colors[:]
    depth_order = rect_colors[:]
    for _ in range(80):
        random.shuffle(depth_order)
        if depth_order != input_order and depth_order[-1] != rect_colors[0]:
            break
    else:
        return None

    scene = _render_rectangles((height, width), rectangles, input_order)
    if not _visible_bbox_matches(scene, rectangles):
        return None

    expected = _compose_crop(rectangles, depth_order, crop_r0, crop_r1, crop_c0, crop_c1)
    crop_scene = scene[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    if np.array_equal(crop_scene, expected):
        return None
    if len(set(map(int, np.unique(expected))) - {BG}) < 2:
        return None
    if not np.any(expected == BG):
        return None

    grid = scene.copy()
    grid[: random.choice(EDGE_TICK_LENGTHS), crop_c0] = top_marker_color
    grid[: random.choice(EDGE_TICK_LENGTHS), crop_c1] = top_marker_color
    grid[crop_r0, : random.choice(EDGE_TICK_LENGTHS)] = left_marker_color
    grid[crop_r1, : random.choice(EDGE_TICK_LENGTHS)] = left_marker_color

    legend_rows = list(range(height - rect_count, height))
    for row, color in zip(legend_rows, depth_order):
        grid[row, width - 1] = color

    if list(map(int, np.flatnonzero(grid[:, width - 1] != BG))) != legend_rows:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
