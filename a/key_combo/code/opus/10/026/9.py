from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
FRAME_COUNT_CHOICES = (3, 4)
GRID_ROWS = (13, 18)
GRID_COLS = (13, 18)
FRAME_HEIGHTS = tuple(range(4, 10))
FRAME_WIDTHS = tuple(range(4, 10))
MARGIN = 1
MIN_SHARED = 2
MAX_RETRIES = 600


Box = tuple[int, int, int, int]


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is not None:
        colors = [color for color in colors if color not in exclude]
    return colors


def _bbox(mask: np.ndarray) -> Box | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dims(box: Box) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_area(box: Box) -> int:
    h, w = _dims(box)
    return int(h * w)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _frame_mask(shape: tuple[int, int], box: Box) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _side_cells(box: Box, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown side {side}")


def _shared_row(box: Box, side: str) -> int:
    return int(box[0] if side == "top" else box[2])


def _shared_col(box: Box, side: str) -> int:
    return int(box[1] if side == "left" else box[3])


def _valid_col_starts(width: int, total_cols: int, other_box: Box) -> list[int]:
    starts = []
    other0, other1 = int(other_box[1]), int(other_box[3])
    for c0 in range(MARGIN, total_cols - width - MARGIN + 1):
        overlap = _interval_overlap(c0, c0 + width - 1, other0, other1)
        if MIN_SHARED <= overlap < width:
            starts.append(int(c0))
    return starts


def _valid_row_starts(height: int, total_rows: int, other_box: Box) -> list[int]:
    starts = []
    other0, other1 = int(other_box[0]), int(other_box[2])
    for r0 in range(MARGIN, total_rows - height - MARGIN + 1):
        overlap = _interval_overlap(r0, r0 + height - 1, other0, other1)
        if MIN_SHARED <= overlap < height:
            starts.append(int(r0))
    return starts


def _random_box(shape: tuple[int, int]) -> Box | None:
    rows, cols = shape
    for _ in range(120):
        h = random.choice(FRAME_HEIGHTS)
        w = random.choice(FRAME_WIDTHS)
        if h > rows - 2 * MARGIN or w > cols - 2 * MARGIN:
            continue
        r0 = random.randint(MARGIN, rows - h - MARGIN)
        c0 = random.randint(MARGIN, cols - w - MARGIN)
        return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)
    return None


def _place_one_axis(shape: tuple[int, int], base_box: Box, axis: str, occupied: list[Box]) -> Box | None:
    rows, cols = shape
    for _ in range(240):
        h = random.choice(FRAME_HEIGHTS)
        w = random.choice(FRAME_WIDTHS)
        if h > rows - 2 * MARGIN or w > cols - 2 * MARGIN:
            continue
        if axis == "h":
            own_side = random.choice(("top", "bottom"))
            base_side = random.choice(("top", "bottom"))
            shared = _shared_row(base_box, base_side)
            if own_side == "top":
                r0, r1 = shared, shared + h - 1
            else:
                r0, r1 = shared - h + 1, shared
            if r0 < MARGIN or r1 >= rows - MARGIN:
                continue
            starts = _valid_col_starts(w, cols, base_box)
            if not starts:
                continue
            c0 = random.choice(starts)
            box = (int(r0), int(c0), int(r1), int(c0 + w - 1))
        else:
            own_side = random.choice(("left", "right"))
            base_side = random.choice(("left", "right"))
            shared = _shared_col(base_box, base_side)
            if own_side == "left":
                c0, c1 = shared, shared + w - 1
            else:
                c0, c1 = shared - w + 1, shared
            if c0 < MARGIN or c1 >= cols - MARGIN:
                continue
            starts = _valid_row_starts(h, rows, base_box)
            if not starts:
                continue
            r0 = random.choice(starts)
            box = (int(r0), int(c0), int(r0 + h - 1), int(c1))
        if box not in occupied:
            return box
    return None


def _place_two_axes(
    shape: tuple[int, int],
    h_base: Box,
    v_base: Box,
    occupied: list[Box],
) -> Box | None:
    rows, cols = shape
    for _ in range(400):
        h = random.choice(FRAME_HEIGHTS)
        w = random.choice(FRAME_WIDTHS)
        if h > rows - 2 * MARGIN or w > cols - 2 * MARGIN:
            continue

        own_h = random.choice(("top", "bottom"))
        base_h = random.choice(("top", "bottom"))
        shared_row = _shared_row(h_base, base_h)
        if own_h == "top":
            r0, r1 = shared_row, shared_row + h - 1
        else:
            r0, r1 = shared_row - h + 1, shared_row
        if r0 < MARGIN or r1 >= rows - MARGIN:
            continue

        own_v = random.choice(("left", "right"))
        base_v = random.choice(("left", "right"))
        shared_col = _shared_col(v_base, base_v)
        if own_v == "left":
            c0, c1 = shared_col, shared_col + w - 1
        else:
            c0, c1 = shared_col - w + 1, shared_col
        if c0 < MARGIN or c1 >= cols - MARGIN:
            continue

        if _interval_overlap(c0, c1, h_base[1], h_base[3]) < MIN_SHARED:
            continue
        if _interval_overlap(c0, c1, h_base[1], h_base[3]) >= w:
            continue
        if _interval_overlap(r0, r1, v_base[0], v_base[2]) < MIN_SHARED:
            continue
        if _interval_overlap(r0, r1, v_base[0], v_base[2]) >= h:
            continue

        box = (int(r0), int(c0), int(r1), int(c1))
        if box not in occupied:
            return box
    return None


def _render_scene(shape: tuple[int, int], colors_front_to_back: list[int], boxes_front_to_back: list[Box]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, box in reversed(list(zip(colors_front_to_back, boxes_front_to_back))):
        grid[_frame_mask(shape, box)] = int(color)
    return grid


def _reconstruct_frames(grid: np.ndarray) -> dict[int, Box]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = _colors_present(g, exclude={BG})
    assert 3 <= len(colors) <= 4, "expected 3 or 4 frame colors"

    frames: dict[int, Box] = {}
    for color in colors:
        mask = g == int(color)
        box = _bbox(mask)
        assert box is not None, f"color {color} must be visible"
        full = _frame_mask(g.shape, box)
        assert np.all(mask <= full), f"color {color} must lie on one rectangular frame border"
        for side in ("top", "bottom", "left", "right"):
            assert any(mask[r, c] for r, c in _side_cells(box, side)), f"color {color} lost an entire side"
        frames[int(color)] = box
    return frames


def _infer_edges(grid: np.ndarray, frames: dict[int, Box]) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    masks = {color: _frame_mask(g.shape, box) for color, box in frames.items()}
    colors = list(frames)
    edges = {color: set() for color in colors}
    for idx, color_a in enumerate(colors):
        for color_b in colors[idx + 1 :]:
            overlap = masks[color_a] & masks[color_b]
            if not np.any(overlap):
                continue
            seen_a = bool(np.any(g[overlap] == int(color_a)))
            seen_b = bool(np.any(g[overlap] == int(color_b)))
            assert not (seen_a and seen_b), "pairwise overlap implies contradictory depth order"
            if seen_a:
                edges[color_a].add(int(color_b))
            elif seen_b:
                edges[color_b].add(int(color_a))
    return edges


def _unique_toposort(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        zeros = sorted(color for color in remaining if indeg[color] == 0)
        assert len(zeros) == 1, "depth order must be uniquely determined"
        cur = zeros[0]
        order.append(int(cur))
        remaining.remove(cur)
        for nxt in edges[cur]:
            indeg[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the front-to-back order of overlapping incomplete rectangular frames and output that color sequence as a one-row strip.

    Input structure
    - A black-background grid contains 3 or 4 distinct non-black colors.
    - Each color belongs to one single-pixel-thick rectangular frame, but deeper frames may have visible gaps where shallower frames cover shared border cells.
    - The visible pixels of each color still determine one axis-aligned frame location, and the overall scene comes from a total front-to-back drawing order.

    Rule
    - Reconstruct each color's complete rectangular frame from the bounding box of its visible border pixels.
    - For every pair of reconstructed frames, inspect their shared border cells: if a shared cell shows color A while frame B should also pass through it, then A is in front of B.
    - Topologically sort all inferred pairwise blocking relations and output a 1xK row whose entries list the frame colors from front to back.

    Key insight
    - The missing pixels matter only after reconstructing the full frame geometry: the visible color on a shared border cell is a direct witness of which reconstructed frame was drawn later.
    """
    frames = _reconstruct_frames(np.asarray(grid, dtype=int))
    edges = _infer_edges(np.asarray(grid, dtype=int), frames)
    order = _unique_toposort(list(frames), edges)
    return np.asarray([order], dtype=int)


# --------------- generator (o5x) ---------------


def _side_occlusion_count(grid: np.ndarray, color: int, box: Box) -> int:
    count = 0
    for side in ("top", "bottom", "left", "right"):
        cells = _side_cells(box, side)
        if any(int(grid[r, c]) not in (BG, int(color)) for r, c in cells):
            count += 1
    return count


def _build_boxes(shape: tuple[int, int], count: int) -> list[Box] | None:
    first_axis = random.choice(("h", "v"))
    box0 = _random_box(shape)
    if box0 is None:
        return None

    if first_axis == "h":
        box1 = _place_one_axis(shape, box0, "h", [box0])
        if box1 is None:
            return None
        box2 = _place_two_axes(shape, box0, box1, [box0, box1])
        if box2 is None:
            return None
        boxes = [box0, box1, box2]
        if count == 4:
            box3 = _place_two_axes(shape, box1, box2, boxes)
            if box3 is None:
                return None
            boxes.append(box3)
        return boxes

    box1 = _place_one_axis(shape, box0, "v", [box0])
    if box1 is None:
        return None
    box2 = _place_two_axes(shape, box1, box0, [box0, box1])
    if box2 is None:
        return None
    boxes = [box0, box1, box2]
    if count == 4:
        box3 = _place_two_axes(shape, box2, box1, boxes)
        if box3 is None:
            return None
        boxes.append(box3)
    return boxes


def _semantic_checks(
    grid: np.ndarray,
    colors_front_to_back: list[int],
    boxes_front_to_back: list[Box],
) -> bool:
    shape = grid.shape
    frame_masks = {color: _frame_mask(shape, box) for color, box in zip(colors_front_to_back, boxes_front_to_back)}

    top_color = colors_front_to_back[0]
    if int(np.count_nonzero(grid == int(top_color))) != int(np.count_nonzero(frame_masks[top_color])):
        return False

    for color, box in zip(colors_front_to_back, boxes_front_to_back):
        visible = grid == int(color)
        if not np.any(visible):
            return False
        if _bbox(visible) != box:
            return False
        if not np.all(visible <= frame_masks[color]):
            return False
        if int(np.count_nonzero(visible)) < 6:
            return False

    overlap_counts = np.zeros(shape, dtype=int)
    for box in boxes_front_to_back:
        overlap_counts += _frame_mask(shape, box).astype(int)
    union = int(np.count_nonzero(overlap_counts))
    multi = int(np.count_nonzero(overlap_counts >= 2))
    if union == 0 or multi * 10 > union * 4:
        return False

    if not any(_side_occlusion_count(grid, color, box) >= 2 for color, box in zip(colors_front_to_back[1:], boxes_front_to_back[1:])):
        return False

    for i in range(len(colors_front_to_back) - 1):
        front_color = colors_front_to_back[i]
        overlap = frame_masks[front_color] & frame_masks[colors_front_to_back[i + 1]]
        if int(np.count_nonzero(overlap)) < MIN_SHARED:
            return False
        if not np.any(grid[overlap] == int(front_color)):
            return False

    for i, box in enumerate(boxes_front_to_back):
        for other in boxes_front_to_back[:i]:
            if _bbox_area(box) == _bbox_area(other) and box == other:
                return False

    try:
        reconstructed = _reconstruct_frames(grid)
        inferred = _unique_toposort(list(reconstructed), _infer_edges(grid, reconstructed))
    except AssertionError:
        return False
    if inferred != colors_front_to_back:
        return False

    areas = {color: _bbox_area(box) for color, box in zip(colors_front_to_back, boxes_front_to_back)}
    visibles = {color: int(np.count_nonzero(grid == int(color))) for color in colors_front_to_back}
    area_desc = [color for color, _ in sorted(areas.items(), key=lambda item: (-item[1], item[0]))]
    area_asc = [color for color, _ in sorted(areas.items(), key=lambda item: (item[1], item[0]))]
    visible_desc = [color for color, _ in sorted(visibles.items(), key=lambda item: (-item[1], item[0]))]
    visible_asc = [color for color, _ in sorted(visibles.items(), key=lambda item: (item[1], item[0]))]
    if inferred in (area_desc, area_asc, visible_desc, visible_asc):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    count = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(PALETTE, count)
    boxes = _build_boxes((rows, cols), count)
    if boxes is None:
        return None

    grid = _render_scene((rows, cols), colors, boxes)
    expected = np.asarray([colors], dtype=int)
    if not _semantic_checks(grid, colors, boxes):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame depth input")
