from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

import numpy as np


Box = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> Box | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_size(box: Box) -> int:
    return max(box[2] - box[0] + 1, box[3] - box[1] + 1)


@lru_cache(maxsize=None)
def _border_cells(box: Box) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return frozenset(cells)


def _infer_frames(grid: np.ndarray) -> tuple[int, dict[int, Box]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: dict[int, Box] = {}
    for color in _colors_present(g, exclude={bg}):
        box = _bbox(g == color)
        assert box is not None, "each frame color must remain visible"
        visible = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        assert visible <= _border_cells(box), "visible frame pixels must lie on their bbox border"
        frames[color] = box
    assert frames, "expected at least one non-background frame"
    return bg, frames


def _occlusion_edges(grid: np.ndarray, frames: dict[int, Box]) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    colors = list(frames)
    edges: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        cells_a = _border_cells(frames[color_a])
        for color_b in colors[i + 1 :]:
            overlap = cells_a & _border_cells(frames[color_b])
            if not overlap:
                continue

            visible = {int(g[r, c]) for r, c in overlap if int(g[r, c]) in (color_a, color_b)}
            if not visible:
                continue
            assert len(visible) == 1, "a pairwise overlap must show a single visible upper frame"

            top = visible.pop()
            if top == color_a:
                edges.add((color_b, color_a))
            else:
                edges.add((color_a, color_b))
    return edges


def _unique_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    adjacency = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in adjacency[lower]:
            adjacency[lower].add(upper)
            indegree[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zero_indegree = sorted(color for color in remaining if indegree[color] == 0)
        assert len(zero_indegree) == 1, "visible overlaps must determine a unique full layer order"
        color = zero_indegree[0]
        remaining.remove(color)
        order.append(color)
        for nxt in adjacency[color]:
            indegree[nxt] -= 1
    return order


def _render_concentric(order: list[int]) -> np.ndarray:
    n = len(order)
    side = 2 * n - 1
    out = np.empty((side, side), dtype=int)
    for inset, color in enumerate(order):
        out[inset : side - inset, inset : side - inset] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder overlapped hollow frames into concentric rings by inferred occlusion depth.

    Input structure:
    - The background is the dominant color.
    - Each non-background color belongs to one 1-pixel-thick hollow rectangular frame,
      possibly partially hidden where higher frames cross it.
    - Frame sizes and positions are unrelated to the final nesting order.

    Rule:
    - Reconstruct each frame as the border of the bbox of its visible pixels.
    - At every border-overlap between two reconstructed frames, whichever frame color is
      visible there is above the other one; combine all such constraints into one unique
      bottom-to-top order.
    - Output a `(2n-1) x (2n-1)` grid for `n` frames, placing the bottom frame on the
      outer ring and moving inward one layer per higher frame until the top frame becomes
      the center pixel.

    Key insight:
    - The intended nesting order is encoded by local occlusion evidence at crossings,
      not by the physical size of the input frames.
    """
    g = np.asarray(grid, dtype=int)
    _bg, frames = _infer_frames(g)
    order = _unique_order(list(frames), _occlusion_edges(g, frames))
    return _render_concentric(order)


GEN_BG = 0
GEN_FRAME_COUNT_RANGE = (3, 5)
GEN_SIZE_MIN = 5
GEN_MAX_ATTEMPTS = 500
GEN_MAX_SHAPE_SPAN = 24
GEN_PADDING_RANGE = (1, 4)
_SEEN_OUTPUT_SIGNATURES: set[tuple[bytes, tuple[int, int]]] = set()


def _box_from_size(r: int, c: int, size: int) -> Box:
    return r, c, r + size - 1, c + size - 1


def _border_overlap_count(box_a: Box, box_b: Box) -> int:
    return len(_border_cells(box_a) & _border_cells(box_b))


def _sample_sizes(count: int, side: int) -> list[int] | None:
    max_size = min(side - 2, 13)
    choices = list(range(GEN_SIZE_MIN, max_size + 1))
    if len(choices) < count:
        return None

    for _ in range(100):
        sizes = random.sample(choices, count)
        if sizes != sorted(sizes) and sizes != sorted(sizes, reverse=True):
            return sizes
    return None


def _diagonal_step(prev: Box, size: int, direction: str) -> Box:
    r0, c0, r1, c1 = prev
    if direction == "dr":
        return _box_from_size(r1, c1, size)
    if direction == "dl":
        return _box_from_size(r1, c0 - size + 1, size)
    if direction == "ur":
        return _box_from_size(r0 - size + 1, c1, size)
    if direction == "ul":
        return _box_from_size(r0 - size + 1, c0 - size + 1, size)
    raise ValueError(f"unknown direction: {direction}")


def _translate_boxes(boxes: list[Box], dr: int, dc: int) -> list[Box]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _union_span(boxes: list[Box]) -> tuple[int, int, int, int]:
    r0 = min(box[0] for box in boxes)
    c0 = min(box[1] for box in boxes)
    r1 = max(box[2] for box in boxes)
    c1 = max(box[3] for box in boxes)
    return r0, c0, r1, c1


def _place_chain(sizes: list[int]) -> list[Box] | None:
    boxes = [_box_from_size(0, 0, sizes[0])]
    directions = ["dr", "dl", "ur", "ul"]

    for size in sizes[1:]:
        options = directions[:]
        random.shuffle(options)
        placed = None
        for direction in options:
            box = _diagonal_step(boxes[-1], size, direction)
            if _border_overlap_count(box, boxes[-1]) != 1:
                continue
            if any(_border_overlap_count(box, older) > 0 for older in boxes[:-1]):
                continue
            placed = box
            break
        if placed is None:
            return None
        boxes.append(placed)

    r0, c0, r1, c1 = _union_span(boxes)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if max(height, width) > GEN_MAX_SHAPE_SPAN:
        return None

    return _translate_boxes(boxes, -r0, -c0)


def _paint_frames(side: int, boxes: list[Box], colors: list[int]) -> np.ndarray:
    grid = np.full((side, side), GEN_BG, dtype=int)
    for box, color in zip(boxes, colors):
        for r, c in _border_cells(box):
            grid[r, c] = int(color)
    return grid


def _size_sorted_order(boxes: list[Box], colors: list[int]) -> list[int]:
    ranked = sorted(
        ((_box_size(box), idx, color) for idx, (box, color) in enumerate(zip(boxes, colors))),
        reverse=True,
    )
    return [color for _size, _idx, color in ranked]


def _validate_instance(grid: np.ndarray, boxes: list[Box], colors: list[int]) -> bool:
    try:
        bg, frames = _infer_frames(grid)
        if bg != GEN_BG:
            return False
    except Exception:
        return False

    if np.count_nonzero(grid == GEN_BG) <= max(np.count_nonzero(grid == color) for color in colors):
        return False

    for box, color in zip(boxes, colors):
        if frames.get(color) != box:
            return False

    edges = _occlusion_edges(grid, frames)
    if any((lower, upper) not in edges for lower, upper in zip(colors, colors[1:])):
        return False

    try:
        order = _unique_order(list(frames), edges)
    except Exception:
        return False
    if order != colors:
        return False

    expected = _render_concentric(colors)
    shortcut = _render_concentric(_size_sorted_order(boxes, colors))
    if np.array_equal(expected, shortcut):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    count = random.randint(*GEN_FRAME_COUNT_RANGE)
    sizes = _sample_sizes(count, GEN_MAX_SHAPE_SPAN)
    if sizes is None:
        return None

    boxes = _place_chain(sizes)
    if boxes is None:
        return None

    r0, c0, r1, c1 = _union_span(boxes)
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    max_pad = min(30 - inner_h, 30 - inner_w)
    if max_pad < GEN_PADDING_RANGE[0] * 2:
        return None

    top = random.randint(GEN_PADDING_RANGE[0], min(GEN_PADDING_RANGE[1], 30 - inner_h - GEN_PADDING_RANGE[0]))
    left = random.randint(GEN_PADDING_RANGE[0], min(GEN_PADDING_RANGE[1], 30 - inner_w - GEN_PADDING_RANGE[0]))
    bottom_room = 30 - inner_h - top
    right_room = 30 - inner_w - left
    bottom = random.randint(GEN_PADDING_RANGE[0], min(GEN_PADDING_RANGE[1], bottom_room))
    right = random.randint(GEN_PADDING_RANGE[0], min(GEN_PADDING_RANGE[1], right_room))
    side = max(inner_h + top + bottom, inner_w + left + right)
    boxes = _translate_boxes(boxes, top, left)

    colors = random.sample(range(1, 10), count)
    grid = _paint_frames(side, boxes, colors)
    if not _validate_instance(grid, boxes, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            out = main(grid)
            signature = (out.tobytes(), tuple(map(int, out.shape)))
            if signature in _SEEN_OUTPUT_SIGNATURES:
                continue
            _SEEN_OUTPUT_SIGNATURES.add(signature)
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order frame puzzle")
