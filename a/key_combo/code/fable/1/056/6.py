from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _frame_patch(color: int, height: int, width: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _render_scene(
    shape: tuple[int, int],
    rectangles: dict[int, tuple[int, int, int, int]],
    order_bottom_to_top: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order_bottom_to_top:
        r0, c0, r1, c1 = rectangles[int(color)]
        patch = _frame_patch(int(color), r1 - r0 + 1, c1 - c0 + 1)
        paste(out, patch, r0, c0, transparent=BG)
    return out


def _extract_rectangles(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))
    assert 3 <= len(colors) <= 4, "expected three or four frame colors"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    borders: dict[int, np.ndarray] = {}
    for color in colors:
        visible = grid == int(color)
        box = bbox(visible)
        assert box is not None, "every frame color must be visible"
        r0, c0, r1, c1 = map(int, box)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must be at least 3x3"

        border = _border_mask(grid.shape, box)
        assert bool(np.all(~visible | border)), "visible pixels must lie on a rectangle border"

        visible_crop = crop_bbox(visible.astype(int), box).astype(bool)
        assert bool(visible_crop[0, :].any())
        assert bool(visible_crop[-1, :].any())
        assert bool(visible_crop[:, 0].any())
        assert bool(visible_crop[:, -1].any())

        boxes[int(color)] = box
        borders[int(color)] = border

    return colors, boxes, borders


def _unique_topological_order(colors: list[int], lower_to_upper: set[tuple[int, int]]) -> list[int]:
    outgoing = {int(color): set() for color in colors}
    indegree = {int(color): 0 for color in colors}
    for low, high in lower_to_upper:
        if int(high) not in outgoing[int(low)]:
            outgoing[int(low)].add(int(high))
            indegree[int(high)] += 1

    frontier = sorted(color for color in colors if indegree[int(color)] == 0)
    order: list[int] = []
    while frontier:
        assert len(frontier) == 1, "occlusion clues must determine a unique total order"
        low = int(frontier.pop())
        order.append(low)
        for high in sorted(outgoing[low]):
            indegree[high] -= 1
            if indegree[high] == 0:
                frontier.append(high)
        frontier.sort()

    assert len(order) == len(colors), "occlusion graph must be acyclic and complete enough"
    return order


def _infer_order(
    grid: np.ndarray,
    bg: int,
    colors: list[int],
    borders: dict[int, np.ndarray],
) -> list[int]:
    lower_to_upper: set[tuple[int, int]] = set()
    seen_overlap = False

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            intersection = borders[int(color_a)] & borders[int(color_b)]
            if not bool(intersection.any()):
                continue

            seen_overlap = True
            visible = {int(v) for v in grid[intersection] if int(v) != int(bg)}
            assert visible <= {int(color_a), int(color_b)}
            assert len(visible) == 1, "each border overlap must reveal exactly one upper frame"
            top = visible.pop()
            if top == int(color_a):
                lower_to_upper.add((int(color_b), int(color_a)))
            else:
                lower_to_upper.add((int(color_a), int(color_b)))

    assert seen_overlap, "expected overlapping frames"
    return _unique_topological_order(colors, lower_to_upper)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full hollow-rectangle stack from occlusion gaps, then redraw it with the stacking order fully reversed.

    Input structure:
    - A black background contains three or four differently colored hollow axis-aligned rectangles.
    - Rectangle borders overlap, and only the upper frame's border pixel remains visible at each crossing.
    - Lower frames therefore appear broken exactly where higher frames cover them.

    Rule:
    - For each color, reconstruct its complete rectangle from the visible border fragments.
    - Compare the recovered full borders at every overlap point to infer the unique global bottom-to-top stacking order.
    - Reverse that total order.
    - Redraw the same complete hollow rectangles on the same canvas from the new bottom layer to the new top layer.

    Key insight:
    - The missing pixels are not part of the rectangles' intrinsic shapes; they are occlusion evidence that encodes both the full box outlines and the original layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    colors, rectangles, borders = _extract_rectangles(g, bg)
    order_bottom_to_top = _infer_order(g, bg, colors, borders)
    return _render_scene(g.shape, rectangles, list(reversed(order_bottom_to_top)))


def _strict_offsets(count: int) -> list[int]:
    steps = [random.randint(1, 3) for _ in range(int(count) - 1)]
    offsets = [0]
    for step in steps:
        offsets.append(offsets[-1] + int(step))
    return offsets


def _sample_rectangles(rect_count: int) -> dict[int, tuple[int, int, int, int]] | None:
    common_h = random.randint(4, 8)
    common_w = random.randint(4, 8)

    up = list(reversed(_strict_offsets(rect_count)))
    down = list(reversed(_strict_offsets(rect_count)))
    left_expand = _strict_offsets(rect_count)
    right_expand = _strict_offsets(rect_count)

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 3)

    common_top = pad_top + up[0]
    common_left = pad_left + left_expand[-1]
    common_bottom = common_top + common_h - 1
    common_right = common_left + common_w - 1

    boxes = {}
    for idx in range(int(rect_count)):
        r0 = common_top - up[idx]
        r1 = common_bottom + down[idx]
        c0 = common_left - left_expand[idx]
        c1 = common_right + right_expand[idx]
        boxes[idx] = (int(r0), int(c0), int(r1), int(c1))

    height = max(box[2] for box in boxes.values()) + pad_bottom + 1
    width = max(box[3] for box in boxes.values()) + pad_right + 1
    if height > 30 or width > 30:
        return None

    return boxes


def _maybe_transpose_boxes(
    boxes: dict[int, tuple[int, int, int, int]],
    shape: tuple[int, int],
    transpose: bool,
) -> tuple[dict[int, tuple[int, int, int, int]], tuple[int, int]]:
    if not transpose:
        return boxes, shape

    height, width = map(int, shape)
    transposed: dict[int, tuple[int, int, int, int]] = {}
    for key, (r0, c0, r1, c1) in boxes.items():
        transposed[int(key)] = (int(c0), int(r0), int(c1), int(r1))
    return transposed, (width, height)


def _pairwise_intersections_exist(shape: tuple[int, int], rectangles: dict[int, tuple[int, int, int, int]]) -> bool:
    masks = {key: _border_mask(shape, box) for key, box in rectangles.items()}
    keys = sorted(rectangles)
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            if not bool((masks[int(key_a)] & masks[int(key_b)]).any()):
                return False
    return True


def _has_three_way_overlap(shape: tuple[int, int], rectangles: dict[int, tuple[int, int, int, int]]) -> bool:
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles.values():
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return bool(np.any(cover >= 3))


def _validate_generated_scene(
    inp: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
    order_bottom_to_top: list[int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    if int(get_bg(inp)) != BG:
        return False

    if not _pairwise_intersections_exist(inp.shape, rectangles):
        return False
    if not _has_three_way_overlap(inp.shape, rectangles):
        return False

    try:
        colors, inferred_boxes, borders = _extract_rectangles(inp, BG)
        inferred_order = _infer_order(inp, BG, colors, borders)
    except Exception:
        return False

    expected_boxes = {int(color): tuple(map(int, box)) for color, box in rectangles.items()}
    if inferred_boxes != expected_boxes:
        return False
    if inferred_order != [int(color) for color in order_bottom_to_top]:
        return False

    expected = _render_scene(inp.shape, rectangles, list(reversed(order_bottom_to_top)))
    try:
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    return True


def _gen_attempt(rect_count: int) -> np.ndarray | None:
    geometry = _sample_rectangles(int(rect_count))
    if geometry is None:
        return None

    colors = random.sample(PALETTE, int(rect_count))
    rectangles = {int(color): tuple(map(int, geometry[idx])) for idx, color in enumerate(colors)}
    order_bottom_to_top = random.sample(colors, len(colors))

    union_r0 = min(box[0] for box in rectangles.values())
    union_c0 = min(box[1] for box in rectangles.values())
    union_r1 = max(box[2] for box in rectangles.values())
    union_c1 = max(box[3] for box in rectangles.values())
    shape = (union_r1 + random.randint(2, 4), union_c1 + random.randint(2, 4))

    rectangles, shape = _maybe_transpose_boxes(rectangles, shape, transpose=bool(random.getrandbits(1)))
    inp = _render_scene(shape, rectangles, order_bottom_to_top)

    if not _validate_generated_scene(inp, rectangles, order_bottom_to_top):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    target_count = 3 if _GEN_CALL_COUNT % 2 == 0 else 4
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt(target_count)
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate a valid reversed-stacking scene with {target_count} frames")
