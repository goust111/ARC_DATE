from __future__ import annotations

import random

import numpy as np


BG = 0

# Generator configuration.
MIN_RECTS = 3
MAX_RECTS = 5
TOP_SIZE_MIN = 2
TOP_SIZE_MAX = 6
LAYER_MARGIN_MIN = 1
LAYER_MARGIN_MAX = 3
GRID_MAX = 30
PAD_MIN = 1
PAD_MAX = 4
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected at least one visible cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_has_visible(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _precedence_graph(
    masks: dict[int, np.ndarray], bboxes: dict[int, tuple[int, int, int, int]]
) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {color: set() for color in masks}
    for upper, mask in masks.items():
        for lower, bbox in bboxes.items():
            if upper != lower and _bbox_has_visible(mask, bbox):
                graph[upper].add(lower)
    return graph


def _reachable_from(graph: dict[int, set[int]], start: int) -> set[int]:
    seen: set[int] = set()
    stack = list(graph[start])
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph[cur] - seen)
    return seen


def _bottom_color_from_grid(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert len(colors) >= 3, "expected at least three visible rectangles"

    masks = {color: grid == color for color in colors}
    bboxes = {color: _bbox_from_mask(mask) for color, mask in masks.items()}
    graph = _precedence_graph(masks, bboxes)
    reach = {color: _reachable_from(graph, color) for color in colors}

    assert all(color not in reach[color] for color in colors), "occlusion graph must be acyclic"
    below_counts = {
        color: sum(color in reach[other] for other in colors if other != color)
        for color in colors
    }
    candidates = [color for color, count in below_counts.items() if count == len(colors) - 1]
    assert len(candidates) == 1, "expected a unique bottom-most rectangle"
    bottom = candidates[0]
    return bottom, bboxes[bottom]


def _paint_rect(
    canvas: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each rectangle's full bbox from its visible pixels, infer the layering
    order from visible-over-inside-bbox evidence, and return only the bottom rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color, bbox = _bottom_color_from_grid(g)
    out = np.zeros_like(g)
    _paint_rect(out, color, bbox)
    return out


def _make_nested_rectangles() -> tuple[tuple[int, int], list[tuple[int, tuple[int, int, int, int]]]]:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)

    rects_inner_to_outer: list[tuple[int, int, int, int]] = []
    top_h = random.randint(TOP_SIZE_MIN, TOP_SIZE_MAX)
    top_w = random.randint(TOP_SIZE_MIN, TOP_SIZE_MAX)
    rects_inner_to_outer.append((0, 0, top_h - 1, top_w - 1))

    for _ in range(n_rects - 1):
        r0, c0, r1, c1 = rects_inner_to_outer[-1]
        expand_top = random.randint(LAYER_MARGIN_MIN, LAYER_MARGIN_MAX)
        expand_bottom = random.randint(LAYER_MARGIN_MIN, LAYER_MARGIN_MAX)
        expand_left = random.randint(LAYER_MARGIN_MIN, LAYER_MARGIN_MAX)
        expand_right = random.randint(LAYER_MARGIN_MIN, LAYER_MARGIN_MAX)
        rects_inner_to_outer.append(
            (
                r0 - expand_top,
                c0 - expand_left,
                r1 + expand_bottom,
                c1 + expand_right,
            )
        )

    outer_r0, outer_c0, outer_r1, outer_c1 = rects_inner_to_outer[-1]
    outer_h = outer_r1 - outer_r0 + 1
    outer_w = outer_c1 - outer_c0 + 1

    if outer_h + 2 * PAD_MIN > GRID_MAX or outer_w + 2 * PAD_MIN > GRID_MAX:
        raise ValueError("nested stack does not fit")

    top_pad = random.randint(PAD_MIN, min(PAD_MAX, GRID_MAX - outer_h - PAD_MIN))
    bottom_pad = random.randint(PAD_MIN, min(PAD_MAX, GRID_MAX - outer_h - top_pad))
    left_pad = random.randint(PAD_MIN, min(PAD_MAX, GRID_MAX - outer_w - PAD_MIN))
    right_pad = random.randint(PAD_MIN, min(PAD_MAX, GRID_MAX - outer_w - left_pad))

    shift_r = top_pad - outer_r0
    shift_c = left_pad - outer_c0
    rects_bottom_to_top: list[tuple[int, tuple[int, int, int, int]]] = []
    colors = random.sample(PALETTE, n_rects)
    for color, bbox in zip(colors, reversed(rects_inner_to_outer), strict=True):
        r0, c0, r1, c1 = bbox
        rects_bottom_to_top.append((color, (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)))

    grid_shape = (outer_h + top_pad + bottom_pad, outer_w + left_pad + right_pad)
    return grid_shape, rects_bottom_to_top


def _render_scene(
    grid_shape: tuple[int, int], rects_bottom_to_top: list[tuple[int, tuple[int, int, int, int]]]
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for color, bbox in rects_bottom_to_top:
        _paint_rect(grid, color, bbox)
    return grid


def _expected_output(
    grid_shape: tuple[int, int], rects_bottom_to_top: list[tuple[int, tuple[int, int, int, int]]]
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    bottom_color, bottom_bbox = rects_bottom_to_top[0]
    _paint_rect(out, bottom_color, bottom_bbox)
    return out


def _valid_instance(
    grid: np.ndarray,
    rects_bottom_to_top: list[tuple[int, tuple[int, int, int, int]]],
) -> bool:
    colors = {color for color, _bbox in rects_bottom_to_top}
    if len(colors) != len(rects_bottom_to_top):
        return False

    for color, bbox in rects_bottom_to_top:
        mask = grid == color
        if not mask.any():
            return False
        if _bbox_from_mask(mask) != bbox:
            return False

    expected = _expected_output(grid.shape, rects_bottom_to_top)
    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid_shape, rects_bottom_to_top = _make_nested_rectangles()
        except ValueError:
            continue

        grid = _render_scene(grid_shape, rects_bottom_to_top)
        if _valid_instance(grid, rects_bottom_to_top):
            return grid

    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
