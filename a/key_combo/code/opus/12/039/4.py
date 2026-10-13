from __future__ import annotations

from functools import cmp_to_key

import numpy as np


BACKGROUND = 0
RECTANGLE_COLORS = (1, 2, 3, 4)
BAR_COLORS = (6, 7, 8, 9)
MIN_OBJECTS = 3
MAX_OBJECTS = 4
MIN_SCENE_SIZE = 11
MAX_SCENE_SIZE = 15
MAX_GENERATION_ATTEMPTS = 600


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return int((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def _render_layers(
    shape: tuple[int, int],
    layers: list[dict[str, int | tuple[int, int, int, int]]],
    recolor: dict[int, int] | None = None,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for layer in layers:
        src_color = int(layer["color"])
        color = int(recolor.get(src_color, src_color)) if recolor is not None else src_color
        r0, c0, r1, c1 = layer["box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _ratio_compare(a: tuple[int, int], b: tuple[int, int]) -> int:
    occluded_a, area_a = a
    occluded_b, area_b = b
    left = occluded_a * area_b
    right = occluded_b * area_a
    if left > right:
        return -1
    if left < right:
        return 1
    return 0


def _topological_order(
    colors: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
    scene: np.ndarray,
) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = _intersection(boxes[color_a], boxes[color_b])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            patch = scene[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(patch == color_a))
            b_visible = bool(np.any(patch == color_b))
            assert not (a_visible and b_visible), "overlap must reveal at most one lower/higher relation"
            if not a_visible and not b_visible:
                continue
            lower, higher = (color_b, color_a) if a_visible else (color_a, color_b)
            if higher not in above[lower]:
                above[lower].add(higher)
                indegree[higher] += 1

    order: list[int] = []
    queue = sorted((color for color in colors if indegree[color] == 0), key=lambda color: (boxes[color][0], boxes[color][1], color))
    while queue:
        color = queue.pop(0)
        order.append(color)
        for higher in sorted(above[color]):
            indegree[higher] -= 1
            if indegree[higher] == 0:
                queue.append(higher)
                queue.sort(key=lambda c: (boxes[c][0], boxes[c][1], c))

    assert len(order) == len(colors), "layer relation must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore overlapped solid rectangles, rank them by occlusion ratio, then recolor them from the bottom strip before redrawing by the original layer order.

    Input structure:
    - The background is black, and the last row is a color strip containing one non-black pixel per rectangle.
    - Above the strip are 3 or 4 distinct-colored solid rectangles that may partially overlap, with higher rectangles hiding parts of lower ones.
    - Every rectangle still exposes enough of its boundary that its full axis-aligned bounding box is recoverable from the visible pixels.

    Rule:
    - Remove the last row from consideration and treat each remaining non-black color as one rectangle.
    - Restore each rectangle to the full filled bounding box of its visible pixels, compute its area `A`, its hidden pixel count `O = A - visible_count`, and its occlusion ratio `O / A`.
    - Sort rectangles from largest to smallest occlusion ratio, map that ranking onto the bottom strip colors from left to right, then redraw the restored rectangles with those new colors in the original bottom-to-top overlap order.

    Key insight:
    - The bottom strip does not correspond to position or original color; it is a ranking legend for how much of each rectangle was hidden, so solving the recoloring requires restoring the occluded rectangles first and using the overlap relations only afterwards for the final redraw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid plus a bottom color strip"

    scene = g[:-1]
    strip = g[-1]
    strip_colors = [int(color) for color in strip if int(color) != BACKGROUND]
    assert len(strip_colors) == len(set(strip_colors)), "strip colors must be distinct"

    rectangle_colors = sorted(int(color) for color in np.unique(scene) if int(color) != BACKGROUND)
    assert len(rectangle_colors) == len(strip_colors), "strip length must match rectangle count"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    ratios: dict[int, tuple[int, int]] = {}
    for color in rectangle_colors:
        coords = np.argwhere(scene == color)
        assert coords.size > 0, "each rectangle color must appear"
        box = _bbox_of_coords(coords)
        boxes[color] = box
        visible_count = int(coords.shape[0])
        area = _box_area(box)
        occluded = area - visible_count
        ratios[color] = (int(occluded), int(area))

    order = _topological_order(rectangle_colors, boxes, scene)
    ranked = sorted(rectangle_colors, key=cmp_to_key(lambda a, b: _ratio_compare(ratios[a], ratios[b])))

    for i, color_a in enumerate(ranked):
        for color_b in ranked[i + 1 :]:
            assert _ratio_compare(ratios[color_a], ratios[color_b]) != 0, "occlusion ratios must be unique"

    recolor = {color: strip_colors[idx] for idx, color in enumerate(ranked)}
    rebuilt = [{"color": color, "box": boxes[color]} for color in order]
    return _render_layers(scene.shape, rebuilt, recolor)


def _visible_corner_count(scene: np.ndarray, color: int, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return sum(int(scene[r, c] == color) for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)))


def _sides_visible(scene: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        np.any(scene[r0, c0 : c1 + 1] == color)
        and np.any(scene[r1, c0 : c1 + 1] == color)
        and np.any(scene[r0 : r1 + 1, c0] == color)
        and np.any(scene[r0 : r1 + 1, c1] == color)
    )


def _ratio_key(occluded: int, area: int) -> tuple[int, int]:
    return int(occluded), int(area)


def _candidate_size(rng: np.random.Generator, scene_h: int, scene_w: int, layer_idx: int) -> tuple[int, int]:
    min_h = 4 if layer_idx < 2 else 3
    min_w = 4 if layer_idx < 2 else 3
    max_h = max(min_h, min(scene_h - 2, 8 - layer_idx))
    max_w = max(min_w, min(scene_w - 2, 8 - layer_idx))
    height = int(rng.integers(min_h, max_h + 1))
    width = int(rng.integers(min_w, max_w + 1))
    return height, width


def _build_expected_output(
    scene_shape: tuple[int, int],
    layers: list[dict[str, int | tuple[int, int, int, int]]],
    strip_colors: list[int],
) -> np.ndarray:
    scene = _render_layers(scene_shape, layers)
    ratios: list[tuple[int, int, int]] = []
    for layer in layers:
        color = int(layer["color"])
        box = layer["box"]
        visible = int(np.count_nonzero(scene == color))
        area = _box_area(box)
        ratios.append((color, area - visible, area))

    ranked = [
        color
        for color, _occluded, _area in sorted(
            ratios,
            key=cmp_to_key(
                lambda a, b: _ratio_compare((int(a[1]), int(a[2])), (int(b[1]), int(b[2])))
            ),
        )
    ]
    recolor = {color: strip_colors[idx] for idx, color in enumerate(ranked)}
    return _render_layers(scene_shape, layers, recolor)


def _scene_is_valid(
    scene_shape: tuple[int, int],
    layers: list[dict[str, int | tuple[int, int, int, int]]],
) -> tuple[bool, np.ndarray | None]:
    scene = _render_layers(scene_shape, layers)
    ratios: list[tuple[int, int]] = []
    overlapped_rectangles = 0

    for layer_idx, layer in enumerate(layers):
        color = int(layer["color"])
        box = layer["box"]
        coords = np.argwhere(scene == color)
        if coords.size == 0:
            return False, None
        if _bbox_of_coords(coords) != box:
            return False, None
        if not _sides_visible(scene, color, box):
            return False, None
        if _visible_corner_count(scene, color, box) < 2:
            return False, None

        r0, c0, r1, c1 = box
        if np.any(scene[r0 : r1 + 1, c0 : c1 + 1] == BACKGROUND):
            return False, None

        area = _box_area(box)
        occluded = area - int(coords.shape[0])
        if layer_idx < len(layers) - 1 and occluded == 0:
            return False, None
        if occluded * 100 >= area * 55:
            return False, None
        if occluded > 0:
            overlapped_rectangles += 1
        ratios.append(_ratio_key(occluded, area))

    for i, ratio_a in enumerate(ratios):
        for ratio_b in ratios[i + 1 :]:
            if _ratio_compare(ratio_a, ratio_b) == 0:
                return False, None

    overlap_pairs = 0
    for i, layer_a in enumerate(layers):
        box_a = layer_a["box"]
        area_a = _box_area(box_a)
        for layer_b in layers[i + 1 :]:
            box_b = layer_b["box"]
            overlap = _intersection(box_a, box_b)
            if overlap is None:
                continue
            overlap_pairs += 1
            area_b = _box_area(box_b)
            overlap_area = _box_area(overlap)
            if overlap_area * 100 > min(area_a, area_b) * 45:
                return False, None

    if overlap_pairs < len(layers) - 1:
        return False, None
    if overlapped_rectangles < len(layers) - 1:
        return False, None

    return True, scene


def _try_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    scene_h = int(rng.integers(MIN_SCENE_SIZE, MAX_SCENE_SIZE + 1))
    scene_w = int(rng.integers(MIN_SCENE_SIZE, MAX_SCENE_SIZE + 1))
    n_objects = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))

    layer_colors = [int(color) for color in rng.permutation(RECTANGLE_COLORS)[:n_objects]]
    layers: list[dict[str, int | tuple[int, int, int, int]]] = []
    scene_shape = (scene_h, scene_w)

    for layer_idx, color in enumerate(layer_colors):
        placed = False
        for _ in range(220):
            height, width = _candidate_size(rng, scene_h, scene_w, layer_idx)
            r0 = int(rng.integers(0, scene_h - height + 1))
            c0 = int(rng.integers(0, scene_w - width + 1))
            box = (r0, c0, r0 + height - 1, c0 + width - 1)

            if layer_idx > 0 and not any(_intersection(box, prev["box"]) is not None for prev in layers):
                continue

            candidate_layers = layers + [{"color": color, "box": box}]
            ok, _scene = _scene_is_valid(scene_shape, candidate_layers)
            if not ok:
                continue

            layers = candidate_layers
            placed = True
            break

        if not placed:
            return None

    ok, scene = _scene_is_valid(scene_shape, layers)
    if not ok or scene is None:
        return None

    strip_palette = [int(color) for color in rng.permutation(BAR_COLORS)[:n_objects]]
    strip = np.full(scene_w, BACKGROUND, dtype=int)
    columns = np.sort(rng.choice(np.arange(scene_w), size=n_objects, replace=False))
    strip[columns] = np.array(strip_palette, dtype=int)

    inp = np.vstack([scene, strip])
    expected = _build_expected_output(scene_shape, layers, strip_palette)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-ratio recoloring puzzle")
