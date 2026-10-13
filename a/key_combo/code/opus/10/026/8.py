from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_BASE_SHAPES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 1), (2, 1), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)},
    {(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)},
]


def _normalize_shape(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_variants(base: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    variants = []
    points = list(base)
    for rot in range(4):
        rotated = points
        for _ in range(rot):
            rotated = [(-c, r) for r, c in rotated]
        for flip in (False, True):
            transformed = [(-r, c) if flip else (r, c) for r, c in rotated]
            norm = _normalize_shape(transformed)
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)
    return variants


_SHAPE_LIBRARY = []
for _base in _BASE_SHAPES:
    for _variant in _shape_variants(_base):
        if _variant not in _SHAPE_LIBRARY:
            _SHAPE_LIBRARY.append(_variant)


def _bbox_from_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in _DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    pts = set(comp)
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return pts == expected


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components_of_color(grid, GREY):
        if not _is_rectangular_frame(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_h": int(r1 - r0 - 1),
                "inner_w": int(c1 - c0 - 1),
                "inner_area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
            }
        )
    return frames


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1 + 1, c1 + 1])
    total -= int(prefix[r0, c1 + 1])
    total -= int(prefix[r1 + 1, c0])
    total += int(prefix[r0, c0])
    return total


def _all_black_rectangles(canvas: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = canvas.shape
    black = (canvas == BLACK).astype(int)
    prefix = np.pad(black, ((1, 0), (1, 0)), constant_values=0).cumsum(axis=0).cumsum(axis=1)
    rects = []
    for r0 in range(h):
        for r1 in range(r0 + 1, h):
            for c0 in range(w):
                for c1 in range(c0 + 1, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _rect_sum(prefix, r0, c0, r1, c1) == area:
                        rects.append((area, r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    return rects


def _largest_black_rectangle(canvas: np.ndarray) -> tuple[int, int, int, int, int]:
    rects = _all_black_rectangles(canvas)
    assert rects, "expected at least one all-black rectangle of size >=2x2"
    best_area = max(area for area, *_ in rects)
    best = [rect for rect in rects if rect[0] == best_area]
    assert len(best) == 1, "expected a unique largest all-black rectangle"
    return best[0]


def _extract_template(template_crop: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(template_crop == BLUE)
    assert len(coords) >= 3, "template needs at least three blue pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), offsets


def _largest_black_component_area(canvas: np.ndarray) -> int:
    components = _components_of_color(canvas, BLACK)
    return max(len(comp) for comp in components)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the larger framed scene and stamp the smaller framed blue template into its unique largest empty rectangle.

    Input structure:
    - A black background contains exactly two complete one-pixel-thick grey rectangular frames.
    - The larger frame encloses a canvas of black space plus several small red foreground objects.
    - The smaller frame encloses one compact blue 4-connected object on black, which acts as the template.

    Rule:
    - Identify the two grey frames and assign roles by interior area: the larger interior is the canvas, the smaller interior is the template source.
    - Crop each frame to its interior, then take the blue template object's bounding box and record its blue-pixel offsets.
    - In the large interior, enumerate every axis-aligned all-black rectangle with height and width at least 2, choose the unique maximum-area one, and align the template bbox's top-left corner to that rectangle's top-left corner.
    - Stamp only the template's blue pixels at that aligned position and return the modified large interior.

    Key insight:
    - The two similar-looking frames have different roles, and the placement location is inferred from the canvas's unique largest empty rectangle rather than from any explicit marker pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    frames = _find_frames(g)
    assert len(frames) == 2, "expected exactly two grey rectangular frames"
    frames = sorted(frames, key=lambda item: item["inner_area"], reverse=True)
    big, small = frames
    assert big["inner_area"] > small["inner_area"], "frame roles must be uniquely determined by interior area"

    br0, bc0, br1, bc1 = big["bbox"]
    sr0, sc0, sr1, sc1 = small["bbox"]
    canvas = g[br0 + 1 : br1, bc0 + 1 : bc1].copy()
    template_crop = g[sr0 + 1 : sr1, sc0 + 1 : sc1]

    template_h, template_w, offsets = _extract_template(template_crop)
    _area, rect_r0, rect_c0, rect_h, rect_w = _largest_black_rectangle(canvas)
    assert rect_h >= template_h and rect_w >= template_w, "largest empty rectangle must fit the template bbox"

    out = canvas.copy()
    for dr, dc in offsets:
        out[rect_r0 + dr, rect_c0 + dc] = BLUE
    return out


def _place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], color: int, anchor_r: int, anchor_c: int) -> None:
    for dr, dc in shape:
        grid[anchor_r + dr, anchor_c + dc] = int(color)


def _can_place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], anchor_r: int, anchor_c: int) -> bool:
    h, w = grid.shape
    cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in shape}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BLACK:
            return False
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == RED and (nr, nc) not in cells:
                return False
    return True


def _random_canvas() -> tuple[np.ndarray, tuple[int, int, int, int, int]]:
    for _ in range(500):
        h = random.randint(8, 11)
        w = random.randint(8, 11)
        canvas = np.zeros((h, w), dtype=int)
        object_count = random.randint(2, 4)
        placed = 0
        for _obj in range(object_count):
            shape_choices = random.sample(_SHAPE_LIBRARY, k=min(len(_SHAPE_LIBRARY), 24))
            placed_here = False
            for shape in shape_choices:
                sh, sw = _bbox_from_cells(shape)
                anchors = [
                    (r, c)
                    for r in range(h - sh + 1)
                    for c in range(w - sw + 1)
                    if _can_place_shape(canvas, shape, r, c)
                ]
                if not anchors:
                    continue
                anchor_r, anchor_c = random.choice(anchors)
                _place_shape(canvas, shape, RED, anchor_r, anchor_c)
                placed_here = True
                placed += 1
                break
            if not placed_here:
                break
        if placed != object_count:
            continue

        red_components = _components_of_color(canvas, RED)
        if len(red_components) != object_count:
            continue
        if not all(3 <= len(comp) <= 6 for comp in red_components):
            continue

        rects = _all_black_rectangles(canvas)
        if not rects:
            continue
        best_area = max(area for area, *_ in rects)
        best = [rect for rect in rects if rect[0] == best_area]
        if len(best) != 1:
            continue
        target = best[0]
        _, _r0, _c0, rect_h, rect_w = target
        if rect_h < 2 or rect_w < 2:
            continue

        if _largest_black_component_area(canvas) <= best_area + 2:
            continue

        return canvas, target
    raise RuntimeError("failed to generate a canvas with a unique largest empty rectangle")


def _template_choices(max_h: int, max_w: int) -> list[tuple[tuple[int, int], ...]]:
    choices = []
    for shape in _SHAPE_LIBRARY:
        h, w = _bbox_from_cells(shape)
        if 2 <= h <= min(4, max_h) and 2 <= w <= min(4, max_w):
            choices.append(shape)
    return choices


def _random_template(rect_h: int, rect_w: int) -> tuple[np.ndarray, tuple[int, int], tuple[tuple[int, int], ...]]:
    shapes = _template_choices(rect_h, rect_w)
    assert shapes, "expected at least one template shape to fit the chosen empty rectangle"
    shape = random.choice(shapes)
    sh, sw = _bbox_from_cells(shape)
    inner_h = random.randint(max(3, sh), max(3, sh) if sh == 4 else 4)
    inner_w = random.randint(max(3, sw), max(3, sw) if sw == 4 else 4)
    crop = np.zeros((inner_h, inner_w), dtype=int)
    top = random.randint(0, inner_h - sh)
    left = random.randint(0, inner_w - sw)
    for dr, dc in shape:
        crop[top + dr, left + dc] = BLUE
    offsets = tuple(sorted((int(dr), int(dc)) for dr, dc in shape))
    return crop, (sh, sw), offsets


def _draw_frame_with_interior(grid: np.ndarray, top: int, left: int, interior: np.ndarray) -> tuple[int, int, int, int]:
    h, w = interior.shape
    bottom = top + h + 1
    right = left + w + 1
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top + 1 : bottom, left + 1 : right] = interior
    return top, left, bottom, right


def _compose_grid(canvas: np.ndarray, template_crop: np.ndarray) -> np.ndarray:
    big_h, big_w = canvas.shape
    small_h, small_w = template_crop.shape
    big_outer_h, big_outer_w = big_h + 2, big_w + 2
    small_outer_h, small_outer_w = small_h + 2, small_w + 2

    if random.random() < 0.5:
        gap = random.randint(1, 4)
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        total_h = max(big_outer_h, small_outer_h) + top_margin + bottom_margin
        total_w = big_outer_w + small_outer_w + gap + left_margin + right_margin
        grid = np.zeros((total_h, total_w), dtype=int)
        if random.random() < 0.5:
            big_left = left_margin
            small_left = left_margin + big_outer_w + gap
        else:
            small_left = left_margin
            big_left = left_margin + small_outer_w + gap
        big_top = top_margin + random.randint(0, total_h - top_margin - bottom_margin - big_outer_h)
        small_top = top_margin + random.randint(0, total_h - top_margin - bottom_margin - small_outer_h)
    else:
        gap = random.randint(1, 4)
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        total_h = big_outer_h + small_outer_h + gap + top_margin + bottom_margin
        total_w = max(big_outer_w, small_outer_w) + left_margin + right_margin
        grid = np.zeros((total_h, total_w), dtype=int)
        if random.random() < 0.5:
            big_top = top_margin
            small_top = top_margin + big_outer_h + gap
        else:
            small_top = top_margin
            big_top = top_margin + small_outer_h + gap
        big_left = left_margin + random.randint(0, total_w - left_margin - right_margin - big_outer_w)
        small_left = left_margin + random.randint(0, total_w - left_margin - right_margin - small_outer_w)

    _draw_frame_with_interior(grid, big_top, big_left, canvas)
    _draw_frame_with_interior(grid, small_top, small_left, template_crop)
    return grid


def _expected_output(canvas: np.ndarray, target: tuple[int, int, int, int, int], offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = canvas.copy()
    _, rect_r0, rect_c0, _rect_h, _rect_w = target
    for dr, dc in offsets:
        out[rect_r0 + dr, rect_c0 + dc] = BLUE
    return out


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    target: tuple[int, int, int, int, int],
    template_bbox: tuple[int, int],
    template_offsets: tuple[tuple[int, int], ...],
) -> bool:
    frames = _find_frames(grid)
    if len(frames) != 2:
        return False
    frames = sorted(frames, key=lambda item: item["inner_area"], reverse=True)
    if frames[0]["inner_area"] <= frames[1]["inner_area"]:
        return False

    br0, bc0, br1, bc1 = frames[0]["bbox"]
    sr0, sc0, sr1, sc1 = frames[1]["bbox"]
    canvas = grid[br0 + 1 : br1, bc0 + 1 : bc1]
    template_crop = grid[sr0 + 1 : sr1, sc0 + 1 : sc1]

    red_components = _components_of_color(canvas, RED)
    if not (2 <= len(red_components) <= 4):
        return False
    if not all(3 <= len(comp) <= 6 for comp in red_components):
        return False

    detected_h, detected_w, detected_offsets = _extract_template(template_crop)
    if (detected_h, detected_w) != template_bbox:
        return False
    if detected_offsets != template_offsets:
        return False

    try:
        detected_target = _largest_black_rectangle(canvas)
    except AssertionError:
        return False
    if detected_target != target:
        return False
    if target[3] < template_bbox[0] or target[4] < template_bbox[1]:
        return False
    if _largest_black_component_area(canvas) <= target[0] + 2:
        return False

    return np.array_equal(main(grid), expected)


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(500):
        canvas, target = _random_canvas()
        template_crop, template_bbox, template_offsets = _random_template(target[3], target[4])
        grid = _compose_grid(canvas, template_crop)
        expected = _expected_output(canvas, target, template_offsets)
        if _validate_instance(grid, expected, target, template_bbox, template_offsets):
            return grid, expected
    raise RuntimeError("failed to generate a valid example for task 8")


def generate_input() -> np.ndarray:
    grid, _expected = _generate_instance()
    return grid
