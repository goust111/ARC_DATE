from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MIN_SIDE = 4
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120
COMPONENT_PADDING = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _rect_fill_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    return int(r0), int(c0), int(r1), int(c1)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _component_key(rect_by_color: dict[int, dict], color: int) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = rect_by_color[color]["bbox"]
    return int(r0), int(c0), int(r1), int(c1), int(color)


def _transitive_closure(colors: list[int], edges: dict[int, set[int]]) -> dict[int, set[int]]:
    closure = {int(color): set() for color in colors}
    for src in colors:
        stack = list(edges[src])
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(int(cur))
            stack.extend(edges[cur] - seen)
        closure[src] = seen
    return closure


def _topological_order(
    colors: list[int], edges: dict[int, set[int]], rect_by_color: dict[int, dict]
) -> list[int]:
    indeg = {int(color): 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1

    remaining = set(map(int, colors))
    order: list[int] = []
    while remaining:
        zeros = [color for color in remaining if indeg[color] == 0]
        assert zeros, "layer constraints must form a DAG"
        color = min(zeros, key=lambda value: _component_key(rect_by_color, value))
        remaining.remove(color)
        order.append(color)
        for dst in edges[color]:
            indeg[dst] -= 1
    return order


def _render_hollow(shape: tuple[int, int], rects: list[dict], top_to_bottom: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    rect_by_color = {int(rect["color"]): rect for rect in rects}
    for color in reversed(top_to_bottom):
        out[rect_by_color[color]["border_mask"]] = color
    return out


def _render_solid(shape: tuple[int, int], rects: list[dict], top_to_bottom: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    rect_by_color = {int(rect["color"]): rect for rect in rects}
    for color in reversed(top_to_bottom):
        r0, c0, r1, c1 = rect_by_color[color]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _infer_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects: list[dict] = []
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "every rectangle color must remain visible"
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= MIN_SIDE and c1 - c0 + 1 >= MIN_SIDE, "expected non-degenerate hollow rectangles"
        border_mask = _rect_border_mask(grid.shape, bbox)
        assert border_mask[coords[:, 0], coords[:, 1]].all(), "visible pixels must lie on the inferred border"
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "border_mask": border_mask,
                "fill_mask": _rect_fill_mask(grid.shape, bbox),
            }
        )
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full solid occlusion scene from crossing evidence between hollow rectangle frames.

    Input structure:
    - The grid has a black background and several one-cell-thick hollow axis-aligned rectangles in distinct colors.
    - Rectangle borders may cross, and each crossing cell shows only the color of the higher layer.
    - Visible pixels for one color are fragmented border remnants of a single rectangle.

    Rule:
    - Recover each rectangle's bounding box from the extent of its visible border pixels.
    - At every cell where multiple recovered borders pass, treat the visible color as the rectangle drawn above the other border owners, then close these relations transitively.
    - Output a same-size grid obtained by filling every rectangle's full bounding box and rendering the solid rectangles from bottom to top.

    Key insight:
    - The sparse outline image is not the final geometry: the important information is which rectangle color survives at each border crossing, because those local wins determine the global layer order used to redraw the solid scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    rects = _infer_rectangles(g, bg)
    rect_by_color = {int(rect["color"]): rect for rect in rects}
    colors = [int(rect["color"]) for rect in rects]

    direct_edges = {int(color): set() for color in colors}
    for r, c in np.argwhere(g != bg):
        owners = [color for color in colors if rect_by_color[color]["border_mask"][r, c]]
        visible = int(g[r, c])
        assert visible in owners, "visible crossing color must belong to one recovered rectangle"
        for other in owners:
            if other != visible:
                direct_edges[visible].add(int(other))

    closure = _transitive_closure(colors, direct_edges)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if not _bboxes_overlap(rect_by_color[color_a]["bbox"], rect_by_color[color_b]["bbox"]):
                continue
            a_above_b = color_b in closure[color_a]
            b_above_a = color_a in closure[color_b]
            assert a_above_b ^ b_above_a, "every overlapping rectangle pair must have a unique layer order"

    top_to_bottom = _topological_order(colors, closure, rect_by_color)
    return _render_solid(g.shape, rects, top_to_bottom)


def _sample_chain_component(rng: random.Random, size: int) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    assert 2 <= size <= 4
    min_patch_h = max(10, 2 * size + 4)
    min_patch_w = max(10, 2 * size + 4)
    for _ in range(PLACEMENT_ATTEMPTS):
        patch_h = rng.randint(min_patch_h, min(16, 10 + 2 * size))
        patch_w = rng.randint(min_patch_w, min(16, 10 + 2 * size))
        row_pool = list(range(1, patch_h - 1))
        col_pool = list(range(1, patch_w - 1))
        if len(row_pool) < 2 * size or len(col_pool) < 2 * size:
            continue
        rows = sorted(rng.sample(row_pool, 2 * size))
        cols = sorted(rng.sample(col_pool, 2 * size))
        bboxes = []
        ok = True
        for i in range(size):
            r0 = rows[i]
            r1 = rows[size + i]
            c0 = cols[i]
            c1 = cols[2 * size - 1 - i]
            if r1 - r0 + 1 < MIN_SIDE or c1 - c0 + 1 < MIN_SIDE:
                ok = False
                break
            bboxes.append((int(r0), int(c0), int(r1), int(c1)))
        if ok:
            return (patch_h, patch_w), bboxes
    return None


def _transform_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    transpose: bool,
    flip_r: bool,
    flip_c: bool,
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if transpose:
        h, w = w, h
        r0, c0, r1, c1 = c0, r0, c1, r1
    if flip_r:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_c:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return (h, w), (int(r0), int(c0), int(r1), int(c1))


def _transform_component(
    shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]], rng: random.Random
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    transpose = rng.random() < 0.5
    flip_r = rng.random() < 0.5
    flip_c = rng.random() < 0.5
    new_shape = shape
    out = []
    for bbox in bboxes:
        new_shape, new_bbox = _transform_bbox(
            bbox, shape, transpose=transpose, flip_r=flip_r, flip_c=flip_c
        )
        out.append(new_bbox)
    return new_shape, out


def _translate_bboxes(
    bboxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in bboxes]


def _pad_bbox(
    bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _place_component(
    grid_shape: tuple[int, int],
    local_shape: tuple[int, int],
    local_bboxes: list[tuple[int, int, int, int]],
    used_component_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    h, w = grid_shape
    patch_h, patch_w = local_shape
    if patch_h > h or patch_w > w:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        dr = rng.randint(0, h - patch_h)
        dc = rng.randint(0, w - patch_w)
        placed = _translate_bboxes(local_bboxes, dr, dc)
        union = _bbox_union(placed)
        padded = _pad_bbox(union, COMPONENT_PADDING, grid_shape)
        if any(_bboxes_overlap(padded, other) for other in used_component_boxes):
            continue
        return placed
    return None


def _sample_isolated_rect(
    grid_shape: tuple[int, int], used_component_boxes: list[tuple[int, int, int, int]], rng: random.Random
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    for _ in range(PLACEMENT_ATTEMPTS):
        rh = rng.randint(MIN_SIDE, 9)
        rw = rng.randint(MIN_SIDE, 9)
        if rh > h or rw > w:
            continue
        r0 = rng.randint(0, h - rh)
        c0 = rng.randint(0, w - rw)
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        padded = _pad_bbox(bbox, COMPONENT_PADDING, grid_shape)
        if any(_bboxes_overlap(padded, other) for other in used_component_boxes):
            continue
        return bbox
    return None


def _crossing_cells(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    cells = set()
    for rr in (r0a, r1a):
        for cc in (c0b, c1b):
            if c0a < cc < c1a and r0b < rr < r1b:
                cells.add((int(rr), int(cc)))
    for rr in (r0b, r1b):
        for cc in (c0a, c1a):
            if c0b < cc < c1b and r0a < rr < r1a:
                cells.add((int(rr), int(cc)))
    return sorted(cells)


def _has_bad_pair_geometry(rects: list[dict]) -> bool:
    bboxes = [rect["bbox"] for rect in rects]
    pair_crossings: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, bbox_a in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            bbox_b = bboxes[j]
            cross = _crossing_cells(bbox_a, bbox_b)
            inter = _rect_intersection(bbox_a, bbox_b)
            if inter is not None and not cross:
                return True
            pair_crossings[(i, j)] = cross
    seen: dict[tuple[int, int], int] = {}
    for cross in pair_crossings.values():
        for cell in cross:
            seen[cell] = seen.get(cell, 0) + 1
            if seen[cell] > 1:
                return True
    return False


def _visible_bbox_is_recoverable(inp: np.ndarray, rect: dict) -> bool:
    color = int(rect["color"])
    bbox = rect["bbox"]
    coords = np.argwhere(inp == color)
    if coords.size == 0 or _bbox_coords(coords) != bbox:
        return False
    r0, c0, r1, c1 = bbox
    mask = inp == color
    return bool(
        mask[r0, c0 : c1 + 1].any()
        and mask[r1, c0 : c1 + 1].any()
        and mask[r0 : r1 + 1, c0].any()
        and mask[r0 : r1 + 1, c1].any()
    )


def _make_rect_dicts(bboxes: list[tuple[int, int, int, int]], colors: list[int], shape: tuple[int, int]) -> list[dict]:
    rects = []
    for bbox, color in zip(bboxes, colors):
        rects.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, bbox)),
                "border_mask": _rect_border_mask(shape, bbox),
                "fill_mask": _rect_fill_mask(shape, bbox),
            }
        )
    return rects


def _sample_plan(rng: random.Random) -> tuple[list[int], int]:
    plans = [([2], 2), ([3], 1), ([4], 1), ([2, 2], 1), ([3, 2], 0)]
    return rng.choice(plans)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    grid_shape = (height, width)

    group_sizes, isolate_cap = _sample_plan(rng)
    isolate_count = rng.randint(0, isolate_cap)
    total_rects = sum(group_sizes) + isolate_count
    if total_rects > len(PALETTE):
        return None
    colors = rng.sample(PALETTE, total_rects)

    used_component_boxes: list[tuple[int, int, int, int]] = []
    placed_bboxes: list[tuple[int, int, int, int]] = []
    offset = 0
    for size in group_sizes:
        sampled = _sample_chain_component(rng, size)
        if sampled is None:
            return None
        local_shape, local_bboxes = sampled
        local_shape, local_bboxes = _transform_component(local_shape, local_bboxes, rng)
        placed = _place_component(grid_shape, local_shape, local_bboxes, used_component_boxes, rng)
        if placed is None:
            return None
        used_component_boxes.append(_pad_bbox(_bbox_union(placed), COMPONENT_PADDING, grid_shape))
        placed_bboxes.extend(placed)
        offset += size

    for _ in range(isolate_count):
        bbox = _sample_isolated_rect(grid_shape, used_component_boxes, rng)
        if bbox is None:
            return None
        used_component_boxes.append(_pad_bbox(bbox, COMPONENT_PADDING, grid_shape))
        placed_bboxes.append(bbox)

    rects = _make_rect_dicts(placed_bboxes, colors, grid_shape)
    if _has_bad_pair_geometry(rects):
        return None

    top_to_bottom = rng.sample(colors, len(colors))
    inp = _render_hollow(grid_shape, rects, top_to_bottom)
    expected = _render_solid(grid_shape, rects, top_to_bottom)

    if np.count_nonzero(inp) < 20:
        return None
    for rect in rects:
        if not _visible_bbox_is_recoverable(inp, rect):
            return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-rectangle layering input")
