import random

import numpy as np


INPUT_COLORS = (1, 2, 3, 4, 7)
FILL_SEQUENCE = (5, 9, 8, 6, 5)
CORNERS = ("tl", "tr", "br", "bl")
OPPOSITE_CORNER = {"tl": "br", "tr": "bl", "br": "tl", "bl": "tr"}
MIN_SIZE = 4
MAX_SIZE = 7
GEN_RETRIES = 500
CHAIN_RECURSE_TRIES = 40


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_filled_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], fill: int, border: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill
    _paint_border(grid, bbox, border)


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _corner_of_bbox(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1
    if corner == "br":
        return r1, c1
    if corner == "bl":
        return r1, c0
    raise ValueError(f"unknown corner {corner}")


def _bbox_from_corner(anchor: tuple[int, int], corner: str, h: int, w: int) -> tuple[int, int, int, int]:
    r, c = anchor
    if corner == "tl":
        return r, c, r + h - 1, c + w - 1
    if corner == "tr":
        return r, c - w + 1, r + h - 1, c
    if corner == "br":
        return r - h + 1, c - w + 1, r, c
    if corner == "bl":
        return r - h + 1, c, r, c + w - 1
    raise ValueError(f"unknown corner {corner}")


def _in_margin(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin: int = 1) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return margin <= r0 <= r1 < h - margin and margin <= c0 <= c1 < w - margin


def _render_from_order(
    shape: tuple[int, int],
    order: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rank, color in reversed(list(enumerate(order))):
        fill = FILL_SEQUENCE[rank % len(FILL_SEQUENCE)]
        _paint_filled_rect(out, boxes[color], fill, color)
    return out


def _render_input_from_order(
    shape: tuple[int, int],
    order: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in reversed(order):
        _paint_border(out, boxes[color], color)
    return out


def _infer_boxes(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    assert 4 <= len(colors) <= 5, "expected 4 or 5 colored rectangle borders"
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert MIN_SIZE <= (r1 - r0 + 1) <= MAX_SIZE
        assert MIN_SIZE <= (c1 - c0 + 1) <= MAX_SIZE
        border = _border_mask(g.shape, bbox)
        visible = g == color
        assert np.all(border[visible]), "visible colored pixels must lie on a hollow-rectangle border"
        boxes[color] = bbox
    return boxes


def _overlap_relations(
    grid: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(boxes)
    border_masks = {color: _border_mask(g.shape, bbox) for color, bbox in boxes.items()}
    edges: dict[int, set[int]] = {color: set() for color in colors}
    overlap_owner: dict[tuple[int, int], tuple[int, int]] = {}
    occupancy = np.zeros(g.shape, dtype=int)
    for color in colors:
        occupancy += border_masks[color].astype(int)
    assert occupancy.max() <= 2, "at most two rectangles may share a pixel"

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            overlap = border_masks[a] & border_masks[b]
            if not overlap.any():
                continue
            vis = {int(v) for v in np.unique(g[overlap]) if int(v) != 0}
            assert len(vis) == 1, "every overlap pair must show a unique visible front color"
            front = vis.pop()
            assert front in (a, b)
            back = b if front == a else a
            edges[front].add(back)
            for r, c in np.argwhere(overlap):
                overlap_owner[(int(r), int(c))] = (front, back)
    return edges, overlap_owner


def _unique_topological_order(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for node, outs in edges.items():
        for nxt in outs:
            assert nxt in indeg and nxt != node
            indeg[nxt] += 1

    order: list[int] = []
    remaining = set(edges)
    while remaining:
        ready = sorted(node for node in remaining if indeg[node] == 0)
        assert len(ready) == 1, "overlap evidence must imply a unique global depth order"
        node = ready[0]
        remaining.remove(node)
        order.append(node)
        for nxt in edges[node]:
            indeg[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the front-to-back order of overlapping hollow rectangles, then fill each rectangle by depth.

    Input structure:
    - Black background with 4-5 hollow axis-aligned rectangles whose border colors are all distinct.
    - Rectangles are one pixel thick, sized 4x4 to 7x7, and may overlap only where two border pixels coincide.
    - At every shared border pixel, only the front rectangle's color is visible in the input.

    Rule:
    - For each visible border color, recover its rectangle bbox from the colored border pixels.
    - Reconstruct the complete border of every rectangle, examine every pairwise border overlap, and record which color is in front at each shared pixel.
    - Use those pairwise front/back relations to derive the unique global depth order from front to back.
    - Fill rectangle interiors by depth using the repeating sequence grey, maroon, teal, pink, grey, then redraw the original border colors with the same depth layering.

    Key insight:
    - A single overlap reveals only a local front/back fact; the full output needs all overlaps combined into one unique topological depth order before any fill color can be chosen.
    """
    g = np.asarray(grid, dtype=int)
    boxes = _infer_boxes(g)
    edges, _ = _overlap_relations(g, boxes)
    order = _unique_topological_order(edges)
    return _render_from_order(g.shape, order, boxes)


def _valid_new_box(
    existing: list[dict],
    new_box: tuple[int, int, int, int],
    shared_point: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    new_border = _border_mask(shape, new_box)
    for idx, rect in enumerate(existing):
        inter = _bbox_intersection(rect["bbox"], new_box)
        if idx == len(existing) - 1:
            if inter != (shared_point[0], shared_point[1], shared_point[0], shared_point[1]):
                return False
            prev_border = _border_mask(shape, rect["bbox"])
            overlap = prev_border & new_border
            coords = {tuple(map(int, rc)) for rc in np.argwhere(overlap)}
            if coords != {shared_point}:
                return False
        elif inter is not None:
            return False
    return True


def _extend_chain(
    rects: list[dict],
    order: list[int],
    shape: tuple[int, int],
) -> bool:
    if len(rects) == len(order):
        return True

    prev = rects[-1]
    exit_corners = [corner for corner in CORNERS if corner != prev["entry_corner"]]
    random.shuffle(exit_corners)
    for exit_corner in exit_corners:
        shared_point = _corner_of_bbox(prev["bbox"], exit_corner)
        entry_corner = OPPOSITE_CORNER[exit_corner]
        for _ in range(CHAIN_RECURSE_TRIES):
            h = random.randint(MIN_SIZE, MAX_SIZE)
            w = random.randint(MIN_SIZE, MAX_SIZE)
            bbox = _bbox_from_corner(shared_point, entry_corner, h, w)
            if not _in_margin(bbox, shape, margin=1):
                continue
            if not _valid_new_box(rects, bbox, shared_point, shape):
                continue
            rects.append(
                {
                    "color": order[len(rects)],
                    "bbox": bbox,
                    "entry_corner": entry_corner,
                }
            )
            if _extend_chain(rects, order, shape):
                return True
            rects.pop()
    return False


def _side_visibility(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = bbox
    return all(
        (
            np.any(grid[r0, c0 : c1 + 1] == color),
            np.any(grid[r1, c0 : c1 + 1] == color),
            np.any(grid[r0 : r1 + 1, c0] == color),
            np.any(grid[r0 : r1 + 1, c1] == color),
        )
    )


def _scene_semantics_ok(order: list[int], boxes: dict[int, tuple[int, int, int, int]], inp: np.ndarray) -> bool:
    if set(order) != set(boxes):
        return False
    if not (4 <= len(order) <= 5):
        return False

    occupancy = np.zeros(inp.shape, dtype=int)
    for color in order:
        occupancy += _border_mask(inp.shape, boxes[color]).astype(int)
        if not _side_visibility(inp, boxes[color], color):
            return False
    if occupancy.max() > 2:
        return False

    edges, overlap_owner = _overlap_relations(inp, boxes)
    for idx in range(len(order) - 1):
        if order[idx + 1] not in edges[order[idx]]:
            return False
    if len(overlap_owner) != len(order) - 1:
        return False
    try:
        recovered = _unique_topological_order(edges)
    except AssertionError:
        return False
    return recovered == order


def _generate_scene() -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], list[int]] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    n_rect = random.choice((4, 5))
    colors = list(INPUT_COLORS)
    random.shuffle(colors)
    order = colors[:n_rect]

    r_h = random.randint(MIN_SIZE, MAX_SIZE)
    r_w = random.randint(MIN_SIZE, MAX_SIZE)
    r0 = random.randint(1, h - r_h - 2)
    c0 = random.randint(1, w - r_w - 2)
    first = {
        "color": order[0],
        "bbox": (r0, c0, r0 + r_h - 1, c0 + r_w - 1),
        "entry_corner": None,
    }
    rects = [first]
    if not _extend_chain(rects, order, (h, w)):
        return None

    boxes = {rect["color"]: rect["bbox"] for rect in rects}
    inp = _render_input_from_order((h, w), order, boxes)
    if not _scene_semantics_ok(order, boxes, inp):
        return None
    expected = _render_from_order((h, w), order, boxes)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp, boxes, order


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scene = _generate_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid overlapping-rectangle scene")
