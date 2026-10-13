from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random

import numpy as np


BG = 0
FRAME = 5
SCHEMES = {
    0: (2, 1),  # odd, even
    1: (3, 4),
    2: (6, 7),
}

MAX_DEPTH = 2
GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 26
MAX_GENERATION_ATTEMPTS = 500

LEAF_SIDE_BOUNDS = {
    0: (5, 8),
    1: (5, 7),
    2: (5, 6),
}

# Every pattern has at least two roots so depth cannot be inferred from "the only object".
TREE_PATTERNS = [
    [-1, 0, 1, -1],
    [-1, 0, 0, -1],
    [-1, 0, -1, 2],
    [-1, 0, 1, 0, -1],
    [-1, 0, 0, -1, 3],
    [-1, 0, 1, -1, 3],
    [-1, 0, 1, -1, 3, -1],
    [-1, 0, 1, 0, -1, 4],
    [-1, 0, 0, -1, 3, 3],
    [-1, 0, 1, -1, 3, 4],
]


BBox = tuple[int, int, int, int]


@dataclass
class Node:
    idx: int
    children: list["Node"] = field(default_factory=list)
    depth: int = 0
    size: tuple[int, int] | None = None
    child_offsets: list[tuple[int, int]] = field(default_factory=list)
    bbox: BBox | None = None


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strict_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _boxes_too_close(a: BBox, b: BBox) -> bool:
    return not (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_rectangles(grid: np.ndarray) -> list[BBox]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(np.unique(g).tolist()) <= {BG, FRAME}

    rectangles: list[BBox] = []
    for coords in _components_mask(g == FRAME):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5

        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert np.array_equal(local, _frame_mask(h, w))
        rectangles.append(bbox)

    assert rectangles, "expected at least one grey rectangular frame"
    return rectangles


def _depths_for_rectangles(rectangles: list[BBox]) -> dict[BBox, int]:
    depth_by_box: dict[BBox, int] = {}
    for box in rectangles:
        containers = [other for other in rectangles if other != box and _strict_contains(other, box)]
        for i in range(len(containers)):
            for j in range(i + 1, len(containers)):
                a = containers[i]
                b = containers[j]
                assert _strict_contains(a, b) or _strict_contains(b, a), "containment must form a chain"
        depth = len(containers)
        assert 0 <= depth <= MAX_DEPTH
        depth_by_box[box] = depth
    return depth_by_box


def _paint_rectangle(out: np.ndarray, bbox: BBox, depth: int) -> None:
    odd_color, even_color = SCHEMES[depth]
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            layer = min(r - r0, r1 - r, c - c0, c1 - c)
            if layer == 0:
                out[r, c] = FRAME
            elif layer % 2 == 1:
                out[r, c] = odd_color
            else:
                out[r, c] = even_color


def _oracle_from_bboxes(shape: tuple[int, int], rectangles: list[BBox]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    depth_by_box = _depths_for_rectangles(rectangles)
    for box in rectangles:
        _paint_rectangle(out, box, depth_by_box[box])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey rectangular frame with concentric layers chosen by its bbox-containment depth.

    Input structure:
    - The grid has a black background and several grey axis-aligned rectangular frame objects.
    - Frames are 8-connected, one pixel thick, and some frame bboxes are strictly nested with at least one black-cell gap.
    - The nesting relation forms a tree, so each frame has a unique containment depth.

    Rule:
    - Find every grey rectangular frame and compute how many other frame bboxes strictly contain its bbox.
    - Depth 0 frames use palette (odd layer red 2, even layer blue 1), depth 1 uses (green 3, yellow 4), and depth 2 uses (pink 6, orange 7).
    - Keep the outer border grey, then fill the rectangle interior by rectangular layer depth: cells at layer 1 use the odd color, layer 2 use the even color, and so on.

    Key insight:
    - The geometry of concentric filling is local to each rectangle, but the chosen color pair is a global property determined only by the rectangle's bbox-containment depth.
    """
    g = np.asarray(grid, dtype=int)
    rectangles = _extract_rectangles(g)
    depth_by_box = _depths_for_rectangles(rectangles)
    out = np.array(g, copy=True)
    for box in rectangles:
        _paint_rectangle(out, box, depth_by_box[box])
    return out


def _build_forest(parents: list[int]) -> tuple[list[Node], list[Node]]:
    nodes = [Node(idx=i) for i in range(len(parents))]
    roots: list[Node] = []
    for i, parent in enumerate(parents):
        if parent == -1:
            roots.append(nodes[i])
        else:
            nodes[parent].children.append(nodes[i])

    def assign_depth(node: Node, depth: int) -> None:
        node.depth = depth
        for child in node.children:
            assign_depth(child, depth + 1)

    for root in roots:
        assign_depth(root, 0)
    return roots, nodes


def _place_boxes(height: int, width: int, sizes: list[tuple[int, int]], margin: int) -> list[BBox] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: dict[int, BBox] = {}
    current: list[BBox] = []

    for idx in order:
        h, w = sizes[idx]
        row_max = height - h - margin
        col_max = width - w - margin
        if row_max < margin or col_max < margin:
            return None

        candidates = [
            (r, c)
            for r in range(margin, row_max + 1)
            for c in range(margin, col_max + 1)
        ]
        random.shuffle(candidates)

        chosen = None
        for r, c in candidates:
            box = (r, c, r + h - 1, c + w - 1)
            if any(_boxes_too_close(box, other) for other in current):
                continue
            if any(_strict_contains(box, other) or _strict_contains(other, box) for other in current):
                continue
            chosen = box
            break
        if chosen is None:
            return None
        placed[idx] = chosen
        current.append(chosen)

    return [placed[i] for i in range(len(sizes))]


def _sample_subtree(node: Node) -> bool:
    for child in node.children:
        if not _sample_subtree(child):
            return False

    if not node.children:
        lo, hi = LEAF_SIDE_BOUNDS[node.depth]
        node.size = (random.randint(lo, hi), random.randint(lo, hi))
        node.child_offsets = []
        return True

    child_sizes = [child.size for child in node.children]
    assert all(size is not None for size in child_sizes)
    child_sizes = [size for size in child_sizes if size is not None]

    if len(child_sizes) == 1:
        min_h = child_sizes[0][0] + 4
        min_w = child_sizes[0][1] + 4
    else:
        if random.random() < 0.5:
            min_h = max(size[0] for size in child_sizes) + 4
            min_w = sum(size[1] for size in child_sizes) + 6
        else:
            min_h = sum(size[0] for size in child_sizes) + 6
            min_w = max(size[1] for size in child_sizes) + 4

    max_h = min(GRID_MAX_SIDE - 2, min_h + 4)
    max_w = min(GRID_MAX_SIDE - 2, min_w + 4)
    if min_h > max_h or min_w > max_w:
        return False

    for _ in range(50):
        height = random.randint(min_h, max_h)
        width = random.randint(min_w, max_w)
        boxes = _place_boxes(height, width, child_sizes, margin=2)
        if boxes is None:
            continue
        node.size = (height, width)
        node.child_offsets = [(box[0], box[1]) for box in boxes]
        return True
    return False


def _assign_absolute_bboxes(node: Node, top: int, left: int) -> None:
    assert node.size is not None
    h, w = node.size
    node.bbox = (top, left, top + h - 1, left + w - 1)
    for child, (dr, dc) in zip(node.children, node.child_offsets, strict=False):
        _assign_absolute_bboxes(child, top + dr, left + dc)


def _render_input(shape: tuple[int, int], nodes: list[Node]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for node in nodes:
        assert node.bbox is not None
        r0, c0, r1, c1 = node.bbox
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
    return grid


def _validate_generated(grid: np.ndarray, nodes: list[Node]) -> bool:
    rectangles = _extract_rectangles(grid)
    expected_boxes = sorted(node.bbox for node in nodes if node.bbox is not None)
    if sorted(rectangles) != expected_boxes:
        return False

    depth_by_box = _depths_for_rectangles(rectangles)
    expected_depths = {node.bbox: node.depth for node in nodes if node.bbox is not None}
    if depth_by_box != expected_depths:
        return False

    used_depths = set(depth_by_box.values())
    if used_depths == {0} or 1 not in used_depths:
        return False

    if max(used_depths) == 0:
        return False

    # Anti-shortcut: at least one nested frame is larger than some top-level frame.
    root_areas = []
    child_areas = []
    for box, depth in depth_by_box.items():
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        if depth == 0:
            root_areas.append(area)
        elif depth == 1:
            child_areas.append(area)
    if root_areas and child_areas and max(child_areas) <= min(root_areas):
        return False

    expected = _oracle_from_bboxes(grid.shape, rectangles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    # Every used palette must be visibly exercised by odd and even layers.
    for box, depth in depth_by_box.items():
        r0, c0, r1, c1 = box
        region = expected[r0 : r1 + 1, c0 : c1 + 1]
        odd_color, even_color = SCHEMES[depth]
        if odd_color not in region or even_color not in region:
            return False

    return True


def _attempt_generation() -> np.ndarray | None:
    parents = random.choice(TREE_PATTERNS)
    roots, nodes = _build_forest(parents)

    if max(node.depth for node in nodes) == 0:
        return None

    for root in roots:
        if not _sample_subtree(root):
            return None

    root_sizes = [root.size for root in roots]
    assert all(size is not None for size in root_sizes)
    root_sizes = [size for size in root_sizes if size is not None]

    for _ in range(60):
        height = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
        width = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
        boxes = _place_boxes(height, width, root_sizes, margin=1)
        if boxes is None:
            continue

        for root, box in zip(roots, boxes, strict=False):
            _assign_absolute_bboxes(root, box[0], box[1])

        grid = _render_input((height, width), nodes)
        if _validate_generated(grid, nodes):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle input")
