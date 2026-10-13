from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAX_GRID_SIZE = 30
COLOR_POOL = tuple(range(1, 10))

# --------------- generator configuration ---------------
GEN_RETRIES = 400
ROOT_COUNT_CHOICES = (1, 1, 2)
ROOT_DEPTH_CHOICES = (2, 3, 4)
OUTER_MARGIN_RANGE = (1, 3)
ROOT_GAP_RANGE = (2, 4)
CHILD_GAP_RANGE = (2, 4)
LEAF_MIN_SIDE = 5
EXTRA_CHILD_PROB = 0.45
CANVAS_EXTRA_MAX = 6


class _Frame:
    def __init__(
        self,
        color: int,
        bbox: tuple[int, int, int, int],
        parent: int | None = None,
        children: list[int] | None = None,
    ) -> None:
        self.color = color
        self.bbox = bbox
        self.parent = parent
        self.children = [] if children is None else children


class _GenNode:
    def __init__(
        self,
        children: list["_GenNode"] | None = None,
        orientation: str | None = None,
        min_h: int = 0,
        min_w: int = 0,
        bbox: tuple[int, int, int, int] | None = None,
        color: int = -1,
    ) -> None:
        self.children = [] if children is None else children
        self.orientation = orientation
        self.min_h = min_h
        self.min_w = min_w
        self.bbox = bbox
        self.color = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill or shrink nested rectangular frames according to their containment depth.

    Input structure:
    - The background is Black.
    - The foreground consists of one-pixel-thick monochrome rectangular frames.
    - Frames may be disjoint or nested; deeper frames lie strictly inside larger frames.
    - The innermost frame interiors contain only background.

    Rule:
    - Build the nesting tree of frames, with outermost frames at depth 1.
    - Odd-depth frames keep their border and fill every interior cell that is not inside a direct child frame's full rectangle.
    - Even-depth frames clear their original border and redraw the same-colored border one cell inward.
    - Apply the transformations from outer depth to inner depth so outer fills do not overwrite inner frames.

    Key insight:
    - The action for a frame is not determined by its color or size, but by the parity of its depth in the global containment tree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    for idx, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if j != idx and _contains(other.bbox, frame.bbox)]
        if containers:
            parent = min(containers, key=lambda j: _bbox_area(frames[j].bbox))
            frames[idx].parent = parent
            frames[parent].children.append(idx)

    depth_cache: dict[int, int] = {}

    def depth(idx: int) -> int:
        if idx in depth_cache:
            return depth_cache[idx]
        parent = frames[idx].parent
        value = 1 if parent is None else depth(parent) + 1
        depth_cache[idx] = value
        return value

    out = g.copy()
    order = sorted(range(len(frames)), key=lambda idx: (depth(idx), _bbox_area(frames[idx].bbox)))
    for idx in order:
        frame = frames[idx]
        if depth(idx) % 2 == 1:
            child_boxes = [frames[child].bbox for child in frame.children]
            _fill_frame_interior(out, frame.bbox, frame.color, child_boxes)
        else:
            _shrink_frame(out, frame.bbox, frame.color)
    return out


def _find_frames(grid: np.ndarray) -> list[_Frame]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    frames: list[_Frame] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            frame = _component_to_frame(coords, color)
            assert frame is not None, "all foreground objects must be rectangular frame components"
            frames.append(frame)
    return frames


def _component_to_frame(coords: list[tuple[int, int]], color: int) -> _Frame | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    actual = set(coords)
    expected = set(_perimeter_cells((r0, c0, r1, c1)))
    if actual != expected:
        return None
    return _Frame(color=color, bbox=(r0, c0, r1, c1))


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _fill_frame_interior(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    child_boxes: list[tuple[int, int, int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    patch = out[r0 + 1 : r1, c0 + 1 : c1]
    if patch.size == 0:
        return
    fill_mask = np.ones(patch.shape, dtype=bool)
    for cr0, cc0, cr1, cc1 in child_boxes:
        fill_mask[cr0 - r0 - 1 : cr1 - r0, cc0 - c0 - 1 : cc1 - c0] = False
    patch[fill_mask] = color


def _shrink_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _perimeter_cells(bbox):
        out[r, c] = BG
    r0, c0, r1, c1 = bbox
    inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    if inner_bbox[0] > inner_bbox[2] or inner_bbox[1] > inner_bbox[3]:
        return
    _draw_frame(out, inner_bbox, color)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_tree(levels: int) -> _GenNode:
    node = _GenNode()
    if levels == 1:
        return node
    node.children.append(_make_tree(levels - 1))
    if random.random() < EXTRA_CHILD_PROB:
        node.children.append(_make_tree(random.randint(1, levels - 1)))
        random.shuffle(node.children)
        node.orientation = random.choice(("h", "v"))
    return node


def _count_nodes(node: _GenNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _compute_min_size(node: _GenNode) -> tuple[int, int]:
    if not node.children:
        node.min_h = LEAF_MIN_SIDE
        node.min_w = LEAF_MIN_SIDE
        return node.min_h, node.min_w

    child_sizes = [_compute_min_size(child) for child in node.children]
    if len(node.children) == 1:
        node.min_h = child_sizes[0][0] + 4
        node.min_w = child_sizes[0][1] + 4
        return node.min_h, node.min_w

    gap = CHILD_GAP_RANGE[0]
    if node.orientation == "h":
        node.min_h = max(size[0] for size in child_sizes) + 4
        node.min_w = sum(size[1] for size in child_sizes) + gap + 4
    else:
        node.min_h = sum(size[0] for size in child_sizes) + gap + 4
        node.min_w = max(size[1] for size in child_sizes) + 4
    return node.min_h, node.min_w


def _place_node(node: _GenNode, slot: tuple[int, int, int, int]) -> bool:
    sr0, sc0, sr1, sc1 = slot
    slot_h = sr1 - sr0 + 1
    slot_w = sc1 - sc0 + 1
    if slot_h < node.min_h or slot_w < node.min_w:
        return False

    h = random.randint(node.min_h, slot_h)
    w = random.randint(node.min_w, slot_w)
    r0 = random.randint(sr0, sr1 - h + 1)
    c0 = random.randint(sc0, sc1 - w + 1)
    node.bbox = (r0, c0, r0 + h - 1, c0 + w - 1)

    if not node.children:
        return True

    ir0, ic0 = r0 + 2, c0 + 2
    ir1, ic1 = r0 + h - 3, c0 + w - 3
    if len(node.children) == 1:
        return _place_node(node.children[0], (ir0, ic0, ir1, ic1))

    first, second = node.children
    gap_min, gap_cap = CHILD_GAP_RANGE
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    if node.orientation == "h":
        max_gap = min(gap_cap, inner_w - first.min_w - second.min_w)
        if max_gap < gap_min:
            return False
        gap = random.randint(gap_min, max_gap)
        extra = inner_w - first.min_w - second.min_w - gap
        extra_left = random.randint(0, extra)
        slot1_w = first.min_w + extra_left
        slot2_w = second.min_w + (extra - extra_left)
        slot1 = (ir0, ic0, ir0 + inner_h - 1, ic0 + slot1_w - 1)
        slot2 = (ir0, slot1[3] + gap + 1, ir0 + inner_h - 1, ic0 + inner_w - 1)
    else:
        max_gap = min(gap_cap, inner_h - first.min_h - second.min_h)
        if max_gap < gap_min:
            return False
        gap = random.randint(gap_min, max_gap)
        extra = inner_h - first.min_h - second.min_h - gap
        extra_top = random.randint(0, extra)
        slot1_h = first.min_h + extra_top
        slot2_h = second.min_h + (extra - extra_top)
        slot1 = (ir0, ic0, ir0 + slot1_h - 1, ic0 + inner_w - 1)
        slot2 = (slot1[2] + gap + 1, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1)
    return _place_node(first, slot1) and _place_node(second, slot2)


def _iter_nodes(node: _GenNode) -> list[_GenNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(_iter_nodes(child))
    return nodes


def _draw_node(grid: np.ndarray, node: _GenNode) -> None:
    assert node.bbox is not None and node.color != -1
    _draw_frame(grid, node.bbox, node.color)
    for child in node.children:
        _draw_node(grid, child)


def _gen_attempt() -> np.ndarray | None:
    root_count = random.choice(ROOT_COUNT_CHOICES)
    roots = [_make_tree(random.choice(ROOT_DEPTH_CHOICES)) for _ in range(root_count)]
    if sum(_count_nodes(root) for root in roots) > len(COLOR_POOL):
        return None

    for root in roots:
        _compute_min_size(root)

    top = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)

    arrangement = None
    gap = 0
    if root_count == 2:
        arrangement = random.choice(("h", "v"))
        gap = random.randint(*ROOT_GAP_RANGE)

    if root_count == 1:
        min_h = roots[0].min_h + top + bottom
        min_w = roots[0].min_w + left + right
    elif arrangement == "h":
        min_h = max(root.min_h for root in roots) + top + bottom
        min_w = sum(root.min_w for root in roots) + gap + left + right
    else:
        min_h = sum(root.min_h for root in roots) + gap + top + bottom
        min_w = max(root.min_w for root in roots) + left + right

    if min_h > MAX_GRID_SIZE or min_w > MAX_GRID_SIZE:
        return None

    max_h = min(MAX_GRID_SIZE, min_h + CANVAS_EXTRA_MAX)
    max_w = min(MAX_GRID_SIZE, min_w + CANVAS_EXTRA_MAX)
    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    if root_count == 1:
        slots = [(top, left, grid_h - bottom - 1, grid_w - right - 1)]
    elif arrangement == "h":
        inner_h = grid_h - top - bottom
        inner_w = grid_w - left - right - gap
        extra = inner_w - roots[0].min_w - roots[1].min_w
        left_extra = random.randint(0, extra)
        slot1_w = roots[0].min_w + left_extra
        slot2_w = roots[1].min_w + (extra - left_extra)
        slot1 = (top, left, top + inner_h - 1, left + slot1_w - 1)
        slot2 = (top, slot1[3] + gap + 1, top + inner_h - 1, slot1[3] + gap + slot2_w)
        slots = [slot1, slot2]
    else:
        inner_h = grid_h - top - bottom - gap
        inner_w = grid_w - left - right
        extra = inner_h - roots[0].min_h - roots[1].min_h
        top_extra = random.randint(0, extra)
        slot1_h = roots[0].min_h + top_extra
        slot2_h = roots[1].min_h + (extra - top_extra)
        slot1 = (top, left, top + slot1_h - 1, left + inner_w - 1)
        slot2 = (slot1[2] + gap + 1, left, slot1[2] + gap + slot2_h, left + inner_w - 1)
        slots = [slot1, slot2]

    for root, slot in zip(roots, slots):
        if not _place_node(root, slot):
            return None

    all_nodes: list[_GenNode] = []
    for root in roots:
        all_nodes.extend(_iter_nodes(root))

    colors = random.sample(COLOR_POOL, len(all_nodes))
    for node, color in zip(all_nodes, colors):
        node.color = color
    for root in roots:
        _draw_node(grid, root)

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid nested-frame input after {GEN_RETRIES} attempts")
