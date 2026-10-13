from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    color: int = -1
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    children: list["_Node"] = field(default_factory=list)
    parent: _Node | None = None
    depth: int = 0
    h: int = 0
    w: int = 0
    orient: str = "h"
    gap: int = 0
    margins: tuple[int, int, int, int] = (0, 0, 0, 0)


PATTERN_LIBRARY = [
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
]

RNG = np.random.default_rng()

BG = 0
LEGEND_COLOR = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (22, 34)
MAX_DEPTH_RANGE = (1, 4)
MAX_FRAME_COUNT = 7
LEAF_SIDE_RANGE = (4, 8)
GAP_RANGE = (2, 4)
MARGIN_RANGE = (1, 4)
GEN_ATTEMPTS = 300
ROOT_PLACE_ATTEMPTS = 200


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray], tuple[int, int, int, int]]:
    h, w = grid.shape
    colors = sorted(set(map(int, np.unique(grid))) - {bg})
    for color in colors:
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h != 3 or box_w < 3 or (box_w + 1) % 4 != 0:
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue

        patterns: list[np.ndarray] = []
        count = (box_w + 1) // 4
        ok = True
        for i in range(count):
            cc = c0 + 4 * i
            block = grid[r0 : r0 + 3, cc : cc + 3] == color
            if block.shape != (3, 3) or not bool(block.any()):
                ok = False
                break
            sub_pts = np.argwhere(block)
            br0, bc0 = sub_pts.min(axis=0)
            br1, bc1 = sub_pts.max(axis=0)
            if (int(br0), int(bc0), int(br1), int(bc1)) != (0, 0, 2, 2):
                ok = False
                break
            patterns.append(block.copy())
            if i < count - 1 and bool((grid[r0 : r0 + 3, cc + 3] != bg).any()):
                ok = False
                break
        if ok:
            return int(color), patterns, (r0, c0, r1, c1)

    raise AssertionError("legend not found")


def _find_frames(grid: np.ndarray, bg: int, legend_color: int, legend_bbox: tuple[int, int, int, int]) -> list[_Node]:
    r0, c0, r1, c1 = legend_bbox
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    frames: list[_Node] = []
    colors = sorted(set(map(int, np.unique(grid))) - {bg, legend_color})
    for color in colors:
        color_mask = (grid == color) & (~legend_mask)
        for comp in _connected_components(color_mask):
            rr0, cc0 = comp.min(axis=0)
            rr1, cc1 = comp.max(axis=0)
            rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
            box_h = rr1 - rr0 + 1
            box_w = cc1 - cc0 + 1
            if box_h < 3 or box_w < 3:
                continue
            patch = grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == color
            frame_mask = np.zeros((box_h, box_w), dtype=bool)
            frame_mask[0, :] = True
            frame_mask[-1, :] = True
            frame_mask[:, 0] = True
            frame_mask[:, -1] = True
            if np.array_equal(patch, frame_mask):
                frames.append(_Node(color=int(color), bbox=(rr0, cc0, rr1, cc1)))

    assert frames, "at least one frame is required"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _build_tree(nodes: list[_Node]) -> list[_Node]:
    for node in nodes:
        node.children = []
        node.parent = None
        node.depth = 0

    for node in nodes:
        containers = [other for other in nodes if other is not node and _contains(other.bbox, node.bbox)]
        if containers:
            node.parent = min(containers, key=lambda item: _bbox_area(item.bbox))
            node.parent.children.append(node)

    roots = [node for node in nodes if node.parent is None]
    assert roots, "a containment tree needs at least one root"

    def assign_depth(node: _Node, depth: int) -> None:
        node.depth = depth
        for child in node.children:
            assign_depth(child, depth + 1)

    for root in roots:
        assign_depth(root, 1)
    return roots


def _collect_nodes(roots: list[_Node]) -> list[_Node]:
    out: list[_Node] = []

    def visit(node: _Node) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return out


def _render_scene(shape: tuple[int, int], roots: list[_Node], patterns: list[np.ndarray], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    nodes = _collect_nodes(roots)

    for node in nodes:
        r0, c0, r1, c1 = node.bbox
        out[r0, c0 : c1 + 1] = node.color
        out[r1, c0 : c1 + 1] = node.color
        out[r0 : r1 + 1, c0] = node.color
        out[r0 : r1 + 1, c1] = node.color

    for node in sorted(nodes, key=lambda item: item.depth):
        r0, c0, r1, c1 = node.bbox
        if r1 - r0 <= 1 or c1 - c0 <= 1:
            continue
        pattern = patterns[node.depth - 1]
        blocked = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
        for child in node.children:
            cr0, cc0, cr1, cc1 = child.bbox
            blocked[cr0 - (r0 + 1) : cr1 - r0, cc0 - (c0 + 1) : cc1 - c0] = True
        for rr in range(r1 - r0 - 1):
            for cc in range(c1 - c0 - 1):
                if blocked[rr, cc]:
                    continue
                if pattern[rr % 3, cc % 3]:
                    out[r0 + 1 + rr, c0 + 1 + cc] = node.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every rectangular frame with the legend pattern selected by nesting depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_bg(g)
    legend_color, patterns, legend_bbox = _find_legend(g, bg)
    frames = _find_frames(g, bg, legend_color, legend_bbox)
    roots = _build_tree(frames)
    max_depth = max(node.depth for node in _collect_nodes(roots))
    assert len(patterns) == max_depth, "legend width must match the deepest nesting level"
    return _render_scene(g.shape, roots, patterns, bg)


def _make_tree(max_depth: int) -> list[_Node]:
    chain = [_Node() for _ in range(max_depth)]
    for i in range(max_depth - 1):
        chain[i].children.append(chain[i + 1])
    roots = [chain[0]]
    budget = MAX_FRAME_COUNT - max_depth

    for node in chain[:-1]:
        if budget > 0 and float(RNG.random()) < 0.55:
            node.children.append(_Node())
            budget -= 1

    if max_depth >= 2 and budget > 1 and float(RNG.random()) < 0.35:
        roots.append(_Node(children=[_Node()]))
        budget -= 2
    elif budget > 0 and float(RNG.random()) < 0.65:
        roots.append(_Node())
        budget -= 1

    if budget > 0 and len(roots) < 3 and float(RNG.random()) < 0.35:
        roots.append(_Node())

    return roots


def _assign_depths(roots: list[_Node]) -> None:
    def visit(node: _Node, depth: int) -> None:
        node.depth = depth
        node.parent = None
        for child in node.children:
            child.parent = node
            visit(child, depth + 1)

    for root in roots:
        visit(root, 1)


def _assign_sizes(node: _Node) -> None:
    for child in node.children:
        _assign_sizes(child)

    if not node.children:
        node.h = int(RNG.integers(*LEAF_SIDE_RANGE))
        node.w = int(RNG.integers(*LEAF_SIDE_RANGE))
        return

    perm = list(RNG.permutation(len(node.children)))
    node.children = [node.children[i] for i in perm]
    node.orient = "h" if float(RNG.random()) < 0.5 else "v"
    node.gap = int(RNG.integers(*GAP_RANGE))
    top = int(RNG.integers(*MARGIN_RANGE))
    bottom = int(RNG.integers(*MARGIN_RANGE))
    left = int(RNG.integers(*MARGIN_RANGE))
    right = int(RNG.integers(*MARGIN_RANGE))
    node.margins = (top, bottom, left, right)

    if node.orient == "h":
        content_h = max(child.h for child in node.children)
        content_w = sum(child.w for child in node.children) + node.gap * (len(node.children) - 1)
    else:
        content_h = sum(child.h for child in node.children) + node.gap * (len(node.children) - 1)
        content_w = max(child.w for child in node.children)

    node.h = content_h + top + bottom + 2
    node.w = content_w + left + right + 2


def _set_bboxes(node: _Node, top: int, left: int) -> None:
    node.bbox = (top, left, top + node.h - 1, left + node.w - 1)
    if not node.children:
        return

    margin_top, margin_bottom, margin_left, margin_right = node.margins
    inner_top = top + 1 + margin_top
    inner_left = left + 1 + margin_left
    if node.orient == "h":
        content_h = node.h - 2 - margin_top - margin_bottom
        cur_left = inner_left
        for child in node.children:
            slack = content_h - child.h
            child_top = inner_top + (0 if slack <= 0 else int(RNG.integers(0, slack + 1)))
            _set_bboxes(child, child_top, cur_left)
            cur_left += child.w + node.gap
    else:
        content_w = node.w - 2 - margin_left - margin_right
        cur_top = inner_top
        for child in node.children:
            slack = content_w - child.w
            child_left = inner_left + (0 if slack <= 0 else int(RNG.integers(0, slack + 1)))
            _set_bboxes(child, cur_top, child_left)
            cur_top += child.h + node.gap


def _legend_bbox(shape: tuple[int, int], depth: int, corner: str) -> tuple[int, int, int, int]:
    h, w = shape
    box_w = 4 * depth - 1
    if corner == "tl":
        return (0, 0, 2, box_w - 1)
    if corner == "tr":
        return (0, w - box_w, 2, w - 1)
    if corner == "bl":
        return (h - 3, 0, h - 1, box_w - 1)
    return (h - 3, w - box_w, h - 1, w - 1)


def _mark_rect(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(mask.shape[0] - 1, r1 + pad)
    c1 = min(mask.shape[1] - 1, c1 + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _rect_free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(mask.shape[0] - 1, r1 + pad)
    c1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _sample_anchor(limit: int, size: int) -> int:
    hi = limit - size
    if hi <= 1:
        return 0
    lo = 1
    hi -= 1
    if hi < lo:
        lo = 0
        hi = limit - size
    return int(RNG.integers(lo, hi + 1))


def _place_roots(roots: list[_Node], shape: tuple[int, int], legend_box: tuple[int, int, int, int]) -> bool:
    occ = np.zeros(shape, dtype=bool)
    _mark_rect(occ, legend_box, pad=1)

    for root in sorted(roots, key=lambda item: item.h * item.w, reverse=True):
        placed = False
        for _ in range(ROOT_PLACE_ATTEMPTS):
            top = _sample_anchor(shape[0], root.h)
            left = _sample_anchor(shape[1], root.w)
            bbox = (top, left, top + root.h - 1, left + root.w - 1)
            if not _rect_free(occ, bbox, pad=1):
                continue
            _set_bboxes(root, top, left)
            _mark_rect(occ, bbox, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _draw_input(shape: tuple[int, int], roots: list[_Node], patterns: list[np.ndarray], legend_box: tuple[int, int, int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for node in _collect_nodes(roots):
        r0, c0, r1, c1 = node.bbox
        grid[r0, c0 : c1 + 1] = node.color
        grid[r1, c0 : c1 + 1] = node.color
        grid[r0 : r1 + 1, c0] = node.color
        grid[r0 : r1 + 1, c1] = node.color

    lr0, lc0, _, _ = legend_box
    for i, pattern in enumerate(patterns):
        patch = grid[lr0 : lr0 + 3, lc0 + 4 * i : lc0 + 4 * i + 3]
        patch[pattern] = LEGEND_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    max_depth = int(RNG.integers(MAX_DEPTH_RANGE[0], MAX_DEPTH_RANGE[1] + 1))
    roots = _make_tree(max_depth)
    _assign_depths(roots)

    nodes = _collect_nodes(roots)
    if len(nodes) > len(FRAME_COLORS):
        return None
    colors = list(RNG.permutation(FRAME_COLORS))[: len(nodes)]
    for node, color in zip(nodes, colors):
        node.color = int(color)

    for root in roots:
        _assign_sizes(root)

    max_root_h = max(root.h for root in roots)
    max_root_w = max(root.w for root in roots)
    legend_w = 4 * max_depth - 1
    min_h = max(GRID_H_RANGE[0], max_root_h + 3)
    min_w = max(GRID_W_RANGE[0], max(max_root_w + 3, legend_w + 2))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = int(RNG.integers(min_h, GRID_H_RANGE[1] + 1))
    width = int(RNG.integers(min_w, GRID_W_RANGE[1] + 1))

    patterns = [PATTERN_LIBRARY[i].copy() for i in RNG.choice(len(PATTERN_LIBRARY), size=max_depth, replace=False)]
    corner = str(RNG.choice(["tl", "tr", "bl", "br"]))
    legend_box = _legend_bbox((height, width), max_depth, corner)
    if not _place_roots(roots, (height, width), legend_box):
        return None

    grid = _draw_input((height, width), roots, patterns, legend_box)
    expected = _render_scene((height, width), roots, patterns, BG)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
