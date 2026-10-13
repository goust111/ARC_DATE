from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 300


@dataclass
class _Rect:
    bbox: tuple[int, int, int, int]
    color: int
    parent: int | None = None
    depth: int = 0


@dataclass
class _Node:
    children: list["_Node"] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)
    child_offsets: list[tuple[int, int]] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    depth: int = 0
    color: int = 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            parts.append(np.asarray(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _read_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    assert int(grid[h - 1, w - 1]) != BG, "bottom-right legend pixel must be non-black"
    start = h - 1
    while start > 0 and int(grid[start - 1, w - 1]) != BG:
        start -= 1
    colors = [int(grid[r, w - 1]) for r in range(start, h)]
    mask = np.zeros_like(grid, dtype=bool)
    mask[start:h, w - 1] = True
    return colors, mask


def _component_to_rect(grid: np.ndarray, coords: np.ndarray) -> _Rect | None:
    colors = np.unique(grid[coords[:, 0], coords[:, 1]])
    if colors.size != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return None

    observed = np.zeros((hh, ww), dtype=bool)
    observed[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros((hh, ww), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True

    if not np.array_equal(observed, expected):
        return None
    return _Rect(bbox=(r0, c0, r1, c1), color=int(colors[0]))


def _contains(outer: _Rect, inner: _Rect) -> bool:
    or0, oc0, or1, oc1 = outer.bbox
    ir0, ic0, ir1, ic1 = inner.bbox
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _extract_rectangles(grid: np.ndarray, legend_mask: np.ndarray) -> list[_Rect]:
    comps = _components((grid != BG) & ~legend_mask)
    rects: list[_Rect] = []
    for comp in comps:
        rect = _component_to_rect(grid, comp)
        assert rect is not None, "all non-legend foreground components must be 1-pixel rectangle borders"
        rects.append(rect)
    assert rects, "expected at least one rectangle border"

    for idx, rect in enumerate(rects):
        containers = [j for j, other in enumerate(rects) if j != idx and _contains(other, rect)]
        if containers:
            rect.parent = min(containers, key=lambda j: _area(rects[j].bbox))

    memo: dict[int, int] = {}

    def depth_of(i: int) -> int:
        if i in memo:
            return memo[i]
        parent = rects[i].parent
        memo[i] = 0 if parent is None else depth_of(parent) + 1
        return memo[i]

    for i, rect in enumerate(rects):
        rect.depth = depth_of(i)
    return rects


def _apply_fill_rule(grid: np.ndarray, rects: list[_Rect], legend_colors: list[int]) -> np.ndarray:
    out = grid.copy()
    original_black = grid == BG
    for rect in sorted(rects, key=lambda item: (item.depth, -_area(item.bbox))):
        assert rect.depth < len(legend_colors), "legend must provide a fill color for every depth"
        r0, c0, r1, c1 = rect.bbox
        if r1 - r0 <= 1 or c1 - c0 <= 1:
            continue
        fill = legend_colors[rect.depth]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        mask = original_black[r0 + 1 : r1, c0 + 1 : c1]
        patch[mask] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested hollow rectangles according to a depth-color legend in the lower-right corner.

    Input structure:
    - The background is black.
    - The grid contains several one-pixel-thick hollow rectangular borders, each drawn in a single color.
    - Rectangles are either disjoint or strictly nested, forming a containment tree.
    - The lower-right corner contains a vertical legend: a contiguous column of colored pixels on the last column, read top-to-bottom as depth 0, depth 1, depth 2, and so on.

    Rule:
    - Detect every hollow rectangle border and recover the nesting tree from strict geometric containment.
    - Read the lower-right legend as a depth-to-fill-color lookup table.
    - For each rectangle, fill black pixels inside its border with the legend color for that rectangle's depth, processing from outer depth to inner depth so deeper rectangles overwrite shallower fills where they overlap.
    - Keep all rectangle borders and legend pixels unchanged.

    Key insight:
    - The border colors identify the rectangles, but only the containment depth and the lower-right legend together determine each interior fill color.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _read_legend(g)
    rects = _extract_rectangles(g, legend_mask)
    return _apply_fill_rule(g, rects, legend_colors)


def _make_chain(length: int) -> _Node:
    root = _Node()
    cur = root
    for _ in range(length - 1):
        child = _Node()
        cur.children = [child]
        cur = child
    return root


def _build_tree(rng: random.Random) -> list[_Node]:
    mode = rng.choice(("chain", "branch", "two_roots", "mixed"))

    if mode == "chain":
        return [_make_chain(rng.randint(2, 4))]

    if mode == "branch":
        root = _Node(children=[_Node() for _ in range(rng.randint(2, 3))])
        if rng.random() < 0.75:
            target = rng.choice(root.children)
            target.children = [_Node()]
            if rng.random() < 0.35:
                target.children[0].children = [_Node()]
        return [root]

    if mode == "two_roots":
        root_a = _make_chain(rng.randint(2, 3))
        root_b = _Node()
        if rng.random() < 0.55:
            root_b.children = [_Node()]
        return [root_a, root_b]

    root_a = _Node(children=[_Node(), _Node()])
    if rng.random() < 0.6:
        rng.choice(root_a.children).children = [_Node()]
    root_b = _make_chain(rng.randint(2, 3))
    return [root_a, root_b]


def _collect_nodes(roots: list[_Node]) -> list[_Node]:
    nodes: list[_Node] = []

    def walk(node: _Node, depth: int) -> None:
        node.depth = depth
        nodes.append(node)
        for child in node.children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return nodes


def _arrange_boxes(sizes: list[tuple[int, int]], rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = len(sizes)
    assert n >= 1
    if n == 1:
        return [(0, 0)], sizes[0]

    indices = list(range(n))
    rng.shuffle(indices)
    candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []

    def build_row(order: list[int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
        gaps = [rng.randint(2, 3) for _ in range(len(order) - 1)]
        extra = rng.randint(0, 2)
        height = max(sizes[i][0] for i in order) + extra
        pos = [(0, 0)] * n
        x = 0
        for k, idx in enumerate(order):
            h_i, w_i = sizes[idx]
            y = rng.randint(0, height - h_i)
            pos[idx] = (y, x)
            x += w_i
            if k + 1 < len(order):
                x += gaps[k]
        return pos, (height, x)

    def build_col(order: list[int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
        gaps = [rng.randint(2, 3) for _ in range(len(order) - 1)]
        extra = rng.randint(0, 2)
        width = max(sizes[i][1] for i in order) + extra
        pos = [(0, 0)] * n
        y = 0
        for k, idx in enumerate(order):
            h_i, w_i = sizes[idx]
            x = rng.randint(0, width - w_i)
            pos[idx] = (y, x)
            y += h_i
            if k + 1 < len(order):
                y += gaps[k]
        return pos, (y, width)

    candidates.append(build_row(indices))
    candidates.append(build_col(indices))

    if n == 3:
        for top_count in (1, 2):
            top = indices[:top_count]
            bottom = indices[top_count:]
            top_pos, (top_h, top_w) = build_row(top)
            bottom_pos, (bottom_h, bottom_w) = build_row(bottom)
            width = max(top_w, bottom_w) + rng.randint(0, 2)
            gap = rng.randint(2, 3)
            pos = [(0, 0)] * n
            top_shift = rng.randint(0, width - top_w)
            bottom_shift = rng.randint(0, width - bottom_w)
            for idx in top:
                tr, tc = top_pos[idx]
                pos[idx] = (tr, tc + top_shift)
            for idx in bottom:
                br, bc = bottom_pos[idx]
                pos[idx] = (top_h + gap + br, bc + bottom_shift)
            candidates.append((pos, (top_h + gap + bottom_h, width)))

    best = min(candidates, key=lambda item: (max(item[1]), item[1][0] * item[1][1]))
    if rng.random() < 0.5:
        return best
    return rng.choice(candidates)


def _layout_subtree(node: _Node, rng: random.Random) -> None:
    for child in node.children:
        _layout_subtree(child, rng)

    if not node.children:
        node.size = (rng.randint(4, 6), rng.randint(4, 6))
        node.child_offsets = []
        return

    sizes = [child.size for child in node.children]
    child_offsets, (content_h, content_w) = _arrange_boxes(sizes, rng)
    top = rng.randint(2, 3)
    bottom = rng.randint(2, 3)
    left = rng.randint(2, 3)
    right = rng.randint(2, 3)
    node.child_offsets = [(top + dr, left + dc) for dr, dc in child_offsets]
    node.size = (top + content_h + bottom, left + content_w + right)


def _assign_bboxes(node: _Node, top: int, left: int) -> None:
    h, w = node.size
    node.bbox = (top, left, top + h - 1, left + w - 1)
    for child, (dr, dc) in zip(node.children, node.child_offsets):
        _assign_bboxes(child, top + dr, left + dc)


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_expected(grid: np.ndarray, roots: list[_Node], legend_colors: list[int]) -> np.ndarray:
    rects = [_Rect(bbox=node.bbox, color=node.color, depth=node.depth) for node in _collect_nodes(roots)]  # type: ignore[arg-type]
    return _apply_fill_rule(grid, rects, legend_colors)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    roots = _build_tree(rng)
    nodes = _collect_nodes(roots)

    for node in roots:
        _layout_subtree(node, rng)

    root_sizes = [root.size for root in roots]
    root_offsets, (content_h, content_w) = _arrange_boxes(root_sizes, rng)

    max_depth = max(node.depth for node in nodes)
    legend_colors = rng.sample(COLORS, max_depth + 1)

    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    right_gap = rng.randint(2, 4)

    height = top + content_h + bottom
    width = left + content_w + right_gap + 1
    if height < len(legend_colors) + 1:
        height = len(legend_colors) + 1
    if height > 30 or width > 30:
        return None

    border_colors = rng.sample(COLORS, len(nodes))
    for node, color in zip(nodes, border_colors):
        node.color = color

    grid = np.zeros((height, width), dtype=int)

    for root, (dr, dc) in zip(roots, root_offsets):
        _assign_bboxes(root, top + dr, left + dc)

    for node in nodes:
        assert node.bbox is not None
        _draw_border(grid, node.bbox, node.color)

    legend_start = height - len(legend_colors)
    for idx, color in enumerate(legend_colors):
        grid[legend_start + idx, width - 1] = color

    expected = _render_expected(grid, roots, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
