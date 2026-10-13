from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
CANVAS_SIZE = 30
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
    if len(points) != expected:
        return None

    point_set = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _mask_child_bbox(
    inner_mask: np.ndarray,
    parent_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int],
) -> None:
    pr0, pc0, _, _ = parent_bbox
    cr0, cc0, cr1, cc1 = child_bbox
    rs = cr0 - (pr0 + 1)
    re = cr1 - pr0
    cs = cc0 - (pc0 + 1)
    ce = cc1 - pc0
    inner_mask[rs:re, cs:ce] = False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangular frames, propagate fill colors down the nesting tree,
    and paint each rectangle's visible interior while preserving frame colors.
    """
    g = np.asarray(grid, dtype=int)
    rects: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                rects.append({"bbox": bbox, "border": color})

    if not rects:
        return g.copy()

    n = len(rects)
    parents: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        bbox_i = rects[i]["bbox"]
        best_parent = None
        best_area = None
        for j in range(n):
            if i == j:
                continue
            bbox_j = rects[j]["bbox"]
            if _contains(bbox_j, bbox_i):
                area_j = _bbox_area(bbox_j)
                if best_parent is None or area_j < best_area:
                    best_parent = j
                    best_area = area_j
        parents[i] = best_parent
        if best_parent is not None:
            children[best_parent].append(i)

    fill_colors = [BG] * n

    def assign_colors(idx: int) -> None:
        parent = parents[idx]
        r0, c0, _, _ = rects[idx]["bbox"]
        if parent is None:
            fill_colors[idx] = int(g[r0 + 1, c0 + 1])
        else:
            fill_colors[idx] = int(rects[parent]["border"])
        for child_idx in children[idx]:
            assign_colors(child_idx)

    out = g.copy()

    def paint(idx: int) -> None:
        bbox = rects[idx]["bbox"]
        r0, c0, r1, c1 = bbox
        inner = np.ones((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
        for child_idx in children[idx]:
            _mask_child_bbox(inner, bbox, rects[child_idx]["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1][inner] = fill_colors[idx]
        for child_idx in children[idx]:
            paint(child_idx)

    for i in range(n):
        if parents[i] is None:
            assign_colors(i)
    for i in range(n):
        if parents[i] is None:
            paint(i)

    return out


@dataclass
class LayoutNode:
    children: list["LayoutNode"] = field(default_factory=list)
    orient: str | None = None
    h: int = 0
    w: int = 0
    bbox: tuple[int, int, int, int] | None = None
    border_color: int = 0
    seed_color: int | None = None


def _chain(depth: int) -> LayoutNode:
    root = LayoutNode()
    cur = root
    for _ in range(depth - 1):
        child = LayoutNode()
        cur.children = [child]
        cur = child
    return root


def _branch(with_grandchild: bool) -> LayoutNode:
    kids = [LayoutNode(), LayoutNode()]
    random.shuffle(kids)
    root = LayoutNode(children=kids, orient=random.choice(("h", "v")))
    if with_grandchild:
        target = random.choice(kids)
        target.children = [LayoutNode()]
    return root


def _sample_forest() -> list[LayoutNode]:
    profile = random.choice(("chain", "branch", "double", "mixed"))
    if profile == "chain":
        forest = [_chain(random.randint(2, 4))]
    elif profile == "branch":
        forest = [_branch(random.choice((False, True)))]
    elif profile == "double":
        forest = [_chain(random.randint(2, 3)), _chain(random.randint(1, 2))]
    else:
        forest = [_branch(False), _chain(random.randint(1, 2))]
        if random.random() < 0.5:
            forest.append(LayoutNode())
    random.shuffle(forest)
    return forest


def _walk(node: LayoutNode) -> list[LayoutNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(_walk(child))
    return nodes


def _finalize_sizes(node: LayoutNode) -> None:
    for child in node.children:
        _finalize_sizes(child)

    if not node.children:
        node.h = random.randint(3, 6)
        node.w = random.randint(3, 6)
        return

    if len(node.children) == 1:
        child = node.children[0]
        node.h = child.h + 4 + random.randint(0, 3)
        node.w = child.w + 4 + random.randint(0, 3)
        return

    a, b = node.children
    if node.orient == "h":
        node.h = max(a.h, b.h) + 4 + random.randint(0, 2)
        node.w = a.w + b.w + 5 + random.randint(0, 3)
    else:
        node.h = a.h + b.h + 5 + random.randint(0, 3)
        node.w = max(a.w, b.w) + 4 + random.randint(0, 2)


def _split_extra(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _place_node(node: LayoutNode, r0: int, c0: int) -> None:
    node.bbox = (r0, c0, r0 + node.h - 1, c0 + node.w - 1)
    if not node.children:
        return

    if len(node.children) == 1:
        child = node.children[0]
        cr0 = random.randint(r0 + 2, r0 + node.h - child.h - 2)
        cc0 = random.randint(c0 + 2, c0 + node.w - child.w - 2)
        _place_node(child, cr0, cc0)
        return

    a, b = node.children
    if node.orient == "h":
        extra = (node.w - 2) - (a.w + b.w + 3)
        left_pad, gap_pad, _ = _split_extra(extra, 3)
        ca0 = c0 + 2 + left_pad
        cb0 = ca0 + a.w + 1 + gap_pad
        ra0 = random.randint(r0 + 2, r0 + node.h - a.h - 2)
        rb0 = random.randint(r0 + 2, r0 + node.h - b.h - 2)
        _place_node(a, ra0, ca0)
        _place_node(b, rb0, cb0)
        return

    extra = (node.h - 2) - (a.h + b.h + 3)
    top_pad, gap_pad, _ = _split_extra(extra, 3)
    ra0 = r0 + 2 + top_pad
    rb0 = ra0 + a.h + 1 + gap_pad
    ca0 = random.randint(c0 + 2, c0 + node.w - a.w - 2)
    cb0 = random.randint(c0 + 2, c0 + node.w - b.w - 2)
    _place_node(a, ra0, ca0)
    _place_node(b, rb0, cb0)


def _place_roots(roots: list[LayoutNode]) -> bool:
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    for node in sorted(roots, key=lambda cur: cur.h * cur.w, reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, CANVAS_SIZE - node.h):
            for c0 in range(1, CANVAS_SIZE - node.w):
                rr0 = max(0, r0 - 1)
                rr1 = min(CANVAS_SIZE, r0 + node.h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(CANVAS_SIZE, c0 + node.w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        _place_node(node, r0, c0)
        rr0 = max(0, r0 - 1)
        rr1 = min(CANVAS_SIZE, r0 + node.h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(CANVAS_SIZE, c0 + node.w + 1)
        occupied[rr0:rr1, cc0:cc1] = True
    return True


def _assign_colors(roots: list[LayoutNode]) -> bool:
    nodes: list[LayoutNode] = []
    for root in roots:
        nodes.extend(_walk(root))
    if len(nodes) > 6:
        return False

    border_colors = random.sample(PALETTE, len(nodes))
    remaining = [color for color in PALETTE if color not in border_colors]
    if len(remaining) < len(roots):
        return False
    seed_colors = random.sample(remaining, len(roots))

    for node, color in zip(nodes, border_colors):
        node.border_color = int(color)
    for root, color in zip(roots, seed_colors):
        root.seed_color = int(color)
    return True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_expected(out: np.ndarray, node: LayoutNode, fill_color: int) -> None:
    assert node.bbox is not None
    r0, c0, r1, c1 = node.bbox
    inner = np.ones((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
    for child in node.children:
        assert child.bbox is not None
        _mask_child_bbox(inner, node.bbox, child.bbox)
    out[r0 + 1 : r1, c0 + 1 : c1][inner] = fill_color
    for child in node.children:
        _paint_expected(out, child, node.border_color)


def _render_pair(roots: list[LayoutNode]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for root in roots:
        for node in _walk(root):
            assert node.bbox is not None
            _draw_frame(inp, node.bbox, node.border_color)
        assert root.bbox is not None and root.seed_color is not None
        r0, c0, _, _ = root.bbox
        inp[r0 + 1, c0 + 1] = root.seed_color

    out = inp.copy()
    for root in roots:
        assert root.seed_color is not None
        _paint_expected(out, root, root.seed_color)
    return inp, out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(inp != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top = random.randint(0, min(3, int(r0)))
    left = random.randint(0, min(3, int(c0)))
    bottom = random.randint(0, min(3, int(inp.shape[0] - 1 - r1)))
    right = random.randint(0, min(3, int(inp.shape[1] - 1 - c1)))
    rs = int(r0 - top)
    re = int(r1 + bottom + 1)
    cs = int(c0 - left)
    ce = int(c1 + right + 1)
    return inp[rs:re, cs:ce], out[rs:re, cs:ce]


def _gen_attempt() -> np.ndarray | None:
    roots = _sample_forest()
    for root in roots:
        _finalize_sizes(root)
    if not _place_roots(roots):
        return None
    if not _assign_colors(roots):
        return None

    inp, expected = _render_pair(roots)
    inp, expected = _crop_pair(inp, expected)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle input")
