from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_DEPTH = 4
MAX_CHILDREN = 2
MAX_TOTAL_NODES = 7
MAX_GENERATION_ATTEMPTS = 400
MIN_LEAF_SIDE = 3
MAX_LEAF_SIDE = 6
MIN_GAP = 1
MAX_GAP = 2
MAX_SLACK = 2
ROOT_PAD = 1


@dataclass
class TreeNode:
    color: int = -1
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class PlacedNode:
    color: int
    bbox: tuple[int, int, int, int]
    children: list["PlacedNode"] = field(default_factory=list)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _shift_tree(node: PlacedNode, dr: int, dc: int) -> PlacedNode:
    r0, c0, r1, c1 = node.bbox
    return PlacedNode(
        color=node.color,
        bbox=(r0 + dr, c0 + dc, r1 + dr, c1 + dc),
        children=[_shift_tree(child, dr, dc) for child in node.children],
    )


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BLACK
    sub[mask] = patch[mask]


def _draw_ring(patch: np.ndarray, color: int) -> None:
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color


def _connected_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _ring_records(grid: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for comp in _connected_components(grid):
        rows = [r for r, _ in comp["coords"]]
        cols = [c for _, c in comp["coords"]]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "expected each component to be a hollow rectangle"
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in comp["coords"]:
            mask[rr - r0, cc - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(mask, border), "expected each component to be exactly a rectangular ring"
        records.append(
            {
                "color": int(comp["color"]),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(h * w),
            }
        )
    assert records, "expected at least one ring"
    return records


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every even-depth rectangular ring to its direct parent's color.

    Input structure:
    - The background is black.
    - Non-black cells form one-pixel-thick hollow rectangular rings, each as its own 4-connected monochrome component.
    - Rings may be arranged as several disjoint outer roots, and smaller rings can be fully nested inside larger rings without crossing.

    Rule:
    - Recover the containment forest by finding, for each ring, the smallest larger ring whose rectangle strictly contains it.
    - Depth 1 rings are roots; contained rings are one level deeper than their direct parent.
    - Rings at odd depths keep their original color, while rings at even depths are recolored to their direct parent's original color.

    Key insight:
    The deciding feature is not local shape or color but global nesting depth inferred from rectangle containment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings = _ring_records(g)
    parents: list[int | None] = [None] * len(rings)
    for i, ring in enumerate(rings):
        candidates = [
            j for j, other in enumerate(rings)
            if i != j and _contains_bbox(other["bbox"], ring["bbox"])
        ]
        if candidates:
            parents[i] = min(candidates, key=lambda j: rings[j]["area"])

    depths: dict[int, int] = {}

    def _depth(idx: int) -> int:
        if idx in depths:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 1 if parent is None else _depth(parent) + 1
        return depths[idx]

    out = g.copy()
    for i, ring in enumerate(rings):
        d = _depth(i)
        if d % 2 == 1:
            continue
        parent = parents[i]
        assert parent is not None
        new_color = int(rings[parent]["color"])
        for r, c in ring["coords"]:
            out[r, c] = new_color
    return out


def _iter_nodes(roots: list[TreeNode], depth: int = 1):
    for node in roots:
        yield node, depth
        yield from _iter_nodes(node.children, depth + 1)


def _node_count(roots: list[TreeNode]) -> int:
    return sum(1 for _node, _depth in _iter_nodes(roots))


def _build_shape_forest() -> list[TreeNode] | None:
    max_depth = random.randint(1, MAX_DEPTH)
    max_root_count = min(3, MAX_TOTAL_NODES - max_depth + 1)
    root_count = random.randint(1, max_root_count)

    if max_depth == 1:
        total_nodes = root_count
    else:
        min_nodes = root_count + max_depth - 1
        total_nodes = random.randint(min_nodes, MAX_TOTAL_NODES)

    roots = [TreeNode() for _ in range(root_count)]
    chain_root = random.choice(roots)
    current = chain_root
    for _ in range(max_depth - 1):
        child = TreeNode()
        current.children.append(child)
        current = child

    extra = total_nodes - _node_count(roots)
    while extra > 0:
        candidates = [
            node for node, depth in _iter_nodes(roots)
            if depth < max_depth and len(node.children) < MAX_CHILDREN
        ]
        if not candidates:
            return None
        parent = random.choice(candidates)
        parent.children.append(TreeNode())
        extra -= 1

    colors = random.sample(NON_BLACK_COLORS, _node_count(roots))
    color_iter = iter(colors)
    for node, _depth in _iter_nodes(roots):
        node.color = next(color_iter)
        random.shuffle(node.children)
    return roots


def _render_tree(node: TreeNode) -> tuple[np.ndarray, PlacedNode]:
    child_results = [_render_tree(child) for child in node.children]
    if not child_results:
        h = random.randint(MIN_LEAF_SIDE, MAX_LEAF_SIDE)
        w = random.randint(MIN_LEAF_SIDE, MAX_LEAF_SIDE)
        patch = np.zeros((h, w), dtype=int)
        _draw_ring(patch, node.color)
        return patch, PlacedNode(node.color, (0, 0, h - 1, w - 1), [])

    if len(child_results) == 1:
        child_patch, child_node = child_results[0]
        top = random.randint(MIN_GAP, MAX_GAP)
        bottom = random.randint(MIN_GAP, MAX_GAP)
        left = random.randint(MIN_GAP, MAX_GAP)
        right = random.randint(MIN_GAP, MAX_GAP)
        h = child_patch.shape[0] + top + bottom + 2
        w = child_patch.shape[1] + left + right + 2
        patch = np.zeros((h, w), dtype=int)
        _draw_ring(patch, node.color)
        off_r, off_c = 1 + top, 1 + left
        _paste_nonzero(patch, child_patch, off_r, off_c)
        return patch, PlacedNode(
            node.color,
            (0, 0, h - 1, w - 1),
            [_shift_tree(child_node, off_r, off_c)],
        )

    random.shuffle(child_results)
    (patch_a, node_a), (patch_b, node_b) = child_results[:2]
    top = random.randint(MIN_GAP, MAX_GAP)
    bottom = random.randint(MIN_GAP, MAX_GAP)
    left = random.randint(MIN_GAP, MAX_GAP)
    right = random.randint(MIN_GAP, MAX_GAP)
    sep = random.randint(MIN_GAP, MAX_GAP)
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        extra = random.randint(0, MAX_SLACK)
        h = 2 + top + max(patch_a.shape[0], patch_b.shape[0]) + bottom + extra
        w = 2 + left + patch_a.shape[1] + sep + patch_b.shape[1] + right
        patch = np.zeros((h, w), dtype=int)
        _draw_ring(patch, node.color)
        x_a = 1 + left
        x_b = x_a + patch_a.shape[1] + sep
        y_min = 1 + top
        y_max_a = h - 1 - bottom - patch_a.shape[0]
        y_max_b = h - 1 - bottom - patch_b.shape[0]
        y_a = random.randint(y_min, y_max_a)
        y_b = random.randint(y_min, y_max_b)
        _paste_nonzero(patch, patch_a, y_a, x_a)
        _paste_nonzero(patch, patch_b, y_b, x_b)
        children = [_shift_tree(node_a, y_a, x_a), _shift_tree(node_b, y_b, x_b)]
    else:
        extra = random.randint(0, MAX_SLACK)
        h = 2 + top + patch_a.shape[0] + sep + patch_b.shape[0] + bottom
        w = 2 + left + max(patch_a.shape[1], patch_b.shape[1]) + right + extra
        patch = np.zeros((h, w), dtype=int)
        _draw_ring(patch, node.color)
        y_a = 1 + top
        y_b = y_a + patch_a.shape[0] + sep
        x_min = 1 + left
        x_max_a = w - 1 - right - patch_a.shape[1]
        x_max_b = w - 1 - right - patch_b.shape[1]
        x_a = random.randint(x_min, x_max_a)
        x_b = random.randint(x_min, x_max_b)
        _paste_nonzero(patch, patch_a, y_a, x_a)
        _paste_nonzero(patch, patch_b, y_b, x_b)
        children = [_shift_tree(node_a, y_a, x_a), _shift_tree(node_b, y_b, x_b)]

    return patch, PlacedNode(node.color, (0, 0, h - 1, w - 1), children)


def _bbox_clear(bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], pad: int) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _place_roots(rendered: list[tuple[np.ndarray, PlacedNode]]) -> tuple[np.ndarray, list[PlacedNode]] | None:
    max_h = max(patch.shape[0] for patch, _node in rendered)
    max_w = max(patch.shape[1] for patch, _node in rendered)
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch, _node in rendered)
    dim_options = [
        (h, w)
        for h in range(max_h, 31)
        for w in range(max_w, 31)
        if h * w >= total_area + 4 * (len(rendered) - 1)
    ]
    random.shuffle(dim_options)

    order = sorted(range(len(rendered)), key=lambda i: rendered[i][0].shape[0] * rendered[i][0].shape[1], reverse=True)
    for h, w in dim_options[:120]:
        grid = np.zeros((h, w), dtype=int)
        placed_bboxes: list[tuple[int, int, int, int]] = []
        placed_nodes: list[PlacedNode] = [None] * len(rendered)  # type: ignore[list-item]
        ok = True
        for idx in order:
            patch, node = rendered[idx]
            ph, pw = map(int, patch.shape)
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(positions)
            chosen: tuple[int, int] | None = None
            for r0, c0 in positions:
                bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _bbox_clear(bbox, placed_bboxes, ROOT_PAD):
                    chosen = (r0, c0)
                    placed_bboxes.append(bbox)
                    break
            if chosen is None:
                ok = False
                break
            r0, c0 = chosen
            _paste_nonzero(grid, patch, r0, c0)
            placed_nodes[idx] = _shift_tree(node, r0, c0)
        if ok:
            return grid, placed_nodes
    return None


def _paint_expected(input_grid: np.ndarray, roots: list[PlacedNode]) -> np.ndarray:
    out = input_grid.copy()

    def visit(node: PlacedNode, depth: int, parent_color: int | None) -> None:
        color = node.color if depth % 2 == 1 else int(parent_color)
        r0, c0, r1, c1 = node.bbox
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        for child in node.children:
            visit(child, depth + 1, node.color)

    for root in roots:
        visit(root, 1, None)
    return out


def _gen_attempt() -> np.ndarray | None:
    roots = _build_shape_forest()
    if roots is None:
        return None

    rendered = [_render_tree(root) for root in roots]
    placement = _place_roots(rendered)
    if placement is None:
        return None
    input_grid, placed_roots = placement
    expected = _paint_expected(input_grid, placed_roots)

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
