from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

MIN_H = 14
MAX_H = 24
MIN_W = 18
MAX_W = 32
MAX_TOTAL_RECTS = 8
MAX_DEPTH = 3  # depth 0..3 -> at most 4 nested levels
MAX_GENERATION_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    if np.any(vals == 0):
        return 0
    return int(vals[counts.argmax()])


def _rect_sort_key(node: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = node["bbox"]
    return int(c0), int(r0), int(c1 - c0), int(node["color"])


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    rects: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(g == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "children": [],
            }
        )
    return rects


def _contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _build_forest(rects: list[dict]) -> list[dict]:
    by_color = {int(node["color"]): node for node in rects}
    has_parent: set[int] = set()
    for node in rects:
        node["children"] = []

    for node in rects:
        parent = None
        candidates = [cand for cand in rects if cand is not node and _contains(cand, node)]
        if candidates:
            parent = min(
                candidates,
                key=lambda cand: (
                    int(cand["area"]),
                    int(cand["bbox"][0]),
                    int(cand["bbox"][1]),
                    int(cand["color"]),
                ),
            )
        if parent is not None:
            by_color[int(parent["color"])]["children"].append(node)
            has_parent.add(int(node["color"]))

    roots = [node for node in rects if int(node["color"]) not in has_parent]
    for node in rects:
        node["children"].sort(key=_rect_sort_key)
    roots.sort(key=_rect_sort_key)
    return roots


def _preorder(nodes: list[dict]) -> list[tuple[dict, int]]:
    order: list[tuple[dict, int]] = []

    def visit(node: dict, depth: int) -> None:
        order.append((node, depth))
        for child in node["children"]:
            visit(child, depth + 1)

    for root in nodes:
        visit(root, 0)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract nested rectangles, linearize them by preorder, and encode depth by vertical offset.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _background(g)
    rects = _extract_rectangles(g)
    if not rects:
        return np.full((h, 0), bg, dtype=int)

    roots = _build_forest(rects)
    order = _preorder(roots)
    out_w = sum(int(node["w"]) for node, _depth in order) + max(0, len(order) - 1)
    out = np.full((h, out_w), bg, dtype=int)

    col = 0
    for node, depth in order:
        top = 2 * depth
        height = int(node["h"])
        width = int(node["w"])
        assert top + height <= h, "depth-coded placement must fit output height"
        out[top : top + height, col : col + width] = int(node["color"])
        col += width + 1

    return out


def _distribute(total: int, parts: int, minima: list[int], rng: random.Random) -> list[int]:
    assert parts == len(minima)
    assert total >= sum(minima)
    vals = list(minima)
    extra = total - sum(minima)
    for _ in range(extra):
        vals[rng.randrange(parts)] += 1
    return vals


def _min_dim_for_budget(node_budget: int, max_depth: int) -> int:
    layers = min(node_budget, max_depth + 1)
    return max(4, 2 * layers)


def _choose_mode(node_budget: int, rng: random.Random) -> str:
    if node_budget <= 2:
        return rng.choice(["chain", "mixed"])
    return rng.choice(["chain", "branch", "mixed"])


def _choose_child_count(
    max_children: int,
    depth: int,
    max_depth: int,
    mode: str,
    rng: random.Random,
) -> int:
    counts = list(range(max_children + 1))
    weight_table = {
        "chain": [1, 8, 2, 1],
        "branch": [1, 2, 7, 5],
        "mixed": [2, 5, 4, 2],
    }
    weights = [weight_table[mode][count] for count in counts]
    if depth >= max_depth:
        return 0
    if depth == max_depth - 1:
        weights = [max(1, w) for w in weights]
    return rng.choices(counts, weights=weights, k=1)[0]


def _build_subtree(
    r0: int,
    c0: int,
    height: int,
    width: int,
    depth: int,
    max_depth: int,
    node_budget: int,
    mode: str,
    rng: random.Random,
) -> dict:
    node = {
        "color": None,
        "bbox": (r0, c0, r0 + height - 1, c0 + width - 1),
        "h": height,
        "w": width,
        "children": [],
    }
    if depth >= max_depth or node_budget <= 1:
        return node

    inner_h = height - 2
    inner_w = width - 2
    if inner_h < 2 or inner_w < 2:
        return node

    max_children_space = min(3, (inner_w + 1) // 3)
    max_children = min(max_children_space, node_budget - 1)
    if max_children <= 0:
        return node

    child_count = _choose_child_count(max_children, depth, max_depth, mode, rng)
    if child_count == 0:
        return node

    child_budgets = _distribute(node_budget - 1, child_count, [1] * child_count, rng)
    minima: list[int] = [0]
    for i in range(child_count):
        minima.append(2)
        minima.append(0 if i == child_count - 1 else 1)
    slots = _distribute(inner_w, len(minima), minima, rng)

    cursor = c0 + 1 + slots[0]
    for i in range(child_count):
        child_w = slots[2 * i + 1]
        child_budget = child_budgets[i]

        min_child_h = 2
        if child_budget > 1 and inner_h >= 4 and rng.random() < 0.75:
            min_child_h = 4
        child_h = rng.randint(min_child_h, inner_h)

        start_min = r0 + 1
        start_max = r0 + height - child_h - 1
        child_r0 = rng.randint(start_min, start_max)

        child = _build_subtree(
            child_r0,
            cursor,
            child_h,
            child_w,
            depth + 1,
            max_depth,
            child_budget,
            mode,
            rng,
        )
        node["children"].append(child)
        cursor += child_w + slots[2 * i + 2]

    node["children"].sort(key=lambda child: (child["bbox"][1], child["bbox"][0]))
    return node


def _iter_nodes(node: dict) -> list[dict]:
    out = [node]
    for child in node["children"]:
        out.extend(_iter_nodes(child))
    return out


def _iter_forest(roots: list[dict]) -> list[dict]:
    out: list[dict] = []
    for root in roots:
        out.extend(_iter_nodes(root))
    return out


def _max_actual_depth(node: dict) -> int:
    if not node["children"]:
        return 0
    return 1 + max(_max_actual_depth(child) for child in node["children"])


def _render_tree(canvas: np.ndarray, node: dict) -> None:
    r0, c0, r1, c1 = node["bbox"]
    canvas[r0 : r1 + 1, c0 : c1 + 1] = int(node["color"])
    for child in node["children"]:
        _render_tree(canvas, child)


def _expected_output(roots: list[dict], height: int) -> np.ndarray:
    order = _preorder(roots)
    out_w = sum(int(node["w"]) for node, _depth in order) + max(0, len(order) - 1)
    out = np.full((height, out_w), BG, dtype=int)
    col = 0
    for node, depth in order:
        r0 = 2 * depth
        r1 = r0 + int(node["h"])
        c1 = col + int(node["w"])
        out[r0:r1, col:c1] = int(node["color"])
        col = c1 + 1
    return out


def _structure_signature(roots: list[dict]) -> tuple[tuple[int, int, tuple[int, int, int, int]], ...]:
    sig: list[tuple[int, int, tuple[int, int, int, int]]] = []

    def visit(node: dict, depth: int) -> None:
        sig.append((int(node["color"]), depth, tuple(int(x) for x in node["bbox"])))
        for child in node["children"]:
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return tuple(sig)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_depth = rng.choice([1, 2, 3])
    root_count = rng.choice([1, 2, 3])
    total_nodes = rng.randint(root_count + 1, min(MAX_TOTAL_RECTS, root_count + 5))
    root_budgets = _distribute(total_nodes, root_count, [1] * root_count, rng)

    min_h = max(MIN_H, 2 * (max_depth + 1) + 4)
    min_root_widths = [_min_dim_for_budget(budget, max_depth) for budget in root_budgets]
    min_w = max(MIN_W, sum(min_root_widths) + 2 * (root_count - 1))
    if min_h > MAX_H or min_w > MAX_W:
        return None

    height = rng.randint(min_h, MAX_H)
    width = rng.randint(min_w, MAX_W)

    minima: list[int] = [0]
    for i, min_root_w in enumerate(min_root_widths):
        minima.append(min_root_w)
        minima.append(0 if i == root_count - 1 else 2)
    slots = _distribute(width, len(minima), minima, rng)

    roots: list[dict] = []
    cursor = slots[0]
    for i, budget in enumerate(root_budgets):
        root_w = slots[2 * i + 1]
        min_root_h = _min_dim_for_budget(budget, max_depth)
        root_h = rng.randint(min_root_h, min(height, min_root_h + 6))
        root_r0 = rng.randint(0, height - root_h)
        mode = _choose_mode(budget, rng)
        roots.append(
            _build_subtree(
                root_r0,
                cursor,
                root_h,
                root_w,
                0,
                max_depth,
                budget,
                mode,
                rng,
            )
        )
        cursor += root_w + slots[2 * i + 2]

    roots.sort(key=lambda node: (node["bbox"][1], node["bbox"][0]))
    nodes = _iter_forest(roots)
    actual_count = len(nodes)
    if actual_count < root_count + 1 or actual_count > len(PALETTE):
        return None
    if max(_max_actual_depth(root) for root in roots) < 1:
        return None

    colors = rng.sample(PALETTE, actual_count)
    for node, color in zip(nodes, colors):
        node["color"] = int(color)

    inp = np.full((height, width), BG, dtype=int)
    for root in roots:
        _render_tree(inp, root)

    extracted = _build_forest(_extract_rectangles(inp))
    if _structure_signature(roots) != _structure_signature(extracted):
        return None

    expected = _expected_output(roots, height)
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate a valid instance after {MAX_GENERATION_ATTEMPTS} attempts")
