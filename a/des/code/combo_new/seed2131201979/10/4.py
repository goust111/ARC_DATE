from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
RNG = np.random.default_rng()

MARGIN = 1
GAP = 1
NODE_COUNT_RANGE = (3, 9)
LEAF_H_RANGE = (2, 5)
LEAF_W_RANGE = (2, 5)
PADDING_RANGE = (1, 4)
ROOT_EXTRA_LIMIT = 6
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return bool(or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1)


def _draw_square_frame(grid: np.ndarray, inset: int, color: int) -> None:
    last = int(grid.shape[0] - 1 - inset)
    grid[inset, inset : last + 1] = color
    grid[last, inset : last + 1] = color
    grid[inset : last + 1, inset] = color
    grid[inset : last + 1, last] = color


def _find_root_marker(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    h, w = map(int, grid.shape)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    markers = [(r, c) for r, c in corners if int(grid[r, c]) != BG]
    assert len(markers) == 1, "expected exactly one colored corner marker"
    marker = markers[0]
    return int(grid[marker]), marker


def _extract_rectangles(
    grid: np.ndarray, marker: tuple[int, int]
) -> dict[int, dict[str, Any]]:
    work = np.array(grid, dtype=int, copy=True)
    work[marker] = BG

    rectangles: dict[int, dict[str, Any]] = {}
    for color in map(int, np.unique(work)):
        if color == BG:
            continue
        coords = np.argwhere(work == color)
        assert len(coords) > 0, "rectangle color must remain after removing marker"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = work[r0 : r1 + 1, c0 : c1 + 1]
        assert bool((patch[0, :] == color).all())
        assert bool((patch[-1, :] == color).all())
        assert bool((patch[:, 0] == color).all())
        assert bool((patch[:, -1] == color).all())
        rectangles[color] = {
            "bbox": (r0, c0, r1, c1),
            "area": _bbox_area((r0, c0, r1, c1)),
        }

    assert rectangles, "expected at least one rectangle"
    return rectangles


def _tree_from_rectangles(
    rectangles: dict[int, dict[str, Any]], root_color: int
) -> dict[int, list[int]]:
    parent: dict[int, int | None] = {}
    colors = list(rectangles)
    for color in colors:
        bbox = rectangles[color]["bbox"]
        candidates = [
            other
            for other in colors
            if other != color
            and _strictly_contains(rectangles[other]["bbox"], bbox)
        ]
        if not candidates:
            parent[color] = None
            continue
        parent[color] = min(candidates, key=lambda other: rectangles[other]["area"])

    assert parent.get(root_color) is None, "marked rectangle must be the unique root"

    children: dict[int, list[int]] = defaultdict(list)
    roots = [color for color, par in parent.items() if par is None]
    assert roots == [root_color], "all rectangles must belong to the marked root"
    for color, par in parent.items():
        if par is not None:
            children[par].append(color)

    for color, kids in children.items():
        kids.sort(
            key=lambda kid: (
                -int(rectangles[kid]["area"]),
                int(rectangles[kid]["bbox"][0]),
                int(rectangles[kid]["bbox"][1]),
            )
        )
    return children


def _dfs_colors(root: int, children: dict[int, list[int]]) -> list[int]:
    order = [root]
    for child in children.get(root, []):
        order.extend(_dfs_colors(child, children))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the rectangle containment tree and render its depth-first order as concentric square frames.

    Input structure:
    - Black background.
    - Colored axis-aligned rectangles, one per color, possibly nested but never partially overlapping.
    - One grid corner contains a singleton marker pixel whose color matches the root rectangle.

    Rule:
    - Remove the corner marker from consideration, recover each rectangle's bounding box, and connect rectangles by direct strict containment.
    - The marked rectangle is the root; siblings are ordered by rectangle area from large to small.
    - Traverse the tree depth first in that sibling order.
    - Output a square of side 2N-1, where N is the number of rectangles, and draw one-pixel concentric square frames from outermost to innermost using the traversal colors.

    Key insight:
    Rectangle colors encode a hidden laminar family: once bounding boxes reveal the containment tree, the output is just a depth-first linearization rendered as nested frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    root_color, marker = _find_root_marker(g)
    rectangles = _extract_rectangles(g, marker)
    assert root_color in rectangles, "marker color must match a rectangle"

    children = _tree_from_rectangles(rectangles, root_color)
    order = _dfs_colors(root_color, children)
    assert len(order) == len(rectangles), "tree traversal must cover every rectangle"

    side = 2 * len(order) - 1
    out = np.full((side, side), BG, dtype=int)
    for inset, color in enumerate(order):
        _draw_square_frame(out, inset, int(color))
    return out


def _randint_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    values: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _multinomial_split(total: int, buckets: int, rng: np.random.Generator) -> list[int]:
    if total <= 0:
        return [0] * buckets
    return [int(x) for x in rng.multinomial(int(total), np.full(buckets, 1.0 / buckets))]


def _build_tree(total_nodes: int, rng: np.random.Generator) -> dict[str, Any]:
    node: dict[str, Any] = {"children": []}
    if total_nodes == 1:
        return node

    max_children = min(3, total_nodes - 1)
    if total_nodes == 2:
        child_count = 1
    else:
        choices = np.arange(1, max_children + 1)
        if max_children == 1:
            child_count = 1
        else:
            weights = np.ones(len(choices), dtype=float)
            if total_nodes >= 5:
                weights[0] = 1.2
                if len(weights) > 1:
                    weights[1] = 1.5
                if len(weights) > 2:
                    weights[2] = 1.3
            child_count = int(rng.choice(choices, p=weights / weights.sum()))

    parts = _random_composition(total_nodes - 1, int(child_count), rng)
    node["layout"] = "h" if bool(rng.integers(0, 2)) else "v"
    node["children"] = [_build_tree(part, rng) for part in parts]
    return node


def _annotate_mins(node: dict[str, Any], rng: np.random.Generator) -> tuple[int, int]:
    children = node["children"]
    if not children:
        min_h = _randint_inclusive(rng, LEAF_H_RANGE[0], LEAF_H_RANGE[1])
        min_w = _randint_inclusive(rng, LEAF_W_RANGE[0], LEAF_W_RANGE[1])
    elif len(children) == 1:
        child_h, child_w = _annotate_mins(children[0], rng)
        min_h = int(child_h + 2 * MARGIN)
        min_w = int(child_w + 2 * MARGIN)
    else:
        child_dims = [_annotate_mins(child, rng) for child in children]
        if node["layout"] == "h":
            min_h = int(max(h for h, _ in child_dims) + 2 * MARGIN)
            min_w = int(sum(w for _, w in child_dims) + GAP * (len(children) - 1) + 2 * MARGIN)
        else:
            min_h = int(sum(h for h, _ in child_dims) + GAP * (len(children) - 1) + 2 * MARGIN)
            min_w = int(max(w for _, w in child_dims) + 2 * MARGIN)

    node["min_h"] = int(min_h)
    node["min_w"] = int(min_w)
    node["node_count"] = 1 + sum(int(child.get("node_count", 1)) for child in children)
    return int(min_h), int(min_w)


def _place_node(
    node: dict[str, Any], r0: int, c0: int, height: int, width: int, rng: np.random.Generator
) -> bool:
    node["bbox"] = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
    children = node["children"]
    if not children:
        return True

    inner_h = int(height - 2 * MARGIN)
    inner_w = int(width - 2 * MARGIN)

    for _ in range(PLACE_ATTEMPTS):
        placements: list[tuple[int, int, int, int]] = []

        if len(children) == 1:
            child = children[0]
            child_h = int(child["min_h"] + rng.integers(inner_h - child["min_h"] + 1))
            child_w = int(child["min_w"] + rng.integers(inner_w - child["min_w"] + 1))
            off_r = int(rng.integers(inner_h - child_h + 1))
            off_c = int(rng.integers(inner_w - child_w + 1))
            placements.append((int(r0 + MARGIN + off_r), int(c0 + MARGIN + off_c), child_h, child_w))
        elif node["layout"] == "h":
            min_total_w = int(sum(int(child["min_w"]) for child in children) + GAP * (len(children) - 1))
            slack = _multinomial_split(inner_w - min_total_w, len(children) + 2, rng)
            current_c = int(c0 + MARGIN + slack[0])
            for index, child in enumerate(children):
                child_w = int(child["min_w"] + slack[index + 1])
                child_h = int(child["min_h"] + rng.integers(inner_h - child["min_h"] + 1))
                off_r = int(rng.integers(inner_h - child_h + 1))
                placements.append((int(r0 + MARGIN + off_r), current_c, child_h, child_w))
                current_c += int(child_w + GAP)
        else:
            min_total_h = int(sum(int(child["min_h"]) for child in children) + GAP * (len(children) - 1))
            slack = _multinomial_split(inner_h - min_total_h, len(children) + 2, rng)
            current_r = int(r0 + MARGIN + slack[0])
            for index, child in enumerate(children):
                child_h = int(child["min_h"] + slack[index + 1])
                child_w = int(child["min_w"] + rng.integers(inner_w - child["min_w"] + 1))
                off_c = int(rng.integers(inner_w - child_w + 1))
                placements.append((current_r, int(c0 + MARGIN + off_c), child_h, child_w))
                current_r += int(child_h + GAP)

        sibling_areas = [int(h * w) for _, _, h, w in placements]
        if len(set(sibling_areas)) != len(sibling_areas):
            continue

        ok = True
        for child, (cr0, cc0, ch, cw) in zip(children, placements):
            if not _place_node(child, cr0, cc0, ch, cw, rng):
                ok = False
                break
        if ok:
            return True

    return False


def _collect_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    for child in node["children"]:
        nodes.extend(_collect_nodes(child))
    return nodes


def _assign_colors(root: dict[str, Any], rng: np.random.Generator) -> None:
    nodes = _collect_nodes(root)
    colors = [int(x) for x in rng.choice(np.array(PALETTE), size=len(nodes), replace=False)]
    rng.shuffle(colors)
    for node, color in zip(nodes, colors):
        node["color"] = int(color)


def _paint_tree(node: dict[str, Any], grid: np.ndarray) -> None:
    r0, c0, r1, c1 = node["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(node["color"])
    for child in node["children"]:
        _paint_tree(child, grid)


def _sibling_areas_unique(node: dict[str, Any]) -> bool:
    children = node["children"]
    if children:
        areas = [_bbox_area(child["bbox"]) for child in children]
        if len(set(areas)) != len(areas):
            return False
    return all(_sibling_areas_unique(child) for child in children)


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    node_count = _randint_inclusive(rng, NODE_COUNT_RANGE[0], NODE_COUNT_RANGE[1])
    root = _build_tree(node_count, rng)
    min_h, min_w = _annotate_mins(root, rng)

    top = _randint_inclusive(rng, PADDING_RANGE[0], PADDING_RANGE[1])
    bottom = _randint_inclusive(rng, PADDING_RANGE[0], PADDING_RANGE[1])
    left = _randint_inclusive(rng, PADDING_RANGE[0], PADDING_RANGE[1])
    right = _randint_inclusive(rng, PADDING_RANGE[0], PADDING_RANGE[1])

    max_root_h = 30 - top - bottom
    max_root_w = 30 - left - right
    if min_h > max_root_h or min_w > max_root_w:
        return None

    root_h = int(min_h + rng.integers(min(ROOT_EXTRA_LIMIT, max_root_h - min_h) + 1))
    root_w = int(min_w + rng.integers(min(ROOT_EXTRA_LIMIT, max_root_w - min_w) + 1))

    if not _place_node(root, top, left, root_h, root_w, rng):
        return None
    if not _sibling_areas_unique(root):
        return None

    _assign_colors(root, rng)

    height = int(root_h + top + bottom)
    width = int(root_w + left + right)
    grid = np.full((height, width), BG, dtype=int)
    _paint_tree(root, grid)

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    marker = corners[int(rng.integers(len(corners)))]
    grid[marker] = int(root["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (2 * node_count - 1, 2 * node_count - 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate(RNG)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
