from __future__ import annotations

import random

import numpy as np


BG = 0
INPUT_COLOR = 5
DEPTH_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow

MARGIN_TO_CHILD = 2
SIBLING_GAP = 2
ROOT_GAP = 2
LEAF_MIN_SIZE = 4
LEAF_MAX_SIZE = 8
GRID_H_RANGE = (20, 32)
GRID_W_RANGE = (20, 32)
MAX_GENERATION_ATTEMPTS = 400
ROOT_PLACEMENT_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            parts.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return parts


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _containment_depths(parts: list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]) -> list[int]:
    parents = [-1] * len(parts)
    for i, part in enumerate(parts):
        bbox = part["bbox"]
        containers = [
            j
            for j, other in enumerate(parts)
            if i != j and _strictly_contains(other["bbox"], bbox)
        ]
        if containers:
            parents[i] = min(containers, key=lambda j: _bbox_area(parts[j]["bbox"]))

    depths = [-1] * len(parts)

    def visit(idx: int) -> int:
        if depths[idx] != -1:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 0 if parent == -1 else visit(parent) + 1
        assert depths[idx] < len(DEPTH_COLORS), "nesting depth exceeds supported palette"
        return depths[idx]

    for i in range(len(parts)):
        visit(i)
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by its depth in the bbox-containment forest.

    Input structure:
    - Black background.
    - Several same-color 4-connected foreground objects.
    - Some objects have bounding boxes strictly inside another object's bbox.

    Rule:
    - Build the direct-parent relation from strict bbox containment.
    - Roots have depth 0.
    - Recolor depths 0/1/2/3 to blue/red/green/yellow.
    - Keep the background unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    parts = _components(g != BG)
    assert parts, "expected at least one foreground object"
    depths = _containment_depths(parts)

    out = np.full_like(g, BG)
    for part, depth in zip(parts, depths, strict=False):
        coords = part["coords"]
        out[coords[:, 0], coords[:, 1]] = DEPTH_COLORS[depth]
    return out


def _sample_tree(depth_left: int) -> dict:
    node: dict[str, object] = {"children": []}
    if depth_left > 1:
        child_count = random.choices([1, 2], weights=[3, 2], k=1)[0]
        deep_idx = random.randrange(child_count)
        children = []
        for idx in range(child_count):
            child_depth = depth_left - 1 if idx == deep_idx else random.randint(1, depth_left - 1)
            children.append(_sample_tree(child_depth))
        random.shuffle(children)
        node["children"] = children
        node["layout"] = "single" if child_count == 1 else random.choice(("h", "v"))
    _annotate_min_size(node)
    return node


def _annotate_min_size(node: dict) -> None:
    children = node["children"]
    if not children:
        node["min_h"] = random.randint(LEAF_MIN_SIZE, LEAF_MAX_SIZE)
        node["min_w"] = random.randint(LEAF_MIN_SIZE, LEAF_MAX_SIZE)
        return

    if len(children) == 1:
        child = children[0]
        node["min_h"] = child["min_h"] + 2 * MARGIN_TO_CHILD
        node["min_w"] = child["min_w"] + 2 * MARGIN_TO_CHILD
        return

    if node["layout"] == "h":
        node["min_h"] = max(child["min_h"] for child in children) + 2 * MARGIN_TO_CHILD
        node["min_w"] = sum(child["min_w"] for child in children) + SIBLING_GAP + 2 * MARGIN_TO_CHILD
    else:
        node["min_h"] = sum(child["min_h"] for child in children) + SIBLING_GAP + 2 * MARGIN_TO_CHILD
        node["min_w"] = max(child["min_w"] for child in children) + 2 * MARGIN_TO_CHILD


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_roots(nodes: list[dict], height: int, width: int) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    for node in sorted(nodes, key=lambda item: item["min_h"] * item["min_w"], reverse=True):
        ok = False
        for _ in range(ROOT_PLACEMENT_ATTEMPTS):
            max_h = min(height, node["min_h"] + 4)
            max_w = min(width, node["min_w"] + 4)
            box_h = random.randint(node["min_h"], max_h)
            box_w = random.randint(node["min_w"], max_w)
            r0 = random.randint(0, height - box_h)
            c0 = random.randint(0, width - box_w)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_boxes_too_close(bbox, other, ROOT_GAP) for other in placed):
                continue
            node["bbox"] = bbox
            placed.append(bbox)
            ok = True
            break
        if not ok:
            return False
    return True


def _layout_children(node: dict) -> bool:
    children = node["children"]
    if not children:
        node["tabs"] = _sample_tabs(node["bbox"])
        return True

    r0, c0, r1, c1 = node["bbox"]
    ir0 = r0 + MARGIN_TO_CHILD
    ic0 = c0 + MARGIN_TO_CHILD
    inner_h = r1 - r0 + 1 - 2 * MARGIN_TO_CHILD
    inner_w = c1 - c0 + 1 - 2 * MARGIN_TO_CHILD
    if inner_h <= 0 or inner_w <= 0:
        return False

    if len(children) == 1:
        child = children[0]
        if child["min_h"] > inner_h or child["min_w"] > inner_w:
            return False
        box_h = random.randint(child["min_h"], inner_h)
        box_w = random.randint(child["min_w"], inner_w)
        child_r0 = ir0 + random.randint(0, inner_h - box_h)
        child_c0 = ic0 + random.randint(0, inner_w - box_w)
        child["bbox"] = (child_r0, child_c0, child_r0 + box_h - 1, child_c0 + box_w - 1)
        return _layout_children(child)

    if node["layout"] == "h":
        total_min_w = sum(child["min_w"] for child in children) + SIBLING_GAP
        max_min_h = max(child["min_h"] for child in children)
        if total_min_w > inner_w or max_min_h > inner_h:
            return False

        extra_w = inner_w - total_min_w
        add0 = random.randint(0, extra_w)
        add1 = random.randint(0, extra_w - add0)
        widths = [children[0]["min_w"] + add0, children[1]["min_w"] + add1]
        spare = inner_w - (widths[0] + widths[1] + SIBLING_GAP)
        lead = random.randint(0, spare)
        between = random.randint(0, spare - lead)
        cols = [
            ic0 + lead,
            ic0 + lead + widths[0] + SIBLING_GAP + between,
        ]
        for idx, child in enumerate(children):
            box_h = random.randint(child["min_h"], inner_h)
            child_r0 = ir0 + random.randint(0, inner_h - box_h)
            child["bbox"] = (child_r0, cols[idx], child_r0 + box_h - 1, cols[idx] + widths[idx] - 1)
    else:
        total_min_h = sum(child["min_h"] for child in children) + SIBLING_GAP
        max_min_w = max(child["min_w"] for child in children)
        if total_min_h > inner_h or max_min_w > inner_w:
            return False

        extra_h = inner_h - total_min_h
        add0 = random.randint(0, extra_h)
        add1 = random.randint(0, extra_h - add0)
        heights = [children[0]["min_h"] + add0, children[1]["min_h"] + add1]
        spare = inner_h - (heights[0] + heights[1] + SIBLING_GAP)
        lead = random.randint(0, spare)
        between = random.randint(0, spare - lead)
        rows = [
            ir0 + lead,
            ir0 + lead + heights[0] + SIBLING_GAP + between,
        ]
        for idx, child in enumerate(children):
            box_w = random.randint(child["min_w"], inner_w)
            child_c0 = ic0 + random.randint(0, inner_w - box_w)
            child["bbox"] = (rows[idx], child_c0, rows[idx] + heights[idx] - 1, child_c0 + box_w - 1)

    return all(_layout_children(child) for child in children)


def _sample_tabs(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    tabs: set[tuple[int, int]] = set()
    tab_count = random.choices([0, 1, 2], weights=[3, 3, 1], k=1)[0]
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides[:tab_count]:
        if side in {"top", "bottom"} and w < 4:
            continue
        if side in {"left", "right"} and h < 4:
            continue
        if side == "top":
            col = random.randint(c0 + 1, c1 - 1)
            length = random.randint(1, max(1, h - 3))
            tabs.update((r0 + step, col) for step in range(1, length + 1))
        elif side == "bottom":
            col = random.randint(c0 + 1, c1 - 1)
            length = random.randint(1, max(1, h - 3))
            tabs.update((r1 - step, col) for step in range(1, length + 1))
        elif side == "left":
            row = random.randint(r0 + 1, r1 - 1)
            length = random.randint(1, max(1, w - 3))
            tabs.update((row, c0 + step) for step in range(1, length + 1))
        else:
            row = random.randint(r0 + 1, r1 - 1)
            length = random.randint(1, max(1, w - 3))
            tabs.update((row, c1 - step) for step in range(1, length + 1))
    return sorted(tabs)


def _paint_node(
    inp: np.ndarray,
    out: np.ndarray,
    node: dict,
    depth: int,
) -> None:
    r0, c0, r1, c1 = node["bbox"]
    inp[r0, c0 : c1 + 1] = INPUT_COLOR
    inp[r1, c0 : c1 + 1] = INPUT_COLOR
    inp[r0 : r1 + 1, c0] = INPUT_COLOR
    inp[r0 : r1 + 1, c1] = INPUT_COLOR
    for rr, cc in node.get("tabs", []):
        inp[rr, cc] = INPUT_COLOR

    color = DEPTH_COLORS[depth]
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    for rr, cc in node.get("tabs", []):
        out[rr, cc] = color

    for child in node["children"]:
        _paint_node(inp, out, child, depth + 1)


def _render(nodes: list[dict], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for node in nodes:
        _paint_node(inp, out, node, depth=0)
    return inp, out


def _collect_nodes(node: dict) -> list[dict]:
    items = [node]
    for child in node["children"]:
        items.extend(_collect_nodes(child))
    return items


def _gen_attempt() -> np.ndarray | None:
    max_depth = random.choices([1, 2, 3, 4], weights=[1, 2, 3, 3], k=1)[0]
    root_depths = [max_depth]
    extra_roots = random.randint(0, 2 if max_depth <= 2 else 1)
    for _ in range(extra_roots):
        root_depths.append(random.randint(1, max_depth))

    nodes = [_sample_tree(depth) for depth in root_depths]
    min_height = max(GRID_H_RANGE[0], max(node["min_h"] for node in nodes))
    min_width = max(GRID_W_RANGE[0], max(node["min_w"] for node in nodes))
    if min_height > GRID_H_RANGE[1] or min_width > GRID_W_RANGE[1]:
        return None
    height = random.randint(min_height, GRID_H_RANGE[1])
    width = random.randint(min_width, GRID_W_RANGE[1])

    if sum(node["min_h"] * node["min_w"] for node in nodes) > height * width:
        return None
    if not _place_roots(nodes, height, width):
        return None
    if not all(_layout_children(node) for node in nodes):
        return None

    inp, expected = _render(nodes, height, width)
    parts = _components(inp != BG)
    expected_count = sum(len(_collect_nodes(node)) for node in nodes)
    if len(parts) != expected_count:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-containment task")
