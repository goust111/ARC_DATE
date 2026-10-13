from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GAP = 2
OFFSET = GAP + 1
MAX_ATTEMPTS = 500


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _bbox_key(obj: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return int(r0), int(c0), int(r1), int(c1), int(obj["area"])


def _contains_bbox(outer: dict, inner: dict) -> bool:
    ar0, ac0, ar1, ac1 = outer["bbox"]
    br0, bc0, br1, bc1 = inner["bbox"]
    return (
        int(outer["area"]) > int(inner["area"])
        and ar0 <= br0
        and ac0 <= bc0
        and br1 <= ar1
        and bc1 <= ac1
        and (ar0, ac0, ar1, ac1) != (br0, bc0, br1, bc1)
    )


def _components(g: np.ndarray) -> list[dict]:
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []

    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or int(g[sr, sc]) == BLACK:
                continue
            color = int(g[sr, sc])
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": bbox,
                    "children": [],
                }
            )

    return objects


def _build_tree(objects: list[dict]) -> list[dict]:
    for obj in objects:
        obj["children"] = []

    parent: dict[int, int] = {}
    for i, child in enumerate(objects):
        containers = [j for j, obj in enumerate(objects) if i != j and _contains_bbox(obj, child)]
        if not containers:
            continue
        containers.sort(key=lambda j: (_bbox_area(objects[j]["bbox"]), int(objects[j]["area"]), _bbox_key(objects[j])))
        parent[i] = containers[0]

    for child_i, parent_i in parent.items():
        objects[parent_i]["children"].append(objects[child_i])

    for obj in objects:
        obj["children"].sort(key=_bbox_key)

    roots = [obj for i, obj in enumerate(objects) if i not in parent]
    roots.sort(key=_bbox_key)

    def depth(obj: dict) -> int:
        if not obj["children"]:
            return 1
        return 1 + max(depth(child) for child in obj["children"])

    assert roots, "expected at least one foreground object"
    assert max(depth(root) for root in roots) <= 3, "containment depth must be at most three"
    return roots


def _draw_frame(out: np.ndarray, color: int) -> None:
    out[0, :] = int(color)
    out[-1, :] = int(color)
    out[:, 0] = int(color)
    out[:, -1] = int(color)


def _render_node(obj: dict, leaf_size: int) -> np.ndarray:
    color = int(obj["color"])
    children = list(obj["children"])
    if not children:
        return np.full((leaf_size, leaf_size), color, dtype=int)

    patches = [_render_node(child, leaf_size) for child in children]
    group_h = max(int(p.shape[0]) for p in patches)
    group_w = sum(int(p.shape[1]) for p in patches) + GAP * (len(patches) - 1)
    out = np.zeros((group_h + 2 * OFFSET, group_w + 2 * OFFSET), dtype=int)
    _draw_frame(out, color)

    c = OFFSET
    for patch in patches:
        r = OFFSET + (group_h - int(patch.shape[0])) // 2
        out[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
        c += int(patch.shape[1]) + GAP
    return out


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray:
    if len(patches) == 1:
        return patches[0]

    height = max(int(p.shape[0]) for p in patches)
    width = sum(int(p.shape[1]) for p in patches) + GAP * (len(patches) - 1)
    if height <= 30 and width <= 30:
        out = np.zeros((height, width), dtype=int)
        c = 0
        for patch in patches:
            r = (height - int(patch.shape[0])) // 2
            out[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
            c += int(patch.shape[1]) + GAP
        return out

    height = sum(int(p.shape[0]) for p in patches) + GAP * (len(patches) - 1)
    width = max(int(p.shape[1]) for p in patches)
    out = np.zeros((height, width), dtype=int)
    r = 0
    for patch in patches:
        c = (width - int(patch.shape[1])) // 2
        out[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
        r += int(patch.shape[0]) + GAP
    return out


def _render_forest(roots: list[dict], leaf_size: int = 2) -> np.ndarray:
    return _pack_patches([_render_node(root, leaf_size) for root in roots])


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the containment tree of every object and redraw it as nested boxes.

    Input structure:
    - The background is Black.
    - Each non-Black 4-connected monochrome component is a foreground object with
      its own visible color.
    - Larger objects may contain smaller objects when the smaller object's
      bounding box lies completely inside the larger object's bounding box.
    - The containment depth is at most three levels.

    Rule:
    - Find all foreground objects, their pixel areas, and their bounding boxes.
    - Give each object the smallest visible container whose bbox contains it and
      whose pixel area is larger, producing a containment forest.
    - Redraw each outer object as a one-pixel rectangular border in its color.
      Children are drawn inside the parent with two Black pixels of separation.
    - Objects with no children are drawn as small solid squares; multiple roots
      and siblings are arranged in input reading order.

    Key insight:
    - The original object shapes and exact positions are discarded: only the
      bounding-box containment hierarchy and object colors survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _components(g)
    roots = _build_tree(objects)
    out = _render_forest(roots, leaf_size=2)
    if max(out.shape) > 30:
        out = _render_forest(roots, leaf_size=1)
    assert out.ndim == 2 and max(out.shape) <= 30, "output grid exceeds ARC size limit"
    return out


def _node(children: list[dict] | None = None) -> dict:
    return {"children": list(children or [])}


def _all_nodes(nodes: list[dict]) -> list[dict]:
    out: list[dict] = []

    def visit(node: dict) -> None:
        out.append(node)
        for child in node["children"]:
            visit(child)

    for node in nodes:
        visit(node)
    return out


def _spec_patterns() -> list[list[dict]]:
    return [
        [_node([_node() for _ in range(random.randint(1, 4))])],
        [
            _node(
                [
                    _node([_node() for _ in range(random.randint(1, 2))]),
                    _node() if random.random() < 0.5 else _node([_node()]),
                ]
            )
        ],
        [_node([_node() for _ in range(random.randint(1, 2))]) for _ in range(random.randint(2, 3))],
        [_node([_node([_node()])]), _node([_node() for _ in range(random.randint(1, 2))])],
    ]


def _choose_specs() -> list[dict] | None:
    roots = random.choice(_spec_patterns())
    if len(_all_nodes(roots)) > len(COLORS):
        return None
    return roots


def _layout_node(node: dict) -> None:
    children = node["children"]
    if not children:
        node["height"] = random.randint(1, 3)
        node["width"] = random.randint(1, 3)
        return

    for child in children:
        _layout_node(child)

    orient = random.choice(("horizontal", "vertical"))
    gap = random.randint(2, 4)
    margin = random.randint(3, 5)
    extra_h = random.randint(0, 3)
    extra_w = random.randint(0, 3)

    if orient == "horizontal":
        group_h = max(int(child["height"]) for child in children)
        group_w = sum(int(child["width"]) for child in children) + gap * (len(children) - 1)
        node["height"] = group_h + 2 * margin + extra_h
        node["width"] = group_w + 2 * margin + extra_w
        r = margin + random.randint(0, extra_h)
        c = margin + random.randint(0, extra_w)
        for child in children:
            child["rel_r"] = r
            child["rel_c"] = c
            c += int(child["width"]) + gap
    else:
        group_h = sum(int(child["height"]) for child in children) + gap * (len(children) - 1)
        group_w = max(int(child["width"]) for child in children)
        node["height"] = group_h + 2 * margin + extra_h
        node["width"] = group_w + 2 * margin + extra_w
        r = margin + random.randint(0, extra_h)
        c = margin + random.randint(0, extra_w)
        for child in children:
            child["rel_r"] = r
            child["rel_c"] = c
            r += int(child["height"]) + gap


def _set_absolute(node: dict, r0: int, c0: int) -> None:
    h = int(node["height"])
    w = int(node["width"])
    node["bbox"] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    for child in node["children"]:
        _set_absolute(child, r0 + int(child["rel_r"]), c0 + int(child["rel_c"]))


def _pack_specs(roots: list[dict]) -> tuple[int, int] | None:
    root_gap = random.randint(2, 4)
    horizontal = random.choice((True, False))
    choices = [horizontal, not horizontal]

    for use_horizontal in choices:
        if use_horizontal:
            group_h = max(int(root["height"]) for root in roots)
            group_w = sum(int(root["width"]) for root in roots) + root_gap * (len(roots) - 1)
        else:
            group_h = sum(int(root["height"]) for root in roots) + root_gap * (len(roots) - 1)
            group_w = max(int(root["width"]) for root in roots)
        if group_h > 30 or group_w > 30:
            continue

        height = random.randint(group_h, 30)
        width = random.randint(group_w, 30)
        base_r = random.randint(0, height - group_h)
        base_c = random.randint(0, width - group_w)
        if use_horizontal:
            c = base_c
            for root in roots:
                _set_absolute(root, base_r, c)
                c += int(root["width"]) + root_gap
        else:
            r = base_r
            for root in roots:
                _set_absolute(root, r, base_c)
                r += int(root["height"]) + root_gap
        return height, width
    return None


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    height, width = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(height - 1, r1 + pad), min(width - 1, c1 + pad)


def _inside_any(r: int, c: int, bboxes: list[tuple[int, int, int, int]]) -> bool:
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in bboxes)


def _descendant_bboxes(node: dict) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for child in node["children"]:
        out.append(child["bbox"])
        out.extend(_descendant_bboxes(child))
    return out


def _paint_internal(grid: np.ndarray, node: dict) -> None:
    r0, c0, r1, c1 = node["bbox"]
    color = int(node["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    protected = [_expanded_bbox(b, 1, grid.shape) for b in _descendant_bboxes(node)]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    bump_budget = random.randint(0, max(1, (height + width) // 5))
    border_cells: list[tuple[int, int, int, int]] = []
    for c in range(c0 + 1, c1):
        border_cells.append((r0, c, 1, 0))
        border_cells.append((r1, c, -1, 0))
    for r in range(r0 + 1, r1):
        border_cells.append((r, c0, 0, 1))
        border_cells.append((r, c1, 0, -1))
    random.shuffle(border_cells)
    for r, c, dr, dc in border_cells[:bump_budget]:
        rr, cc = r + dr, c + dc
        if r0 < rr < r1 and c0 < cc < c1 and not _inside_any(rr, cc, protected):
            grid[rr, cc] = color


def _leaf_mask(height: int, width: int) -> np.ndarray:
    if height == 1 and width == 1:
        return np.ones((1, 1), dtype=bool)
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=bool)
    mode = random.choice(("solid", "l", "path"))
    if mode == "solid":
        return np.ones((height, width), dtype=bool)
    if mode == "l":
        mask = np.zeros((height, width), dtype=bool)
        if random.random() < 0.5:
            mask[:, 0] = True
            mask[-1, :] = True
        else:
            mask[0, :] = True
            mask[:, -1] = True
        return mask

    mask = np.zeros((height, width), dtype=bool)
    r = 0
    c = 0
    mask[r, c] = True
    while r < height - 1 or c < width - 1:
        if r == height - 1:
            c += 1
        elif c == width - 1:
            r += 1
        elif random.random() < 0.5:
            r += 1
        else:
            c += 1
        mask[r, c] = True
    return mask


def _paint_node(grid: np.ndarray, node: dict) -> None:
    if node["children"]:
        _paint_internal(grid, node)
    else:
        r0, c0, r1, c1 = node["bbox"]
        mask = _leaf_mask(r1 - r0 + 1, c1 - c0 + 1)
        grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(node["color"])

    for child in node["children"]:
        _paint_node(grid, child)


def _spec_to_obj(node: dict) -> dict:
    obj = {
        "color": int(node["color"]),
        "bbox": node["bbox"],
        "area": 1,
        "children": [_spec_to_obj(child) for child in node["children"]],
    }
    obj["children"].sort(key=_bbox_key)
    return obj


def _expected_from_specs(roots: list[dict]) -> np.ndarray:
    objs = [_spec_to_obj(root) for root in roots]
    objs.sort(key=_bbox_key)
    return _render_forest(objs, leaf_size=2)


def _gen_attempt() -> np.ndarray | None:
    roots = _choose_specs()
    if roots is None:
        return None
    for root in roots:
        _layout_node(root)

    packed = _pack_specs(roots)
    if packed is None:
        return None
    height, width = packed

    nodes = _all_nodes(roots)
    colors = random.sample(COLORS, len(nodes))
    for node, color in zip(nodes, colors):
        node["color"] = int(color)

    expected = _expected_from_specs(roots)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    for root in roots:
        _paint_node(grid, root)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/8/3 input")
