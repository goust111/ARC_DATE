from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        cells: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components_mask(grid != bg):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each foreground object must be monochromatic"
        objects.append(
            {
                "coords": coords.copy(),
                "color": int(colors[0]),
            }
        )
    return objects


def _occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "objects must not overlap"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _movable_right(objects: list[dict], occ: np.ndarray) -> set[int]:
    _, w = occ.shape
    movable = set(range(len(objects)))
    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            coords = objects[idx]["coords"]
            shifted_cols = coords[:, 1] + 1
            if np.any(shifted_cols >= w):
                movable.remove(idx)
                changed = True
                continue
            labels = occ[coords[:, 0], shifted_cols]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True
    return movable


def _settle_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict]:
    settled = [
        {
            "coords": obj["coords"].copy(),
            "color": int(obj["color"]),
        }
        for obj in objects
    ]
    _, w = shape
    for _step in range(w):
        occ = _occupancy(shape, settled)
        movable = _movable_right(settled, occ)
        if not movable:
            return settled
        for idx in movable:
            settled[idx]["coords"][:, 1] += 1
    raise AssertionError("rightward settling exceeded grid width")


def _support_map(objects: list[dict], shape: tuple[int, int]) -> dict[int, int]:
    h, w = shape
    occ = _occupancy((h, w), objects)
    support: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        touching_wall = bool(np.any(coords[:, 1] == w - 1))
        inside = coords[:, 1] + 1 < w
        labels = occ[coords[inside, 0], coords[inside, 1] + 1]
        blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
        assert len(blockers) <= 1, "each object must have at most one right-side supporter"
        if blockers:
            assert not touching_wall, "wall-root cannot also be blocked by another object"
            support[idx] = next(iter(blockers))
        else:
            assert touching_wall, "stable object without right neighbor must touch the wall"
    return support


def _root_index(idx: int, support: dict[int, int], cache: dict[int, int]) -> int:
    if idx in cache:
        return cache[idx]
    if idx not in support:
        cache[idx] = idx
    else:
        cache[idx] = _root_index(support[idx], support, cache)
    return cache[idx]


def _recolor_settled(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    support = _support_map(objects, shape)
    root_cache: dict[int, int] = {}
    out = np.full(shape, bg, dtype=int)
    for idx, obj in enumerate(objects):
        root = _root_index(idx, support, root_cache)
        color = int(objects[root]["color"])
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every foreground object rightward until it settles, then recolor each object by the wall-root at the end of its support chain.

    Input structure:
    - The grid has one dominant background color and several separated monochromatic 4-connected foreground objects.
    - Foreground row bands overlap so that after eastward settling some objects stop against the left side of other objects.
    - In the settled state each object has at most one distinct right-side neighbor, though a single object may support multiple objects on its left.

    Rule:
    - Treat every foreground object as a rigid body and repeatedly apply one synchronous one-cell slide to the right whenever that object is not blocked by the border or by an object that is already unable to move in that step.
    - After settling, define a support edge A -> B when some cell of A is immediately left-adjacent to a cell of B.
    - Objects that touch the right border keep their own colors; every other object follows its support edge chain to the unique border-touching root and is recolored uniformly to that root's color.

    Key insight:
    - The final color is determined by the entire collision chain, not just by the object that directly blocks a sprite or by the rightmost color visible in the same rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    assert objects, "expected at least one foreground object"

    settled = _settle_objects(objects, g.shape)
    return _recolor_settled(settled, g.shape, bg)


# --------------- generator ---------------

GEN_ATTEMPTS = 400
WIDTH_RANGE = (16, 26)
HEIGHT_RANGE = (14, 24)
RECT_H_SMALL = (2, 4)
RECT_H_LEAF = (2, 3)
RECT_W_RANGE = (2, 4)
MAX_LEFT_SHIFT = 8
TREE_GAP_RANGE = (1, 2)


def _overlap_top(parent_top: int, parent_h: int, child_h: int) -> int:
    return random.randint(parent_top - child_h + 1, parent_top + parent_h - 1)


def _normalize_rows(nodes: list[dict]) -> tuple[list[dict], int]:
    min_r = min(int(node["r"]) for node in nodes)
    out: list[dict] = []
    for node in nodes:
        cur = dict(node)
        cur["r"] = int(cur["r"]) - min_r
        out.append(cur)
    height = max(int(node["r"]) + int(node["h"]) for node in out)
    return out, height


def _assign_local_cols(nodes: list[dict]) -> tuple[list[dict], int]:
    path_widths: list[int] = []
    for idx, node in enumerate(nodes):
        width = int(node["w"])
        parent = node["parent"]
        if parent is None:
            path_widths.append(width)
        else:
            path_widths.append(path_widths[int(parent)] + width)
    used_width = max(path_widths)
    out: list[dict] = []
    for node, path_w in zip(nodes, path_widths):
        cur = dict(node)
        cur["c"] = used_width - path_w
        out.append(cur)
    return out, used_width


def _build_primary_tree() -> dict:
    h_leaf_a = random.randint(*RECT_H_LEAF)
    h_leaf_b = random.randint(*RECT_H_LEAF)
    h_mid = h_leaf_a + h_leaf_b + 1 + random.randint(0, 2)
    h_root = random.randint(*RECT_H_SMALL)

    nodes = [
        {"parent": None, "r": 0, "h": h_root, "w": random.randint(*RECT_W_RANGE)},
        {"parent": 0, "r": _overlap_top(0, h_root, h_mid), "h": h_mid, "w": random.randint(*RECT_W_RANGE)},
    ]

    slack = h_mid - (h_leaf_a + h_leaf_b + 1)
    top_slack = random.randint(0, slack)
    mid_top = int(nodes[1]["r"])
    leaf_specs = [(h_leaf_a, random.randint(*RECT_W_RANGE)), (h_leaf_b, random.randint(*RECT_W_RANGE))]
    if random.random() < 0.5:
        leaf_specs = [leaf_specs[1], leaf_specs[0]]
    top_leaf_h = int(leaf_specs[0][0])
    leaf_rows = [mid_top + top_slack, mid_top + top_slack + top_leaf_h + 1]
    for (leaf_h, leaf_w), leaf_r in zip(leaf_specs, leaf_rows):
        nodes.append({"parent": 1, "r": leaf_r, "h": leaf_h, "w": leaf_w})

    nodes, height = _normalize_rows(nodes)
    nodes, used_width = _assign_local_cols(nodes)
    return {"nodes": nodes, "height": height, "used_width": used_width}


def _build_secondary_tree() -> dict:
    if random.random() < 0.25:
        nodes = [
            {"parent": None, "r": 0, "h": random.randint(*RECT_H_SMALL), "w": random.randint(*RECT_W_RANGE)},
        ]
    else:
        h_root = random.randint(*RECT_H_SMALL)
        h_child = random.randint(*RECT_H_SMALL)
        nodes = [
            {"parent": None, "r": 0, "h": h_root, "w": random.randint(*RECT_W_RANGE)},
            {"parent": 0, "r": _overlap_top(0, h_root, h_child), "h": h_child, "w": random.randint(*RECT_W_RANGE)},
        ]
    nodes, height = _normalize_rows(nodes)
    nodes, used_width = _assign_local_cols(nodes)
    return {"nodes": nodes, "height": height, "used_width": used_width}


def _pack_forest(trees: list[dict]) -> tuple[list[dict], tuple[int, int]] | None:
    random.shuffle(trees)
    gaps = [random.randint(*TREE_GAP_RANGE) for _ in range(len(trees) - 1)]
    used_height = sum(int(tree["height"]) for tree in trees) + sum(gaps)
    global_used_width = max(int(tree["used_width"]) for tree in trees)

    min_h = max(HEIGHT_RANGE[0], used_height + 2)
    max_h = min(HEIGHT_RANGE[1], 30)
    min_w = max(WIDTH_RANGE[0], global_used_width + 4)
    max_w = min(WIDTH_RANGE[1], 30)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    top_pad = random.randint(0, h - used_height)
    extra_left = w - global_used_width

    objects: list[dict] = []
    row = top_pad
    for tree_idx, tree in enumerate(trees):
        left_offset = global_used_width - int(tree["used_width"]) + extra_left
        base = len(objects)
        for node in tree["nodes"]:
            parent = node["parent"]
            objects.append(
                {
                    "parent": None if parent is None else base + int(parent),
                    "r": row + int(node["r"]),
                    "c": left_offset + int(node["c"]),
                    "h": int(node["h"]),
                    "w": int(node["w"]),
                }
            )
        row += int(tree["height"])
        if tree_idx < len(gaps):
            row += gaps[tree_idx]

    return objects, (h, w)


def _assign_colors(objects: list[dict], bg: int) -> None:
    palette = [color for color in range(10) if color != bg]
    assert len(objects) <= len(palette), "need unique visible colors for all objects"
    colors = random.sample(palette, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)


def _children_map(objects: list[dict]) -> tuple[int, dict[int, list[int]]]:
    children = {idx: [] for idx in range(len(objects))}
    root = None
    for idx, obj in enumerate(objects):
        parent = obj["parent"]
        if parent is None:
            root = idx
        else:
            children[int(parent)].append(idx)
    assert root is not None
    return root, children


def _forest_roots(objects: list[dict]) -> list[int]:
    return [idx for idx, obj in enumerate(objects) if obj["parent"] is None]


def _assign_shifts(objects: list[dict]) -> dict[int, int] | None:
    children = {idx: [] for idx in range(len(objects))}
    for idx, obj in enumerate(objects):
        parent = obj["parent"]
        if parent is not None:
            children[int(parent)].append(idx)

    caps: dict[int, int] = {}

    def visit(idx: int) -> int:
        cap = min(MAX_LEFT_SHIFT, int(objects[idx]["c"]))
        for child in children[idx]:
            cap = min(cap, visit(child) - 1)
        caps[idx] = cap
        return cap

    for root in _forest_roots(objects):
        if visit(root) < 1:
            return None

    shifts: dict[int, int] = {}

    def assign(idx: int, min_shift: int) -> bool:
        if min_shift > caps[idx]:
            return False
        shifts[idx] = random.randint(min_shift, caps[idx])
        for child in children[idx]:
            if not assign(child, shifts[idx] + 1):
                return False
        return True

    for root in _forest_roots(objects):
        if not assign(root, 1):
            return None
    return shifts


def _place_rect(grid: np.ndarray, obj: dict, color: int, col_shift: int = 0) -> bool:
    r = int(obj["r"])
    c = int(obj["c"]) - int(col_shift)
    h = int(obj["h"])
    w = int(obj["w"])
    if r < 0 or c < 0 or r + h > grid.shape[0] or c + w > grid.shape[1]:
        return False
    if np.any(grid[r : r + h, c : c + w] != _get_bg(grid)):
        return False
    grid[r : r + h, c : c + w] = int(color)
    return True


def _render_input(objects: list[dict], shape: tuple[int, int], bg: int, shifts: dict[int, int]) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    for idx, obj in enumerate(objects):
        if not _place_rect(grid, obj, int(obj["color"]), col_shift=int(shifts[idx])):
            return None
    return grid


def _root_of_spec(idx: int, objects: list[dict], cache: dict[int, int]) -> int:
    if idx in cache:
        return cache[idx]
    parent = objects[idx]["parent"]
    if parent is None:
        cache[idx] = idx
    else:
        cache[idx] = _root_of_spec(int(parent), objects, cache)
    return cache[idx]


def _render_expected(objects: list[dict], shape: tuple[int, int], bg: int, immediate_only: bool = False) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    root_cache: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        if immediate_only and obj["parent"] is not None:
            color = int(objects[int(obj["parent"])]["color"])
        else:
            color = int(objects[_root_of_spec(idx, objects, root_cache)]["color"])
        r = int(obj["r"])
        c = int(obj["c"])
        h = int(obj["h"])
        w = int(obj["w"])
        out[r : r + h, c : c + w] = color
    return out


def _support_map_by_color(grid: np.ndarray) -> dict[int, int | None]:
    bg = _get_bg(grid)
    objects = _extract_objects(grid, bg)
    settled = _settle_objects(objects, grid.shape)
    support = _support_map(settled, grid.shape)
    return {
        int(obj["color"]): None if idx not in support else int(settled[support[idx]]["color"])
        for idx, obj in enumerate(settled)
    }


def _spec_support_by_color(objects: list[dict]) -> dict[int, int | None]:
    return {
        int(obj["color"]): None if obj["parent"] is None else int(objects[int(obj["parent"])]["color"])
        for obj in objects
    }


def _depths(objects: list[dict]) -> list[int]:
    depth_cache: dict[int, int] = {}

    def depth(idx: int) -> int:
        if idx in depth_cache:
            return depth_cache[idx]
        parent = objects[idx]["parent"]
        if parent is None:
            depth_cache[idx] = 0
        else:
            depth_cache[idx] = 1 + depth(int(parent))
        return depth_cache[idx]

    return [depth(idx) for idx in range(len(objects))]


def _validate_case(inp: np.ndarray, objects: list[dict], bg: int) -> bool:
    expected = _render_expected(objects, inp.shape, bg)
    immediate = _render_expected(objects, inp.shape, bg, immediate_only=True)
    if np.array_equal(expected, immediate):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if len(_extract_objects(inp, bg)) != len(objects):
        return False
    if _support_map_by_color(inp) != _spec_support_by_color(objects):
        return False
    depths = _depths(objects)
    if max(depths) < 2:
        return False
    child_counts = Counter(
        int(obj["parent"])
        for obj in objects
        if obj["parent"] is not None
    )
    if max(child_counts.values(), default=0) < 2:
        return False
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return False
    return True


def _generate_case() -> np.ndarray | None:
    bg = random.randint(0, 9)
    trees = [_build_primary_tree()]
    if random.random() < 0.7:
        trees.append(_build_secondary_tree())

    packed = _pack_forest(trees)
    if packed is None:
        return None
    objects, shape = packed
    _assign_colors(objects, bg)

    shifts = _assign_shifts(objects)
    if shifts is None:
        return None

    inp = _render_input(objects, shape, bg, shifts)
    if inp is None:
        return None
    if not _validate_case(inp, objects, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
