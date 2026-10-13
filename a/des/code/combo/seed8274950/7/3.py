from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400
MAX_SCENE_SIDE = 30
MIN_SCENE_SIDE = 8


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "bbox_area": _bbox_area(bbox),
                    "area": int(arr.shape[0]),
                }
            )

    return objects


def _enclosed_region_labels(obj: dict) -> np.ndarray:
    r0, c0, r1, c1 = obj["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    rel = obj["coords"] - np.asarray([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = (~mask) & (~outside)
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for rr in range(h):
        for cc in range(w):
            if not enclosed[rr, cc] or labels[rr, cc] != 0:
                continue
            next_label += 1
            labels[rr, cc] = next_label
            q.append((rr, cc))
            while q:
                cr, cc2 = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc2 + dc
                    if 0 <= nr < h and 0 <= nc < w and enclosed[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))

    return labels


def _parent_indices(objects: list[dict]) -> list[int | None]:
    labels_cache = [_enclosed_region_labels(obj) for obj in objects]
    parents: list[int | None] = [None] * len(objects)

    for child_idx, child in enumerate(objects):
        candidates: list[int] = []
        cr0, cc0, cr1, cc1 = child["bbox"]
        for parent_idx, parent in enumerate(objects):
            if child_idx == parent_idx:
                continue
            pr0, pc0, pr1, pc1 = parent["bbox"]
            if not (pr0 <= cr0 and pc0 <= cc0 and cr1 <= pr1 and cc1 <= pc1):
                continue

            rel = child["coords"] - np.asarray([pr0, pc0], dtype=int)
            region_ids = labels_cache[parent_idx][rel[:, 0], rel[:, 1]]
            first = int(region_ids[0])
            if first > 0 and bool(np.all(region_ids == first)):
                candidates.append(parent_idx)

        if candidates:
            candidates.sort(
                key=lambda idx: (
                    objects[idx]["bbox_area"],
                    objects[idx]["bbox"][0],
                    objects[idx]["bbox"][1],
                    objects[idx]["bbox"][2],
                    objects[idx]["bbox"][3],
                )
            )
            parents[child_idx] = candidates[0]

    return parents


def _object_order(objects: list[dict]) -> list[int]:
    parents = _parent_indices(objects)
    depths: list[int | None] = [None] * len(objects)

    def depth_of(idx: int) -> int:
        if depths[idx] is not None:
            return int(depths[idx])
        parent = parents[idx]
        depths[idx] = 0 if parent is None else depth_of(parent) + 1
        return int(depths[idx])

    order = list(range(len(objects)))
    order.sort(
        key=lambda idx: (
            depth_of(idx),
            -objects[idx]["area"],
            objects[idx]["bbox"][0],
            objects[idx]["bbox"][1],
            objects[idx]["bbox"][2] - objects[idx]["bbox"][0],
            objects[idx]["bbox"][3] - objects[idx]["bbox"][1],
        )
    )
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order objects by containment depth, then by area within each depth.

    The output is a 1-row grid whose entries are the object colors, listed from
    the outermost objects to the innermost objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _extract_objects(g, bg)
    if not objects:
        return np.zeros((1, 0), dtype=int)

    order = _object_order(objects)
    colors = [objects[idx]["color"] for idx in order]
    return np.asarray([colors], dtype=int)


def _leaf_mask(style: str, h: int, w: int, variant: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if style == "solid":
        mask[:, :] = True
    elif style == "cross":
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    else:
        if variant == 0:
            mask[0, :] = True
            mask[:, 0] = True
        elif variant == 1:
            mask[0, :] = True
            mask[:, -1] = True
        elif variant == 2:
            mask[-1, :] = True
            mask[:, 0] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
    return mask


def _frame_mask(h: int, w: int, thickness: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if h > 2 * thickness and w > 2 * thickness:
        mask[thickness : h - thickness, thickness : w - thickness] = False
    return mask


def _leaf_node() -> dict:
    return {"children": []}


def _frame_node(children: list[dict]) -> dict:
    return {"children": children}


def _build_skeleton(rng: random.Random) -> list[dict]:
    mode = rng.choice(["chain", "branch", "forest"])

    if mode == "chain":
        length = rng.randint(3, 4)
        node = _leaf_node()
        for _ in range(length - 1):
            node = _frame_node([node])
        roots = [node]
        if rng.random() < 0.65:
            roots.append(_leaf_node())
        return roots

    if mode == "branch":
        child_count = rng.choice([2, 3])
        children = [_leaf_node() for _ in range(child_count)]
        if rng.random() < 0.7:
            idx = rng.randrange(child_count)
            grandchild = _leaf_node()
            if rng.random() < 0.35:
                grandchild = _frame_node([grandchild])
            children[idx] = _frame_node([grandchild])
        roots = [_frame_node(children)]
        if rng.random() < 0.5:
            roots.append(_leaf_node())
        return roots

    first_children = [_leaf_node() for _ in range(rng.choice([1, 2]))]
    if rng.random() < 0.5:
        idx = rng.randrange(len(first_children))
        first_children[idx] = _frame_node([_leaf_node()])
    root_a = _frame_node(first_children)
    root_b = _frame_node([_leaf_node()]) if rng.random() < 0.5 else _leaf_node()
    return [root_a, root_b]


def _collect_nodes(roots: list[dict]) -> list[dict]:
    out: list[dict] = []

    def visit(node: dict, depth: int) -> None:
        node["depth"] = depth
        out.append(node)
        for child in node["children"]:
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return out


def _assign_colors(nodes: list[dict], bg: int, rng: random.Random) -> None:
    palette = [color for color in range(10) if color != bg]
    colors = rng.sample(palette, len(nodes))
    for node, color in zip(nodes, colors):
        node["color"] = color


def _assign_sizes(node: dict, rng: random.Random) -> None:
    if not node["children"]:
        style = rng.choices(["solid", "cross", "ell"], weights=[4, 2, 3], k=1)[0]
        variant = rng.randrange(4)
        if style == "cross":
            h = rng.randint(3, 5)
            w = rng.randint(3, 5)
        else:
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
        node["style"] = style
        node["variant"] = variant
        node["height"] = h
        node["width"] = w
        node["mask"] = _leaf_mask(style, h, w, variant)
        node["area"] = int(node["mask"].sum())
        return

    for child in node["children"]:
        _assign_sizes(child, rng)

    child_heights = [child["height"] for child in node["children"]]
    child_widths = [child["width"] for child in node["children"]]
    layout = "single" if len(node["children"]) == 1 else rng.choice(["horizontal", "vertical"])
    gap = 0 if len(node["children"]) == 1 else rng.randint(1, 2)

    if layout == "horizontal":
        content_h = max(child_heights)
        content_w = sum(child_widths) + gap * (len(child_widths) - 1)
    elif layout == "vertical":
        content_h = sum(child_heights) + gap * (len(child_heights) - 1)
        content_w = max(child_widths)
    else:
        content_h = child_heights[0]
        content_w = child_widths[0]

    thickness = 1
    if min(content_h, content_w) >= 4 and rng.random() < 0.2:
        thickness = 2
    pad = thickness + 1

    h = content_h + 2 * pad + rng.randint(0, 2)
    w = content_w + 2 * pad + rng.randint(0, 2)
    node["style"] = "frame"
    node["thickness"] = thickness
    node["layout"] = layout
    node["gap"] = gap
    node["height"] = h
    node["width"] = w
    node["mask"] = _frame_mask(h, w, thickness)
    node["area"] = int(node["mask"].sum())


def _place_node(node: dict, top: int, left: int, rng: random.Random) -> bool:
    h = int(node["height"])
    w = int(node["width"])
    node["bbox"] = (int(top), int(left), int(top + h - 1), int(left + w - 1))

    if not node["children"]:
        return True

    thickness = int(node["thickness"])
    inner_top = top + thickness + 1
    inner_left = left + thickness + 1
    inner_bottom = top + h - thickness - 2
    inner_right = left + w - thickness - 2
    avail_h = inner_bottom - inner_top + 1
    avail_w = inner_right - inner_left + 1
    if avail_h <= 0 or avail_w <= 0:
        return False

    children = node["children"]
    if node["layout"] == "single":
        child = children[0]
        dy = avail_h - child["height"]
        dx = avail_w - child["width"]
        if dy < 0 or dx < 0:
            return False
        child_top = inner_top + rng.randint(0, dy)
        child_left = inner_left + rng.randint(0, dx)
        return _place_node(child, child_top, child_left, rng)

    gap = int(node["gap"])
    if node["layout"] == "horizontal":
        total_w = sum(child["width"] for child in children) + gap * (len(children) - 1)
        if total_w > avail_w:
            return False
        cursor = inner_left + rng.randint(0, avail_w - total_w)
        for child in children:
            dy = avail_h - child["height"]
            if dy < 0:
                return False
            child_top = inner_top + rng.randint(0, dy)
            if not _place_node(child, child_top, cursor, rng):
                return False
            cursor += child["width"] + gap
        return True

    total_h = sum(child["height"] for child in children) + gap * (len(children) - 1)
    if total_h > avail_h:
        return False
    cursor = inner_top + rng.randint(0, avail_h - total_h)
    for child in children:
        dx = avail_w - child["width"]
        if dx < 0:
            return False
        child_left = inner_left + rng.randint(0, dx)
        if not _place_node(child, cursor, child_left, rng):
            return False
        cursor += child["height"] + gap
    return True


def _layout_scene(roots: list[dict], rng: random.Random) -> tuple[int, int] | None:
    for _ in range(60):
        top_margin = rng.randint(1, 4)
        bottom_margin = rng.randint(1, 4)
        left_margin = rng.randint(1, 4)
        right_margin = rng.randint(1, 4)
        extra_h = rng.randint(0, 3)
        extra_w = rng.randint(0, 3)

        if len(roots) == 1:
            root = roots[0]
            h = top_margin + bottom_margin + root["height"] + extra_h
            w = left_margin + right_margin + root["width"] + extra_w
            if not (MIN_SCENE_SIDE <= h <= MAX_SCENE_SIDE and MIN_SCENE_SIDE <= w <= MAX_SCENE_SIDE):
                continue
            root_top = top_margin + rng.randint(0, extra_h)
            root_left = left_margin + rng.randint(0, extra_w)
            if _place_node(root, root_top, root_left, rng):
                return h, w
            continue

        orientation = rng.choice(["horizontal", "vertical"])
        gap = rng.randint(2, 4)
        a, b = roots

        if orientation == "horizontal":
            h = top_margin + bottom_margin + max(a["height"], b["height"]) + extra_h
            w = left_margin + right_margin + a["width"] + b["width"] + gap + extra_w
            if not (MIN_SCENE_SIDE <= h <= MAX_SCENE_SIDE and MIN_SCENE_SIDE <= w <= MAX_SCENE_SIDE):
                continue
            cursor = left_margin + rng.randint(0, extra_w)
            a_top = top_margin + rng.randint(0, h - top_margin - bottom_margin - a["height"])
            b_top = top_margin + rng.randint(0, h - top_margin - bottom_margin - b["height"])
            if _place_node(a, a_top, cursor, rng) and _place_node(b, b_top, cursor + a["width"] + gap, rng):
                return h, w
            continue

        h = top_margin + bottom_margin + a["height"] + b["height"] + gap + extra_h
        w = left_margin + right_margin + max(a["width"], b["width"]) + extra_w
        if not (MIN_SCENE_SIDE <= h <= MAX_SCENE_SIDE and MIN_SCENE_SIDE <= w <= MAX_SCENE_SIDE):
            continue
        cursor = top_margin + rng.randint(0, extra_h)
        a_left = left_margin + rng.randint(0, w - left_margin - right_margin - a["width"])
        b_left = left_margin + rng.randint(0, w - left_margin - right_margin - b["width"])
        if _place_node(a, cursor, a_left, rng) and _place_node(b, cursor + a["height"] + gap, b_left, rng):
            return h, w

    return None


def _render_node(grid: np.ndarray, node: dict) -> None:
    r0, c0, r1, c1 = node["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    sub[node["mask"]] = int(node["color"])
    for child in node["children"]:
        _render_node(grid, child)


def _expected_output(nodes: list[dict]) -> np.ndarray:
    ordered = sorted(
        nodes,
        key=lambda node: (
            int(node["depth"]),
            -int(node["area"]),
            int(node["bbox"][0]),
            int(node["bbox"][1]),
            int(node["height"]),
            int(node["width"]),
        ),
    )
    return np.asarray([[int(node["color"]) for node in ordered]], dtype=int)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    roots = _build_skeleton(rng)
    nodes = _collect_nodes(roots)
    bg = rng.randrange(10)
    _assign_colors(nodes, bg, rng)
    for root in roots:
        _assign_sizes(root, rng)

    scene_shape = _layout_scene(roots, rng)
    if scene_shape is None:
        return None

    h, w = scene_shape
    grid = np.full((h, w), bg, dtype=int)
    for root in roots:
        _render_node(grid, root)

    total_fg = sum(int(node["area"]) for node in nodes)
    bg_count = int(grid.size - total_fg)
    if bg_count <= max(int(node["area"]) for node in nodes):
        return None

    counts = Counter(int(v) for v in grid.ravel())
    most_common = counts.most_common(2)
    if int(most_common[0][0]) != bg:
        return None
    if len(most_common) > 1 and int(most_common[0][1]) == int(most_common[1][1]):
        return None

    objects = _extract_objects(grid, bg)
    if len(objects) != len(nodes):
        return None

    expected = _expected_output(nodes)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
