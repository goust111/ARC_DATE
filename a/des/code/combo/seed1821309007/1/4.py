import random
from collections import Counter, deque

import numpy as np


BG = 8
COUNT_COLORS = {1: 2, 2: 3, 3: 4, 4: 6}
PALETTE = [c for c in range(10) if c != BG]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    return comps


def bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def build_tree(components: list[dict]) -> tuple[list[int], list[int | None], list[list[int]]]:
    n = len(components)
    parents: list[int | None] = [None] * n
    for i, comp in enumerate(components):
        bbox = comp["bbox"]
        candidates = []
        for j, maybe_parent in enumerate(components):
            if i == j:
                continue
            parent_bbox = maybe_parent["bbox"]
            if bbox_contains(parent_bbox, bbox):
                area = (parent_bbox[2] - parent_bbox[0] + 1) * (parent_bbox[3] - parent_bbox[1] + 1)
                candidates.append((area, j))
        if candidates:
            parents[i] = min(candidates)[1]

    children: list[list[int]] = [[] for _ in range(n)]
    roots = []
    for i, parent in enumerate(parents):
        if parent is None:
            roots.append(i)
        else:
            children[parent].append(i)
    return roots, parents, children


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor bbox-nested objects as a tree.

    - Direct parent/child links come from strict bbox containment with no
      smaller containing bbox in between.
    - Leaves inherit their parent's original color.
    - Non-leaves are recolored by direct-child count:
      1 -> red, 2 -> green, 3 -> yellow, 4 -> magenta.
    - A root with no children keeps its original color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    roots, parents, children = build_tree(components)

    out = g.copy()
    original_colors = [int(comp["color"]) for comp in components]

    for i, comp in enumerate(components):
        child_count = len(children[i])
        if child_count > 0:
            new_color = COUNT_COLORS[child_count]
        elif parents[i] is not None:
            new_color = original_colors[parents[i]]
        else:
            new_color = original_colors[i]
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)

    return out


def _make_leaf_patch(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    return np.ones((h, w), dtype=int)


def _partition_sizes(total: int, parts: int, gap: int, min_size: int = 1) -> list[int]:
    remaining = total - gap * (parts - 1) - min_size * parts
    sizes = [min_size] * parts
    for _ in range(remaining):
        sizes[random.randrange(parts)] += 1
    return sizes


def _build_node(rng: random.Random, depth: int, max_depth: int, force_internal: bool) -> dict:
    make_leaf = (depth >= max_depth) or (not force_internal and depth > 0 and rng.random() < 0.45)
    if make_leaf:
        patch = _make_leaf_patch(rng)
        return {
            "children": [],
            "patch": patch,
            "h": int(patch.shape[0]),
            "w": int(patch.shape[1]),
        }

    child_count = rng.randint(1, 4)
    children = [
        _build_node(rng, depth + 1, max_depth, force_internal=(depth + 1 < max_depth and rng.random() < 0.35))
        for _ in range(child_count)
    ]
    layout = rng.choice(["row", "col"])
    gap = rng.randint(1, 2)
    margin = rng.randint(1, 2)

    if layout == "row":
        content_h = max(child["h"] for child in children)
        total_w = sum(child["w"] for child in children) + gap * (child_count - 1)
        extra_h = rng.randint(0, 2)
        h = content_h + 2 * (margin + 1) + extra_h
        w = total_w + 2 * (margin + 1)
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = 1
        patch[-1, :] = 1
        patch[:, 0] = 1
        patch[:, -1] = 1
        cur_c = margin + 1
        placements = []
        for child in children:
            free = h - 2 * (margin + 1) - child["h"]
            off_r = margin + 1 + rng.randint(0, max(0, free))
            placements.append((off_r, cur_c, child))
            cur_c += child["w"] + gap
    else:
        content_w = max(child["w"] for child in children)
        total_h = sum(child["h"] for child in children) + gap * (child_count - 1)
        extra_w = rng.randint(0, 2)
        h = total_h + 2 * (margin + 1)
        w = content_w + 2 * (margin + 1) + extra_w
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = 1
        patch[-1, :] = 1
        patch[:, 0] = 1
        patch[:, -1] = 1
        cur_r = margin + 1
        placements = []
        for child in children:
            free = w - 2 * (margin + 1) - child["w"]
            off_c = margin + 1 + rng.randint(0, max(0, free))
            placements.append((cur_r, off_c, child))
            cur_r += child["h"] + gap

    for off_r, off_c, child in placements:
        child["offset"] = (off_r, off_c)

    return {"children": children, "patch": patch, "h": h, "w": w}


def _assign_colors(node: dict, colors: list[int], index_ref: list[int]) -> None:
    node["color"] = colors[index_ref[0]]
    index_ref[0] += 1
    for child in node["children"]:
        _assign_colors(child, colors, index_ref)


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(child) for child in node["children"])


def _render_node(grid: np.ndarray, top: int, left: int, node: dict) -> None:
    coords = np.argwhere(node["patch"] == 1)
    for rr, cc in coords:
        grid[top + int(rr), left + int(cc)] = int(node["color"])
    for child in node["children"]:
        off_r, off_c = child["offset"]
        _render_node(grid, top + off_r, left + off_c, child)


def _sample_roots(rng: random.Random) -> list[dict]:
    roots = []
    tree_roots = rng.randint(1, 2)
    include_isolated = rng.random() < 0.5
    for _ in range(tree_roots):
        roots.append(_build_node(rng, depth=0, max_depth=rng.randint(1, 3), force_internal=True))
    if include_isolated:
        for _ in range(rng.randint(1, 2)):
            roots.append(_build_node(rng, depth=0, max_depth=0, force_internal=False))
    rng.shuffle(roots)
    return roots


def _place_roots(rng: random.Random, roots: list[dict]) -> tuple[int, int, list[tuple[int, int]]]:
    max_w = max(root["w"] for root in roots)
    total_h = sum(root["h"] for root in roots) + 3 * (len(roots) - 1)
    max_h = max(root["h"] for root in roots)
    total_w = sum(root["w"] for root in roots) + 3 * (len(roots) - 1)

    if rng.random() < 0.5:
        canvas_h = total_h + rng.randint(4, 10)
        canvas_w = max(max_w + rng.randint(4, 12), 18)
        positions = []
        cur_r = rng.randint(1, 3)
        for root in roots:
            left = rng.randint(1, max(1, canvas_w - root["w"] - 2))
            positions.append((cur_r, left))
            cur_r += root["h"] + rng.randint(2, 4)
        if cur_r + 1 > canvas_h:
            raise ValueError("canvas too short")
    else:
        canvas_h = max(max_h + rng.randint(4, 12), 18)
        canvas_w = total_w + rng.randint(4, 10)
        positions = []
        cur_c = rng.randint(1, 3)
        for root in roots:
            top = rng.randint(1, max(1, canvas_h - root["h"] - 2))
            positions.append((top, cur_c))
            cur_c += root["w"] + rng.randint(2, 4)
        if cur_c + 1 > canvas_w:
            raise ValueError("canvas too narrow")

    if canvas_h > 30 or canvas_w > 30:
        raise ValueError("canvas too large")
    return canvas_h, canvas_w, positions


def _validate_generated(grid: np.ndarray) -> bool:
    components = find_components(grid, BG)
    if not components:
        return False
    colors = [int(comp["color"]) for comp in components]
    if len(colors) != len(set(colors)):
        return False
    roots, parents, children = build_tree(components)
    if not roots:
        return False
    for i, parent in enumerate(parents):
        if parent is not None and len(children[parent]) > 4:
            return False
        if parent is not None:
            pb = components[parent]["bbox"]
            b = components[i]["bbox"]
            if not bbox_contains(pb, b):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    roots = _sample_roots(rng)
    total_nodes = sum(_count_nodes(root) for root in roots)
    if total_nodes > len(PALETTE):
        return None

    try:
        h, w, positions = _place_roots(rng, roots)
    except ValueError:
        return None

    colors = rng.sample(PALETTE, total_nodes)
    idx = [0]
    for root in roots:
        _assign_colors(root, colors, idx)

    grid = np.full((h, w), BG, dtype=int)
    for root, (top, left) in zip(roots, positions):
        if top + root["h"] > h or left + root["w"] > w:
            return None
        _render_node(grid, top, left, root)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
