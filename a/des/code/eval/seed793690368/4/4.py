import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

MAX_GRID_SIZE = 30
MAX_PATCH_DIM = 24
MAX_GENERATION_ATTEMPTS = 400
ANALYSIS_CONN = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ANALYSIS_CONN:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _is_rect_ring(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(component["coords"]) == expected


def _contains(parent: dict, child: dict) -> bool:
    pr0, pc0, pr1, pc1 = parent["bbox"]
    cr0, cc0, cr1, cc1 = child["bbox"]
    return pr0 < cr0 and pc0 < cc0 and cr1 < pr1 and cc1 < pc1


def _ring_area_key(ring: dict) -> tuple[int, int, tuple[int, int, int, int]]:
    return (int(ring["box_area"]), int(ring["inner_area"]), tuple(ring["bbox"]))


def _analyze_rings(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _same_color_components(g, bg)

    rings: list[dict] = []
    for comp in components:
        if not _is_rect_ring(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rings.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "box_area": h * w,
                "inner_area": (h - 2) * (w - 2),
                "parent": None,
                "children": [],
            }
        )

    for child_idx, child in enumerate(rings):
        containers = [i for i, parent in enumerate(rings) if i != child_idx and _contains(parent, child)]
        if containers:
            parent_idx = min(containers, key=lambda i: _ring_area_key(rings[i]))
            child["parent"] = parent_idx
            rings[parent_idx]["children"].append(child_idx)

    for ring in rings:
        ring["children"].sort(key=lambda i: _ring_area_key(rings[i]))
    return bg, rings, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular rings based on child containment.

    Leaf rings nested inside a parent fill their own interiors. Rings with
    direct children fill the background part of their interior with the color
    of the smallest direct child. Top-level isolated rings stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, rings, _ = _analyze_rings(g)
    out = g.copy()

    for ring in sorted(rings, key=_ring_area_key):
        r0, c0, r1, c1 = ring["bbox"]
        region = np.zeros_like(out, dtype=bool)
        region[r0 + 1 : r1, c0 + 1 : c1] = True

        if ring["children"]:
            fill_color = int(rings[ring["children"][0]]["color"])
            for child_idx in ring["children"]:
                cr0, cc0, cr1, cc1 = rings[child_idx]["bbox"]
                region[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
            out[region & (out == bg)] = fill_color
        elif ring["parent"] is not None:
            out[region & (out == bg)] = int(ring["color"])

    return out


def _shift_rings(rings: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rings]


def _make_leaf(rng: random.Random) -> dict:
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    return {"h": h, "w": w, "rings": [(0, 0, h - 1, w - 1)]}


def _wrap_subtree(rng: random.Random, child: dict) -> dict | None:
    for _ in range(40):
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        left = rng.randint(1, 2)
        right = rng.randint(1, 2)
        h = child["h"] + top + bottom + 2
        w = child["w"] + left + right + 2
        if h > MAX_PATCH_DIM or w > MAX_PATCH_DIM:
            continue
        rings = [(0, 0, h - 1, w - 1)]
        rings.extend(_shift_rings(child["rings"], 1 + top, 1 + left))
        return {"h": h, "w": w, "rings": rings}
    return None


def _make_chain(rng: random.Random, depth: int) -> dict | None:
    tree = _make_leaf(rng)
    for _ in range(depth - 1):
        wrapped = _wrap_subtree(rng, tree)
        if wrapped is None:
            return None
        tree = wrapped
    return tree


def _compose_branch(rng: random.Random, children: list[dict], orientation: str) -> dict | None:
    pad_t = rng.randint(1, 2)
    pad_b = rng.randint(1, 2)
    pad_l = rng.randint(1, 2)
    pad_r = rng.randint(1, 2)
    gap = rng.randint(1, 2)

    if orientation == "horizontal":
        content_h = max(child["h"] for child in children)
        content_w = sum(child["w"] for child in children) + gap * (len(children) - 1)
        h = content_h + pad_t + pad_b + 2
        w = content_w + pad_l + pad_r + 2
        if h > MAX_PATCH_DIM or w > MAX_PATCH_DIM:
            return None
        rings = [(0, 0, h - 1, w - 1)]
        cur_c = 1 + pad_l
        for child in children:
            offset_r = 1 + pad_t + rng.randint(0, content_h - child["h"])
            rings.extend(_shift_rings(child["rings"], offset_r, cur_c))
            cur_c += child["w"] + gap
        return {"h": h, "w": w, "rings": rings}

    content_h = sum(child["h"] for child in children) + gap * (len(children) - 1)
    content_w = max(child["w"] for child in children)
    h = content_h + pad_t + pad_b + 2
    w = content_w + pad_l + pad_r + 2
    if h > MAX_PATCH_DIM or w > MAX_PATCH_DIM:
        return None
    rings = [(0, 0, h - 1, w - 1)]
    cur_r = 1 + pad_t
    for child in children:
        offset_c = 1 + pad_l + rng.randint(0, content_w - child["w"])
        rings.extend(_shift_rings(child["rings"], cur_r, offset_c))
        cur_r += child["h"] + gap
    return {"h": h, "w": w, "rings": rings}


def _make_branch(rng: random.Random) -> dict | None:
    for _ in range(80):
        depths = [rng.choice((1, 2, 2, 3)), rng.choice((1, 1, 2, 3))]
        children = []
        ok = True
        for depth in depths:
            child = _make_chain(rng, depth)
            if child is None:
                ok = False
                break
            children.append(child)
        if not ok:
            continue
        if len({child["h"] * child["w"] for child in children}) != len(children):
            continue

        orientations = ["horizontal", "vertical"]
        rng.shuffle(orientations)
        branch = None
        for orientation in orientations:
            branch = _compose_branch(rng, children, orientation)
            if branch is not None:
                break
        if branch is None:
            continue

        if rng.random() < 0.45:
            wrapped = _wrap_subtree(rng, branch)
            if wrapped is not None and len(wrapped["rings"]) <= 7:
                branch = wrapped

        return branch
    return None


def _pack_patches(rng: random.Random, patches: list[dict]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    order = patches[:]
    rng.shuffle(order)

    gaps_row = [rng.randint(2, 4) for _ in range(max(0, len(order) - 1))]
    total_w = sum(patch["w"] for patch in order) + sum(gaps_row)
    max_h = max(patch["h"] for patch in order)
    if total_w + 2 <= MAX_GRID_SIZE and max_h + 2 <= MAX_GRID_SIZE:
        w = rng.randint(total_w + 2, MAX_GRID_SIZE)
        h = rng.randint(max_h + 2, MAX_GRID_SIZE)
        left_margin = rng.randint(1, w - total_w - 1)
        positions = []
        cur_c = left_margin
        for idx, patch in enumerate(order):
            row = rng.randint(1, h - patch["h"] - 1)
            positions.append((row, cur_c))
            cur_c += patch["w"]
            if idx < len(gaps_row):
                cur_c += gaps_row[idx]
        original_positions = {id(patch): pos for patch, pos in zip(order, positions)}
        return (h, w), [original_positions[id(patch)] for patch in patches]

    gaps_col = [rng.randint(2, 4) for _ in range(max(0, len(order) - 1))]
    total_h = sum(patch["h"] for patch in order) + sum(gaps_col)
    max_w = max(patch["w"] for patch in order)
    if total_h + 2 <= MAX_GRID_SIZE and max_w + 2 <= MAX_GRID_SIZE:
        h = rng.randint(total_h + 2, MAX_GRID_SIZE)
        w = rng.randint(max_w + 2, MAX_GRID_SIZE)
        top_margin = rng.randint(1, h - total_h - 1)
        positions = []
        cur_r = top_margin
        for idx, patch in enumerate(order):
            col = rng.randint(1, w - patch["w"] - 1)
            positions.append((cur_r, col))
            cur_r += patch["h"]
            if idx < len(gaps_col):
                cur_r += gaps_col[idx]
        original_positions = {id(patch): pos for patch, pos in zip(order, positions)}
        return (h, w), [original_positions[id(patch)] for patch in patches]

    return None


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _generated_scene_ok(grid: np.ndarray) -> bool:
    bg, rings, components = _analyze_rings(grid)
    if bg != BG:
        return False
    if len(components) != len(rings):
        return False
    if len(rings) < 4:
        return False

    has_independent = any(ring["parent"] is None and not ring["children"] for ring in rings)
    has_nested_leaf = any(ring["parent"] is not None and not ring["children"] for ring in rings)
    has_parent = any(ring["children"] for ring in rings)
    has_choice_parent = any(len(ring["children"]) >= 2 for ring in rings)
    if not (has_independent and has_nested_leaf and has_parent and has_choice_parent):
        return False

    for ring in rings:
        if len(ring["children"]) >= 2:
            child_areas = [rings[idx]["box_area"] for idx in ring["children"]]
            if len(child_areas) != len(set(child_areas)):
                return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = _make_branch(rng)
    if branch is None:
        return None

    independent = _make_leaf(rng)
    patches = [branch, independent]

    extra_budget = 9 - sum(len(patch["rings"]) for patch in patches)
    if extra_budget >= 2 and rng.random() < 0.6:
        extra = _make_leaf(rng) if rng.random() < 0.5 else _make_chain(rng, 2)
        if extra is not None and len(extra["rings"]) <= extra_budget:
            patches.append(extra)

    packed = _pack_patches(rng, patches)
    if packed is None:
        return None
    (h, w), positions = packed

    ring_bboxes: list[tuple[int, int, int, int]] = []
    for patch, (dr, dc) in zip(patches, positions):
        ring_bboxes.extend(_shift_rings(patch["rings"], dr, dc))

    if len(ring_bboxes) > len(PALETTE):
        return None

    colors = rng.sample(PALETTE, len(ring_bboxes))
    grid = np.full((h, w), BG, dtype=int)
    for bbox, color in zip(ring_bboxes, colors):
        _draw_ring(grid, bbox, color)

    if not _generated_scene_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
