from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _inside(outer: BBox, inner: BBox) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(inner[2]) < int(outer[2])
        and int(inner[3]) < int(outer[3])
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(pts, dtype=int))
    return out


def _border_set(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _extract_objects(g: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in [int(c) for c in np.unique(g) if int(c) != int(bg)]:
        for coords in _components(g == int(color)):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(coords.shape[0])
            if area == h * w:
                kind = "solid"
            else:
                assert h >= 3 and w >= 3, "rectangular frames need a real interior"
                assert {tuple(map(int, p)) for p in coords} == _border_set(box), (
                    "foreground objects must be rectangular frames or solid rectangles"
                )
                kind = "frame"
            objects.append(
                {
                    "id": -1,
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "kind": kind,
                    "area": _bbox_area(box),
                }
            )

    objects.sort(key=lambda obj: (-int(obj["area"]), obj["bbox"]))
    for i, obj in enumerate(objects):
        obj["id"] = int(i)
    return objects


def _linear_chain(root: int, objects: list[dict], children: list[list[int]]) -> list[int] | None:
    chain: list[int] = []
    cur = int(root)
    while True:
        chain.append(cur)
        kids = children[cur]
        if not kids:
            if objects[cur]["kind"] != "solid" or len(chain) < 3:
                return None
            colors = [int(objects[i]["color"]) for i in chain]
            if len(colors) != len(set(colors)):
                return None
            return chain
        if objects[cur]["kind"] != "frame" or len(kids) != 1:
            return None
        cur = int(kids[0])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    assert len(objects) >= 3, "expected nested rectangular foreground objects"

    n = len(objects)
    parents = [-1] * n
    for i, obj in enumerate(objects):
        inner = obj["bbox"]
        containers = [j for j, other in enumerate(objects) if i != j and _inside(other["bbox"], inner)]
        if containers:
            parents[i] = min(containers, key=lambda j: int(objects[j]["area"]))

    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    for i, obj in enumerate(objects):
        assert not children[i] or obj["kind"] == "frame", "solid blocks cannot contain children"

    roots = [i for i, p in enumerate(parents) if p < 0]
    assert roots, "expected at least one containment tree"

    depths = [0] * n

    def visit(node: int, depth: int) -> None:
        assert depths[node] == 0, "containment tree must be acyclic"
        depths[node] = int(depth)
        for child in children[node]:
            visit(int(child), int(depth) + 1)

    for root in roots:
        visit(int(root), 1)

    candidates = []
    for root in roots:
        chain = _linear_chain(int(root), objects, children)
        if chain is not None:
            candidates.append(chain)
    assert len(candidates) == 1, "expected exactly one strict linear codebook chain"

    code_chain = candidates[0]
    code_colors = [int(objects[i]["color"]) for i in code_chain]
    code_ids = set(code_chain)
    for i in range(n):
        if i not in code_ids:
            assert int(depths[i]) <= len(code_colors), "codebook must cover every target depth"

    return {
        "grid": g,
        "bg": int(bg),
        "objects": objects,
        "parents": parents,
        "children": children,
        "roots": roots,
        "depths": depths,
        "code_chain": code_chain,
        "code_colors": code_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the only one-child-per-layer nested chain as a depth-to-color codebook, then erase it.

    Input structure:
    - The dominant background surrounds several trees of closed rectangular frames, with solid terminal blocks allowed as leaves.
    - Containment is defined by strict bbox inclusion: a direct child is the smallest object bbox inside its parent frame.
    - Exactly one root tree is a strict linear chain whose non-leaf nodes each have one child, whose leaf is solid, and whose colors are all distinct.
    - Every other containment tree branches at some level and uses distracting, non-systematic colors.

    Rule:
    - Read the codebook chain from outside to inside; its depth d color is C(d), with the outermost object at depth 1.
    - For every object outside the codebook tree, recolor its existing cells to C(d), where d is its depth in its own tree.
    - Replace every cell belonging to the codebook chain with the background color.

    Key insight:
    - The legend is disguised as an ordinary nested scene object; only the containment topology, not size, position, or original color, identifies it.
    """
    parsed = _analyze(grid)
    g = parsed["grid"]
    bg = int(parsed["bg"])
    objects = parsed["objects"]
    depths = parsed["depths"]
    code_chain = set(parsed["code_chain"])
    code_colors = parsed["code_colors"]

    out = g.copy()
    for i, obj in enumerate(objects):
        coords = obj["coords"]
        if i in code_chain:
            color = bg
        else:
            color = int(code_colors[int(depths[i]) - 1])
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _solid_structure() -> dict:
    h, w = random.choice(((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)))
    return {
        "height": int(h),
        "width": int(w),
        "objects": [{"bbox": (0, 0, int(h) - 1, int(w) - 1), "kind": "solid", "depth": 1}],
    }


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _margin() -> int:
    return 2 + (1 if random.random() < 0.2 else 0)


def _wrap_structure(child: dict) -> dict:
    top = _margin()
    bottom = _margin()
    left = _margin()
    right = _margin()
    h = int(child["height"]) + top + bottom
    w = int(child["width"]) + left + right
    objects = [{"bbox": (0, 0, h - 1, w - 1), "kind": "frame", "depth": 1}]
    for obj in child["objects"]:
        objects.append(
            {
                "bbox": _shift_box(obj["bbox"], top, left),
                "kind": obj["kind"],
                "depth": int(obj["depth"]) + 1,
            }
        )
    return {"height": int(h), "width": int(w), "objects": objects}


def _linear_structure(depth: int) -> dict:
    assert int(depth) >= 1
    struct = _solid_structure()
    for _ in range(int(depth) - 1):
        struct = _wrap_structure(struct)
    return struct


def _combine_under_branch_frame(a: dict, b: dict) -> dict:
    if random.random() < 0.5:
        a, b = b, a
    gap = random.choice((2, 2, 3))
    top = _margin()
    bottom = _margin()
    left = _margin()
    right = _margin()
    orientation = random.choice(("h", "v"))

    if orientation == "h":
        inner_h = max(int(a["height"]), int(b["height"]))
        h = top + inner_h + bottom
        w = left + int(a["width"]) + gap + int(b["width"]) + right
        ar = top + random.randint(0, inner_h - int(a["height"]))
        br = top + random.randint(0, inner_h - int(b["height"]))
        ac = left
        bc = left + int(a["width"]) + gap
    else:
        inner_w = max(int(a["width"]), int(b["width"]))
        h = top + int(a["height"]) + gap + int(b["height"]) + bottom
        w = left + inner_w + right
        ar = top
        br = top + int(a["height"]) + gap
        ac = left + random.randint(0, inner_w - int(a["width"]))
        bc = left + random.randint(0, inner_w - int(b["width"]))

    objects = [{"bbox": (0, 0, h - 1, w - 1), "kind": "frame", "depth": 1}]
    for child, dr, dc in ((a, ar, ac), (b, br, bc)):
        for obj in child["objects"]:
            objects.append(
                {
                    "bbox": _shift_box(obj["bbox"], int(dr), int(dc)),
                    "kind": obj["kind"],
                    "depth": int(obj["depth"]) + 1,
                }
            )
    return {"height": int(h), "width": int(w), "objects": objects}


def _branch_structure(total_depth: int, branch_depth: int) -> dict:
    assert 1 <= int(branch_depth) < int(total_depth)
    deep = _linear_structure(int(total_depth) - int(branch_depth))
    side_depth = random.randint(1, int(total_depth) - int(branch_depth))
    side = _linear_structure(side_depth)
    struct = _combine_under_branch_frame(deep, side)
    for _ in range(int(branch_depth) - 1):
        struct = _wrap_structure(struct)
    return struct


def _paint_object(patch: np.ndarray, obj: dict, color: int) -> None:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    if obj["kind"] == "solid":
        patch[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    else:
        patch[r0, c0 : c1 + 1] = int(color)
        patch[r1, c0 : c1 + 1] = int(color)
        patch[r0 : r1 + 1, c0] = int(color)
        patch[r0 : r1 + 1, c1] = int(color)


def _render_structure(struct: dict, bg: int, code_colors: list[int] | None = None) -> np.ndarray:
    patch = np.full((int(struct["height"]), int(struct["width"])), int(bg), dtype=int)
    for obj in struct["objects"]:
        depth = int(obj["depth"])
        if code_colors is None:
            blocked = int(_CURRENT_CODE_COLORS[depth - 1])
            choices = [c for c in PALETTE if c != int(bg) and c != blocked]
            color = random.choice(choices)
        else:
            color = int(code_colors[depth - 1])
        _paint_object(patch, obj, int(color))
    return patch


_CURRENT_CODE_COLORS: list[int] = []


def _boxes_overlap(a: BBox, b: BBox, padding: int = 1) -> bool:
    pad = int(padding)
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _place_patches(patches: list[np.ndarray], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    used: list[BBox] = []
    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, int(h) - int(ph)):
            for c in range(1, int(w) - int(pw)):
                box = (int(r), int(c), int(r + ph - 1), int(c + pw - 1))
                if not any(_boxes_overlap(box, old, padding=1) for old in used):
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (int(r), int(c))
        used.append((int(r), int(c), int(r + ph - 1), int(c + pw - 1)))
    return [p for p in placements if p is not None]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r: int, c: int, bg: int) -> None:
    target = grid[int(r) : int(r) + patch.shape[0], int(c) : int(c) + patch.shape[1]]
    mask = patch != int(bg)
    target[mask] = patch[mask]


def _tree_nodes(root: int, children: list[list[int]]) -> set[int]:
    found: set[int] = set()
    stack = [int(root)]
    while stack:
        node = stack.pop()
        if node in found:
            continue
        found.add(int(node))
        stack.extend(int(k) for k in children[node])
    return found


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        parsed = _analyze(grid)
        out = main(grid)
    except AssertionError:
        return False

    bg = int(parsed["bg"])
    objects = parsed["objects"]
    children = parsed["children"]
    roots = parsed["roots"]
    depths = parsed["depths"]
    code_chain = set(parsed["code_chain"])
    code_root = int(parsed["code_chain"][0])
    code_colors = parsed["code_colors"]
    non_code_roots = [int(r) for r in roots if int(r) != code_root]
    if len(non_code_roots) < 2:
        return False

    has_root_branch = any(len(children[root]) > 1 for root in non_code_roots)
    has_deep_branch = any(
        any(node != root and len(children[node]) > 1 for node in _tree_nodes(root, children))
        for root in non_code_roots
    )
    if not (has_root_branch and has_deep_branch):
        return False

    for root in non_code_roots:
        if not any(len(children[node]) > 1 for node in _tree_nodes(root, children)):
            return False

    target_ids = [i for i in range(len(objects)) if i not in code_chain]
    depth_set = {int(depths[i]) for i in target_ids}
    if depth_set != set(range(1, len(code_colors) + 1)):
        return False

    changed_depths: set[int] = set()
    for i, obj in enumerate(objects):
        coords = obj["coords"]
        values = out[coords[:, 0], coords[:, 1]]
        if i in code_chain:
            if not bool(np.all(values == bg)):
                return False
            continue
        expected = int(code_colors[int(depths[i]) - 1])
        if not bool(np.all(values == expected)):
            return False
        original = grid[coords[:, 0], coords[:, 1]]
        if bool(np.any(original != expected)):
            changed_depths.add(int(depths[i]))

    return changed_depths == set(range(1, len(code_colors) + 1))


def _gen_attempt() -> np.ndarray | None:
    global _CURRENT_CODE_COLORS

    bg = random.choice(PALETTE)
    depth = random.choice((3, 3, 4))
    code_colors = random.sample([c for c in PALETTE if c != int(bg)], int(depth))
    _CURRENT_CODE_COLORS = list(map(int, code_colors))

    codebook = _linear_structure(depth)
    root_branch = _branch_structure(depth, 1)
    deep_total_depth = 3 if depth == 3 else random.choice((3, 4))
    deep_branch_depth = random.randint(2, deep_total_depth - 1)
    deep_branch = _branch_structure(deep_total_depth, deep_branch_depth)

    patches = [
        _render_structure(codebook, bg, code_colors),
        _render_structure(root_branch, bg, None),
        _render_structure(deep_branch, bg, None),
    ]
    if any(max(p.shape) > 26 for p in patches):
        return None

    for _ in range(30):
        min_h = max(int(p.shape[0]) for p in patches) + 2
        min_w = max(int(p.shape[1]) for p in patches) + 2
        h = random.randint(max(22, min_h), 30)
        w = random.randint(max(22, min_w), 30)
        placements = _place_patches(patches, h, w)
        if placements is None:
            continue
        grid = np.full((h, w), int(bg), dtype=int)
        order = list(range(len(patches)))
        random.shuffle(order)
        for idx in order:
            r, c = placements[idx]
            _paste_patch(grid, patches[idx], r, c, bg)
        if _get_bg(grid) != int(bg):
            continue
        if _semantic_check(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
