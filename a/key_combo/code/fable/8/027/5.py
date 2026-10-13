from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, find_components, paste, random_free_location


BG = Color.BLACK
FG_COLORS = tuple(Color.NOT_BLACK)
MAX_ATTEMPTS = 400


@dataclass
class BranchBundle:
    patch: np.ndarray
    leaf_coords: np.ndarray
    used_colors: set[int]
    depth: int


@dataclass
class EntityBundle:
    input_patch: np.ndarray
    output_patch: np.ndarray


def _bbox_area(b: tuple[int, int, int, int]) -> int:
    return bbox_h(b) * bbox_w(b)


def _bbox_contains_strict(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _subtree_nodes(root: int, children: list[list[int]]) -> list[int]:
    out = [root]
    for child in children[root]:
        out.extend(_subtree_nodes(child, children))
    return out


def _build_forest(objects: list[dict]) -> tuple[list[int | None], list[list[int]], list[int]]:
    parents: list[int | None] = [None] * len(objects)
    for i, obj in enumerate(objects):
        containers = []
        for j, other in enumerate(objects):
            if i == j:
                continue
            if _bbox_contains_strict(other["bbox"], obj["bbox"]):
                containers.append(( _bbox_area(other["bbox"]), int(other["area"]), j))
        if containers:
            containers.sort()
            parents[i] = containers[0][2]
    children = [[] for _ in objects]
    for child, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(child)
    roots = [i for i, parent in enumerate(parents) if parent is None]
    return parents, children, roots


def _collapse_tree_patch(
    shape: tuple[int, int],
    root_bbox: tuple[int, int, int, int],
    root_color: int,
    leaf_coords: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    group_bbox = bbox_coords(leaf_coords)
    gr0, gc0, _gr1, _gc1 = group_bbox
    group_h = bbox_h(group_bbox)
    group_w = bbox_w(group_bbox)
    group_patch = np.zeros((group_h, group_w), dtype=int)
    rel = leaf_coords - np.array([gr0, gc0], dtype=int)
    group_patch[rel[:, 0], rel[:, 1]] = int(root_color)
    rr0, rc0, rr1, rc1 = root_bbox
    dest_r = rr0 + (bbox_h(root_bbox) - group_h) // 2
    dest_c = rc0 + (bbox_w(root_bbox) - group_w) // 2
    assert rr0 <= dest_r and dest_r + group_h - 1 <= rr1
    assert rc0 <= dest_c and dest_c + group_w - 1 <= rc1
    paste(out, group_patch, dest_r, dest_c, transparent=BG)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Collapse each bbox-nesting tree into innermost shapes recolored by the outermost shell and recentered in that shell.

    Input structure:
    - Black is the background, and every non-black 8-connected monochromatic component is one object.
    - Some objects form strict bounding-box containment trees: a larger object's bbox strictly contains one or more smaller-object bboxes, possibly through multiple layers.
    - Other objects are independent and do not participate in any containment relation.

    Rule:
    - Compute all object bboxes and connect objects by immediate strict bbox containment to form nesting trees.
    - For every tree with at least two objects, erase every layer, keep only the innermost leaf shapes, recolor those leaves to the tree root's color, preserve the leaves' relative offsets, and center the whole leaf group inside the root bbox.
    - Leave every independent non-nested object unchanged.

    Key insight:
    - The output mixes three sources: leaf geometry, root color, and root-centered placement, so intermediate shells are only structural clues and must disappear.
    """
    g = np.asarray(input_grid, dtype=int)
    objects = find_components(g, bg=BG, connectivity=8, monochromatic=True, return_format="dict")
    if not objects:
        return g.copy()

    _parents, children, roots = _build_forest(objects)
    out = g.copy()

    for root in roots:
        nodes = _subtree_nodes(root, children)
        if len(nodes) == 1:
            continue

        for idx in nodes:
            coords = objects[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

        leaves = [idx for idx in nodes if not children[idx]]
        leaf_coords = np.concatenate([objects[idx]["coords"] for idx in leaves], axis=0)
        collapsed = _collapse_tree_patch(g.shape, objects[root]["bbox"], int(objects[root]["color"]), leaf_coords)
        paste(out, collapsed, 0, 0, transparent=BG)

    return out


def _choose_color(rng: np.random.Generator, forbidden: set[int]) -> int:
    choices = [c for c in FG_COLORS if c not in forbidden]
    if not choices:
        raise ValueError("ran out of distinct colors")
    return int(rng.choice(choices))


def _neighbors8(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _random_leaf_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    min_cells = max(3, (h * w + 2) // 3)
    max_cells = max(min_cells, h * w - 1)
    for _ in range(300):
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        target = int(rng.integers(min_cells, max_cells + 1))
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                frontier.update(_neighbors8(r, c, h, w))
            frontier.difference_update(cells)
            if not frontier:
                break

            def _score(cell: tuple[int, int]) -> tuple[int, float]:
                r, c = cell
                touches = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
                return touches, float(rng.random())

            cell = max(frontier, key=_score)
            cells.add(cell)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.all(mask):
            continue
        if bbox_coords(np.argwhere(mask)) != (0, 0, h - 1, w - 1):
            continue
        return mask
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1:2, 1:-1:2] = False
    return mask


def _make_leaf_branch(
    rng: np.random.Generator,
    used_colors: set[int],
    max_h: int = 5,
    max_w: int = 5,
) -> BranchBundle:
    h = int(rng.integers(2, max_h + 1))
    w = int(rng.integers(2, max_w + 1))
    color = _choose_color(rng, used_colors)
    mask = _random_leaf_mask(rng, h, w)
    patch = np.zeros((h, w), dtype=int)
    patch[mask] = color
    return BranchBundle(
        patch=patch,
        leaf_coords=np.argwhere(mask),
        used_colors=used_colors | {color},
        depth=1,
    )


def _make_branch(
    rng: np.random.Generator,
    depth: int,
    used_colors: set[int],
    max_h: int = 8,
    max_w: int = 8,
) -> BranchBundle:
    if depth <= 1 or max_h < 5 or max_w < 5:
        return _make_leaf_branch(rng, used_colors, min(max_h, 5), min(max_w, 5))

    inner = _make_branch(rng, depth - 1, used_colors, max_h - 2, max_w - 2)
    ih, iw = inner.patch.shape
    for _ in range(200):
        outer_h = int(rng.integers(ih + 2, min(max_h, ih + 4) + 1))
        outer_w = int(rng.integers(iw + 2, min(max_w, iw + 4) + 1))
        extra_h = outer_h - ih
        extra_w = outer_w - iw
        top = int(rng.integers(1, extra_h))
        left = int(rng.integers(1, extra_w))
        bottom = extra_h - top
        right = extra_w - left
        if bottom < 1 or right < 1:
            continue
        color = _choose_color(rng, inner.used_colors)
        patch = np.zeros((outer_h, outer_w), dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        paste(patch, inner.patch, top, left, transparent=BG)
        return BranchBundle(
            patch=patch,
            leaf_coords=inner.leaf_coords + np.array([top, left], dtype=int),
            used_colors=inner.used_colors | {color},
            depth=inner.depth + 1,
        )
    raise RuntimeError("failed to build nested branch")


def _place_boxes(
    rng: np.random.Generator,
    patches: list[np.ndarray],
    *,
    min_h: int = 1,
    max_h: int = 30,
    min_w: int = 1,
    max_w: int = 30,
    padding: int = 1,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sizes = [patch.shape for patch in patches]
    need_h = max([min_h] + [h for h, _ in sizes])
    need_w = max([min_w] + [w for _, w in sizes])
    area = sum(h * w for h, w in sizes)
    lower_h = max(need_h, int(np.sqrt(area)) - 1)
    lower_w = max(need_w, int(np.sqrt(area)) - 1)

    for _ in range(250):
        H = int(rng.integers(lower_h, max_h + 1))
        W = int(rng.integers(lower_w, max_w + 1))
        if H * W < area + len(patches) * 3:
            continue

        occ = np.zeros((H, W), dtype=int)
        placements: list[tuple[int, int] | None] = [None] * len(patches)
        order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

        ok = True
        for idx in order:
            box = np.ones(patches[idx].shape, dtype=int)
            try:
                loc = random_free_location(
                    occ,
                    box,
                    transparent=BG,
                    border=0,
                    padding=padding,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            paste(occ, box, loc[0], loc[1], transparent=BG)
            placements[idx] = loc
        if ok:
            return occ, [loc for loc in placements if loc is not None]
    raise RuntimeError("failed to place patches")


def _collapse_local_tree(root_shape: tuple[int, int], root_color: int, leaf_coords: np.ndarray) -> np.ndarray:
    return _collapse_tree_patch(root_shape, (0, 0, root_shape[0] - 1, root_shape[1] - 1), root_color, leaf_coords)


def _tree_depth_pattern(rng: np.random.Generator) -> list[int]:
    branch_count = int(rng.choice([1, 2, 3], p=[0.35, 0.4, 0.25]))
    options = {
        1: [[1], [2], [3]],
        2: [[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1]],
        3: [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 1], [2, 1, 2], [1, 2, 2]],
    }
    return list(options[branch_count][int(rng.integers(0, len(options[branch_count])))])


def _make_tree_entity(rng: np.random.Generator) -> EntityBundle:
    for _ in range(300):
        try:
            branch_depths = _tree_depth_pattern(rng)
            if 1 + sum(branch_depths) > len(FG_COLORS):
                continue

            used_colors: set[int] = set()
            branches: list[BranchBundle] = []
            for depth in branch_depths:
                branch = _make_branch(rng, depth, used_colors)
                used_colors = set(branch.used_colors)
                branches.append(branch)

            container, placements = _place_boxes(
                rng,
                [branch.patch for branch in branches],
                min_h=6,
                max_h=13,
                min_w=6,
                max_w=13,
                padding=1,
            )

            leaf_groups = []
            for branch, (r0, c0) in zip(branches, placements):
                leaf_groups.append(branch.leaf_coords + np.array([r0, c0], dtype=int))
            leaf_coords = np.concatenate(leaf_groups, axis=0)
            group_bbox = bbox_coords(leaf_coords)
            group_h = bbox_h(group_bbox)
            group_w = bbox_w(group_bbox)

            margin_choices = []
            for top in range(1, 4):
                for bottom in range(1, 4):
                    for left in range(1, 4):
                        for right in range(1, 4):
                            root_h = container.shape[0] + top + bottom
                            root_w = container.shape[1] + left + right
                            if root_h > 18 or root_w > 18:
                                continue
                            if (root_h - group_h) % 2 == 0 and (root_w - group_w) % 2 == 0:
                                margin_choices.append((top, bottom, left, right))
            if not margin_choices:
                continue
            top, bottom, left, right = margin_choices[int(rng.integers(0, len(margin_choices)))]

            root_color = _choose_color(rng, used_colors)
            root_h = container.shape[0] + top + bottom
            root_w = container.shape[1] + left + right
            input_patch = np.zeros((root_h, root_w), dtype=int)
            input_patch[0, :] = root_color
            input_patch[-1, :] = root_color
            input_patch[:, 0] = root_color
            input_patch[:, -1] = root_color

            for branch, (r0, c0) in zip(branches, placements):
                paste(input_patch, branch.patch, top + r0, left + c0, transparent=BG)
            shifted_leaf_coords = leaf_coords + np.array([top, left], dtype=int)
            output_patch = _collapse_local_tree(input_patch.shape, root_color, shifted_leaf_coords)

            if np.array_equal(input_patch, output_patch):
                continue
            return EntityBundle(input_patch=input_patch, output_patch=output_patch)
        except (RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to build tree entity")


def _make_standalone_entity(rng: np.random.Generator) -> EntityBundle:
    bundle = _make_leaf_branch(rng, set(), max_h=5, max_w=5)
    return EntityBundle(input_patch=bundle.patch, output_patch=bundle.patch.copy())


def _compose_scene(rng: np.random.Generator, entities: list[EntityBundle]) -> tuple[np.ndarray, np.ndarray]:
    sizes = [entity.input_patch.shape for entity in entities]
    need_h = max(h for h, _ in sizes)
    need_w = max(w for _, w in sizes)
    total_area = sum(h * w for h, w in sizes)
    lower_h = max(need_h, min(30, int(np.sqrt(total_area * 1.5))))
    lower_w = max(need_w, min(30, int(np.sqrt(total_area * 1.5))))

    for _ in range(300):
        H = int(rng.integers(lower_h, 31))
        W = int(rng.integers(lower_w, 31))
        if H * W < total_area + len(entities) * 10:
            continue
        if any(entity.input_patch.shape[0] > H or entity.input_patch.shape[1] > W for entity in entities):
            continue

        occ = np.zeros((H, W), dtype=int)
        inp = np.zeros((H, W), dtype=int)
        expected = np.zeros((H, W), dtype=int)
        order = sorted(
            range(len(entities)),
            key=lambda i: entities[i].input_patch.shape[0] * entities[i].input_patch.shape[1],
            reverse=True,
        )

        ok = True
        placements: dict[int, tuple[int, int]] = {}
        for idx in order:
            box = np.ones(entities[idx].input_patch.shape, dtype=int)
            try:
                loc = random_free_location(
                    occ,
                    box,
                    transparent=BG,
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            placements[idx] = loc
            paste(occ, box, loc[0], loc[1], transparent=BG)
            paste(inp, entities[idx].input_patch, loc[0], loc[1], transparent=BG)
            paste(expected, entities[idx].output_patch, loc[0], loc[1], transparent=BG)
        if ok and np.count_nonzero(inp) > 0:
            return inp, expected
    raise RuntimeError("failed to compose scene")


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(150):
        tree_count = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
        standalone_count = int(rng.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15]))
        try:
            entities = [_make_tree_entity(rng) for _ in range(tree_count)]
            entities.extend(_make_standalone_entity(rng) for _ in range(standalone_count))
            inp, expected = _compose_scene(rng, entities)
        except (RuntimeError, ValueError):
            continue

        if not np.array_equal(main(inp), expected):
            continue

        objects = find_components(inp, bg=BG, connectivity=8, monochromatic=True, return_format="dict")
        _parents, children, roots = _build_forest(objects)
        transformed_roots = [root for root in roots if len(_subtree_nodes(root, children)) > 1]
        if not transformed_roots:
            continue
        if np.array_equal(inp, expected):
            continue
        return inp, expected
    raise RuntimeError("failed to generate valid case")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, _expected = _generate_case(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
