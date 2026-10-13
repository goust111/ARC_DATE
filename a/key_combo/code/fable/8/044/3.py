from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg


BG = 0
COLORS = tuple(range(1, 10))
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_TRIES = 600


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    local = coords - np.array([r0, c0], dtype=int)
    cells = tuple(sorted((int(r), int(c)) for r, c in local))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), cells


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    local = coords - np.array([r0, c0], dtype=int)
    cells = tuple(sorted((int(r), int(c)) for r, c in local))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), cells


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _aligned_union(masks: list[np.ndarray]) -> np.ndarray:
    h = max(int(mask.shape[0]) for mask in masks)
    w = max(int(mask.shape[1]) for mask in masks)
    union = np.zeros((h, w), dtype=bool)
    for mask in masks:
        mh, mw = map(int, mask.shape)
        union[:mh, :mw] |= mask
    return _crop_mask(union)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each left-panel hole with the color whose right-panel aligned fragment-union has the same shape.

    Input structure:
    - One full-height vertical separator splits the grid into a left template panel and a right fragment panel.
    - The left panel contains a single-color template on the background; its internal background components are holes.
    - The right panel contains several colors, and each color appears as 2 or 3 disconnected partial fragments.

    Rule:
    - For each color on the right, crop every fragment to its own bbox, align those crops by their top-left corners,
      and take their union to recover that color's complete shape.
    - Each internal hole on the left matches exactly one recovered complete shape up to translation, with no rotation
      or reflection.
    - Return only the left panel after filling every hole with the color of its matched recovered shape.

    Key insight:
    - No single right-panel fragment is a valid key; the correct key appears only after same-color fragments are
      normalized to their own bboxes and unioned at the top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = get_bg(g)

    sep_cols = [
        c
        for c in range(1, w - 1)
        if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])
    ]
    assert len(sep_cols) == 1
    sep = sep_cols[0]

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0

    left_colors = sorted(set(map(int, np.unique(left))) - {bg})
    assert len(left_colors) == 1

    hole_components = [
        coords
        for coords in components_mask(left == bg, connectivity=4)
        if not _touches_border(coords, left.shape)
    ]
    assert hole_components

    shape_to_color: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    right_colors = sorted(set(map(int, np.unique(right))) - {bg})
    assert right_colors
    for color in right_colors:
        comps = components_mask(right == color, connectivity=4)
        assert 2 <= len(comps) <= 3
        fragment_masks = []
        for coords in comps:
            r0, c0, r1, c1 = bbox_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            fragment_masks.append(local)
        full_mask = _aligned_union(fragment_masks)
        key = _mask_key(full_mask)
        assert key not in shape_to_color
        shape_to_color[key] = int(color)

    assert len(shape_to_color) == len(hole_components)

    out = left.copy()
    for coords in hole_components:
        key = _coords_key(coords)
        assert key in shape_to_color
        out[coords[:, 0], coords[:, 1]] = shape_to_color[key]
    return out


def _random_tree_shape() -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]] | None:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    min_area = max(5, int(np.ceil(0.42 * h * w)))
    max_area = min(h * w - 1, int(np.floor(0.72 * h * w)))
    if min_area > max_area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(0, 0): None}
    mask[0, 0] = True

    bottom_target = (h - 1, random.randint(0, w - 2))
    right_target = (random.randint(0, h - 2), w - 1)
    for target in (bottom_target, right_target):
        r, c = 0, 0
        while (r, c) != target:
            choices = []
            if r < target[0]:
                choices.append((r + 1, c))
            if c < target[1]:
                choices.append((r, c + 1))
            nr, nc = random.choice(choices)
            if not mask[nr, nc]:
                nbrs = [(rr, cc) for rr, cc in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)) if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]]
                if len(nbrs) != 1:
                    return None
                parent[(nr, nc)] = nbrs[0]
                mask[nr, nc] = True
            r, c = nr, nc

    base_area = int(mask.sum())
    if base_area > max_area:
        return None
    target_area = random.randint(max(min_area, base_area), max_area)
    while int(mask.sum()) < target_area:
        candidates = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                nbrs = [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]]
                if len(nbrs) == 1:
                    candidates.append((r, c, nbrs[0]))
        if not candidates:
            return None
        r, c, p = random.choice(candidates)
        mask[r, c] = True
        parent[(r, c)] = p

    if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    if np.array_equal(mask, np.ones_like(mask, dtype=bool)):
        return None

    children = {node: 0 for node in parent}
    for node, p in parent.items():
        if p is not None:
            children[p] += 1
    leaves = [node for node, count in children.items() if count == 0 and node != (0, 0)]
    if len(leaves) < 3:
        return None
    return mask, parent


def _tree_leaves(parent: dict[tuple[int, int], tuple[int, int] | None]) -> list[tuple[int, int]]:
    children = {node: 0 for node in parent}
    for node, p in parent.items():
        if p is not None:
            children[p] += 1
    return [node for node, count in children.items() if count == 0 and node != (0, 0)]


def _path_to_root(node: tuple[int, int], parent: dict[tuple[int, int], tuple[int, int] | None]) -> list[tuple[int, int]]:
    path = []
    cur: tuple[int, int] | None = node
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path


def _fragments_from_shape(mask: np.ndarray, parent: dict[tuple[int, int], tuple[int, int] | None], count: int) -> list[np.ndarray] | None:
    leaves = _tree_leaves(parent)
    if len(leaves) < count:
        return None

    exclusive = random.sample(leaves, count)
    leaf_assignments: dict[tuple[int, int], set[int]] = {}
    for i, leaf in enumerate(exclusive):
        leaf_assignments[leaf] = {i}
    for leaf in leaves:
        if leaf in leaf_assignments:
            continue
        subset_size = random.choices(
            population=tuple(range(1, count + 1)),
            weights=tuple(count + 1 - s for s in range(1, count + 1)),
            k=1,
        )[0]
        leaf_assignments[leaf] = set(random.sample(range(count), subset_size))

    fragments = []
    for i in range(count):
        frag = np.zeros_like(mask, dtype=bool)
        for leaf, owners in leaf_assignments.items():
            if i not in owners:
                continue
            for cell in _path_to_root(leaf, parent):
                frag[cell] = True
        fragments.append(frag)

    union = np.zeros_like(mask, dtype=bool)
    for frag in fragments:
        union |= frag
    if not np.array_equal(union, mask):
        return None
    if any(np.array_equal(frag, mask) for frag in fragments):
        return None
    if len({_mask_key(frag) for frag in fragments}) != len(fragments):
        return None
    for i in range(count):
        other_union = np.zeros_like(mask, dtype=bool)
        for j, frag in enumerate(fragments):
            if i != j:
                other_union |= frag
        if np.array_equal(other_union, mask):
            return None
    return fragments


def _place_rects(
    canvas_h: int,
    canvas_w: int,
    sizes: list[tuple[int, int]],
    *,
    min_r: int,
    min_c: int,
    max_r: int,
    max_c: int,
    padding: int,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx in order:
        h, w = sizes[idx]
        candidates = []
        for r in range(min_r, max_r - h + 1 + 1):
            for c in range(min_c, max_c - w + 1 + 1):
                rr0 = max(0, r - padding)
                cc0 = max(0, c - padding)
                rr1 = min(canvas_h, r + h + padding)
                cc1 = min(canvas_w, c + w + padding)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (r, c)
        occupied[r : r + h, c : c + w] = True
    return [placement for placement in placements if placement is not None]


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.where(mask)
    grid[top + rr, left + cc] = int(color)


def _build_scene() -> tuple[np.ndarray, np.ndarray, dict] | None:
    hole_count = random.randint(2, 4)
    shapes = []
    used_keys = set()
    for _ in range(hole_count):
        shape = None
        for _ in range(80):
            candidate = _random_tree_shape()
            if candidate is None:
                continue
            mask, parent = candidate
            key = _mask_key(mask)
            if key in used_keys:
                continue
            used_keys.add(key)
            shape = {"mask": mask, "parent": parent, "key": key}
            break
        if shape is None:
            return None
        shapes.append(shape)

    fragment_counts = [3 if random.random() < 0.55 else 2 for _ in range(hole_count)]
    if 3 not in fragment_counts:
        fragment_counts[random.randrange(hole_count)] = 3

    any_shrunken_fragment = False
    full_keys = {shape["key"] for shape in shapes}
    for shape, count in zip(shapes, fragment_counts):
        fragments = None
        for _ in range(120):
            candidate = _fragments_from_shape(shape["mask"], shape["parent"], count)
            if candidate is None:
                continue
            fragment_keys = {_mask_key(frag) for frag in candidate}
            if fragment_keys & full_keys:
                continue
            if any(_crop_mask(frag).shape != shape["mask"].shape for frag in candidate):
                any_shrunken_fragment = True
            fragments = candidate
            break
        if fragments is None:
            return None
        shape["fragments"] = fragments

    if not any_shrunken_fragment:
        return None

    palette = random.sample(COLORS, hole_count + 2)
    template_color = palette[0]
    separator_color = palette[1]
    fill_colors = palette[2:]
    random.shuffle(fill_colors)

    for shape, color in zip(shapes, fill_colors):
        shape["color"] = int(color)

    grid_h = random.randint(12, 18)
    left_w = random.randint(11, 14)
    right_w = random.randint(10, 14)
    if left_w + 1 + right_w > 30:
        return None

    max_shape_h = max(int(shape["mask"].shape[0]) for shape in shapes)
    max_shape_w = max(int(shape["mask"].shape[1]) for shape in shapes)
    template_h_min = max(max_shape_h + 4, 8)
    template_w_min = max(max_shape_w + 4, 8)
    template_h_max = min(grid_h - 2, grid_h - 2)
    template_w_max = min(left_w - 2, left_w - 2)
    if template_h_min > template_h_max or template_w_min > template_w_max:
        return None

    template_h = random.randint(template_h_min, template_h_max)
    template_w = random.randint(template_w_min, template_w_max)
    interior_h = template_h - 2
    interior_w = template_w - 2
    hole_sizes = [tuple(map(int, shape["mask"].shape)) for shape in shapes]
    hole_positions = _place_rects(
        interior_h,
        interior_w,
        hole_sizes,
        min_r=0,
        min_c=0,
        max_r=interior_h - 1,
        max_c=interior_w - 1,
        padding=1,
    )
    if hole_positions is None:
        return None

    max_top = grid_h - template_h - 1
    max_left = left_w - template_w - 1
    if max_top < 1 or max_left < 1:
        return None
    template_top = random.randint(1, max_top)
    template_left = random.randint(1, max_left)

    left_input = np.full((grid_h, left_w), BG, dtype=int)
    left_output = np.full((grid_h, left_w), BG, dtype=int)
    left_input[template_top : template_top + template_h, template_left : template_left + template_w] = template_color
    left_output[template_top : template_top + template_h, template_left : template_left + template_w] = template_color

    hole_records = []
    for shape, color, (r, c) in zip(shapes, fill_colors, hole_positions):
        top = template_top + 1 + r
        left = template_left + 1 + c
        hole_mask = shape["mask"]
        rr, cc = np.where(hole_mask)
        left_input[top + rr, left + cc] = BG
        left_output[top + rr, left + cc] = int(color)
        hole_records.append(
            {
                "color": int(color),
                "mask": hole_mask,
                "top": int(top),
                "left": int(left),
                "key": shape["key"],
            }
        )

    fragment_patches = []
    for shape in shapes:
        for frag in shape["fragments"]:
            patch = _crop_mask(frag)
            fragment_patches.append({"color": shape["color"], "patch": patch, "full_key": shape["key"]})

    patch_sizes = [tuple(map(int, item["patch"].shape)) for item in fragment_patches]
    fragment_positions = _place_rects(
        grid_h,
        right_w,
        patch_sizes,
        min_r=0,
        min_c=0,
        max_r=grid_h - 1,
        max_c=right_w - 1,
        padding=1,
    )
    if fragment_positions is None:
        return None

    right = np.full((grid_h, right_w), BG, dtype=int)
    fragment_records = []
    for item, (r, c) in zip(fragment_patches, fragment_positions):
        _render_mask(right, item["patch"], r, c, item["color"])
        fragment_records.append(
            {
                "color": int(item["color"]),
                "patch": item["patch"],
                "top": int(r),
                "left": int(c),
                "full_key": item["full_key"],
            }
        )

    sep_col = np.full((grid_h, 1), separator_color, dtype=int)
    inp = np.concatenate([left_input, sep_col, right], axis=1)
    meta = {
        "left_output": left_output,
        "hole_records": hole_records,
        "fragment_records": fragment_records,
        "grid_h": grid_h,
        "left_w": left_w,
        "right_w": right_w,
        "separator_color": separator_color,
    }
    if not _validate_scene(inp, meta):
        return None
    return inp, left_output, meta


def _validate_scene(inp: np.ndarray, meta: dict) -> bool:
    out = meta["left_output"]
    if not np.array_equal(main(inp), out):
        return False

    if len([c for c in range(1, inp.shape[1] - 1) if int(inp[0, c]) != BG and np.all(inp[:, c] == inp[0, c])]) != 1:
        return False

    hole_keys = {record["key"] for record in meta["hole_records"]}
    if len(hole_keys) != len(meta["hole_records"]):
        return False

    by_color: dict[int, list[np.ndarray]] = {}
    full_key_by_color: dict[int, tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = {}
    any_shrunken = False
    for record in meta["fragment_records"]:
        by_color.setdefault(record["color"], []).append(record["patch"])
        full_key_by_color[record["color"]] = record["full_key"]
        if record["patch"].shape != full_key_by_color[record["color"]][0]:
            any_shrunken = True

    if not any_shrunken:
        return False

    if set(full_key_by_color.values()) != hole_keys:
        return False

    for color, patches in by_color.items():
        if len(patches) not in (2, 3):
            return False
        union_key = _mask_key(_aligned_union(patches))
        if union_key != full_key_by_color[color]:
            return False
        for i, patch in enumerate(patches):
            if _mask_key(patch) in hole_keys:
                return False
            if _mask_key(patch) == union_key:
                return False
            other_union = _aligned_union([p for j, p in enumerate(patches) if j != i])
            if _mask_key(other_union) == union_key:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        built = _build_scene()
        if built is not None:
            inp, _out, _meta = built
            return inp
    raise RuntimeError("failed to generate a valid union-complete ARC input")
