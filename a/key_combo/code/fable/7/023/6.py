from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, object_interior, paste


BG = 0
MIN_SIDE = 5
MAX_SIDE = 8
MAX_ATTEMPTS = 500


def _separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    assert len(rows) == 1, "expected exactly one solid separator row"
    r = int(rows[0])
    return r, int(grid[r, 0])


def _component_patch(comp: dict, bg: int) -> np.ndarray:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _hole_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    return object_interior(patch, background=bg) & (patch == bg)


def _count_holes(patch: np.ndarray, bg: int) -> int:
    holes = _hole_mask(patch, bg)
    if not holes.any():
        return 0
    comps = find_components(
        holes.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[1],
        return_format="dict",
    )
    return len(comps)


def _filled_signature(patch: np.ndarray, bg: int) -> np.ndarray:
    return object_interior(patch, background=bg)


def _rect_capacity(h: int, w: int) -> int:
    inner_h = max(0, h - 2)
    inner_w = max(0, w - 2)
    return (inner_h * inner_w + 1) // 2


def _choose_holes(h: int, w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    for _ in range(300):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(abs(r - rr) + abs(c - cc) != 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    raise RuntimeError("failed to place non-adjacent holes")


def _make_rect_object(h: int, w: int, color: int, hole_count: int) -> np.ndarray:
    assert _rect_capacity(h, w) >= hole_count
    patch = np.full((h, w), int(color), dtype=int)
    for r, c in _choose_holes(h, w, hole_count):
        patch[r, c] = BG
    return patch


def _dims_for_holes(hole_count: int, *, exclude: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    dims = []
    for h in range(MIN_SIDE, MAX_SIDE + 1):
        for w in range(MIN_SIDE, MAX_SIDE + 1):
            if exclude is not None and (h, w) == exclude:
                continue
            if _rect_capacity(h, w) >= hole_count:
                dims.append((h, w))
    return dims


def _dims_for_exact_area(area: int, hole_count: int, *, exclude: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    dims = []
    for h in range(MIN_SIDE, MAX_SIDE + 1):
        for w in range(MIN_SIDE, MAX_SIDE + 1):
            if exclude is not None and (h, w) == exclude:
                continue
            if h * w - hole_count != area:
                continue
            if _rect_capacity(h, w) >= hole_count:
                dims.append((h, w))
    return dims


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def _row_extent(patches: list[np.ndarray]) -> tuple[int, int]:
    if not patches:
        return 0, 0
    gap = 2
    width = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1)
    height = max(p.shape[0] for p in patches)
    return int(width), int(height)


def _analyze_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep_r, _sep_color = _separator_row(g, bg)
    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]

    key_comps = find_components(upper, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(key_comps) == 1, "upper panel must contain exactly one key object"
    key_comp = key_comps[0]
    key_patch = _component_patch(key_comp, bg)
    key_holes = _count_holes(key_patch, bg)
    key_area = int(np.count_nonzero(key_patch != bg))
    key_color = int(key_comp["color"])
    key_sig = _filled_signature(key_patch, bg)

    lower_comps = find_components(lower, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(lower_comps) >= 4, "lower panel must contain several candidate locks"

    objs = []
    for comp in lower_comps:
        patch = _component_patch(comp, bg)
        objs.append(
            {
                "comp": comp,
                "patch": patch,
                "color": int(comp["color"]),
                "holes": _count_holes(patch, bg),
                "area": int(np.count_nonzero(patch != bg)),
                "sig": _filled_signature(patch, bg),
            }
        )

    hole_counts = [obj["holes"] for obj in objs]
    assert len(set(hole_counts)) == len(hole_counts), "lock hole counts must be unique"
    targets = [obj for obj in objs if obj["holes"] == key_holes]
    assert len(targets) == 1, "exactly one lock must match the key hole count"
    target = targets[0]

    same_color = [obj for obj in objs if obj["color"] == key_color and obj["holes"] != key_holes]
    same_area = [obj for obj in objs if obj["area"] == key_area and obj["holes"] != key_holes]
    same_shape = [obj for obj in objs if obj["holes"] != key_holes and np.array_equal(obj["sig"], key_sig)]
    assert len(same_color) == 1, "need exactly one same-color distractor"
    assert len(same_area) == 1, "need exactly one same-area distractor"
    assert len(same_shape) == 1, "need exactly one same-shape distractor"

    expected = target["patch"].copy()
    expected[_hole_mask(expected, bg)] = key_color

    return {
        "bg": bg,
        "key_holes": key_holes,
        "key_area": key_area,
        "key_color": key_color,
        "target": target,
        "expected": expected,
        "lower_count": len(objs),
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique lower lock whose hole count matches the upper key, then pour the key color into that lock's holes.

    Input structure:
    - One solid non-background separator row splits the grid into an upper key panel and a lower lock panel.
    - The upper panel contains exactly one monochrome connected key object with one or more enclosed background holes.
    - The lower panel contains several separated monochrome objects whose enclosed-hole counts are all different.

    Rule:
    - Count the number of enclosed background hole components inside the upper key.
    - In the lower panel, find the unique object with the same hole count.
    - Return that lock object's tight crop, keep its body color unchanged, and recolor each enclosed hole region to the key's color.

    Key insight:
    - The match is not based on visible color, area, or outline; it is the object's topological hole count, and the key's color is reused only after the correct lock is identified.
    """
    info = _analyze_case(np.asarray(input_grid, dtype=int))
    out = info["expected"]
    b = bbox(out != info["bg"])
    assert b is not None
    return crop_bbox(out, b)


def _build_case() -> np.ndarray | None:
    sep_color, key_color = random.sample(range(1, 10), 2)
    for _ in range(120):
        key_dims = random.choice(_dims_for_holes(1))
        max_key_holes = min(4, _rect_capacity(*key_dims))
        key_holes = random.randint(1, max_key_holes)
        key_h, key_w = key_dims
        key_area = key_h * key_w - key_holes

        shape_choices = [n for n in range(0, min(6, _rect_capacity(key_h, key_w)) + 1) if n != key_holes]
        if not shape_choices:
            continue
        shape_holes = random.choice(shape_choices)

        area_choices = [
            n
            for n in range(0, 7)
            if n not in {key_holes, shape_holes}
            and _dims_for_exact_area(key_area, n, exclude=key_dims)
        ]
        if not area_choices:
            continue
        area_holes = random.choice(area_choices)
        area_dims = random.choice(_dims_for_exact_area(key_area, area_holes, exclude=key_dims))

        target_dims_choices = _dims_for_holes(key_holes, exclude=key_dims)
        if not target_dims_choices:
            continue
        target_dims = random.choice(target_dims_choices)

        same_color_choices = []
        for n in range(0, 7):
            if n in {key_holes, shape_holes, area_holes}:
                continue
            dims = [d for d in _dims_for_holes(n, exclude=key_dims) if d[0] * d[1] - n != key_area]
            if dims:
                same_color_choices.append((n, dims))
        if not same_color_choices:
            continue
        same_color_holes, same_color_dims_pool = random.choice(same_color_choices)
        same_color_dims = random.choice(same_color_dims_pool)

        extra_specs = []
        remaining_counts = [n for n in range(0, 7) if n not in {key_holes, shape_holes, area_holes, same_color_holes}]
        random.shuffle(remaining_counts)
        if remaining_counts and random.random() < 0.6:
            for extra_holes in remaining_counts:
                extra_dims_pool = [d for d in _dims_for_holes(extra_holes, exclude=key_dims) if d[0] * d[1] - extra_holes != key_area]
                if extra_dims_pool:
                    extra_specs.append((extra_holes, random.choice(extra_dims_pool)))
                    break

        lower_other_colors = [c for c in range(1, 10) if c not in {sep_color, key_color}]
        if len(lower_other_colors) < 3 + len(extra_specs):
            continue
        target_color, shape_color, area_color, *extra_colors = random.sample(lower_other_colors, 3 + len(extra_specs))

        key_patch = _make_rect_object(key_h, key_w, key_color, key_holes)
        target_patch = _make_rect_object(target_dims[0], target_dims[1], target_color, key_holes)
        shape_patch = _make_rect_object(key_h, key_w, shape_color, shape_holes)
        area_patch = _make_rect_object(area_dims[0], area_dims[1], area_color, area_holes)
        same_color_patch = _make_rect_object(same_color_dims[0], same_color_dims[1], key_color, same_color_holes)

        lower_patches = [target_patch, shape_patch, area_patch, same_color_patch]
        for (extra_holes, extra_dims), color in zip(extra_specs, extra_colors):
            lower_patches.append(_make_rect_object(extra_dims[0], extra_dims[1], color, extra_holes))

        random.shuffle(lower_patches)
        split = 2 if len(lower_patches) == 4 else 3
        top_row = lower_patches[:split]
        bottom_row = lower_patches[split:]
        top_row_w, top_row_h = _row_extent(top_row)
        bottom_row_w, bottom_row_h = _row_extent(bottom_row)

        base_w = max(key_patch.shape[1] + 2, top_row_w + 2, bottom_row_w + 2)
        base_top_h = key_patch.shape[0] + 2
        base_bottom_h = top_row_h + bottom_row_h + 3
        if base_w > 30 or base_top_h + 1 + base_bottom_h > 30:
            continue

        panel_w = random.randint(base_w, min(30, base_w + 4))
        top_h = random.randint(base_top_h, min(30 - 1 - base_bottom_h, base_top_h + 2))
        bottom_h = random.randint(base_bottom_h, 30 - 1 - top_h)

        grid = np.full((top_h + 1 + bottom_h, panel_w), BG, dtype=int)
        grid[top_h, :] = sep_color

        key_r = random.randint(0, top_h - key_patch.shape[0])
        key_c = random.randint(1, panel_w - key_patch.shape[1] - 1)
        paste(grid, key_patch, key_r, key_c, transparent=BG)

        top_slack = bottom_h - base_bottom_h
        extra_top = random.randint(0, top_slack)
        extra_mid = random.randint(0, top_slack - extra_top)
        row1_r = 1 + extra_top
        row2_r = row1_r + top_row_h + 1 + extra_mid

        row1_c = random.randint(1, panel_w - top_row_w - 1)
        c = row1_c
        for patch in top_row:
            paste(grid, patch, top_h + 1 + row1_r, c, transparent=BG)
            c += patch.shape[1] + 2

        row2_c = random.randint(1, panel_w - bottom_row_w - 1)
        c = row2_c
        for patch in bottom_row:
            paste(grid, patch, top_h + 1 + row2_r, c, transparent=BG)
            c += patch.shape[1] + 2

        try:
            info = _analyze_case(grid)
        except Exception:
            continue
        if not np.array_equal(main(grid), info["expected"]):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-lock ARC input")
