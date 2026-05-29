# Variant of c7f57c3e
# Dimension: D1 — 分组匹配条件 (match_cond)
# Change: 按颜色模式（如 teal/green 组 vs red/green 组）将精灵分为两组，同组精灵共享相同颜色组合... → 按精灵大小分组
#
# Design analysis
# ---------------
# variant_spec:
#   source_task_baseline:
#     original_c7f57c3e: "group identity came from prototype/color pattern; every sprite toggled to the other prototype"
#     reference_31f7f899: "a visible scalar measurement (height) determines redistribution; complexity stays at one clear ranking/counting step"
#   concrete_visualization:
#     sprites: "still two multicolor prototypes with the same bbox shape and the same color-1 anchor semantics"
#     size_groups:
#       small: "all scale-1 sprites"
#       large: "all scale-2 or scale-3 sprites, but within one grid they share the same larger scale"
#     input_patterning:
#       small_group: "all small sprites share one prototype/color pattern"
#       large_group: "all large sprites share the other prototype/color pattern"
#     output_patterning:
#       swap: "small sprites take the large group's pattern; large sprites take the small group's pattern"
#     preserved_properties: "each sprite keeps its own location, own scale, and anchor alignment"
#   behavior_trace:
#     original_flow:
#       - "find two smallest prototypes"
#       - "classify each sprite by which prototype it already matches"
#       - "replace it with the opposite prototype"
#     variant_flow:
#       - "measure bbox area to split sprites into small vs large groups"
#       - "read the current prototype shown by the small group and by the large group"
#       - "swap those two group-level patterns while keeping each sprite's own scale"
#     example:
#       - "if the two 4x4 sprites are red/green and the two 8x8 sprites are teal/yellow, output turns the 4x4 sprites teal/yellow and the 8x8 sprites red/green"
#   complexity_calibration:
#     kept_simple: "only two size levels per grid, so humans can see the partition from 2-4 examples"
#     not_too_simple: "each size group appears multiple times, preventing a lone outlier from making the rule ambiguous"
#     visibility_support: "prototype color pairs are chosen disjointly so the swap is easy to notice"
#   implicit_assumption_audit:
#     old_smallest_two_distinct_prototypes: "no longer valid, because the smallest sprites now all belong to one size group and intentionally share one prototype"
#     old_per_sprite_prototype_toggle: "no longer valid, because target prototype depends on group size, not the sprite's current prototype identity"
#     still_valid:
#       - "sprites remain tight connected components"
#       - "both prototypes still share one base bbox shape"
#       - "integer nearest-neighbor scaling is still used"
#       - "color-1 anchor pixels still define alignment"
#     new_solver_assumptions:
#       - "exactly two bbox-area levels exist"
#       - "all small sprites normalize to one prototype"
#       - "all large sprites normalize to the other prototype"
#       - "large bbox dimensions are integer multiples of small bbox dimensions"
#     generator_guarantees_these: true
#   placement_legality:
#     strategy: "output-first with anchor-aligned union bboxes, same as the original seed"
#     spacing: "reserve map plus padding keeps components separated in both input and output"
#     bounds: "grid size is chosen from the largest scaled sprite dimension upward"
#   visual_observability:
#     visible_signal:
#       - "size difference is explicit in pixel extent"
#       - "pattern difference is explicit in color pair and flip structure"
#       - "the rule is consistent across all sprites of the same size group"
#     avoided_hidden_signal:
#       - "no sorting by color id"
#       - "no invisible metadata"
#   cleanliness:
#     layout: "only four sprites, with padding between union bboxes, avoids clutter"
#     overlap: "none between sprites in either input or output"

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for extra in (ROOT, ROOT / "solvers"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.append(extra_s)

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick a stable anchor bbox inside the sprite array (color-1 pixels; always present)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _downscale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "scaled sprite must be divisible by its integer scale"
    return arr[::scale, ::scale].copy()


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the prototype assignment between the small-sprite group and the large-sprite group.

    Input structure:
    - bg = most common color. Two distinct multicolor prototypes appear as separated sprite components.
    - Sprites come in exactly two visible size levels (small vs large), and all sprites of the same size share one prototype pattern.

    Rule:
    - Split sprites into the small group and the large group by bounding-box area.
    - Read which prototype pattern the small group currently uses and which prototype pattern the large group currently uses.
    - Replace every small sprite with the large-group prototype and every large sprite with the small-group prototype, preserving each sprite's own scale and anchor alignment.

    Key insight: Prototype identity is no longer attached to each sprite individually; it is attached to a size class, so the solver must measure size first and then exchange the two group-level patterns.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 4, "grid must contain multiple sprites from both size groups"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [_bbox_area(s["bbox"]) for s in sprites]
    area_levels = sorted(set(areas))
    assert len(area_levels) == 2, "variant expects exactly two bbox-area levels"
    small_area, large_area = area_levels

    small_sprites = [s for s, area in zip(sprites, areas, strict=False) if area == small_area]
    large_sprites = [s for s, area in zip(sprites, areas, strict=False) if area == large_area]
    assert len(small_sprites) >= 2 and len(large_sprites) >= 2, "both size groups must have repeated support"

    small_proto = small_sprites[0]["arr"].copy()
    base_h, base_w = small_proto.shape

    large_h, large_w = large_sprites[0]["arr"].shape
    scale_h = large_h // base_h
    scale_w = large_w // base_w
    assert scale_h == scale_w and scale_h >= 2, "large sprite must be an integer upscale of the small prototype size"
    large_scale = int(scale_h)
    large_proto = _downscale_nn(large_sprites[0]["arr"], large_scale)

    for s in small_sprites[1:]:
        assert np.array_equal(s["arr"], small_proto), "all small sprites must share one prototype"
    for s in large_sprites:
        norm = _downscale_nn(s["arr"], large_scale)
        assert np.array_equal(norm, large_proto), "all large sprites must share one normalized prototype"
    assert not np.array_equal(small_proto, large_proto), "the two size groups must show different patterns"

    out = g.copy()

    for s, area in zip(sprites, areas, strict=False):
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        target_proto = large_proto if area == small_area else small_proto
        scale = int(arr.shape[0] // base_h)
        assert arr.shape[1] == base_w * scale, "sprite width must follow the same integer scale"

        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, _ar1, _ac1 = anc

        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = bg

        scaled_target = _scale_nn(target_proto, scale)
        target_anchor = _pick_anchor_bbox(target_proto, bg)
        pr0, pc0, _pr1, _pc1 = target_anchor
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = scaled_target.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = scaled_target[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub_p[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.0

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SMALL_COUNT = 2
LARGE_COUNT = 2
LARGE_SCALES = [2]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: VARIABLE_ROW_WEIGHTS,
        3: VARIABLE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        entries = weights_by_role[role]
        kinds, wts = zip(*entries)
        chosen_kind = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen_kind, w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, a_cols), _colorize(np.flipud(labels), b_cols)


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    large_scale = random.choice(LARGE_SCALES)
    small_out_type = random.randrange(2)
    large_out_type = 1 - small_out_type

    instances = []
    for _ in range(SMALL_COUNT):
        instances.append({"scale": 1, "out_type": small_out_type})
    for _ in range(LARGE_COUNT):
        instances.append({"scale": large_scale, "out_type": large_out_type})
    random.shuffle(instances)
    instances.sort(key=lambda item: int(item["scale"]), reverse=True)

    max_dim = max(max(protos[int(item["out_type"])].shape) * int(item["scale"]) for item in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for item in instances:
        out_type = int(item["out_type"])
        scale = int(item["scale"])
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[1 - out_type], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D1 input")
