# Variant of c7f57c3e
# Dimension: D13 — 干扰项处理 (meta)
# Change: 输入中不包含干扰项，所有非背景精灵都参与分组和交换... → 存在不参与交换的干扰精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D13 variant
# ==================================
# a) Concrete behavior and trace
#    - The original grid contains only swap-participating sprites. Every non-background
#      8-connected component is a scaled copy of one of two anchor-bearing prototypes.
#    - This variant injects 1-2 extra sprite-like distractors. They are still compact,
#      multi-color, and comparable in visual size to the real sprites, but they never
#      use anchor color 1 and they use colors not seen in either prototype family.
#      In the output these distractors stay exactly where they are, unchanged.
#    - Small trace:
#      old pipeline: components -> smallest bbox components define the two prototypes
#      -> all components get swapped.
#      new pipeline: components -> keep only anchor-bearing components as swap candidates
#      -> smallest anchor-bearing components define the two prototypes -> only those
#      candidates get swapped, while non-anchor distractors remain untouched.
# b) Complexity calibration
#    - Distractors are not random noise pixels. They are 3x3 or 4x4 asymmetric,
#      two-color shapes so the visual difference from the original task is obvious.
#    - Their colors are disjoint from the prototype colors, but there are only 1-2
#      distractors per grid and each uses exactly two colors, keeping the palette
#      interpretable from 2-4 examples.
# c) Implicit-assumption audit of the original main()
#    - `components_mask(g != bg)` assumed every component was a real sprite. No longer true.
#    - `_pick_anchor_bbox(arr, bg)` assumed every component had color 1. Distractors break it.
#    - `min_area = min(areas)` assumed the globally smallest components were real prototypes.
#      With distractors, that can be false.
#    - The replacement loop assumed every extracted component should be cleared and redrawn.
#      In the variant, only the anchor-bearing family members should be modified.
#    - Fix: filter the component list before prototype discovery and before replacement.
# d) Placement legality
#    - The generator still uses output-first construction for the swap family, preserving
#      the original anchor alignment and scale rules.
#    - Distractors are placed with the same `random_free_location` reserve canvas, so they
#      do not overlap any real sprite, do not cross borders, and keep the padding rule.
# e) Visual observability
#    - Across examples, humans can see that only the repeated anchor-colored family swaps.
#      The extra uniquely colored shapes remain fixed from input to output, so D13 is visible.
# f) Visual cleanliness
#    - All objects are separated by the reserve padding. There is no layering or ambiguous
#      overlap, and distractors are whole shapes rather than scattered pixels.
# g) Color-change inference
#    - The swap rule itself is unchanged: scale and anchor position are preserved while the
#      sprite changes to the opposite prototype. Distractor colors do not participate in any
#      mapping; they merely mark objects that are excluded from the family.
# 9) New solver assumptions and generator guarantees
#    - New assumption: every swappable sprite contains anchor color 1, and every distractor
#      avoids color 1. `_make_distractor_patch()` enforces this.
#    - New assumption: among anchor-bearing components, the smallest bbox area still yields
#      exactly two distinct scale-1 prototypes. The instance construction preserves this.
#    - New assumption: distractor colors are disjoint from the main family so distractors are
#      visibly separable. The generator reserves separate color pools for them.


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 pixels; active sprites are guaranteed to have them."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "active sprite has no anchor pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _sprite_record(grid: np.ndarray, pts: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "arr": arr,
        "has_anchor": bool(np.any(arr == 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the true sprite family to the opposite prototype and leave distractors untouched.

    Input structure:
    - bg is the dominant color. The grid mixes two families of anchor-bearing sprites with 1-2 extra distractor sprites.
    - Real family members are scaled copies of one of two prototypes and always contain color 1; distractors use separate colors and do not contain the anchor color.

    Rule:
    - Ignore components that are not part of the anchor-bearing family.
    - Recover the two scale-1 prototypes from the smallest anchor-bearing sprites, infer each real sprite's scale, and replace it with the other prototype at the same scale.
    - Keep every distractor sprite exactly as it appears in the input.

    Key insight:
    - The old swap logic still works after filtering: only the repeated components carrying the shared color-1 anchor belong to the transformable family, so distractors can be excluded before prototype matching and then left untouched.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 3, "grid must contain swap sprites and at least one distractor"

    records = [_sprite_record(g, pts) for pts in comps]
    sprites = [rec for rec in records if rec["has_anchor"]]
    assert len(sprites) >= 2, "grid must contain at least 2 anchor-bearing sprites"

    # Determine the two real prototypes using only anchor-bearing sprites.
    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest anchor-bearing sprites must yield exactly 2 prototypes"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert base_anc_h == anc1[2] - anc1[0] + 1
    assert base_anc_w == anc1[3] - anc1[1] + 1

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0, pc0 = panc[0] * scale, panc[1] * scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            ph, pw = sp.shape
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + ph)
            cc1 = min(w, left + pw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue

            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            m = sub_p != bg
            score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
            if score > best_score:
                best_score = score
                best_idx = idx

        repl_idx = 1 - best_idx

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        pr0, pc0 = panc[0] * scale, panc[1] * scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (D13 variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]
DISTRACTOR_COUNTS = [1, 2]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}

_DISTRACTOR_TEMPLATES = [
    np.array(
        [
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 0, 2],
            [0, 1, 2, 2],
            [0, 0, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 2, 0],
            [0, 2, 2, 2],
        ],
        dtype=int,
    ),
]


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


def _make_distractor_patch(bg: int, colors: tuple[int, int]) -> np.ndarray:
    template = random.choice(_DISTRACTOR_TEMPLATES)
    patch = np.full(template.shape, bg, dtype=int)
    patch[template == 1] = colors[0]
    patch[template == 2] = colors[1]
    return patch


def _grid_lower_bound(items: list[np.ndarray]) -> int:
    max_dim = max(max(item.shape) for item in items)
    total_area = sum(int(item.shape[0] * item.shape[1]) for item in items)
    area_bound = int(np.ceil(np.sqrt(total_area))) + 6
    return max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, area_bound)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    main_colors = sorted((set(map(int, proto_a.flatten())) | set(map(int, proto_b.flatten()))) - {bg, 1})
    distractor_count = random.choice(DISTRACTOR_COUNTS)
    remaining = [c for c in range(10) if c not in (bg, 1) and c not in main_colors]
    need = 2 * distractor_count
    if len(remaining) < need:
        return None
    distractor_colors = random.sample(remaining, need)
    distractors = [
        _make_distractor_patch(bg, (distractor_colors[2 * i], distractor_colors[2 * i + 1]))
        for i in range(distractor_count)
    ]

    placement_items = []
    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[1 - out_type], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        placement_items.append(marker)
    placement_items.extend(distractors)

    lower = _grid_lower_bound(placement_items)
    if lower > GRID_SIZE_MAX:
        return None

    n = random.randint(lower, GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
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
        in_r, in_c = out_r + dr, out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

    for patch in distractors:
        try:
            rr, cc = random_free_location(
                reserve,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(reserve, patch, rr, cc, transparent=bg)
        paste(out, patch, rr, cc, transparent=bg)
        paste(inp, patch, rr, cc, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D13 input")
