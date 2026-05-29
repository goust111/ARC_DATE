# Variant of c7f57c3e
# Dimension: D7 — 尺度保持规则 (meta)
# Change: 交换颜色模式时保持精灵的尺度（scale）不变，同尺度精灵配对... → 统一缩放到最大/最小尺度

# Design analysis for this D7 variant
# - Concrete behavior:
#   The input still contains 3-4 isolated multicolor sprites built from exactly two prototype patterns.
#   Color 1 marks an internal anchor region. As in the original task, every sprite changes to the
#   opposite prototype. The D7 change is global scale normalization: instead of keeping each sprite at
#   its own scale, every output sprite is redrawn at the largest scale present anywhere in the input.
#   Example: if input scales are [1, 1, 3], the original task would output [1, 1, 3] with swapped
#   patterns, while this variant outputs [3, 3, 3] with swapped patterns. The anchor top-left still
#   stays fixed, so the enlarged small sprites "grow" around the same internal reference point.
# - Complexity calibration:
#   The generator always includes one large sprite (scale 2 or 3) and at least two scale-1 sprites,
#   one from each prototype. This guarantees repeated evidence for the normalization rule and prevents
#   degeneration to the original same-scale behavior. Keeping only one large scale tier preserves
#   learnability: solvers and humans only need to infer "everything becomes as large as the biggest."
# - Implicit assumptions from the original main():
#   1. Smallest-area components must reveal the two unscaled prototypes. This remains valid only if
#      the input still contains both prototypes at scale 1, so the generator guarantees that.
#   2. Matching a sprite against a prototype assumes the sprite's own scale can be inferred from the
#      anchor bbox size. This remains valid because scaling is still nearest-neighbor and isotropic.
#   3. The original code clears and redraws one sprite at a time. With enlargement, a later clear
#      could erase an already-drawn expanded neighbor if unions were not disjoint. To make the new
#      assumption explicit and robust, the solver now clears all input sprites first, then redraws all
#      normalized outputs in a second pass.
# - Placement legality:
#   The generator remains output-first, but "output" now means the normalized-scale swapped sprites.
#   For each object we reserve the full union bbox of its small input patch and its enlarged output
#   patch under anchor alignment. Distinct unions are placed with padding, so inputs do not overlap,
#   outputs do not overlap, and no output invades another object's input area.
# - Visual observability:
#   The change is visible without hidden metadata because at least two small sprites expand to match
#   the single largest sprite in every example. The rule uses only visible size relations, never color
#   ids or arbitrary ordering. The prototypes keep holes and asymmetric vertical relation so the swap
#   remains distinguishable after scaling.
# - Visual cleanliness and color inference:
#   No extra markers are introduced. Objects stay separated by rectangular reserves plus padding, which
#   keeps the scene readable even after upscaling. Colors still only encode prototype identity as in
#   the source task; D7 does not add any new color mapping burden.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of the color-1 anchor region inside a sprite array."""
    del bg
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "sprite array must contain color-1 anchor pixels"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _infer_scale_from_anchor(
    arr: np.ndarray, bg: int, base_anchor_h: int, base_anchor_w: int
) -> tuple[tuple[int, int, int, int], int]:
    """Infer integer sprite scale from the anchor bbox dimensions."""
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_anc_h = ar1 - ar0 + 1
    inst_anc_w = ac1 - ac0 + 1
    scale_h = max(1, inst_anc_h // max(1, base_anchor_h))
    scale_w = max(1, inst_anc_w // max(1, base_anchor_w))
    scale = int(max(scale_h, scale_w))
    return (ar0, ac0, ar1, ac1), scale


def _aligned_union_geometry(
    output_patch: np.ndarray, input_patch: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    """Return output/input offsets plus union size when anchor top-lefts are aligned."""
    out_anchor = _pick_anchor_bbox(output_patch, bg=0)
    in_anchor = _pick_anchor_bbox(input_patch, bg=0)
    rel_r = out_anchor[0] - in_anchor[0]
    rel_c = out_anchor[1] - in_anchor[1]

    min_r = min(0, rel_r)
    min_c = min(0, rel_c)
    max_r = max(output_patch.shape[0], rel_r + input_patch.shape[0])
    max_c = max(output_patch.shape[1], rel_c + input_patch.shape[1])

    out_r = -min_r
    out_c = -min_c
    in_r = rel_r - min_r
    in_c = rel_c - min_c
    return out_r, out_c, in_r, in_c, max_r - min_r, max_c - min_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then normalize all outputs to the largest scale.

    Input structure:
    - The background is the most frequent color, and the foreground consists of 3-4 disconnected sprites.
    - Exactly two prototype patterns are present; each sprite is a nearest-neighbor scaled copy of one of them.
      Both prototypes contain a color-1 anchor region plus other colors and holes.

    Rule:
    - Recover the two unscaled prototypes from the smallest sprites, identify each sprite's prototype and scale,
      and compute the maximum scale appearing anywhere in the input.
    - Replace every sprite with the opposite prototype, but redraw all replacements at that single maximum scale,
      aligned so the replacement anchor's top-left matches the original anchor's top-left.

    Key insight:
    - The prototype swap is still local to each sprite, but the size rule is now global: one large sprite reveals
      the normalization target, and every smaller sprite expands to that same scale while staying pinned by the
      shared color-1 anchor region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(np.asarray(s["arr"]).ravel().tolist())
        if key not in uniq:
            uniq[key] = np.asarray(s["arr"])
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert base_anc_h == anc1[2] - anc1[0] + 1 and base_anc_w == anc1[3] - anc1[1] + 1

    records: list[dict[str, object]] = []
    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = np.asarray(s["arr"])
        anc, scale = _infer_scale_from_anchor(arr, bg, base_anc_h, base_anc_w)
        ar0, ac0, ar1, ac1 = anc
        del r1, c1, ar1, ac1

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0 = panc[0] * scale
            pc0 = panc[1] * scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + sp.shape[0])
            cc1 = min(w, left + sp.shape[1])
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            mask = sub_p != bg
            score = int(np.sum(sub_g[mask] == sub_p[mask])) if np.any(mask) else 0
            if score > best_score:
                best_score = score
                best_idx = idx

        records.append(
            {
                "pts": s["pts"],
                "bbox": s["bbox"],
                "anchor": anc,
                "scale": scale,
                "type_idx": best_idx,
            }
        )

    target_scale = max(int(rec["scale"]) for rec in records)
    out = g.copy()

    for rec in records:
        for rr, cc in rec["pts"]:
            out[rr, cc] = bg

    for rec in records:
        r0, c0, _r1, _c1 = rec["bbox"]
        ar0, ac0, _ar1, _ac1 = rec["anchor"]
        repl = protos[1 - int(rec["type_idx"])]
        sp = _scale_nn(repl, target_scale)
        panc = _pick_anchor_bbox(repl, bg)
        pr0 = panc[0] * target_scale
        pc0 = panc[1] * target_scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + sp.shape[0])
        cc1 = min(w, left + sp.shape[1])
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub_p[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
NORMALIZED_SCALES = [2, 3]
SPRITE_COUNTS_BY_SCALE = {2: [3, 4], 3: [3]}

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
        kinds, wts = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

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


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    target_scale = random.choice(NORMALIZED_SCALES)
    count = random.choice(SPRITE_COUNTS_BY_SCALE[target_scale])

    # Input sprites: two scale-1 prototypes plus one already-large exemplar.
    input_instances = [(0, 1), (1, 1), (random.randrange(2), target_scale)]
    if count == 4:
        input_instances.append((random.randrange(2), 1))
    random.shuffle(input_instances)

    max_dim = max(max(proto.shape) * target_scale for proto in protos)
    if target_scale == 3:
        n = GRID_SIZE_MAX
    else:
        n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for input_type, input_scale in input_instances:
        input_patch = _scale(protos[input_type], input_scale)
        output_patch = _scale(protos[1 - input_type], target_scale)
        out_r0, out_c0, in_r0, in_c0, uh, uw = _aligned_union_geometry(output_patch, input_patch)
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

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, ur + out_r0, uc + out_c0, transparent=bg)
        paste(inp, input_patch, ur + in_r0, uc + in_c0, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e input")
