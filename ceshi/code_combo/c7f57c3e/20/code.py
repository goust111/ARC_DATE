# Combo variant of c7f57c3e (index 20)
# combo_spec: D3/2+D7/2
# Name: 位置由尺度排序 + 统一缩放到同一尺度
#
# Design analysis:
# - Concrete fused behavior:
#   The source task swaps each sprite to the other prototype while keeping both scale and place.
#   D3/2 removes the old "stay near yourself" placement rule: the sprites now define three visible
#   slots on one horizontal or vertical line, and output order is decided globally by original scale.
#   D7/2 removes the old "keep your own scale" rule: after swapping prototype identity, every output
#   sprite is redrawn at the single largest scale present in the input. The fused behavior is therefore:
#   recover each sprite's original type and original scale, sort the swapped sprites by original scale,
#   then place those swapped sprites into the ordered slots using one shared largest scale for all of them.
#   Example: horizontal slots contain [scale2 of B, scale1 of A, scale1 of B].
#   Original task output: [scale2 of A, scale1 of B, scale1 of A] in the same three places.
#   Fused output here: sort by original scale to get [(scale1 A), (scale1 B), (scale2 B)], swap type,
#   normalize all to scale2, and place them left to right as [scale2 of B, scale2 of A, scale2 of A].
#   The difference from original is twofold at once: the large sprite's identity moves to the size-sorted
#   slot, and the two small sprites visibly enlarge to match the maximum scale.
# - Complexity calibration:
#   To keep the rule learnable and non-degenerate, the generator fixes the scene to exactly three sprites
#   with original scales [1, 1, 2]. This preserves the original prototype-recovery trick because the two
#   scale-1 sprites expose the two base prototypes directly, and it also guarantees repeated evidence for
#   both fused dimensions in every sample: one sprite is visibly larger in the input, while two sprites
#   visibly enlarge in the output. Scale 3 is deliberately disallowed here because three scale-3-capable
#   slots cannot fit inside the original 18-28 canvas bounds for these 3x4/4x4 prototypes.
# - Implicit assumption review:
#   1. The original solver assumes each output sprite should be anchor-aligned back onto its own input
#      location. D3/2 breaks that, so this fused solver must first recover the global slot order and only
#      then redraw into destination slots.
#   2. The original solver clears and redraws per sprite. D7/2 can enlarge small sprites, so redrawing
#      against the original grid risks stale pixels if outputs outgrow inputs. The fused solver therefore
#      redraws from a blank background grid.
#   3. Prototype matching still relies on the smallest components being the unscaled prototypes and on
#      scale being inferable from the color-1 anchor bbox. The generator explicitly preserves both facts.
#   4. Sorting introduces a new stable tie-break requirement for the two scale-1 sprites. The solver uses
#      slot order as the tie-breaker, and the generator constructs the intended output with that same rule.
# - Placement legality:
#   Because D3/2 turns positions into reusable slots, the generator preallocates three equal-capacity slots
#   sized for the normalized scale-2 outputs. Every input sprite is top-left aligned inside one slot, and
#   every output sprite is also top-left aligned inside a slot. This guarantees no overlap, no clipping,
#   and no ambiguity about destination geometry after normalization.
# - Visual observability:
#   A human can infer the rule from 2-3 examples because the same two regularities repeat every time:
#   the largest input sprite does not stay in place, and the smaller sprites expand to the same size as
#   that largest sprite. The sort key is the visibly larger-vs-smaller relation only; it never depends on
#   hidden metadata such as color ids or generation order.
# - Visual cleanliness and color inference:
#   There are no extra markers or overlays. Each slot contains exactly one sprite, spacing is explicit,
#   and colors still only encode which of the two prototypes is present after the swap. Since all outputs
#   share one scale, the scene remains neat rather than mixing several enlarged footprints in one region.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of the color-1 anchor region inside a sprite array."""
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
    """Infer sprite scale from the size of its color-1 anchor bbox."""
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_anc_h = ar1 - ar0 + 1
    inst_anc_w = ac1 - ac0 + 1
    scale_h = max(1, inst_anc_h // max(1, base_anchor_h))
    scale_w = max(1, inst_anc_w // max(1, base_anchor_w))
    scale = int(max(scale_h, scale_w))
    return (ar0, ac0, ar1, ac1), scale


def _match_sprite(arr: np.ndarray, protos: list[np.ndarray], scale: int) -> int:
    """Identify which prototype exactly matches this scaled sprite array."""
    for idx, proto in enumerate(protos):
        if np.array_equal(arr, _scale_nn(proto, scale)):
            return idx
    raise AssertionError("sprite does not match either prototype at inferred scale")


def _sprite_center_key(sprite: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _ordered_slots(sprites: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the three visible slots in reading order along their shared axis."""
    centers = [_sprite_center_key(s) for s in sprites]
    row_range = max(rc[0] for rc in centers) - min(rc[0] for rc in centers)
    col_range = max(rc[1] for rc in centers) - min(rc[1] for rc in centers)
    horizontal = col_range >= row_range
    if horizontal:
        return sorted(sprites, key=lambda s: (int(s["bbox"][1]), int(s["bbox"][0])))
    return sorted(sprites, key=lambda s: (int(s["bbox"][0]), int(s["bbox"][1])))


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _slot_offsets(total: int, count: int, span: int, border: int, gap: int) -> list[int]:
    used = 2 * border + count * span + (count - 1) * gap
    if used > total:
        raise ValueError("slots do not fit")
    extra = total - used
    gaps = [0] * (count + 1)
    for _ in range(extra):
        gaps[random.randrange(count + 1)] += 1
    offsets: list[int] = []
    cur = border + gaps[0]
    for i in range(count):
        offsets.append(cur)
        cur += span
        if i < count - 1:
            cur += gap + gaps[i + 1]
    return offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, sort destinations by original scale, and redraw all outputs at one shared maximum scale.

    Input structure:
    - The background is the most frequent color, and exactly three disconnected multicolor sprites appear on one horizontal row of slots or one vertical column of slots.
    - Exactly two scale-1 sprites reveal the two base prototypes, and one additional sprite is a larger nearest-neighbor copy of one of those prototypes.
    - Every prototype contains a color-1 anchor region plus two variable-color regions and background holes.

    Rule:
    - Recover the two unscaled prototypes from the smallest sprites, infer each sprite's prototype identity and its original integer scale, and read the slot order from left-to-right or top-to-bottom.
    - Compute the maximum original scale in the scene, sort the sprites by their original scale from small to large using slot order to break ties, then swap each sorted sprite to the opposite prototype.
    - Place those swapped sprites back into the ordered slots, but redraw every one of them at the single maximum scale rather than at its own original scale.

    Key insight:
    - The output position and the output size are controlled by different pieces of visible information: original size ranks decide which slot a sprite's swapped identity goes to, while the single largest input scale decides how large every output sprite becomes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 3, "grid must contain exactly 3 sprite components"

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
            uniq[key] = np.asarray(s["arr"]).copy()
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert base_anc_h == anc1[2] - anc1[0] + 1 and base_anc_w == anc1[3] - anc1[1] + 1

    slots = _ordered_slots(sprites)
    records: list[dict[str, int]] = []
    for slot_index, sprite in enumerate(slots):
        r0, c0, _r1, _c1 = sprite["bbox"]
        arr = np.asarray(sprite["arr"])
        _anc, scale = _infer_scale_from_anchor(arr, bg, base_anc_h, base_anc_w)
        proto_idx = _match_sprite(arr, protos, scale)
        records.append(
            {
                "slot_index": int(slot_index),
                "slot_r0": int(r0),
                "slot_c0": int(c0),
                "scale": int(scale),
                "proto_idx": int(proto_idx),
            }
        )

    target_scale = max(int(rec["scale"]) for rec in records)
    sorted_shapes = sorted(records, key=lambda rec: (int(rec["scale"]), int(rec["slot_index"])))
    out = np.full_like(g, bg)

    for dst, shape in zip(records, sorted_shapes, strict=True):
        repl = _scale_nn(protos[1 - int(shape["proto_idx"])], target_scale)
        paste(out, repl, int(dst["slot_r0"]), int(dst["slot_c0"]), transparent=bg)

    return out


# --------------- generator (combo) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_BORDER = 1
SLOT_GAP = 1
MAX_RETRIES = 300

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: fused slot layout ---
ORIENTATIONS = ["horizontal", "vertical"]
TARGET_SCALE = 2

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


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
    for r, (role, mask) in enumerate(zip(roles, masks, strict=True)):
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


def _assemble_scene(
    n: int,
    bg: int,
    protos: list[np.ndarray],
    slots: list[tuple[int, int]],
    assignments: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), (typ, scale) in zip(slots, assignments, strict=True):
        patch = _scale(protos[typ], scale)
        paste(grid, patch, int(r0), int(c0), transparent=bg)
    return grid


def _ordered_output_assignments(assignments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_shapes = sorted(
        list(enumerate(assignments)),
        key=lambda item: (int(item[1][1]), int(item[0])),
    )
    return [(1 - int(typ), TARGET_SCALE) for _idx, (typ, _scale_value) in sorted_shapes]


def _normalized_no_sort_assignments(assignments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(1 - int(typ), TARGET_SCALE) for typ, _scale_value in assignments]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    instances = [(0, 1), (1, 1), (random.randrange(2), TARGET_SCALE)]
    orientation = random.choice(ORIENTATIONS)

    max_h = max(_scale(proto, TARGET_SCALE).shape[0] for proto in protos)
    max_w = max(_scale(proto, TARGET_SCALE).shape[1] for proto in protos)

    if orientation == "horizontal":
        need = max(2 * GRID_BORDER + max_h, 2 * GRID_BORDER + 3 * max_w + 2 * SLOT_GAP)
    else:
        need = max(2 * GRID_BORDER + max_w, 2 * GRID_BORDER + 3 * max_h + 2 * SLOT_GAP)
    if need > GRID_SIZE_MAX:
        return None
    n = random.randint(max(GRID_SIZE_MIN, need), GRID_SIZE_MAX)

    if orientation == "horizontal":
        row = random.randint(GRID_BORDER, n - GRID_BORDER - max_h)
        cols = _slot_offsets(n, 3, max_w, GRID_BORDER, SLOT_GAP)
        slots = [(int(row), int(c)) for c in cols]
    else:
        col = random.randint(GRID_BORDER, n - GRID_BORDER - max_w)
        rows = _slot_offsets(n, 3, max_h, GRID_BORDER, SLOT_GAP)
        slots = [(int(r), int(col)) for r in rows]

    perm = list(range(3))
    random.shuffle(perm)
    while perm == [0, 1, 2] or perm.index(2) == 2:
        random.shuffle(perm)
    input_assignments = [instances[i] for i in perm]
    output_assignments = _ordered_output_assignments(input_assignments)
    if output_assignments == _normalized_no_sort_assignments(input_assignments):
        return None

    inp = _assemble_scene(n, bg, protos, slots, input_assignments)
    out = _assemble_scene(n, bg, protos, slots, output_assignments)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
