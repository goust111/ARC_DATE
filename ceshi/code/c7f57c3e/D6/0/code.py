# Variant of c7f57c3e
# Dimension: D6 — 颜色模式交换方式 (transform)
# Change: 两组之间完全交换颜色模式（A组获得B组的颜色，B组获得A组的颜色），同时保持尺度不变... → 单向颜色传递（非交换）

import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D6 variant
# a) Concrete behavior:
#    The input still shows 3-4 separated scaled sprites built from exactly two small
#    prototypes. The two prototypes keep the original relation: proto_B is the vertical
#    flip of proto_A with recolored variable regions, and both contain color 1 as an
#    anchor sub-bbox. The variant changes only the transform direction: sprites of the
#    "upper-anchor" family (proto_A) adopt the full proto_B pattern, while sprites that
#    already belong to the "lower-anchor" family (proto_B) stay as they are. The output
#    therefore contains only proto_B-looking sprites. Example at scale 1:
#      original task: A -> B and B -> A
#      this variant:  A -> B and B -> B
#    The changed step is the replacement choice after prototype matching; placement,
#    scale inference, and anchor alignment stay the same.
# b) Complexity calibration:
#    Keeping two prototypes and 3-4 instances preserves the original matching load and
#    avoids trivial single-object cases. A and B both appear at scale 1 in every input,
#    so the solver can infer both prototypes locally. Output collapsing to one family is
#    visually strong, but not ambiguous, because the input always contains repeated
#    evidence for both families and at least one receiver sprite must actually change.
# c) Hidden assumptions audit:
#    The original `repl_idx = 1 - best_idx` depends on symmetric swapping; that is no
#    longer valid. Prototype ordering also cannot stay arbitrary now, because "the other
#    prototype" is not the rule anymore. The new solver explicitly orders the two base
#    prototypes by anchor height: upper-anchor = receiver, lower-anchor = sender. The
#    generator guarantees this direction cue by rejecting vertically ambiguous label
#    grids, so the ordering used by `main()` is stable and human-visible.
# d) Placement legality:
#    Generation remains output-first. For each input sprite, the output sprite is the
#    sender prototype at the same scale, anchor-aligned against the input sprite. The
#    union bbox of input/output patches is reserved before placement, so transformed
#    patches cannot overlap or leave the grid. The same spacing/border rules as the
#    original task are retained.
# e) Visual observability:
#    The direction is visible from 2-3 examples because one orientation consistently
#    survives and the opposite orientation consistently turns into it. No hidden sort by
#    color id is used; the receiver/sender choice depends only on anchor location inside
#    the visible shape. The anchor is forced into the upper half of proto_A, hence into
#    the lower half of proto_B after vertical flip.
# f) Visual cleanliness:
#    Sprites remain isolated connected components with at least 2 cells of padding.
#    There are no extra markers or overlapping annotations, so each input/output pair is
#    readable as "one family dominates the final scene".
# g) Color inferability:
#    Each prototype still uses color 1 plus exactly two variable colors. The sender
#    palette is copied onto receiver sprites by reusing the full sender prototype, so the
#    palette change is coupled to a structural cue rather than being a free-form mapping.
# h) Variant-specific generator checks:
#    We reject label grids whose vertical flip keeps the anchor at the same height or
#    makes the two families too similar. That prevents accidental degeneration where the
#    one-way rule would look identical to the old two-way swap on some sample.
# i) New solver assumptions reviewed:
#    - smallest-area sprites must still expose exactly two distinct scale-1 prototypes
#    - anchor bbox sizes of both prototypes must match after ordering
#    - the two prototypes must have different direction keys
#    The generator constructs and verifies all three conditions before returning a grid.


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


def _direction_key(arr: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Order prototypes by visible vertical anchor placement: upper-anchor comes first."""
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    non_bg = arr != bg
    above = int(np.sum(non_bg[:ar0, :]))
    below = int(np.sum(non_bg[ar1 + 1 :, :]))
    return (ar0 + ar1, above - below, ac0 + ac1)


def _order_prototypes(protos: list[np.ndarray], bg: int) -> tuple[list[np.ndarray], list[int]]:
    keyed = sorted(
        [(idx, _direction_key(proto, bg), proto) for idx, proto in enumerate(protos)],
        key=lambda item: item[1],
    )
    assert keyed[0][1] != keyed[1][1], "prototype direction must be visually distinguishable"
    return [item[2] for item in keyed], [int(item[0]) for item in keyed]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint every sprite to the lower-anchor prototype while preserving scale and position.

    Input structure:
    - The background is the most common color. Two distinct sprite prototypes appear as
      small scale-1 examples and as larger nearest-neighbor scaled copies.
    - The prototypes share an internal color-1 anchor region, but one family places that
      anchor higher in its bbox while the other places it lower.

    Rule:
    - Recover the two base prototypes from the smallest sprites and order them by anchor
      height: upper-anchor family first, lower-anchor family second.
    - For each sprite, infer its scale and align the lower-anchor prototype by the color-1
      anchor box, then write that prototype at the same scale. Sprites already matching the
      lower-anchor family therefore remain unchanged.

    Key insight:
    - The transform is no longer a symmetric toggle. All sprites converge to the same
      destination family, and the visible anchor height inside the prototype tells which
      family is the receiver and which family is the persistent donor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    raw_protos = [uniq[key] for key in list(uniq.keys())[:2]]
    protos, _ = _order_prototypes(raw_protos, bg)
    receiver_proto, sender_proto = protos

    rec_anc = _pick_anchor_bbox(receiver_proto, bg)
    snd_anc = _pick_anchor_bbox(sender_proto, bg)
    rec_anc_h = rec_anc[2] - rec_anc[0] + 1
    rec_anc_w = rec_anc[3] - rec_anc[1] + 1
    snd_anc_h = snd_anc[2] - snd_anc[0] + 1
    snd_anc_w = snd_anc[3] - snd_anc[1] + 1
    assert (rec_anc_h, rec_anc_w) == (snd_anc_h, snd_anc_w), "prototype anchors must share the same size"

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = s["arr"]
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, rec_anc_h))
        scale_w = max(1, inst_anc_w // max(1, rec_anc_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            pr0, pc0, pr1, pc1 = _pick_anchor_bbox(p, bg)
            pr0 *= scale
            pc0 *= scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            score = 0
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
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
            if score > best_score:
                best_score = score
                best_idx = idx

        assert best_score >= 0, "each sprite must match one of the two prototypes"
        # Matching is still performed to preserve the original identification pipeline;
        # the D6 change is that both matched families now resolve to the same sender prototype.
        target_proto = sender_proto

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        sp = _scale_nn(target_proto, scale)
        pr0, pc0, pr1, pc1 = _pick_anchor_bbox(target_proto, bg)
        pr0 *= scale
        pc0 *= scale
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


# --------------- generator (o5x) ---------------

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

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300
LABEL_RETRIES = 200

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


def _make_labels_once() -> np.ndarray:
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


def _labels_are_directional(labels: np.ndarray) -> bool:
    if len(components_mask(labels != 0, connectivity=8)) != 1:
        return False
    flipped = np.flipud(labels)
    if np.array_equal(labels, flipped):
        return False
    a_key = _direction_key(labels, 0)
    b_key = _direction_key(flipped, 0)
    if not (a_key < b_key):
        return False
    ar0, ac0, ar1, ac1 = _anchor_bbox(labels)
    br0, bc0, br1, bc1 = _anchor_bbox(flipped)
    if ar0 + ar1 == br0 + br1:
        return False
    return True


def _make_labels() -> np.ndarray:
    for _ in range(LABEL_RETRIES):
        labels = _make_labels_once()
        if _labels_are_directional(labels):
            return labels
    raise RuntimeError("failed to construct directional labels")


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

    proto_a = _colorize(labels, a_cols)
    proto_b = _colorize(np.flipud(labels), b_cols)
    assert _direction_key(proto_a, bg) < _direction_key(proto_b, bg)
    return proto_a, proto_b


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
    sender_idx = 1

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    if not any(inp_type == 0 for inp_type, _ in instances):
        return None
    if not any(inp_type == 1 for inp_type, _ in instances):
        return None

    max_dim = max(max(protos[inp_type].shape) * scale for inp_type, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    changed = 0
    unchanged = 0
    for inp_type, scale in instances:
        output_patch = _scale(protos[sender_idx], scale)
        input_patch = _scale(protos[inp_type], scale)
        if inp_type == sender_idx:
            unchanged += 1
        else:
            changed += 1
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

    if changed == 0 or unchanged == 0:
        return None

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
    raise RuntimeError("failed to generate a valid c7f57c3e D6 input")
