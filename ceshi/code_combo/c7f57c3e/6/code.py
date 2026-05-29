# Combo variant of c7f57c3e (index 6)
# combo_spec: D6/0+D9/0
# Name: 单向颜色传递 + 迭代交换

import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D6/0 + D9/0 combo variant
# a) Concrete behavior:
#    The original task toggles every sprite to the opposite prototype. This combo removes
#    the symmetry and makes the change iterative. We still recover two scale-1 prototypes,
#    but they are ordered by visible anchor height: the upper-anchor family is A
#    (receiver) and the lower-anchor family is B (donor). B never changes. A no longer
#    turns into the whole B prototype; instead it keeps its own structure and receives B's
#    two variable colors in two rounds. Because ARC supplies only one final output grid,
#    the intermediate state is externalized by scale tier: scale-1 A sprites stop after
#    round 1, while enlarged A sprites continue to round 2. Example:
#      input:  A1, B1, A3, B2
#      output: A1 gets only B's upper variable color; B1 stays B;
#              A3 gets both B variable colors; B2 stays B.
#    This makes the visible states {unchanged donor, round-1 receiver, round-2 receiver}
#    coexist in one output.
# b) Complexity calibration:
#    The rule uses only two visible axes: family identity (anchor high vs low) and size
#    tier (prototype-sized vs enlarged). That keeps the branch count small enough to infer
#    from 2-4 examples. We deliberately keep exactly two donor colors and exactly two
#    receiver rounds, so the palette transfer is structured rather than free-form.
# c) Hidden assumptions audit:
#    The original `repl_idx = 1 - best_idx` is invalid twice over: the transform is no
#    longer bidirectional, and the target of an A sprite depends on how far the iterative
#    process proceeds. The new solver therefore:
#    - orders the two prototypes explicitly by anchor height
#    - matches each sprite only to determine whether it is A or B
#    - derives two hybrid A prototypes (round 1 and round 2) instead of using only the
#      two raw input prototypes
#    - uses scale==1 versus scale>1 to choose which hybrid state to write
#    Another latent assumption in the original code is that arbitrary prototype ordering
#    is harmless; here it is not, so the generator rejects vertically ambiguous labels.
# d) Placement legality:
#    Generation remains output-first. Each instance reserves the union bbox of its input
#    patch and output patch under anchor alignment before anything is pasted, so neither
#    grid can overlap or go out of bounds. Receiver hybrids keep the receiver geometry,
#    donors keep donor geometry, and the same anchor-alignment math is used in solver and
#    generator.
# e) Visual observability:
#    The direction cue is human-visible because B survives unchanged at both small and big
#    scales, while only A recolors. The iterative cue is also human-visible because the
#    small A and enlarged A do not end in the same palette state. No hidden ordering by
#    color id is used: "first" and "second" transferred colors are defined by vertical
#    slot order inside the visible prototype.
# f) Visual cleanliness:
#    The grid still contains only 4 isolated sprites with at least 2 cells of padding, so
#    the three output states are easy to compare. We avoid extra markers or overlays.
# g) Color inferability:
#    The two donor colors are forced to be disjoint from the two receiver colors. That
#    prevents degenerate samples where a transfer step would be invisible. Each variable
#    slot is a coherent region, so people can see that round 1 changes one slot and round
#    2 changes the remaining slot.
# h) Variant-specific generator checks:
#    The label builder rejects patterns where flipping does not visibly change anchor
#    height. The color builder uses four distinct variable colors, and the generator always
#    includes one scale-1 A, one enlarged A, one scale-1 B, and one enlarged B, ensuring
#    the direction and the two iterative stages are all present in every sample.
# i) New solver assumptions reviewed:
#    `main()` assumes:
#    - smallest-area sprites expose exactly two raw prototypes
#    - the upper-anchor prototype is visually distinguishable from the lower-anchor one
#    - each prototype has exactly two variable-color regions, separable by vertical order
#    - enlarged sprites are integer scale copies inferred from the anchor bbox
#    - scale-1 receiver sprites encode round 1 and enlarged receiver sprites encode round 2
#    The generator constructs and oracle-checks all of these conditions.


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _direction_key(arr: np.ndarray, bg: int) -> tuple[int, int, int]:
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    non_bg = arr != bg
    above = int(np.sum(non_bg[:ar0, :]))
    below = int(np.sum(non_bg[ar1 + 1 :, :]))
    return (ar0 + ar1, above - below, ac0 + ac1)


def _order_prototypes(protos: list[np.ndarray], bg: int) -> list[np.ndarray]:
    keyed = sorted(
        [(idx, _direction_key(proto, bg), proto) for idx, proto in enumerate(protos)],
        key=lambda item: item[1],
    )
    assert keyed[0][1] != keyed[1][1], "prototype direction must be distinguishable"
    return [item[2] for item in keyed]


def _ordered_variable_slots(arr: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    colors = [int(c) for c in np.unique(arr) if int(c) not in (bg, 1)]
    assert len(colors) == 2, "each prototype must have exactly 2 variable colors"
    slots = []
    for color in colors:
        ys, xs = np.where(arr == color)
        key = (float(np.mean(ys)), float(np.mean(xs)))
        slots.append((key, color, arr == color))
    slots.sort(key=lambda item: item[0])
    assert slots[0][0] != slots[1][0], "variable slots must be vertically distinguishable"
    return [(color, mask) for _key, color, mask in slots]


def _make_receiver_round(receiver_proto: np.ndarray, donor_proto: np.ndarray, bg: int, rounds: int) -> np.ndarray:
    receiver_slots = _ordered_variable_slots(receiver_proto, bg)
    donor_slots = _ordered_variable_slots(donor_proto, bg)
    out = receiver_proto.copy()
    for idx in range(min(rounds, 2)):
        donor_color, _donor_mask = donor_slots[idx]
        _receiver_color, receiver_mask = receiver_slots[idx]
        out[receiver_mask] = donor_color
    return out


def _infer_scale(
    arr: np.ndarray,
    base_anchor_h: int,
    base_anchor_w: int,
    bg: int,
) -> tuple[int, tuple[int, int, int, int]]:
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_anc_h = ar1 - ar0 + 1
    inst_anc_w = ac1 - ac0 + 1
    scale_h = max(1, inst_anc_h // max(1, base_anchor_h))
    scale_w = max(1, inst_anc_w // max(1, base_anchor_w))
    return int(max(scale_h, scale_w)), anc


def _match_prototype(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    protos: list[np.ndarray],
    scale: int,
    anc: tuple[int, int, int, int],
    bg: int,
) -> int:
    h, w = g.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anc
    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = (r0 + ar0) - (pr0 * scale)
        left = (c0 + ac0) - (pc0 * scale)

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sub_g = g[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        score = int(np.sum(sub_g[mask] == sub_p[mask])) if np.any(mask) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    assert best_score >= 0, "each sprite must match a prototype"
    return best_idx


def _place_aligned(
    out: np.ndarray,
    proto: np.ndarray,
    scale: int,
    sprite_bbox: tuple[int, int, int, int],
    inst_anchor: tuple[int, int, int, int],
    bg: int,
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = sprite_bbox
    ar0, ac0, _ar1, _ac1 = inst_anchor
    scaled = _scale_nn(proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
    top = (r0 + ar0) - (pr0 * scale)
    left = (c0 + ac0) - (pc0 * scale)

    ph, pw = scaled.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    mask = sub_p != bg
    out_slice = out[rr0:rr1, cc0:cc1]
    out_slice[mask] = sub_p[mask]
    out[rr0:rr1, cc0:cc1] = out_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit donor colors onto only the upper-anchor family, showing round-1 and round-2 states by scale.

    Input structure:
    - The most common color is the background. Two base sprite prototypes appear as isolated scale-1 examples, plus enlarged nearest-neighbor copies.
    - Both prototypes share a color-1 anchor region and have two variable-color slots, but one family places the anchor visibly higher (receiver A) and the other lower (donor B).

    Rule:
    - Recover the two raw prototypes from the smallest-area sprites and order them by anchor height: upper-anchor first, lower-anchor second.
    - Keep every donor-B sprite unchanged. For each receiver-A sprite, preserve its own geometry and anchor alignment, but recolor its variable slots using B's ordered colors.
    - Prototype-sized A sprites stop after round 1 and receive only B's first visible slot color; enlarged A sprites continue to round 2 and receive both B slot colors.

    Key insight:
    - The transform is directional and iterative at once: family identity decides who can change, while size tier decides how many transfer rounds are completed. The unchanged B sprites prove the direction, and the small-versus-large A sprites reveal the intermediate state.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [_bbox_area(s["bbox"]) for s in sprites]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for sprite in base:
        key = tuple(sprite["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = sprite["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 raw prototypes"

    raw_protos = _order_prototypes([uniq[key] for key in list(uniq.keys())[:2]], bg)
    receiver_proto, donor_proto = raw_protos
    round1_proto = _make_receiver_round(receiver_proto, donor_proto, bg, rounds=1)
    round2_proto = _make_receiver_round(receiver_proto, donor_proto, bg, rounds=2)

    receiver_anchor = _pick_anchor_bbox(receiver_proto, bg)
    donor_anchor = _pick_anchor_bbox(donor_proto, bg)
    receiver_anchor_h = receiver_anchor[2] - receiver_anchor[0] + 1
    receiver_anchor_w = receiver_anchor[3] - receiver_anchor[1] + 1
    donor_anchor_h = donor_anchor[2] - donor_anchor[0] + 1
    donor_anchor_w = donor_anchor[3] - donor_anchor[1] + 1
    assert (receiver_anchor_h, receiver_anchor_w) == (donor_anchor_h, donor_anchor_w), "prototype anchors must share a size"

    out = g.copy()

    for sprite in sprites:
        scale, anc = _infer_scale(sprite["arr"], receiver_anchor_h, receiver_anchor_w, bg)
        family_idx = _match_prototype(g, sprite["bbox"], raw_protos, scale, anc, bg)
        if family_idx == 1:
            target_proto = donor_proto
        else:
            target_proto = round1_proto if scale == 1 else round2_proto

        for rr, cc in sprite["pts"]:
            out[rr, cc] = bg
        _place_aligned(out, target_proto, scale, sprite["bbox"], anc, bg)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SMALL_SCALE = 1
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


def _role_centroid(labels: np.ndarray, role: int) -> tuple[float, float]:
    ys, xs = np.where(labels == role)
    assert ys.size > 0, f"role {role} must be present"
    return float(np.mean(ys)), float(np.mean(xs))


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
        kinds, weights = zip(*weights_by_role[role])
        chosen = random.choices(kinds, weights=weights)[0]
        masks.append(_row_mask(chosen, w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
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
    if not (_direction_key(labels, 0) < _direction_key(flipped, 0)):
        return False
    ar0, _ac0, ar1, _ac1 = _anchor_bbox(labels)
    br0, _bc0, br1, _bc1 = _anchor_bbox(flipped)
    if ar0 + ar1 == br0 + br1:
        return False
    if _role_centroid(labels, 2) == _role_centroid(labels, 3):
        return False
    if _role_centroid(flipped, 2) == _role_centroid(flipped, 3):
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
    chosen = random.sample(pool, 4)
    a_cols = chosen[:2]
    b_cols = chosen[2:]

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    proto_a = _colorize(labels, a_cols)
    proto_b = _colorize(np.flipud(labels), b_cols)
    ordered = _order_prototypes([proto_a, proto_b], bg)
    receiver_proto, donor_proto = ordered
    receiver_colors = {color for color, _mask in _ordered_variable_slots(receiver_proto, bg)}
    donor_colors = {color for color, _mask in _ordered_variable_slots(donor_proto, bg)}
    assert receiver_colors.isdisjoint(donor_colors), "receiver and donor colors must be distinct"
    return receiver_proto, donor_proto


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    out_anchor = _anchor_bbox(patch_a)
    in_anchor = _anchor_bbox(patch_b)
    dr = out_anchor[0] - in_anchor[0]
    dc = out_anchor[1] - in_anchor[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _build_instances() -> list[tuple[int, int, int]]:
    receiver_big = random.choice(BIG_SCALES)
    donor_big = random.choice(BIG_SCALES)
    instances = [
        (0, SMALL_SCALE, 1),
        (1, SMALL_SCALE, 0),
        (0, receiver_big, 2),
        (1, donor_big, 0),
    ]
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    receiver_proto, donor_proto = _make_prototypes(bg)
    round1_proto = _make_receiver_round(receiver_proto, donor_proto, bg, rounds=1)
    round2_proto = _make_receiver_round(receiver_proto, donor_proto, bg, rounds=2)

    assert not np.array_equal(receiver_proto, round1_proto), "round-1 state must be visible"
    assert not np.array_equal(round1_proto, round2_proto), "round-2 state must add visible change"

    raw_protos = [receiver_proto, donor_proto]
    output_protos = {
        (0, 1): round1_proto,
        (0, 2): round2_proto,
        (1, 0): donor_proto,
    }
    instances = _build_instances()

    max_dim = max(max(output_protos[(family, stage)].shape) * scale for family, scale, stage in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for family, scale, stage in instances:
        input_patch = _scale(raw_protos[family], scale)
        output_patch = _scale(output_protos[(family, stage)], scale)
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
        paste(inp, input_patch, in_r, in_c, transparent=bg)
        paste(out, output_patch, out_r, out_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
