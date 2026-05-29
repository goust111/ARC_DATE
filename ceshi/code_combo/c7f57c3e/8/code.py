# Combo variant of c7f57c3e (index 8)
# combo_spec: D5/1+D8/0
# Name: 堆叠到公共区域 + 紧凑裁剪

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# combo_spec:
#   core_change_1:
#     old_dst: each swapped sprite stays near its source through local anchor alignment
#     new_dst: all swapped sprites are moved into one shared vertical stack
#   core_change_2:
#     old_grid_shape: output keeps the full input canvas
#     new_grid_shape: after stacking, keep only the tight crop around all non-bg pixels
#   preserved_parts:
#     - exactly two prototype families still define the task
#     - every sprite is still an integer-scaled copy of one prototype
#     - every sprite still swaps to the opposite prototype at the same scale
#     - the smallest-area scale-1 sprites still expose the two base prototypes
#
# a) concretization_and_trace:
#   - Input: 3-4 separated multicolor sprites on a large square background; one sprite has scale 2
#     or 3, the others have scale 1, and both prototype families appear.
#   - Original behavior: a bottom-right scale-3 proto_A instance would swap to scale-3 proto_B and
#     stay near bottom-right because the anchor bbox fixes the local replacement position.
#   - New behavior: the same instance is still recognized as proto_A at scale 3, but after the
#     swap it is appended into a shared top-to-bottom stack together with all other swapped sprites.
#   - Final difference: the old output preserved the input canvas and approximate positions; the
#     new output discards absolute placement entirely and returns only the cropped stack.
#
# b) complexity_calibration:
#   - Prototype recognition is unchanged, so the solver still has to infer family and scale rather
#     than reading them off from color ids.
#   - A single vertical stack keeps the relocation rule visually strong without introducing many
#     layout modes. Reading order is geometric and human-visible.
#   - Tight cropping amplifies the D5 change: the output is not just rearranged inside the same
#     canvas, it also becomes much smaller, which makes the fusion easy to notice from examples.
#
# c) implicit_assumption_review_for_original_main:
#   - Connected-component extraction still works only if input sprites do not touch; the generator
#     keeps padding between sprite bboxes.
#   - "Smallest-area sprites reveal the two prototypes" still has to hold, so the generator keeps
#     one scale-1 instance of each prototype family.
#   - Scale inference still depends on the color-1 anchor region scaling by an integer factor.
#   - The original output started from a copy of the input and cleared/replaced locally; that
#     assumption is invalid now because the whole dispersed layout is discarded.
#   - Absolute input coordinates are no longer output coordinates; they are used only to sort the
#     sprites in reading order before stacking.
#
# d) placement_legality:
#   - Input sprites are placed with border and padding constraints, so they stay in-bounds and
#     remain separate components.
#   - Only one sprite is allowed to have scale 2 or 3; all others stay at scale 1, which preserves
#     the original prototype-discovery logic.
#   - Output placement is deterministic: stacked sprites never overlap, and a 1-cell gap prevents
#     accidental touching that would blur object boundaries after cropping.
#
# e) visual_observability:
#   - Across examples, the transformation is readable by eye: scattered sprites become a compact
#     top-to-bottom column of swapped sprites.
#   - The crop is also visible because the output size shrinks from a large square input to a much
#     narrower rectangle.
#   - Ordering is based on visible input geometry, not on hidden attributes such as color numbers.
#
# f) visual_cleanliness:
#   - The output contains a single organized stack with uniform spacing and no overlay.
#   - Each sprite keeps its own scaled shape, so large-vs-small structure remains legible even
#     after all items move into one area.
#
# g) color_inference:
#   - No extra recoloring rule is introduced. Colors still come directly from the opposite
#     prototype, so the only new ideas are global stacking and tight cropping.
#
# h) generator_non_degeneracy_metric:
#   - `crop_area_ratio = output_area / input_area` should stay clearly below 1, showing that D8 is
#     active rather than leaving the full square canvas intact.
#   - `shape_changed = output.shape != input.shape` should hold for generated samples, confirming
#     that the fused task does not collapse back to the original same-size output behavior.
#
# i) new_main_assumptions_to_check:
#   - Sorting assumes sprite bbox top-left corners are distinct enough to define reading order; the
#     generator enforces non-overlap so ties are not problematic.
#   - Tight crop assumes at least one swapped non-bg pixel exists; every prototype contains color 1
#     and other foreground pixels, so this is guaranteed.
#   - Stacking assumes a deterministic gap policy; both solver and generator share the same
#     `STACK_GAP`, so expected outputs remain stable.
# ---------------------------------------------------------------------------


STACK_GAP = 1


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 anchor pixels inside a sprite array."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _stack_required_height(patches: list[np.ndarray]) -> int:
    total = sum(int(p.shape[0]) for p in patches)
    if len(patches) >= 2:
        total += STACK_GAP * (len(patches) - 1)
    return total


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != bg)
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (int(r0), int(c0), int(r1), int(c1)), "arr": arr})
    return sprites


def _discover_prototypes(
    sprites: list[dict],
    bg: int,
) -> tuple[list[np.ndarray], int, int]:
    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    protos = [arr.copy() for arr in list(uniq.values())[:2]]
    a0r0, a0c0, a0r1, a0c1 = _pick_anchor_bbox(protos[0], bg)
    base_anc_h = int(a0r1 - a0r0 + 1)
    base_anc_w = int(a0c1 - a0c0 + 1)
    return protos, base_anc_h, base_anc_w


def _match_sprite(
    grid: np.ndarray,
    sprite: dict,
    protos: list[np.ndarray],
    base_anc_h: int,
    base_anc_w: int,
    bg: int,
) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    arr = sprite["arr"]

    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_anc_h = int(ar1 - ar0 + 1)
    inst_anc_w = int(ac1 - ac0 + 1)
    scale_h = max(1, inst_anc_h // max(1, base_anc_h))
    scale_w = max(1, inst_anc_w // max(1, base_anc_w))
    scale = int(max(scale_h, scale_w))

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = int(r0 + ar0 - pr0 * scale)
        left = int(c0 + ac0 - pc0 * scale)

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + scaled.shape[0])
        cc1 = min(w, left + scaled.shape[1])
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx

    assert best_score >= 0, "failed to match sprite to a prototype"
    return int(best_idx), scale


def _build_stacked_output(bg: int, ordered_patches: list[np.ndarray]) -> np.ndarray:
    assert ordered_patches, "need at least one patch to stack"
    stack_h = _stack_required_height(ordered_patches)
    max_w = max(int(p.shape[1]) for p in ordered_patches)
    canvas = np.full((stack_h, max_w), int(bg), dtype=int)
    r = 0
    for i, patch in enumerate(ordered_patches):
        paste(canvas, patch, r, 0, transparent=bg)
        r += int(patch.shape[0])
        if i + 1 < len(ordered_patches):
            r += STACK_GAP
    return _tight_crop(canvas, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, gather all replacements into one shared stack,
    then return only the stack's tight crop.

    Input structure:
    - bg is the most common color. Two distinct small multicolor prototype patterns appear at
      several integer scales.
    - Each foreground component is a scaled copy of one prototype, and the components are scattered
      across a larger square canvas.

    Rule:
    - Recover the two prototypes from the smallest-area sprites.
    - For each sprite, infer which prototype it matches and its scale from the color-1 anchor bbox.
    - Replace it with the other prototype at the same scale, sort all replacements in reading
      order of the input sprites, stack them top-to-bottom with a 1-cell gap, and tight-crop.

    Key insight: Prototype recognition still follows the original anchor-based matching, but once
    the swaps are known, absolute positions are thrown away; only reading order survives into a
    compact stacked output.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprites = _extract_sprites(g, bg)
    protos, base_anc_h, base_anc_w = _discover_prototypes(sprites, bg)

    ordered: list[tuple[int, int, np.ndarray]] = []
    for sprite in sprites:
        proto_idx, scale = _match_sprite(g, sprite, protos, base_anc_h, base_anc_w, bg)
        swapped = _scale_nn(protos[1 - proto_idx], scale)
        r0, c0, _r1, _c1 = sprite["bbox"]
        ordered.append((int(r0), int(c0), swapped))

    ordered.sort(key=lambda item: (item[0], item[1]))
    return _build_stacked_output(bg, [patch for _r0, _c0, patch in ordered])


# --------------- generator ---------------

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
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

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


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


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


def _expected_output(bg: int, protos: list[np.ndarray], placed: list[dict]) -> np.ndarray:
    ordered = sorted(placed, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    swapped_patches = [_scale_nn(protos[1 - item["type"]], item["scale"]) for item in ordered]
    return _build_stacked_output(bg, swapped_patches)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))

    records = []
    for typ, scale in instances:
        patch = _scale_nn(protos[typ], scale)
        records.append(
            {
                "type": int(typ),
                "scale": int(scale),
                "patch": patch,
                "area": int(patch.shape[0] * patch.shape[1]),
            }
        )

    max_dim = max(max(int(rec["patch"].shape[0]), int(rec["patch"].shape[1])) for rec in records)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None

    n = random.randint(min_n, GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    placed = []
    for rec in sorted(records, key=lambda item: item["area"], reverse=True):
        marker = np.full(rec["patch"].shape, 1, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(inp, rec["patch"], r0, c0, transparent=bg)
        paste(reserve, marker, r0, c0)
        placed.append(
            {
                "type": rec["type"],
                "scale": rec["scale"],
                "bbox": (int(r0), int(c0), int(r0 + rec["patch"].shape[0] - 1), int(c0 + rec["patch"].shape[1] - 1)),
            }
        )

    expected = _expected_output(bg, protos, placed)
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if produced.shape == inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
