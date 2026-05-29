# Variant of c7f57c3e
# Dimension: D5 — 目标位置/锚定方式 (dst)
# Change: 交换后精灵保持在与配对精灵最大重叠的位置，本质上近似原位交换... → 堆叠到公共区域

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
#
# variant_spec:
#   core_change:
#     old_dst: each swapped sprite stays near its original location via anchor alignment
#     new_dst: all swapped sprites are moved into one shared vertical strip and stacked
#              from top to bottom, in input reading order (top-to-bottom, then left-to-right)
#   preserved_parts:
#     - two prototype families still exist
#     - every sprite is still an integer-scaled copy of one prototype
#     - every sprite still swaps to the opposite prototype at the same scale
#     - smallest-area sprites still reveal the two base prototypes
#
# a) concretization_and_trace:
#   input_view:
#     - 3-4 scattered multicolor sprites on a square background
#     - both prototype families appear; exactly one sprite has scale 2 or 3
#     - sprites are separated, so connected-component extraction still isolates them cleanly
#   output_view:
#     - the original scattered layout disappears entirely
#     - swapped sprites are re-drawn in a common left-side strip
#     - they are stacked vertically with a 1-cell gap between consecutive sprites
#   local_trace_example:
#     - original task: a scale-3 proto_A instance at bottom-right is replaced by scale-3 proto_B
#       and kept anchor-aligned near bottom-right
#     - new task: the same instance is still identified as proto_A at scale 3, but after swapping
#       it is pasted into the next slot of the shared stack strip instead of its old location
#     - difference: identification logic is unchanged; only destination layout changes
#
# b) complexity_calibration:
#   - keeping two nontrivial prototypes plus multiple scales preserves the original matching burden
#   - stacking all outputs into one strip makes the D5 change visually strong
#   - ordering uses visible geometry only (reading order of input bboxes), not color ids
#   - using a vertical stack avoids accidental simplification where outputs still resemble the
#     original dispersed layout
#
# c) implicit_assumption_review_for_original_main:
#   - extracting connected components still works because sprites remain non-overlapping
#   - "smallest-area sprites reveal the two prototypes" must still hold, so the generator keeps
#     one scale-1 instance of each prototype
#   - scale inference still assumes the anchor region scales by the same integer factor
#   - the old code's in-place clearing and anchor-based re-placement depended on destination
#     staying local; this is the exact assumption removed here
#   - the new output cannot start from a copy of the input, otherwise stale original sprites would
#     remain; it must start from an all-background canvas
#
# d) placement_legality:
#   - generator places input sprites with bbox padding and without overlap, exactly as before
#   - generator also computes the total stacked height in advance and only samples grid sizes that
#     can contain the final vertical stack
#   - the shared stack strip is reserved away from input placements so the input visibly differs
#     from the output and so generator validation is stable
#
# e) visual_observability:
#   - across examples, the distinctive change is easy to see: scattered objects become a vertical
#     column on the left
#   - scale is preserved, so large objects remain large inside the stack
#   - prototype swapping is still visible because the two families keep different internal colors
#     and a vertical flip relationship
#
# f) visual_cleanliness:
#   - output contains only one organized strip; there is no overlap between stacked sprites
#   - a 1-cell gap between stacked objects prevents ambiguous touching/merging
#   - reserving the strip in the input avoids clutter from input objects already occupying the same
#     area that the output later uses
#
# g) color_inference:
#   - no new color rule is introduced
#   - colors still belong to the swapped prototype itself; the variant changes destination only
#
# h) generator_non_degeneracy_metric:
#   - the variant-specific metric is "moved_components": the number of output sprites whose
#     top-left corner differs from the corresponding input sprite top-left corner
#   - the intended distribution is all components moved in every generated sample
#
# i) new_main_assumptions_to_check:
#   - reading-order sorting assumes bbox top-lefts are well-defined and distinct; generator ensures
#     sprites do not overlap so ties do not occur
#   - stacked output assumes the total required height fits in the canvas; generator enforces this
#   - prototype matching still assumes every sprite is an exact scaled prototype instance
# ---------------------------------------------------------------------------


STACK_GAP = 1
STACK_COL = 1

# --------------- shared helpers ---------------


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


def _stack_required_height(patches: list[np.ndarray]) -> int:
    total = sum(int(p.shape[0]) for p in patches)
    if len(patches) >= 2:
        total += STACK_GAP * (len(patches) - 1)
    return total


def _build_stacked_output(
    shape: tuple[int, int],
    bg: int,
    ordered_patches: list[np.ndarray],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    r = 0
    for i, patch in enumerate(ordered_patches):
        paste(out, patch, r, STACK_COL, transparent=bg)
        r += int(patch.shape[0])
        if i + 1 < len(ordered_patches):
            r += STACK_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then stack all swapped sprites into one shared strip.

    Input structure:
    - bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
    - Each sprite instance is a scaled copy of one of the two prototypes, scattered on the grid.

    Rule:
    - Identify the two prototype patterns from the smallest-area sprites.
    - For each sprite, determine which prototype it matches and its integer scale factor.
    - Replace it with the other prototype at the same scale, then discard the old layout and stack
      all replacements from top to bottom in reading order inside a common left-side area.

    Key insight: Prototype recognition is the same as the original task; the variant only changes
    the destination policy, turning a local swap into a global re-layout in one shared stack zone.
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

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

    ordered = []
    swapped_patches = []

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
            pr0, pc0, _pr1, _pc1 = panc
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

        repl_idx = 1 - best_idx
        swapped = _scale_nn(protos[repl_idx], scale)
        ordered.append((int(r0), int(c0), swapped))
        swapped_patches.append(swapped)

    ordered.sort(key=lambda item: (item[0], item[1]))
    stack_h = _stack_required_height([patch for _, _, patch in ordered])
    max_w = max(int(p.shape[1]) for p in swapped_patches)
    assert stack_h <= h, "stacked output must fit vertically"
    assert STACK_COL + max_w <= w, "stacked output must fit horizontally"

    return _build_stacked_output((h, w), bg, [patch for _, _, patch in ordered])


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28

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
INPUT_STACK_STRIP_BUFFER = 1

# --- CFG: generation retries ---
MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


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


def _reserve_left_strip(canvas: np.ndarray, width: int) -> None:
    canvas[:, :width] = 1


def _expected_output(
    inp: np.ndarray,
    bg: int,
    protos: list[np.ndarray],
    placed: list[dict],
) -> np.ndarray:
    ordered = sorted(placed, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    patches = [_scale(protos[1 - item["type"]], item["scale"]) for item in ordered]
    return _build_stacked_output(inp.shape, bg, patches)


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
        patch = _scale(protos[typ], scale)
        records.append(
            {
                "type": int(typ),
                "scale": int(scale),
                "patch": patch,
                "area": int(patch.shape[0] * patch.shape[1]),
            }
        )

    max_patch_w = max(int(rec["patch"].shape[1]) for rec in records)
    max_patch_h = max(int(rec["patch"].shape[0]) for rec in records)
    stack_h = _stack_required_height([rec["patch"] for rec in records])
    strip_w = STACK_COL + max_patch_w + INPUT_STACK_STRIP_BUFFER

    min_n = max(
        GRID_SIZE_MIN,
        stack_h,
        strip_w + max_patch_w + 1,
        max_patch_h + 2,
    )
    if min_n > GRID_SIZE_MAX:
        return None

    n = random.randint(min_n, GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)
    _reserve_left_strip(reserve, strip_w)

    placed = []
    for rec in sorted(records, key=lambda item: item["area"], reverse=True):
        patch = rec["patch"]
        marker = np.full(patch.shape, 1, dtype=int)
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

        paste(inp, patch, r0, c0, transparent=bg)
        paste(reserve, marker, r0, c0)
        placed.append(
            {
                "type": rec["type"],
                "scale": rec["scale"],
                "bbox": (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)),
            }
        )

    expected = _expected_output(inp, bg, protos, placed)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D5 input")
