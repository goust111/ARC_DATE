# Variant of c7f57c3e
# Dimension: D10 — 精灵数量与分组基数 (multi_step)
# Change: 精灵数量不定，分为恰好两组，组间1:1或多:多配对... → 每个精灵独立一组（全排列）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the original "small multicolor sprite with color-1 anchor + integer scaling" setup.
#    - Replace the old two-prototype world with one shared geometric template and 3-4 sprites whose
#      two variable colors are all unique, so every sprite has its own visible color mode.
#    - Sort sprites by horizontal position (bbox center, left to right). Output rotates these color
#      modes cyclically: sprite i receives sprite i+1's mode; the rightmost sprite receives the
#      leftmost mode.
#    - Small example with 3 sprites A/B/C from left to right:
#      input  = [A, B, C]
#      output = [B, C, A]
#      The original code would have tried to classify each sprite into one of two prototypes and
#      toggle A<->B. That classification logic is invalid once all sprites are unique.
# b) Complexity calibration:
#    - Using 3-4 sprites avoids the degenerate 2-sprite case, which would collapse back to a swap.
#    - Each sprite gets an ordered pair of colors with no reuse across sprites, so humans can track
#      full modes rather than single cells.
#    - Exactly one sprite is scaled up (2x or 3x), preserving the original scale variation while
#      keeping inference manageable.
# c) Implicit-assumption audit against the old solver:
#    - "smallest-area sprites yield exactly 2 distinct prototype patterns" no longer holds; the
#      variant can have 3-4 distinct patterns at scale 1.
#    - "replacement index = 1 - best_idx" assumes a binary toggle; this must become cyclic routing.
#    - Prototype matching by score against two candidates is unnecessary; instead we recover each
#      sprite's canonical base pattern by exact NN downscaling.
#    - The old placement code used anchor alignment to compensate for different prototype geometry.
#      We keep anchor alignment in the solver even though the generator uses one shared silhouette,
#      so scale/placement handling stays robust and close to the original task family.
# d) Placement legality:
#    - Generator still places bbox-sized markers with border/padding checks and pastes only inside
#      the grid.
#    - Sprites are validated with the solver itself before accepting a sample.
# e) Visual observability:
#    - Ordering uses visible left-to-right position only; no color-id sorting or hidden metadata.
#    - Unique color pairs make the cyclic shift obvious in 2-4 demonstrations.
# f) Visual cleanliness:
#    - Sprites remain disconnected, with bbox padding and an extra check that horizontal centers are
#      separated enough to suggest a readable sequence.
# g) Color inference:
#    - Only two variable colors move per sprite, always inside the same structural roles. Humans see
#      that entire color modes travel, not arbitrary per-pixel recoloring.
# h) New solver assumptions introduced here:
#    - All sprites share one unscaled silhouette and anchor layout.
#    - At least one sprite appears at scale 1, so the base shape can be recovered from the smallest
#      bbox.
#    - Sprite horizontal centers are distinct, so left-to-right order is well-defined.
# i) Generator guarantees for those assumptions:
#    - It creates all sprites from one label template, includes multiple scale-1 sprites, and rejects
#      placements whose horizontal centers are too close.


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 anchor pixels; they are guaranteed to exist."""
    del bg
    ys, xs = np.where(arr == 1)
    assert len(ys) > 0, "sprite array has no color-1 pixels"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _downscale_nn_exact(arr: np.ndarray, scale: int) -> np.ndarray:
    """Recover the unscaled sprite and verify the instance is an exact NN enlargement."""
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "sprite size is not divisible by scale"
    base = arr[::scale, ::scale].copy()
    assert np.array_equal(_scale_nn(base, scale), arr), "sprite is not an exact NN-scaled copy"
    return base


def _sprite_order_key(sprite: dict) -> tuple[int, int]:
    """Visible left-to-right order by bbox center; row breaks ties."""
    r0, c0, r1, c1 = sprite["bbox"]
    return (c0 + c1, r0 + r1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite's unique color mode to the previous position in left-to-right order.

    Input structure:
    - bg is the most common color. Several disconnected sprites share one geometric template and a
      color-1 anchor sub-region, but each sprite has its own ordered pair of variable colors.
    - Every sprite is an integer-scaled nearest-neighbor copy of that template; typically one sprite
      is larger while the others stay at base scale.

    Rule:
    - Detect every sprite, recover its canonical unscaled pattern, and sort sprites by visible
      horizontal position from left to right.
    - Replace each sprite with the next sprite's canonical pattern in that order, wrapping the last
      sprite back to the first, while preserving the recipient sprite's scale and anchor alignment.

    Key insight:
    - Geometry is shared across all sprites, so the only thing that circulates is the full color
      mode. Downscale each instance to a common template, rotate those templates by position, then
      rescale the donor template back onto the recipient sprite.
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

    base_sprite = min(
        sprites,
        key=lambda s: (s["arr"].shape[0] * s["arr"].shape[1], s["arr"].shape[0], s["arr"].shape[1]),
    )
    base_arr = base_sprite["arr"]
    base_h, base_w = base_arr.shape
    ba_r0, ba_c0, ba_r1, ba_c1 = _pick_anchor_bbox(base_arr, bg)
    base_anc_h = ba_r1 - ba_r0 + 1
    base_anc_w = ba_c1 - ba_c0 + 1

    for sprite in sprites:
        arr = sprite["arr"]
        sh, sw = arr.shape
        assert sh % base_h == 0 and sw % base_w == 0, "sprite bbox is not a scaled copy of base shape"
        scale_h = sh // base_h
        scale_w = sw // base_w
        assert scale_h == scale_w, "non-uniform scaling is not allowed"
        scale = int(scale_h)

        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        anc_h = ar1 - ar0 + 1
        anc_w = ac1 - ac0 + 1
        assert anc_h == base_anc_h * scale and anc_w == base_anc_w * scale, "anchor size mismatch"

        sprite["scale"] = scale
        sprite["anchor"] = (ar0, ac0, ar1, ac1)
        sprite["base_arr"] = _downscale_nn_exact(arr, scale)

    ordered = sorted(sprites, key=_sprite_order_key)
    out = g.copy()

    for sprite in sprites:
        for rr, cc in sprite["pts"]:
            out[rr, cc] = bg

    for idx, sprite in enumerate(ordered):
        donor = ordered[(idx + 1) % len(ordered)]["base_arr"]
        scale = sprite["scale"]
        repl = _scale_nn(donor, scale)

        r0, c0, _r1, _c1 = sprite["bbox"]
        ar0, ac0, _ar1, _ac1 = sprite["anchor"]
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(donor, bg)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = repl.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub = repl[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (D10 variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes / palette ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

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
MIN_CENTER_GAP = 3

# --- CFG: generation retries ---
MAX_RETRIES = 300

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    """Build a shared structural template with anchor cells, two variable regions, and holes."""
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
        masks.append(_row_mask(random.choices(kinds, weights=weights)[0], w))

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


def _colorize(labels: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    arr = np.full(labels.shape, bg, dtype=int)
    arr[labels == 1] = 1
    arr[labels == 2] = colors[0]
    arr[labels == 3] = colors[1]
    return arr


def _make_color_modes(bg: int, count: int) -> list[tuple[int, int]]:
    """Pick count ordered color pairs with no reuse across sprites."""
    pool = [c for c in range(10) if c not in (bg, 1)]
    chosen = random.sample(pool, 2 * count)
    return [(chosen[2 * i], chosen[2 * i + 1]) for i in range(count)]


def _center_x(record: dict) -> int:
    r0, c0, r1, c1 = record["bbox"]
    del r0, r1
    return c0 + c1


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    labels = _make_labels()
    base_h, base_w = labels.shape

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    scales = [1, 1, big_scale]
    if count == 4:
        scales.append(1)

    max_dim = max(base_h, base_w) * max(scales)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    reserve = np.full((n, n), bg, dtype=int)

    placements = []
    for scale in sorted(scales, reverse=True):
        marker = np.full((base_h * scale, base_w * scale), 1, dtype=int)
        try:
            r, c = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(reserve, marker, r, c, transparent=bg)
        placements.append({"bbox": (r, c, r + marker.shape[0] - 1, c + marker.shape[1] - 1), "scale": scale})

    ordered = sorted(placements, key=lambda rec: (_center_x(rec), rec["bbox"][0] + rec["bbox"][2]))
    centers = [_center_x(rec) for rec in ordered]
    if any(b - a < MIN_CENTER_GAP for a, b in zip(centers, centers[1:], strict=False)):
        return None

    modes = _make_color_modes(bg, count)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)

    for i, rec in enumerate(ordered):
        scale = rec["scale"]
        r0, c0, _r1, _c1 = rec["bbox"]
        inp_patch = _scale_nn(_colorize(labels, modes[i], bg), scale)
        out_patch = _scale_nn(_colorize(labels, modes[(i + 1) % count], bg), scale)
        paste(inp, inp_patch, r0, c0, transparent=bg)
        paste(out, out_patch, r0, c0, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D10 input")
