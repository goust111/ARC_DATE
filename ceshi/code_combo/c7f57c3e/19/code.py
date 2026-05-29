# Combo variant of c7f57c3e (index 19)
# combo_spec: D7/1+D10/2
# Name: 尺度决定颜色顺序 + 全排列轮换

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Keep the original family structure: several disconnected multicolor sprites, each containing
#      color 1 as an anchor region and two variable-color regions, all created by integer NN scaling.
#    - Replace the original binary "proto_A <-> proto_B" toggle with three independent color modes on
#      one shared silhouette. The three visible sprites use scales 3, 2, and 1 exactly once each.
#    - The output performs a cyclic mode transfer, but the cycle is ordered by scale instead of by
#      pairing or position. If the input has large=A, medium=B, small=C, then the output is
#      large=C, medium=A, small=B because large passes to medium, medium passes to small, and small
#      closes the loop back to large.
#    - Compared with the original task, scale no longer selects "which opposite prototype to use";
#      it only determines the routing order of whole color modes.
# b) Complexity calibration:
#    - Exactly three sprites avoids the 2-sprite degenerate case, which would collapse the cycle back
#      to a simple swap.
#    - Using scales 1/2/3 exactly once removes tie ambiguity in the scale order while keeping the
#      visual vocabulary close to the source task.
#    - Each sprite uses its own ordered color pair with no reuse across sprites, so humans can track
#      full modes rather than single-cell recolorings.
# c) Implicit-assumption audit against the old solver:
#    - The old assumption "smallest-area sprites yield exactly 2 prototypes" is invalid; here the
#      scale-1 sprite is only the shared geometric template with one particular color mode.
#    - The old binary replacement "1 - best_idx" is invalid because replacement is now a 3-cycle.
#    - The old prototype-scoring logic is unnecessary; all sprites share one silhouette, so exact
#      NN downscaling recovers a canonical base pattern directly.
#    - The old meaning of scale as a preserved pairing constraint is gone. In this combo, scale is
#      the ordering key of the permutation chain.
# d) Placement legality:
#    - The generator places three bbox markers with border/padding checks, so sprites never overlap
#      and never leave the canvas.
#    - It validates each candidate by running the solver and only accepts samples whose output matches
#      the intended construction.
# e) Visual observability:
#    - The ordering signal is fully visible: one sprite is largest, one medium, one smallest.
#    - The generator rejects left-to-right scale permutations whose cyclic routing would coincide with
#      a position-based rotation rule, preventing accidental degeneracy.
# f) Visual cleanliness:
#    - Sprites stay disconnected with at least a two-cell gap and a one-cell border margin.
#    - Reusing one silhouette keeps the scene tidy while the scale contrast makes the ranking obvious.
# g) Change inferability:
#    - Only two variable colors move per sprite, and they move together as a full ordered mode tied
#      to the same structural slots. This gives enough repeated structure for humans to infer that the
#      whole mode, not individual pixels, is being routed by scale.
# h) New solver assumptions introduced here:
#    - All sprites share one base silhouette and one anchor layout.
#    - The instance scales are exactly {1, 2, 3}, so descending scale order is unique.
#    - Every sprite is an exact NN-scaled copy of the same base template.
# i) Generator guarantees for those assumptions:
#    - It constructs every sprite from one label template, assigns scales [1, 2, 3], uses distinct
#      color pairs, and rejects candidates where spatial order would induce the same cycle.


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
    """Recover the unscaled pattern and verify exact NN scaling."""
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "sprite size is not divisible by scale"
    base = arr[::scale, ::scale].copy()
    assert np.array_equal(_scale_nn(base, scale), arr), "sprite is not an exact NN-scaled copy"
    return base


def _center_key(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (c0 + c1, r0 + r1)


def _scale_order_key(sprite: dict) -> tuple[int, int, int]:
    """Largest scale first; visible position is only a fallback tie-break."""
    return (-sprite["scale"],) + _center_key(sprite["bbox"])


def _extract_sprites(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 3, "grid must contain exactly 3 sprite components"

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

    scales_seen = set()
    for sprite in sprites:
        arr = sprite["arr"]
        sh, sw = arr.shape
        assert sh % base_h == 0 and sw % base_w == 0, "sprite bbox is not a scaled copy of base shape"
        scale_h = sh // base_h
        scale_w = sw // base_w
        assert scale_h == scale_w, "non-uniform scaling is not allowed"
        scale = int(scale_h)
        assert scale in {1, 2, 3}, "unexpected scale"

        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        anc_h = ar1 - ar0 + 1
        anc_w = ac1 - ac0 + 1
        assert anc_h == base_anc_h * scale and anc_w == base_anc_w * scale, "anchor size mismatch"

        sprite["scale"] = scale
        sprite["anchor"] = (ar0, ac0, ar1, ac1)
        sprite["base_arr"] = _downscale_nn_exact(arr, scale)
        scales_seen.add(scale)

    assert scales_seen == {1, 2, 3}, "sprites must use scales 1, 2, and 3 exactly once"
    return sprites


def _place_aligned(out: np.ndarray, sprite: dict, donor_base: np.ndarray, bg: int) -> None:
    """Paste a scaled donor pattern using anchor alignment onto the recipient sprite location."""
    h, w = out.shape
    scale = sprite["scale"]
    repl = _scale_nn(donor_base, scale)

    r0, c0, _r1, _c1 = sprite["bbox"]
    ar0, ac0, _ar1, _ac1 = sprite["anchor"]
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(donor_base, bg)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite's full color mode around a cycle ordered by sprite scale.

    Input structure:
    - The background is the most common color. Exactly three disconnected sprites appear, all sharing
      one geometric template with a color-1 anchor region plus two variable-color regions.
    - The three sprites are exact nearest-neighbor enlargements of that template at scales 1, 2, and
      3, and each scale carries its own ordered pair of visible colors.

    Rule:
    - Recover each sprite's canonical unscaled pattern, sort the sprites by descending scale, and
      treat that order as a cyclic chain.
    - Transfer the largest sprite's color mode to the medium sprite, the medium sprite's mode to the
      smallest sprite, and the smallest sprite's mode back to the largest, while preserving each
      recipient sprite's own scale and anchor alignment.

    Key insight:
    - Shape and anchor stay fixed across all instances, so the only moving information is the full
      color mode. Once every sprite is downscaled to the common template, the task reduces to a
      scale-ranked cyclic permutation followed by rescaling onto the original footprints.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprites = _extract_sprites(g, bg)
    ordered = sorted(sprites, key=_scale_order_key)

    out = g.copy()
    for sprite in sprites:
        for rr, cc in sprite["pts"]:
            out[rr, cc] = bg

    for idx, sprite in enumerate(ordered):
        donor = ordered[(idx - 1) % len(ordered)]["base_arr"]
        _place_aligned(out, sprite, donor, bg)

    return out


# --------------- generator (combo variant) ---------------

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
SCALES = [1, 2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8
MIN_CENTER_GAP = 4

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


def _make_color_modes(bg: int) -> dict[int, tuple[int, int]]:
    """Pick one ordered color pair per scale with no reuse across sprites."""
    pool = [c for c in range(10) if c not in (bg, 1)]
    chosen = random.sample(pool, 2 * len(SCALES))
    modes: dict[int, tuple[int, int]] = {}
    for idx, scale in enumerate(SCALES):
        modes[scale] = (chosen[2 * idx], chosen[2 * idx + 1])
    return modes


def _scale_to_donor(scale: int) -> int:
    """Descending scale cycle: 3->2, 2->1, 1->3 means recipient gets previous donor."""
    ordered = sorted(SCALES, reverse=True)
    idx = ordered.index(scale)
    return ordered[(idx - 1) % len(ordered)]


def _center_x(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    del r0, r1
    return c0 + c1


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    labels = _make_labels()
    base_h, base_w = labels.shape

    max_dim = max(base_h, base_w) * max(SCALES)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    reserve = np.full((n, n), bg, dtype=int)

    placements = []
    for scale in sorted(SCALES, reverse=True):
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

    ordered_x = sorted(placements, key=lambda rec: (_center_x(rec["bbox"]), rec["bbox"][0] + rec["bbox"][2]))
    centers = [_center_x(rec["bbox"]) for rec in ordered_x]
    if any(b - a < MIN_CENTER_GAP for a, b in zip(centers, centers[1:], strict=False)):
        return None
    if tuple(rec["scale"] for rec in ordered_x) not in {(1, 3, 2), (2, 1, 3)}:
        return None

    modes = _make_color_modes(bg)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)

    for rec in placements:
        scale = rec["scale"]
        r0, c0, _r1, _c1 = rec["bbox"]
        inp_patch = _scale_nn(_colorize(labels, modes[scale], bg), scale)
        donor_scale = _scale_to_donor(scale)
        out_patch = _scale_nn(_colorize(labels, modes[donor_scale], bg), scale)
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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
