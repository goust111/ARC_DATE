# Variant of 8e5c0c38
# Dimension: D7 — 网格尺寸与布局 (grid_shape)
# Change: 22x22固定网格，3个精灵放置在预定义的互不重叠区域（slot布局），精灵间最小间距>=4... → 紧凑裁剪输出

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, paste

# Design analysis
# ===============
# spec:
#   preserved_core:
#     input_canvas: fixed 22x22 noisy scene with exactly 3 non-bg sprite colors
#     per_sprite_rule: each sprite is almost vertically symmetric around its own best axis
#     corruption: extra same-color pixels appear without mirrored partners and must be removed
#     placement: sprites stay in separated slots so colors do not interact spatially
#   variant_change:
#     output_canvas: no longer keep the repaired scene on the full 22x22 canvas
#     new_output_rule: after all colors are denoised, crop to the minimum bbox covering every repaired non-bg pixel
#   behavior_trace:
#     original:
#       if repaired sprites occupy rows 1..18 and cols 2..19, the solver still returned a 22x22 grid
#       containing a blank border around the cleaned sprites.
#     variant:
#       the same denoising step happens first, but the final answer becomes rows 1..18 by cols 2..19 only,
#       so the border disappears and the output shape changes to 18x18.
#     difference:
#       the per-color symmetry inference is unchanged; the extra reasoning step is to infer the global crop box
#       after reconstruction, not from the noisy bbox.
#   complexity_calibration:
#     keep the original 3-sprite slot layout, color separation, symmetry-axis search, and local noise model.
#     This makes the only new freedom the crop boundary, which is visually obvious but not combinatorially messy.
#     Rejecting uncropped 22x22 outputs ensures the D7 change is always visible in examples.
#   implicit_assumption_review_main:
#     - `get_bg(g)` still assumes one majority background color; generator preserves that.
#     - iterating `colors_present(g, exclude={bg})` still assumes each foreground color belongs to one sprite;
#       generator keeps colors distinct across sprites.
#     - scanning all vertical axes across the full input width still works because the input remains 22x22.
#     - deleting unpaired pixels into `out` is still safe because sprites do not overlap by color or position.
#     - new crop step assumes at least one repaired foreground pixel remains; generator constructs symmetric cores
#       with 6-46 pixels, so denoising never deletes everything.
#   placement_legality:
#     slots remain non-overlapping and far apart, so noisy expansions cannot merge objects.
#     The clean output bbox is taken after denoising, so crop boundaries are determined by repaired sprites only.
#   visual_observability:
#     humans can still see "broken bilateral shapes become repaired", and now also see that the surrounding
#     empty frame disappears. This is readable from 2-3 examples because the output shape consistently shrinks.
#   cleanliness:
#     no extra markers or color remapping are introduced; only empty border removal changes the layout.


def _tight_crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    """Crop to the minimum bbox containing all non-bg pixels."""
    fg = grid != int(bg)
    bb = bbox(fg)
    if bb is None:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(grid, bb)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each color layer by vertical symmetry, then trim the empty outer frame.

    Input structure:
    - The input is still a 22x22 scene with one dominant background color and several non-bg sprite colors.
    - Each non-bg color forms one nearly vertically symmetric sprite, plus a few same-color noise pixels without mirrored partners.

    Rule:
    - For every non-bg color independently, find the vertical axis that best matches its current pixels and delete unpaired pixels.
    - After all sprites are repaired, return only the smallest rectangular crop that contains every remaining non-bg pixel.

    Key insight: Symmetry recovery is still per color, but the final frame is not fixed anymore — the output bounds are the global tight crop of the repaired scene.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Candidate axes use k in [0, 2*(W-1)], representing the vertical line c = k/2.
    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (w - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < w and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        k = int(best_k)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < w and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return _tight_crop_foreground(out, bg)


# --------------- generator (codex-att3, D7 tight-crop variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int = GRID_SIZE) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate vertical axis."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[k] = score
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int) -> int:
    """Return the highest axis score excluding the true axis."""
    scores = _axis_scores(coords)
    return max(s for k, s in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying sprite constraints."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
            if target_count % 2 != 0:
                continue
            for side_w in EVEN_SIDE_W_RANGE:
                w = 2 * side_w
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick a random clean-sprite spec, mildly favoring denser bboxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells while keeping the local sprite reasonably coherent."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    """Return a boolean row mask indicating which rows receive a center-column pixel."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a vertically symmetric sprite and return (sprite_array, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[r, c] = color
            sprite[r, mirror_c] = color
        if center_count > 0:
            sprite[centers, side_w] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, true_k) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place clean sprites into their slot regions on a fresh 22x22 canvas."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = r0
        spec["c0"] = c0
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Find 4-adjacent cells that can host asymmetric noise without pairing."""
    _h, w = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    cols = np.arange(w)
    mirror_cols = true_k - cols

    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    valid_mirror = (mirror_cols >= 0) & (mirror_cols < w) & ~on_axis
    for c in np.where(valid_mirror)[0]:
        adj[:, c] &= ~clean_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add asymmetric noise pixels for one sprite; return False if constraints fail."""
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_k)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _second_best_score(current_coords + [cell], true_k)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = cell
        forbidden.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    if clean_count - _second_best_score(current_coords, true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build a noisy 22x22 input whose repaired answer is a strict tight crop."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "axis_k_local": axis_k_local})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out_full, placed_specs = result

    inp = out_full.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out_full, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out_full = np.flipud(out_full)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out_full = np.fliplr(out_full)

    expected = _tight_crop_foreground(out_full, bg)
    if expected.shape == (GRID_SIZE, GRID_SIZE):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
