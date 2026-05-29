# Variant of 8e5c0c38
# Dimension: D1 — 对称修复的变换方式 (transform)
# Change: 删除没有镜像对应像素的噪声像素，使精灵恢复垂直镜像对称... → 补全缺失的镜像像素而非删除多余像素

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original_rule: keep the best vertical axis per color, then erase same-color pixels
#       whose mirror partner is missing.
#     new_rule: keep the same per-color axis search, but interpret every unmatched
#       pixel as evidence that its mirror partner is absent. The solver therefore adds
#       the missing mirror pixel instead of deleting the visible one.
#   concrete_grid_behavior:
#     input:
#       - exactly 3 non-background sprites, each a different color
#       - every sprite is a partially missing version of a vertically symmetric shape
#       - missing cells only remove one side of a mirrored pair; axis cells are never
#         deleted, and both members of the same pair are never deleted together
#     output:
#       - the completed symmetric sprites
#       - same colors, same object positions, but more filled pixels than the input
#   behavior_trace:
#     - Suppose the true axis is k=8 and a row contains color at columns 3 and 5
#       (since 3 + 5 = 8). In the original task, an extra pixel at 3 with no mate at 5
#       would be removed. In this variant, if column 5 is absent while column 3 remains,
#       the solver fills column 5. The observed asymmetry now means "missing partner",
#       not "noise to cut away".
#   complexity_calibration:
#     - Keep the original 22x22 canvas, 3 sprites, distinct colors, reserved placement
#       slots, and axis-gap validation so the task still requires per-object axis
#       inference rather than a trivial whole-grid mirror.
#     - Each sprite loses 1-12 pixels, roughly 11%-33% of the clean sprite, so the
#       completion is visually obvious across a few examples.
#   hidden_assumption_review:
#     - The original axis search still works because the input remains a subset of the
#       target symmetric sprite, so many mirrored pairs survive and the correct axis can
#       still be identified by maximum pair score.
#     - The old deletion loop relied on "unmatched means wrong". That assumption is no
#       longer valid. The new loop must read from the original grid `g` and write the
#       mirror into `out`, otherwise newly filled pixels could affect the interpretation
#       of other unmatched pixels.
#     - Filling only background cells avoids accidental overwrite; the generator
#       guarantees the intended mirror locations are empty in the input.
#   placement_legality:
#     - Generation is still output-first: build the full symmetric sprite, place it in a
#       non-overlapping slot, then delete one side of selected mirrored pairs to form the
#       input. Because deletions only shrink the sprite, placement remains in-bounds and
#       non-overlapping, and solver-added pixels stay inside the original placed output.
#   visual_observability:
#     - The difference from the source task is directly visible: outputs are larger than
#       inputs, and asymmetric protrusions are mirrored rather than pruned away.
#   color_reasoning:
#     - No color remapping is introduced; color is only used to separate sprites, which
#       keeps the rule human-inferable from shape changes alone.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color layer by restoring vertical mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors each form an almost-symmetric sprite about some vertical axis.
    - The asymmetry comes from missing mirror pixels, so each visible sprite is a subset of its intended symmetric shape.

    Rule:
    - For each non-bg color, find the vertical axis that best matches its existing pixels.
    - Whenever a same-color pixel has no mirror partner across that axis, add the partner at the reflected location.

    Key insight: Each color has its own independent symmetry axis, and an unmatched pixel is a clue that its mirror counterpart should be filled in rather than erased.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Search all possible vertical axes: k in [0, 2*(W-1)] corresponds to axis at c = k/2.
    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (W - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < W and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        # Fill any missing mirror partner under the chosen vertical axis.
        k = int(best_k)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if 0 <= cc < W and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


# --------------- generator (codex-att3, D1 inversion) ---------------

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

# ── CFG: missing mirror pixels ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
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


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    """Return the highest axis score excluding the true axis."""
    scores = _axis_scores(coords, width)
    return max(s for k, s in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying spec constraints."""
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
    """Pick a random (h, w, target_count, center_count), weighting toward denser fills."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left_count cells from the left half-grid (h x side_w), ensuring mild structure."""
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
    """Return boolean mask of which rows have a center-column pixel."""
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
        if _second_best_score(coords, true_k, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into slot regions on a fresh grid. Returns (grid, placed_specs) or None."""
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


def _removal_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Return removable cells: one side of a mirrored pair, never axis cells."""
    H, W = clean_mask.shape
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(clean_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if 0 <= mirror_c < W and clean_mask[r, mirror_c]:
            candidates.append((r, c))
    return candidates


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one-sided mirror partners from inp so main() must fill them back in."""
    bg = int(get_bg(out))
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    candidates = _removal_candidates(clean_mask, true_k)
    pair_count = len(candidates) // 2
    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)
    if lo > hi:
        return False

    missing_target = random.randint(lo, hi)
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)

    for _ in range(missing_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell in current_coords]
        if not viable:
            return False

        scored = []
        for cell in viable:
            test_coords = [pt for pt in current_coords if pt != cell]
            alt = _second_best_score(test_coords, true_k, GRID_SIZE)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)

        chosen.append(cell)
        current_coords.remove(cell)
        r, c = cell
        forbidden.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    scores = _axis_scores(current_coords, GRID_SIZE)
    true_score = scores[true_k]
    if true_score - max(s for k, s in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = bg
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, place them, delete mirrors, validate via solver."""
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
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _remove_pixels(inp, out, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
