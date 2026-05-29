# Combo variant of 8e5c0c38 (index 32)
# combo_spec: D1/0+D4/2
# Name: 缺失像素补全修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# combo_behavior:
#   original_task:
#     - Each non-background color is treated independently.
#     - The solver searches for the best vertical axis for that color.
#     - Any same-color pixel without a mirrored partner is deleted as additive noise.
#   fused_task:
#     - Keep the same per-color axis search, but reinterpret asymmetry.
#     - The visible shape is now a damaged subset of a symmetric sprite:
#       some pixels that should exist were removed from one side of mirrored pairs.
#     - The solver therefore restores the missing partner instead of deleting the
#       surviving pixel.
#   concrete_trace:
#     - Example row under axis k=8: a clean symmetric pair would occupy columns 3 and 5.
#     - Original behavior on an observed lone pixel at column 3: erase column 3.
#     - New behavior on the same observation: fill column 5.
#     - Difference: the source task shrinks shapes toward symmetry; this variant grows
#       them back to symmetry.
# complexity_calibration:
#   - Preserve the 22x22 canvas, exactly 3 colored sprites, distinct colors, and the
#     reserved placement slots from the source generator so the task still requires
#     inferring a separate axis per object rather than mirroring the whole scene.
#   - Preserve the original sprite size, density, and axis-gap constraints so the best
#     axis remains visually recoverable from a few surviving mirrored pairs.
#   - Missing pixels stay in the original 11%-33% / at most 12 range so the effect is
#     obvious but does not destroy the axis evidence.
# hidden_assumption_review:
#   - The axis-scoring loop assumes the correct axis still maximizes mirrored matches.
#     That remains valid because the input is a subset of the true symmetric output,
#     not an arbitrary corruption.
#   - The old deletion loop assumed "unmatched means wrong". That assumption is broken
#     by D1/0 and D4/2. The new loop must instead treat each unmatched visible pixel as
#     evidence that its mirror partner is missing.
#   - The fill loop must read from the original grid `g`, not from `out`, otherwise a
#     newly inserted mirror could cascade and justify extra fills that were not present
#     in the evidence set. The generator guarantees one-sided deletions only, so one
#     reflection step is sufficient.
#   - The fill loop also assumes the reflected destination is background. The generator
#     guarantees this by starting from the clean output and deleting pixels rather than
#     moving colors around, so no other object can occupy a repaired cell.
# placement_legality:
#   - Generation stays output-first: first build clean symmetric sprites, then place
#     them into non-overlapping slots, then delete selected non-axis pixels from one
#     side of mirrored pairs to form the input.
#   - Because corruption only removes pixels, legality is preserved automatically:
#     nothing new is added beyond the already placed clean output, so repairs remain
#     in-bounds and cannot overlap other objects.
# visual_observability:
#   - Across examples, outputs are systematically larger than inputs and specifically
#     fill missing mirror counterparts. This is visibly different from the source task,
#     where outputs are smaller due to pruning.
#   - No hidden ordering by color id or other non-visual property is used. Color only
#     separates objects; the rule is purely geometric.
# cleanliness_and_inference:
#   - Using only 3 sprites and wide slot gaps keeps the scene uncluttered.
#   - Missing cells belong to existing objects, so the repaired pixels appear exactly
#     where a human would expect a mirror completion, making the fusion rule inferable
#     from 2-4 examples.
# new_main_assumptions_checked_against_generator:
#   - `colors_present(..., exclude={bg})`: generator always creates exactly 3 non-bg colors.
#   - axis search over all columns: generator records a unique true axis with a score gap.
#   - `out[rr, cc] == bg` before fill: generator only deletes from the placed clean shape.
#   - one pass over original points is enough: generator never deletes both sides of the
#     same mirrored pair and never deletes axis cells.


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: mirrored-match count} for every candidate vertical axis."""
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


def _best_axis(coords: np.ndarray, width: int) -> int:
    """Pick the axis k maximizing mirrored matches for the given coordinates."""
    pts = {(int(r), int(c)) for r, c in coords.tolist()}
    best_score = -1
    best_k = 0
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    """Return the highest mirrored-match score excluding the true axis."""
    scores = _axis_scores(coords, width)
    return max(score for k, score in scores.items() if k != true_k)


def _prune_unmatched(grid: np.ndarray) -> np.ndarray:
    """Reference the original task behavior: delete unmatched pixels after axis search."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        k = _best_axis(pts, w)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < w and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = bg
    return out


def _completion_count(inp: np.ndarray, out: np.ndarray) -> int:
    """Count how many background cells are filled by the repaired output."""
    bg = int(get_bg(inp))
    return int(np.sum((inp == bg) & (out != bg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each colored sprite by completing its vertical mirror partners.

    Input structure:
    - bg is the dominant background color, and each other color forms its own nearly symmetric sprite.
    - The asymmetry comes from missing pixels: one side of some mirrored pairs was deleted, leaving a partial version of the intended sprite.

    Rule:
    - For each non-background color, find the vertical axis that best matches the observed pixels of that color.
    - Whenever a visible pixel lacks its same-color mirror partner across that axis, fill the reflected background cell with the same color.

    Key insight: The symmetry axis is still inferred independently per color, but an unmatched pixel now means "its partner is absent" rather than "this pixel is noise", so the repair is additive instead of subtractive.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        k = _best_axis(pts, w)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if 0 <= cc < w and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == bg:
                out[rr, cc] = int(col)

    return out


# --------------- generator (combo D1/0 + D4/2) ---------------

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

# ── CFG: missing-pixel corruption ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

# ── CFG: axis-gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying the sprite spec."""
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
    """Pick a random clean sprite spec, mildly biased toward denser shapes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose a structured subset of the left half-grid."""
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
    """Return a mask telling which rows receive a center-column pixel."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a vertically symmetric sprite and return (sprite, local_axis_k)."""
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
    """Place the clean sprites into disjoint slot regions."""
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
    """Return deletable cells: one side of a mirrored pair, never axis cells."""
    _h, w = clean_mask.shape
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(clean_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if 0 <= mirror_c < w and clean_mask[r, mirror_c]:
            candidates.append((r, c))
    return candidates


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one side of mirrored pairs so the solver must complete them."""
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
        scored.sort(key=lambda item: (item[0], item[1]))
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
    if true_score - max(score for k, score in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = bg
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Build one (input, output) pair with missing-pixel asymmetry."""
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

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    # Anti-degeneration check: the sample must genuinely require completion and must
    # not collapse back to the original "prune unmatched pixels" behavior.
    if _completion_count(inp, solved) <= 0:
        return None
    if np.array_equal(_prune_unmatched(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            inp, _out = pair
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
