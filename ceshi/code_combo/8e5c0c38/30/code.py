# Combo variant of 8e5c0c38 (index 30)
# combo_spec: D1/0+D7/0
# Name: 紧凑裁剪补全修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   preserved_base:
#     - input is still a 22x22 scene with one dominant background color
#     - exactly 3 foreground colors appear, each color corresponding to one sprite
#     - each sprite is organized around its own vertical symmetry axis, which may be
#       an integer or half-integer column
#   fused_changes:
#     - D1/0 flips the repair direction: unmatched pixels are not deleted; instead
#       their reflected partners are filled in
#     - D7/0 changes the output frame: after repair, return only the tight bbox that
#       contains all repaired foreground pixels
#   concrete_behavior_trace:
#     - Small example: suppose a clean scene would occupy rows 1..6 and cols 2..9.
#       A leftmost sprite has true pixels in cols 2 and 5 that mirror across its axis.
#       The input removes the col-2 pixel but keeps the col-5 pixel. In the source task,
#       the visible col-5 pixel would be erased. In this combo, the solver restores
#       the missing col-2 pixel, so the repaired scene again reaches global col 2.
#       Only then is the whole scene cropped. If one cropped the input first, the bbox
#       would begin at col 3 instead; the combo therefore requires "complete first,
#       crop second".
#   complexity_calibration:
#     - Keep the original slot-based separation and axis-gap checks so the task still
#       needs per-color axis inference rather than whole-grid mirroring.
#     - Missing pixels remain 11%-33% of a sprite, so completion is plainly visible.
#     - To make the crop rule non-degenerate, generation forces at least one missing
#       pixel on the global left or right boundary of the repaired scene. Therefore
#       the cropped output is wider than the tight crop of the input, and the crop
#       boundary truly depends on the repaired extent.
#   implicit_assumption_review_main:
#     - `get_bg(g)` still assumes one majority background color; generation preserves
#       a mostly empty 22x22 canvas.
#     - `colors_present(..., exclude={bg})` assumes each foreground color is a single
#       sprite; generation uses 3 distinct colors and non-overlapping placements.
#     - Axis search across the full input width is still valid because the input canvas
#       remains 22x22 before cropping.
#     - The fill loop must read symmetry evidence from the original grid `g`, not from
#       the evolving `out`, otherwise newly added pixels could recursively justify more
#       additions than the intended one-step mirror completion.
#     - Filling only background cells avoids overwriting other sprites; generation keeps
#       sprites spatially separated, so intended mirror targets stay empty in the input.
#     - The final crop assumes repaired foreground exists; generation starts from clean
#       sprites with 6-46 pixels and removes only one side of mirrored pairs.
#   placement_legality:
#     - Generation is output-first: build full symmetric sprites, place them in the
#       original non-overlapping slots, then delete selected pixels to form the input.
#       Because deletions only shrink sprites, inputs remain in-bounds and disjoint.
#     - Forced boundary deletions are taken from the global leftmost or rightmost clean
#       column only, so restoring them never expands beyond the original clean layout.
#   visual_observability:
#     - Across examples, outputs have both more pixels than inputs and a smaller frame
#       than 22x22. More importantly, the output crop is based on the repaired scene,
#       not the damaged one, so at least one missing outer column reappears in the
#       cropped answer. This makes the fusion visible from a few examples.
#   cleanliness:
#     - No color remapping or hidden ordering is used. Colors only separate sprites,
#       and the visual story remains "repair bilateral shapes, then trim blank border".


def _tight_crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    """Crop to the smallest bbox containing all non-background pixels."""
    bb = bbox(grid != int(bg))
    if bb is None:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(grid, bb)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color layer by vertical mirroring, then crop the repaired scene tightly.

    Input structure:
    - The input is a 22x22 grid with one dominant background color and three separated non-bg sprite colors.
    - Each sprite is almost vertically symmetric around its own axis, but some mirror partners are missing, so the visible shape is a subset of the intended symmetric one.

    Rule:
    - For each non-bg color independently, find the vertical axis that best matches the observed pixels and fill any missing same-color mirror partners across that axis.
    - After every sprite has been completed, return only the minimum rectangular crop that contains all repaired non-bg pixels.

    Key insight: The asymmetry is evidence of absent mirror pixels rather than noise, and the final output bounds must be inferred from the completed scene, not from the damaged input silhouette.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Candidate axes are encoded by k in [0, 2*(w-1)], representing column c = k/2.
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
            if 0 <= cc < w and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return _tight_crop_foreground(out, bg)


# --------------- generator (codex-att3, combo D1 fill + D7 tight crop) ---------------

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


def _axis_scores(coords: list[tuple[int, int]] | set[tuple[int, int]], width: int) -> dict[int, int]:
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


def _second_best_score(coords: list[tuple[int, int]] | set[tuple[int, int]], true_k: int, width: int) -> int:
    """Return the highest axis score excluding the true axis."""
    scores = _axis_scores(coords, width)
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
    """Choose left-half cells while keeping the local sprite visually coherent."""
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
        if _second_best_score(coords, true_k, w) > target_count - SPRITE_AXIS_GAP_MIN:
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


def _removal_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Return removable cells: one side of a mirrored pair, never axis cells."""
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


def _missing_lo(clean_count: int) -> int:
    return max(1, math.ceil(MISSING_FRAC_LO * clean_count))


def _missing_hi(clean_count: int, pair_count: int) -> int:
    return min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)


def _boundary_forced_removals(out_full: np.ndarray, placed_specs: list[dict], bg: int) -> dict[int, list[tuple[int, int]]] | None:
    """
    Choose a global left or right boundary column whose cells will be removed in the input.

    This guarantees that the tight crop after completion is strictly different from the
    tight crop of the damaged input, so D7 is coupled to the D1 completion step.
    """
    fg = out_full != int(bg)
    bb = bbox(fg)
    if bb is None:
        return None
    _r0, c0, _r1, c1 = bb

    per_color_info: dict[int, dict[str, object]] = {}
    for spec in placed_specs:
        color = int(spec["color"])
        clean_mask = out_full == color
        candidates = set(_removal_candidates(clean_mask, int(spec["true_k"])))
        clean_count = int(np.count_nonzero(clean_mask))
        pair_count = len(candidates) // 2
        per_color_info[color] = {
            "candidates": candidates,
            "max_forced": _missing_hi(clean_count, pair_count),
        }

    side_cols = [c0, c1]
    random.shuffle(side_cols)
    for boundary_col in side_cols:
        rows = np.where(fg[:, boundary_col])[0]
        if rows.size == 0:
            continue

        forced_by_color: dict[int, list[tuple[int, int]]] = {}
        ok = True
        for r in rows.tolist():
            cell = (int(r), int(boundary_col))
            color = int(out_full[cell])
            info = per_color_info[color]
            if cell not in info["candidates"]:
                ok = False
                break
            forced_by_color.setdefault(color, []).append(cell)

        if not ok:
            continue
        if any(len(cells) > int(per_color_info[color]["max_forced"]) for color, cells in forced_by_color.items()):
            continue
        return forced_by_color

    return None


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict, forced: list[tuple[int, int]]) -> bool:
    """Delete one-sided mirror partners from inp so main() must fill them back in."""
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = {tuple(x) for x in np.argwhere(out == color)}
    clean_mask = out == color
    clean_count = len(clean_coords)

    candidates = _removal_candidates(clean_mask, true_k)
    candidate_set = set(candidates)
    pair_count = len(candidates) // 2
    lo = _missing_lo(clean_count)
    hi = _missing_hi(clean_count, pair_count)
    if lo > hi:
        return False
    if len(forced) > hi or any(cell not in candidate_set for cell in forced):
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = set(clean_coords)

    for cell in forced:
        if cell in forbidden or cell not in current_coords:
            return False
        chosen.append(cell)
        current_coords.remove(cell)
        forbidden.add(cell)
        mirror = (cell[0], true_k - cell[1])
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    target_lo = max(lo, len(forced))
    if target_lo > hi:
        return False
    missing_target = random.randint(target_lo, hi)

    for _ in range(missing_target - len(forced)):
        viable = [cell for cell in candidates if cell not in forbidden and cell in current_coords]
        if not viable:
            return False

        scored = []
        for cell in viable:
            test_coords = current_coords - {cell}
            alt = _second_best_score(test_coords, true_k, GRID_SIZE)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)

        chosen.append(cell)
        current_coords.remove(cell)
        forbidden.add(cell)
        mirror = (cell[0], true_k - cell[1])
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
    """Build a damaged 22x22 input whose completed answer is a strict tight crop."""
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

    forced_by_color = _boundary_forced_removals(out_full, placed_specs, bg)
    if forced_by_color is None:
        return None

    inp = out_full.copy()
    for spec in placed_specs:
        forced = forced_by_color.get(int(spec["color"]), [])
        if not _remove_pixels(inp, out_full, spec, forced):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out_full = np.flipud(out_full)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out_full = np.fliplr(out_full)

    expected = _tight_crop_foreground(out_full, bg)
    cropped_input = _tight_crop_foreground(inp, bg)
    if expected.shape == (GRID_SIZE, GRID_SIZE):
        return None
    if np.array_equal(cropped_input, expected):
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
