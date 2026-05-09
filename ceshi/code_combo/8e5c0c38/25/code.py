# Combo variant of 8e5c0c38 (index 25)
# combo_spec: D1/0+D3/3
# Name: 空洞精灵补全修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# fusion_summary:
#   - D1/0 flips the original repair direction: asymmetric evidence means a mirror
#     partner is missing and should be added, not that the visible pixel is noise.
#   - D3/3 changes the clean object family from generic solid-ish sprites to hollow
#     sprites, so legal background holes must survive the repair.
# concrete_object_choice:
#   - To keep the hole visually obvious and the generator reliable, each clean object
#     is a 1-pixel-thick hollow rectangular frame.
#   - The input is formed by deleting one side of selected mirrored border pairs.
#     No pair loses both members, and axis cells are never deleted.
#   - The output is the original full frame, so every sample simultaneously shows
#     "output grows by completion" and "the center hole stays empty".
# behavior_trace:
#   - Example clean frame (color X, true axis through the middle):
#       XXXXXXX
#       X.....X
#       X.....X
#       XXXXXXX
#   - Input after removals:
#       X.XXXXX
#       X.....X
#       X.....X
#       XXXXX.X
#   - Original task logic would delete the unmatched border pixels and shrink the
#     shape further. A naive D1-only mirror fill would add the missing border mates,
#     but the combo solver must also treat the interior background as intentional.
#   - The combo solver therefore finds the best axis, mirrors the surviving border
#     support to recover the full bbox, then redraws the hollow frame so the hole
#     remains background instead of being interpreted as a region to fill.
# complexity_calibration:
#   - The 22x22 canvas, three colors, slot-based placement, odd/even widths, and
#     axis-gap checks are retained from the source task so per-object axis inference
#     still matters.
#   - Frames use varied sizes with nontrivial hole area, and each object loses a
#     visible fraction of its border, so the difference from the source task is easy
#     to see in 2-4 examples.
# implicit_assumption_review_for_main:
#   - The original solver assumed "unmatched pixel => delete it". That fails under
#     D1/0 and is replaced with "visible border support can imply a missing mate".
#   - Plain mirror completion on an arbitrary subset does not encode the hole; to
#     make D3/3 meaningful, main() redraws the object family (hollow frames) after
#     the axis-guided completion step.
#   - This redraw assumes each color corresponds to one frame and that the completed
#     support recovers the true bbox. The generator guarantees both by using distinct
#     colors, non-overlapping slots, and one-sided deletions only.
# placement_legality:
#   - Generation remains output-first. Full frames are placed inside fixed slots, so
#     every later deletion only shrinks objects and cannot introduce overlap or go
#     out of bounds.
# visual_observability_and_cleanliness:
#   - Every output is larger than its input, but the enlarged region traces a neat
#     rectangular ring instead of filling the hole. The hole is a large blank area,
#     so the D3 change is human-visible without relying on color ids or hidden state.
# color_reasoning:
#   - Colors still only separate objects. There is no color remapping or ordering.


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: same-color mirror-match count} for every vertical axis."""
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
    """Return the highest vertical-axis score excluding the intended axis."""
    scores = _axis_scores(coords, width)
    return max(s for k, s in scores.items() if k != true_k)


def _best_axis(coords: np.ndarray, grid: np.ndarray, color: int) -> int:
    """Return the vertical axis k with the strongest same-color mirror support."""
    _h, w = grid.shape
    best_score = -1
    best_k = 0
    for k in range(2 * (w - 1) + 1):
        score = 0
        for r, c in coords:
            cc = int(k) - int(c)
            if 0 <= cc < w and int(grid[int(r), cc]) == int(color):
                score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each hollow sprite by completing its vertical mirror structure.

    Input structure:
    - The background fills most cells. Three other colors each form one nearly symmetric hollow sprite with a legal background hole.
    - The asymmetry comes from missing border pixels on one side of mirrored pairs, while the hole itself is intentional empty space.

    Rule:
    - For each non-background color, find the vertical axis with the strongest same-color mirror agreement.
    - Reflect the surviving same-color support across that axis to recover the sprite's full bounding box, then redraw the complete hollow rectangular frame and keep its interior as background.

    Key insight:
    - Missing border pixels should be completed, but empty cells inside the hole are not missing data. Symmetry identifies the frame's extent; the object family tells you that only the border is colored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        k = _best_axis(pts, g, int(col))
        completed: set[tuple[int, int]] = set()
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            completed.add((rr, cc))
            mc = int(k) - cc
            if 0 <= mc < g.shape[1]:
                completed.add((rr, mc))

        rows = [r for r, _ in completed]
        cols = [c for _, c in completed]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        out[out == int(col)] = int(bg)
        out[r0, c0 : c1 + 1] = int(col)
        out[r1, c0 : c1 + 1] = int(col)
        out[r0 : r1 + 1, c0] = int(col)
        out[r0 : r1 + 1, c1] = int(col)

    return out


# --------------- generator (combo D1/0 + D3/3) ---------------

# canvas and layout
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# clean hollow-frame geometry
PIXEL_COUNT_MIN = 12
PIXEL_COUNT_MAX = 40
FRAME_HEIGHT_MIN = 5
FRAME_HEIGHT_MAX = 9
FRAME_WIDTH_MIN = 5
FRAME_WIDTH_MAX = 11
MIN_HOLE_AREA = 3

# missing-border corruption
MISSING_FRAC_LO = 0.12
MISSING_FRAC_HI = 0.30
MISSING_MAX = 10

# axis checks
SPRITE_AXIS_GAP_MIN = 4
FINAL_AXIS_GAP_MIN = 3

# retries / augmentation
SPRITE_RETRIES = 120
GEN_RETRIES = 400
FLIP_PROB = 0.5


def _frame_options(max_h: int, max_w: int) -> list[tuple[int, int]]:
    """Enumerate feasible (h, w) pairs for hollow rectangular frames."""
    options: list[tuple[int, int]] = []
    for h in range(FRAME_HEIGHT_MIN, min(FRAME_HEIGHT_MAX, max_h) + 1):
        for w in range(FRAME_WIDTH_MIN, min(FRAME_WIDTH_MAX, max_w) + 1):
            perimeter = 2 * h + 2 * w - 4
            hole_area = (h - 2) * (w - 2)
            if not (PIXEL_COUNT_MIN <= perimeter <= PIXEL_COUNT_MAX):
                continue
            if hole_area < MIN_HOLE_AREA:
                continue
            options.append((h, w))
    return options


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a hollow symmetric frame and return (sprite_array, local_axis_k)."""
    options = _frame_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible hollow frame spec")

    for _ in range(SPRITE_RETRIES):
        h, w = random.choice(options)
        sprite = np.zeros((h, w), dtype=int)
        sprite[0, :] = color
        sprite[-1, :] = color
        sprite[:, 0] = color
        sprite[:, -1] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = w - 1
        if _second_best_score(coords, true_k, width=w) > len(coords) - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k

    raise ValueError("failed to build clean hollow frame")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place full clean sprites into non-overlapping slot regions."""
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
        spec["r1"] = r0 + h - 1
        spec["c1"] = c0 + w - 1
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        placed.append(spec)

    return out, placed


def _border_groups(spec: dict) -> list[list[tuple[int, int]]]:
    """Return canonical mirrored border groups for one clean frame."""
    r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
    true_k = int(spec["true_k"])
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    groups: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r, c in sorted(border):
        mc = true_k - c
        if not (c0 <= mc <= c1):
            continue
        mate = (r, mc)
        if mate not in border:
            continue
        if mate == (r, c):
            continue
        group = tuple(sorted([(r, c), mate]))
        if group in seen:
            continue
        seen.add(group)
        groups.append(list(group))
    return groups


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one member from selected mirrored border pairs."""
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_count = len(clean_coords)

    groups = _border_groups(spec)
    if not groups:
        return False

    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), len(groups))
    if lo > hi:
        return False

    target = random.randint(lo, hi)
    chosen_groups = random.sample(groups, target)
    removed: list[tuple[int, int]] = [random.choice(group) for group in chosen_groups]

    remaining = [pt for pt in clean_coords if pt not in removed]
    scores = _axis_scores(remaining, GRID_SIZE)
    if scores[true_k] - max(s for k, s in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in removed:
        inp[r, c] = bg
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build clean frames, remove mirrored border mates, and validate with main()."""
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
