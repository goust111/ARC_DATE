# Combo variant of 8e5c0c38 (index 12)
# combo_spec: D3/3+D4/2
# Name: 空洞精灵缺失像素修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# - concrete object family:
#   Each non-background color still owns exactly one sprite and one vertical
#   symmetry axis, but the clean sprite is now hollow: it is a filled rectangle
#   with either one centered internal hole or a mirrored pair of internal holes.
#   The holes are real background regions that belong to the target shape.
# - behavior trace on one sprite:
#   Clean 6x7 sprite with a centered hole:
#     #######
#     #######
#     ##...##
#     ##...##
#     #######
#     #######
#   Input noise is no longer "extra attached pixels". Instead, one side of a
#   mirrored pair is deleted, e.g. row 2 column 1 disappears while row 2 column
#   5 remains, or one pixel on the hole rim is removed so the hole grows only on
#   one side. The original solver would wrongly erase the surviving partner,
#   shrinking the sprite. The fused solver keeps the axis-search step but flips
#   the action: a visible pixel now proves that its reflected partner is the
#   missing one, so the output fills the background mirror cell. True holes stay
#   background because both mirrored positions are background there.
# - complexity calibration:
#   This stays close to the original ARC task: 22x22 grid, 3 colors, one sprite
#   per color, independent per-color axis inference, integer or half-integer
#   axes, and output-first generation with axis-gap validation. The visible
#   change is meaningful: outputs are larger than inputs, and some repaired
#   pixels sit beside internal cavities, so the solver must infer "hole vs
#   missing notch" rather than blindly mirror the whole grid.
# - implicit-assumption audit for main():
#   1. `pts = np.argwhere(g == col)` assumes one logical sprite per color.
#      Generator keeps distinct colors and non-overlapping slots, so that holds.
#   2. Best-axis search assumes enough mirrored evidence survives the damage.
#      Generator only deletes one side of a mirrored pair and rejects samples
#      whose true axis is not clearly best after deletion.
#   3. The original deletion loop assumed "unmatched means spurious". That fails
#      here because unmatched colored pixels are often the surviving half of a
#      missing pair. The new loop therefore reads from the original grid and
#      paints the reflected background cell instead of deleting the visible one.
#   4. Filling `out[r, cc]` assumes the mirror location is background, not some
#      other sprite. Output-first placement in disjoint slots guarantees this.
# - placement legality:
#   Sprites are built locally, pasted into fixed slot layouts, and then only
#   shrink by deletion to form the input. Since deleted pixels come from the
#   already-placed clean sprite, the repaired output always stays in-bounds and
#   cannot collide with other colors.
# - visual observability:
#   Across a few samples a human can see two repeated facts: every colored object
#   contains a real symmetric cavity, and every asymmetric dent is repaired by
#   reflecting the surviving side. Background inside a hole that is symmetric on
#   both sides never fills, so the "keep holes, fix one-sided bites" rule is
#   visually recoverable.
# - visual cleanliness:
#   Only three sprites appear, each in its own reserved region. Holes are at
#   least a few cells large and deletions are limited to boundary cells, so the
#   objects remain legible rather than turning into noisy speckle.
# - color reasoning:
#   Colors are unchanged. They only separate the three independent sprites; no
#   hidden ordering or color-id-specific behavior is introduced.
# - new solver assumptions checked against generator:
#   1. Every intended repair cell has a same-color mirror witness in the input.
#      Guaranteed because axis cells are never deleted and both sides of one
#      mirrored pair are never deleted together.
#   2. Real holes remain symmetric background on both sides. Guaranteed because
#      the clean sprite is built first and deletion noise only removes colored
#      cells; it never adds fill inside a hole.
#   3. The target is exactly the symmetry-closure of the observed color layer.
#      Guaranteed by deletion-only noise plus per-sprite symmetry.


def _best_axis(coords: np.ndarray, grid: np.ndarray, color: int) -> int:
    """Return the vertical axis k maximizing same-color mirror matches."""
    _h, w = grid.shape
    best_score = -1
    best_k = 0
    for k in range(0, 2 * (w - 1) + 1):
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
    Repair each hollow sprite by restoring its missing mirrored pixels.

    Input structure:
    - The background fills most of the grid. Three other colors each form one nearly vertically symmetric sprite with genuine background holes inside it.
    - Noise appears as deletions: some colored cells are missing on only one side of the sprite, so dents can affect either the outer rim or a hole boundary.

    Rule:
    - For each non-background color, find the vertical axis with the strongest same-color mirror agreement.
    - Reflect every observed same-color pixel across that axis and fill the mirror location when it is still background.

    Key insight:
    - In this variant, a background cell is repaired only when the opposite side still contains the matching colored witness; symmetric background on both sides is an intended hole and must remain empty.
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

        k = _best_axis(pts, g, int(col))
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if 0 <= cc < w and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


# --------------- generator (combo hollow-deletion variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean hollow sprite geometry ──
OUTER_HEIGHT_MIN = 5
OUTER_HEIGHT_MAX = 9
OUTER_WIDTH_MIN = 5
OUTER_WIDTH_MAX = 11
CLEAN_COUNT_MIN = 14
CLEAN_COUNT_MAX = 46
MIN_TOTAL_HOLE_AREA = 3
CENTER_HOLE_WEIGHT = 3.0
PAIR_HOLE_WEIGHT = 2.0

# ── CFG: deletion noise ──
MISSING_MIN = 2
MISSING_FRAC_LO = 0.12
MISSING_FRAC_HI = 0.30
MISSING_MAX = 12

# ── CFG: axis checks ──
SPRITE_AXIS_GAP_MIN = 4
FINAL_AXIS_GAP_MIN = 3

# ── CFG: retry limits ──
SPRITE_RETRIES = 180
GEN_RETRIES = 450

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
    scores = _axis_scores(coords, width=width)
    return max(s for k, s in scores.items() if k != true_k)


def _center_hole_options(h: int, w: int) -> list[list[tuple[int, int, int, int]]]:
    """Return symmetric centered-hole rectangle options."""
    options: list[list[tuple[int, int, int, int]]] = []
    for hole_h in range(1, h - 1):
        for r0 in range(1, h - hole_h):
            for hole_w in range(1, w - 1):
                if (w - hole_w) % 2 != 0:
                    continue
                c0 = (w - hole_w) // 2
                if c0 < 1 or c0 + hole_w > w - 1:
                    continue
                if hole_h * hole_w < MIN_TOTAL_HOLE_AREA:
                    continue
                options.append([(r0, c0, r0 + hole_h - 1, c0 + hole_w - 1)])
    return options


def _paired_hole_options(h: int, w: int) -> list[list[tuple[int, int, int, int]]]:
    """Return mirrored left/right hole rectangle options."""
    options: list[list[tuple[int, int, int, int]]] = []
    side_w = w // 2
    for hole_h in range(1, h - 1):
        for r0 in range(1, h - hole_h):
            for hole_w in range(1, side_w + 1):
                max_left_c0 = side_w - hole_w
                for left_c0 in range(1, max_left_c0 + 1):
                    right_c0 = w - left_c0 - hole_w
                    left_c1 = left_c0 + hole_w - 1
                    right_c1 = right_c0 + hole_w - 1
                    if right_c0 <= left_c1:
                        continue
                    if right_c0 < 1 or right_c1 > w - 2:
                        continue
                    if 2 * hole_h * hole_w < MIN_TOTAL_HOLE_AREA:
                        continue
                    options.append(
                        [
                            (r0, left_c0, r0 + hole_h - 1, left_c1),
                            (r0, right_c0, r0 + hole_h - 1, right_c1),
                        ]
                    )
    return options


def _carve_rectangles(mask: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy of mask with the listed rectangles carved out."""
    out = mask.copy()
    for r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = False
    return out


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a vertically symmetric hollow sprite and return (sprite_array, local_axis_k)."""
    h_hi = min(OUTER_HEIGHT_MAX, max_h)
    w_hi = min(OUTER_WIDTH_MAX, max_w)
    if h_hi < OUTER_HEIGHT_MIN or w_hi < OUTER_WIDTH_MIN:
        raise ValueError("slot too small for hollow sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.randint(OUTER_HEIGHT_MIN, h_hi)
        w = random.randint(OUTER_WIDTH_MIN, w_hi)

        center_opts = _center_hole_options(h, w)
        pair_opts = _paired_hole_options(h, w)
        hole_options: list[list[tuple[int, int, int, int]]] = []
        weights: list[float] = []
        if center_opts:
            hole_options.extend(center_opts)
            weights.extend([CENTER_HOLE_WEIGHT] * len(center_opts))
        if pair_opts:
            hole_options.extend(pair_opts)
            weights.extend([PAIR_HOLE_WEIGHT] * len(pair_opts))
        if not hole_options:
            continue

        rects = random.choices(hole_options, weights=weights, k=1)[0]
        mask = _carve_rectangles(np.ones((h, w), dtype=bool), rects)
        clean_count = int(np.count_nonzero(mask))
        hole_area = int(mask.size - clean_count)
        if not (CLEAN_COUNT_MIN <= clean_count <= CLEAN_COUNT_MAX):
            continue
        if hole_area < MIN_TOTAL_HOLE_AREA:
            continue

        coords = [tuple(x) for x in np.argwhere(mask)]
        true_k = w - 1
        if _second_best_score(coords, true_k, width=w) > clean_count - SPRITE_AXIS_GAP_MIN:
            continue

        sprite = np.zeros((h, w), dtype=int)
        sprite[mask] = color
        return sprite, true_k

    raise ValueError("failed to build clean hollow sprite")


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
        spec["r1"] = r0 + h - 1
        spec["c1"] = c0 + w - 1
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        placed.append(spec)
    return out, placed


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return which foreground cells touch non-foreground in 4-neighborhood."""
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            out[r, c] = True
            continue
        if not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
            out[r, c] = True
    return out


def _removal_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Return removable cells: one side of a mirrored pair, restricted to visible boundaries."""
    h, w = clean_mask.shape
    boundary = _boundary_mask(clean_mask)
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(clean_mask & boundary)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if 0 <= mirror_c < w and bool(clean_mask[r, mirror_c]):
            candidates.append((r, c))
    return candidates


def _unique_pair_count(candidates: list[tuple[int, int]], true_k: int) -> int:
    """Count mirrored pairs represented by candidate cells."""
    pairs: set[tuple[int, int, int]] = set()
    for r, c in candidates:
        mc = true_k - c
        pairs.add((int(r), min(int(c), int(mc)), max(int(c), int(mc))))
    return len(pairs)


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one-sided boundary pixels from inp so main() must fill them back in."""
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    candidates = _removal_candidates(clean_mask, true_k)
    pair_count = _unique_pair_count(candidates, true_k)
    lo = max(MISSING_MIN, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)
    if lo > hi:
        return False

    missing_target = random.randint(lo, hi)
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    current_set = set(clean_coords)

    for _ in range(missing_target):
        viable = []
        for cell in candidates:
            if cell in forbidden or cell not in current_set:
                continue
            mirror = (cell[0], true_k - cell[1])
            if mirror not in current_set:
                continue
            viable.append(cell)
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
        current_set.remove(cell)
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
    """Single attempt: build clean hollow sprites, delete mirror partners, validate via solver."""
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
