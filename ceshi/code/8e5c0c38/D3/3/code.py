# Variant of 8e5c0c38
# Dimension: D3 — 被变换的源对象类型 (src)
# Change: 3个独立颜色的近对称精灵，每个精灵使用单一颜色，精灵间互不重叠... → 精灵具有空洞（内部有背景色区域）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste

# Design analysis for the D3 variant
# - source_task: none, so there is no extra reference solver to inspect.
# - concrete object family:
#   Each non-background object is now a single-color hollow rectangular frame.
#   The clean output contains three vertically symmetric frames with 1-pixel walls
#   and background-colored holes inside. The input adds two kinds of noise:
#   1. asymmetric exterior pixels attached around the outside border;
#   2. interior hole-fill pixels placed inside the hole, often in mirror-paired
#      positions that the original solver would incorrectly keep.
# - behavior trace on one sprite:
#   Clean frame (5x7):
#     #######
#     #.....#
#     #.....#
#     #.....#
#     #######
#   Noisy input may contain one extra left-side exterior pixel and a symmetric
#   pair of filled cells inside the hole. The original solver's "keep every
#   mirrored pixel" rule would delete the exterior pixel but preserve the filled
#   hole pair. The new rule still uses the best vertical axis, but it only uses
#   the mirror-paired pixels to recover the true bounding box; it then redraws
#   the bbox border and leaves the interior as background, restoring the hole.
# - complexity calibration:
#   The objects stay simple enough for ARC inspection, but the filled-hole noise
#   is visually obvious and materially changes the logic: there are examples
#   where symmetric same-color pixels must still be removed. Width parity varies
#   so some axes are integer-column and some are half-integer, matching the
#   original task's axis ambiguity without adding unrelated color logic.
# - implicit-assumption audit for main():
#   The original code assumed the clean target is exactly the set of mirror-
#   paired pixels. That assumption fails here because mirror-paired pixels may
#   sit inside a hole and should be deleted. The new code assumes instead that
#   each color forms one hollow rectangular frame and that asymmetric exterior
#   noise is not needed to locate the true bbox. The generator guarantees this.
# - placement legality:
#   Sprites are still generated output-first inside fixed non-overlapping slots.
#   The clean frame is pasted first, then hole-fill and exterior noise are added
#   without crossing slot boundaries or colliding with other colors.
# - visual observability:
#   Across a few samples, the human-visible invariant is that each color object
#   becomes a neat hollow frame. Some interior colored pixels disappear even when
#   they are symmetrically placed, so the hole itself is clearly part of the
#   structure to recover.
# - visual cleanliness:
#   One-pixel-thick frames keep the hole readable. Exterior noise is sparse and
#   stays near one sprite, so the result is legible even in a 22x22 grid.
# - color reasoning:
#   Colors are untouched except for deletion; there is no hidden ordering or
#   color-id-dependent behavior.
# - new solver assumptions checked against generator:
#   1. One object per non-bg color. Guaranteed by distinct-color frame placement.
#   2. The mirror-paired subset shares the clean frame bbox. Guaranteed because
#      all exterior noise is asymmetric and all hole-fill noise stays strictly
#      inside the clean bbox.
#   3. The target shape is a 1-pixel rectangular frame. Guaranteed by the clean
#      sprite builder; main redraws exactly that shape.


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
    Restore each hollow sprite by recovering its vertical frame.

    Input structure:
    - The background fills most of the grid. Three other colors each form one nearly mirrored hollow frame.
    - Noise may appear outside a frame or inside its background hole, and some hole-filling noise can itself be mirror-paired.

    Rule:
    - For each non-background color, find the vertical axis with the strongest same-color mirror agreement.
    - Use the mirror-paired subset only to recover that sprite's bounding box, then redraw the box border in that color and keep its interior as background.

    Key insight:
    - Symmetry alone no longer identifies the target pixels, because hole-filling noise may also be symmetric; the stable invariant is the hollow rectangular frame implied by the symmetric bbox.
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
        paired: list[tuple[int, int]] = []
        for r, c in pts:
            cc = int(k) - int(c)
            if 0 <= cc < g.shape[1] and int(g[int(r), cc]) == int(col):
                paired.append((int(r), int(c)))

        if not paired:
            out[out == int(col)] = int(bg)
            continue

        rows = [r for r, _ in paired]
        cols = [c for _, c in paired]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        out[out == int(col)] = int(bg)
        out[r0, c0 : c1 + 1] = int(col)
        out[r1, c0 : c1 + 1] = int(col)
        out[r0 : r1 + 1, c0] = int(col)
        out[r0 : r1 + 1, c1] = int(col)

    return out


# --------------- generator (D3 hollow-frame variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean hollow frames ──
PIXEL_COUNT_MIN = 12
PIXEL_COUNT_MAX = 40
FRAME_HEIGHT_MIN = 5
FRAME_HEIGHT_MAX = 9
FRAME_WIDTH_MIN = 5
FRAME_WIDTH_MAX = 11
MIN_HOLE_AREA = 3

# ── CFG: noise ──
HOLE_GROUPS_MIN = 1
HOLE_GROUPS_MAX = 4
OUTER_NOISE_MIN = 1
OUTER_NOISE_MAX = 5
MAX_TOTAL_NOISE_PIXELS = 12

# ── CFG: axis checks ──
SPRITE_AXIS_GAP_MIN = 4
FINAL_AXIS_GAP_MIN = 4

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 400

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


def _frame_options(max_h: int, max_w: int) -> list[tuple[int, int]]:
    """Enumerate feasible (h, w) pairs for 1-pixel hollow rectangular frames."""
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


def _hole_groups(spec: dict) -> list[list[tuple[int, int]]]:
    """Return mirror-canonical candidate groups inside the clean hole."""
    r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
    true_k = spec["true_k"]
    groups: list[list[tuple[int, int]]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            mc = true_k - c
            if not (c0 < mc < c1):
                continue
            if c > mc:
                continue
            if mc == c:
                groups.append([(r, c)])
            else:
                groups.append([(r, c), (r, mc)])
    return groups


def _add_hole_noise(inp: np.ndarray, spec: dict) -> int:
    """Add mirror-paired noise inside the hole. Returns number of added pixels."""
    groups = _hole_groups(spec)
    if not groups:
        return -1

    max_groups = min(HOLE_GROUPS_MAX, len(groups))
    if max_groups < HOLE_GROUPS_MIN:
        return -1
    group_count = random.randint(HOLE_GROUPS_MIN, max_groups)
    chosen_groups = random.sample(groups, group_count)

    added = 0
    color = int(spec["color"])
    for group in chosen_groups:
        for r, c in group:
            inp[r, c] = color
            added += 1
    return added


def _outer_noise_candidates(inp: np.ndarray, spec: dict, bg: int) -> list[tuple[int, int]]:
    """Find exterior 4-adjacent cells outside the clean bbox."""
    r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
    color = int(spec["color"])
    candidates: set[tuple[int, int]] = set()

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(inp[r, c]) != color:
                continue
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE):
                    continue
                if int(inp[rr, cc]) != bg:
                    continue
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    continue
                candidates.add((rr, cc))

    return sorted(candidates)


def _add_outer_noise(inp: np.ndarray, spec: dict, bg: int, budget: int) -> bool:
    """Add asymmetric exterior noise near the frame."""
    if budget <= 0:
        return True

    candidates = _outer_noise_candidates(inp, spec, bg)
    true_k = spec["true_k"]
    color = int(spec["color"])
    if len(candidates) < OUTER_NOISE_MIN:
        return False

    hi = min(OUTER_NOISE_MAX, budget, len(candidates))
    if hi < OUTER_NOISE_MIN:
        return False
    target = random.randint(OUTER_NOISE_MIN, hi)

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    for _ in range(target):
        viable = []
        for r, c in candidates:
            if (r, c) in forbidden:
                continue
            mc = true_k - c
            if mc == c:
                continue
            if 0 <= mc < GRID_SIZE and int(inp[r, mc]) == color:
                continue
            viable.append((r, c))
        if not viable:
            return False
        cell = random.choice(viable)
        chosen.append(cell)
        forbidden.add(cell)
        mc = true_k - cell[1]
        if 0 <= mc < GRID_SIZE:
            forbidden.add((cell[0], mc))

    for r, c in chosen:
        inp[r, c] = color
    return True


def _validate_noisy_sprite(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Ensure the intended axis remains clearly best for this color."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    coords = [tuple(x) for x in np.argwhere(inp == color)]
    clean_count = int(np.count_nonzero(out == color))
    return clean_count - _second_best_score(coords, true_k, width=GRID_SIZE) >= FINAL_AXIS_GAP_MIN


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean frames, place them, add noise, validate via solver."""
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
        hole_added = _add_hole_noise(inp, spec)
        if hole_added < 1:
            return None
        if hole_added > MAX_TOTAL_NOISE_PIXELS:
            return None
        if not _add_outer_noise(inp, spec, bg, budget=MAX_TOTAL_NOISE_PIXELS - hole_added):
            return None
        if not _validate_noisy_sprite(inp, out, spec):
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
