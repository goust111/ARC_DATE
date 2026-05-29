# Combo variant of 8e5c0c38 (index 34)
# combo_spec: D1/0+D3/3+D4/2
# Name: 空洞精灵缺失补全

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    The original task treated asymmetry as extra pixels to erase. This combo
#    flips that logic and also changes the clean object family: each color now
#    forms one vertically symmetric hollow sprite with a genuine enclosed
#    background hole. A small row-wise example makes the difference explicit.
#    Clean target around axis k=6:
#      ..###..
#      .#####.
#      .#...#.
#      .#####.
#      ..###..
#    Here the central '.' cells are a legal hole because both sides are empty.
#    If the input deletes the left outer pixel of row 1, the original solver
#    would interpret the surviving right pixel as unmatched noise and remove it,
#    shrinking the sprite. The fused solver instead keeps the right pixel and
#    restores its left mirror. The hole remains empty because no visible pixel
#    points into it from either side, so there is nothing to mirror-fill there.
#    The operational distinction is therefore:
#    - original: unmatched colored cell -> delete it
#    - combo:    unmatched colored cell -> add its missing mirror partner
#    - hole test: mirrored empty/empty locations stay background
#
# b) Complexity calibration
#    The variant should still feel like an ARC symmetry problem, not a trivial
#    frame redraw. To preserve that:
#    - the canvas stays 22x22 with exactly 3 non-bg colors in separated slots;
#    - each sprite keeps its own independently inferred vertical axis;
#    - clean sprites are not fixed rectangles: each row has its own symmetric
#      outer span, producing blob-like hollow silhouettes rather than one frame
#      template;
#    - widths can be odd or even, so axes can be integer or half-integer.
#    This keeps the axis-search ambiguity from the source task while making the
#    hole feature visually obvious across samples.
#
# c) Implicit assumption audit of main()
#    The source main() relied on these assumptions:
#    1. A color layer is best explained by one vertical axis. Still true.
#    2. Unmatched pixels are wrong and should be removed. No longer true; this
#       is the core inversion, so unmatched pixels now imply missing partners.
#    3. Every empty mirror location of an unmatched pixel should be filled only
#       if it is background. Still required, otherwise colors could overwrite.
#    4. Updating the output must not affect later mirror checks for the same
#       color. Still important, so checks read from the original grid `g`.
#    5. Background holes should not be filled just because they are interior.
#       The new logic satisfies this automatically: a hole is empty on both
#       sides, so no existing pixel votes to fill it.
#    The combo therefore only needs a D1-style fill solver, but the generator
#    must guarantee that legal hole cells are represented by paired emptiness
#    rather than by removed one-sided pixels.
#
# d) Placement legality
#    Generation remains output-first. We build a clean hollow sprite locally,
#    paste it into a reserved slot, then delete one side of selected mirrored
#    boundary pairs. Because deletion only shrinks the placed sprite, objects
#    never overlap and every solver-added pixel stays within the original clean
#    footprint. The hole is created before placement, so it is also guaranteed
#    to stay in-bounds.
#
# e) Visual observability
#    A human seeing 2-4 examples can detect two stable facts:
#    - outputs are larger than inputs, so this is completion, not pruning;
#    - interior holes remain holes after completion, so the rule is not
#      "fill all cavities" but specifically "restore mirror-missing pixels".
#    That makes the fusion visible from both the added perimeter pixels and the
#    preserved enclosed background.
#
# f) Visual cleanliness
#    Deletions are restricted to the exterior boundary and are validated so they
#    do not open the enclosed hole to the outside. This avoids messy shapes
#    where the notion of a hole disappears, keeping each object readable as one
#    hollow sprite with a few dents.
#
# g) Inferability of changes
#    No color remapping is used. Color only separates objects, which means the
#    entire rule can be inferred from geometry alone: each colored sprite is
#    repaired independently by vertical mirroring while respecting genuine hole
#    background.
#
# New solver-side assumptions checked against the generator
# - Each non-bg color appears in exactly one isolated sprite. Guaranteed by
#   distinct colors plus slot-based non-overlapping placement.
# - The best axis for a partially deleted sprite is still the true clean axis.
#   Guaranteed by rejection on axis-gap thresholds after deletions.
# - Missing pixels occur only on one side of mirrored colored pairs, never both.
#   Guaranteed by group-wise deletion that forbids removing both partners.
# - Genuine hole cells stay empty on both sides. Guaranteed because deletions are
#   validated not to reduce the clean enclosed-hole area.


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: same-color mirror-match count} for each vertical axis."""
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
    """Return the best mirror-match score among all wrong vertical axes."""
    scores = _axis_scores(coords, width=width)
    return max(score for k, score in scores.items() if k != true_k)


def _best_axis(coords: np.ndarray, grid: np.ndarray, color: int) -> int:
    """Return the vertical axis k maximizing same-color mirror agreement."""
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
    Complete each hollow color sprite by restoring vertical mirror symmetry.

    Input structure:
    - The background fills most of the grid. Three other colors each form one almost-symmetric hollow sprite with its own vertical axis.
    - The asymmetry comes from deleted colored pixels, while genuine interior holes remain background on both mirrored sides.

    Rule:
    - For each non-background color, find the vertical axis with the strongest same-color mirror agreement.
    - Whenever a colored pixel lacks its mirror partner across that axis, fill the reflected partner if that cell is background.

    Key insight: An unmatched colored pixel means its mirror counterpart is missing, but a legal hole stays empty because neither side contains a colored witness that would justify filling it.
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
            if 0 <= cc < w and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


# --------------- generator (combo hollow-sprite completion) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean hollow sprite geometry ──
PIXEL_COUNT_MIN = 10
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 5
HEIGHT_MAX = 9
WIDTH_MIN = 5
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89
MIN_HOLE_AREA = 3
MIN_HOLE_ROWS = 2
MIN_HOLE_WIDTH = 1
OUTER_SPAN_MIN = 3

# ── CFG: deleted mirror pixels ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 180
REMOVAL_RETRIES = 40
GEN_RETRIES = 400

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _enclosed_bg_local(mask: np.ndarray) -> np.ndarray:
    """Return background cells that are not 4-connected to the local bbox border."""
    h, w = mask.shape
    bg = ~mask
    exterior = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                stack.append((rr, cc))

    return bg & ~exterior


def _enclosed_bg_mask(mask: np.ndarray) -> np.ndarray:
    """Return enclosed background cells inside the bbox of a full-grid color mask."""
    enclosed = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return enclosed
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    local = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    enclosed[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = _enclosed_bg_local(local)
    return enclosed


def _count_hole_area(mask: np.ndarray) -> int:
    """Count enclosed background cells within the bbox of a color mask."""
    return int(np.count_nonzero(_enclosed_bg_mask(mask)))


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build one vertically symmetric hollow sprite and return (sprite, local_axis_k)."""
    h_lo = max(HEIGHT_MIN, 5)
    h_hi = min(HEIGHT_MAX, max_h)
    w_lo = max(WIDTH_MIN, 5)
    w_hi = min(WIDTH_MAX, max_w)
    if h_lo > h_hi or w_lo > w_hi:
        raise ValueError("no feasible hollow sprite size")

    for _ in range(SPRITE_RETRIES):
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        max_left = max(0, (w - OUTER_SPAN_MIN) // 2)
        left_edges = [random.randint(0, min(max_left, 1 if max_left > 0 else 0))]
        for _row in range(1, h):
            if random.random() < 0.25:
                nxt = random.randint(0, max_left)
            else:
                step = random.choice((-1, 0, 1))
                nxt = max(0, min(max_left, left_edges[-1] + step))
            left_edges.append(nxt)
        left_edges[0] = min(left_edges[0], 1)
        left_edges[-1] = min(left_edges[-1], 1)

        hole_h = random.randint(MIN_HOLE_ROWS, h - 2)
        hr0 = random.randint(1, h - hole_h - 1)
        hr1 = hr0 + hole_h - 1
        hole_left_min = max(left_edges[hr0 : hr1 + 1]) + 1
        hole_left_max = (w - MIN_HOLE_WIDTH) // 2
        if hole_left_min > hole_left_max:
            continue
        hole_left = random.randint(hole_left_min, hole_left_max)
        hole_right = w - 1 - hole_left

        sprite = np.zeros((h, w), dtype=int)
        for r, left in enumerate(left_edges):
            sprite[r, left : w - left] = int(color)
        sprite[hr0 : hr1 + 1, hole_left : hole_right + 1] = 0

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        count = len(coords)
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        density = count / float(h * w)
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue
        hole_area = int(np.count_nonzero(_enclosed_bg_local(sprite == int(color))))
        if hole_area < MIN_HOLE_AREA:
            continue

        true_k = w - 1
        if _second_best_score(coords, true_k, width=w) > count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k

    raise ValueError("failed to build clean hollow sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into reserved slots. Return (grid, placed_specs) or None."""
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


def _exterior_bg_mask(bg_mask: np.ndarray) -> np.ndarray:
    """Return background cells 4-connected to the outer grid border."""
    h, w = bg_mask.shape
    exterior = np.zeros_like(bg_mask, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and bg_mask[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                stack.append((rr, cc))

    return exterior


def _removal_groups(out: np.ndarray, spec: dict, bg: int) -> list[list[tuple[int, int]]]:
    """Return mirrored boundary-pair groups eligible for one-sided deletion."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    color_mask = out == color
    hole_mask = _enclosed_bg_mask(color_mask)
    exterior_bg = _exterior_bg_mask(out == bg)

    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    rows, cols = np.where(color_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if not (0 <= mirror_c < GRID_SIZE and color_mask[r, mirror_c]):
            continue

        touches_exterior = False
        touches_hole = False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                if exterior_bg[rr, cc]:
                    touches_exterior = True
                if hole_mask[rr, cc]:
                    touches_hole = True
        if not touches_exterior or touches_hole:
            continue

        key = (r, min(c, mirror_c), max(c, mirror_c))
        groups.setdefault(key, []).append((r, c))

    return list(groups.values())


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict, bg: int) -> bool:
    """Delete one side of mirrored exterior pairs so main() must fill them back."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_count = len(clean_coords)
    clean_hole_area = _count_hole_area(out == color)
    groups = _removal_groups(out, spec, bg)

    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), len(groups))
    if lo > hi:
        return False

    clean_set = set(clean_coords)
    for _ in range(REMOVAL_RETRIES):
        target = random.randint(lo, hi)
        chosen_groups = random.sample(groups, target)
        current_set = set(clean_set)
        chosen_cells: list[tuple[int, int]] = []

        for group in chosen_groups:
            scored: list[tuple[int, float, tuple[int, int]]] = []
            for cell in group:
                if cell not in current_set:
                    continue
                trial = list(current_set - {cell})
                alt = _second_best_score(trial, true_k, width=GRID_SIZE)
                scored.append((alt, random.random(), cell))
            if not scored:
                break
            scored.sort(key=lambda item: (item[0], item[1]))
            best_alt = scored[0][0]
            pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
            cell = random.choice(pool)
            current_set.remove(cell)
            chosen_cells.append(cell)
        else:
            trial_mask = np.zeros_like(out, dtype=bool)
            for r, c in current_set:
                trial_mask[r, c] = True
            if _count_hole_area(trial_mask) != clean_hole_area:
                continue

            scores = _axis_scores(list(current_set), width=GRID_SIZE)
            if scores[true_k] - max(score for k, score in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
                continue

            for r, c in chosen_cells:
                inp[r, c] = bg
            return True

    return False


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean hollow sprites, delete mirror partners, validate."""
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
        if not _remove_pixels(inp, out, spec, bg):
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
