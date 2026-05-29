# Combo variant of 8e5c0c38 (index 4)
# combo_spec: D1/0+D5/0
# Name: 标记轴补全修复

from __future__ import annotations

"""
Design analysis for the combo variant.

combo_summary:
  - Start from the original per-sprite vertical-symmetry task, but fuse two
    changes at once:
    1. D1 inversion: asymmetry means a missing mirror partner that must be
       added, not a noisy pixel that must be deleted.
    2. D5 marker-axis: the symmetry axis is no longer inferred from the sprite;
       it is given explicitly by a nearby red marker component.

concrete_pixel_behavior:
  input:
    - 22x22 grid with one dominant background color.
    - Exactly 3 payload sprites, each monochrome and each using a different
      non-red color.
    - Each payload sprite is an incomplete odd-width vertically symmetric shape:
      some non-axis pixels from one side of mirrored pairs are missing.
    - Each sprite has one external red marker (color 2), either a single pixel
      or a short vertical segment of length 2 or 3, placed one cell above or
      below the sprite on its true symmetry column.
  output:
    - The red markers disappear.
    - Each sprite is completed into the full clean symmetric shape by adding the
      missing mirror partners back in.

behavior_trace:
  - Small example on one row. Suppose a sprite's marked axis is column 7. The
    clean row should contain the same color at columns 5 and 9, mirrored around
    7. If the input keeps column 5 but column 9 is absent, then:
      original task: if axis were inferred and the unmatched pixel treated as
        noise, column 5 would be deleted.
      D1-only variant: infer an axis from the visible pixels, then fill column 9.
      fused combo: do not infer the axis at all; read column 7 from the marker,
        then fill column 9.
    Difference from the source task: the output becomes larger, not smaller.
    Difference from pure D1: the axis comes from metadata, and the generator
    intentionally includes cases where visible pixels alone are not enough to
    recover the correct axis reliably.

complexity_calibration:
  - Marker color is fixed and visually special, so humans can quickly notice
    that red disappears and therefore encodes control information.
  - Marker styles use only a few repeated forms (1, 2, or 3 cells in one
    column), which keeps the visual language narrow.
  - Each grid still contains 3 repeated sprite-marker pairs, giving enough
    evidence that markers indicate vertical axes rather than being payload.
  - Only integer-column axes are used. This avoids introducing a second marker
    encoding for half-columns and keeps the combo centered on the intended two
    changes instead of adding a third representational wrinkle.

implicit_assumption_review_of_original_main:
  - "colors_present(... exclude={bg})":
      broken. Red is no longer payload; it is metadata and must be excluded.
  - "for each color, search all candidate axes":
      broken. The axis is no longer inferred from the payload.
  - "unmatched pixel means wrong pixel":
      broken. In the combo, unmatched means its mirror partner is missing.
  - "copying the grid is harmless":
      broken unless markers are cleared first, because marker cells must vanish
      from the output.
  - "one color layer equals one stable connected object":
      weakened by D1 inversion, because removing one side of pairs can fragment
      the visible sprite. The new solver therefore groups payload by color and
      bbox, not by connected component.

placement_legality:
  - Generation remains output-first: build the clean symmetric output sprites,
    place them into separated slots, then delete selected mirrored pixels and
    add marker cells to form the input.
  - Because deletions only shrink sprites and markers use reserved above/below
    space inside each slot, all placements stay in-bounds and non-overlapping.
  - Each placed sprite contains axis-column pixels that are never deleted, so
    the visible sprite bbox still spans the marked axis column.

visual_observability:
  - The input/output difference is easy to see: red markers disappear, and the
    colored sprites gain pixels on the reflected side.
  - The marker is always aligned to exactly one nearby sprite and separated by a
    one-cell gap, so the correspondence is human-readable from a few examples.
  - The generator rejects samples where a no-marker baseline solver would still
    reconstruct the same output. This makes the explicit-axis change active
    rather than decorative.

visual_cleanliness:
  - Marker cells never overlap payload cells.
  - The reserved gap cell prevents marker-payload merging.
  - Slot spacing keeps the three sprite-marker bundles separated.

non_degeneracy_metric:
  - For validation after implementation, compare the true output against a
    baseline solver that ignores the marker and searches the best axis per
    color. A sample is accepted only if:
      1. the true solver adds at least one payload pixel, and
      2. the baseline output differs from the clean output.
    This directly checks that both D1 and D5 are contributing.

new_main_assumptions_checked_against_generator:
  - Assumption: marker components are single-column vertical pieces.
    guaranteed by generator: every marker is length 1/2/3 in one column.
  - Assumption: every payload color belongs to exactly one sprite.
    guaranteed by generator: the 3 payload colors are unique.
  - Assumption: each sprite bbox still spans its marked column after deletions.
    guaranteed by generator: only non-axis cells are removed.
  - Assumption: marker-to-sprite pairing by bbox span and proximity is unique.
    guaranteed by generator: bundles are placed in separated slots with one
    marker per sprite.
"""

import math
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import colors_present, get_bg, label_mask, paste


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = map(int, b1)
    r0b, c0b, r1b, c1b = map(int, b2)
    if r1a < r0b:
        dr = int(r0b - r1a - 1)
    elif r1b < r0a:
        dr = int(r0a - r1b - 1)
    else:
        dr = 0
    if c1a < c0b:
        dc = int(c0b - c1a - 1)
    elif c1b < c0a:
        dc = int(c0a - c1b - 1)
    else:
        dc = 0
    return int(dr + dc)


def _marker_components(g: np.ndarray, mark: int) -> list[dict]:
    marker_lab, marker_n = label_mask(g == int(mark), connectivity=4)
    comps: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        cols = {c for _r, c in coords}
        if len(cols) != 1:
            continue
        rows = [r for r, _c in coords]
        axis_c = int(next(iter(cols)))
        comps.append(
            {
                "coords": coords,
                "axis_c": axis_c,
                "bbox": (int(min(rows)), int(axis_c), int(max(rows)), int(axis_c)),
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each marked sprite by restoring vertical mirror symmetry.

    Input structure:
    - bg is the dominant background color. Red (2) is a special marker color, not payload.
    - Three non-red monochrome sprites appear with nearby red single-pixel or short vertical markers that indicate their symmetry columns.
    - Each sprite is an incomplete version of a vertically symmetric shape: some non-axis pixels are missing on one side of mirrored pairs.

    Rule:
    - Remove the red markers from the output.
    - Pair each marker with the nearby sprite whose visible color bbox spans the marker column.
    - Use that marked column as the vertical reflection axis, and whenever a sprite pixel lacks its same-color reflected partner, add the missing partner on background.

    Key insight: The marker already gives the correct axis, so the asymmetry should be interpreted as missing reflected structure to complete, not as noise to erase or as evidence for axis search.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, W = g.shape
    MARK = 2

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    payload_colors = colors_present(g, exclude={bg, MARK})
    assert payload_colors, "grid must contain at least one payload color"

    sprites: list[dict] = []
    for color in payload_colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == int(color))]
        if not coords:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        sprites.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    used_colors: set[int] = set()
    for marker in sorted(_marker_components(g, MARK), key=lambda rec: rec["bbox"][0]):
        axis_c = int(marker["axis_c"])
        candidates = [
            sprite
            for sprite in sprites
            if sprite["color"] not in used_colors and int(sprite["bbox"][1]) <= axis_c <= int(sprite["bbox"][3])
        ]
        if not candidates:
            continue

        sprite = min(
            candidates,
            key=lambda rec: (
                _bbox_gap(marker["bbox"], rec["bbox"]),
                abs(axis_c - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2)),
            ),
        )
        used_colors.add(int(sprite["color"]))

        color = int(sprite["color"])
        k = int(2 * axis_c)
        for r, c in sprite["coords"]:
            cc = int(k - c)
            if 0 <= cc < W and int(g[int(r), cc]) != color and int(out[int(r), cc]) == int(bg):
                out[int(r), int(cc)] = int(color)

    return out


# --------------- generator (combo D1 inversion + D5 marker-axis) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
MARK = 2
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 9, 8), (0, 13, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 9, 9), (0, 12, 9, 20), (12, 3, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 6
ODD_SIDE_W_RANGE = range(1, 6)  # widths 3..11
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: missing mirror pixels ──
MISSING_FRAC_LO = 0.16
MISSING_FRAC_HI = 0.40
MISSING_MAX = 12

# ── CFG: marker placement ──
MARKER_GAP = 1
MARKER_LENGTHS = (1, 2, 3)

# ── CFG: retry limits ──
SPRITE_RETRIES = 200
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = int(k - c)
            if 0 <= cc < width and (int(r), int(cc)) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1):
        for side_w in ODD_SIDE_W_RANGE:
            w = 2 * side_w + 1
            if w > max_w:
                continue
            area = h * w
            lo = max(PIXEL_COUNT_MIN, math.ceil(DENSITY_LO * area))
            hi = min(PIXEL_COUNT_MAX, math.floor(DENSITY_HI * area))
            if lo > hi:
                continue
            for target_count in range(lo, hi + 1):
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    weights = [1.0 / (h * w) for h, w, _t, _c in options]
    return random.choices(options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((int(r), int(c)))

    remaining = int(left_count - len(cells))
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(int(r), int(c)) for r in range(h) for c in range(side_w) if (int(r), int(c)) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    rows[random.sample(range(h), center_count)] = True
    return rows


def _crop_to_bbox(sprite: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(sprite == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = int(w - 1 - c)
            sprite[int(r), int(c)] = int(color)
            sprite[int(r), int(mirror_c)] = int(color)
        sprite[centers, int(side_w)] = int(color)
        sprite = _crop_to_bbox(sprite, color)

        if sprite.shape[0] > max_h or sprite.shape[1] > max_w or sprite.shape[1] % 2 == 0:
            continue
        count = int(np.sum(sprite == int(color)))
        area = int(sprite.shape[0] * sprite.shape[1])
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        density = count / area
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue
        lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1 or int(np.max(lab)) == 0:
            continue
        return sprite
    raise ValueError("failed to build clean sprite")


def _choose_marker_styles() -> tuple[list[int], list[str]]:
    lengths = [1, random.choice([2, 3]), random.choice(MARKER_LENGTHS)]
    sides = ["above", "below", random.choice(["above", "below"])]
    random.shuffle(lengths)
    random.shuffle(sides)
    return lengths, sides


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        marker_len = int(spec["marker_len"])
        marker_side = str(spec["marker_side"])
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)
        bundle_h = int(h + MARKER_GAP + marker_len)
        if bundle_h > slot_h or w > slot_w:
            return None

        top = random.randint(int(r0s), int(r1s - bundle_h + 1))
        c0 = random.randint(int(c0s), int(c1s - w + 1))
        axis_c = int(c0 + w // 2)
        if marker_side == "above":
            marker_r0 = int(top)
            sprite_r0 = int(top + marker_len + MARKER_GAP)
        else:
            sprite_r0 = int(top)
            marker_r0 = int(sprite_r0 + h + MARKER_GAP)

        marker_cells = [(int(marker_r0 + i), int(axis_c)) for i in range(marker_len)]
        paste(out, sprite, int(sprite_r0), int(c0), transparent=0)
        spec["sprite_r0"] = int(sprite_r0)
        spec["sprite_c0"] = int(c0)
        spec["axis_c"] = int(axis_c)
        spec["true_k"] = int(2 * axis_c)
        spec["marker_cells"] = marker_cells
        placed.append(spec)

    return out, placed


def _removal_candidates(clean_mask: np.ndarray, true_k: int, side: str) -> list[tuple[int, int]]:
    rows, cols = np.where(clean_mask)
    candidates: list[tuple[int, int]] = []
    axis_c = true_k // 2
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_c = int(true_k - c)
        if mirror_c == c:
            continue
        if not (0 <= mirror_c < clean_mask.shape[1] and clean_mask[int(r), int(mirror_c)]):
            continue
        if side == "left" and int(c) >= axis_c:
            continue
        if side == "right" and int(c) <= axis_c:
            continue
        candidates.append((int(r), int(c)))
    return candidates


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_mask = out == int(color)
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    side = random.choice(["left", "right"])
    candidates = _removal_candidates(clean_mask, true_k, side)
    pair_count = len(candidates)
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
            scores = _axis_scores(test_coords, GRID_SIZE)
            true_score = int(scores[true_k])
            other_best = max(s for k, s in scores.items() if k != true_k)
            scored.append((int(true_score - other_best), int(true_score), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        best_gap = scored[0][0]
        pool = [cell for gap, _true, _rnd, cell in scored if gap <= best_gap + 1]
        cell = random.choice(pool)

        chosen.append(cell)
        current_coords.remove(cell)
        forbidden.add(cell)
        mirror = (int(cell[0]), int(true_k - cell[1]))
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    for r, c in chosen:
        inp[int(r), int(c)] = int(bg)
    return True


def _baseline_infer_axis_and_complete(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, W = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)
    for color in colors_present(g, exclude={bg, MARK}):
        pts = np.argwhere(g == int(color))
        if pts.size == 0:
            continue
        best_score = -1
        best_k = 0
        for k in range(0, 2 * (W - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k - c)
                if 0 <= cc < W and int(g[int(r), int(cc)]) == int(color):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            cc = int(best_k - c)
            if 0 <= cc < W and int(g[int(r), int(cc)]) != int(color) and int(out[int(r), int(cc)]) == int(bg):
                out[int(r), int(cc)] = int(color)
    return out


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    solved = main(inp)
    baseline = _baseline_infer_axis_and_complete(inp)
    added_payload = int(np.count_nonzero((inp == int(get_bg(inp))) & (solved != int(get_bg(inp)))))
    marker_cells = int(np.count_nonzero(inp == int(MARK)))
    baseline_diff = int(np.count_nonzero(baseline != out))
    return {
        "added_payload": added_payload,
        "marker_cells": marker_cells,
        "baseline_diff": baseline_diff,
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    marker_lengths, marker_sides = _choose_marker_styles()

    sprite_specs: list[dict] = []
    try:
        for color, slot, marker_len, marker_side in zip(obj_colors, slots, marker_lengths, marker_sides):
            max_h = int(slot[2] - slot[0] + 1 - MARKER_GAP - marker_len)
            max_w = int(slot[3] - slot[1] + 1)
            sprite = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "marker_len": int(marker_len),
                    "marker_side": str(marker_side),
                }
            )
    except ValueError:
        return None

    placed = _slot_place_sprites(int(bg), sprite_specs, slots)
    if placed is None:
        return None
    out, placed_specs = placed

    inp = out.copy()
    for spec in placed_specs:
        if not _remove_pixels(inp, out, spec):
            return None
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None

    metrics = _fusion_metrics(inp, out)
    if metrics["added_payload"] <= 0 or metrics["marker_cells"] < NUM_SPRITES or metrics["baseline_diff"] <= 0:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
