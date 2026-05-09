# Combo variant of 8e5c0c38 (index 26)
# combo_spec: D5/2+D16/0
# Name: 整数轴外接矩形中心修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise each colored sprite by mirroring only around the integer axis implied by its own bbox.

    Input structure:
    - The background fills most cells, and each non-background color forms one nearly symmetric sprite.
    - Extra pixels may enlarge a sprite's bbox, so the bbox center can disagree with the visually best symmetry axis.

    Rule:
    - For each non-background color, compute its tight bounding box and take the bbox center as the vertical axis.
    - Only integer axes are allowed: if the bbox has two middle columns, use the left middle column and delete every pixel without a same-color mirror across that column.

    Key insight:
    - The axis is fixed by the current bbox rather than searched from the pixel evidence, so a convincing half-column symmetry is ignored if the bbox says to keep the left-center integer column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        axis_col = (c0 + c1) // 2

        for r, c in pts:
            rr = int(r)
            cc = 2 * axis_col - int(c)
            if not (0 <= cc < w and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return out


# Design analysis for combo D5/2 + D16/0
# a) Concrete trace:
#    Start with a clean width-5 sprite on local columns 0..4, symmetric about column 2.
#    Add a few noise pixels in a new column 5, aligned with rows that already use column 0.
#    The original task would search all k and can prefer k=5 (axis 2.5), because each new col-5
#    pixel now pairs with an existing col-0 pixel. The fused task instead uses the noisy bbox
#    [0..5], whose center is 2.5, but integer-only rounding chooses the left middle column 2.
#    Therefore col-5 noise is deleted even when the half-axis looks more convincing.
# b) Complexity calibration:
#    Every board still has only 3 separated sprites, and every sprite repeats the same right-fringe
#    cue across multiple rows. That makes the combo visible without adding many unrelated cases.
# c) Hidden assumptions changed in main():
#    - One object per color: bbox is computed over all cells of that color.
#    - Axis tie rule is explicit: floor((c0 + c1) / 2), i.e. the left middle column.
#    - Mirror checks use the original grid, so edit order does not matter.
#    The generator enforces all three.
# d) Placement legality:
#    Clean sprites have odd width; noisy sprites are exactly one column wider on the right.
#    Placement is validated against the noisy width so nothing overlaps or leaves the slot.
# e-f) Visual observability and cleanliness:
#    Noise is a single extra fringe column, always 4-adjacent to the clean sprite, so the
#    input/output difference is easy to see and objects remain visually separated.
# g) No invisible color logic:
#    Colors are sampled randomly and play no role in deciding the axis or deletions.
# h) Non-degeneration target:
#    The generator rejects any sprite unless the original best-axis search would prefer the
#    half-axis immediately to the right of the fused integer axis.
# i) New solver assumptions audited against the generator:
#    main() assumes the color bbox is meaningful and that no same-color objects are disjointly
#    mixed across the board. The generator uses distinct colors and fixed non-overlapping slots.


# --------------- generator ---------------

GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 3
HEIGHT_MAX = 8
SIDE_W_MIN = 2
SIDE_W_MAX = 4
DENSITY_LO = 0.35
DENSITY_HI = 0.89

NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

SPRITE_RETRIES = 200
GEN_RETRIES = 400
FLIP_UD_PROB = 0.5


def _axis_score(coords: list[tuple[int, int]], k: int, width: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    score = 0
    for r, c in pts:
        cc = int(k) - int(c)
        if 0 <= cc < width and (int(r), cc) in pts:
            score += 1
    return score


def _best_axis(coords: list[tuple[int, int]], width: int) -> tuple[int, int]:
    best_k = 0
    best_score = -1
    for k in range(0, 2 * (width - 1) + 1):
        score = _axis_score(coords, k, width)
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k), int(best_score)


def _apply_fixed_axis_local(grid: np.ndarray, color: int) -> np.ndarray:
    pts = np.argwhere(grid == color)
    out = grid.copy()
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    axis_col = (c0 + c1) // 2
    for r, c in pts:
        cc = 2 * axis_col - int(c)
        if not (0 <= cc < grid.shape[1] and int(grid[int(r), cc]) == color):
            out[int(r), int(c)] = 0
    return out


def _make_clean_sprite(color: int, max_h: int, max_w_input: int) -> np.ndarray:
    max_h = min(max_h, HEIGHT_MAX)
    max_side = min(SIDE_W_MAX, (max_w_input - 2) // 2)
    if max_h < HEIGHT_MIN or max_side < SIDE_W_MIN:
        raise ValueError("slot too small for fused sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.randint(HEIGHT_MIN, max_h)
        sides = list(range(SIDE_W_MIN, max_side + 1))
        side_w = random.choices(sides, weights=[1.8 ** (-abs(s - 2)) for s in sides], k=1)[0]
        inner_fill = random.uniform(0.58, 0.9)
        center_fill = random.uniform(0.15, 0.45)

        left = np.zeros((h, side_w), dtype=bool)
        left[:, 0] = True
        if side_w > 1:
            left[:, 1:] = np.random.random((h, side_w - 1)) < inner_fill
            left[random.randrange(h), -1] = True

        centers = np.random.random(h) < center_fill
        centers[random.randrange(h)] = True

        w = 2 * side_w + 1
        sprite = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(side_w):
                if left[r, c]:
                    sprite[r, c] = color
                    sprite[r, w - 1 - c] = color
            if centers[r]:
                sprite[r, side_w] = color

        count = int(np.count_nonzero(sprite))
        density = count / float(sprite.size)
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue

        noise_lo = max(1, math.ceil(NOISE_FRAC_LO * count))
        noise_hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * count), h)
        if noise_lo > noise_hi:
            continue
        return sprite

    raise ValueError("failed to build clean sprite")


def _make_noisy_variant(clean: np.ndarray, color: int) -> tuple[np.ndarray, dict] | None:
    h, w = clean.shape
    axis_col = w // 2
    fixed_k = 2 * axis_col
    alt_k = fixed_k + 1
    width_input = w + 1

    clean_coords = [tuple(x) for x in np.argwhere(clean == color)]
    clean_count = len(clean_coords)
    alt_clean_score = _axis_score(clean_coords, alt_k, width_input)

    noise_lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    noise_hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count), h)
    needed = max(noise_lo, (clean_count - alt_clean_score) // 2 + 1)
    if needed > noise_hi:
        return None

    noise_target = random.randint(needed, noise_hi)
    rows = sorted(random.sample(range(h), noise_target))

    inp = np.zeros((h, width_input), dtype=int)
    inp[:, :w] = clean
    inp[rows, w] = color
    target = np.zeros((h, width_input), dtype=int)
    target[:, :w] = clean

    input_coords = [tuple(x) for x in np.argwhere(inp == color)]
    best_k, best_score = _best_axis(input_coords, width_input)
    fixed_score = _axis_score(input_coords, fixed_k, width_input)
    alt_score = _axis_score(input_coords, alt_k, width_input)
    if best_k != alt_k:
        return None
    if alt_score <= fixed_score:
        return None
    if not np.array_equal(_apply_fixed_axis_local(inp, color), target):
        return None

    meta = {
        "fixed_k_local": fixed_k,
        "best_k_local": best_k,
        "fixed_score_local": fixed_score,
        "best_score_local": best_score,
        "noise_rows": rows,
    }
    return inp, meta


def _build_sprite_pair(color: int, max_h: int, max_w_input: int) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(SPRITE_RETRIES):
        clean = _make_clean_sprite(color, max_h, max_w_input)
        variant = _make_noisy_variant(clean, color)
        if variant is None:
            continue
        inp, meta = variant
        return clean, inp, meta
    raise ValueError("failed to build fused sprite pair")


def _place_case(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        clean = spec["clean"]
        noisy = spec["noisy"]
        h, w_in = noisy.shape
        w_out = clean.shape[1]
        if h > r1s - r0s + 1 or w_in > c1s - c0s + 1:
            return None

        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w_in + 1)
        paste(inp, noisy, r0, c0, transparent=0)
        paste(out, clean, r0, c0, transparent=0)

        meta = dict(spec["meta"])
        meta["color"] = spec["color"]
        meta["r0"] = r0
        meta["c0"] = c0
        meta["height"] = h
        meta["fixed_axis_col_abs"] = c0 + (w_out // 2)
        meta["fixed_k_abs"] = 2 * meta["fixed_axis_col_abs"]
        meta["best_k_abs"] = meta["fixed_k_abs"] + 1
        placed.append(meta)

    return inp, out, placed


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict] = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w_input = slot[3] - slot[1] + 1
            clean, noisy, meta = _build_sprite_pair(color, max_h, max_w_input)
            sprite_specs.append({"color": color, "clean": clean, "noisy": noisy, "meta": meta})
    except ValueError:
        return None

    placed = _place_case(bg, sprite_specs, slots)
    if placed is None:
        return None
    inp, out, meta = placed

    if random.random() < FLIP_UD_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
        for item in meta:
            item["r0"] = GRID_SIZE - item["r0"] - item["height"]

    if not np.array_equal(main(inp), out):
        return None
    return inp, out, meta


def _generate_case() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")


def generate_input() -> np.ndarray:
    inp, _out, _meta = _generate_case()
    return inp
