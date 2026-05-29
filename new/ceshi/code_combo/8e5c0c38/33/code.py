# Combo variant of 8e5c0c38 (index 33)
# combo_spec: D1/0+D2/1+D6/0
# Name: 自适应方向四重对称补全

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - Original task: for each color, search only vertical axes, then delete pixels that
#      lack a same-color mirror partner.
#    - Combo task: each color is now a damaged subset of a symmetric target, so the
#      solver must add missing pixels. There are three object modes in one grid:
#      vertical-only, horizontal-only, and fourfold (both axes at once).
#    - Small trace on one 2x2 fourfold orbit with true axes through its center:
#      clean orbit = {(1,1),(1,2),(2,1),(2,2)}.
#      If input keeps {(1,1),(1,2)} only, a vertical-only completion does nothing,
#      a horizontal-only completion adds the bottom pair, and fourfold completion also
#      adds the bottom pair. If a second damaged orbit in the same sprite keeps
#      {(4,4),(5,4)} only, horizontal-only completion now fails there while vertical-only
#      succeeds there. The mixed evidence across both damaged orbits forces a fourfold
#      interpretation of that sprite.
# b) Complexity calibration:
#    - Every generated grid contains exactly 3 sprites: one vertical-only, one
#      horizontal-only, and one fourfold. That keeps the freedom set small but makes the
#      fusion visible in every example.
#    - The original 22x22 canvas, 3 separated slots, distinct object colors, and
#      6-46 clean-pixel regime are preserved to avoid overlap ambiguity.
#    - Fourfold sprites are required to keep some intact full orbits plus a mix of
#      row-pair and column-pair fragments, so neither single-axis model can explain the
#      whole object even though each direction has real support.
# c) Hidden assumption review of the original main():
#    - Original assumption: every sprite is vertical. Broken by D2/1, so the solver now
#      scores vertical and horizontal axes independently.
#    - Original assumption: asymmetry means noise to delete. Broken by D1/0, so the
#      solver must read the original pixels as evidence and fill mirrors in `out`.
#    - Original assumption: one mirror partner check is enough. Broken by D6/0, because
#      some sprites need both axes simultaneously. The new solver therefore first infers
#      the model type, then either fills one mirrored partner or an entire fourfold orbit.
#    - New implicit assumption: same-color pixels belong to one sprite with one dominant
#      symmetry model. The generator preserves one distinct color per object and rejects
#      samples where the shared `_infer_model()` would disagree.
# d) Placement legality:
#    - Generation stays output-first: build clean sprites, place them into the legacy
#      non-overlapping slots, then delete pixels to create the input.
#    - All damage is subtractive, so inputs remain in-bounds and separated; solver-added
#      pixels can only reoccupy cells that were already part of the clean output.
# e) Visual observability:
#    - Humans can see three different repair behaviors in the same grid: one object grows
#      left-right, one grows up-down, and one needs both directions across different local
#      fragments. That makes the fusion distinguishable from each single variant.
# f) Visual cleanliness:
#    - Objects remain monochrome and isolated in three far-apart slots. No auxiliary
#      markers, sorting tricks, or color-coded metadata are introduced.
# g) Color reasoning:
#    - Color is only an object separator. The repair rule depends on geometry, not on the
#      numeric color ids.
# h) Quantified anti-degeneration target:
#    - The generator requires its fourfold sprite to reconstruct differently from both
#      pure vertical completion and pure horizontal completion on the same damaged input.
#      This prevents collapse to the D1+D2 subcase.
# i) New solver assumptions and generator guarantees:
#    - Assumption: the inferred axes are identifiable from surviving mirrored evidence.
#      Guarantee: single-axis sprites must keep an axis-score gap, and fourfold sprites
#      must keep balanced vertical/horizontal support plus intact full orbits.
#    - Assumption: fourfold completion should only trigger when both directions matter.
#      Guarantee: the fourfold sprite includes both row-pair and column-pair fragments.


# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: single-axis clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: damage ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12
FOURFOLD_PLAN_RETRIES = 120

# ── CFG: inference thresholds ──
SINGLE_AXIS_GAP_MIN = 3
FINAL_SINGLE_GAP_MIN = 2
FOURFOLD_BALANCE_RATIO = 0.68
FOURFOLD_COMPLETE_MIN = 4
FOURFOLD_MIN_PAIR_SCORE = 4

# ── CFG: retry limits ──
SPRITE_RETRIES = 320
GEN_RETRIES = 700

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < width and (int(r), cc) in pts:
                    score += 1
            scores[k] = score
    elif direction == "horizontal":
        for k in range(0, 2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < height and (rr, int(c)) in pts:
                    score += 1
            scores[k] = score
    else:
        raise ValueError(f"unknown direction: {direction}")
    return scores


def _best_axis(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> tuple[int, int]:
    best_k = 0
    best_score = -1
    for k, score in _axis_scores(coords, height, width, direction).items():
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k, best_score


def _orbit_cells(
    r: int, c: int, ky: int, kx: int, height: int, width: int
) -> tuple[tuple[int, int], ...] | None:
    rows = {int(r), int(ky) - int(r)}
    cols = {int(c), int(kx) - int(c)}
    orbit: set[tuple[int, int]] = set()
    for rr in rows:
        for cc in cols:
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            orbit.add((int(rr), int(cc)))
    return tuple(sorted(orbit))


def _complete_orbit_pixels(
    pts: set[tuple[int, int]], ky: int, kx: int, height: int, width: int
) -> int:
    seen: set[tuple[tuple[int, int], ...]] = set()
    total = 0
    for r, c in pts:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is None or orbit in seen:
            continue
        seen.add(orbit)
        if set(orbit) <= pts:
            total += len(orbit)
    return total


def _infer_model(coords: list[tuple[int, int]], height: int, width: int) -> dict[str, int | str]:
    pts = {(int(r), int(c)) for r, c in coords}
    kx, vertical_score = _best_axis(coords, height, width, "vertical")
    ky, horizontal_score = _best_axis(coords, height, width, "horizontal")
    complete_pixels = _complete_orbit_pixels(pts, ky, kx, height, width)

    if (
        min(vertical_score, horizontal_score) >= FOURFOLD_MIN_PAIR_SCORE
        and min(vertical_score, horizontal_score) >= FOURFOLD_BALANCE_RATIO * max(vertical_score, horizontal_score)
        and complete_pixels >= FOURFOLD_COMPLETE_MIN
    ):
        return {
            "kind": "fourfold",
            "kx": int(kx),
            "ky": int(ky),
            "vertical_score": int(vertical_score),
            "horizontal_score": int(horizontal_score),
            "complete_pixels": int(complete_pixels),
        }

    if vertical_score >= horizontal_score:
        return {
            "kind": "vertical",
            "k": int(kx),
            "score": int(vertical_score),
            "other_score": int(horizontal_score),
        }
    return {
        "kind": "horizontal",
        "k": int(ky),
        "score": int(horizontal_score),
        "other_score": int(vertical_score),
    }


def _completed_coords(
    coords: list[tuple[int, int]], height: int, width: int, model: dict[str, int | str]
) -> set[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c in coords}
    out = set(pts)
    if model["kind"] == "vertical":
        k = int(model["k"])
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width:
                out.add((r, int(cc)))
    elif model["kind"] == "horizontal":
        k = int(model["k"])
        for r, c in pts:
            rr = k - r
            if 0 <= rr < height:
                out.add((int(rr), c))
    else:
        ky = int(model["ky"])
        kx = int(model["kx"])
        for r, c in pts:
            orbit = _orbit_cells(r, c, ky, kx, height, width)
            if orbit is not None:
                out.update(orbit)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color layer by choosing the strongest symmetric repair model.

    Input structure:
    - bg = background. Three non-bg colors appear as separate sprites in distant slots.
    - Some sprites are damaged versions of a single-axis mirror shape, while others are damaged versions of a shape that is symmetric in both vertical and horizontal directions.
    - The damage removes pixels from the clean target, so the input is always a subset of the intended output.

    Rule:
    - For each non-bg color independently, score the best vertical axis and the best horizontal axis, then decide whether the sprite is better explained as vertical-only, horizontal-only, or fourfold.
    - If the chosen model is single-axis, fill the missing mirror partner for every observed pixel across that axis.
    - If the chosen model is fourfold, fill the entire orbit induced by the chosen vertical and horizontal axes.

    Key insight: The task is no longer "erase bad pixels"; every visible pixel is trusted evidence. The real challenge is deciding, per sprite, whether one direction is enough or whether mixed left-right and up-down clues require full fourfold completion.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts_arr = np.argwhere(g == int(col))
        if pts_arr.size == 0:
            continue

        coords = [(int(r), int(c)) for r, c in pts_arr]
        model = _infer_model(coords, height, width)

        if model["kind"] == "vertical":
            k = int(model["k"])
            for r, c in coords:
                cc = k - c
                if 0 <= cc < width and int(out[r, cc]) == bg:
                    out[r, cc] = int(col)
        elif model["kind"] == "horizontal":
            k = int(model["k"])
            for r, c in coords:
                rr = k - r
                if 0 <= rr < height and int(out[rr, c]) == bg:
                    out[rr, c] = int(col)
        else:
            ky = int(model["ky"])
            kx = int(model["kx"])
            for r, c in coords:
                orbit = _orbit_cells(r, c, ky, kx, height, width)
                if orbit is None:
                    continue
                for rr, cc in orbit:
                    if int(out[rr, cc]) == bg:
                        out[rr, cc] = int(col)

    return out


def _feasible_single_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                lo = math.ceil(DENSITY_LO * area)
                hi = math.floor(DENSITY_HI * area)
                if not (lo <= target_count <= hi):
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
                lo = math.ceil(DENSITY_LO * area)
                hi = math.floor(DENSITY_HI * area)
                if not (lo <= target_count <= hi):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_single_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options = _feasible_single_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible single-axis sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


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
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _vertical_candidate(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    h, w, target_count, center_count = _choose_single_shape_spec(max_h, max_w)
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
    return sprite, w - 1


def _build_single_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(color, max_h, max_w)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(color, max_w, max_h)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown single-axis direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if not coords:
            continue
        model = _infer_model(coords, *sprite.shape)
        if model["kind"] != direction:
            continue
        if int(model["k"]) != true_k:
            continue
        if int(model["score"]) - int(model["other_score"]) < SINGLE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build single-axis sprite")


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    forced = {(0, 0)}
    if left_w > 1:
        forced.add((0, left_w - 1))
    if top_h > 1:
        forced.add((top_h - 1, 0))
    if top_h > 1 and left_w > 1:
        forced.add((top_h - 1, left_w - 1))

    cells = set(forced)
    fill_prob = random.uniform(0.25, 0.78)
    for r in range(top_h):
        for c in range(left_w):
            if (r, c) in forced:
                continue
            if random.random() < fill_prob:
                cells.add((r, c))
    if len(cells) == len(forced) and top_h * left_w > len(forced):
        pool = [(r, c) for r in range(top_h) for c in range(left_w) if (r, c) not in cells]
        cells.add(random.choice(pool))
    return cells


def _mirror_canonical(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in cells:
        rows = {int(r), int(h - 1 - r)}
        cols = {int(c), int(w - 1 - c)}
        for rr in rows:
            for cc in cols:
                pts.add((rr, cc))
    return pts


def _four_cell_orbits(
    pts: set[tuple[int, int]], ky: int, kx: int, height: int, width: int
) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r, c in sorted(pts):
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is None or orbit in seen:
            continue
        seen.add(orbit)
        if len(orbit) == 4 and set(orbit) <= pts:
            out.append(orbit)
    return out


def _build_fourfold_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    for _ in range(SPRITE_RETRIES):
        h = random.randint(max(4, HEIGHT_MIN), min(HEIGHT_MAX, max_h))
        w = random.randint(max(4, WIDTH_MIN), min(WIDTH_MAX, max_w))
        top_h = (h + 1) // 2
        left_w = (w + 1) // 2
        canonical = _canonical_cells(top_h, left_w)
        pts = _mirror_canonical(canonical, h, w)

        count = len(pts)
        area = h * w
        if count < 14:
            continue
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue

        sprite = np.zeros((h, w), dtype=int)
        for r, c in pts:
            sprite[r, c] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        model = _infer_model(coords, h, w)
        if model["kind"] != "fourfold":
            continue
        if int(model["ky"]) != h - 1 or int(model["kx"]) != w - 1:
            continue
        if len(_four_cell_orbits(set(coords), h - 1, w - 1, h, w)) < 2:
            continue
        return sprite, h - 1, w - 1
    raise ValueError("failed to build fourfold sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict[str, object]], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict[str, object]] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        current = dict(spec)
        sprite = np.array(current["sprite"], dtype=int)
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        current["r0"] = int(r0)
        current["c0"] = int(c0)
        if current["kind"] == "fourfold":
            current["true_ky"] = int(current["axis_ky_local"]) + 2 * int(r0)
            current["true_kx"] = int(current["axis_kx_local"]) + 2 * int(c0)
        elif current["kind"] == "vertical":
            current["true_k"] = int(current["axis_k_local"]) + 2 * int(c0)
        else:
            current["true_k"] = int(current["axis_k_local"]) + 2 * int(r0)
        placed.append(current)
    return out, placed


def _single_removal_candidates(
    clean_pts: set[tuple[int, int]], direction: str, true_k: int, height: int, width: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in sorted(clean_pts):
        if direction == "vertical":
            rr, cc = r, true_k - c
        else:
            rr, cc = true_k - r, c
        if (rr, cc) == (r, c):
            continue
        if 0 <= rr < height and 0 <= cc < width and (int(rr), int(cc)) in clean_pts:
            candidates.append((r, c))
    return candidates


def _remove_single_pixels(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> bool:
    bg = int(get_bg(out))
    color = int(spec["color"])
    direction = str(spec["kind"])
    true_k = int(spec["true_k"])
    clean_pts = {tuple(x) for x in np.argwhere(out == color)}
    clean_count = len(clean_pts)

    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), len(clean_pts) // 2)
    if lo > hi:
        return False

    candidates = _single_removal_candidates(clean_pts, direction, true_k, GRID_SIZE, GRID_SIZE)
    if len(candidates) < lo:
        return False

    missing_target = random.randint(lo, hi)
    current = set(clean_pts)
    forbidden: set[tuple[int, int]] = set()
    removed: list[tuple[int, int]] = []

    for _ in range(missing_target):
        viable: list[tuple[int, float, tuple[int, int]]] = []
        for cell in candidates:
            if cell in forbidden or cell not in current:
                continue
            test_coords = sorted(current - {cell})
            model = _infer_model(test_coords, GRID_SIZE, GRID_SIZE)
            if model["kind"] != direction:
                continue
            if int(model["k"]) != true_k:
                continue
            if int(model["score"]) - int(model["other_score"]) < FINAL_SINGLE_GAP_MIN:
                continue
            viable.append((int(model["other_score"]), random.random(), cell))
        if not viable:
            return False

        viable.sort(key=lambda item: (item[0], item[1]))
        cell = viable[0][2]
        current.remove(cell)
        removed.append(cell)
        forbidden.add(cell)
        if direction == "vertical":
            mirror = (cell[0], true_k - cell[1])
        else:
            mirror = (true_k - cell[0], cell[1])
        if 0 <= mirror[0] < GRID_SIZE and 0 <= mirror[1] < GRID_SIZE:
            forbidden.add((int(mirror[0]), int(mirror[1])))

    for r, c in removed:
        inp[r, c] = bg
    return True


def _row_pair_options(orbit: tuple[tuple[int, int], ...]) -> list[set[tuple[int, int]]]:
    rows = sorted({r for r, _ in orbit})
    return [
        {cell for cell in orbit if cell[0] == rows[0]},
        {cell for cell in orbit if cell[0] == rows[1]},
    ]


def _col_pair_options(orbit: tuple[tuple[int, int], ...]) -> list[set[tuple[int, int]]]:
    cols = sorted({c for _, c in orbit})
    return [
        {cell for cell in orbit if cell[1] == cols[0]},
        {cell for cell in orbit if cell[1] == cols[1]},
    ]


def _extra_fragment_options(orbit: tuple[tuple[int, int], ...]) -> list[set[tuple[int, int]]]:
    opts: list[set[tuple[int, int]]] = []
    opts.extend(_row_pair_options(orbit))
    opts.extend(_col_pair_options(orbit))
    for missing in orbit:
        opts.append(set(orbit) - {missing})  # keep 3, remove 1
    for keep in orbit:
        opts.append({keep})  # keep 1, remove 3
    return opts


def _remove_fourfold_pixels(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> bool:
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    clean_pts = {tuple(x) for x in np.argwhere(out == color)}
    clean_count = len(clean_pts)

    lo = max(4, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count))
    if lo > hi:
        return False

    orbits = _four_cell_orbits(clean_pts, true_ky, true_kx, GRID_SIZE, GRID_SIZE)
    if len(orbits) < 2:
        return False

    targets = list(range(lo, hi + 1))
    random.shuffle(targets)

    for missing_target in targets:
        for _ in range(FOURFOLD_PLAN_RETRIES):
            row_orbit, col_orbit = random.sample(orbits, 2)
            kept: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {
                row_orbit: random.choice(_row_pair_options(row_orbit)),
                col_orbit: random.choice(_col_pair_options(col_orbit)),
            }
            removed_total = 4
            remaining = missing_target - removed_total
            if remaining < 0:
                continue

            extras = [orbit for orbit in orbits if orbit not in kept]
            random.shuffle(extras)
            success = True
            for orbit in extras:
                if remaining == 0:
                    break
                choices = [opt for opt in _extra_fragment_options(orbit) if 0 < 4 - len(opt) <= remaining]
                if not choices:
                    continue
                exact = [opt for opt in choices if 4 - len(opt) == remaining]
                if exact:
                    choice = random.choice(exact)
                else:
                    choices.sort(key=lambda opt: abs((4 - len(opt)) - remaining))
                    choice = random.choice(choices[: min(4, len(choices))])
                kept[orbit] = set(choice)
                delta = 4 - len(choice)
                removed_total += delta
                remaining -= delta

            if remaining != 0:
                success = False
            if not success:
                continue

            removed_cells: set[tuple[int, int]] = set()
            for orbit, keep_cells in kept.items():
                removed_cells.update(set(orbit) - set(keep_cells))

            current = sorted(clean_pts - removed_cells)
            model = _infer_model(current, GRID_SIZE, GRID_SIZE)
            if model["kind"] != "fourfold":
                continue
            if int(model["ky"]) != true_ky or int(model["kx"]) != true_kx:
                continue

            vertical_only = _completed_coords(current, GRID_SIZE, GRID_SIZE, {"kind": "vertical", "k": true_kx})
            horizontal_only = _completed_coords(current, GRID_SIZE, GRID_SIZE, {"kind": "horizontal", "k": true_ky})
            if vertical_only == clean_pts:
                continue
            if horizontal_only == clean_pts:
                continue
            if _completed_coords(current, GRID_SIZE, GRID_SIZE, model) != clean_pts:
                continue

            for r, c in removed_cells:
                inp[r, c] = bg
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)
    kinds = ["vertical", "horizontal", "fourfold"]
    random.shuffle(kinds)

    sprite_specs: list[dict[str, object]] = []
    try:
        for color, slot, kind in zip(obj_colors, slots, kinds):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            if kind == "fourfold":
                sprite, axis_ky_local, axis_kx_local = _build_fourfold_sprite(color, max_h, max_w)
                sprite_specs.append(
                    {
                        "color": color,
                        "sprite": sprite,
                        "kind": kind,
                        "axis_ky_local": axis_ky_local,
                        "axis_kx_local": axis_kx_local,
                    }
                )
            else:
                sprite, axis_k_local = _build_single_sprite(color, max_h, max_w, kind)
                sprite_specs.append(
                    {
                        "color": color,
                        "sprite": sprite,
                        "kind": kind,
                        "axis_k_local": axis_k_local,
                    }
                )
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if spec["kind"] == "fourfold":
            if not _remove_fourfold_pixels(inp, out, spec):
                return None
        else:
            if not _remove_single_pixels(inp, out, spec):
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
