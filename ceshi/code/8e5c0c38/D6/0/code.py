# Variant of 8e5c0c38
# Dimension: D6 — 精灵的对称类型 (transform)
# Change: 垂直镜像对称（bilateral mirror symmetry about a vertical axis）... → 四重对称（同时垂直和水平轴对称）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior:
#    Each non-background color still represents one noisy sprite. The clean target is no
#    longer just left/right mirrored: every kept pixel must belong to a full orbit under
#    both a vertical axis and a horizontal axis of that sprite. In grid terms, a pixel at
#    (r, c) must retain same-color support at (r, mc), (mr, c), and (mr, mc), with
#    mc = kx - c and mr = ky - r, modulo the usual axis degeneracies when a pixel lies on
#    one of the axes.
#    Small trace:
#    - Original logic: choose only kx, keep a pixel if its left/right mate exists.
#    - Variant logic: choose (ky, kx), keep a pixel only if its whole fourfold orbit exists.
#    - Difference: a noise pair that is vertically mirrored but appears only above the
#      horizontal axis survives the old rule and is deleted by the new rule.
# b) Complexity calibration:
#    The clean sprites are generated with irregular quadrant structure and mirrored into
#    fourfold symmetry. Noise is not limited to singletons; some samples contain vertical
#    noise pairs that deliberately satisfy the old vertical rule but fail the new
#    horizontal requirement. This makes the D6 change visible rather than collapsing back
#    to the original task.
# c) Hidden assumptions from the original main():
#    - Searching only vertical axes is no longer valid; we must search both axis families.
#    - Checking one mirror partner is insufficient; a 3-of-4 orbit would leave residual
#      asymmetric pixels after one-pass deletion.
#    - Scoring by vertical pair count alone can be fooled by vertically matched noise
#      pairs, so the search score is upgraded to "how many pixels belong to complete
#      fourfold orbits" for a candidate axis pair.
# d) Placement legality:
#    The original slot layout is preserved so sprites remain separated and in bounds.
#    Each color is still used for exactly one sprite, so per-color axis search remains
#    well-posed. Noise is chosen from background cells adjacent to the clean sprite and
#    from orbits disjoint from the clean orbits, which prevents accidental valid pixels.
# e) Visual observability:
#    A human can see that some extra pixels come in left/right pairs yet disappear in the
#    output because they lack top/bottom support. That is the central observable contrast
#    with the source task.
# f) Visual cleanliness:
#    Sprites stay in three separated regions, remain monochrome, and noise is kept close
#    to the owning sprite. No cross-object overlays are introduced.
# g) Color reasoning:
#    No color remapping is added. Colors remain identifiers for independent sprites, so
#    the new rule is inferred entirely from shape behavior.
# h) Quantified anti-degeneration target:
#    The generator rejects samples for which the legacy vertical-only cleaner produces the
#    same output as the new solver. At least one sprite therefore exhibits genuinely
#    stronger fourfold denoising behavior.
# i) New solver assumptions and generator guarantees:
#    - Assumption: each color corresponds to one sprite with one axis pair.
#      Guarantee: generator still uses three distinct non-background colors.
#    - Assumption: the true axis pair is identifiable by complete-orbit score.
#      Guarantee: clean sprites and noisy sprites must pass pair-gap checks.
#    - Assumption: deleting all incomplete orbits yields the target.
#      Guarantee: every injected noise fragment occupies only a strict subset of a clean
#      fourfold orbit, never a full orbit.


def _orbit_cells(r: int, c: int, ky: int, kx: int, height: int, width: int) -> set[tuple[int, int]] | None:
    rows = {int(r), int(ky) - int(r)}
    cols = {int(c), int(kx) - int(c)}
    orbit: set[tuple[int, int]] = set()
    for rr in rows:
        for cc in cols:
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            orbit.add((int(rr), int(cc)))
    return orbit


def _fourfold_score(pts: set[tuple[int, int]], ky: int, kx: int, height: int, width: int) -> int:
    score = 0
    for r, c in pts:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is not None and orbit <= pts:
            score += 1
    return score


def _best_fourfold_axes(pts: set[tuple[int, int]], height: int, width: int) -> tuple[int, int, int]:
    best_score = -1
    best_ky = 0
    best_kx = 0
    for ky in range(0, 2 * (height - 1) + 1):
        for kx in range(0, 2 * (width - 1) + 1):
            score = _fourfold_score(pts, ky, kx, height, width)
            if score > best_score:
                best_score = int(score)
                best_ky = int(ky)
                best_kx = int(kx)
    return best_ky, best_kx, best_score


def _pair_scores(coords: list[tuple[int, int]], height: int, width: int) -> dict[tuple[int, int], int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[tuple[int, int], int] = {}
    for ky in range(0, 2 * (height - 1) + 1):
        for kx in range(0, 2 * (width - 1) + 1):
            scores[(ky, kx)] = _fourfold_score(pts, ky, kx, height, width)
    return scores


def _second_best_pair_score(
    coords: list[tuple[int, int]],
    true_pair: tuple[int, int],
    height: int,
    width: int,
) -> int:
    scores = _pair_scores(coords, height, width)
    return max(score for pair, score in scores.items() if pair != true_pair)


def _legacy_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_kx = 0
        for kx in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                mc = int(kx) - int(c)
                if 0 <= mc < width and int(g[int(r), mc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_kx = int(kx)
        for r, c in pts:
            mc = int(best_kx) - int(c)
            if not (0 <= mc < width and int(g[int(r), mc]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing simultaneous horizontal and vertical mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors appear as separate sprites, each close to a fourfold-symmetric shape around its own axis pair.
    - Some same-color pixels are noise fragments that may satisfy one mirror direction but not the other.

    Rule:
    - For each non-bg color, search for the horizontal axis and vertical axis that best explain complete same-color mirror orbits.
    - Delete any pixel whose full orbit under that axis pair is incomplete, so only pixels supported in both directions remain.

    Key insight: The denoising unit is still each color independently, but the survivor test is stricter than the original task: a pixel must belong to a complete fourfold orbit, not merely have a left-right partner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue

        best_ky, best_kx, _best_score = _best_fourfold_axes(pts, height, width)
        for r, c in pts:
            orbit = _orbit_cells(r, c, best_ky, best_kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 80

# ── CFG: axis gap thresholds ──
CLEAN_PAIR_GAP_MIN = 4
FINAL_PAIR_GAP_MIN = 3

# ── CFG: retry limits ──
SPRITE_RETRIES = 240
GEN_RETRIES = 400

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    forced = {(0, 0)}
    if left_w > 1:
        forced.add((0, left_w - 1))
    if top_h > 1:
        forced.add((top_h - 1, 0))
    if top_h > 1 and left_w > 1 and random.random() < 0.7:
        forced.add((top_h - 1, left_w - 1))

    cells = set(forced)
    fill_prob = random.uniform(0.25, 0.8)
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
                pts.add((int(rr), int(cc)))
    return pts


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    for _ in range(SPRITE_RETRIES):
        h = random.randint(HEIGHT_MIN, min(HEIGHT_MAX, max_h))
        w = random.randint(WIDTH_MIN, min(WIDTH_MAX, max_w))
        top_h = (h + 1) // 2
        left_w = (w + 1) // 2
        canonical = _canonical_cells(top_h, left_w)
        pts = _mirror_canonical(canonical, h, w)

        count = len(pts)
        area = h * w
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue

        sprite = np.zeros((h, w), dtype=int)
        for r, c in pts:
            sprite[int(r), int(c)] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_pair = (h - 1, w - 1)
        if _second_best_pair_score(coords, true_pair, h, w) > count - CLEAN_PAIR_GAP_MIN:
            continue
        return sprite, h - 1, w - 1
    raise ValueError("failed to build clean fourfold sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
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
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_ky"] = int(spec["axis_ky_local"] + 2 * r0)
        spec["true_kx"] = int(spec["axis_kx_local"] + 2 * c0)
        spec["clean_count"] = int(np.count_nonzero(sprite))
        placed.append(spec)
    return out, placed


def _adjacent_mask(clean_mask: np.ndarray) -> np.ndarray:
    adj = np.zeros_like(clean_mask, dtype=bool)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    return adj & ~clean_mask


def _fragment_candidates(clean_mask: np.ndarray, true_ky: int, true_kx: int) -> dict[str, list[dict]]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask)
    clean_pts = {tuple(x) for x in np.argwhere(clean_mask)}

    vertical: dict[tuple[tuple[int, int], ...], dict] = {}
    horizontal: dict[tuple[tuple[int, int], ...], dict] = {}
    singletons: dict[tuple[tuple[int, int], ...], dict] = {}

    for r, c in np.argwhere(adj):
        r = int(r)
        c = int(c)
        orbit = _orbit_cells(r, c, true_ky, true_kx, height, width)
        if orbit is None or orbit & clean_pts:
            continue
        mr = int(true_ky) - r
        mc = int(true_kx) - c
        if mr == r or mc == c:
            continue
        if not adj[r, c]:
            continue

        vpair = tuple(sorted(((r, c), (r, mc))))
        if 0 <= mc < width and all(adj[rr, cc] for rr, cc in vpair):
            vertical[vpair] = {
                "kind": "vertical_pair",
                "cells": vpair,
                "orbit": frozenset(orbit),
            }

        hpair = tuple(sorted(((r, c), (mr, c))))
        if 0 <= mr < height and all(adj[rr, cc] for rr, cc in hpair):
            horizontal[hpair] = {
                "kind": "horizontal_pair",
                "cells": hpair,
                "orbit": frozenset(orbit),
            }

        key = ((r, c),)
        singletons[key] = {
            "kind": "singleton",
            "cells": key,
            "orbit": frozenset(orbit),
        }

    return {
        "vertical_pair": list(vertical.values()),
        "horizontal_pair": list(horizontal.values()),
        "singleton": list(singletons.values()),
    }


def _choose_noise_fragments(
    candidates: dict[str, list[dict]],
    lo: int,
    hi: int,
    require_vertical_pair: bool,
) -> list[dict] | None:
    minimum = max(lo, 2 if require_vertical_pair else lo)
    if minimum > hi:
        return None

    targets = list(range(minimum, hi + 1))
    random.shuffle(targets)
    all_candidates = (
        candidates["vertical_pair"] + candidates["horizontal_pair"] + candidates["singleton"]
    )

    for target in targets:
        for _ in range(NOISE_PLAN_RETRIES):
            chosen: list[dict] = []
            used_orbit_cells: set[tuple[int, int]] = set()
            remaining = int(target)

            if require_vertical_pair:
                starters = [cand for cand in candidates["vertical_pair"] if len(cand["cells"]) <= remaining]
                if not starters:
                    return None
                first = random.choice(starters)
                chosen.append(first)
                used_orbit_cells |= set(first["orbit"])
                remaining -= len(first["cells"])

            while remaining > 0:
                viable = [
                    cand
                    for cand in all_candidates
                    if len(cand["cells"]) <= remaining and set(cand["orbit"]).isdisjoint(used_orbit_cells)
                ]
                if not viable:
                    break

                exact = [cand for cand in viable if len(cand["cells"]) == remaining]
                if exact:
                    cand = random.choice(exact)
                elif remaining % 2 == 1:
                    ones = [cand for cand in viable if len(cand["cells"]) == 1]
                    if not ones:
                        break
                    cand = random.choice(ones)
                else:
                    twos = [cand for cand in viable if len(cand["cells"]) == 2]
                    cand = random.choice(twos if twos and random.random() < 0.7 else viable)

                chosen.append(cand)
                used_orbit_cells |= set(cand["orbit"])
                remaining -= len(cand["cells"])

            if remaining == 0:
                return chosen
    return None


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, *, require_vertical_pair: bool) -> bool:
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    clean_mask = out == color
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    candidates = _fragment_candidates(clean_mask, true_ky, true_kx)
    fragments = _choose_noise_fragments(candidates, lo, hi, require_vertical_pair)
    if not fragments:
        return False

    noise_cells = sorted({cell for fragment in fragments for cell in fragment["cells"]})
    current_coords = clean_coords + noise_cells
    if clean_count - _second_best_pair_score(current_coords, (true_ky, true_kx), GRID_SIZE, GRID_SIZE) < FINAL_PAIR_GAP_MIN:
        return False

    for r, c in noise_cells:
        inp[int(r), int(c)] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_ky_local, axis_kx_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append(
                {
                    "color": color,
                    "sprite": sprite,
                    "axis_ky_local": axis_ky_local,
                    "axis_kx_local": axis_kx_local,
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    eligible = [spec for spec in placed_specs if spec["clean_count"] >= 8]
    random.shuffle(eligible)
    required_spec = None
    for spec in eligible:
        candidates = _fragment_candidates(out == int(spec["color"]), spec["true_ky"], spec["true_kx"])
        if candidates["vertical_pair"]:
            required_spec = spec
            break
    if required_spec is None:
        return None

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, require_vertical_pair=(spec is required_spec)):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_vertical_clean(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
