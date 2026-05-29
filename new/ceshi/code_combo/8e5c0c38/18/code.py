# Combo variant of 8e5c0c38 (index 18)
# combo_spec: D6/0+D11/2
# Name: 两阶段四重对称修复

from __future__ import annotations

import itertools
import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Every non-background color is still one isolated sprite, but the clean target is now
#    fourfold symmetric: each kept pixel must be mirrored across both a vertical axis kx
#    and a horizontal axis ky. The intended processing is explicitly two-stage:
#    1. Find all sprites' ky/kx values on the original noisy grid.
#    2. Using those frozen axes, remove every pixel whose full orbit is incomplete.
#    Small trace:
#    - Clean orbit around (ky=4, kx=4): {(1,1), (1,3), (3,1), (3,3)}.
#    - Add vertical-only noise {(0,1), (0,3)} and horizontal-only noise {(1,0), (3,0)}.
#    - Original task keeps {(0,1), (0,3)} because they have vertical mates.
#    - This combo first locks both axes, then deletes all four noise cells because neither
#      pair completes the 4-cell orbit required by both symmetries.
#    - A gradual repair that deletes vertical violations before searching ky loses the
#      horizontal-only evidence first; that is the D11/2 failure mode the generator tests.
# b) Complexity calibration:
#    Sprites are built from irregular quadrant seeds and mirrored into true fourfold
#    shapes, so the target is visibly different from the source task. Noise is drawn from
#    incomplete orbits close to the sprite and includes vertical pairs, horizontal pairs,
#    and occasional singletons. At least one sample sprite carries both pair types so the
#    stricter rule is visually observable.
# c) Hidden assumptions audit:
#    - The original main() assumed one vertical axis per color; that no longer holds, so
#      ky and kx are both searched.
#    - The original deletion test checked only one mirrored partner; now a survivor must
#      belong to a complete orbit under both axes.
#    - The original per-color "search then immediately delete" flow risks coupling later
#      searches to earlier deletions. The new main() stores every color's axes first and
#      only then edits the output grid.
# d) Placement legality:
#    The original slot layout is preserved. Clean sprites are placed fully inside their
#    slots, colors remain distinct, and noise candidates must be adjacent to the owning
#    sprite, in bounds, and on background cells disjoint from all clean orbits.
# e) Visual observability:
#    Human-solvable evidence comes from paired noise that looks symmetric in only one
#    direction. Across a few examples, the input shows near-cross-symmetric blobs while
#    the output removes side-pairs above the center and top-bottom pairs left of center.
# f) Visual cleanliness:
#    Objects remain separated in three regions and stay monochrome. Noise never overlaps
#    another sprite and is kept adjacent to the owning shape, so the correspondence stays
#    legible.
# g) Color reasoning:
#    No recoloring is introduced. Color still serves only to separate independent sprites,
#    which keeps the inferred rule about geometry rather than palette values.
# h) Quantified anti-degeneration target:
#    The generator rejects a sample if the legacy vertical-only cleaner matches the clean
#    output, so the D6 fourfold change is always active. It also forces at least one
#    sprite to carry both a vertical-only noise pair and a horizontal-only noise pair;
#    this mixed noise is the observable witness for the D11/2 two-stage reading, and we
#    measure gradual-baseline disagreement separately during validation rather than making
#    every sample pay that acceptance cost.
# i) Solver assumptions and generator guarantees:
#    - Assumption: each non-bg color corresponds to exactly one sprite with one axis pair.
#      Guarantee: generator still uses exactly three distinct object colors in disjoint
#      slots.
#    - Assumption: the true ky and kx are identifiable from the noisy point set by their
#      own marginal mirror scores.
#      Guarantee: clean and noisy samples must pass separate vertical and horizontal gap
#      checks before acceptance.
#    - Assumption: deleting all pixels outside complete fourfold orbits yields the target.
#      Guarantee: injected noise occupies only strict subsets of an otherwise empty orbit.


def _orbit_cells(
    r: int,
    c: int,
    ky: int,
    kx: int,
    height: int,
    width: int,
) -> set[tuple[int, int]] | None:
    rows = {int(r), int(ky) - int(r)}
    cols = {int(c), int(kx) - int(c)}
    orbit: set[tuple[int, int]] = set()
    for rr in rows:
        for cc in cols:
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            orbit.add((int(rr), int(cc)))
    return orbit


def _vertical_score(pts: set[tuple[int, int]], kx: int, width: int) -> int:
    score = 0
    for r, c in pts:
        mc = int(kx) - int(c)
        if 0 <= mc < width and (int(r), int(mc)) in pts:
            score += 1
    return score


def _horizontal_score(pts: set[tuple[int, int]], ky: int, height: int) -> int:
    score = 0
    for r, c in pts:
        mr = int(ky) - int(r)
        if 0 <= mr < height and (int(mr), int(c)) in pts:
            score += 1
    return score


def _best_vertical_axis(pts: set[tuple[int, int]], width: int) -> tuple[int, int]:
    best_score = -1
    best_kx = 0
    for kx in range(0, 2 * (width - 1) + 1):
        score = _vertical_score(pts, kx, width)
        if score > best_score:
            best_score = int(score)
            best_kx = int(kx)
    return best_kx, best_score


def _best_horizontal_axis(pts: set[tuple[int, int]], height: int) -> tuple[int, int]:
    best_score = -1
    best_ky = 0
    for ky in range(0, 2 * (height - 1) + 1):
        score = _horizontal_score(pts, ky, height)
        if score > best_score:
            best_score = int(score)
            best_ky = int(ky)
    return best_ky, best_score


def _second_best_vertical_score(coords: list[tuple[int, int]], true_kx: int, width: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    scores = [_vertical_score(pts, kx, width) for kx in range(0, 2 * (width - 1) + 1) if kx != true_kx]
    return max(scores)


def _second_best_horizontal_score(coords: list[tuple[int, int]], true_ky: int, height: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    scores = [_horizontal_score(pts, ky, height) for ky in range(0, 2 * (height - 1) + 1) if ky != true_ky]
    return max(scores)


def _complete_orbit_score(
    pts: set[tuple[int, int]],
    ky: int,
    kx: int,
    height: int,
    width: int,
) -> int:
    score = 0
    for r, c in pts:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is not None and orbit <= pts:
            score += 1
    return score


def _best_axes_two_stage(pts: set[tuple[int, int]], height: int, width: int) -> tuple[int, int]:
    best_ky, _ = _best_horizontal_axis(pts, height)
    best_kx, _ = _best_vertical_axis(pts, width)
    return best_ky, best_kx


def _legacy_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue
        best_kx, _ = _best_vertical_axis(pts, width)
        for r, c in pts:
            mc = int(best_kx) - int(c)
            if not (0 <= mc < width and (int(r), int(mc)) in pts):
                out[int(r), int(c)] = int(bg)
    return out


def _sequential_axis_clean(grid: np.ndarray, order: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue

        current = set(pts)
        if order == "vh":
            best_kx, _ = _best_vertical_axis(current, width)
            current = {(r, c) for r, c in current if (r, int(best_kx) - c) in current}
            if current:
                best_ky, _ = _best_horizontal_axis(current, height)
                current = {(r, c) for r, c in current if (int(best_ky) - r, c) in current}
        elif order == "hv":
            best_ky, _ = _best_horizontal_axis(current, height)
            current = {(r, c) for r, c in current if (int(best_ky) - r, c) in current}
            if current:
                best_kx, _ = _best_vertical_axis(current, width)
                current = {(r, c) for r, c in current if (r, int(best_kx) - c) in current}
        else:
            raise ValueError(f"unknown order: {order}")

        for r, c in pts:
            if (int(r), int(c)) not in current:
                out[int(r), int(c)] = int(bg)

    return out


def _progressive_research_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        current = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not current:
            continue

        while True:
            ky, kx = _best_axes_two_stage(current, height, width)
            nxt = set()
            for r, c in current:
                orbit = _orbit_cells(r, c, ky, kx, height, width)
                if orbit is not None and orbit <= current:
                    nxt.add((r, c))
            if nxt == current:
                break
            current = nxt

        original = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        for r, c in original:
            if (r, c) not in current:
                out[r, c] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing fourfold mirror symmetry with a frozen two-stage axis search.

    Input structure:
    - bg = background. Several non-bg colors appear as separate sprites, each close to a shape that should mirror across both one vertical axis and one horizontal axis.
    - Some same-color pixels are noise fragments that may look valid in only one mirror direction or may bias axis search if cleaned too early.

    Rule:
    - First, for every non-bg color, locate the best horizontal axis and best vertical axis on the original noisy pixel set and store them.
    - Then, using those stored axes without any re-search, delete each pixel whose full orbit under the axis pair is incomplete.

    Key insight: The task is still independent per color, but the correct repair is not a gradual left-right then top-bottom cleanup. The axes must be inferred before any deletions, and a pixel survives only if it is supported by both mirror directions at once.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    point_sets: dict[int, set[tuple[int, int]]] = {}
    axes_by_color: dict[int, tuple[int, int]] = {}

    for col in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue
        point_sets[int(col)] = pts
        axes_by_color[int(col)] = _best_axes_two_stage(pts, height, width)

    for col in colors:
        pts = point_sets.get(int(col))
        if not pts:
            continue
        ky, kx = axes_by_color[int(col)]
        for r, c in pts:
            orbit = _orbit_cells(r, c, ky, kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

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
WIDTH_MIN = 3
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 120
NOISE_LAYOUT_RETRIES = 120

# ── CFG: axis gap thresholds ──
CLEAN_VERTICAL_GAP_MIN = 2
CLEAN_HORIZONTAL_GAP_MIN = 2
FINAL_VERTICAL_GAP_MIN = 1
FINAL_HORIZONTAL_GAP_MIN = 1

# ── CFG: retry limits ──
SPRITE_RETRIES = 260
GEN_RETRIES = 500

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
    fill_prob = random.uniform(0.2, 0.78)
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
        true_ky = h - 1
        true_kx = w - 1
        if _second_best_horizontal_score(coords, true_ky, h) > count - CLEAN_HORIZONTAL_GAP_MIN:
            continue
        if _second_best_vertical_score(coords, true_kx, w) > count - CLEAN_VERTICAL_GAP_MIN:
            continue
        return sprite, true_ky, true_kx
    raise ValueError("failed to build clean fourfold sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for idx, (spec, (r0s, c0s, r1s, c1s)) in enumerate(zip(sprite_specs, slots)):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["index"] = int(idx)
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


def _fragment_candidates(canvas: np.ndarray, bg: int, color: int, true_ky: int, true_kx: int) -> dict[str, list[dict]]:
    height, width = canvas.shape
    clean_mask = canvas == int(color)
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
        if any(int(canvas[rr, cc]) != int(bg) for rr, cc in orbit):
            continue

        mr = int(true_ky) - r
        mc = int(true_kx) - c
        if mr == r or mc == c:
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

        singletons[((r, c),)] = {
            "kind": "singleton",
            "cells": ((r, c),),
            "orbit": frozenset(orbit),
        }

    return {
        "vertical_pair": list(vertical.values()),
        "horizontal_pair": list(horizontal.values()),
        "singleton": list(singletons.values()),
    }


def _pick_required_fragments(
    candidates: dict[str, list[dict]],
    min_vertical_pairs: int,
    min_horizontal_pairs: int,
) -> list[dict] | None:
    if min_vertical_pairs == 0 and min_horizontal_pairs == 0:
        return []

    vfrags = candidates["vertical_pair"]
    hfrags = candidates["horizontal_pair"]
    if len(vfrags) < min_vertical_pairs or len(hfrags) < min_horizontal_pairs:
        return None

    options: list[list[dict]] = []
    for vchoice in itertools.combinations(vfrags, min_vertical_pairs):
        used = set()
        ok = True
        for frag in vchoice:
            orbit = set(frag["orbit"])
            if orbit & used:
                ok = False
                break
            used |= orbit
        if not ok:
            continue
        for hchoice in itertools.combinations(hfrags, min_horizontal_pairs):
            used2 = set(used)
            ok2 = True
            for frag in hchoice:
                orbit = set(frag["orbit"])
                if orbit & used2:
                    ok2 = False
                    break
                used2 |= orbit
            if ok2:
                options.append(list(vchoice) + list(hchoice))
    if not options:
        return None
    return random.choice(options)


def _choose_noise_fragments(
    candidates: dict[str, list[dict]],
    lo: int,
    hi: int,
    *,
    min_vertical_pairs: int,
    min_horizontal_pairs: int,
) -> list[dict] | None:
    minimum = max(lo, 2 * min_vertical_pairs + 2 * min_horizontal_pairs)
    if minimum > hi:
        return None

    targets = list(range(minimum, hi + 1))
    random.shuffle(targets)
    all_candidates = candidates["vertical_pair"] + candidates["horizontal_pair"] + candidates["singleton"]

    for target in targets:
        for _ in range(NOISE_PLAN_RETRIES):
            chosen = _pick_required_fragments(candidates, min_vertical_pairs, min_horizontal_pairs)
            if chosen is None:
                return None

            used_orbit_cells: set[tuple[int, int]] = set()
            remaining = int(target)
            for frag in chosen:
                used_orbit_cells |= set(frag["orbit"])
                remaining -= len(frag["cells"])
            if remaining < 0:
                continue

            if min_vertical_pairs > 0 and min_horizontal_pairs > 0 and remaining > 0:
                mixed_bonus = []
                for bucket in ("vertical_pair", "horizontal_pair", "singleton"):
                    for frag in candidates[bucket]:
                        if set(frag["orbit"]).isdisjoint(used_orbit_cells):
                            mixed_bonus.append(frag)
                random.shuffle(mixed_bonus)
                for frag in mixed_bonus:
                    if len(frag["cells"]) <= remaining:
                        chosen.append(frag)
                        used_orbit_cells |= set(frag["orbit"])
                        remaining -= len(frag["cells"])
                        break

            while remaining > 0:
                viable = [
                    frag
                    for frag in all_candidates
                    if len(frag["cells"]) <= remaining and set(frag["orbit"]).isdisjoint(used_orbit_cells)
                ]
                if not viable:
                    break

                exact = [frag for frag in viable if len(frag["cells"]) == remaining]
                if exact:
                    frag = random.choice(exact)
                elif remaining % 2 == 1:
                    singles = [frag for frag in viable if len(frag["cells"]) == 1]
                    if not singles:
                        break
                    frag = random.choice(singles)
                else:
                    pairs = [frag for frag in viable if len(frag["cells"]) == 2]
                    frag = random.choice(pairs if pairs and random.random() < 0.75 else viable)

                chosen.append(frag)
                used_orbit_cells |= set(frag["orbit"])
                remaining -= len(frag["cells"])

            if remaining == 0:
                return chosen
    return None


def _plan_noise(
    canvas: np.ndarray,
    bg: int,
    spec: dict,
    *,
    min_vertical_pairs: int,
    min_horizontal_pairs: int,
) -> list[tuple[int, int]] | None:
    color = int(spec["color"])
    clean_count = int(spec["clean_count"])
    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return None

    candidates = _fragment_candidates(canvas, bg, color, int(spec["true_ky"]), int(spec["true_kx"]))
    fragments = _choose_noise_fragments(
        candidates,
        lo,
        hi,
        min_vertical_pairs=min_vertical_pairs,
        min_horizontal_pairs=min_horizontal_pairs,
    )
    if not fragments:
        return None
    return sorted({(int(r), int(c)) for frag in fragments for r, c in frag["cells"]})


def _final_axis_gaps(coords: list[tuple[int, int]], true_ky: int, true_kx: int) -> tuple[int, int]:
    true_v = _vertical_score({(int(r), int(c)) for r, c in coords}, true_kx, GRID_SIZE)
    true_h = _horizontal_score({(int(r), int(c)) for r, c in coords}, true_ky, GRID_SIZE)
    return (
        int(true_v - _second_best_vertical_score(coords, true_kx, GRID_SIZE)),
        int(true_h - _second_best_horizontal_score(coords, true_ky, GRID_SIZE)),
    )


def _passes_noisy_axis_gap(spec: dict, noise_cells: list[tuple[int, int]]) -> bool:
    clean_mask = spec["canvas"] == int(spec["color"])
    coords = [tuple(x) for x in np.argwhere(clean_mask)]
    coords.extend((int(r), int(c)) for r, c in noise_cells)
    vgap, hgap = _final_axis_gaps(coords, int(spec["true_ky"]), int(spec["true_kx"]))
    return vgap >= FINAL_VERTICAL_GAP_MIN and hgap >= FINAL_HORIZONTAL_GAP_MIN


def _build_candidate_pair() -> tuple[np.ndarray, np.ndarray] | None:
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

    for spec in placed_specs:
        spec["canvas"] = out

    eligible = []
    for spec in placed_specs:
        candidates = _fragment_candidates(out, bg, int(spec["color"]), int(spec["true_ky"]), int(spec["true_kx"]))
        if candidates["vertical_pair"] and candidates["horizontal_pair"] and spec["clean_count"] >= 10:
            eligible.append(spec)
    if not eligible:
        return None
    required_spec = random.choice(eligible)

    for _ in range(NOISE_LAYOUT_RETRIES):
        noise_by_color: dict[int, list[tuple[int, int]]] = {}
        ok = True
        for spec in placed_specs:
            noise_cells = _plan_noise(
                out,
                bg,
                spec,
                min_vertical_pairs=int(spec is required_spec),
                min_horizontal_pairs=int(spec is required_spec),
            )
            if noise_cells is None or not _passes_noisy_axis_gap(spec, noise_cells):
                ok = False
                break
            noise_by_color[int(spec["color"])] = noise_cells
        if not ok:
            continue

        inp = out.copy()
        for spec in placed_specs:
            color = int(spec["color"])
            for r, c in noise_by_color[color]:
                inp[int(r), int(c)] = color

        flip_ud = random.random() < FLIP_PROB
        flip_lr = random.random() < FLIP_PROB
        if flip_ud:
            inp = np.flipud(inp)
            out_eval = np.flipud(out)
        else:
            out_eval = out.copy()
        if flip_lr:
            inp = np.fliplr(inp)
            out_eval = np.fliplr(out_eval)
        return inp, out_eval

    return None


def _validate_candidate(inp: np.ndarray, out: np.ndarray) -> str | None:
    if not np.array_equal(main(inp), out):
        return "solver_mismatch"
    if np.array_equal(_legacy_vertical_clean(inp), out):
        return "legacy_match"
    return None


def _gen_attempt() -> np.ndarray | None:
    candidate = _build_candidate_pair()
    if candidate is None:
        return None
    inp, out = candidate
    if _validate_candidate(inp, out) is not None:
        return None
    return inp


def _gen_attempt_debug() -> tuple[np.ndarray | None, str]:
    candidate = _build_candidate_pair()
    if candidate is None:
        return None, "construction_failed"
    inp, out = candidate
    reason = _validate_candidate(inp, out)
    if reason is not None:
        return None, reason
    return inp, "ok"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
