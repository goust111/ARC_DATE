# Combo variant of 8e5c0c38 (index 27)
# combo_spec: D6/0+D15/2
# Name: 轴标记四重对称修复

from __future__ import annotations

import itertools
import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Each sprite is now judged against two marked symmetry axes, not a discovered single
#    vertical axis. The input contains one special-color pixel at the center of each
#    sprite: its row gives the horizontal mirror axis and its column gives the vertical
#    mirror axis. A same-color pixel survives only if its full orbit under both mirrors is
#    present.
#    Small trace:
#    - Original task: pick one best vertical axis per color, keep pixels with a left/right
#      partner.
#    - D6/0 effect: keep only complete fourfold orbits under one axis pair.
#    - D15/2 effect in this combo: the axis pair is supplied by a center marker, so the
#      solver must trust the marked row/column instead of re-searching.
#    - Observable difference: a left/right noise pair above the center row is deleted even
#      though it would survive the original rule; a compact 4-pixel cluster near the sprite
#      can also be deleted because it is symmetric around a nearby false center rather than
#      the marked one.
# b) Complexity calibration:
#    The generator keeps exactly three separated sprites, so humans can associate each
#    marker with one nearby object. To make the fusion visible, at least one sprite gets a
#    vertical-only noise pair and at least one sprite gets a false-center fourfold noise
#    orbit. This avoids collapsing back to either the original vertical task or the
#    marker-free D6 search variant.
# c) Hidden assumptions from the original main():
#    - "Each color independently reveals its own axis by score search" no longer holds:
#      the solver must first identify the dedicated marker color, then attach each marker to
#      a nearby sprite color.
#    - "A single mirrored partner is enough" no longer holds: complete horizontal and
#      vertical support is required.
#    - "All non-bg colors are sprite colors" no longer holds: one non-bg color is metadata
#      and must be removed in the output.
# d) Placement legality:
#    The original 22x22 three-slot layout is preserved. Sprites use odd height and odd
#    width so the marked center lies on a real cell. The clean sprite keeps that center
#    empty, preventing overlap between marker pixels and sprite pixels.
# e) Visual observability:
#    In examples, a human can spot the single marker inside each sprite and infer that the
#    center row and center column are privileged. Noise close to the sprite disappears when
#    it fails to mirror across both of those marked axes.
# f) Visual cleanliness:
#    Markers are only single pixels, sprites stay monochrome, and the slot gaps from the
#    source task keep different objects visually separate. The marker color is used nowhere
#    else, so it reads as an annotation rather than another object.
# g) Color reasoning:
#    Sprite colors are still identity-only. The only extra color semantics are that the
#    least-populated non-background color is the marker color and must be stripped. No
#    hidden ordering by numeric color id is used.
# h) Quantified anti-degeneration target:
#    Generation rejects any sample where the old vertical cleaner matches the target, so
#    the D6 fourfold requirement is always active. For the D15 marker dimension, validation
#    is done statistically after generation: we measure how often a marker-free fourfold
#    search differs from the marked-center solver and how often the marked center is not the
#    unique best-scoring axis pair. This keeps generation fast while still checking that the
#    explicit marker instruction is present and sometimes decision-relevant.
# i) New solver assumptions and generator guarantees:
#    - Assumption: one unique marker pixel exists per sprite and marker color is distinct.
#      Guarantee: generator uses one special color exactly three times.
#    - Assumption: each marker belongs to exactly one nearby sprite color.
#      Guarantee: sprites occupy three well-separated slots with one center marker each.
#    - Assumption: the marked center cell is not part of the clean sprite.
#      Guarantee: clean sprite synthesis leaves the center cell as background.
#    - Assumption: deleting incomplete true-center orbits yields the output.
#      Guarantee: injected noise never completes a full orbit around the marked axes.


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


def _find_marker_color(g: np.ndarray, bg: int) -> int:
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain non-background colors"
    return min(colors, key=lambda col: int(np.count_nonzero(g == int(col))))


def _assign_markers_to_colors(
    markers: list[tuple[int, int]],
    color_points: dict[int, np.ndarray],
) -> list[tuple[tuple[int, int], int]]:
    colors = list(color_points)
    assert len(markers) == len(colors), "expected one marker per sprite color"

    best_score = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(colors):
        score = 0
        for (mr, mc), col in zip(markers, perm):
            pts = color_points[col]
            dists = np.abs(pts[:, 0] - int(mr)) + np.abs(pts[:, 1] - int(mc))
            score += int(dists.min())
        if best_score is None or score < best_score:
            best_score = int(score)
            best_perm = tuple(int(x) for x in perm)

    assert best_perm is not None
    return list(zip(markers, best_perm))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each sprite by using a marked center to enforce simultaneous horizontal and vertical symmetry.

    Input structure:
    - bg = background. Three non-bg sprite colors appear in separate regions, each close to a fourfold-symmetric shape.
    - A fourth non-bg color appears only as one-pixel center markers; each marker sits at the crossing of one sprite's intended horizontal and vertical axes.

    Rule:
    - Identify the marker color, pair each marker with its nearby sprite color, and treat the marker's row and column as that sprite's symmetry axes.
    - Remove every sprite pixel whose full orbit under those two axes is incomplete, then erase the marker pixels themselves.

    Key insight: The solver no longer needs to infer axes from the noisy shape. The center marker explicitly supplies the correct axis pair, and denoising succeeds only when pixels are supported in both mirror directions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    marker_color = int(_find_marker_color(g, bg))
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == marker_color)]
    sprite_colors = colors_present(g, exclude={bg, marker_color})
    assert markers and sprite_colors, "expected sprites and marker pixels"

    color_points = {
        int(col): np.argwhere(g == int(col)).astype(int)
        for col in sprite_colors
    }
    assignments = _assign_markers_to_colors(markers, color_points)

    out = g.copy()
    out[out == marker_color] = int(bg)

    for (mr, mc), col in assignments:
        pts = {(int(r), int(c)) for r, c in color_points[int(col)]}
        ky = 2 * int(mr)
        kx = 2 * int(mc)
        for r, c in pts:
            orbit = _orbit_cells(r, c, ky, kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 3
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 11
DENSITY_LO = 0.28
DENSITY_HI = 0.86
SPRITE_RETRIES = 240

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 120
FALSE_CENTER_DELTAS = [
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
]

# ── CFG: retry limits ──
GEN_RETRIES = 700

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _adjacent_mask(mask: np.ndarray) -> np.ndarray:
    adj = np.zeros_like(mask, dtype=bool)
    adj[1:, :] |= mask[:-1, :]
    adj[:-1, :] |= mask[1:, :]
    adj[:, 1:] |= mask[:, :-1]
    adj[:, :-1] |= mask[:, 1:]
    return adj & ~mask


def _dilate4(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(steps)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    center = (top_h - 1, left_w - 1)
    all_cells = [
        (r, c)
        for r in range(top_h)
        for c in range(left_w)
        if (r, c) != center
    ]
    off_axis = [(r, c) for r, c in all_cells if r < top_h - 1 and c < left_w - 1]

    forced = {(0, 0)}
    if top_h > 1:
        forced.add((top_h - 1, 0))
    if left_w > 1:
        forced.add((0, left_w - 1))
    if off_axis:
        forced.add(random.choice(off_axis))

    cells = set(forced)
    fill_prob = random.uniform(0.2, 0.75)
    for cell in all_cells:
        if cell in forced:
            continue
        if random.random() < fill_prob:
            cells.add(cell)

    if len(cells) == len(forced):
        extras = [cell for cell in all_cells if cell not in cells]
        if extras:
            cells.add(random.choice(extras))
    return cells


def _mirror_canonical(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    center_r = h // 2
    center_c = w // 2
    pts: set[tuple[int, int]] = set()
    for r, c in cells:
        rows = {int(r), int(h - 1 - r)}
        cols = {int(c), int(w - 1 - c)}
        for rr in rows:
            for cc in cols:
                if rr == center_r and cc == center_c:
                    continue
                pts.add((int(rr), int(cc)))
    return pts


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    height_choices = [h for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1) if h % 2 == 1]
    width_choices = [w for w in range(WIDTH_MIN, min(WIDTH_MAX, max_w) + 1) if w % 2 == 1]
    if not height_choices or not width_choices:
        raise ValueError("no feasible odd sprite size")

    for _ in range(SPRITE_RETRIES):
        h = random.choice(height_choices)
        w = random.choice(width_choices)
        canonical = _canonical_cells((h + 1) // 2, (w + 1) // 2)
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
        sprite[h // 2, w // 2] = 0
        return sprite, h - 1, w - 1
    raise ValueError("failed to build clean marked sprite")


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
        spec["center"] = (int(r0 + h // 2), int(c0 + w // 2))
        spec["clean_count"] = int(np.count_nonzero(sprite))
        placed.append(spec)
    return out, placed


def _true_fragment_candidates(
    clean_mask: np.ndarray,
    true_ky: int,
    true_kx: int,
    blocked: set[tuple[int, int]],
) -> dict[str, list[dict]]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask)
    clean_pts = {tuple(map(int, rc)) for rc in np.argwhere(clean_mask)}

    vertical: dict[tuple[tuple[int, int], ...], dict] = {}
    horizontal: dict[tuple[tuple[int, int], ...], dict] = {}
    singletons: dict[tuple[tuple[int, int], ...], dict] = {}

    for r, c in np.argwhere(adj):
        r = int(r)
        c = int(c)
        if (r, c) in blocked:
            continue
        orbit = _orbit_cells(r, c, true_ky, true_kx, height, width)
        if orbit is None or orbit & clean_pts or orbit & blocked:
            continue

        singletons[((r, c),)] = {
            "kind": "singleton",
            "cells": ((r, c),),
            "guard": frozenset(orbit),
        }

        mr = int(true_ky) - r
        mc = int(true_kx) - c
        if len(orbit) != 4:
            continue

        vpair = tuple(sorted(((r, c), (r, mc))))
        if all(adj[rr, cc] and (rr, cc) not in blocked for rr, cc in vpair):
            vertical[vpair] = {
                "kind": "vertical_pair",
                "cells": vpair,
                "guard": frozenset(orbit),
            }

        hpair = tuple(sorted(((r, c), (mr, c))))
        if all(adj[rr, cc] and (rr, cc) not in blocked for rr, cc in hpair):
            horizontal[hpair] = {
                "kind": "horizontal_pair",
                "cells": hpair,
                "guard": frozenset(orbit),
            }

    return {
        "vertical_pair": list(vertical.values()),
        "horizontal_pair": list(horizontal.values()),
        "singleton": list(singletons.values()),
    }


def _false_orbit_candidates(
    clean_mask: np.ndarray,
    true_ky: int,
    true_kx: int,
    blocked: set[tuple[int, int]],
) -> list[dict]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask)
    near = _dilate4(clean_mask, 2) & ~clean_mask
    clean_pts = {tuple(map(int, rc)) for rc in np.argwhere(clean_mask)}
    candidates: dict[tuple[tuple[int, int], ...], dict] = {}

    for dky, dkx in FALSE_CENTER_DELTAS:
        false_ky = int(true_ky + dky)
        false_kx = int(true_kx + dkx)
        for r, c in np.argwhere(adj):
            r = int(r)
            c = int(c)
            orbit = _orbit_cells(r, c, false_ky, false_kx, height, width)
            if orbit is None or len(orbit) != 4:
                continue
            if orbit & clean_pts or orbit & blocked:
                continue
            if not all(near[rr, cc] for rr, cc in orbit):
                continue
            if not any(adj[rr, cc] for rr, cc in orbit):
                continue

            true_guard: set[tuple[int, int]] = set()
            valid = True
            for rr, cc in orbit:
                torbit = _orbit_cells(rr, cc, true_ky, true_kx, height, width)
                if torbit is None or torbit & clean_pts or torbit & blocked:
                    valid = False
                    break
                if torbit <= orbit:
                    valid = False
                    break
                true_guard |= torbit
            if not valid:
                continue

            key = tuple(sorted(orbit))
            candidates[key] = {
                "kind": "false_orbit",
                "cells": key,
                "guard": frozenset(true_guard),
                "false_pair": (false_ky, false_kx),
            }

    return list(candidates.values())


def _choose_noise_fragments(
    candidates: dict[str, list[dict]],
    lo: int,
    hi: int,
    required: list[dict],
) -> list[dict] | None:
    base_min = sum(len(cand["cells"]) for cand in required)
    if base_min > hi:
        return None

    extras_master = (
        candidates["horizontal_pair"]
        + candidates["singleton"]
        + candidates["vertical_pair"]
        + candidates["false_orbit"]
    )

    for _ in range(NOISE_PLAN_RETRIES):
        chosen: list[dict] = []
        used_cells: set[tuple[int, int]] = set()
        used_guard: set[tuple[int, int]] = set()
        total = 0
        ok = True

        for cand in required:
            if (
                set(cand["cells"]).isdisjoint(used_cells)
                and set(cand["guard"]).isdisjoint(used_guard)
                and total + len(cand["cells"]) <= hi
            ):
                chosen.append(cand)
                used_cells |= set(cand["cells"])
                used_guard |= set(cand["guard"])
                total += len(cand["cells"])
            else:
                ok = False
                break
        if not ok:
            continue

        extras = list(extras_master)
        random.shuffle(extras)
        while total < lo:
            viable = [
                cand
                for cand in extras
                if set(cand["cells"]).isdisjoint(used_cells)
                and set(cand["guard"]).isdisjoint(used_guard)
                and total + len(cand["cells"]) <= hi
            ]
            if not viable:
                ok = False
                break

            deficit = lo - total
            exact = [cand for cand in viable if len(cand["cells"]) == deficit]
            if exact:
                cand = random.choice(exact)
            else:
                fit = [cand for cand in viable if len(cand["cells"]) <= deficit]
                if fit:
                    fit.sort(key=lambda cand: (abs(deficit - len(cand["cells"])), -len(cand["cells"])))
                    best = fit[0]
                    pool = [cand for cand in fit if len(cand["cells"]) == len(best["cells"])]
                    cand = random.choice(pool)
                else:
                    singles = [cand for cand in viable if len(cand["cells"]) == 1]
                    if not singles:
                        ok = False
                        break
                    cand = random.choice(singles)

            chosen.append(cand)
            used_cells |= set(cand["cells"])
            used_guard |= set(cand["guard"])
            total += len(cand["cells"])

        if not ok:
            continue

        while total < hi:
            viable = [
                cand
                for cand in extras
                if set(cand["cells"]).isdisjoint(used_cells)
                and set(cand["guard"]).isdisjoint(used_guard)
                and total + len(cand["cells"]) <= hi
            ]
            if not viable or random.random() < 0.55:
                break
            exact = [cand for cand in viable if total + len(cand["cells"]) == hi]
            cand = random.choice(exact if exact else viable)
            chosen.append(cand)
            used_cells |= set(cand["cells"])
            used_guard |= set(cand["guard"])
            total += len(cand["cells"])

        if lo <= total <= hi:
            return chosen
    return None


def _add_noise(
    inp: np.ndarray,
    out: np.ndarray,
    spec: dict,
    *,
    required_vertical_pair: dict | None,
    required_false_orbit: dict | None,
) -> bool:
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    center = tuple(spec["center"])
    clean_mask = out == color
    clean_count = int(np.count_nonzero(clean_mask))

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    blocked = {center}
    candidates = _true_fragment_candidates(clean_mask, true_ky, true_kx, blocked)
    candidates["false_orbit"] = _false_orbit_candidates(clean_mask, true_ky, true_kx, blocked)
    required = []
    if required_false_orbit is not None:
        required.append(required_false_orbit)
    if required_vertical_pair is not None:
        required.append(required_vertical_pair)
    fragments = _choose_noise_fragments(candidates, lo, hi, required)
    if not fragments:
        return False

    for frag in fragments:
        for r, c in frag["cells"]:
            inp[int(r), int(c)] = int(color)
    return True


def _legacy_vertical_clean(grid: np.ndarray, marker_color: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    out[out == marker_color] = int(bg)
    work = out.copy()
    _height, width = work.shape

    for col in colors_present(work, exclude={bg}):
        pts = np.argwhere(work == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_kx = 0
        for kx in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                mc = int(kx) - int(c)
                if 0 <= mc < width and int(work[int(r), mc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_kx = int(kx)
        for r, c in pts:
            mc = int(best_kx) - int(c)
            if not (0 <= mc < width and int(work[int(r), mc]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def _search_fourfold_clean(grid: np.ndarray, marker_color: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    out[out == marker_color] = int(bg)
    work = out.copy()
    height, width = work.shape

    for col in colors_present(work, exclude={bg}):
        pts = {(int(r), int(c)) for r, c in np.argwhere(work == int(col))}
        if not pts:
            continue
        best_ky, best_kx, _score = _best_fourfold_axes(pts, height, width)
        for r, c in pts:
            orbit = _orbit_cells(r, c, best_ky, best_kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    colors = random.sample(range(10), NUM_SPRITES + 2)
    bg = int(colors[0])
    marker_color = int(colors[1])
    obj_colors = [int(c) for c in colors[2:]]
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

    d6_eligible: list[tuple[dict, dict]] = []
    d15_eligible: list[tuple[dict, dict]] = []
    for spec in placed_specs:
        clean_mask = out == int(spec["color"])
        blocked = {tuple(spec["center"])}
        true_cands = _true_fragment_candidates(clean_mask, spec["true_ky"], spec["true_kx"], blocked)
        if true_cands["vertical_pair"]:
            d6_eligible.append((spec, random.choice(true_cands["vertical_pair"])))
        false_cands = _false_orbit_candidates(clean_mask, spec["true_ky"], spec["true_kx"], blocked)
        if false_cands:
            d15_eligible.append((spec, random.choice(false_cands)))
    if not d6_eligible or not d15_eligible:
        return None

    d6_spec, d6_frag = random.choice(d6_eligible)
    distinct_d15 = [(spec, frag) for spec, frag in d15_eligible if spec is not d6_spec]
    d15_pool = distinct_d15 if distinct_d15 else d15_eligible
    d15_spec, d15_frag = min(d15_pool, key=lambda item: int(item[0]["clean_count"]))

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(
            inp,
            out,
            spec,
            required_vertical_pair=(d6_frag if spec is d6_spec else None),
            required_false_orbit=(d15_frag if spec is d15_spec else None),
        ):
            return None

    for spec in placed_specs:
        rr, cc = spec["center"]
        inp[int(rr), int(cc)] = marker_color

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_vertical_clean(inp, marker_color), out):
        return None

    metrics = {
        "marker_count": int(np.count_nonzero(inp == marker_color)),
        "removed_pixels": int(np.count_nonzero((inp != out) & (inp != marker_color))),
    }
    return inp, out, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _out, _metrics = result
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
