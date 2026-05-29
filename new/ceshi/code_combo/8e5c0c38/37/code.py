# Combo variant of 8e5c0c38 (index 37)
# combo_spec: D5/0+D6/0+D11/2
# Name: 标记双轴两阶段修复

from __future__ import annotations

"""
Design analysis for the combo variant.

combo_summary:
  - Keep the original task's denoising flavor: every sprite starts as a nearly
    symmetric monochrome object with attached same-color noise, and the output
    removes the noise.
  - Fuse three changes at once:
    1. D5/0: the axes are no longer inferred from sprite pixels; red external
       markers explicitly specify them.
    2. D6/0: the target symmetry is fourfold, so each sprite must satisfy both
       a vertical axis and a horizontal axis simultaneously.
    3. D11/2: the solver must work in two stages, first reading all axis markers
       for all sprites, then performing one global denoising pass.

concrete_pixel_behavior:
  input:
    - 22x22 grid with one dominant background color and exactly three payload
      sprite colors. Red (2) is reserved for markers.
    - Each sprite is close to fourfold symmetry around an integer row and an
      integer column.
    - Every sprite has two external red markers:
      * one vertical red line segment above or below the sprite, aligned to the
        sprite's vertical axis column;
      * one horizontal red line segment left or right of the sprite, aligned to
        the sprite's horizontal axis row.
      Both markers sit one background cell away from the sprite so they are
      visible but not merged with payload.
    - Noise pixels are same-color fragments attached near the sprite. Some are
      deliberately chosen as left/right pairs without top/bottom support, so
      they pass a vertical-only test but fail fourfold symmetry.
  output:
    - All red markers disappear.
    - For each payload sprite, only pixels belonging to complete mirror orbits
      under both marked axes survive.

behavior_trace:
  - Original task:
    1. Collect one color layer.
    2. Search for the best vertical axis from the noisy pixels.
    3. Remove pixels lacking a left/right mate under that axis.
  - Combo task:
    1. Detect every red marker component first and classify it as vertical-axis
       metadata or horizontal-axis metadata from its line orientation.
    2. Match those markers to the nearby payload components and record every
       sprite's full axis pair before deleting anything.
    3. Remove any payload pixel whose fourfold orbit is incomplete under the
       recorded axis pair.
  - Key difference:
    a vertically mirrored noise pair above the horizontal axis survives the
    original rule but is deleted here because its lower reflected pair is absent.

complexity_calibration:
  - Only one metadata color is introduced, and it always repeats in the same two
    roles: vertical line = vertical axis, horizontal line = horizontal axis.
  - Each sample contains three sprites, so the marker convention is repeated
    multiple times per grid and becomes learnable from few examples.
  - Marker lengths vary only in a small range (2 or 3), which keeps the visual
    class narrow while still preventing a brittle "exactly one shape" shortcut.
  - The generator forces at least one sprite to contain a vertical noise pair
    that the old vertical-only cleaner would keep, making the D6 change visible.

implicit_assumption_review_of_original_main:
  - "all non-bg colors are payload": no longer true, because red is metadata and
    must be removed from the output rather than denoised as an object.
  - "one color layer implies one axis search problem": no longer sufficient,
    because the solver must first group marker components with payload
    components, then use explicit axes instead of searching.
  - "checking one mirror partner is enough": false after D6/0. A valid survivor
    now needs its entire orbit under both axes.
  - "processing objects independently in discovery order is harmless": weakened
    by D11/2. The new implementation explicitly separates axis-discovery from
    deletion so all assignments are fixed before any payload is erased.

placement_legality:
  - Three large slots keep sprites separated and leave room for one vertical and
    one horizontal marker around each sprite.
  - Marker cells and the one-cell gaps beside them are reserved and blocked from
    noise placement.
  - Every marker axis coordinate lies inside exactly one sprite bbox in the
    generator, so nearest valid-span matching is stable.

visual_observability_and_cleanliness:
  - A human sees two red lines near each sprite, one vertical and one
    horizontal, so the "double-axis" instruction is visible rather than latent.
  - The output removes both the red metadata and asymmetric payload fragments.
  - Objects never overlap and markers never overwrite payload, so the grid stays
    readable.

anti_degeneracy_target:
  - The generator rejects samples for which a vertical-only marked-axis cleaner
    produces the same result as the fused solver. This quantifies that the
    horizontal-axis requirement is doing real work on generated inputs.

new_main_assumptions_checked_against_generator:
  - Assumption: marker components are straight 4-connected lines with unambiguous
    orientation. Guarantee: lengths are always 2 or 3, never 1.
  - Assumption: each payload component receives exactly one vertical marker and
    one horizontal marker. Guarantee: placement creates one of each per sprite.
  - Assumption: each sprite stays a single 4-connected payload component even
    after noise. Guarantee: clean sprites are connected and noise is added only
    4-adjacent to the owning sprite.
  - Assumption: using original-stage coordinates for deletion is valid. Guarantee:
    all markers are external and do not share cells with payload.
"""

import math
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import get_bg, label_mask, paste


MARK = 2
GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 9, 9), (0, 12, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 9, 10), (0, 11, 9, 20), (12, 3, 21, 18)],
]

PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 3
HEIGHT_MAX = 7
WIDTH_MIN = 3
WIDTH_MAX = 7
DENSITY_LO = 0.35
DENSITY_HI = 0.89

NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 80

MARKER_GAP = 1
MARKER_LENGTHS = (2, 3)

SPRITE_RETRIES = 240
GEN_RETRIES = 500
FLIP_PROB = 0.5


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


def _orbit_cells(
    r: int, c: int, ky: int, kx: int, height: int, width: int
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


def _parse_scene(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    marker_lab, marker_n = label_mask(g == int(MARK), connectivity=4)

    objects: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        objects.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    markers: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        rs = sorted({int(r) for r, _c in coords})
        cs = sorted({int(c) for _r, c in coords})
        if len(cs) == 1 and len(rs) >= 2:
            markers.append(
                {
                    "id": int(lbl),
                    "kind": "vertical",
                    "axis_c": int(cs[0]),
                    "bbox": (int(rs[0]), int(cs[0]), int(rs[-1]), int(cs[0])),
                }
            )
        elif len(rs) == 1 and len(cs) >= 2:
            markers.append(
                {
                    "id": int(lbl),
                    "kind": "horizontal",
                    "axis_r": int(rs[0]),
                    "bbox": (int(rs[0]), int(cs[0]), int(rs[0]), int(cs[-1])),
                }
            )

    return objects, markers


def _assign_axes(objects: list[dict], markers: list[dict]) -> dict[int, dict[str, int]]:
    axes: dict[int, dict[str, int]] = {int(obj["id"]): {} for obj in objects}

    for kind, sort_key in (("vertical", lambda rec: rec["bbox"][0]), ("horizontal", lambda rec: rec["bbox"][1])):
        used_obj_ids: set[int] = set()
        subset = [marker for marker in markers if marker["kind"] == kind]
        for marker in sorted(subset, key=sort_key):
            if kind == "vertical":
                axis_c = int(marker["axis_c"])
                candidates = [
                    obj
                    for obj in objects
                    if int(obj["id"]) not in used_obj_ids
                    and int(obj["bbox"][1]) <= axis_c <= int(obj["bbox"][3])
                ]
                if not candidates:
                    continue
                obj = min(
                    candidates,
                    key=lambda rec: (
                        _bbox_gap(marker["bbox"], rec["bbox"]),
                        abs(axis_c - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2)),
                    ),
                )
                used_obj_ids.add(int(obj["id"]))
                axes[int(obj["id"])]["kx"] = int(2 * axis_c)
            else:
                axis_r = int(marker["axis_r"])
                candidates = [
                    obj
                    for obj in objects
                    if int(obj["id"]) not in used_obj_ids
                    and int(obj["bbox"][0]) <= axis_r <= int(obj["bbox"][2])
                ]
                if not candidates:
                    continue
                obj = min(
                    candidates,
                    key=lambda rec: (
                        _bbox_gap(marker["bbox"], rec["bbox"]),
                        abs(axis_r - ((int(rec["bbox"][0]) + int(rec["bbox"][2])) // 2)),
                    ),
                )
                used_obj_ids.add(int(obj["id"]))
                axes[int(obj["id"])]["ky"] = int(2 * axis_r)

    return axes


def _vertical_only_marked_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    out = g.copy()
    out[g == int(MARK)] = int(bg)

    objects, markers = _parse_scene(g, bg)
    axes = _assign_axes(objects, markers)

    for obj in objects:
        obj_axes = axes.get(int(obj["id"]), {})
        if "kx" not in obj_axes:
            continue
        pts = {(int(r), int(c)) for r, c in obj["coords"]}
        color = int(obj["color"])
        kx = int(obj_axes["kx"])
        for r, c in pts:
            mc = int(kx) - int(c)
            if not (0 <= mc < width and int(g[int(r), mc]) == color):
                out[int(r), int(c)] = int(bg)
    return out


def fourfold_gain_metric(grid: np.ndarray) -> int:
    """Extra payload pixels removed by the fused fourfold cleaner versus vertical-only cleanup."""
    fused = main(grid)
    vertical_only = _vertical_only_marked_clean(grid)
    bg = int(get_bg(np.array(grid, dtype=int)))
    return int(np.count_nonzero((vertical_only != int(bg)) & (fused == int(bg))))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove unmatched sprite pixels using explicitly marked horizontal and vertical axes.

    Input structure:
    - bg is the dominant background color. Red (2) is a metadata color, not a payload color.
    - Three colored sprites appear with one nearby red vertical line segment and one nearby red horizontal line segment, together marking the sprite's axis pair.
    - Each sprite is almost fourfold symmetric, but some same-color noise fragments fail one or both mirror directions.

    Rule:
    - First read every red marker, classify its orientation, and match all markers to their nearby sprites to recover each sprite's horizontal and vertical axes.
    - Then, using those fixed axis pairs, delete every payload pixel whose full orbit under both reflections is incomplete, and erase all red markers.

    Key insight: The puzzle is no longer about inferring symmetry from noisy payload alone; the visible red lines explicitly encode both axes, and the real work is two-stage reasoning: collect all axis metadata first, then keep only complete fourfold mirror orbits.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    objects, markers = _parse_scene(g, bg)
    axes = _assign_axes(objects, markers)

    for obj in objects:
        obj_axes = axes.get(int(obj["id"]), {})
        if "ky" not in obj_axes or "kx" not in obj_axes:
            continue
        pts = {(int(r), int(c)) for r, c in obj["coords"]}
        ky = int(obj_axes["ky"])
        kx = int(obj_axes["kx"])
        for r, c in pts:
            orbit = _orbit_cells(r, c, ky, kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)

    return out


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (int(h - 1), int(w - 1))
    mask[seed] = True
    while int(mask.sum()) < int(target):
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if (r > 0 and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c]) or (c > 0 and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1]):
                    frontier.append((int(r), int(c)))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[int(rr), int(cc)] = True
    return mask


def _mirror_canonical(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    sprite = np.zeros((h, w), dtype=bool)
    top_h, left_w = mask.shape
    for r, c in np.argwhere(mask):
        rows = {int(r), int(h - 1 - r)}
        cols = {int(c), int(w - 1 - c)}
        for rr in rows:
            for cc in cols:
                sprite[int(rr), int(cc)] = True
    return sprite


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> np.ndarray:
    odd_heights = [h for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1) if h % 2 == 1]
    odd_widths = [w for w in range(WIDTH_MIN, min(WIDTH_MAX, max_w) + 1) if w % 2 == 1]
    if not odd_heights or not odd_widths:
        raise ValueError("no odd dimensions available for fourfold sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.choice(odd_heights)
        w = random.choice(odd_widths)
        top_h = (h + 1) // 2
        left_w = (w + 1) // 2
        target = random.randint(2, top_h * left_w)
        canonical = _grow_connected_mask(top_h, left_w, target)
        full_mask = _mirror_canonical(canonical, h, w)
        full_mask = _crop_mask(full_mask)

        hh, ww = full_mask.shape
        if hh > max_h or ww > max_w or hh % 2 == 0 or ww % 2 == 0:
            continue
        count = int(full_mask.sum())
        area = int(hh * ww)
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue
        lab, n = label_mask(full_mask, connectivity=4)
        if int(n) != 1:
            continue

        sprite = np.zeros((hh, ww), dtype=int)
        sprite[full_mask] = int(color)
        return sprite

    raise ValueError("failed to build clean fourfold sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        vlen = int(spec["v_marker_len"])
        hlen = int(spec["h_marker_len"])
        vside = str(spec["v_marker_side"])
        hside = str(spec["h_marker_side"])

        extra_top = int(vlen + MARKER_GAP) if vside == "above" else 0
        extra_bottom = int(vlen + MARKER_GAP) if vside == "below" else 0
        extra_left = int(hlen + MARKER_GAP) if hside == "left" else 0
        extra_right = int(hlen + MARKER_GAP) if hside == "right" else 0
        bundle_h = int(extra_top + h + extra_bottom)
        bundle_w = int(extra_left + w + extra_right)
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)
        if bundle_h > slot_h or bundle_w > slot_w:
            return None

        bundle_r0 = random.randint(int(r0s), int(r1s - bundle_h + 1))
        bundle_c0 = random.randint(int(c0s), int(c1s - bundle_w + 1))
        sprite_r0 = int(bundle_r0 + extra_top)
        sprite_c0 = int(bundle_c0 + extra_left)
        axis_r = int(sprite_r0 + h // 2)
        axis_c = int(sprite_c0 + w // 2)

        if vside == "above":
            v_marker_cells = [(int(sprite_r0 - MARKER_GAP - vlen + i), int(axis_c)) for i in range(vlen)]
            v_gap_cells = [(int(sprite_r0 - 1), int(axis_c))]
        else:
            v_marker_cells = [(int(sprite_r0 + h + MARKER_GAP + i), int(axis_c)) for i in range(vlen)]
            v_gap_cells = [(int(sprite_r0 + h), int(axis_c))]

        if hside == "left":
            h_marker_cells = [(int(axis_r), int(sprite_c0 - MARKER_GAP - hlen + i)) for i in range(hlen)]
            h_gap_cells = [(int(axis_r), int(sprite_c0 - 1))]
        else:
            h_marker_cells = [(int(axis_r), int(sprite_c0 + w + MARKER_GAP + i)) for i in range(hlen)]
            h_gap_cells = [(int(axis_r), int(sprite_c0 + w))]

        paste(out, sprite, int(sprite_r0), int(sprite_c0), transparent=0)
        for r, c in v_marker_cells + h_marker_cells + v_gap_cells + h_gap_cells:
            blocked[int(r), int(c)] = True

        spec["r0"] = int(sprite_r0)
        spec["c0"] = int(sprite_c0)
        spec["true_ky"] = int(2 * axis_r)
        spec["true_kx"] = int(2 * axis_c)
        spec["v_marker_cells"] = v_marker_cells
        spec["h_marker_cells"] = h_marker_cells
        placed.append(spec)

    return out, placed, blocked


def _adjacent_mask(clean_mask: np.ndarray) -> np.ndarray:
    adj = np.zeros_like(clean_mask, dtype=bool)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    return adj & ~clean_mask


def _fragment_candidates(
    clean_mask: np.ndarray, true_ky: int, true_kx: int, blocked: np.ndarray
) -> dict[str, list[dict]]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask) & ~blocked
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
        mr = int(true_ky - r)
        mc = int(true_kx - c)
        if mr == r or mc == c:
            continue

        key = ((int(r), int(c)),)
        singletons[key] = {"kind": "singleton", "cells": key, "orbit": frozenset(orbit)}

        vpair = tuple(sorted(((int(r), int(c)), (int(r), int(mc)))))
        if 0 <= mc < width and all(adj[rr, cc] for rr, cc in vpair):
            vertical[vpair] = {"kind": "vertical_pair", "cells": vpair, "orbit": frozenset(orbit)}

        hpair = tuple(sorted(((int(r), int(c)), (int(mr), int(c)))))
        if 0 <= mr < height and all(adj[rr, cc] for rr, cc in hpair):
            horizontal[hpair] = {"kind": "horizontal_pair", "cells": hpair, "orbit": frozenset(orbit)}

    return {
        "vertical_pair": list(vertical.values()),
        "horizontal_pair": list(horizontal.values()),
        "singleton": list(singletons.values()),
    }


def _choose_noise_fragments(
    candidates: dict[str, list[dict]], lo: int, hi: int, require_vertical_pair: bool
) -> list[dict] | None:
    minimum = max(lo, 2 if require_vertical_pair else lo)
    if minimum > hi:
        return None

    targets = list(range(minimum, hi + 1))
    random.shuffle(targets)
    all_candidates = candidates["vertical_pair"] + candidates["horizontal_pair"] + candidates["singleton"]

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


def _add_noise(
    inp: np.ndarray, out: np.ndarray, spec: dict, blocked: np.ndarray, *, require_vertical_pair: bool
) -> bool:
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    clean_mask = out == int(color)
    clean_count = int(np.count_nonzero(clean_mask))

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    candidates = _fragment_candidates(clean_mask, true_ky, true_kx, blocked)
    fragments = _choose_noise_fragments(candidates, lo, hi, require_vertical_pair)
    if not fragments:
        return False

    for r, c in sorted({cell for fragment in fragments for cell in fragment["cells"]}):
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(10) if c != int(MARK)]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict] = []
    try:
        for color, slot in zip(obj_colors, slots):
            v_marker_len = int(random.choice(MARKER_LENGTHS))
            h_marker_len = int(random.choice(MARKER_LENGTHS))
            v_marker_side = random.choice(["above", "below"])
            h_marker_side = random.choice(["left", "right"])
            max_h = int(min(HEIGHT_MAX, slot[2] - slot[0] + 1 - MARKER_GAP - v_marker_len))
            max_w = int(min(WIDTH_MAX, slot[3] - slot[1] + 1 - MARKER_GAP - h_marker_len))
            sprite = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "v_marker_len": int(v_marker_len),
                    "h_marker_len": int(h_marker_len),
                    "v_marker_side": str(v_marker_side),
                    "h_marker_side": str(h_marker_side),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs, blocked = result

    eligible = [spec for spec in placed_specs if int(np.count_nonzero(out == int(spec["color"]))) >= 8]
    random.shuffle(eligible)
    required_spec = None
    for spec in eligible:
        candidates = _fragment_candidates(out == int(spec["color"]), spec["true_ky"], spec["true_kx"], blocked)
        if candidates["vertical_pair"]:
            required_spec = spec
            break
    if required_spec is None:
        return None

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, blocked, require_vertical_pair=(spec is required_spec)):
            return None
        for r, c in spec["v_marker_cells"] + spec["h_marker_cells"]:
            inp[int(r), int(c)] = int(MARK)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_vertical_only_marked_clean(inp), out):
        return None
    if fourfold_gain_metric(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
