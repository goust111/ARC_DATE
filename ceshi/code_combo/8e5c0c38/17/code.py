# Combo variant of 8e5c0c38 (index 17)
# combo_spec: D5/0+D6/0
# Name: 标记双轴四重对称修复

from __future__ import annotations

"""
Design analysis for combo D5/0 + D6/0.

a) Concrete behavior and trace:
   - Original task:
     each sprite color is handled independently; the solver searches a vertical
     axis from the noisy pixels, then removes same-color pixels lacking a
     left/right mate.
   - Combo task:
     each sprite now has two external markers:
       * red vertical segment = the sprite's vertical symmetry axis
       * yellow horizontal segment = the sprite's horizontal symmetry axis
     The solver no longer searches any axis. It pairs both markers to the
     nearby sprite and keeps only pixels whose full 4-way orbit exists under
     both marked axes.
   - Small pixel trace:
     suppose a sprite has marked axes at row 8 and col 6. A payload pixel at
     (6,4) survives only if the same color also exists at (6,8), (10,4), and
     (10,8), modulo axis degeneracy when a point lies on a marked axis.
     Under the original task, the pair (6,4)/(6,8) would already be enough.
     Under the combo task that vertically matched pair is still deleted if the
     bottom half is missing.

b) Complexity calibration:
   - Marker colors are fixed and reused on all three objects, so the visible
     codebook is small: red always means the column axis, yellow always means
     the row axis.
   - Marker geometry is also narrow: a short 1xk or kx1 segment outside the
     sprite, always aligned to the relevant axis.
   - To make the D6 change observable, the generator forces at least one sprite
     to contain a vertically mirrored noise pair that would survive a
     vertical-only cleaner but is removed by the fourfold rule.
   - Odd sprite widths and heights are enforced. This removes half-integer axis
     ambiguity and lets one visible row marker and one visible column marker
     specify the axes unambiguously.

c) Hidden-assumption review of the original main():
   - "colors_present(g, exclude={bg})":
     no longer valid, because some non-bg colors are metadata markers, not
     payload. The new solver excludes marker colors and erases them in output.
   - "for each color, search all vertical axes":
     replaced. The new task assumes the axes are given explicitly and must be
     read from nearby marker components.
   - "a pixel survives if its vertical mirror exists":
     too weak now. Survival requires the full orbit under both marked axes.
   - "out starts as copy of g":
     still fine only after clearing markers, since markers disappear in output.
   - "grouping by color is sufficient":
     not robust anymore. The relevant grouping is sprite-component plus its two
     nearby markers, because the markers use shared colors across all sprites.

d) Placement legality:
   - The generator still uses three non-overlapping slots.
   - Within a slot, the sprite is placed first, then one row-axis marker is put
     to its left or right and one column-axis marker is put above or below.
   - The one-cell gaps between marker and sprite are reserved, and marker cells
     are blocked from noise placement.
   - Slot sizes were widened relative to the original so that sprite, two
     markers, and local noise all fit without overlap or clipping.

e) Visual observability:
   - Across 2-4 examples a human can notice:
       * red vertical ticks and yellow horizontal ticks vanish in the output
       * each cleaned sprite becomes symmetric in both directions around the
         marked row/column pair
       * some noisy left/right pairs still disappear, signaling that vertical
         pairing alone is not the rule
   - No numeric color ordering is used; only visible geometry and repeated
     marker roles matter.

f) Visual cleanliness:
   - Markers are external, not embedded, so payload shape and metadata stay
     visually separable.
   - Slot spacing prevents cross-object ambiguity.
   - Noise remains 4-adjacent to its owner sprite, keeping the local "repair"
     story readable instead of scattering pixels around the canvas.

g) Change inferability:
   - There is no recoloring of payload objects. Only the fixed marker colors
     red/yellow act as metadata and disappear.
   - Because the same marker colors repeat for all sprites, the mapping
     red->vertical-axis and yellow->horizontal-axis is inferable from few
     examples.

h) Anti-degeneration target:
   - The generator rejects any sample where a legacy "marked vertical-only"
     cleaner already produces the target output. At least one sprite must
     therefore require the new horizontal constraint.

i) New solver assumptions checked against the generator:
   - Assumption: each payload sprite is one 4-connected non-marker component.
     Guarantee: clean sprites are connectivity-checked, and noise is added only
     4-adjacent to that sprite.
   - Assumption: each sprite has exactly one nearby red marker whose column lies
     inside its bbox, and one nearby yellow marker whose row lies inside its
     bbox.
     Guarantee: placement constructs exactly one of each per sprite.
   - Assumption: marker candidates can be disambiguated by bbox-span plus
     nearest-gap pairing.
     Guarantee: slot spacing keeps every marker much nearer to its owner than to
     any other sprite.
   - Assumption: deleting all incomplete orbits is correct.
     Guarantee: every injected noise fragment occupies only a strict subset of a
     fourfold orbit, never a full orbit.
"""

import math
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import get_bg, label_mask, paste


GRID_SIZE = 22
MARK_VERT = 2
MARK_HOR = 4
NUM_SPRITES = 3

# ── CFG: slot layout ──
LAYOUTS = [
    [(0, 0, 10, 9), (0, 12, 10, 21), (12, 4, 21, 17)],
    [(0, 1, 10, 10), (0, 11, 10, 20), (12, 3, 21, 18)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 7
PIXEL_COUNT_MAX = 46
HEIGHT_VALUES = (3, 5, 7)
WIDTH_VALUES = (3, 5, 7, 9)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: markers ──
MARKER_GAP = 1
MARKER_LENGTHS = (1, 2, 3)

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
NOISE_PLAN_RETRIES = 100

# ── CFG: retry limits ──
SPRITE_RETRIES = 300
GEN_RETRIES = 500

# ── CFG: augmentation ──
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


def _extract_payload_components(
    g: np.ndarray,
    *,
    bg: int,
) -> list[dict]:
    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK_VERT)) & (g != int(MARK_HOR)), connectivity=4)
    comps: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "pts": set(coords),
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def _extract_axis_markers(g: np.ndarray, marker_color: int, axis_kind: str) -> list[dict]:
    lab, n = label_mask(g == int(marker_color), connectivity=4)
    markers: list[dict] = []
    for lbl in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(lbl))]
        if not coords:
            continue
        rs = {r for r, _c in coords}
        cs = {c for _r, c in coords}
        if axis_kind == "vertical":
            if len(cs) != 1:
                continue
            axis_pos = int(next(iter(cs)))
        else:
            if len(rs) != 1:
                continue
            axis_pos = int(next(iter(rs)))
        r_vals = [r for r, _c in coords]
        c_vals = [c for _r, c in coords]
        markers.append(
            {
                "id": int(lbl),
                "coords": coords,
                "axis_pos": int(axis_pos),
                "bbox": (int(min(r_vals)), int(min(c_vals)), int(max(r_vals)), int(max(c_vals))),
            }
        )
    return markers


def _assign_markers_to_objects(objs: list[dict], markers: list[dict], axis_kind: str) -> dict[int, int]:
    assignments: dict[int, int] = {}
    used: set[int] = set()
    span_lo = 1 if axis_kind == "vertical" else 0
    span_hi = 3 if axis_kind == "vertical" else 2

    for obj in sorted(objs, key=lambda rec: (rec["bbox"][0], rec["bbox"][1])):
        candidates = [
            marker
            for marker in markers
            if marker["id"] not in used
            and int(obj["bbox"][span_lo]) <= int(marker["axis_pos"]) <= int(obj["bbox"][span_hi])
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda marker: (_bbox_gap(obj["bbox"], marker["bbox"]), marker["bbox"][0], marker["bbox"][1]))
        assignments[int(obj["id"])] = int(best["axis_pos"])
        used.add(int(best["id"]))
    return assignments


def _clear_markers(g: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    out[(g == int(MARK_VERT)) | (g == int(MARK_HOR))] = int(bg)
    return out


def _marked_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = _clear_markers(g, bg)
    objs = _extract_payload_components(g, bg=bg)
    vmarkers = _extract_axis_markers(g, MARK_VERT, "vertical")
    v_assign = _assign_markers_to_objects(objs, vmarkers, "vertical")
    width = g.shape[1]

    for obj in objs:
        axis_c = v_assign.get(int(obj["id"]))
        if axis_c is None:
            continue
        kx = 2 * int(axis_c)
        color = int(obj["color"])
        for r, c in obj["coords"]:
            mc = int(kx) - int(c)
            if not (0 <= mc < width and (int(r), int(mc)) in obj["pts"] and int(g[int(r), int(mc)]) == color):
                out[int(r), int(c)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each marked sprite by enforcing fourfold symmetry around given row and column axes.

    Input structure:
    - bg is the dominant background color. Red (2) marks a sprite's vertical axis and yellow (4) marks its horizontal axis.
    - Three non-marker colored sprites appear in separate regions, each close to a fourfold-symmetric shape.
    - Same-color noise pixels may be attached to a sprite; some even come in left/right pairs but still miss their top/bottom mates.

    Rule:
    - Identify payload components while ignoring the two marker colors, then pair each component with the nearby red marker whose column lies inside its bbox and the nearby yellow marker whose row lies inside its bbox.
    - Use those marked axes directly instead of searching for symmetry axes.
    - Remove every payload pixel whose full orbit under the marked horizontal and vertical reflections is incomplete, and erase all markers.

    Key insight: The visible markers solve the axis-location problem; the real task is to use the given row/column pair to test complete fourfold support, which is stricter than ordinary left/right mirroring.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = _clear_markers(g, bg)
    objs = _extract_payload_components(g, bg=bg)
    vmarkers = _extract_axis_markers(g, MARK_VERT, "vertical")
    hmarkers = _extract_axis_markers(g, MARK_HOR, "horizontal")
    v_assign = _assign_markers_to_objects(objs, vmarkers, "vertical")
    h_assign = _assign_markers_to_objects(objs, hmarkers, "horizontal")

    for obj in objs:
        axis_c = v_assign.get(int(obj["id"]))
        axis_r = h_assign.get(int(obj["id"]))
        if axis_c is None or axis_r is None:
            continue
        ky = 2 * int(axis_r)
        kx = 2 * int(axis_c)
        pts = obj["pts"]
        for r, c in obj["coords"]:
            orbit = _orbit_cells(int(r), int(c), ky, kx, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)
    return out


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    cells = {(0, 0), (top_h - 1, 0), (0, left_w - 1)}
    if top_h > 1 and left_w > 1 and random.random() < 0.7:
        cells.add((top_h - 1, left_w - 1))

    fill_prob = random.uniform(0.3, 0.8)
    for r in range(top_h):
        for c in range(left_w):
            if (r, c) in cells:
                continue
            if random.random() < fill_prob:
                cells.add((r, c))

    if len(cells) < 2 and top_h * left_w > 1:
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


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> np.ndarray:
    allowed_h = [h for h in HEIGHT_VALUES if h <= max_h]
    allowed_w = [w for w in WIDTH_VALUES if w <= max_w]
    if not allowed_h or not allowed_w:
        raise ValueError("slot too small for marked fourfold sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.choice(allowed_h)
        w = random.choice(allowed_w)
        pts = _mirror_canonical(_canonical_cells((h + 1) // 2, (w + 1) // 2), h, w)
        count = len(pts)
        area = h * w
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue

        sprite = np.zeros((h, w), dtype=int)
        for r, c in pts:
            sprite[int(r), int(c)] = int(color)
        _lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1:
            continue
        return sprite
    raise ValueError("failed to build clean sprite")


def _choose_marker_styles() -> tuple[list[int], list[str], list[int], list[str]]:
    v_lengths = [1, random.choice([2, 3]), random.choice(MARKER_LENGTHS)]
    h_lengths = [1, random.choice([2, 3]), random.choice(MARKER_LENGTHS)]
    v_sides = ["above", "below", random.choice(["above", "below"])]
    h_sides = ["left", "right", random.choice(["left", "right"])]
    random.shuffle(v_lengths)
    random.shuffle(h_lengths)
    random.shuffle(v_sides)
    random.shuffle(h_sides)
    return v_lengths, v_sides, h_lengths, h_sides


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
        v_len = int(spec["v_marker_len"])
        h_len = int(spec["h_marker_len"])
        v_side = str(spec["v_marker_side"])
        h_side = str(spec["h_marker_side"])

        extra_top = v_len + MARKER_GAP if v_side == "above" else 0
        extra_bottom = v_len + MARKER_GAP if v_side == "below" else 0
        extra_left = h_len + MARKER_GAP if h_side == "left" else 0
        extra_right = h_len + MARKER_GAP if h_side == "right" else 0

        row_lo = int(r0s + extra_top)
        row_hi = int(r1s - extra_bottom - h + 1)
        col_lo = int(c0s + extra_left)
        col_hi = int(c1s - extra_right - w + 1)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        sprite_r0 = random.randint(row_lo, row_hi)
        sprite_c0 = random.randint(col_lo, col_hi)
        axis_r = int(sprite_r0 + h // 2)
        axis_c = int(sprite_c0 + w // 2)

        if v_side == "above":
            v_rows = list(range(int(sprite_r0 - MARKER_GAP - v_len), int(sprite_r0 - MARKER_GAP)))
            v_gap = [(int(sprite_r0 - 1), int(axis_c))]
        else:
            v_rows = list(range(int(sprite_r0 + h + MARKER_GAP), int(sprite_r0 + h + MARKER_GAP + v_len)))
            v_gap = [(int(sprite_r0 + h), int(axis_c))]
        v_cells = [(int(r), int(axis_c)) for r in v_rows]

        if h_side == "left":
            h_cols = list(range(int(sprite_c0 - MARKER_GAP - h_len), int(sprite_c0 - MARKER_GAP)))
            h_gap = [(int(axis_r), int(sprite_c0 - 1))]
        else:
            h_cols = list(range(int(sprite_c0 + w + MARKER_GAP), int(sprite_c0 + w + MARKER_GAP + h_len)))
            h_gap = [(int(axis_r), int(sprite_c0 + w))]
        h_cells = [(int(axis_r), int(c)) for c in h_cols]

        paste(out, sprite, int(sprite_r0), int(sprite_c0), transparent=0)
        for r, c in v_cells + h_cells + v_gap + h_gap:
            blocked[int(r), int(c)] = True

        spec["r0"] = int(sprite_r0)
        spec["c0"] = int(sprite_c0)
        spec["axis_r"] = int(axis_r)
        spec["axis_c"] = int(axis_c)
        spec["true_ky"] = int(2 * axis_r)
        spec["true_kx"] = int(2 * axis_c)
        spec["v_cells"] = v_cells
        spec["h_cells"] = h_cells
        spec["gap_cells"] = v_gap + h_gap
        spec["clean_count"] = int(np.count_nonzero(sprite))
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
    base_grid: np.ndarray,
    clean_mask: np.ndarray,
    true_ky: int,
    true_kx: int,
    blocked: np.ndarray,
    *,
    bg: int,
) -> dict[str, list[dict]]:
    height, width = clean_mask.shape
    adj = _adjacent_mask(clean_mask) & (base_grid == int(bg)) & ~blocked
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
        if any(blocked[int(rr), int(cc)] for rr, cc in orbit):
            continue
        if any(int(base_grid[int(rr), int(cc)]) != int(bg) for rr, cc in orbit):
            continue

        mr = int(true_ky) - r
        mc = int(true_kx) - c
        if mr == r and mc == c:
            continue

        vpair = tuple(sorted({(int(r), int(c)), (int(r), int(mc))}))
        if 0 <= mc < width and len(vpair) == 2 and all(adj[int(rr), int(cc)] for rr, cc in vpair):
            vertical[vpair] = {
                "kind": "vertical_pair",
                "cells": vpair,
                "orbit": frozenset(orbit),
            }

        hpair = tuple(sorted({(int(r), int(c)), (int(mr), int(c))}))
        if 0 <= mr < height and len(hpair) == 2 and all(adj[int(rr), int(cc)] for rr, cc in hpair):
            horizontal[hpair] = {
                "kind": "horizontal_pair",
                "cells": hpair,
                "orbit": frozenset(orbit),
            }

        singletons[((int(r), int(c)),)] = {
            "kind": "singleton",
            "cells": ((int(r), int(c)),),
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
    *,
    require_vertical_pair: bool,
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
            used_orbits: set[tuple[int, int]] = set()
            remaining = int(target)

            if require_vertical_pair:
                starters = [cand for cand in candidates["vertical_pair"] if len(cand["cells"]) <= remaining]
                if not starters:
                    return None
                first = random.choice(starters)
                chosen.append(first)
                used_orbits |= set(first["orbit"])
                remaining -= len(first["cells"])

            while remaining > 0:
                viable = [
                    cand
                    for cand in all_candidates
                    if len(cand["cells"]) <= remaining and set(cand["orbit"]).isdisjoint(used_orbits)
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
                    pairs = [cand for cand in viable if len(cand["cells"]) == 2]
                    cand = random.choice(pairs if pairs and random.random() < 0.7 else viable)

                chosen.append(cand)
                used_orbits |= set(cand["orbit"])
                remaining -= len(cand["cells"])

            if remaining == 0:
                return chosen
    return None


def _add_noise(
    inp: np.ndarray,
    clean_grid: np.ndarray,
    spec: dict,
    blocked: np.ndarray,
    *,
    bg: int,
    require_vertical_pair: bool,
) -> tuple[bool, dict]:
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    clean_mask = clean_grid == int(color)
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False, {}

    candidates = _fragment_candidates(clean_grid, clean_mask, true_ky, true_kx, blocked, bg=bg)
    fragments = _choose_noise_fragments(candidates, lo, hi, require_vertical_pair=require_vertical_pair)
    if not fragments:
        return False, {}

    noise_cells = sorted({cell for frag in fragments for cell in frag["cells"]})
    for r, c in noise_cells:
        inp[int(r), int(c)] = int(color)

    metrics = {
        "noise_cells": int(len(noise_cells)),
        "vertical_pair_fragments": int(sum(frag["kind"] == "vertical_pair" for frag in fragments)),
        "horizontal_pair_fragments": int(sum(frag["kind"] == "horizontal_pair" for frag in fragments)),
        "singleton_fragments": int(sum(frag["kind"] == "singleton" for frag in fragments)),
    }
    return True, metrics


def _apply_markers(inp: np.ndarray, placed_specs: list[dict]) -> None:
    for spec in placed_specs:
        for r, c in spec["v_cells"]:
            inp[int(r), int(c)] = int(MARK_VERT)
        for r, c in spec["h_cells"]:
            inp[int(r), int(c)] = int(MARK_HOR)


def _gen_attempt(return_meta: bool = False):
    palette = [c for c in range(10) if c not in {MARK_VERT, MARK_HOR}]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    v_lengths, v_sides, h_lengths, h_sides = _choose_marker_styles()

    sprite_specs: list[dict] = []
    try:
        for color, slot, v_len, v_side, h_len, h_side in zip(obj_colors, slots, v_lengths, v_sides, h_lengths, h_sides):
            max_h = int(slot[2] - slot[0] + 1 - (v_len + MARKER_GAP))
            max_w = int(slot[3] - slot[1] + 1 - (h_len + MARKER_GAP))
            sprite = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "v_marker_len": int(v_len),
                    "v_marker_side": str(v_side),
                    "h_marker_len": int(h_len),
                    "h_marker_side": str(h_side),
                }
            )
    except ValueError:
        return None

    placed_result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if placed_result is None:
        return None
    clean_grid, placed_specs, blocked = placed_result

    eligible = []
    for spec in placed_specs:
        candidates = _fragment_candidates(
            clean_grid,
            clean_grid == int(spec["color"]),
            int(spec["true_ky"]),
            int(spec["true_kx"]),
            blocked,
            bg=int(bg),
        )
        if candidates["vertical_pair"]:
            eligible.append(spec)
    if not eligible:
        return None
    required_spec = random.choice(eligible)

    inp = clean_grid.copy()
    per_sprite_metrics: list[dict] = []
    for spec in placed_specs:
        ok, metrics = _add_noise(
            inp,
            clean_grid,
            spec,
            blocked,
            bg=int(bg),
            require_vertical_pair=(spec is required_spec),
        )
        if not ok:
            return None
        metrics["color"] = int(spec["color"])
        metrics["required_vertical_pair"] = bool(spec is required_spec)
        per_sprite_metrics.append(metrics)

    _apply_markers(inp, placed_specs)
    out = clean_grid.copy()

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_marked_vertical_clean(inp), out):
        return None

    if return_meta:
        total_vertical_pairs = sum(int(m["vertical_pair_fragments"]) for m in per_sprite_metrics)
        return {
            "input": inp,
            "output": out,
            "metrics": {
                "total_vertical_pair_fragments": int(total_vertical_pairs),
                "required_sprite_has_vertical_pair": True,
                "per_sprite": per_sprite_metrics,
            },
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(return_meta=False)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")


def sample_generation_metadata() -> dict:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(return_meta=True)
        if sample is not None:
            return sample
    raise RuntimeError(f"failed to generate valid sample after {GEN_RETRIES} attempts")
