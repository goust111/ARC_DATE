# Combo variant of 8e5c0c38 (index 5)
# combo_spec: D1/0+D6/0
# Name: 四重对称补全修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The source task searched one vertical axis per color and deleted unmatched pixels.
#    D1 inverts that direction: unmatched evidence means "a mirror partner is missing",
#    so the solver must add pixels instead of removing them. D6 strengthens the symmetry
#    class from left/right mirror to simultaneous left/right and top/bottom mirror.
#    The fused rule is therefore:
#      - search one horizontal axis and one vertical axis for each monochrome sprite;
#      - treat every observed pixel as belonging to a fourfold orbit under that axis pair;
#      - fill every missing member of that orbit.
#    Small trace on a 4-cell orbit:
#      clean orbit = {(1,1), (1,4), (4,1), (4,4)} under axes ky=5, kx=5.
#      original task: if only (1,1) survives, it would search only kx and either delete
#      it (source) or add (1,4) (D1-only).
#      fused task: the same singleton implies three missing partners, so the output adds
#      (1,4), (4,1), and (4,4). This is the visible signature of the combo.
# b) Complexity calibration:
#    The generator keeps the original 22x22 canvas, 3 separated colors, and fixed slot
#    layouts. Clean sprites are irregular fourfold-symmetric monochrome shapes, not
#    trivial rectangles, and each sample includes at least one orbit with only a single
#    survivor so the output expansion is visibly stronger than plain vertical completion.
# c) Hidden assumptions from the original main():
#    - "one vertical axis is enough" no longer holds; both axis families must be searched.
#    - "unmatched means noise" is inverted; unmatched observed pixels are preserved and
#      used as seeds for completion.
#    - "one opposite-side partner is the full test" no longer holds; we now reason in
#      orbit closures of size 1, 2, or 4.
#    - Because filling writes new cells, the read set must stay anchored to the original
#      input points. Otherwise earlier fills could recursively trigger unrelated fills.
# d) Placement legality:
#    Generation stays output-first. We build the complete fourfold-symmetric target,
#    place each sprite wholly inside one reserved slot, then delete interior orbit cells.
#    Deletions only shrink sprites, so there is no risk of overlap or out-of-bounds
#    writes; the solver only refills cells that already belonged to the placed target.
# e) Visual observability:
#    Across a few examples, the key pattern is readable by eye: inputs are nearly
#    symmetric in both directions but have "missing quadrants" or "missing lower halves"
#    that reappear in the outputs. The outputs are larger than the inputs, unlike the
#    source denoiser, and at least one damaged orbit per sample requires horizontal
#    reasoning beyond pure left/right mirroring.
# f) Visual cleanliness:
#    Colors still identify independent sprites, slots keep the sprites far apart, and
#    missing cells are confined to each sprite's own bbox. No cross-object interactions or
#    color remappings are introduced.
# g) New solver assumptions and generator guarantees:
#    - Assumption: each non-background color belongs to one sprite with one axis pair.
#      Guarantee: generator still uses exactly 3 distinct non-background colors.
#    - Assumption: the true axis pair is recoverable from the surviving complete orbits.
#      Guarantee: the generator keeps bbox-anchor orbits intact and rejects samples unless
#      the new solver reconstructs the exact target.
#    - Assumption: filling the orbit-closure of every observed point yields the target.
#      Guarantee: the input is constructed strictly by deleting proper subsets of target
#      orbits; no foreign same-color noise is added.


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


def _candidate_metrics(
    pts: set[tuple[int, int]],
    ky: int,
    kx: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    orbit_counts: dict[frozenset[tuple[int, int]], int] = {}
    for r, c in pts:
        orbit = _orbit_cells(r, c, ky, kx, height, width)
        if orbit is None:
            return (-1, -1, -10**9, -10**9)
        sig = frozenset(orbit)
        orbit_counts[sig] = orbit_counts.get(sig, 0) + 1

    complete = 0
    supported = 0
    closure_size = 0
    for sig, seen in orbit_counts.items():
        orbit_size = len(sig)
        closure_size += orbit_size
        if seen == orbit_size:
            complete += seen
        if seen >= 2:
            supported += seen
    additions = closure_size - len(pts)
    return (int(complete), int(supported), -int(additions), -int(closure_size))


def _best_fourfold_axes(
    pts: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[int, int, tuple[int, int, int, int]]:
    best_metrics = (-1, -1, -10**9, -10**9)
    best_ky = 0
    best_kx = 0
    for ky in range(0, 2 * (height - 1) + 1):
        for kx in range(0, 2 * (width - 1) + 1):
            metrics = _candidate_metrics(pts, ky, kx, height, width)
            if metrics > best_metrics:
                best_metrics = metrics
                best_ky = int(ky)
                best_kx = int(kx)
    return best_ky, best_kx, best_metrics


def _axis_pair_scores(
    coords: list[tuple[int, int]],
    height: int,
    width: int,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ky in range(0, 2 * (height - 1) + 1):
        for kx in range(0, 2 * (width - 1) + 1):
            scores[(ky, kx)] = _candidate_metrics(pts, ky, kx, height, width)
    return scores


def _second_best_pair_metrics(
    coords: list[tuple[int, int]],
    true_pair: tuple[int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    scores = _axis_pair_scores(coords, height, width)
    return max(score for pair, score in scores.items() if pair != true_pair)


def _vertical_pair_score(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for kx in range(0, 2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            mc = kx - c
            if 0 <= mc < width and (r, mc) in pts:
                score += 1
        scores[kx] = score
    return scores


def _legacy_vertical_fill(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        scores = _vertical_pair_score([tuple(x) for x in pts], width)
        best_kx = max(scores, key=scores.get)
        for r, c in pts:
            rr = int(r)
            mc = int(best_kx) - int(c)
            if 0 <= mc < width and int(g[rr, mc]) != int(col) and int(out[rr, mc]) == int(bg):
                out[rr, mc] = int(col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color layer by restoring full fourfold mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors appear as separate monochrome sprites, each close to a shape symmetric across both a horizontal axis and a vertical axis.
    - The visible pixels are an incomplete subset of the intended sprite, so one surviving pixel may be missing up to three mirrored companions.

    Rule:
    - For each non-bg color, search for the horizontal axis and vertical axis that best explain the surviving same-color pixels as partial fourfold orbits.
    - For every observed same-color pixel, fill all missing cells in its orbit under that axis pair.

    Key insight: Each color is still solved independently, but asymmetry now means missing orbit members rather than removable noise; the repair step is to close fourfold orbits, not to prune pixels.
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

        best_ky, best_kx, _best_metrics = _best_fourfold_axes(pts, height, width)
        orbit_cache: dict[frozenset[tuple[int, int]], set[tuple[int, int]]] = {}
        for r, c in pts:
            orbit = _orbit_cells(r, c, best_ky, best_kx, height, width)
            if orbit is None:
                continue
            orbit_cache.setdefault(frozenset(orbit), orbit)

        for orbit in orbit_cache.values():
            for rr, cc in orbit:
                if int(out[rr, cc]) in (bg, int(col)):
                    out[rr, cc] = int(col)

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

# ── CFG: missing pixels ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12
REMOVAL_PLAN_RETRIES = 120

# ── CFG: axis distinctness ──
CLEAN_PAIR_GAP_MIN = 4

# ── CFG: retry limits ──
SPRITE_RETRIES = 240
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _canonical_cells(top_h: int, left_w: int) -> set[tuple[int, int]]:
    forced = {(0, 0)}
    if left_w > 1:
        forced.add((0, left_w - 1))
    if top_h > 1:
        forced.add((top_h - 1, 0))
    if top_h > 1 and left_w > 1:
        forced.add((top_h - 1, left_w - 1))

    cells = set(forced)
    fill_prob = random.uniform(0.25, 0.82)
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
        second_best = _second_best_pair_metrics(coords, true_pair, h, w)
        if count - second_best[0] < CLEAN_PAIR_GAP_MIN:
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


def _unique_orbits(
    mask: np.ndarray,
    true_ky: int,
    true_kx: int,
) -> list[tuple[tuple[int, int], ...]]:
    height, width = mask.shape
    seen: set[frozenset[tuple[int, int]]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r, c in np.argwhere(mask):
        orbit = _orbit_cells(int(r), int(c), true_ky, true_kx, height, width)
        if orbit is None or not (orbit <= {tuple(x) for x in np.argwhere(mask)}):
            continue
        sig = frozenset(orbit)
        if sig not in seen:
            seen.add(sig)
            orbits.append(tuple(sorted(sig)))
    return orbits


def _is_anchor_orbit(orbit: tuple[tuple[int, int], ...], mask_shape: tuple[int, int]) -> bool:
    height, width = mask_shape
    for r, c in orbit:
        if r in (0, height - 1) or c in (0, width - 1):
            return True
    return False


def _vertical_fill_orbit(seen: set[tuple[int, int]], kx: int, width: int) -> set[tuple[int, int]]:
    filled = set(seen)
    for r, c in list(seen):
        mc = int(kx) - int(c)
        if 0 <= mc < width:
            filled.add((int(r), int(mc)))
    return filled


def _removal_options(
    orbit: tuple[tuple[int, int], ...],
    true_kx: int,
    width: int,
) -> list[dict]:
    orbit_set = set(orbit)
    by_size: dict[frozenset[tuple[int, int]], dict] = {}

    def add_option(removed: set[tuple[int, int]], label: str) -> None:
        if not removed or removed == orbit_set:
            return
        kept = orbit_set - removed
        strong = _vertical_fill_orbit(kept, true_kx, width) != orbit_set
        key = frozenset(removed)
        by_size[key] = {
            "removed": tuple(sorted(removed)),
            "removed_count": len(removed),
            "strong": bool(strong),
            "triple": len(removed) == 3,
            "label": label,
        }

    cells = list(orbit)
    size = len(cells)
    if size == 4:
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        row_groups = [{cell for cell in cells if cell[0] == r} for r in rows]
        col_groups = [{cell for cell in cells if cell[1] == c} for c in cols]
        for cell in cells:
            add_option({cell}, "drop_one")
            add_option(orbit_set - {cell}, "leave_one")
        for group in row_groups:
            add_option(set(group), "drop_row_pair")
        for group in col_groups:
            add_option(set(group), "drop_col_pair")
    elif size == 2:
        for cell in cells:
            add_option({cell}, "drop_one")
    return list(by_size.values())


def _choose_removals(
    clean_mask: np.ndarray,
    true_ky: int,
    true_kx: int,
    lo: int,
    hi: int,
    *,
    require_strong: bool,
    require_triple: bool,
) -> tuple[list[tuple[int, int]], dict] | None:
    height, width = clean_mask.shape
    all_orbits = _unique_orbits(clean_mask, true_ky, true_kx)
    eligible_orbits = [orbit for orbit in all_orbits if not _is_anchor_orbit(orbit, (height, width))]
    if not eligible_orbits:
        return None

    orbit_options: list[list[dict]] = []
    for orbit in eligible_orbits:
        options = _removal_options(orbit, true_kx, width)
        if options:
            orbit_options.append(options)
    if not orbit_options:
        return None

    strong_options = [opt for opts in orbit_options for opt in opts if opt["strong"]]
    triple_options = [opt for opts in orbit_options for opt in opts if opt["triple"]]
    if require_strong and not strong_options:
        return None
    if require_triple and not triple_options:
        return None

    targets = list(range(lo, hi + 1))
    random.shuffle(targets)
    for target in targets:
        for _ in range(REMOVAL_PLAN_RETRIES):
            plan: list[dict] = []
            used_orbits: set[int] = set()
            remaining = int(target)
            strong_used = False
            triple_used = False

            mandatory_choices: list[tuple[str, list[tuple[int, dict]]]] = []
            if require_triple:
                candidates = [
                    (idx, opt)
                    for idx, opts in enumerate(orbit_options)
                    for opt in opts
                    if opt["triple"] and opt["removed_count"] <= remaining
                ]
                if not candidates:
                    break
                mandatory_choices.append(("triple", candidates))
            if require_strong and not require_triple:
                candidates = [
                    (idx, opt)
                    for idx, opts in enumerate(orbit_options)
                    for opt in opts
                    if opt["strong"] and opt["removed_count"] <= remaining
                ]
                if not candidates:
                    break
                mandatory_choices.append(("strong", candidates))

            feasible = True
            for kind, candidates in mandatory_choices:
                candidates = [(idx, opt) for idx, opt in candidates if idx not in used_orbits]
                if not candidates:
                    feasible = False
                    break
                idx, opt = random.choice(candidates)
                plan.append(opt)
                used_orbits.add(idx)
                remaining -= opt["removed_count"]
                strong_used |= bool(opt["strong"])
                triple_used |= bool(opt["triple"])
            if not feasible or remaining < 0:
                continue

            while remaining > 0:
                viable = [
                    (idx, opt)
                    for idx, opts in enumerate(orbit_options)
                    if idx not in used_orbits
                    for opt in opts
                    if opt["removed_count"] <= remaining
                ]
                if not viable:
                    feasible = False
                    break

                exact = [(idx, opt) for idx, opt in viable if opt["removed_count"] == remaining]
                pool = exact if exact else viable
                weights = []
                for _idx, opt in pool:
                    weight = 1.0
                    if opt["triple"]:
                        weight += 1.4
                    if opt["strong"]:
                        weight += 0.8
                    if opt["removed_count"] == 1:
                        weight += 0.2
                    weights.append(weight)
                idx, opt = random.choices(pool, weights=weights, k=1)[0]
                plan.append(opt)
                used_orbits.add(idx)
                remaining -= opt["removed_count"]
                strong_used |= bool(opt["strong"])
                triple_used |= bool(opt["triple"])

            if not feasible or remaining != 0:
                continue
            if require_strong and not strong_used:
                continue
            if require_triple and not triple_used:
                continue

            removed = sorted({cell for opt in plan for cell in opt["removed"]})
            stats = {
                "removed_count": len(removed),
                "strong_orbits": sum(1 for opt in plan if opt["strong"]),
                "triple_orbits": sum(1 for opt in plan if opt["triple"]),
                "max_removed_from_one_orbit": max(opt["removed_count"] for opt in plan) if plan else 0,
            }
            return removed, stats
    return None


def _remove_pixels(
    inp: np.ndarray,
    out: np.ndarray,
    spec: dict,
    *,
    require_strong: bool,
    require_triple: bool,
) -> dict | None:
    bg = int(get_bg(out))
    color = int(spec["color"])
    true_ky = int(spec["true_ky"])
    true_kx = int(spec["true_kx"])
    clean_mask = out == color
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count))
    if lo > hi:
        return None

    result = _choose_removals(
        clean_mask,
        true_ky,
        true_kx,
        lo,
        hi,
        require_strong=require_strong,
        require_triple=require_triple,
    )
    if result is None:
        return None
    removed, stats = result

    for r, c in removed:
        inp[int(r), int(c)] = int(bg)
    return stats


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(out))
    metrics = {
        "sprites": 0,
        "damaged_orbits": 0,
        "strong_damaged_orbits": 0,
        "triple_orbits": 0,
        "max_missing_per_orbit": 0,
        "filled_pixels": int(np.count_nonzero(out != inp)),
    }
    for color in colors_present(out, exclude={bg}):
        target_pts = {(int(r), int(c)) for r, c in np.argwhere(out == int(color))}
        observed_pts = {(int(r), int(c)) for r, c in np.argwhere(inp == int(color))}
        metrics["sprites"] += 1
        ky, kx, _ = _best_fourfold_axes(target_pts, *out.shape)
        seen: set[frozenset[tuple[int, int]]] = set()
        for r, c in target_pts:
            orbit = _orbit_cells(r, c, ky, kx, *out.shape)
            if orbit is None:
                continue
            sig = frozenset(orbit)
            if sig in seen:
                continue
            seen.add(sig)
            observed = orbit & observed_pts
            missing = len(orbit) - len(observed)
            if 0 < missing:
                metrics["damaged_orbits"] += 1
                metrics["max_missing_per_orbit"] = max(metrics["max_missing_per_orbit"], missing)
                if len(observed) == 1 and len(orbit) == 4:
                    metrics["triple_orbits"] += 1
                if _vertical_fill_orbit(set(observed), kx, out.shape[1]) != orbit:
                    metrics["strong_damaged_orbits"] += 1
    return metrics


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
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

    triple_candidates = [
        spec
        for spec in placed_specs
        if spec["clean_count"] >= 10
        and _choose_removals(
            out == int(spec["color"]),
            spec["true_ky"],
            spec["true_kx"],
            max(1, math.ceil(MISSING_FRAC_LO * spec["clean_count"])),
            min(MISSING_MAX, math.floor(MISSING_FRAC_HI * spec["clean_count"])),
            require_strong=True,
            require_triple=True,
        )
        is not None
    ]
    if not triple_candidates:
        return None
    triple_spec = random.choice(triple_candidates)

    inp = out.copy()
    removal_stats: dict[int, dict] = {}
    for spec in placed_specs:
        stats = _remove_pixels(
            inp,
            out,
            spec,
            require_strong=(spec is triple_spec),
            require_triple=(spec is triple_spec),
        )
        if stats is None:
            return None
        removal_stats[int(spec["color"])] = stats

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_vertical_fill(inp), out):
        return None

    metrics = _fusion_metrics(inp, out)
    if metrics["strong_damaged_orbits"] < 1 or metrics["triple_orbits"] < 1:
        return None
    metrics["required_color"] = int(triple_spec["color"])
    metrics["required_removed"] = int(removal_stats[int(triple_spec["color"])]["removed_count"])
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _metrics = result
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
