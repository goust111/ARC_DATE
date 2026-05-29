# Variant of 9aaea919
# Dimension: D13 — 最近车道匹配策略 (match_cond)
# Change: 红色车道的精灵转移到离它最近(按列中心距离)的绿色车道... → 最远车道匹配

# Design analysis for the D13 variant
# -----------------------------------
# a) Concrete behavior trace
#    - The canvas still has four fixed 5-column lanes. Each lane contains repeated 3x5 plus sprites
#      on the same period-4 vertical lattice, and bottom markers remain 1x5 bars on row 29.
#    - The visible change is not in sprite geometry but in which green-marked lane receives each red
#      lane's transferred sprite count. To make "farthest" observable, the generator now places two
#      green markers instead of one. Example with lane centers [5, 11, 17, 23]:
#      red at 11 and greens at 5 and 23. Original rule would choose 5 (distance 6), while this
#      variant must choose 23 (distance 12). The red lane still turns grey; only the recipient lane
#      changes.
#    - This affects both solver matching and generator role assignment. Output-first generation now
#      accumulates gains separately for each green lane according to farthest-center matching.
# b) Complexity calibration
#    - Exactly two green lanes are enough to expose nearest-vs-farthest without introducing a large
#      branching factor. More green lanes are unnecessary for this dimension and would add ambiguity.
#    - Red lane count remains 1-2, preserving the original combinatorial load and keeping examples
#      readable.
# c) Hidden-assumption audit
#    - The original solver code already supports multiple green markers; the old generator was the
#      only reason the nearest/farthest distinction was invisible.
#    - The line `best = min(...)` was the only direct dependency on the old D13 rule. It becomes
#      `max(...)`.
#    - Placement order into a shared destination lane remains harmless because all transferred
#      sprites keep the same shape and receive the destination lane color.
# d) Placement legality
#    - The generator rejects role assignments where any red lane is equidistant from the two green
#      lanes, so the farthest choice is unique and human-visible.
#    - It also rejects symmetric cases where nearest and farthest would give the same per-green gain
#      totals, because those cases would make the variant visually collapse back to the original task.
#    - It also checks each green lane's final sprite total against the original height budget
#      (`<= 7`), ensuring all stacked sprites fit the existing lattice.
# e) Visual observability
#    - Every generated sample has two green markers, so the solver cannot collapse back to the
#      original single-target behavior.
#    - Because each red lane sees one nearer and one farther green lane, the new rule changes which
#      lane grows in the output; this is visible without using color ids as hidden metadata.
# f) Visual cleanliness
#    - No object shapes or spacing were changed. Lanes stay separated by one empty column on each
#      side, markers stay isolated on the last row, and added sprites remain aligned to the same
#      vertical lattice.
# g) Color inference
#    - Color behavior is still local and structural: red markers identify sources, green markers
#      identify candidate destinations, grey marks emptied source lanes, and transferred sprites use
#      the destination lane's own sprite color. No new color permutation is introduced.
# h) Variant-validation target
#    - A useful quantitative check is the divergence rate from the original nearest solver on inputs
#      produced by this generator. It should be high, ideally 100%, proving the samples genuinely
#      exercise the farthest-match branch.
# i) New solver assumptions and generator guarantees
#    - `main()` now assumes at least two valid green lanes if the variant is to be distinguishable.
#      The generator guarantees exactly two.
#    - `main()` still assumes marker bars and sprites are non-overlapping thin/compact components on
#      the original lattice. The generator preserves that original layout exactly.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are thin 4-connected components of the given marker color.
    """
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _lane_center_from_cols(c0: int, c1: int) -> float:
    return (c0 + c1) / 2.0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-tagged lanes into the most distant green-tagged lane, then restack them.

    Input structure:
    - The background is the dominant color; thin 1x5 bars in red (2) and green (3) sit on the last row.
    - Each marked or unmarked lane is a fixed-width column band containing repeated monochrome plus sprites on a regular vertical lattice.

    Rule:
    - Sprites in every red lane become grey (5), and their counts are transferred to the green lane whose column center is farthest from that red lane.
    - Each transferred sprite uses the source shape shifted into the chosen destination lane and recolored to that destination lane's sprite color, filling empty lattice slots from the bottom upward.
    - All red and green marker bars are erased after the transfer.

    Key insight: The important cue is not "which lane is green" but "which green lane is farther away" along the horizontal axis; sprite counts are conserved while source lanes empty into the farthest candidate destination.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_cols(g, dst_cols, bg)
        dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
        dst_comps = [pts for pts in dst_comps if pts]
        if not dst_comps:
            continue

        dst_sz = max(len(p) for p in dst_comps if len(p) != len(gm))
        dst_sprites = [p for p in dst_comps if len(p) == dst_sz]
        if not dst_sprites:
            continue

        dst_sprites_sorted = []
        for pts in dst_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            dst_sprites_sorted.append((rmin, cmin, pts))
        dst_sprites_sorted.sort()
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        gc = _lane_center_from_cols(dst_c0, dst_c1)
        green_state[id(gm)] = {
            "marker": gm,
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
        src_set = set(rm)
        src_comps = _single_color_components_in_cols(g, src_cols, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        if not src_comps:
            continue

        src_sz = max(len(p) for p in src_comps if len(p) != len(rm))
        src_sprites = [p for p in src_comps if len(p) == src_sz]
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        rc = _lane_center_from_cols(src_c0, src_c1)
        best = max(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = []
        for pts in src_sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            src_sprites_sorted.append((rmin, cmin, pts))
        src_sprites_sorted.sort()
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_top = best["bottom0"] - k * best["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in best["occupied"]:
                continue
            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            for dr, c in src_offsets:
                out[new_top + dr, c] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_GREEN_COUNT = 2
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _lane_center(lane_idx: int) -> float:
    c0 = _LANE_STARTS[lane_idx]
    return c0 + (_SPRITE_W - 1) / 2.0


def _green_gains_for_roles(
    counts: list[int], reds: tuple[int, ...], greens: tuple[int, ...]
) -> dict[int, int] | None:
    farthest_gains = {g: 0 for g in greens}
    nearest_gains = {g: 0 for g in greens}
    for red in reds:
        distances = [(abs(_lane_center(g) - _lane_center(red)), g) for g in greens]
        dist_values = [d for d, _g in distances]
        if len(set(dist_values)) != len(dist_values):
            return None
        farthest = max(distances)[1]
        nearest = min(distances)[1]
        farthest_gains[farthest] += counts[red]
        nearest_gains[nearest] += counts[red]
    if farthest_gains == nearest_gains:
        return None
    if any(counts[g] + farthest_gains[g] > _MAX_LANE_TOTAL for g in greens):
        return None
    return farthest_gains


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]]]:
    """Return feasible (reds, greens, gains) assignments grouped by red-lane count."""
    grouped: dict[int, list[tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]]] = {1: [], 2: []}
    for greens in combinations(range(_NUM_LANES), _GREEN_COUNT):
        others = [i for i in range(_NUM_LANES) if i not in greens]
        for red_count in (1, 2):
            if red_count > len(others):
                continue
            for reds in combinations(others, red_count):
                gains = _green_gains_for_roles(counts, reds, greens)
                if gains is not None:
                    grouped[red_count].append((reds, greens, gains))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, greens, gains = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    green_set = set(greens)

    output_counts = input_counts[:]
    for green, gain in gains.items():
        output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for green in greens:
        c0 = _LANE_STARTS[green]
        gain = gains[green]
        for r0 in _lane_tops(output_counts[green])[:gain]:
            inp[r0:r0 + _SPRITE_H, c0:c0 + _SPRITE_W] = bg
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    if len(green_set) != _GREEN_COUNT:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
