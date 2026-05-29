# Combo variant of 53fb4810 (index 5)
# combo_spec: D3/0+D6/1
# Name: 跳跃平铺距离编码

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Base task: a seed tile attached to one arm of a shared-color plus repeats
#      contiguously until the boundary.
#    - Fusion rule: two things change together.
#      1. Trajectory is dashed: after each visible tile copy, a fixed number of
#         whole-tile slots stay blank before the next copy appears.
#      2. The ray no longer runs to the border: the total number of visible tile
#         copies equals the number of distinct non-background colors in the seed.
#    - Example: if a horizontal seed is [[2, 4], [4, 7]], then its color code is
#      3. With skip_gap = 1 the occupied slots are 0, 2, 4, so the full ray is
#      [seed][blank][copy][blank][copy] and then it stops, even if more room
#      remains before the border.
#    - Difference from the base task: the solver must infer both the spacing
#      between copies and the stopping rule; either one alone is insufficient.
#
# b) Complexity calibration
#    - skip_gap is restricted to {1, 2}. That keeps the dashed rhythm visually
#      obvious and repeated across all pluses in one grid.
#    - The distance code is restricted to 2-4 via the number of distinct seed
#      colors. This gives multiple endpoint lengths without introducing a large
#      symbolic mapping.
#    - Plus shapes are limited to nozzle-width-2 templates so 2x2 seeds can
#      appear; otherwise the distance code would collapse to a constant too often.
#
# c) Hidden assumptions from the original solver that break here
#    - Original extension assumed every outward slot is occupied. That fails
#      because skipped slots must stay background.
#    - Original clue could be treated as a long connected component. That is no
#      longer true: the demonstration ray is disconnected, so only the seed copy
#      touches the plus and detached copies must be read slot-by-slot.
#    - Original stop condition was "until boundary". That now breaks because the
#      solver must stop after an exact number of visible copies inferred from the
#      seed itself, not from available space.
#
# d) Placement legality guarantees in the generator
#    - Generation is still output-first.
#    - Each plus and seed are placed with the original nozzle-alignment rule.
#    - The generator reserves the entire tile-slot corridor from the seed to the
#      boundary, including skipped slots and trailing blank slots after the last
#      visible copy. This prevents another object from contaminating the solver's
#      spacing inference.
#    - Every occupied copy is fully in bounds; no clipped visible tiles appear.
#
# e) Visual observability
#    - The single preserved demonstration ray shows both new signals at once:
#      repeated blank intervals reveal skip_gap, and the fact that the ray stops
#      before the boundary reveals the endpoint is not border-based.
#    - The distance code uses distinct visible colors, which humans can count
#      directly without relying on numeric color ids.
#    - Each generated grid is required to contain at least two different encoded
#      lengths among its seeds, so the output teaches that stop distance varies
#      per marker rather than being one fixed constant.
#
# f) Visual cleanliness
#    - Rays remain axis-aligned, pluses share one color, and whole-lane
#      reservation prevents overlaps. Blank skipped slots are genuinely empty, so
#      the dashed structure and the stopping point stay easy to parse.
#
# g) Change inferability
#    - No color remapping is introduced. Color only serves as a countable
#      property of each seed. Because the same tile is copied verbatim along the
#      ray, the relation "distinct seed colors -> number of visible copies" is
#      structurally supported in the training pairs.
#
# h) New solver assumptions and generator support
#    - main() assumes exactly one compact seed component per plus is adjacent to
#      that plus. Detached clue copies never touch the plus because skip_gap >= 1;
#      the generator enforces that.
#    - main() assumes every full slot after the seed is either an exact tile copy
#      or pure background on the demonstration lane. Whole-lane reservation
#      enforces that.
#    - main() assumes one global skip_gap shared across all pluses in a grid, but
#      a per-seed endpoint determined by that seed's color diversity. The
#      generator samples exactly that regime.


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper (only used by this solver): choose a color that has the most "big" components.

    Big component = connected component whose bbox is at least (min_h x min_w).
    Tie-break: total area of big components (larger wins).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def encoded_ray_units(tile: np.ndarray, *, bg: int) -> int:
    """Distance code: total visible tile copies = number of distinct non-bg colors."""
    cols = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    return max(1, int(len(cols)))


def _slot_origin(
    r0: int,
    c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
    slot_idx: int,
) -> tuple[int, int]:
    """Top-left corner of the slot_idx-th tile slot measured from the seed slot."""
    h, w = tile.shape
    dr, dc = map(int, direction)
    if dc == 0:
        return int(r0 + slot_idx * dr * h), int(c0)
    return int(r0), int(c0 + slot_idx * dc * w)


def _slot_intersects_grid(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    """Whether the tile bbox intersects the grid at all."""
    return not (int(r0) >= int(H) or int(c0) >= int(W) or int(r0 + h) <= 0 or int(c0 + w) <= 0)


def _slot_fully_in_bounds(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    """Whether the whole tile bbox lies inside the grid."""
    return 0 <= int(r0) and 0 <= int(c0) and int(r0 + h) <= int(H) and int(c0 + w) <= int(W)


def _paint_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Paint one tile copy, clipped to the grid."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    bg = int(bg)
    h, w = tile.shape

    for i in range(h):
        rr = int(r0 + i)
        if not (0 <= rr < H):
            continue
        for j in range(w):
            cc = int(c0 + j)
            if not (0 <= cc < W):
                continue
            v = int(tile[i, j])
            if v == bg:
                continue
            if int(out[rr, cc]) in protect:
                continue
            out[rr, cc] = int(v)


def _count_full_slots(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> int:
    """Count how many whole tile slots fit from the seed slot onward."""
    h, w = tile.shape
    slot_idx = 0
    count = 0
    while True:
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, h, w):
            break
        count += 1
        slot_idx += 1
    return int(count)


def infer_skip_gap_from_demo(
    grid: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> tuple[int | None, int]:
    """
    Inspect full tile slots on one candidate lane.

    Returns:
    - inferred skip_gap if this seed is the preserved demonstration ray, else None
    - number of visible copies seen on that lane
    """
    H, W = grid.shape
    h, w = tile.shape
    max_full_slots = _count_full_slots(H, W, tile, int(r0), int(c0), direction)
    occupied_slots = [0]

    for slot_idx in range(1, int(max_full_slots)):
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        patch = grid[int(sr) : int(sr + h), int(sc) : int(sc + w)]
        if np.array_equal(patch, tile):
            occupied_slots.append(int(slot_idx))
        elif np.all(patch == int(bg)):
            continue
        else:
            return None, int(len(occupied_slots))

    if len(occupied_slots) < 3:
        return None, int(len(occupied_slots))

    diffs = [b - a for a, b in zip(occupied_slots, occupied_slots[1:])]
    if len(set(map(int, diffs))) != 1:
        return None, int(len(occupied_slots))

    step_slots = int(diffs[0])
    if step_slots < 2:
        return None, int(len(occupied_slots))

    return int(step_slots - 1), int(len(occupied_slots))


def extend_intermittent_tile_exact(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    total_units: int,
    skip_gap: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Paint exact-length dashed copies of a tile, excluding the seed slot itself."""
    step_slots = int(skip_gap) + 1
    for copy_idx in range(1, int(total_units)):
        slot_idx = int(copy_idx * step_slots)
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        _paint_tile(out, tile, sr, sc, bg=int(bg), protect=protect)


def _extract_markers(grid: np.ndarray) -> tuple[np.ndarray, int | None, list[dict]]:
    """Find plus anchors and their adjacent seed tiles."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return g, None, []

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    markers: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

            dirs: set[tuple[int, int]] = set()
            for pr, pc in pts:
                pr = int(pr)
                pc = int(pc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = pr + dr
                    cc = pc + dc
                    if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                        dirs.add((-int(dr), -int(dc)))
            if len(dirs) != 1:
                continue

            dr, dc = next(iter(dirs))
            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)

            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                seed_cols = list(range(int(c0), int(c1) + 1))
                if seed_cols != nozzle_cols:
                    continue
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                seed_rows = list(range(int(r0), int(r1) + 1))
                if seed_rows != nozzle_rows:
                    continue

            markers.append(
                {
                    "tile": tile,
                    "r0": int(r0),
                    "c0": int(c0),
                    "direction": (int(dr), int(dc)),
                }
            )

    return g, int(plus_col), markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles away from '+' anchors as dashed rays whose spacing is shared but whose lengths are seed-encoded.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms several '+' sprites.
    - Each '+' has exactly one compact seed tile attached flush to one arm, with the seed span matching that arm's nozzle width.
    - Exactly one seed is already shown as a finished dashed ray: identical tile copies recur after blank tile-sized jumps and the ray stops before the boundary.

    Rule:
    - For every seed touching a '+', use the touching side to determine the outward ray direction.
    - Read the preserved dashed ray to infer the global jump spacing in whole-tile slots.
    - For each seed, count its distinct non-background colors; that count is the total number of visible tile copies the ray should contain, including the seed.
    - Repaint the seed along its ray using the shared jump spacing, but stop after that exact number of copies. '+' pixels stay protected.

    Key insight: The demonstration ray teaches the shared rhythm of occupied-slot then skipped-slot, while each seed's own color diversity teaches how many times that rhythm should place the tile before stopping.
    """
    g, plus_col, markers = _extract_markers(grid)
    out = g.copy()
    if plus_col is None:
        return out

    bg = int(get_bg(g))
    skip_gap = 1
    best_visible = -1

    for m in markers:
        cand_gap, visible = infer_skip_gap_from_demo(
            g,
            m["tile"],
            int(m["r0"]),
            int(m["c0"]),
            m["direction"],
            bg=int(bg),
        )
        if cand_gap is None:
            continue
        if int(visible) > int(best_visible):
            skip_gap = int(cand_gap)
            best_visible = int(visible)

    for m in markers:
        total_units = encoded_ray_units(m["tile"], bg=int(bg))
        extend_intermittent_tile_exact(
            out,
            m["tile"],
            int(m["r0"]),
            int(m["c0"]),
            m["direction"],
            total_units=int(total_units),
            skip_gap=int(skip_gap),
            bg=int(bg),
            protect={int(plus_col)},
        )

    return out


# --------------- generator (combo D3/0 + D6/1) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_SKIP_GAP_CHOICES = [1, 2]
_PLACEMENT_TRIES = 140
_GEN_RETRIES = 500
_PRECLUE_MIN_UNITS = 3
_MIN_TRAILING_BLANK_SLOTS = 1
_FORCED_GAP_RETRIES = 320

_PLUS_SHAPES = [
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_GEN_CALL_COUNT = 0


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive nozzle span on the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Anchor the seed tile flush against the chosen nozzle."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        return (int(pr1 + 1), int(pc0 + s)) if dr == 1 else (int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1)) if dc == 1 else (int(pr0 + s), int(pc0 - tw))


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int] | None:
    """Build a valid seed tile and return it together with its encoded length."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k < 2:
        return None
    n_colors = int(rng.integers(2, max_k + 1))

    chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
    vals = chosen.copy()
    if n_cells > n_colors:
        vals.extend(list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True))))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(th, tw)
    return tile, int(n_colors)


def _project_combo_ray(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    skip_gap: int,
    units_total: int,
) -> dict | None:
    """Build the exact dashed ray and reserve the whole lane to the boundary."""
    th, tw = tile.shape
    seed_r0, seed_c0 = _seed_origin(plus_bbox, plus_mask, direction, tile)
    if not _slot_fully_in_bounds(H, W, seed_r0, seed_c0, th, tw):
        return None

    max_full_slots = _count_full_slots(H, W, tile, int(seed_r0), int(seed_c0), direction)
    step_slots = int(skip_gap) + 1
    last_visible_slot = int((units_total - 1) * step_slots)
    if last_visible_slot >= int(max_full_slots):
        return None

    next_visible_r0, next_visible_c0 = _slot_origin(
        int(seed_r0), int(seed_c0), tile, direction, int(units_total * step_slots)
    )
    next_visible_intersects = _slot_intersects_grid(H, W, next_visible_r0, next_visible_c0, th, tw)

    occupied_slots = {int(copy_idx * step_slots) for copy_idx in range(int(units_total))}
    cells: list[tuple[int, int, int]] = []
    lane_cells: set[tuple[int, int]] = set()
    slot_idx = 0

    while True:
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), tile, direction, int(slot_idx))
        if not _slot_intersects_grid(H, W, sr, sc, th, tw):
            break
        for i in range(th):
            rr = int(sr + i)
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = int(sc + j)
                if not (0 <= cc < W):
                    continue
                lane_cells.add((int(rr), int(cc)))
                if int(slot_idx) in occupied_slots:
                    cells.append((int(rr), int(cc), int(tile[i, j])))
        slot_idx += 1

    return {
        "cells": cells,
        "lane_cells": sorted(lane_cells),
        "seed_r0": int(seed_r0),
        "seed_c0": int(seed_c0),
        "units_total": int(units_total),
        "max_full_slots": int(max_full_slots),
        "trailing_blank_slots": int(max_full_slots - last_visible_slot - 1),
        "next_visible_intersects": bool(next_visible_intersects),
    }


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    skip_gap: int,
) -> dict | None:
    """Try to place one plus sprite with one legal dashed exact-length ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) != 0
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
            built = _build_seed_tile(plus_shape, d, palette, rng)
            if built is None:
                continue
            tile, units_total = built
            proj = _project_combo_ray(
                H,
                W,
                (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                plus_shape,
                d,
                tile,
                int(skip_gap),
                int(units_total),
            )
            if proj is None:
                continue
            if any(int(occ[r, c]) != 0 or int(plus_occ[r, c]) == 1 for r, c in proj["lane_cells"]):
                continue

            for r, c in plus_cells:
                occ[int(r), int(c)] = 1
                out[int(r), int(c)] = int(plus_color)
            for r, c in plus_cells:
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[int(rr), int(cc)] = 1

            for r, c in proj["lane_cells"]:
                occ[int(r), int(c)] = 1
            for r, c, v in proj["cells"]:
                out[int(r), int(c)] = int(v)

            return {
                "tile": tile,
                "cells": proj["cells"],
                "seed_r0": int(proj["seed_r0"]),
                "seed_c0": int(proj["seed_c0"]),
                "units_total": int(proj["units_total"]),
                "trailing_blank_slots": int(proj["trailing_blank_slots"]),
                "next_visible_intersects": bool(proj["next_visible_intersects"]),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Keep one full dashed clue ray and erase all non-seed copies from the rest."""
    candidates = [
        i
        for i, p in enumerate(pluses)
        if int(p["units_total"]) >= _PRECLUE_MIN_UNITS
        and int(p["trailing_blank_slots"]) >= _MIN_TRAILING_BLANK_SLOTS
        and bool(p["next_visible_intersects"])
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda i: (int(pluses[i]["units_total"]), int(pluses[i]["trailing_blank_slots"])))
    if len(candidates) > 1 and bool(rng.integers(0, 2)):
        best_units = max(int(pluses[i]["units_total"]) for i in candidates)
        best_pool = [i for i in candidates if int(pluses[i]["units_total"]) == int(best_units)]
        best = int(rng.choice(best_pool))
    pre_idx = int(best)

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator, forced_skip_gap: int | None = None) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    if forced_skip_gap is None:
        skip_gap = int(rng.choice(_SKIP_GAP_CHOICES))
    else:
        skip_gap = int(forced_skip_gap)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(n_plus):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
            int(skip_gap),
        )
        if info is None:
            return None
        pluses.append(info)

    if len({int(p["units_total"]) for p in pluses}) < 2:
        return None
    if sum(int(p["trailing_blank_slots"]) >= _MIN_TRAILING_BLANK_SLOTS for p in pluses) < 2:
        return None

    inp = _build_input_from_output(out, pluses, int(bg), rng)
    if inp is None:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    rng = np.random.default_rng()
    preferred_gap = int(_SKIP_GAP_CHOICES[_GEN_CALL_COUNT % len(_SKIP_GAP_CHOICES)])

    for _ in range(_FORCED_GAP_RETRIES):
        g = _gen_attempt(rng, forced_skip_gap=int(preferred_gap))
        if g is not None:
            _GEN_CALL_COUNT += 1
            return g

    for _ in range(max(0, _GEN_RETRIES - _FORCED_GAP_RETRIES)):
        g = _gen_attempt(rng)
        if g is not None:
            _GEN_CALL_COUNT += 1
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
