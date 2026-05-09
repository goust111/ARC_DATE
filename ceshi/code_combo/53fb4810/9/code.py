# Combo variant of 53fb4810 (index 9)
# combo_spec: D3/0+D5/0
# Name: 大瓦片间歇跳跃

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Base task: a small seed tile touching one arm of a plus repeats continuously
#      outward, filling every tile slot to the boundary.
#    - D3/0 changes the trajectory: some whole tile slots stay blank, so the visible
#      ray becomes tile-gap-tile-gap...
#    - D5/0 changes the source payload: the seed is no longer a tiny 1x2 / 2x2 clue
#      but a larger filled rectangle such as 3x2, 2x3, 3x3, or 4x3 with internal
#      multi-color texture.
#    - In this fused version, the demonstrated ray must therefore be read in units of
#      full large rectangles. Example: if a 3x3 seed is attached below a plus and the
#      global skip gap is 1, the output contains [3x3 tile][3x3 blank][3x3 tile]...
#      down the column. The difference from the base task is visible both in scale
#      (large textured blocks) and spacing (empty whole-tile slots).
#
# b) Complexity calibration
#    - Skip gaps are restricted to {1, 2}. This gives repeated evidence without turning
#      spacing inference into an unrestricted counting problem.
#    - Large seeds are restricted to cross-axis spans {2, 3} and along-axis spans
#      {2, 3, 4}, with a minimum area of 6 and at least 2 colors. This keeps the
#      motif visibly larger than the original while still fitting several copies in
#      one grid.
#    - One long demonstrated ray remains in the input and must show at least three
#      full visible copies, so the skip step is recoverable from repeated evidence.
#
# c) Hidden assumptions broken by the fusion
#    - Original `extend_repeating_tile(...)` assumes every successive tile slot is
#      occupied. That fails once blank slots are legal, so the solver must inspect
#      discrete tile-sized slots and infer a shared step length.
#    - Original small seeds did not need strong texture constraints. With large seeds,
#      a solid or row-constant block could visually collapse into an uninformative
#      band, so the generator enforces multi-color row/column variation.
#    - `mark_lab = label_mask(non_mask, connectivity=4)` still assumes the component
#      touching the plus is the seed itself. The generator guarantees that skipped
#      copies are disconnected from the seed, so this remains valid.
#    - The nozzle-alignment assertions still matter: although the tile is larger, its
#      cross-axis span must still exactly match the chosen arm tip of the plus.
#
# d) Placement legality
#    - The generator reserves the entire ray corridor, including skipped blank slots,
#      before committing an object. This prevents another plus or ray from intruding
#      into a blank gap and corrupting the solver's slot-by-slot reasoning.
#    - Plus sprites share one symmetric shape per grid, remain disjoint, and their
#      rays never overlap or touch another plus's protected neighborhood.
#
# e) Visual observability
#    - A human sees two simultaneous innovations in 2-4 examples: the seed itself is a
#      large textured rectangle, and the one long clue ray is dashed at tile granularity.
#    - Because the same skip gap is reused for every plus in a grid, the single long
#      clue is enough to explain how the compact seeds should extend.
#
# f) Visual cleanliness
#    - Reserved blank corridors keep the empty gaps genuinely empty instead of being
#      polluted by other objects. That preserves the dashed rhythm and keeps the
#      larger tiles visually separable.
#
# g) Post-change solver assumptions and generator support
#    - `main()` assumes at least one marker is a long demonstration ray whose full
#      tile slots are either exact tile copies or all background. The generator
#      constructs the input exactly that way.
#    - `main()` assumes the skip gap is global within a grid. The generator samples
#      one `skip_gap` per grid and uses it for every ray.
#    - `main()` assumes seed bboxes are rectangular repeating periods. The generator
#      uses filled rectangular seed components only.

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


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
    """Paint one tile copy, clipping at the boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    bg = int(bg)

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


def extend_intermittent_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    skip_gap: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a tile outward while leaving fixed-size blank tile slots between copies."""
    H, W = out.shape
    h, w = tile.shape
    step_slots = int(skip_gap) + 1
    slot_idx = int(step_slots)

    while True:
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        if not _slot_intersects_grid(H, W, sr, sc, h, w):
            break
        _paint_tile(out, tile, sr, sc, bg=int(bg), protect=protect)
        slot_idx += int(step_slots)


def _count_full_slots(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> int:
    """Count how many full tile slots fit from the seed slot onward."""
    h, w = tile.shape
    count = 0
    slot_idx = 0
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
    Inspect one candidate ray in whole tile slots.

    Returns:
    - inferred skip_gap if this marker is the demonstrated dashed ray, else None
    - number of visible exact tile copies on that ray
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
            continue
        if np.all(patch == int(bg)):
            continue
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


def _seed_matches_nozzle(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """Check that the seed cross-axis span matches the nozzle edge exactly."""
    r0, c0, r1, c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    pr0 = int(pr0)
    pc0 = int(pc0)
    pr1 = int(pr1)
    pc1 = int(pc1)

    if dc == 0:
        nozzle_row = pr1 if dr == 1 else pr0
        nozzle_cols = sorted(int(cc) for cc in range(pc0, pc1 + 1) if plus_mask[nozzle_row, cc])
        seed_cols = list(range(c0, c1 + 1))
        return seed_cols == nozzle_cols

    nozzle_col = pc1 if dc == 1 else pc0
    nozzle_rows = sorted(int(rr) for rr in range(pr0, pr1 + 1) if plus_mask[rr, nozzle_col])
    seed_rows = list(range(r0, r1 + 1))
    return seed_rows == nozzle_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each large marker rectangle away from its '+' anchor, but only on regularly spaced tile slots.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms several plus-like anchor sprites.
    - Each plus has exactly one adjacent rectangular seed marker whose cross-axis span matches the chosen arm-tip nozzle.
    - In this combo variant the seed is a larger multicolor rectangle, and exactly one plus already shows a longer dashed ray made of repeated whole-tile copies separated by blank tile-sized gaps.

    Rule:
    - Find every marker component touching a plus, infer the attachment side, and treat the component's bbox as the full repeating tile period.
    - Use the one long demonstrated ray to infer the shared jump spacing in whole tile slots.
    - For every seed, propagate the same large rectangle outward in that direction, painting only every fixed step and leaving the skipped slots as background.
    - Plus pixels remain protected and are never overwritten.

    Key insight:
    - The clue ray teaches two things at once: the motif is a large textured rectangle, and the repetition is not continuous but evenly dashed at tile granularity. Once the solver reads those two facts, every compact seed extends in the same rhythm.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    markers: list[dict[str, object]] = []
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

            direction = next(iter(dirs))
            if not _seed_matches_nozzle(plus_mask, plus_coords, (r0, c0, r1, c1), direction):
                continue

            markers.append(
                {
                    "tile": tile,
                    "r0": int(r0),
                    "c0": int(c0),
                    "direction": tuple(map(int, direction)),
                }
            )

    skip_gap = 1
    best_visible = -1
    for marker in markers:
        cand_gap, visible = infer_skip_gap_from_demo(
            g,
            marker["tile"],
            int(marker["r0"]),
            int(marker["c0"]),
            marker["direction"],
            bg=int(bg),
        )
        if cand_gap is None:
            continue
        if int(visible) > int(best_visible):
            skip_gap = int(cand_gap)
            best_visible = int(visible)

    for marker in markers:
        extend_intermittent_tile(
            out,
            marker["tile"],
            int(marker["r0"]),
            int(marker["c0"]),
            marker["direction"],
            skip_gap=int(skip_gap),
            bg=int(bg),
            protect={int(plus_col)},
        )

    return out


# --------------- generator (combo D3/0 + D5/0) ---------------

_H_RANGE = (18, 31)
_W_RANGE = (18, 28)
_N_PLUS_RANGE = (2, 5)
_SEED_ALONG_CHOICES = [2, 3, 4]
_SEED_MAX_COLORS = 4
_SEED_MIN_COLORS = 2
_SEED_ALLOWED_CROSS = {2, 3}
_SKIP_GAP_CHOICES = [1, 2]
_MIN_VISIBLE_COPIES = 2
_PRE_EXT_MIN_VISIBLE = 3
_PLACEMENT_TRIES = 180
_GEN_RETRIES = 600

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
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span of the chosen nozzle."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])

    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    """Reject seeds that visually collapse into a flat stripe or solid block."""
    if len(set(map(int, tile.flatten()))) < _SEED_MIN_COLORS:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    return True


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a large rectangular seed whose cross-axis matches the nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross not in _SEED_ALLOWED_CROSS:
        return None

    for _ in range(24):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        if cross == 2 and along == 2:
            along = 3

        dr, dc = direction
        th, tw = (along, cross) if dc == 0 else (cross, along)
        n_cells = int(th * tw)
        if n_cells < 6:
            continue

        max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
        if max_k < _SEED_MIN_COLORS:
            return None
        n_colors = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile):
            return tile

    return None


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Anchor the seed tile flush to the chosen nozzle."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        return (int(pr1 + 1), int(pc0 + s)) if dr == 1 else (int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1)) if dc == 1 else (int(pr0 + s), int(pc0 - tw))


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    skip_gap: int,
) -> dict[str, object]:
    """
    Build the intermittent large-tile ray layout.

    Returns:
    - cells: painted non-bg cells for all visible tile copies
    - lane_cells: every cell in every tile slot, including skipped slots
    - seed_r0 / seed_c0: top-left of the seed tile
    - visible_count: number of fully in-bounds visible copies
    """
    th, tw = tile.shape
    seed_r0, seed_c0 = _seed_origin(plus_bbox, plus_mask, direction, tile)
    step_slots = int(skip_gap) + 1

    cells: list[tuple[int, int, int]] = []
    lane_cells: set[tuple[int, int]] = set()
    visible_count = 0
    slot_idx = 0

    while True:
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), tile, direction, int(slot_idx))
        if not _slot_intersects_grid(H, W, sr, sc, th, tw):
            break

        full_slot = _slot_fully_in_bounds(H, W, sr, sc, th, tw)
        draw_slot = int(slot_idx) % int(step_slots) == 0

        for i in range(th):
            rr = int(sr + i)
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = int(sc + j)
                if not (0 <= cc < W):
                    continue
                lane_cells.add((int(rr), int(cc)))
                if draw_slot:
                    cells.append((int(rr), int(cc), int(tile[i, j])))

        if draw_slot and full_slot:
            visible_count += 1

        slot_idx += 1

    return {
        "cells": cells,
        "lane_cells": sorted(lane_cells),
        "seed_r0": int(seed_r0),
        "seed_c0": int(seed_c0),
        "visible_count": int(visible_count),
    }


def _plus_touches_existing_lane(
    plus_cells: list[tuple[int, int]],
    lane_occ: np.ndarray,
) -> bool:
    """Reject plus placements that would sit directly on or beside an existing ray corridor."""
    H, W = lane_occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and bool(lane_occ[int(rr), int(cc)]):
                return True
    return False


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    lane_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    skip_gap: int,
) -> dict[str, object] | None:
    """Place one plus together with its full intermittent large-tile ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(bool(occ[r, c]) for r, c in plus_cells):
            continue
        if any(bool(plus_occ[r, c]) for r, c in plus_cells):
            continue
        if _plus_touches_existing_lane(plus_cells, lane_occ):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
            if tile is None:
                continue

            proj = _project_tile_coords(
                H,
                W,
                (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                plus_shape,
                direction,
                tile,
                int(skip_gap),
            )
            if int(proj["visible_count"]) < _MIN_VISIBLE_COPIES:
                continue
            if any(bool(occ[r, c]) or bool(plus_occ[r, c]) for r, c in proj["lane_cells"]):
                continue

            for r, c in plus_cells:
                occ[int(r), int(c)] = True
                out[int(r), int(c)] = int(plus_color)
            for r, c in plus_cells:
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[int(rr), int(cc)] = True

            for r, c in proj["lane_cells"]:
                occ[int(r), int(c)] = True
                lane_occ[int(r), int(c)] = True
            for r, c, v in proj["cells"]:
                out[int(r), int(c)] = int(v)

            return {
                "tile": tile,
                "cells": proj["cells"],
                "seed_r0": int(proj["seed_r0"]),
                "seed_c0": int(proj["seed_c0"]),
                "visible_count": int(proj["visible_count"]),
                "direction": tuple(map(int, direction)),
                "skip_gap": int(skip_gap),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict[str, object]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one long dashed ray and erase all non-seed copies from the others."""
    candidates = [i for i, p in enumerate(pluses) if int(p["visible_count"]) >= _PRE_EXT_MIN_VISIBLE]
    if not candidates:
        raise ValueError("no sufficiently long demonstration ray")

    best_visible = max(int(pluses[i]["visible_count"]) for i in candidates)
    best = [i for i in candidates if int(pluses[i]["visible_count"]) == int(best_visible)]
    pre_idx = int(rng.choice(best))

    inp = out.copy()
    for i, plus in enumerate(pluses):
        if i == pre_idx:
            continue

        th, tw = plus["tile"].shape
        sr0 = int(plus["seed_r0"])
        sc0 = int(plus["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _ in plus["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    skip_gap = int(rng.choice(_SKIP_GAP_CHOICES))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    plus_occ = np.zeros((H, W), dtype=bool)
    lane_occ = np.zeros((H, W), dtype=bool)

    pluses: list[dict[str, object]] = []
    for _ in range(n_plus):
        info = _place_plus_and_ray(
            rng=rng,
            plus_shape=plus_shape,
            palette=palette,
            H=H,
            W=W,
            occ=occ,
            plus_occ=plus_occ,
            lane_occ=lane_occ,
            out=out,
            plus_color=plus_color,
            skip_gap=int(skip_gap),
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["visible_count"]) for p in pluses) < _PRE_EXT_MIN_VISIBLE:
        return None

    try:
        inp = _build_input_from_output(out, pluses, int(bg), rng)
    except ValueError:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
