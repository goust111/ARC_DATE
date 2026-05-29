# Combo variant of 53fb4810 (index 18)
# combo_spec: D3/0+D6/0+D9/0
# Name: 间歇跳跃遇障无线索

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Base task: a compact seed tile touches one side of a shared '+' anchor, and
#      the solver repeats that tile continuously away from the plus.
#    - D3/0 changes the trajectory: visible copies are not contiguous. Here the
#      rhythm is fixed to "paint one whole tile slot, leave one whole tile slot
#      blank, paint the next, ..." so the ray becomes a dashed sequence of tile
#      copies.
#    - D6/0 changes the endpoint: instead of stopping at the border, each ray has
#      a detached obstacle farther out on the same lane. The ray halts before the
#      first visible tile copy that would hit that obstacle.
#    - D9/0 removes the in-grid clue: the input shows only compact seeds plus the
#      detached blockers. No ray is pre-expanded.
#    - Small trace, horizontal 1x2 seed `[2 3]`, fixed skip-gap = 1:
#      original output slots: [23][23][23][23]...
#      fused output slots   : [23][bg][23][bg][23][obstacle]
#      fused input slots    : [23][bg][bg][bg][bg][obstacle]
#      Compared with the original, three stages differ together: no example ray in
#      the input, visible copies appear every other slot, and the sequence ends at
#      a blocker instead of the frame.
#
# b) Complexity calibration
#    - The jump rule is intentionally fixed to a single visible cadence
#      (alternating paint / blank). Without an in-grid clue, a per-instance jump
#      size would be underdetermined from one input alone.
#    - Obstacles use one dedicated color that never appears in seeds, and every ray
#      has one blocker. This makes the new stopping rule repeatedly visible across
#      training pairs instead of being a rare exception.
#    - Every ray keeps at least two visible copies in the output, and at least one
#      ray per generated sample is forced to have three or more visible copies so
#      the dashed repetition is visually recoverable.
#
# c) Hidden assumptions from the original solver that break here
#    - Original `extend_repeating_tile(...)` assumed every future tile slot should
#      be painted. The fused variant must skip every other slot and leave those
#      slots as background.
#    - Original endpoint logic assumed the border was the only stop condition. The
#      fused variant must stop when the next visible tile slot is already occupied.
#    - Original generator left one full clue ray in the input. That disappears
#      here, so `main()` can no longer infer spacing from the input and instead
#      uses the task-level fixed alternating cadence.
#    - Original `label_mask(non_mask)` was safe because only the compact seeds
#      mattered. This remains safe here because the generator guarantees blockers
#      stay detached; `main()` only trusts components that are 4-neighbor-adjacent
#      to a plus.
#
# d) Placement legality guarantees in the generator
#    - The generator is still output-first. It chooses plus positions, seed tiles,
#      dashed ray lengths, and obstacle positions before erasing hidden copies.
#    - It reserves the entire lane from the seed through the obstacle slot,
#      including blank skipped slots, so no other plus, lane, or obstacle can
#      intrude into the invisible spacing that the solver relies on.
#    - Obstacles are placed on the first stripe of the next visible tile slot, so
#      the solver's "stop on first occupied target slot" rule matches generation
#      exactly.
#
# e) Visual observability
#    - Across 2-4 training pairs, a human can see that outputs add dashed rays,
#      not solid rays, because copies reappear with one blank tile-sized interval
#      between them.
#    - The detached same-color blockers explain why the dashed rays terminate early
#      instead of reaching the border.
#    - The absence of any pre-expanded input ray is itself directly visible: every
#      input only contains compact seeds, so the rule must be learned from the
#      training examples rather than from one special clue line.
#
# f) Visual cleanliness
#    - Lanes are pairwise disjoint and fully reserved, pluses have a one-cell
#      safety margin, and blockers are thin aligned bars rather than noisy blobs.
#    - This keeps the three roles visually separated: plus anchors, compact seeds,
#      and detached blockers.
#
# g) Color reasoning
#    - No hidden ordering by color id is used. Seed colors repeat unchanged.
#    - The blocker color is dedicated and globally shared, but the solver does not
#      need its numeric identity; blockers are recognized operationally because
#      they are occupied cells that the next visible copy would hit.
#
# h) New solver assumptions audited against the generator
#    - `main()` assumes each plus has exactly one adjacent compact seed component.
#      The generator enforces one seed per plus and keeps all blockers detached.
#    - `main()` assumes seed bbox cross-axis span matches the chosen nozzle. The
#      generator builds seeds from the nozzle span exactly.
#    - `main()` assumes previously drawn rays never block unrelated rays. The
#      generator reserves full dashed lanes to keep them disjoint.
#    - `main()` assumes the alternating cadence is task-level fixed. The generator
#      uses the same fixed cadence in every sample of this combo.

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


_SKIP_GAP = 1
_STEP_SLOTS = _SKIP_GAP + 1


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper: choose the color that has the most "big" components.

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


def _adjacent_component_ids(mark_lab: np.ndarray, plus_mask: np.ndarray) -> set[int]:
    """Return payload component ids that are 4-neighbor-adjacent to one plus."""
    H, W = mark_lab.shape
    adj_ids: set[int] = set()
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                adj_ids.add(int(mark_lab[rr, cc]))
    return adj_ids


def _marker_direction_and_alignment(
    pts: np.ndarray,
    plus_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    """Return ((dr,dc), bbox) if a payload component is a valid seed for this plus."""
    if pts.size == 0:
        return None

    H, W = plus_mask.shape
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
        return None

    r0, c0, r1, c1 = bbox_coords(pts)
    dr, dc = next(iter(dirs))
    plus_coords = np.argwhere(plus_mask)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        nozzle_cols = sorted(int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc])
        seed_cols = list(range(int(c0), int(c1) + 1))
        if seed_cols != nozzle_cols:
            return None
    else:
        nozzle_col = int(pc1) if dc == 1 else int(pc0)
        nozzle_rows = sorted(int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col])
        seed_rows = list(range(int(r0), int(r1) + 1))
        if seed_rows != nozzle_rows:
            return None

    return (int(dr), int(dc)), (int(r0), int(c0), int(r1), int(c1))


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
    """Paint one full tile copy."""
    protect = set() if protect is None else set(map(int, protect))
    h, w = tile.shape
    for i in range(h):
        for j in range(w):
            v = int(tile[i, j])
            if v == int(bg):
                continue
            rr = int(r0 + i)
            cc = int(c0 + j)
            if int(out[rr, cc]) in protect:
                continue
            out[rr, cc] = int(v)


def extend_intermittent_tile_until_blocked(
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
    """Paint the tile into every (skip_gap+1)-th slot until the next visible slot is occupied."""
    H, W = out.shape
    h, w = tile.shape
    step_slots = int(skip_gap) + 1
    slot_idx = int(step_slots)

    while True:
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, h, w):
            break

        blocked = False
        for i in range(h):
            for j in range(w):
                v = int(tile[i, j])
                if v == int(bg):
                    continue
                if int(out[int(sr + i), int(sc + j)]) != int(bg):
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            break

        _paint_tile(out, tile, sr, sc, bg=int(bg), protect=protect)
        slot_idx += int(step_slots)


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return (start, end) inclusive of nozzle cells along the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
    *,
    max_cross: int,
    seed_along_choices: list[int],
    seed_max_colors: int,
) -> np.ndarray | None:
    """Build a seed tile whose cross-axis exactly matches the nozzle span."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > int(max_cross):
        return None

    along = int(rng.choice(seed_along_choices))
    if along == 1 and cross == 1:
        along = 2

    _dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)
    max_k = min(int(seed_max_colors), n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Anchor the compact seed flush against the chosen nozzle."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        return (int(pr1 + 1), int(pc0 + s)) if dr == 1 else (int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1)) if dc == 1 else (int(pr0 + s), int(pc0 - tw))


def _count_full_slots(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> int:
    """Count consecutive full tile slots from the seed slot onward."""
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


def _obstacle_cells_for_slot(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    slot_idx: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """Place a one-stripe blocker on the leading face of a candidate visible slot."""
    dr, dc = direction
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    if dr == 1:
        row = int(seed_r0 + slot_idx * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dr == -1:
        row = int(seed_r0 - 1 - (slot_idx - 1) * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dc == 1:
        col = int(seed_c0 + slot_idx * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    else:
        col = int(seed_c0 - 1 - (slot_idx - 1) * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    return cells


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    """Whether every listed cell coordinate lies inside the grid."""
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _build_ray_layout(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    visible_copies: int,
    obstacle_color: int,
) -> dict | None:
    """
    Build one dashed ray layout.

    Returns:
    - paint_cells: all visible tile copies, including the seed
    - lane_cells: every reserved cell from the seed slot through the obstacle slot
    - obstacle_cells: one-stripe barrier at the next visible slot
    """
    seed_r0, seed_c0 = _seed_origin(plus_bbox, plus_mask, direction, tile)
    max_full_slots = _count_full_slots(H, W, tile, seed_r0, seed_c0, direction)
    obstacle_slot_idx = int(visible_copies) * int(_STEP_SLOTS)
    if int(visible_copies) < 2 or obstacle_slot_idx >= int(max_full_slots):
        return None

    th, tw = tile.shape
    paint_cells: list[tuple[int, int, int]] = []
    lane_cells: set[tuple[int, int]] = set()

    for slot_idx in range(0, obstacle_slot_idx + 1):
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), tile, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, th, tw):
            return None
        for i in range(th):
            for j in range(tw):
                rr = int(sr + i)
                cc = int(sc + j)
                lane_cells.add((rr, cc))
                if int(slot_idx) % int(_STEP_SLOTS) == 0 and int(slot_idx) < int(obstacle_slot_idx):
                    paint_cells.append((rr, cc, int(tile[i, j])))

    obstacle_cells = _obstacle_cells_for_slot(
        int(seed_r0),
        int(seed_c0),
        direction,
        tile,
        int(obstacle_slot_idx),
        int(obstacle_color),
    )
    if not _all_in_bounds(H, W, obstacle_cells):
        return None

    return {
        "paint_cells": paint_cells,
        "lane_cells": sorted(lane_cells),
        "obstacle_cells": obstacle_cells,
        "seed_r0": int(seed_r0),
        "seed_c0": int(seed_c0),
        "visible_copies": int(visible_copies),
        "obstacle_slot_idx": int(obstacle_slot_idx),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend compact marker tiles away from '+' anchors as dashed rays that stop at blockers.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms several identical '+'-like anchors.
    - Each '+' has exactly one compact adjacent seed tile whose bbox span matches that arm's nozzle.
    - No ray is pre-expanded in the input; only detached obstacle bars sit farther out on the future lanes.

    Rule:
    - For each seed touching a '+', use the touching side to determine the outward direction and use the seed bbox as the repeating tile.
    - Repaint that tile into every other full tile slot away from the '+', leaving one whole tile slot blank between visible copies.
    - Stop when the next visible tile slot would overlap any existing non-background cell; keep that blocker unchanged.
    - Never overwrite the '+' color.

    Key insight:
    - With the example ray removed, the only local geometric cue is which side of the '+' the compact seed touches; the training pairs teach that this seed grows in a fixed dashed rhythm and that detached blockers cut the dashed ray short.
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

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        parsed = _marker_direction_and_alignment(pts, plus_mask)
        if parsed is None:
            continue
        (dr, dc), (r0, c0, r1, c1) = parsed
        tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

        extend_intermittent_tile_until_blocked(
            out,
            tile,
            int(r0),
            int(c0),
            (int(dr), int(dc)),
            skip_gap=int(_SKIP_GAP),
            bg=int(bg),
            protect={int(plus_col)},
        )

    return out


# --------------- generator (output-first, dashed + obstacle + no clue) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 29)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_VISIBLE_COPIES = 2
_IDEAL_VISIBLE_COPIES = 3
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 600

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
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


def _sample_direction_plan(rng: np.random.Generator, n_plus: int) -> list[tuple[int, int]]:
    """
    Choose a per-grid direction plan with explicit branch coverage.

    Every generated puzzle contains at least one vertical and one horizontal ray.
    When three or more pluses are present, prefer three distinct directions so the
    dataset does not collapse to only one sign on an axis.
    """
    vertical = [(1, 0), (-1, 0)]
    horizontal = [(0, 1), (0, -1)]
    rng.shuffle(vertical)
    rng.shuffle(horizontal)

    plan = [vertical[0], horizontal[0]]
    remaining = [d for d in _DIRECTIONS if d not in plan]
    rng.shuffle(remaining)
    while len(plan) < int(n_plus):
        plan.append(remaining.pop(0) if remaining else plan[int(rng.integers(0, len(plan)))])
    rng.shuffle(plan)
    return [(int(dr), int(dc)) for dr, dc in plan]


def _place_plus_seed_and_obstacle(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    *,
    directions_to_try: list[tuple[int, int]],
    plus_color: int,
    obstacle_color: int,
) -> dict | None:
    """Try to place one plus, its dashed output ray, and the detached blocker."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        for direction in directions_to_try:
            tile = _build_seed_tile(
                plus_shape,
                direction,
                palette,
                rng,
                max_cross=int(_SEED_MAX_CROSS),
                seed_along_choices=_SEED_ALONG_CHOICES,
                seed_max_colors=int(_SEED_MAX_COLORS),
            )
            if tile is None:
                continue

            seed_r0, seed_c0 = _seed_origin(
                (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                plus_shape,
                direction,
                tile,
            )
            max_full_slots = _count_full_slots(H, W, tile, seed_r0, seed_c0, direction)
            max_visible = (int(max_full_slots) - 1) // int(_STEP_SLOTS)
            if max_visible < int(_MIN_VISIBLE_COPIES):
                continue

            candidates = list(range(int(_MIN_VISIBLE_COPIES), int(max_visible) + 1))
            candidates.sort(reverse=True)
            for visible_copies in candidates:
                layout = _build_ray_layout(
                    H,
                    W,
                    (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                    plus_shape,
                    direction,
                    tile,
                    int(visible_copies),
                    int(obstacle_color),
                )
                if layout is None:
                    continue

                if any(int(occ[rr, cc]) != 0 or int(plus_occ[rr, cc]) == 1 for rr, cc in layout["lane_cells"]):
                    continue
                if any(int(occ[r, c]) != 0 or int(plus_occ[r, c]) == 1 for r, c, _ in layout["obstacle_cells"]):
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

                for rr, cc in layout["lane_cells"]:
                    occ[int(rr), int(cc)] = 1
                for rr, cc, v in layout["paint_cells"]:
                    out[int(rr), int(cc)] = int(v)
                for rr, cc, v in layout["obstacle_cells"]:
                    occ[int(rr), int(cc)] = 1
                    out[int(rr), int(cc)] = int(v)

                return {
                    "tile": tile,
                    "paint_cells": layout["paint_cells"],
                    "obstacle_cells": layout["obstacle_cells"],
                    "seed_r0": int(layout["seed_r0"]),
                    "seed_c0": int(layout["seed_c0"]),
                    "direction": (int(direction[0]), int(direction[1])),
                    "visible_copies": int(layout["visible_copies"]),
                }

    return None


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every non-seed tile copy so the input keeps only compact seeds plus blockers."""
    inp = out.copy()
    for p in pluses:
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + i), int(sc0 + j)) for i in range(th) for j in range(tw)}
        for r, c, _ in p["paint_cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(
    rng: np.random.Generator,
    *,
    return_meta: bool = False,
) -> np.ndarray | dict | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {plus_color, obstacle_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    direction_plan = _sample_direction_plan(rng, n_plus)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for direction in direction_plan:
        info = _place_plus_seed_and_obstacle(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            directions_to_try=[direction],
            plus_color=int(plus_color),
            obstacle_color=int(obstacle_color),
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["visible_copies"]) for p in pluses) < int(_IDEAL_VISIBLE_COPIES):
        return None

    inp = _build_input_from_output(out, pluses, int(bg))

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    if return_meta:
        return {
            "input": inp,
            "output": out,
            "bg": int(bg),
            "plus_color": int(plus_color),
            "obstacle_color": int(obstacle_color),
            "pluses": pluses,
            "skip_gap": int(_SKIP_GAP),
        }
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
