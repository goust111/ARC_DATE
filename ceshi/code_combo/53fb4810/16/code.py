# Combo variant of 53fb4810 (index 16)
# combo_spec: D1/0+D3/2+D6/0
# Name: 对角线镜像射线遇障停止

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior:
#    The original task launched axis-aligned periodic strips from seeds attached to
#    the arms of a '+'. In this combo, the seed instead sits in one diagonal corner
#    region of the plus bbox, so the launch direction is one of NW/NE/SW/SE. To
#    keep diagonal repetition clean, seeds are only 1x2 or 2x1. Each next unit is
#    shifted by exactly (dr,dc) = (±1,±1), producing a zipper-like diagonal chain.
#    Repeated units alternate between the original seed and its mirror along the
#    seed's long axis, so a horizontal seed [a b] becomes [a b], [b a], [a b], ...
#    Finally, a dedicated obstacle color occupies the footprint of the next diagonal
#    unit, making the ray halt before that blocker instead of reaching the boundary.
#    Small example: a compact NE seed [2 7] above-right of a plus grows as
#    row-1:[2 7], row-2:[7 2], row-3:[2 7], then stops because row-4 has obstacle.
# b) Complexity calibration:
#    Restricting seeds to 2 cells prevents diagonal self-overlap, keeps the mirror
#    visible, and avoids ambiguous blended footprints. Every sample still contains
#    2-3 pluses, exactly one full clue ray, and at least one early stop, so all
#    three changed mechanisms repeat within a grid rather than appearing once.
# c) Assumption audit:
#    - Original code assumed marker components are 4-neighbor attached to a plus arm.
#      That breaks here; seeds touch pluses only diagonally, so component lookup must
#      use 8-neighbor adjacency around each plus.
#    - Original axial alignment checks via nozzle spans no longer apply. Direction is
#      now recovered from which diagonal corner window contains the 2-cell seed.
#    - Original extension was a simple modulo stripe to the frame. That fails twice:
#      diagonal steps are by unit, not by rows/cols alone, and odd/even units use
#      different mirrored tiles. The extension loop must therefore operate unit-by-unit.
#    - With blockers present, extension can no longer assume every future target is
#      background. The next whole unit is tested first; any occupied target cell ends
#      the ray immediately.
#    - A pre-expanded clue component is now a long diagonal chain, not a compact seed.
#      main() therefore extends only 2-cell components and leaves longer clue chains
#      untouched in the copied output.
# d) Placement legality:
#    The generator is output-first. For each plus it chooses a diagonal direction, a
#    1x2 or 2x1 seed, a stop length, and an obstacle occupying the next unit slot.
#    It rejects placements if plus bboxes (expanded by 1) collide, if any ray cell or
#    obstacle overlaps existing objects, or if a future plus would become diagonally
#    adjacent to a foreign ray near its corner windows.
# e) Visual observability:
#    Humans can read the rule from the clue ray because all relevant structure is
#    visible: the clue leaves from a diagonal corner, alternates between two 2-cell
#    orderings, and terminates flush against a same-colored blocker. Compact seeds on
#    the other pluses use the same 2-cell shapes, so the learned continuation transfers.
# f) Visual cleanliness:
#    Rays are thin diagonal zippers, obstacles are aligned copies in one extra color,
#    and pluses are separated by reserved margin. This avoids layered clutter and
#    keeps seed / ray / obstacle roles distinct.
# g) Inferability of change:
#    No hidden color arithmetic is used. Seed colors are copied geometrically under
#    visible mirror transforms, obstacle color is globally shared and never used in
#    seeds, and all launch directions are encoded by literal corner placement.
# Main/generator contract:
#    main() assumes each non-clue plus has exactly one adjacent 2-cell seed located
#    in a legal corner window; generate_input() enforces that and also ensures at
#    least one clue ray is long enough to exhibit diagonal motion, mirror alternation,
#    and obstacle stopping.


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


_DIAG_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_SEED_SHAPES = [(1, 2), (2, 1)]


def _mirror_tile(tile: np.ndarray) -> np.ndarray:
    """Mirror a 2-cell seed along its long axis."""
    h, w = tile.shape
    if (h, w) == (1, 2):
        return np.fliplr(tile)
    if (h, w) == (2, 1):
        return np.flipud(tile)
    raise ValueError(f"_mirror_tile: unsupported seed shape {(h, w)}")


def _tile_variant(tile: np.ndarray, unit_index: int) -> np.ndarray:
    """Even units keep the original seed; odd units use its directional mirror."""
    return _mirror_tile(tile) if int(unit_index) % 2 == 1 else tile


def _candidate_seed_bboxes(
    plus_bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """All legal 2-cell corner windows around one plus bbox."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    return [
        ((-1, -1), (pr0 - 1, pc0 - 1, pr0 - 1, pc0)),      # NW horizontal 1x2
        ((-1, -1), (pr0 - 1, pc0 - 1, pr0, pc0 - 1)),      # NW vertical 2x1
        ((-1, 1), (pr0 - 1, pc1, pr0 - 1, pc1 + 1)),       # NE horizontal 1x2
        ((-1, 1), (pr0 - 1, pc1 + 1, pr0, pc1 + 1)),       # NE vertical 2x1
        ((1, -1), (pr1 + 1, pc0 - 1, pr1 + 1, pc0)),       # SW horizontal 1x2
        ((1, -1), (pr1, pc0 - 1, pr1 + 1, pc0 - 1)),       # SW vertical 2x1
        ((1, 1), (pr1 + 1, pc1, pr1 + 1, pc1 + 1)),        # SE horizontal 1x2
        ((1, 1), (pr1, pc1 + 1, pr1 + 1, pc1 + 1)),        # SE vertical 2x1
    ]


def _bbox_in_bounds(bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W


def _adjacent_component_ids8(mark_lab: np.ndarray, plus_mask: np.ndarray) -> set[int]:
    """Return payload component ids that touch the plus through 8-neighborhood."""
    H, W = mark_lab.shape
    adj_ids: set[int] = set()
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))
    return adj_ids


def _infer_seed_window(
    comp_mask: np.ndarray,
    plus_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    """Recover the diagonal direction and 2-cell seed window for one component."""
    H, W = comp_mask.shape
    matches: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
    for direction, bbox in _candidate_seed_bboxes(plus_bbox):
        if not _bbox_in_bounds(bbox, H, W):
            continue
        r0, c0, r1, c1 = map(int, bbox)
        cells = [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]
        if all(bool(comp_mask[rr, cc]) for rr, cc in cells):
            matches.append((direction, bbox))
    if len(matches) != 1:
        return None
    return matches[0]


def _infer_obstacle_color_and_mark_lab(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Find the color whose removal makes every plus adjacent to exactly one valid
    diagonal seed/ray component.
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})
    H, W = grid.shape

    for obstacle_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(obstacle_col))
        mark_lab, _ = label_mask(non_mask, connectivity=4)
        ok = True
        for pk in range(1, int(plus_lab.max()) + 1):
            plus_mask = plus_lab == int(pk)
            plus_coords = np.argwhere(plus_mask)
            if plus_coords.size == 0:
                ok = False
                break
            plus_bbox = bbox_coords(plus_coords)
            adj_ids = _adjacent_component_ids8(mark_lab, plus_mask)
            if len(adj_ids) != 1:
                ok = False
                break
            cid = next(iter(adj_ids))
            comp_mask = mark_lab == int(cid)
            parsed = _infer_seed_window(comp_mask, plus_bbox)
            if parsed is None:
                ok = False
                break
        if ok:
            return int(obstacle_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def extend_diagonal_mirror_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Extend a 2-cell diagonal ray one shifted unit at a time until blocked."""
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    unit = 1
    while True:
        cur = _tile_variant(tile, unit)
        base_r = int(r0 + unit * dr)
        base_c = int(c0 + unit * dc)
        cells: list[tuple[int, int, int]] = []
        for i in range(h):
            for j in range(w):
                v = int(cur[i, j])
                if v == int(bg):
                    continue
                rr = int(base_r + i)
                cc = int(base_c + j)
                if not (0 <= rr < H and 0 <= cc < W):
                    return
                cells.append((rr, cc, v))
        if any(int(out[rr, cc]) != int(bg) for rr, cc, _ in cells):
            return
        for rr, cc, v in cells:
            out[rr, cc] = int(v)
        unit += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal marker rays from '+' anchors using alternating mirrors and visible blockers.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several identical '+' sprites.
    - Each plus has one 2-cell marker in a diagonal corner window, one extra color
      appears only in detached obstacle copies, and exactly one full diagonal clue
      ray is already present in the input.

    Rule:
    - Determine each marker's launch direction from which corner of the plus bbox
      contains the seed.
    - Extend compact 2-cell seeds by shifting them one cell diagonally per unit,
      alternating original and mirrored order on successive units.
    - Stop before the first non-background blocker, so rays halt at obstacle copies
      instead of continuing to the frame.

    Key insight:
    - The clue ray simultaneously reveals all three hidden controls: corner-based
      diagonal motion, odd/even mirror alternation, and the fact that the ray ends
      because the next aligned unit slot is occupied by a dedicated obstacle color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    _obstacle_col, mark_lab = _infer_obstacle_color_and_mark_lab(
        g,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue
        plus_bbox = bbox_coords(plus_coords)
        adj_ids = _adjacent_component_ids8(mark_lab, plus_mask)
        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        if pts.size == 0:
            continue
        comp_mask = mark_lab == int(cid)
        parsed = _infer_seed_window(comp_mask, plus_bbox)
        if parsed is None:
            continue
        direction, seed_bbox = parsed
        r0, c0, r1, c1 = map(int, seed_bbox)
        tile = crop_bbox(g, (r0, c0, r1, c1))

        # Compact seeds have exactly the 2 cells of the seed window; longer components
        # are the already-expanded clue rays and are left untouched.
        if int(pts.shape[0]) != int(tile.size):
            continue

        extend_diagonal_mirror_until_blocked(
            out,
            tile,
            r0,
            c0,
            direction,
            bg=int(bg),
        )

    return out


# --------------- generator ---------------

_H_RANGE = (14, 29)
_W_RANGE = (14, 23)
_N_PLUS_RANGE = (2, 4)
_MIN_STOP_UNITS = 4
_PRE_EXT_IDEAL_UNITS = 7
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 500

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


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    seed_shape: tuple[int, int],
) -> tuple[int, int]:
    """Top-left corner of a compact seed in one legal diagonal corner window."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    h, w = map(int, seed_shape)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return int(pr0 - 1), int(pc0 - 1)
    if (dr, dc) == (-1, 1):
        return (int(pr0 - 1), int(pc1)) if (h, w) == (1, 2) else (int(pr0 - 1), int(pc1 + 1))
    if (dr, dc) == (1, -1):
        return (int(pr1 + 1), int(pc0 - 1)) if (h, w) == (1, 2) else (int(pr1), int(pc0 - 1))
    if (dr, dc) == (1, 1):
        return (int(pr1 + 1), int(pc1)) if (h, w) == (1, 2) else (int(pr1), int(pc1 + 1))
    raise ValueError(f"_seed_origin: unsupported direction {(dr, dc)}")


def _build_seed_tile(
    seed_shape: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a 2-cell asymmetric seed so mirrored units are visibly different."""
    a, b = map(int, rng.choice(palette, size=2, replace=False))
    vals = np.array([a, b], dtype=int)
    if rng.random() < 0.5:
        vals = vals[::-1]
    return vals.reshape(seed_shape)


def _compute_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    """Maximum number of full diagonal units that fit, including the seed."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    if dr == 1:
        row_units = 1 + (H - (seed_r0 + h))
    else:
        row_units = 1 + seed_r0
    if dc == 1:
        col_units = 1 + (W - (seed_c0 + w))
    else:
        col_units = 1 + seed_c0
    return int(min(row_units, col_units))


def _project_ray_cells(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    units: int,
) -> list[tuple[int, int, int]]:
    """Cells for the first `units` diagonal units, including the seed."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    cells: list[tuple[int, int, int]] = []
    for u in range(int(units)):
        cur = _tile_variant(tile, u)
        base_r = int(seed_r0 + u * dr)
        base_c = int(seed_c0 + u * dc)
        for i in range(h):
            for j in range(w):
                cells.append((int(base_r + i), int(base_c + j), int(cur[i, j])))
    return cells


def _obstacle_cells_after_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    stop_units: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """Obstacle occupies the next unit footprint, forcing the ray to stop."""
    dr, dc = map(int, direction)
    h, w = tile.shape
    base_r = int(seed_r0 + stop_units * dr)
    base_c = int(seed_c0 + stop_units * dc)
    return [
        (int(base_r + i), int(base_c + j), int(obstacle_color))
        for i in range(h)
        for j in range(w)
    ]


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _expanded_bbox_cells(
    plus_bbox: tuple[int, int, int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    cells: list[tuple[int, int]] = []
    for rr in range(max(0, pr0 - 1), min(H - 1, pr1 + 1) + 1):
        for cc in range(max(0, pc0 - 1), min(W - 1, pc1 + 1) + 1):
            cells.append((int(rr), int(cc)))
    return cells


def _place_plus_ray_and_obstacle(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    reserved: np.ndarray,
    out: np.ndarray,
    *,
    plus_color: int,
    obstacle_color: int,
) -> dict | None:
    """Place one plus, its diagonal mirrored ray, and the blocker that stops it."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        zone_cells = _expanded_bbox_cells(plus_bbox, H, W)
        if any(bool(reserved[rr, cc]) or int(occ[rr, cc]) != 0 for rr, cc in zone_cells):
            continue

        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]

        dirs = list(_DIAG_DIRECTIONS)
        rng.shuffle(dirs)
        shapes = list(_SEED_SHAPES)
        rng.shuffle(shapes)

        for direction in dirs:
            for seed_shape in shapes:
                tile = _build_seed_tile(seed_shape, palette, rng)
                seed_r0, seed_c0 = _seed_origin(plus_bbox, direction, seed_shape)
                full_units = _compute_full_units(H, W, seed_r0, seed_c0, direction, tile)
                if full_units <= _MIN_STOP_UNITS:
                    continue

                candidates = list(range(_MIN_STOP_UNITS, full_units))
                rng.shuffle(candidates)
                for stop_units in candidates:
                    ray_cells = _project_ray_cells(seed_r0, seed_c0, direction, tile, stop_units)
                    obstacle_cells = _obstacle_cells_after_units(
                        seed_r0, seed_c0, direction, tile, stop_units, int(obstacle_color)
                    )
                    if not _all_in_bounds(H, W, ray_cells) or not _all_in_bounds(H, W, obstacle_cells):
                        continue
                    if any(int(occ[rr, cc]) != 0 for rr, cc, _ in ray_cells + obstacle_cells):
                        continue
                    if any(bool(reserved[rr, cc]) for rr, cc, _ in ray_cells + obstacle_cells):
                        continue

                    for rr, cc in plus_cells:
                        occ[rr, cc] = 1
                        out[rr, cc] = int(plus_color)
                    for rr, cc in zone_cells:
                        reserved[rr, cc] = True
                    for rr, cc, v in ray_cells:
                        occ[rr, cc] = 2
                        out[rr, cc] = int(v)
                    for rr, cc, v in obstacle_cells:
                        occ[rr, cc] = 3
                        out[rr, cc] = int(v)

                    return {
                        "plus_bbox": plus_bbox,
                        "tile": tile,
                        "direction": (int(direction[0]), int(direction[1])),
                        "seed_r0": int(seed_r0),
                        "seed_c0": int(seed_c0),
                        "units": int(stop_units),
                        "full_units": int(full_units),
                        "cells": ray_cells,
                        "obstacle_cells": obstacle_cells,
                    }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full clue ray; shrink every other ray back to its 2-cell seed."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if int(p["units"]) >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        tile = p["tile"]
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        h, w = tile.shape
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(h) for jj in range(w)}
        for rr, cc, _ in p["cells"]:
            if (int(rr), int(cc)) not in seed_set:
                inp[int(rr), int(cc)] = int(bg)
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
    if len(palette) < 2:
        return None

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_ray_and_obstacle(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            reserved,
            out,
            plus_color=int(plus_color),
            obstacle_color=int(obstacle_color),
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["units"]) for p in pluses) < _PRE_EXT_IDEAL_UNITS:
        return None

    inp = _build_input_from_output(out, pluses, bg, rng)

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
        }
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
