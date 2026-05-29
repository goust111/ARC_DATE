# Combo variant of 53fb4810 (index 25)
# combo_spec: D6/0+D17/0
# Name: 墙壁障碍遇阻停止

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Original 53fb4810 has several identical '+' anchors, one attached seed tile
#    per '+', and exactly one already-expanded clue ray. Each seed repeats away
#    from its '+' until the frame edge. The fused variant inserts a third-role
#    color: static wall bars. Example:
#      original: plus -> seed AB -> AB AB AB AB ... boundary
#      combo   : plus -> seed AB -> AB AB [wall color]
#    The wall is not decorative. It is the only reason the ray stops early, so
#    D17 (obstacle exists) and D6 (new stopping condition) are realized by the
#    same visible object rather than by two unrelated edits.
# b) Complexity calibration:
#    There is one shared wall color across the whole grid, distinct from bg, plus,
#    and all seed colors. Every ray gets its own aligned wall bar, so the stopping
#    rule is repeated multiple times in each sample. Rays still use the original
#    small tile vocabulary (1x2, 2x1, 2x2) to preserve the base task's periodic
#    reasoning without adding a second matching problem.
# c) Hidden assumptions audited:
#    - Original `label_mask(non_mask)` merged any touching non-bg cells. With
#      wall bars touching clue rays at the stopping face, that would fuse payload
#      and wall into one component. The solver therefore first tries to identify
#      a dedicated wall color structurally and exclude it before labeling payloads.
#      A fallback path still exists: if exclusion is ambiguous, later alignment
#      checks discard spurious adjacent components.
#    - Original extension assumed boundary was the only endpoint. Here the next
#      stripe may already contain wall cells, so extension must probe occupancy
#      stripe-by-stripe and stop before the first non-bg collision.
#    - Original order-independence relied on rays not intersecting. The generator
#      keeps pluses, rays, and walls mutually disjoint, so using `out` as the
#      collision surface remains valid.
# d) Placement legality:
#    Generation stays output-first. For each plus we choose a direction, a small
#    aligned seed tile, a full number of tile copies to keep, and then place one
#    wall stripe immediately after that integer number of copies. Candidates are
#    rejected if any plus, ray, or wall cell overlaps an existing object, leaves
#    the grid, or violates the one-cell margin around pluses.
# e) Visual observability:
#    A human can see the same dedicated wall color reappearing away from all
#    anchors, and the pre-expanded clue ray always ends flush against that wall.
#    Across 2-4 training examples this makes the new endpoint rule inferable
#    without relying on hidden ordering or numeric color ids.
# f) Visual cleanliness:
#    Walls are thin bars orthogonal to the travel direction, rays do not cross,
#    pluses remain separated, and the clue ray is still unique. The grid therefore
#    stays readable: plus, payload, and wall roles occupy different visual niches.
# g) Change inferability:
#    No color remapping happens. Seed colors repeat unchanged; the third color is
#    purely structural and always acts as a blocker. Because the same blocker role
#    recurs on every ray, the solver can infer a consistent semantics for it.
# h) New solver assumptions guaranteed by the generator:
#    - Generated samples are filtered so that exactly one removable non-bg/non-plus
#      color makes every '+' adjacent to exactly one valid marker-or-ray component;
#      that color is the wall color.
#    - Each wall sits immediately after an integer number of full tile units, so
#      "stop when the next stripe would collide" reproduces the constructed output.
#    - Any pre-expanded clue component may be longer than the compact seed, but
#      its cross-axis span still matches the nozzle span, so direction recovery
#      from adjacency plus bbox alignment remains valid.


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
    """Return ((dr,dc), bbox) if a payload component is a valid marker/ray for this plus."""
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
        nozzle_cols = sorted(
            int(cc) for cc in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, cc])
        )
        payload_cols = list(range(int(c0), int(c1) + 1))
        if payload_cols != nozzle_cols:
            return None
    else:
        nozzle_col = int(pc1) if dc == 1 else int(pc0)
        nozzle_rows = sorted(
            int(rr) for rr in range(int(pr0), int(pr1) + 1) if bool(plus_mask[rr, nozzle_col])
        )
        payload_rows = list(range(int(r0), int(r1) + 1))
        if payload_rows != nozzle_rows:
            return None

    return (int(dr), int(dc)), (int(r0), int(c0), int(r1), int(c1))


def _infer_wall_color_and_mark_lab(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Find the color whose removal makes every '+' adjacent to exactly one valid
    marker/ray component. Returns (wall_color, mark_lab).
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})

    for wall_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(wall_col))
        mark_lab, _ = label_mask(non_mask, connectivity=4)
        ok = True
        for pk in range(1, int(plus_lab.max()) + 1):
            plus_mask = plus_lab == int(pk)
            adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
            if len(adj_ids) != 1:
                ok = False
                break
            cid = next(iter(adj_ids))
            pts = np.argwhere(mark_lab == int(cid))
            if _marker_direction_and_alignment(pts, plus_mask) is None:
                ok = False
                break
        if ok:
            return int(wall_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def detect_wall_color(
    grid: np.ndarray,
    plus_mask: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> int | None:
    """Recover the dedicated wall color structurally."""
    plus_lab, _ = label_mask(plus_mask, connectivity=4)
    wall_col, _ = _infer_wall_color_and_mark_lab(
        grid,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )
    return wall_col


def extend_repeating_tile_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    wall_color: int | None = None,
) -> None:
    """Extend a repeating tile until the next stripe would hit the wall or another obstacle."""
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def can_paint(cells: list[tuple[int, int, int]]) -> bool:
        active = [(rr, cc) for rr, cc, v in cells if int(v) != int(bg)]
        if wall_color is not None and any(int(out[rr, cc]) == int(wall_color) for rr, cc in active):
            return False
        return all(int(out[rr, cc]) == int(bg) for rr, cc in active)

    def paint(cells: list[tuple[int, int, int]]) -> None:
        for rr, cc, v in cells:
            if int(v) != int(bg):
                out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            stripe = [(r0 + i, tc, int(tile[i, k % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            stripe = [(r0 + i, tc, int(tile[i, (-k) % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            stripe = [(tr, c0 + j, int(tile[k % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            stripe = [(tr, c0 + j, int(tile[(-k) % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    else:
        raise ValueError("extend_repeating_tile_until_blocked: direction must be axial")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' shapes, but stop when the ray reaches a wall.

    Input structure:
    - The most common color is background.
    - One dominant non-background color forms several identical '+'-like anchors.
    - Each anchor has one attached seed or clue ray, and a third color appears as
      detached wall bars farther along those same axes.

    Rule:
    - For every payload component touching a '+', infer which side of the '+' it
      is attached to and therefore which axial direction it should travel.
    - Repeat the component away from the '+' with the same period.
    - Stop immediately before the first non-background stripe ahead, which is the
      wall in valid generated instances.

    Key insight: The attachment side still determines direction, but the endpoint
    is controlled by a dedicated structural wall color rather than by the frame.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    wall_col, mark_lab = _infer_wall_color_and_mark_lab(
        g,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            parsed = _marker_direction_and_alignment(pts, plus_mask)
            if parsed is None:
                continue
            (dr, dc), (r0, c0, r1, c1) = parsed
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

            extend_repeating_tile_until_blocked(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                wall_color=wall_col,
            )

    return out


# --------------- generator (output-first, wall-aware) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_CHOICES = [2, 3, 4]
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_STOP_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 5
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _axis_key(direction: tuple[int, int]) -> str:
    return "v" if int(direction[0]) != 0 else "h"


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
) -> np.ndarray | None:
    """Build a seed tile whose cross-axis matches the nozzle span."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
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
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        return (int(pr1 + 1) if dr == 1 else int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1) if dc == 1 else int(pc0 - tw))


def _compute_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    """Maximum number of full tile units that fit before the boundary."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return 1 + (H - (seed_r0 + th)) // th
    if dr == -1:
        return 1 + seed_r0 // th
    if dc == 1:
        return 1 + (W - (seed_c0 + tw)) // tw
    return 1 + seed_c0 // tw


def _project_tile_coords_for_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    units: int,
) -> list[tuple[int, int, int]]:
    """Return cells for the first `units` full tile copies, including the seed."""
    dr, dc = direction
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    for u in range(int(units)):
        rr0 = int(seed_r0 + u * th * dr)
        cc0 = int(seed_c0 + u * tw * dc)
        for i in range(th):
            for j in range(tw):
                cells.append((int(rr0 + i), int(cc0 + j), int(tile[i, j])))
    return cells


def _wall_cells_after_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    stop_units: int,
    wall_color: int,
) -> list[tuple[int, int, int]]:
    """Place a 1-stripe wall immediately after `stop_units` full tiles."""
    dr, dc = direction
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    if dr == 1:
        row = int(seed_r0 + stop_units * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(wall_color)))
    elif dr == -1:
        row = int(seed_r0 - 1 - (stop_units - 1) * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(wall_color)))
    elif dc == 1:
        col = int(seed_c0 + stop_units * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(wall_color)))
    else:
        col = int(seed_c0 - 1 - (stop_units - 1) * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(wall_color)))
    return cells


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _place_plus_ray_and_wall(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    *,
    plus_color: int,
    wall_color: int,
    preferred_directions: list[tuple[int, int]],
) -> dict | None:
    """Try to place one plus, its repeating ray, and a blocking wall."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        for d in preferred_directions:
            tile = _build_seed_tile(plus_shape, d, palette, rng)
            if tile is None:
                continue

            seed_r0, seed_c0 = _seed_origin((r0, c0, r0 + ph - 1, c0 + pw - 1), plus_shape, d, tile)
            full_units = _compute_full_units(H, W, seed_r0, seed_c0, d, tile)
            if full_units <= _MIN_STOP_UNITS:
                continue

            candidates = list(range(_MIN_STOP_UNITS, full_units))
            rng.shuffle(candidates)
            for stop_units in candidates:
                ray_cells = _project_tile_coords_for_units(seed_r0, seed_c0, d, tile, stop_units)
                wall_cells = _wall_cells_after_units(
                    seed_r0, seed_c0, d, tile, stop_units, int(wall_color)
                )
                if not _all_in_bounds(H, W, ray_cells) or not _all_in_bounds(H, W, wall_cells):
                    continue
                if any(int(occ[r, c]) != 0 for r, c, _ in ray_cells):
                    continue
                if any(int(occ[r, c]) != 0 for r, c, _ in wall_cells):
                    continue
                if any(int(plus_occ[r, c]) == 1 for r, c, _ in wall_cells):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    out[r, c] = int(plus_color)
                    for dr2, dc2 in _DIRECTIONS:
                        rr = r + dr2
                        cc = c + dc2
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[rr, cc] = 1
                for r, c, v in ray_cells:
                    occ[r, c] = 2
                    out[r, c] = int(v)
                for r, c, v in wall_cells:
                    occ[r, c] = 3
                    out[r, c] = int(v)

                return {
                    "tile": tile,
                    "cells": ray_cells,
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "direction": (int(d[0]), int(d[1])),
                    "units": int(stop_units),
                    "full_units": int(full_units),
                    "wall_cells": wall_cells,
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erase non-clue rays from the output, keeping all walls and one full clue ray."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
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
    plus_color, wall_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {plus_color, wall_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.choice(_N_PLUS_CHOICES))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    dir_counts = {d: 0 for d in _DIRECTIONS}
    axis_counts = {"v": 0, "h": 0}
    for _ in range(n_plus):
        priorities = {d: (axis_counts[_axis_key(d)], dir_counts[d], float(rng.random())) for d in _DIRECTIONS}
        preferred_directions = sorted(_DIRECTIONS, key=lambda d: priorities[d])
        info = _place_plus_ray_and_wall(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color=int(plus_color),
            wall_color=int(wall_color),
            preferred_directions=preferred_directions,
        )
        if info is None:
            return None
        pluses.append(info)
        dir_counts[info["direction"]] += 1
        axis_counts[_axis_key(info["direction"])] += 1

    if max(int(p["units"]) for p in pluses) < _PRE_EXT_IDEAL_UNITS:
        return None

    inp = _build_input_from_output(out, pluses, bg, rng)

    plus_lab, _ = label_mask(inp == int(plus_color), connectivity=4)
    inferred_wall, _ = _infer_wall_color_and_mark_lab(
        inp,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_color),
    )
    if inferred_wall != int(wall_color):
        return None

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
            "wall_color": int(wall_color),
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
