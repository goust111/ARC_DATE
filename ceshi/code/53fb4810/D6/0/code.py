# Variant of 53fb4810
# Dimension: D6 — 终点条件 (dst)
# Change: 射线一直延伸到网格边界为止... → 遇障碍物停止

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior:
#    The original task had one clue ray already expanded and every other seed had
#    to be repeated outward until the grid boundary. In this variant, every ray
#    still starts from a small multicolor seed attached to one side of a shared
#    '+' sprite, but a dedicated obstacle color appears as extra bars farther out
#    on those axes. The output repeats the seed away from the '+' only up to the
#    cell immediately before the first obstacle on that axis. Example:
#      original: plus -> seed AB -> AB AB AB AB ... edge
#      variant : plus -> seed AB -> AB AB [obstacle color bar]
#    The changed stage is the ray-extension loop: instead of treating boundary as
#    the only stopping condition, the solver now halts when the next stripe to be
#    painted already contains a non-background cell.
# b) Complexity calibration:
#    Obstacles use one shared color that never appears in seeds. This keeps the
#    extra mechanism visible without introducing a second matching problem. One
#    obstacle is placed on every ray, so the new stopping rule is repeatedly
#    demonstrated in each sample rather than appearing as a rare edge case.
# c) Hidden assumptions from the original main():
#    - `label_mask(non_mask)` merged all touching non-bg colors. With obstacles
#      added flush against ray ends, that would incorrectly fuse ray and obstacle
#      into one component. The solver therefore detects the obstacle color first
#      and excludes it from marker-component labeling.
#    - `extend_repeating_tile(...)` assumed every future target cell was either
#      bg or protected '+' color. That is no longer true; the write loop must
#      test the next stripe before painting and stop on first occupancy.
#    - The original solver was harmlessly order-independent because rays never
#      intersected. This remains an implicit assumption; the generator preserves
#      disjoint rays so that using `out` as the collision surface is safe.
# d) Placement legality:
#    Generator placement is still output-first. Each plus gets exactly one seed,
#    a ray direction, a stop distance, and an aligned obstacle bar. We reject any
#    candidate where plus, ray, or obstacle would overlap an existing object or
#    place the obstacle outside the grid.
# e) Visual observability:
#    Humans can infer the new rule from 2-4 examples because the same dedicated
#    obstacle color appears away from all pluses, the long clue ray terminates
#    exactly at that color, and the shorter seeds expand to the same kind of stop.
#    No ordering depends on invisible color ids.
# f) Visual cleanliness:
#    Rays are kept pairwise disjoint, pluses are separated by a one-cell margin,
#    and obstacles are thin aligned bars rather than irregular clutter. This keeps
#    the grid readable: plus/seed/obstacle roles remain visually distinct.
# g) Color inferability:
#    There is no hidden color remapping. Seed colors repeat unchanged; obstacle
#    color is purely a blocker and is globally shared.
# h) New solver assumptions audited against the generator:
#    - Exactly one color can be removed so that every '+' is adjacent to exactly
#      one valid marker/ray component. The solver identifies obstacles through
#      this structural test instead of relying on raw color adjacency.
#    - Obstacles touch rays only at the stopping face; generator guarantees this
#      by placing each obstacle bar immediately after an integer number of full
#      tile repeats.


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

    best_key: tuple[int, int] | None = None  # (num_big, total_big_area)
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
            if 0 <= rr < H and 0 <= cc < W:
                if bool(plus_mask[rr, cc]):
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


def _infer_obstacle_color_and_mark_lab(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Find the color whose removal makes every '+' adjacent to exactly one valid
    marker/ray component. Returns (obstacle_color, mark_lab).
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})

    for obstacle_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(obstacle_col))
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
            return int(obstacle_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def detect_obstacle_color(grid: np.ndarray, plus_mask: np.ndarray, *, bg: int, plus_col: int) -> int | None:
    """Public helper used by analysis code: recover the obstacle color structurally."""
    plus_lab, _ = label_mask(plus_mask, connectivity=4)
    obstacle_col, _ = _infer_obstacle_color_and_mark_lab(
        grid,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )
    return obstacle_col


def extend_repeating_tile_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Extend a repeating tile from (r0,c0) until the next stripe would hit a non-bg cell."""
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
        return all(int(out[rr, cc]) == int(bg) for rr, cc in active)

    def paint(cells: list[tuple[int, int, int]]) -> None:
        for rr, cc, v in cells:
            if int(v) != int(bg):
                out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):  # right
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            stripe = [(r0 + i, tc, int(tile[i, k % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (0, -1):  # left
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            stripe = [(r0 + i, tc, int(tile[i, (-k) % w])) for i in range(h)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (1, 0):  # down
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            stripe = [(tr, c0 + j, int(tile[k % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    elif (dr, dc) == (-1, 0):  # up
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            stripe = [(tr, c0 + j, int(tile[(-k) % h, j])) for j in range(w)]
            if not can_paint(stripe):
                break
            paint(stripe)
    else:
        raise ValueError("extend_repeating_tile_until_blocked: direction must be axial")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' shapes, but stop when the ray reaches a blocker.

    Input structure:
    - bg is the dominant color.
    - One dominant non-bg color forms several identical '+'-like anchors.
    - Small multicolor seed tiles touch one side of each '+', and one extra color
      appears only in detached obstacle bars farther along the rays.

    Rule:
    - For each seed component touching a '+', infer the side where it attaches.
    - Repeat that component away from the '+' with the same period.
    - Halt as soon as the next stripe would hit any already occupied non-bg cell,
      so rays stop at obstacle bars instead of running to the boundary.

    Key insight: The side of attachment still determines the ray direction, but
    the endpoint is now controlled by visible blockers, not by the outer frame.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

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

        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            parsed = _marker_direction_and_alignment(pts, plus_mask)
            if parsed is None:
                continue
            (dr, dc), (r0, c0, r1, c1) = parsed
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

            extend_repeating_tile_until_blocked(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
            )

    return out


# --------------- generator (output-first, obstacle-aware) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)            # grid height [min, max)
_W_RANGE = (13, 23)            # grid width [min, max)
_N_PLUS_RANGE = (2, 4)         # number of plus sprites [min, max)
_SEED_ALONG_CHOICES = [1, 2]   # along-axis dimension options for seed tile
_SEED_MAX_COLORS = 4           # max distinct colors in a seed tile
_SEED_MAX_CROSS = 2            # max nozzle cross-axis span for valid seed
_MIN_STOP_UNITS = 2            # minimum full tile units kept before an obstacle
_PRE_EXT_IDEAL_UNITS = 5       # preferred minimum units for the pre-extended clue ray
_PLACEMENT_TRIES = 120         # attempts per plus placement
_GEN_RETRIES = 400             # outer retry limit in generate_input

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


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


def _obstacle_cells_after_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    stop_units: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """Place a 1-stripe obstacle bar immediately after `stop_units` full tiles."""
    dr, dc = direction
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    if dr == 1:
        row = int(seed_r0 + stop_units * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dr == -1:
        row = int(seed_r0 - 1 - (stop_units - 1) * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dc == 1:
        col = int(seed_c0 + stop_units * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    else:
        col = int(seed_c0 - 1 - (stop_units - 1) * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    return cells


def _all_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _place_plus_ray_and_obstacle(
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
    obstacle_color: int,
) -> dict | None:
    """Try to place one plus, its repeating ray, and a blocking obstacle."""
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

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
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
                obstacle_cells = _obstacle_cells_after_units(
                    seed_r0, seed_c0, d, tile, stop_units, int(obstacle_color)
                )
                if not _all_in_bounds(H, W, ray_cells) or not _all_in_bounds(H, W, obstacle_cells):
                    continue
                if any(int(occ[r, c]) != 0 for r, c, _ in ray_cells):
                    continue
                if any(int(occ[r, c]) != 0 for r, c, _ in obstacle_cells):
                    continue
                if any(int(plus_occ[r, c]) == 1 for r, c, _ in obstacle_cells):
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
                for r, c, v in obstacle_cells:
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
                    "obstacle_cells": obstacle_cells,
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erase non-clue rays from the output, keeping obstacles and one full example ray."""
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
    plus_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {plus_color, obstacle_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_ray_and_obstacle(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
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
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
