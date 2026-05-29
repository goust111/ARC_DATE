# Variant of 53fb4810
# Dimension: D9 — 预展开射线（示例线索）机制 (multi_step)
# Change: 输入中恰好有一条射线被完整预展开作为示例线索，其余标记仅保留紧凑种子。解题者需从预展开的射线推断规则，然后对未展开的种子... → 部分展开线索

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask

# -----------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# a) Concretization / behavior trace
# - The original task shows one fully drawn ray that already reaches the border.
# - This variant keeps the same plus sprites, same seed attachment rule, and same
#   periodic tiling rule, but the single demonstration ray now stops early after
#   a few repeats and a short extra fragment. Example: a horizontal seed tile of
#   width 2 may appear as 7 visible columns = 3 full periods + 1 extra column.
# - Old logic at the critical step:
#   1. find one connected marker component next to a plus;
#   2. crop the whole component bbox;
#   3. treat that whole bbox as the tile to be repeated.
# - Variant logic at the same step:
#   1. find the marker component and its direction from the plus;
#   2. re-orient the component as a strip from the plus outward;
#   3. recover the smallest period consistent with that visible prefix;
#   4. use only that primitive tile, not the whole prefix, when extending.
# - Difference: with a visible prefix length of 7 and a true period of 2, the old
#   logic repeats a width-7 strip; the new logic repeats a width-2 tile and thus
#   correctly continues the ray to the boundary.
#
# b) Complexity calibration
# - The only structural change is the clue length. Plus shape, attachment, and
#   path family remain unchanged so the task is still recognizable as 53fb4810.
# - To make the change observable and non-degenerate, the generator forces the
#   ray-period along its travel axis to be 2 and the clue length to be "several
#   full periods plus one extra cell/row". This avoids cases where the partial
#   clue is still trivially equivalent to the original full-prefix treatment.
#
# c) Implicit-assumption audit of the original main()
# - `tile = crop_bbox(g, bbox_of_component)` assumed every adjacent component is
#   exactly one seed tile. That no longer holds for the partial clue.
# - `extend_repeating_tile(..., tile, bbox_start, direction)` then relied on the
#   component bbox already being the primitive period. In this variant it may be
#   a longer periodic prefix, so the solver must first compress it back to the
#   smallest period along the ray axis.
# - The cross-axis alignment asserts still remain valid because the partial clue
#   stays a solid strip whose height/width matches the arm nozzle span.
#
# d) Placement legality
# - Output is still generated first using full rays to the boundary.
# - Objects must remain disjoint: pluses do not overlap each other, rays do not
#   collide with other rays or pluses, and the visible partial clue is carved by
#   erasing a tail from one already-valid full ray, so it cannot introduce new
#   overlaps or break adjacency.
#
# e) Visual observability
# - Across training examples the single clue clearly differs from the original:
#   it visibly stops in open background before the border.
# - The extra partial cell/row beyond 3-4 full periods signals that the visible
#   endpoint is not semantically special; it is just a cut-off sample.
#
# f) Visual cleanliness
# - Only one ray is partially shown, and it remains a straight rectangular strip.
# - Other markers stay compact seeds, so the scene remains sparse and readable.
#
# g) Color-rule audit
# - No new color mapping is introduced. Colors remain direct tile payload values.
# - The reasoning burden is purely spatial/periodic, which is visible in-grid.
#
# h) New solver assumptions introduced here
# - Every marker strip is periodic with primitive travel-axis period 2.
# - The strip contains no background holes, so its bbox is the full visible clue.
# - The generator enforces both assumptions.
# -----------------------------------------------------------------------------


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


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):  # right
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):  # left
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):  # down
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):  # up
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def _axis_len(arr: np.ndarray, direction: tuple[int, int]) -> int:
    return int(arr.shape[1] if direction[1] != 0 else arr.shape[0])


def _orient_strip_from_plus(strip: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """Return the visible strip ordered from the plus-adjacent seed outward."""
    dr, dc = direction
    if dc == 1 or dr == 1:
        return strip.copy()
    if dc == -1:
        return strip[:, ::-1].copy()
    return strip[::-1, :].copy()


def _primitive_period_length(strip: np.ndarray, direction: tuple[int, int]) -> int:
    """Smallest travel-axis period whose repeated prefix matches the whole strip."""
    L = _axis_len(strip, direction)
    if direction[1] != 0:
        for p in range(1, L + 1):
            if all(np.array_equal(strip[:, idx], strip[:, idx % p]) for idx in range(L)):
                return int(p)
    else:
        for p in range(1, L + 1):
            if all(np.array_equal(strip[idx, :], strip[idx % p, :]) for idx in range(L)):
                return int(p)
    return int(L)


def _seed_bbox_from_component_bbox(
    bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    direction: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    th, tw = tile.shape
    dr, dc = direction
    if dc == 1:
        return r0, c0, r1, c0 + tw - 1
    if dc == -1:
        return r0, c1 - tw + 1, r1, c1
    if dr == 1:
        return r0, c0, r0 + th - 1, c1
    return r1 - th + 1, c0, r1, c1


def _recover_seed_tile(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recover the primitive seed tile from a visible clue prefix or a compact seed."""
    r0, c0, r1, c1 = map(int, bbox)
    strip = crop_bbox(grid, (r0, c0, r1, c1))
    oriented = _orient_strip_from_plus(strip, direction)
    period = _primitive_period_length(oriented, direction)

    if direction[1] != 0:
        tile = oriented[:, :period] if direction[1] == 1 else oriented[:, :period][:, ::-1]
    else:
        tile = oriented[:period, :] if direction[0] == 1 else oriented[:period, :][::-1, :]

    seed_bbox = _seed_bbox_from_component_bbox((r0, c0, r1, c1), tile, direction)
    return np.array(tile, dtype=int), seed_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue marker strips away from '+' anchors by repeating their primitive tile to the boundary.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several plus-like anchor sprites.
    - Each anchor has one attached marker strip; most strips are only a compact seed, while one strip is already shown as a short periodic prefix that stops before the border.

    Rule:
    - For every strip touching a '+', use the touching side to determine the outward direction.
    - Recover the smallest repeating tile consistent with the visible strip, then continue that tile outward until the grid edge.
    - Plus-colored cells stay protected and are never overwritten.

    Key insight: The longer clue strip is not complete; it is just a prefix of the same periodic ray that every seed should generate. The solver must compress that prefix back to its primitive period before extending it.
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
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

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
            tile, seed_bbox = _recover_seed_tile(g, (int(r0), int(c0), int(r1), int(c1)), (int(dr), int(dc)))
            sr0, sc0, sr1, sc1 = seed_bbox

            # Alignment check: seed cross-axis span must match arm tip span.
            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)
            if dc == 0:  # vertical ray
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc])
                seed_cols = list(range(int(sc0), int(sc1) + 1))
                assert seed_cols == nozzle_cols
            else:  # horizontal ray
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col])
                seed_rows = list(range(int(sr0), int(sr1) + 1))
                assert seed_rows == nozzle_rows

            extend_repeating_tile(
                out,
                tile,
                int(sr0),
                int(sc0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator (variant D9 partial clue) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)                 # grid height [min, max)
_W_RANGE = (13, 23)                 # grid width [min, max)
_N_PLUS_RANGE = (2, 4)              # number of plus sprites [min, max)
_SEED_ALONG_CHOICES = [2]           # force travel-axis period 2 for visible partial clues
_SEED_MAX_COLORS = 4                # max distinct colors in a seed tile
_SEED_MAX_CROSS = 2                 # max nozzle cross-axis span for valid seed
_MIN_RAY_UNITS = 2                  # minimum tile repetitions along ray
_PRE_EXT_IDEAL_UNITS = 7            # prefer a long ray so the partial clue can stop well before the edge
_PARTIAL_FULL_PERIODS = [3, 4]      # visible full periods in the clue prefix (including seed)
_PLACEMENT_TRIES = 100              # attempts per plus placement
_GEN_RETRIES = 300                  # outer retry limit in generate_input

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


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return (start, end) inclusive of nozzle cells along the cross-axis."""
    dr, dc = direction
    if dc == 0:  # vertical ray -- cross-axis is columns
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_primitive_axis_period(tile: np.ndarray, direction: tuple[int, int]) -> bool:
    """Reject tiles that collapse to period 1 along the ray axis."""
    if direction[1] != 0:
        return not np.array_equal(tile[:, 0], tile[:, 1])
    return not np.array_equal(tile[0, :], tile[1, :])


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a seed tile whose cross-axis matches the nozzle span and whose period is exactly 2."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k <= 1:
        return None

    for _ in range(30):
        n_colors = int(rng.integers(2, max_k + 1))
        chosen = list(rng.choice(palette, size=n_colors, replace=False))
        vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_primitive_axis_period(tile, direction):
            return tile
    return None


def _compute_ray_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    r0: int,
    c0: int,
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> int:
    """Compute how many full tile repetitions fit along the ray direction."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return (H - (r0 + ph)) // th
    if dr == -1:
        return r0 // th
    if dc == 1:
        return (W - (c0 + pw)) // tw
    return c0 // tw


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Compute (r, c, value) list for the full projected ray (seed + repetitions)."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:  # vertical
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:  # horizontal
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)

    cells: list[tuple[int, int, int]] = []
    if dc == 0:
        r = r0
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, c0 + j, int(tile[i, j])))
            r += step
    else:
        c = c0
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(tile[i, j])))
            c += step

    return cells, int(r0), int(c0)


def _component_axis_length(cells: list[tuple[int, int, int]], direction: tuple[int, int]) -> int:
    coords = np.array([(r, c) for r, c, _ in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(c1 - c0 + 1 if direction[1] != 0 else r1 - r0 + 1)


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
) -> dict | None:
    """Try to place one plus sprite with its ray. Returns info dict or None."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(occ[r, c] != 0 for r, c in plus_cells):
            continue
        if any(plus_occ[r, c] == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
            tile = _build_seed_tile(plus_shape, d, palette, rng)
            if tile is None:
                continue
            units = _compute_ray_units(d, tile, r0, c0, ph, pw, H, W)
            if units < _MIN_RAY_UNITS:
                continue

            cells, sr0, sc0 = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                d,
                tile,
            )
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = plus_color
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = v

            return {
                "direction": d,
                "tile": tile,
                "cells": cells,
                "seed_r0": sr0,
                "seed_c0": sc0,
                "units": units,
                "axis_len": _component_axis_length(cells, d),
            }

    return None


def _visible_prefix_mask(
    ray: dict,
    visible_len: int,
) -> set[tuple[int, int]]:
    """Cells kept from the demo ray: a prefix measured from the plus outward."""
    sr0 = int(ray["seed_r0"])
    sc0 = int(ray["seed_c0"])
    dr, dc = map(int, ray["direction"])
    keep: set[tuple[int, int]] = set()
    for r, c, _ in ray["cells"]:
        if dc == 1:
            dist = int(c) - sc0
        elif dc == -1:
            dist = (sc0 + ray["tile"].shape[1] - 1) - int(c)
        elif dr == 1:
            dist = int(r) - sr0
        else:
            dist = (sr0 + ray["tile"].shape[0] - 1) - int(r)
        if 0 <= dist < int(visible_len):
            keep.add((int(r), int(c)))
    return keep


def _choose_demo_prefix_length(ray: dict, rng: np.random.Generator) -> int | None:
    period = int(ray["tile"].shape[1] if ray["direction"][1] != 0 else ray["tile"].shape[0])
    total_len = int(ray["axis_len"])
    candidates: list[int] = []
    for full_periods in _PARTIAL_FULL_PERIODS:
        for extra in range(1, period):
            visible_len = full_periods * period + extra
            if visible_len < total_len:
                candidates.append(int(visible_len))
    if not candidates:
        return None
    return int(rng.choice(candidates))


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Keep one partially shown clue ray; erase every other ray back to its compact seed."""
    demo_candidates = [
        i
        for i, p in enumerate(pluses)
        if p["units"] >= _PRE_EXT_IDEAL_UNITS and _choose_demo_prefix_length(p, rng) is not None
    ]
    if not demo_candidates:
        demo_candidates = [
            i for i, p in enumerate(pluses) if _choose_demo_prefix_length(p, rng) is not None
        ]
    if not demo_candidates:
        return None

    demo_idx = int(rng.choice(demo_candidates))
    demo_visible_len = _choose_demo_prefix_length(pluses[demo_idx], rng)
    if demo_visible_len is None:
        return None

    inp = out.copy()
    demo_keep = _visible_prefix_mask(pluses[demo_idx], demo_visible_len)
    for i, p in enumerate(pluses):
        th, tw = p["tile"].shape
        sr0, sc0 = p["seed_r0"], p["seed_c0"]
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        keep = demo_keep if i == demo_idx else seed_set
        for r, c, _ in p["cells"]:
            if (r, c) not in keep:
                inp[r, c] = bg
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
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
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, rng)
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
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
