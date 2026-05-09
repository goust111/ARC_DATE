# Variant of 53fb4810
# Dimension: D11 — 每个精灵的标记数量 (meta)
# Change: 每个'+'精灵恰好附着一个标记瓦片（一个方向一条射线）... → 每个精灵多个标记

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D11 variant
# ===================================
# a) Concrete behavior and pipeline impact
# - Original task: each '+' sprite owns exactly one marker tile, so one sprite -> one ray.
# - D11 variant: each '+' owns 2-4 separate marker components on different arms.
#   Each component is still a compact tile whose bbox aligns with that arm's nozzle span,
#   and each component independently repeats outward away from the plus.
# - Input therefore contains one plus with multiple visible seeds around it; exactly one
#   ray in the whole grid is kept pre-extended as the periodic clue, while every other ray
#   is collapsed back to just its seed tile.
# - Small trace example:
#   Original flow: plus at center, only right seed [2,3] -> output extends rightward.
#   Variant flow: same plus may have up seed [[4],[5]], right seed [2,3], left seed [7,6];
#   output extends three independent periodic rays. The solver stage affected by D11 is the
#   "collect adjacent marker components for each plus" loop: it must process every component
#   touching that plus, not stop after the first one.
#
# b) Complexity calibration
# - Marker count is limited to 2-4 directions, matching the four compass arms and keeping
#   the branching human-readable.
# - The tile shapes remain the same family as the original task: cross-axis must match the
#   nozzle span; along-axis is 1 or 2. This keeps the rule recognizable instead of adding
#   a second new idea.
# - Tiles on the same plus are forced to be pairwise distinct, so the multi-marker variant
#   is visually obvious and does not collapse into "the same sticker copied around".
#
# c) Implicit assumptions from the original main()
# - `mark_lab = label_mask(non_mask, connectivity=4)` assumes each logical marker/ray is one
#   isolated 4-connected component. With D11, this remains valid only if rays from different
#   directions never touch edge-to-edge. Generator now enforces both non-overlap and
#   non-adjacency between distinct rays.
# - `adj_ids` may contain multiple components for one plus; this was already supported by the
#   loop structure, but the original generator never exercised it. D11 relies on that path.
# - `len(dirs) == 1` assumes one component touches exactly one side of exactly one plus. This
#   breaks if a foreign ray runs next to another plus. Generator therefore forbids any ray
#   cell from being 4-neighbor to a non-own plus cell.
# - The nozzle-span alignment checks assume the seed tile is flush with the arm tip. Generator
#   still constructs seeds from the nozzle span, so this assumption is preserved.
#
# d) Placement legality
# - Pluses must be disjoint and not 4-touch any other plus.
# - Every ray cell must stay inside bounds, must not overlap any existing ray or plus, and
#   must not 4-touch a foreign plus or a foreign ray component.
# - The whole plus-with-rays bundle is planned before commit, so a candidate is accepted only
#   if all of its chosen directions are simultaneously valid.
#
# e) Visual observability
# - Humans can now see several attached tiles around the same plus in the input; after looking
#   at the single long exemplar ray, it is natural to project that repetition rule onto the
#   other attached tiles as well.
# - No color-id ordering or hidden numeric convention is used. All choices are geometric:
#   side of attachment determines direction; local tile pattern determines repeated content.
#
# f) Visual cleanliness
# - To keep the grid readable, the number of pluses stays small and ray components are kept
#   separated by at least one background cell from foreign structures.
# - Only one ray is pre-extended in the input, which avoids turning the input into a dense
#   thicket of stripes while still exposing the periodic repetition rule.
#
# g) Color inferability
# - Colors are copied, not remapped. A tile's visible internal pattern is exactly what repeats.
# - Palette size per tile stays small, so a human can match "seed pattern" -> "same pattern
#   periodically repeated" without having to infer any extra color transformation.


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
    """Extend a tile pattern from (r0,c0) until the boundary."""
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

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every marker tile attached to a '+' sprite by periodic repetition to the boundary.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical '+'-like sprites.
    - Around a single plus there may be several separate marker components, each attached to a
      different arm; one component somewhere in the grid may already appear as a long repeated ray.

    Rule:
    - For each non-plus component that is edge-adjacent to a '+' sprite, infer the side where it
      touches that sprite.
    - Treat that component's bounding box as the seed tile, and repeat it outward in the away-from-
      plus direction until the grid edge.
    - Plus pixels are protected, so repeated tiles may fill background but never overwrite the plus.

    Key insight: the task decomposes by attached component, not by whole sprite: a single plus can
    own multiple independent seeds, and each seed defines its own ray by attachment side.
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

        pr0, pc0 = plus_coords.min(axis=0)
        pr1, pc1 = plus_coords.max(axis=0)

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

            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = [
                    int(cc)
                    for cc in range(int(pc0), int(pc1) + 1)
                    if plus_mask[nozzle_row, cc]
                ]
                seed_cols = list(range(int(c0), int(c1) + 1))
                if seed_cols != nozzle_cols:
                    continue
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = [
                    int(rr)
                    for rr in range(int(pr0), int(pr1) + 1)
                    if plus_mask[rr, nozzle_col]
                ]
                seed_rows = list(range(int(r0), int(r1) + 1))
                if seed_rows != nozzle_rows:
                    continue

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator (D11 multi-marker variant) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (16, 31)
_W_RANGE = (16, 27)
_N_PLUS_RANGE = (2, 4)
_MIN_MARKERS_PER_PLUS = 2
_MAX_MARKERS_PER_PLUS = 4
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 9
_PLACEMENT_TRIES = 180
_SUBSET_TRIES = 16
_TILE_TRIES = 20
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
_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return inclusive cross-axis span of the nozzle cells for one arm."""
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
    """Build a seed tile whose cross-axis matches the selected nozzle span."""
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
    tile = np.array(vals, dtype=int).reshape(th, tw)

    # Avoid monotone duplicates when possible: at least two colors or two positions differ.
    if n_cells > 1 and np.all(tile == tile.flat[0]) and len(palette) >= 2:
        alt = int(rng.choice([c for c in palette if c != int(tile.flat[0])]))
        tile.flat[-1] = alt
    return tile


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
    """Compute how many tile repetitions fit along the ray direction."""
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
    """Compute full ray cells (seed + repeated copies) and seed origin."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
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


def _plus_cells_fit(occ: np.ndarray, plus_cells: list[tuple[int, int]]) -> bool:
    """Candidate plus cells must be isolated from existing pluses and rays."""
    H, W = occ.shape
    for r, c in plus_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) != 0:
                return False
    return True


def _ray_cells_fit(
    trial_occ: np.ndarray,
    cells: list[tuple[int, int, int]],
    own_plus_cells: set[tuple[int, int]],
) -> bool:
    """
    Ray cells must stay isolated from foreign structures.

    This enforces the main() assumption that each marker/ray is one separate connected
    component touching exactly one plus arm.
    """
    H, W = trial_occ.shape
    cell_pos = {(int(r), int(c)) for r, c, _ in cells}
    if len(cell_pos) != len(cells):
        return False

    for r, c, _ in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(trial_occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in cell_pos:
                continue
            val = int(trial_occ[rr, cc])
            if val == 2:
                return False
            if val == 1 and (rr, cc) not in own_plus_cells:
                return False
    return True


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.flatten()))


def _valid_directions_for_shape(plus_shape: np.ndarray) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for direction in _DIRECTIONS:
        s, e = _nozzle_span(plus_shape, direction)
        if e - s + 1 <= _SEED_MAX_CROSS:
            valid.append(direction)
    return valid


def _sample_direction_subset(
    directions: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    max_k = min(_MAX_MARKERS_PER_PLUS, len(directions))
    min_k = min(_MIN_MARKERS_PER_PLUS, max_k)
    k = int(rng.integers(min_k, max_k + 1))
    order = directions.copy()
    rng.shuffle(order)
    return order[:k]


def _place_plus_and_rays(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
) -> dict | None:
    """Try to place one plus sprite with 2-4 distinct rays."""
    ph, pw = plus_shape.shape
    valid_dirs = _valid_directions_for_shape(plus_shape)
    if len(valid_dirs) < _MIN_MARKERS_PER_PLUS:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]
        if not _plus_cells_fit(occ, plus_cells):
            continue

        plus_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        own_plus_cells = set((int(r), int(c)) for r, c in plus_cells)

        for _ in range(_SUBSET_TRIES):
            chosen_dirs = _sample_direction_subset(valid_dirs, rng)
            trial_occ = occ.copy()
            for r, c in plus_cells:
                trial_occ[r, c] = 1

            used_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
            rays: list[dict] = []
            success = True

            for direction in chosen_dirs:
                ray_info = None
                for _ in range(_TILE_TRIES):
                    tile = _build_seed_tile(plus_shape, direction, palette, rng)
                    if tile is None:
                        break
                    sig = _tile_signature(tile)
                    if sig in used_tiles:
                        continue

                    units = _compute_ray_units(direction, tile, r0, c0, ph, pw, H, W)
                    if units < _MIN_RAY_UNITS:
                        continue

                    cells, sr0, sc0 = _project_tile_coords(H, W, plus_bbox, plus_shape, direction, tile)
                    if not _ray_cells_fit(trial_occ, cells, own_plus_cells):
                        continue

                    for rr, cc, _v in cells:
                        trial_occ[rr, cc] = 2
                    ray_info = {
                        "direction": tuple(map(int, direction)),
                        "tile": tile,
                        "cells": cells,
                        "seed_r0": int(sr0),
                        "seed_c0": int(sc0),
                        "units": int(units),
                    }
                    used_tiles.add(sig)
                    break

                if ray_info is None:
                    success = False
                    break
                rays.append(ray_info)

            if not success:
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = plus_color
            for ray in rays:
                for rr, cc, vv in ray["cells"]:
                    occ[rr, cc] = 2
                    out[rr, cc] = int(vv)
            return {
                "plus_bbox": plus_bbox,
                "marker_count": len(rays),
                "rays": rays,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full exemplar ray; collapse every other ray back to its seed tile."""
    all_rays: list[tuple[int, int, dict]] = []
    for pi, plus in enumerate(pluses):
        for ri, ray in enumerate(plus["rays"]):
            all_rays.append((pi, ri, ray))

    pre_idx = max(
        range(len(all_rays)),
        key=lambda i: (int(pluses[all_rays[i][0]]["marker_count"]), int(all_rays[i][2]["units"])),
    )
    strong = [
        i
        for i, (pi, _ri, ray) in enumerate(all_rays)
        if int(pluses[pi]["marker_count"]) >= 3 and int(ray["units"]) >= _PRE_EXT_IDEAL_UNITS
    ]
    if strong:
        pre_idx = int(rng.choice(strong))

    keep_pi, keep_ri, _keep_ray = all_rays[pre_idx]
    inp = out.copy()
    for pi, plus in enumerate(pluses):
        for ri, ray in enumerate(plus["rays"]):
            if pi == keep_pi and ri == keep_ri:
                continue
            th, tw = ray["tile"].shape
            sr0 = int(ray["seed_r0"])
            sc0 = int(ray["seed_c0"])
            seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
            for r, c, _ in ray["cells"]:
                if (int(r), int(c)) not in seed_set:
                    inp[r, c] = int(bg)
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
    pluses: list[dict] = []

    for _ in range(N):
        info = _place_plus_and_rays(rng, plus_shape, palette, H, W, occ, out, plus_color)
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, rng)

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
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
