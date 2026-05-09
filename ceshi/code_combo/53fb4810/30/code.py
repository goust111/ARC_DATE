# Combo variant of 53fb4810 (index 30)
# combo_spec: D15/0+D19/1
# Name: 多色精灵干扰射线

from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concretization and behavior trace
# - Original task: one shared plus color marks every anchor, and the input usually keeps one
#   correct long ray while the remaining pluses show only their compact seed tiles.
# - D15 changes the anchor grouping rule: pluses still have the same shape, but each plus uses a
#   different color, so the solver must detect several large monochrome components instead of one
#   dominant color class.
# - D19 changes the input corruption rule: some pluses keep their true outward ray, while others
#   keep the same seed strip but replace the true continuation with a perpendicular distractor ray.
# - In the fused grid, a concrete plus might be color 2 with a seed on its right edge. The true
#   output repeats that seed to the right. The input may instead show a vertical branch rising from
#   the same seed, while another plus of color 7 still shows its correct horizontal clue. The
#   solver must ignore the currently drawn branch direction, recover the seed from the side touching
#   the local plus, and then project away from that plus.
#
# b) Complexity calibration
# - I keep only 2-4 pluses and require them to share one geometry template, so the task does not
#   explode into many object classes.
# - Every generated board contains at least one correct long ray and at least one wrong long ray,
#   making the D19 corruption visible in the same sample rather than across hypothetical cases.
# - Each plus color is unique and seed/ray colors avoid all plus colors. This makes the D15 change
#   visible without adding extra per-color rules.
# - Seed depth along the true ray axis is fixed at 2 cells, matching the single D19 variant. This
#   removes ambiguity when a seed is already attached to a full ray.
#
# c) Implicit-assumption audit of the original main()
# - The original dominant-color detector is invalid because pluses are no longer one color.
# - The original non-bg/non-plus-color marker mask is invalid because other plus colors would be
#   misread as marker material.
# - The original connected-marker bbox logic is invalid under distractor rays because a seed and a
#   long branch form one large component; taking that whole bbox would confuse seed extraction.
# - The original protect={plus_col} overwrite guard is invalid because every plus color must be
#   protected, not just one.
# - The local nozzle-span geometry remains valid. After each plus is isolated by shape and color,
#   the canonical seed strip can still be read directly from the arm tip it touches.
#
# d) Placement legality
# - Output generation remains output-first: place all multi-colored pluses, then paint their true
#   repeated rays, rejecting any placement that collides with another plus or ray.
# - Input corruption happens after the valid output exists. For a chosen wrong-ray plus, the true
#   non-seed continuation is erased and replaced by a perpendicular distractor ray starting from the
#   same seed cells.
# - Distractor rays are chosen by backtracking so they never overlap any plus, any seed, or any
#   still-correct ray.
#
# e) Visual observability
# - A human can see two simultaneous regularities: anchors are identical plus shapes despite using
#   different colors, and the true direction is the side where the seed touches the plus, not the
#   direction of the long branch currently visible.
# - The rule never depends on color ids or hidden sorting; color only separates plus anchors from
#   marker palettes.
#
# f) Visual cleanliness
# - Pluses remain the only large monochrome components with bbox >= 3x3, so they are easy to spot.
# - A one-cell plus-neighborhood buffer prevents accidental extra contacts around the wrong side of
#   a plus.
# - Rays are non-overlapping, so no cell needs to encode two competing objects.
#
# g) Color inferability
# - Plus colors are unique but behaviorally identical. Across 2-4 examples, this visibly supports
#   the idea that color distinguishes anchors rather than changing the projection rule.
# - Marker tiles use at least two colors whenever possible, which preserves the periodic tile cue
#   even when the input contains a distractor branch.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: every plus is a single-color connected component with bbox at least 3x3.
#   Generator guarantee: plus templates are unchanged and each placed plus uses one unique color.
# - Solver assumption: exactly one side of each plus has a fully occupied canonical 2-cell-deep seed
#   strip. Generator guarantee: every plus owns exactly one seed, and spacing rules stop foreign
#   material from occupying another side's canonical strip.
# - Solver assumption: any wrong branch is perpendicular to the true away-from-plus direction.
#   Generator guarantee: distractor directions are chosen only from the perpendicular pair.


def detect_big_plus_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> list[dict]:
    """
    Local helper (only used by this solver): collect every non-bg monochrome component whose bbox
    is at least (min_h x min_w). In this task family, those components are the plus anchors.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps: list[dict] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                comps.append(
                    {
                        "color": int(col),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    }
                )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


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


def _paint_tile(out: np.ndarray, tile: np.ndarray, r0: int, c0: int, *, bg: int) -> None:
    """Stamp the seed tile itself before extending from it."""
    h, w = tile.shape
    for i in range(h):
        for j in range(w):
            v = int(tile[i, j])
            if v != int(bg):
                out[int(r0) + i, int(c0) + j] = v


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


def _seed_shape_for_direction(
    plus_mask: np.ndarray, direction: tuple[int, int], along: int = 2
) -> tuple[int, int]:
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    _dr, dc = direction
    return (int(along), cross) if dc == 0 else (cross, int(along))


def _seed_anchor_from_plus(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile_shape: tuple[int, int],
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile_shape
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)
    return int(r0), int(c0)


def _project_from_seed(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], int]:
    """Return all cells in the repeated ray, including the seed tile itself."""
    th, tw = tile.shape
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []
    units = 0

    if dc == 0:
        r = int(r0)
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            units += 1
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, int(c0) + j, int(tile[i, j])))
            r += step
    else:
        c = int(c0)
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            units += 1
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(r0) + i, cc, int(tile[i, j])))
            c += step

    return cells, int(units)


def _ray_cells_from_plus(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int, int]:
    """Return (cells, seed_r0, seed_c0, units) for the correct ray of one plus."""
    seed_r0, seed_c0 = _seed_anchor_from_plus(plus_bbox, plus_mask, direction, tile.shape)
    cells, units = _project_from_seed(H, W, tile, seed_r0, seed_c0, direction)
    return cells, int(seed_r0), int(seed_c0), int(units)


def _perpendicular_directions(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dc == 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the true outward periodic rays for each differently colored '+' anchor.

    Input structure:
    - The background is the most frequent color.
    - Several large plus-like anchors share one shape but each anchor has its own non-background color.
    - Every plus has exactly one canonical seed strip touching one arm tip; some inputs keep the true
      long ray beyond that seed, while others replace it with a perpendicular distractor branch.

    Rule:
    - Detect every large monochrome plus component independently, regardless of its color.
    - For each plus, locate the unique side whose fixed two-cell-deep strip is fully occupied by
      non-background, non-plus colors; that strip is the true seed.
    - Copy that seed and repeat it away from the plus until the boundary, ignoring any branch drawn
      in another direction and never overwriting any plus color.

    Key insight:
    - Long branches in the input are not trustworthy, and shared color no longer groups the anchors.
    - The reliable cue is purely local geometry: the seed is the compact strip attached to one arm,
      so the correct ray direction is "continue away from that arm" for each colored plus separately.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    pluses = detect_big_plus_components(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if not pluses:
        return g.copy()

    plus_union_mask = np.zeros_like(g, dtype=bool)
    all_plus_colors = {int(comp["color"]) for comp in pluses}
    plus_color_arr = np.array(sorted(all_plus_colors), dtype=int)

    out = np.full_like(g, bg)
    for comp in pluses:
        coords = comp["coords"]
        plus_union_mask[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = int(comp["color"])

    for plus in pluses:
        plus_coords = plus["coords"]
        if plus_coords.size == 0:
            continue

        pr0, pc0, pr1, pc1 = plus["bbox"]
        plus_mask = np.zeros_like(g, dtype=bool)
        plus_mask[plus_coords[:, 0], plus_coords[:, 1]] = True
        local_plus = crop_bbox(
            plus_mask.astype(int), (int(pr0), int(pc0), int(pr1), int(pc1))
        ).astype(bool)

        candidates: list[tuple[tuple[int, int], int, int, int, int]] = []
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            th, tw = _seed_shape_for_direction(local_plus, direction, along=2)
            sr0, sc0 = _seed_anchor_from_plus((pr0, pc0, pr1, pc1), local_plus, direction, (th, tw))
            if sr0 < 0 or sc0 < 0 or sr0 + th > H or sc0 + tw > W:
                continue
            seed = g[sr0 : sr0 + th, sc0 : sc0 + tw]
            if np.any(plus_union_mask[sr0 : sr0 + th, sc0 : sc0 + tw]):
                continue
            if np.any(seed == int(bg)):
                continue
            if np.any(np.isin(seed, plus_color_arr)):
                continue
            candidates.append((direction, int(sr0), int(sc0), int(th), int(tw)))

        if len(candidates) != 1:
            continue

        direction, seed_r0, seed_c0, th, tw = candidates[0]
        tile = g[seed_r0 : seed_r0 + th, seed_c0 : seed_c0 + tw].copy()
        _paint_tile(out, tile, seed_r0, seed_c0, bg=bg)
        extend_repeating_tile(
            out,
            tile,
            seed_r0,
            seed_c0,
            direction,
            bg=bg,
            protect=set(all_plus_colors),
        )

    return out


# --------------- generator (combo D15/0 + D19/1) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 5)
_SEED_ALONG = 2
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PLACEMENT_TRIES = 120
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


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a dense seed tile with fixed depth 2 and cross-axis span matching the nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    _dr, dc = direction
    th, tw = (_SEED_ALONG, cross) if dc == 0 else (cross, _SEED_ALONG)
    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k < 2:
        return None

    n_colors = int(rng.integers(2, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen)
    vals.extend(int(rng.choice(chosen)) for _ in range(n_cells - n_colors))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


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
    """Place one plus with its correct output ray."""
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
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
            if tile is None:
                continue

            cells, sr0, sc0, units = _ray_cells_from_plus(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape.astype(bool),
                direction,
                tile,
            )
            if units < _MIN_RAY_UNITS:
                continue
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            seed_h, seed_w = tile.shape
            seed_set = {(sr0 + i, sc0 + j) for i in range(seed_h) for j in range(seed_w)}
            return {
                "plus_color": int(plus_color),
                "plus_bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
                "plus_mask": plus_shape.astype(bool).copy(),
                "plus_cells": plus_cells,
                "direction": tuple(direction),
                "tile": tile.copy(),
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "seed_set": seed_set,
                "correct_extra": [(r, c, v) for r, c, v in cells if (r, c) not in seed_set],
                "units": int(units),
            }

    return None


def _compute_wrong_candidates(info: dict, H: int, W: int) -> list[dict]:
    out: list[dict] = []
    tile = np.array(info["tile"], dtype=int)
    sr0 = int(info["seed_r0"])
    sc0 = int(info["seed_c0"])
    seed_set = set(info["seed_set"])

    for wrong_dir in _perpendicular_directions(tuple(info["direction"])):
        cells, units = _project_from_seed(H, W, tile, sr0, sc0, wrong_dir)
        if units < _MIN_RAY_UNITS:
            continue
        extra = [(r, c, v) for r, c, v in cells if (r, c) not in seed_set]
        if not extra:
            continue
        out.append(
            {
                "direction": tuple(wrong_dir),
                "cells": cells,
                "extra": extra,
                "extra_set": {(r, c) for r, c, _ in extra},
                "units": int(units),
            }
        )
    return out


def _choose_wrong_assignments(
    pluses: list[dict],
    H: int,
    W: int,
    rng: np.random.Generator,
) -> dict[int, dict] | None:
    n = len(pluses)
    all_ids = list(range(n))
    base_plus_cells = {cell for info in pluses for cell in info["plus_cells"]}
    base_seed_cells = {cell for info in pluses for cell in info["seed_set"]}
    wrong_candidates = [_compute_wrong_candidates(info, H, W) for info in pluses]

    subset_sizes = list(range(1, n))
    rng.shuffle(subset_sizes)

    for k in subset_sizes:
        subset_order = [tuple(s) for s in combinations(all_ids, k)]
        rng.shuffle(subset_order)

        for wrong_ids in subset_order:
            wrong_set = set(wrong_ids)
            if any(not wrong_candidates[i] for i in wrong_ids):
                continue

            occupied = set(base_plus_cells) | set(base_seed_cells)
            for i in all_ids:
                if i in wrong_set:
                    continue
                occupied.update((r, c) for r, c, _ in pluses[i]["correct_extra"])

            order = list(wrong_ids)
            rng.shuffle(order)
            chosen: dict[int, dict] = {}

            def backtrack(pos: int, occ_cells: set[tuple[int, int]]) -> bool:
                if pos == len(order):
                    return True
                idx = order[pos]
                cand_list = list(wrong_candidates[idx])
                rng.shuffle(cand_list)
                for cand in cand_list:
                    if cand["extra_set"] & occ_cells:
                        continue
                    chosen[idx] = cand
                    if backtrack(pos + 1, occ_cells | cand["extra_set"]):
                        return True
                    chosen.pop(idx, None)
                return False

            if backtrack(0, occupied):
                return chosen

    return None


def _build_variant_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]

    max_pluses = min(_N_PLUS_RANGE[1] - 1, len(others) - 2)
    if max_pluses < _N_PLUS_RANGE[0]:
        return None
    n_plus = int(rng.integers(_N_PLUS_RANGE[0], max_pluses + 1))

    plus_colors = list(map(int, rng.choice(others, size=n_plus, replace=False)))
    marker_palette = [c for c in others if c not in plus_colors]
    if len(marker_palette) < 2:
        return None

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for plus_color in plus_colors:
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            marker_palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            int(plus_color),
        )
        if info is None:
            return None
        pluses.append(info)

    assignments = _choose_wrong_assignments(pluses, H, W, rng)
    if assignments is None or not assignments or len(assignments) == len(pluses):
        return None

    inp = out.copy()
    for idx, cand in assignments.items():
        for r, c, _ in pluses[idx]["correct_extra"]:
            inp[r, c] = int(bg)
        for r, c, v in cand["extra"]:
            inp[r, c] = int(v)

    meta = {
        "bg": int(bg),
        "plus_count": len(pluses),
        "plus_colors": [int(info["plus_color"]) for info in pluses],
        "wrong_count": len(assignments),
        "correct_count": len(pluses) - len(assignments),
        "true_dirs": [tuple(info["direction"]) for info in pluses],
        "wrong_dirs": {int(i): tuple(assignments[i]["direction"]) for i in assignments},
    }
    return inp, out, meta


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict] | None:
    built = _build_variant_pair(rng)
    if built is None:
        return None
    inp, out, meta = built

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp, out, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
