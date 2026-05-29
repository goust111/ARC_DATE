# Combo variant of 53fb4810 (index 13)
# combo_spec: D9/0+D11/0
# Name: 无线索多射线

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
# - Original task: each '+' owns exactly one marker tile, and one ray somewhere in the
#   input is already fully expanded as an in-grid example.
# - D9/0 removes that clue: every marker must appear only as a compact seed tile.
# - D11/0 multiplies ownership: one '+' now carries 2-4 separate seeds on different
#   arms, and each seed independently repeats away from the same '+' in the output.
# - Small trace:
#   - Original: one plus with a right seed [2, 3] gives one rightward periodic ray;
#     another plus may show the full expanded ray as a hint.
#   - Combo: the same plus may have an up seed [[4], [5]], a right seed [2, 3], and a
#     left seed [7, 6]; the input shows only those three compact seeds, while the output
#     extends all three rays to the boundary. No grid contains a ready-made long ray.
#
# b) Complexity calibration
# - Each plus uses 2-4 directions, capped by the four compass arms, so the multi-ray
#   behavior is obvious but still readable.
# - Seed tiles stay in the original small family induced by nozzle width: 1x2, 2x1,
#   or 2x2. This preserves the original concept instead of adding a second pattern rule.
# - Tiles on the same plus are forced to have distinct signatures, which makes the
#   multiple attached seeds visibly intentional rather than looking like duplicates.
#
# c) Implicit-assumption audit for main()
# - `detect_dominant_big_component_color(...)` assumes the pluses remain the only large
#   repeated components. This still holds because removing the clue makes non-plus
#   structures smaller, not larger.
# - `mark_lab = label_mask(non_mask, connectivity=4)` assumes each logical seed/ray is a
#   separate 4-connected component. With multiple rays per plus, the generator must now
#   prevent both overlap and 4-neighbor contact between distinct rays.
# - The `adj_ids` collection loop already supports multiple marker components touching one
#   plus; the original generator simply never exercised that path. The combo now relies on it.
# - `len(dirs) == 1` assumes one component touches exactly one arm of exactly one plus.
#   The generator therefore forbids any ray cell from 4-touching a foreign plus.
# - The nozzle-alignment checks assume the seed span matches the arm-tip span exactly.
#   Generation still constructs seeds from the nozzle span, so this remains valid.
#
# d) Placement legality
# - Each plus must be isolated from every existing plus or ray by at least one background
#   cell in the 4-neighborhood.
# - Each candidate ray must stay in bounds, must not overlap existing content, and must not
#   4-touch any foreign ray or foreign plus.
# - A plus candidate is accepted only if all of its chosen rays fit simultaneously.
#
# e) Visual observability
# - Humans can infer the fused rule from 2-4 examples because the change is geometric:
#   multiple seeds visibly hang off the same plus, and none of them is already expanded.
# - No hidden numeric property of color ids is used. Colors are copied literally from each
#   seed tile into its repeated ray.
#
# f) Visual cleanliness
# - Output-first construction plus strict isolation avoids merged stripes and ambiguous
#   attachments. The input is even cleaner because every ray tail is erased back to bg.
# - Grid size is slightly enlarged relative to the single-ray task so 2-4 rays per plus
#   can coexist without crowding.
#
# g) Inferability of the changed behavior
# - The two fused dimensions are directly measurable in the input:
#   1. every attached marker component is compact (no exemplar clue),
#   2. each plus has multiple adjacent marker components.
# - Because seeds remain small and aligned to arm tips, the solver can infer the same
#   repeat-to-boundary rule from training pairs without needing any extra latent mapping.


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
    Grow every compact seed attached to a plus-like anchor into its full outward ray.

    Input structure:
    - The most common color is the background.
    - One dominant non-background color forms several identical '+'-style anchor sprites.
    - A single plus can have several separate adjacent marker components on different arms,
      and all of those markers are compact seeds only: no ray is pre-expanded as a clue.

    Rule:
    - For each non-plus component that edge-touches a plus, infer which arm it is attached to.
    - Treat that component's bounding box as the seed tile and repeat it periodically away
      from the plus until the grid boundary.
    - Preserve plus pixels while drawing, so only background gets replaced by repeated tiles.

    Key insight:
    - The task factors by attached component rather than by whole sprite. With no in-grid
      exemplar ray, the only reliable signal is local geometry: each seed's attachment side
      fixes both its direction and the cross-axis alignment of the repeated pattern.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(
        g, bg=bg, min_h=3, min_w=3, connectivity=4
    )
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


# --------------- generator (combo: clue-free multi-ray) ---------------

_H_RANGE = (16, 31)
_W_RANGE = (16, 27)
_N_PLUS_RANGE = (2, 4)
_MIN_MARKERS_PER_PLUS = 2
_MAX_MARKERS_PER_PLUS = 4
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
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

    _dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(th, tw)

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

    This preserves the solver assumption that every non-plus component corresponds to
    exactly one seed/ray attached to exactly one arm.
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
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    k = int(k)
    max_k = min(_MAX_MARKERS_PER_PLUS, len(directions))
    min_k = min(_MIN_MARKERS_PER_PLUS, max_k)
    if not (min_k <= k <= max_k):
        raise ValueError(f"_sample_direction_subset: invalid k={k} for {len(directions)} directions")
    order = directions.copy()
    rng.shuffle(order)
    return order[:k]


def _sample_marker_counts(n_plus: int, rng: np.random.Generator) -> list[int]:
    """Sample exact per-plus ray counts with visible branch coverage but manageable density."""
    counts = [_MIN_MARKERS_PER_PLUS] * int(n_plus)
    order = list(range(int(n_plus)))
    rng.shuffle(order)

    counts[order[0]] = 4 if float(rng.random()) < 0.25 else 3
    if n_plus >= 3 and float(rng.random()) < 0.35:
        counts[order[1]] = 3
    elif n_plus >= 2 and float(rng.random()) < 0.1:
        counts[order[1]] = 3

    return counts


def _place_plus_and_rays(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    marker_count: int,
) -> dict | None:
    """Try to place one plus sprite with an exact number of distinct rays."""
    ph, pw = plus_shape.shape
    valid_dirs = _valid_directions_for_shape(plus_shape)
    marker_count = int(marker_count)
    if len(valid_dirs) < marker_count:
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
        own_plus_cells = {(int(r), int(c)) for r, c in plus_cells}

        for _ in range(_SUBSET_TRIES):
            chosen_dirs = _sample_direction_subset(valid_dirs, marker_count, rng)
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

                    cells, sr0, sc0 = _project_tile_coords(
                        H, W, plus_bbox, plus_shape, direction, tile
                    )
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


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every ray tail so the input keeps only compact seeds on every arm."""
    inp = out.copy()
    for plus in pluses:
        for ray in plus["rays"]:
            th, tw = ray["tile"].shape
            sr0 = int(ray["seed_r0"])
            sc0 = int(ray["seed_c0"])
            seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
            for r, c, _ in ray["cells"]:
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

    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    marker_counts = _sample_marker_counts(n_plus, rng)
    max_marker_count = max(marker_counts)
    eligible_shapes = [
        shape for shape in _PLUS_SHAPES if len(_valid_directions_for_shape(shape)) >= max_marker_count
    ]
    plus_shape = eligible_shapes[int(rng.integers(0, len(eligible_shapes)))]

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    pluses: list[dict] = []

    for marker_count in marker_counts:
        info = _place_plus_and_rays(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            out,
            plus_color,
            marker_count=int(marker_count),
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg)

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
