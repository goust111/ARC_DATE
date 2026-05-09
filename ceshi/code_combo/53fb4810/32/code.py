# Combo variant of 53fb4810 (index 32)
# combo_spec: D9/0+D18/0
# Name: 无线索边缘编码
#
# Design analysis
# a) Concrete behavior and trace:
#    - Original 53fb4810 uses local geometry twice: a compact seed touches one side of a '+' and that touching side
#      itself determines the ray direction; additionally, one ray is kept fully expanded in the input as an in-grid
#      clue. The solver can copy that demonstrated expansion logic to the other seeds.
#    - In this combo, both of those supports are removed or replaced. Every plus still has exactly one compact seed
#      tile adjacent to one arm, but no seed is pre-expanded in the input. Instead, a dedicated control color appears
#      only on the border as thin strips. The border side encodes the ray direction: top=up, bottom=down,
#      left=left, right=right. The strip span matches the seed bbox cross-axis span so the solver can pair them.
#    - Example trace: a 2x1 seed may sit below a plus because it attaches to the bottom nozzle, but a 2-cell strip on
#      the left edge aligned to those same two rows instructs the solver to project that seed leftward. The original
#      task would fire downward and might also show one long example ray elsewhere; this combo fires leftward and
#      shows no long clue anywhere in the input.
# b) Complexity calibration:
#    - Direction still comes from only four visible categories, so the added symbolic layer is low-entropy.
#    - Matching is kept human-readable by using exact row-span / column-span alignment between each seed and its
#      border strip. The generator enforces that those spans are unique within a grid, avoiding ambiguous pairing.
#    - The payload tiles stay in the original small family driven by plus nozzle width: 1x2, 2x1, or 2x2. This keeps
#      the final repeated pattern easy to recognize without creating large noisy seeds.
# c) Implicit-assumption audit against the original solver:
#    - The old `main()` assumed "payload-touching side == ray direction". That is false here, so direction must be
#      decoded from control strips instead of adjacency.
#    - The old nozzle alignment assert used the inferred direction; after fusion that would be wrong because the seed
#      may attach on one side but fire in another direction. The solver now uses adjacency only to find ownership
#      (which seed belongs to which plus), not to infer where to project.
#    - The old generator left one complete ray in the input. That assumption is removed entirely: every non-control
#      marker component in the input is just a compact seed.
#    - New assumption: there is exactly one border-only control color and every plus has exactly one adjacent
#      non-control seed component. The generator explicitly guarantees both.
# d) Placement legality:
#    - Generation is still output-first: place pluses, seeds, border controls, and full rays into the output; then
#      erase every ray tail back to background to build the input.
#    - Pluses are kept away from the border so control strips are not adjacent to them. Occupancy checks prevent
#      overlaps among pluses, seed/ray cells, and border strips; a protected neighborhood mask also keeps rays from
#      crossing through another plus.
#    - Row spans and column spans of seeds are globally unique so each seed matches exactly one control strip.
# e) Visual observability:
#    - Because no long ray is shown in the input, the only recoverable direction cue is the border symbol, which
#      forces the intended D9+D18 reasoning chain across training examples.
#    - The generator rejects samples unless at least one seed fires in a direction different from its attachment side
#      and unless at least two ray directions appear in the same grid. That makes the symbolic rule visually evident.
# f) Visual cleanliness:
#    - The extra symbol is only a one-cell-thick border strip in one dedicated color. It stays on the frame, so the
#      central scene remains a clean arrangement of pluses plus compact seeds.
#    - Border controls are protected during projection, so the edge code remains legible in both input and output.
# g) Change inferability:
#    - No hidden color ordering is used. The control color is identified structurally as the unique non-background
#      color whose occurrences are all on the border, and its meaning comes from position on the border rather than
#      from the numeric color id.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: control strips are simple 4-connected border components. Generator places only straight 1-cell-thick
#      strips on a single border edge.
#    - Assumption: each plus has exactly one adjacent seed component after removing plus/control colors. Generator
#      places one compact rectangular seed per plus and prevents accidental adjacency/merging.
#    - Assumption: each seed bbox matches exactly one control strip by aligned rows or columns. Generator enforces
#      unique seed row spans and unique seed column spans across the grid.

from __future__ import annotations

import numpy as np

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


def detect_border_only_control_color(
    grid: np.ndarray,
    *,
    bg: int,
    exclude: set[int] | None = None,
) -> int | None:
    """Pick the unique non-bg color whose every occurrence lies on the border."""
    exclude = set() if exclude is None else set(map(int, exclude))
    H, W = grid.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    candidates: list[int] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)} - exclude):
        pts = np.argwhere(grid == int(col))
        if pts.size == 0:
            continue
        if all(bool(border[int(r), int(c)]) for r, c in pts):
            candidates.append(int(col))

    return int(candidates[0]) if len(candidates) == 1 else None


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


def _collect_control_specs(ctrl_mask: np.ndarray) -> list[dict]:
    """Describe each border control strip as a direction + aligned span."""
    H, W = ctrl_mask.shape
    ctrl_lab, cn = label_mask(ctrl_mask, connectivity=4)
    specs: list[dict] = []
    for k in range(1, int(cn) + 1):
        pts = np.argwhere(ctrl_lab == int(k))
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if int(r0) == 0 and int(r1) == 0:
            specs.append({"direction": (-1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(r0) == int(H - 1) and int(r1) == int(H - 1):
            specs.append({"direction": (1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(c0) == 0 and int(c1) == 0:
            specs.append({"direction": (0, -1), "span": tuple(range(int(r0), int(r1) + 1))})
        elif int(c0) == int(W - 1) and int(c1) == int(W - 1):
            specs.append({"direction": (0, 1), "span": tuple(range(int(r0), int(r1) + 1))})
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every compact seed tile to the grid edge using a direction decoded from border controls.

    Input structure:
    - The background is the dominant color.
    - One dominant non-background color forms several identical plus-like anchors.
    - Each plus has exactly one adjacent compact seed component, and a dedicated control color appears only as thin strips on the grid border.

    Rule:
    - Pair each seed with the unique border strip whose aligned span matches that seed bbox: top/bottom strips match by columns, left/right strips by rows.
    - Use the border side of that matched strip as the ray direction, then repeat the seed's bbox tile periodically from its current position until the boundary.
    - Plus pixels and control-strip pixels are protected and remain unchanged.

    Key insight:
    - The seed's touching side only tells you which plus it belongs to; it no longer tells you where to fire. With no pre-expanded example ray available, the border encoding is the only direction cue.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    ctrl_color = detect_border_only_control_color(g, bg=bg, exclude={int(plus_col)})
    if ctrl_color is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    ctrl_specs = _collect_control_specs(g == int(ctrl_color))

    non_mask = (g != int(bg)) & (g != int(plus_col)) & (g != int(ctrl_color))
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

        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        if pts.size == 0:
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
        span_rows = tuple(range(int(r0), int(r1) + 1))
        span_cols = tuple(range(int(c0), int(c1) + 1))

        dirs: list[tuple[int, int]] = []
        for spec in ctrl_specs:
            dr, dc = spec["direction"]
            if dc == 0 and tuple(spec["span"]) == span_cols:
                dirs.append((int(dr), int(dc)))
            elif dr == 0 and tuple(spec["span"]) == span_rows:
                dirs.append((int(dr), int(dc)))

        dirs = sorted(set(dirs))
        if len(dirs) != 1:
            continue

        dr, dc = dirs[0]
        extend_repeating_tile(
            out,
            tile,
            int(r0),
            int(c0),
            (int(dr), int(dc)),
            bg=int(bg),
            protect={int(plus_col), int(ctrl_color)},
        )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
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


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span for one side of the plus shape."""
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
    attach_direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a small rectangular seed whose cross-axis matches the chosen attachment side."""
    s, e = _nozzle_span(plus_mask, attach_direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None
    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    _dr, dc = attach_direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _anchor_seed_tile(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Place the seed flush against the chosen side of the plus."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = attach_direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, attach_direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)
    return int(r0), int(c0)


def _compute_ray_units_from_seed(
    direction: tuple[int, int],
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    H: int,
    W: int,
) -> int:
    """Count the number of full tile placements from the seed position to the border."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return int((H - seed_r0) // th)
    if dr == -1:
        return int((seed_r0 + th) // th)
    if dc == 1:
        return int((W - seed_c0) // tw)
    return int((seed_c0 + tw) // tw)


def _project_tile_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """Return the full seed+ray cells produced by periodic repetition from the seed bbox."""
    th, tw = tile.shape
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []

    if dc == 0:
        r = int(seed_r0)
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((int(rr), int(seed_c0 + j), int(tile[i, j])))
            r += step
    else:
        c = int(seed_c0)
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(seed_r0 + i), int(cc), int(tile[i, j])))
            c += step

    return cells


def _build_control_cells(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Place a border control strip whose span matches the seed cross-axis span."""
    th, tw = tile.shape
    dr, dc = direction
    if dc == 0:
        row = 0 if dr == -1 else int(H - 1)
        return [(int(row), int(seed_c0 + j)) for j in range(tw)]
    col = 0 if dc == -1 else int(W - 1)
    return [(int(seed_r0 + i), int(col)) for i in range(th)]


def _allowed_ray_directions(attach_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Allow forward or sideways projection, but never back through the plus."""
    opposite = (-int(attach_direction[0]), -int(attach_direction[1]))
    return [d for d in _DIRECTIONS if d != opposite]


def _touches_occupied_neighbor(
    cells: list[tuple[int, int]],
    occ: np.ndarray,
    *,
    blocked_values: set[int],
) -> bool:
    """Check whether candidate cells 4-touch an already occupied forbidden cell."""
    blocked_values = set(map(int, blocked_values))
    cell_set = {(int(r), int(c)) for r, c in cells}
    H, W = occ.shape
    for r, c in cell_set:
        for dr, dc in _DIRECTIONS:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in cell_set:
                continue
            if int(occ[rr, cc]) in blocked_values:
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
    out: np.ndarray,
    plus_color: int,
    ctrl_color: int,
    used_row_spans: set[tuple[int, ...]],
    used_col_spans: set[tuple[int, ...]],
    target_ray_direction: tuple[int, int] | None = None,
    require_direction_mismatch: bool = False,
) -> dict | None:
    """Try to place one plus, one seed, its control strip, and its full output ray."""
    ph, pw = plus_shape.shape
    min_r = 2
    max_r = H - ph - 2
    min_c = 2
    max_c = W - pw - 2
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue
        if _touches_occupied_neighbor(plus_cells, occ, blocked_values={2}):
            continue

        attach_dirs = list(_DIRECTIONS)
        rng.shuffle(attach_dirs)
        for attach_direction in attach_dirs:
            if (
                require_direction_mismatch
                and target_ray_direction is not None
                and tuple(map(int, attach_direction)) == tuple(map(int, target_ray_direction))
            ):
                continue

            tile = _build_seed_tile(plus_shape, attach_direction, palette, rng)
            if tile is None:
                continue
            seed_r0, seed_c0 = _anchor_seed_tile(
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                attach_direction,
                tile,
            )
            th, tw = tile.shape
            if seed_r0 < 1 or seed_c0 < 1 or seed_r0 + th > H - 1 or seed_c0 + tw > W - 1:
                continue
            seed_cells = [
                (int(seed_r0 + ii), int(seed_c0 + jj))
                for ii in range(th)
                for jj in range(tw)
            ]
            if any(int(plus_occ[r, c]) == 1 for r, c in seed_cells):
                continue

            seed_rows = tuple(range(int(seed_r0), int(seed_r0 + th)))
            seed_cols = tuple(range(int(seed_c0), int(seed_c0 + tw)))
            if seed_rows in used_row_spans or seed_cols in used_col_spans:
                continue

            if target_ray_direction is not None:
                allowed = {tuple(map(int, d)) for d in _allowed_ray_directions(attach_direction)}
                target = tuple(map(int, target_ray_direction))
                if target not in allowed:
                    continue
                ray_dirs = [target]
            else:
                ray_dirs = [d for d in _allowed_ray_directions(attach_direction) if d != attach_direction]
                rng.shuffle(ray_dirs)
                ray_dirs.append(attach_direction)

            for ray_direction in ray_dirs:
                units = _compute_ray_units_from_seed(
                    ray_direction, tile, int(seed_r0), int(seed_c0), H, W
                )
                if units < _MIN_RAY_UNITS:
                    continue

                cells = _project_tile_coords(H, W, tile, int(seed_r0), int(seed_c0), ray_direction)
                if any(int(occ[r, c]) != 0 for r, c, _ in cells):
                    continue
                if any(int(plus_occ[r, c]) == 1 for r, c, _ in cells):
                    continue
                ray_cells = [(int(r), int(c)) for r, c, _ in cells]
                if _touches_occupied_neighbor(ray_cells, occ, blocked_values={2}):
                    continue

                control_cells = _build_control_cells(
                    H, W, tile, int(seed_r0), int(seed_c0), ray_direction
                )
                if any(int(occ[r, c]) != 0 for r, c in control_cells):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    plus_occ[r, c] = 1
                    out[r, c] = int(plus_color)
                    for dr2, dc2 in _DIRECTIONS:
                        rr = int(r + dr2)
                        cc = int(c + dc2)
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[rr, cc] = 1

                for r, c, v in cells:
                    occ[r, c] = 2
                    out[r, c] = int(v)

                control_set = {(int(r), int(c)) for r, c in control_cells}
                for r, c in control_set:
                    occ[r, c] = 3
                    out[r, c] = int(ctrl_color)

                used_row_spans.add(seed_rows)
                used_col_spans.add(seed_cols)
                return {
                    "tile": tile,
                    "cells": cells,
                    "control_set": control_set,
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "attach_direction": tuple(map(int, attach_direction)),
                    "ray_direction": tuple(map(int, ray_direction)),
                }

    return None


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every ray tail so the input keeps only compact seeds plus border controls."""
    inp = out.copy()
    for p in pluses:
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            rc = (int(r), int(c))
            if rc in p["control_set"]:
                continue
            if rc not in seed_set:
                inp[rc] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    ctrl_color = int(rng.choice([c for c in others if c != plus_color]))
    palette = [c for c in others if c not in {plus_color, ctrl_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    n_distinct_dirs = int(rng.integers(2, min(len(_DIRECTIONS), n_plus) + 1))
    chosen_dirs = list(rng.permutation(_DIRECTIONS))[:n_distinct_dirs]
    target_dirs = [chosen_dirs[i % n_distinct_dirs] for i in range(n_plus)]
    rng.shuffle(target_dirs)
    mismatch_idx = int(rng.integers(0, n_plus))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)
    used_row_spans: set[tuple[int, ...]] = set()
    used_col_spans: set[tuple[int, ...]] = set()

    pluses: list[dict] = []
    for idx in range(n_plus):
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
            ctrl_color,
            used_row_spans,
            used_col_spans,
            target_ray_direction=target_dirs[idx],
            require_direction_mismatch=(idx == mismatch_idx),
        )
        if info is None:
            return None
        pluses.append(info)

    mismatches = sum(int(p["attach_direction"] != p["ray_direction"]) for p in pluses)
    distinct_dirs = {tuple(map(int, p["ray_direction"])) for p in pluses}
    if mismatches < 1 or len(distinct_dirs) < 2:
        return None

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
