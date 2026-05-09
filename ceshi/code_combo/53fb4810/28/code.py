# Combo variant of 53fb4810 (index 28)
# combo_spec: D7/0+D8/1
# Name: 源消失仅输出射线区域
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base task: each '+' source has one adjacent seed tile; every seed should repeat away from
#      its source, but the base output keeps the full canvas and preserves the '+' cells.
#    - D7/0 changes the transform: after inferring and drawing the rays, every '+' footprint is
#      erased back to the background.
#    - D8/1 changes the output frame: after source erasure, keep only the minimal rectangle that
#      contains all non-background ray pixels.
#    - Small trace: if one source has a 1x2 seed on its right and another has a 1x2 seed on its
#      left, the full scene after extension spans left-to-right across the grid. In the combo
#      output, the two horizontal rays remain, the '+' footprints become background holes, and the
#      whole answer is cropped to the tight ray bbox instead of staying at input size.
# b) Complexity calibration:
#    - The periodic extension rule, seed sizes, plus-shape family, and one-demo-ray clue all stay
#      intact, so the puzzle still belongs to the original concept class.
#    - To keep the fused behavior observable, generation uses one shared axis per grid and includes
#      both opposite directions on that axis. This makes the crop visible while also ensuring that
#      source erasure affects the final cropped answer rather than disappearing outside the crop.
#    - Freedom remains limited: only a few plus shapes, 2-3 sources, seed sizes 1x2/2x1/2x2, and
#      four cardinal directions, with repeated structure per sample to support human inference.
# c) Review of old assumptions in main():
#    - Detecting the dominant plus color still works because the input still contains the intact
#      '+' sources; only the output removes them.
#    - Marker extraction via `(g != bg) & (g != plus_col)` still assumes marker tiles never use the
#      plus color. The generator preserves that palette separation.
#    - The alignment asserts still require the seed's cross-axis span to match the plus nozzle
#      exactly. The generator keeps the same adjacency and span construction.
#    - The old solver relied on returning the same-size `out`. That assumption is invalid here, so
#      the new solver must explicitly crop after erasing the plus footprints.
# d) Placement legality:
#    - Generation still constructs a collision-free full scene first, checking that full rays do
#      not overlap existing pluses or rays and that all cells stay in bounds.
#    - Input is then derived by keeping exactly one full demo ray and erasing the repeated suffix
#      of all other rays back to background, leaving their seeds attached to intact pluses.
# e) Visual observability:
#    - Across 2-4 examples, a solver can see both effects directly: compact seeds expand into full
#      periodic rays, and the answer is a cropped subgrid containing ray texture only.
#    - The generator rejects cases where cropping would be invisible or where erasing the sources
#      would have no effect after cropping, preventing degeneration into a near-base task.
# f) Visual cleanliness:
#    - Rays remain collision-free and sparse, so the cropped output is still readable. The only
#      interior background holes come from erased plus footprints, which are structurally coherent.
# g) Change inferability:
#    - No hidden color remapping is introduced. The new reasoning burden is purely geometric:
#      identify source/seed direction, repeat the seed, remove the source, then crop to the ray box.
# h) New solver assumptions checked after the change:
#    - Every plus still has exactly one adjacent marker component with a unique direction.
#    - The union of rays is non-empty after erasure, so the final crop bbox is well-defined.
#    - The generator guarantees accepted samples satisfy the new observable conditions:
#      cropped output is smaller than the input grid, and cropping after source erasure differs from
#      cropping the plus-preserving full scene.
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


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
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


def crop_to_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Return the minimal bbox containing all non-bg cells; fallback to 1x1 bg if empty."""
    pts = np.argwhere(grid != int(bg))
    if pts.size == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    return crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every periodic ray from its '+' source, erase the sources, and return only the ray area.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical '+'-like source sprites.
    - Each source has one adjacent non-plus seed tile, and exactly one source already shows a full
      repeated ray as the clue while the others show only their compact seeds.

    Rule:
    - For each seed touching a '+', infer which side of the source it is attached to.
    - Repeat that seed tile away from the source in the corresponding cardinal direction until the
      grid edge, completing every ray.
    - After all rays are reconstructed, erase every '+' pixel to background and crop the result to
      the minimal rectangle containing the remaining ray pixels.

    Key insight: The '+' objects are still needed in the input to decode direction and alignment,
    but they are not part of the final answer; only the repeated ray texture survives, inside its
    own tight bounding box.
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

            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)
            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                seed_cols = list(range(int(c0), int(c1) + 1))
                assert seed_cols == nozzle_cols
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                seed_rows = list(range(int(r0), int(r1) + 1))
                assert seed_rows == nozzle_rows

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
            )

    out[g == int(plus_col)] = int(bg)
    return crop_to_non_bg(out, int(bg))


# --------------- generator (combo variant) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 10
_MIN_OUTPUT_DENSITY = 0.08
_MIN_CROP_DELTA = 2
_PLACEMENT_TRIES = 100
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


def _shape_supports_axis(plus_shape: np.ndarray, axis: str) -> bool:
    dirs = [(1, 0), (-1, 0)] if axis == "vertical" else [(0, 1), (0, -1)]
    for direction in dirs:
        s, e = _nozzle_span(plus_shape, direction)
        if int(e - s + 1) > _SEED_MAX_CROSS:
            return False
    return True


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

    _dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


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
    """Compute (r, c, value) list for the full projected ray (seed + repetitions)."""
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


def _build_direction_plan(axis: str, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return a same-axis direction plan that contains both opposite directions."""
    if axis == "vertical":
        dirs = [(1, 0), (-1, 0)]
    else:
        dirs = [(0, 1), (0, -1)]
    plan = [dirs[0], dirs[1]]
    for _ in range(count - 2):
        plan.append(dirs[int(rng.integers(0, 2))])
    rng.shuffle(plan)
    return plan


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    scene: np.ndarray,
    plus_color: int,
    direction: tuple[int, int],
) -> dict | None:
    """Try to place one plus sprite with one fixed-axis ray. Returns info dict or None."""
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

        tile = _build_seed_tile(plus_shape, direction, palette, rng)
        if tile is None:
            return None
        units = _compute_ray_units(direction, tile, r0, c0, ph, pw, H, W)
        if units < _MIN_RAY_UNITS:
            continue

        cells, sr0, sc0 = _project_tile_coords(
            H, W, (r0, c0, r0 + ph - 1, c0 + pw - 1), plus_shape, direction, tile
        )
        if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
            continue

        for r, c in plus_cells:
            occ[r, c] = 1
            scene[r, c] = plus_color
            for dr2, dc2 in _DIRECTIONS:
                rr, cc = r + dr2, c + dc2
                if 0 <= rr < H and 0 <= cc < W:
                    plus_occ[rr, cc] = 1
        for r, c, v in cells:
            occ[r, c] = 2
            scene[r, c] = v

        return {
            "tile": tile,
            "cells": cells,
            "seed_r0": sr0,
            "seed_c0": sc0,
            "units": units,
            "plus_cells": plus_cells,
            "direction": direction,
        }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return a copy of scene with every source-plus footprint erased to the background."""
    out = scene.copy()
    for p in pluses:
        for r, c in p["plus_cells"]:
            out[r, c] = int(bg)
    return out


def _combo_observable(
    inp: np.ndarray,
    scene: np.ndarray,
    erased_scene: np.ndarray,
    bg: int,
) -> bool:
    """
    Reject weak cases where the fused crop is barely visible or mostly empty.

    The combo should visibly differ from both the full-scene output and the same-size input.
    """
    out = crop_to_non_bg(erased_scene, int(bg))
    cropped_full_scene = crop_to_non_bg(scene, int(bg))

    if out.shape == inp.shape:
        return False
    if np.array_equal(cropped_full_scene, out):
        return False

    crop_delta_h = int(inp.shape[0] - out.shape[0])
    crop_delta_w = int(inp.shape[1] - out.shape[1])
    if max(crop_delta_h, crop_delta_w) < _MIN_CROP_DELTA:
        return False

    non_bg = int(np.count_nonzero(out != int(bg)))
    if non_bg / float(out.size) < _MIN_OUTPUT_DENSITY:
        return False

    return True


def _build_input_from_scene(
    scene: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep full pluses, but erase the repeated suffix of every non-demo ray."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = scene.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0, sc0 = p["seed_r0"], p["seed_c0"]
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (r, c) not in seed_set:
                inp[r, c] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    N = int(rng.integers(*_N_PLUS_RANGE))
    axis_choices = []
    for axis in ("horizontal", "vertical"):
        valid_shapes = [shape for shape in _PLUS_SHAPES if _shape_supports_axis(shape, axis)]
        if valid_shapes:
            axis_choices.append((axis, valid_shapes))
    if not axis_choices:
        return None

    axis, valid_shapes = axis_choices[int(rng.integers(0, len(axis_choices)))]
    plus_shape = valid_shapes[int(rng.integers(0, len(valid_shapes)))]
    direction_plan = _build_direction_plan(axis, N, rng)

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for direction in direction_plan:
        info = _place_plus_and_ray(
            rng, plus_shape, palette, H, W, occ, plus_occ, scene, plus_color, direction
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_scene(scene, pluses, bg, rng)
    erased_scene = _erase_pluses(scene, pluses, bg)
    out = crop_to_non_bg(erased_scene, int(bg))
    if not _combo_observable(inp, scene, erased_scene, int(bg)):
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
