# Variant of 53fb4810
# Dimension: D7 — 变换规则——源保护 (transform)
# Change: keep_source — '+'精灵的像素永远不被射线覆盖（受保护），射线只覆盖背景像素... → 源精灵消失
#
# Design analysis for the D7 variant
# a) Concrete behavior and trace:
#    - Input still contains several identical '+' sprites in one shared color, each with one
#      adjacent seed tile. Exactly one ray is already pre-extended as the clue, same as the
#      base task.
#    - Output still extends every seed periodically away from its '+' until the boundary.
#    - The new visible difference is that every '+' footprint disappears in the output:
#      those source cells are reset to the background after the ray construction is inferred.
#    - Small trace example: a 3x3 plus with a 1x2 seed attached below originally produced an
#      output that kept the 5 plus pixels while repeating the 1x2 tile downward. In this
#      variant the same downward repetition happens, but the 5 source pixels become background,
#      leaving a hole where the plus used to be.
# b) Complexity calibration:
#    - The geometry, colors, counts, and periodic extension rules stay unchanged so the task
#      remains the same concept class; only the source-retention bit flips.
#    - The plus color remains unique and all pluses share one shape, which keeps object
#      detection stable without introducing extra degrees of freedom.
#    - The disappearance is repeated across 2-3 pluses per grid, so the new rule is obvious
#      from a few examples rather than a one-off artifact.
# c) Review of old assumptions in main():
#    - `detect_dominant_big_component_color(...)` still works because the input still contains
#      the plus sprites; only the output erases them.
#    - `non_mask = (g != bg) & (g != plus_col)` still requires the generator to avoid using
#      the plus color inside marker tiles. The generator preserves that invariant.
#    - The alignment asserts still depend on each seed bbox matching the arm-tip span exactly.
#      The generator keeps the same nozzle/seed alignment logic.
#    - The old solver's effective retention came from starting with `out = g.copy()` and then
#      never clearing plus cells. That must now be inverted explicitly by erasing plus cells
#      after all ray logic has been derived from the unchanged input.
# d) Placement legality:
#    - Generation still places pluses and full rays in a collision-free full scene first.
#    - Rays never overlap existing pluses or other rays because `occ` is checked before commit.
#    - Input is derived by erasing only the repeated suffix of every non-demo ray; output is
#      derived by erasing every plus footprint from the full scene.
# e) Visual observability:
#    - Humans can see two coupled changes in each pair: compact seeds grow into full rays, and
#      every '+' vanishes. The latter is a direct, visible structural change, not a hidden
#      color-id convention.
# f) Visual cleanliness:
#    - Because rays are still collision-free and pluses disappear only on their own footprints,
#      the output stays sparse and readable instead of becoming a layered overlap puzzle.
# g) Color reasoning:
#    - No new color mapping is introduced. The variant changes only whether the source shape is
#      kept or erased, so color inference burden does not increase.
# h) New solver assumptions checked after the change:
#    - All plus sprites in the input still share one dominant color and one shape family.
#    - Each plus has exactly one adjacent marker component, with a unique direction.
#    - Marker palettes still exclude the plus color, so clearing all plus-colored cells in the
#      output does not accidentally erase ray content.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every marker tile outward from its '+' source as a periodic ray, then remove the source.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical '+'-like source sprites.
    - Each source has one small non-plus marker tile attached to one arm; one source already
      shows its long repeated ray as a demonstration.

    Rule:
    - For each marker touching a '+', infer which side of the source it is attached to.
    - Repeat that seed tile away from the source in the same cardinal direction until the edge.
    - After all rays are determined, erase every original '+' pixel back to the background.

    Key insight: The '+' still serves as the directional anchor in the input, but it is not
    part of the final pattern; only the repeated marker trajectory survives in the output.
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

            # Direction from plus -> marker (must be unique).
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

            # Alignment check: seed cross-axis span must match arm tip span.
            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)
            if dc == 0:  # vertical ray
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                seed_cols = list(range(int(c0), int(c1) + 1))
                assert seed_cols == nozzle_cols
            else:  # horizontal ray
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
    return out


# --------------- generator (attempt_1 -> D7 variant) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)            # grid height [min, max)
_W_RANGE = (13, 23)            # grid width [min, max)
_N_PLUS_RANGE = (2, 4)         # number of plus sprites [min, max)
_SEED_ALONG_CHOICES = [1, 2]   # along-axis dimension options for seed tile
_SEED_MAX_COLORS = 4           # max distinct colors in a seed tile
_SEED_MAX_CROSS = 2            # max nozzle cross-axis span for valid seed
_MIN_RAY_UNITS = 2             # minimum tile repetitions along ray
_PRE_EXT_IDEAL_UNITS = 10      # preferred minimum units for the pre-extended ray
_PLACEMENT_TRIES = 100         # attempts per plus placement
_GEN_RETRIES = 300             # outer retry limit in generate_input

# Pre-computed plus shape templates (LR+UD symmetric, bbox >= 3x3)
_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),          # 3x3 classic cross
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int), # 3x4 fat plus
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1],
              [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),                # 5x4 fat diamond
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return (start, end) inclusive of nozzle cells along the cross-axis."""
    dr, dc = direction
    if dc == 0:  # vertical ray -- cross-axis is columns
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    else:  # horizontal ray -- cross-axis is rows
        col = plus_mask.shape[1] - 1 if dc == 1 else 0
        rows = np.where(plus_mask[:, col])[0]
        return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a seed tile whose cross-axis matches the nozzle span.

    Returns None if cross-span exceeds _SEED_MAX_CROSS.
    """
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None
    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2  # avoid degenerate 1x1

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
    """Compute (r, c, value) list for the full projected ray (seed + repetitions).

    Returns (cells, seed_r0, seed_c0).
    """
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    # Anchor the first tile adjacent to the nozzle edge.
    if dc == 0:  # vertical
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:  # horizontal
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)

    cells: list[tuple[int, int, int]] = []
    if dc == 0:  # vertical (up or down)
        r = r0
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, c0 + j, int(tile[i, j])))
            r += step
    else:  # horizontal (left or right)
        c = c0
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(tile[i, j])))
            c += step

    return cells, r0, c0


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
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
            if tile is None:
                continue
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
            }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return a copy of scene with every source-plus footprint erased to the background."""
    out = scene.copy()
    for p in pluses:
        for r, c in p["plus_cells"]:
            out[r, c] = int(bg)
    return out


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

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng, plus_shape, palette, H, W, occ, plus_occ, scene, plus_color
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_scene(scene, pluses, bg, rng)
    out = _erase_pluses(scene, pluses, bg)

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
