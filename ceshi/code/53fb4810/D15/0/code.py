# Variant of 53fb4810
# Dimension: D15 — 背景色与精灵色约束 (meta)
# Change: 背景为单一最常见颜色（0-9任意），'+'精灵为单一非背景颜色，所有精灵共享同一颜色... → 多色精灵

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D15 variant
# a) Concretization and behavior trace
# - Original task: all plus sprites share one color, so the solver can first detect one
#   dominant non-background color and treat everything else as marker material.
# - Variant: each plus sprite has its own color. The repeated marker tiles must avoid
#   every plus color, so color no longer groups the pluses together.
# - Concrete grid view: the board still contains 2-3 identical plus shapes and one seed
#   per plus. Exactly one ray is pre-expanded in the input. The visible difference is that
#   the pluses are now different colors, while the seed/ray colors are drawn from a disjoint
#   marker palette.
# - Small trace: in the original logic, a cell of color 4 could only mean "part of the
#   shared plus color" if 4 was the detected dominant component color. In the variant, a
#   plus of color 4 and another plus of color 7 must both be detected as anchors, and the
#   solver must process the marker next to color 4 independently from the marker next to 7.
#
# b) Complexity calibration
# - I keep the original object geometry, ray periodicity, and one-preexpanded-ray clue.
# - I do not introduce extra transformations or per-color behavior. The only new burden is
#   that multiple large monochrome components must be recognized as separate plus anchors.
# - 2-3 pluses with unique colors are enough to make the change visible without creating a
#   large branching factor.
#
# c) Implicit-assumption audit of the original main()
# - detect_dominant_big_component_color(...): assumes all pluses share one color. Invalid now.
# - plus_lab = label_mask(g == plus_col): assumes one color mask contains every plus. Invalid.
# - non_mask = (g != bg) & (g != plus_col): assumes any non-bg/non-plus_color cell is marker.
#   Invalid when other pluses use different colors.
# - protect={plus_col}: only protects one color from overwrite. Invalid when pluses are multi-color.
# - The direction inference, nozzle-span alignment, and repeat extension remain valid because
#   they depend on local geometry around one plus, not on global color sharing.
#
# d) Placement legality
# - Generator still uses output-first construction and occupancy checks so pluses and rays do
#   not overlap or leave the grid.
# - Each plus keeps the original margin requirements. Seeds remain edge-adjacent to exactly one
#   arm tip and their cross-axis span still matches that arm tip exactly.
# - Marker colors are sampled from colors excluding both background and all chosen plus colors.
#
# e) Visual observability
# - The new rule is visible because multiple identical plus shapes appear in different colors,
#   but each still emits the same style of repeated ray from its adjacent seed.
# - The solver cannot rely on color identity; a human can likewise see that each colored plus
#   acts as its own source.
#
# f) Visual cleanliness
# - I preserve the original non-overlap placement discipline. Rays do not collide with pluses
#   or each other, so no region contains stacked evidence from multiple objects.
# - Marker tiles remain small (cross-axis <= 2), while pluses remain the only components with
#   bbox >= 3 in both dimensions. This keeps anchor-vs-marker separation clean.
#
# g) Color inferability
# - No hidden ordering or color-id arithmetic is used. The only color rule is a visible palette
#   partition: plus colors are unique and marker colors avoid them.
# - Because every sample contains repeated support for "different-colored pluses behave the same
#   way", the D15 change is inferable from a small number of examples.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: every plus is a monochrome connected component whose bbox is at least 3x3.
#   Generator guarantee: plus templates are unchanged and placed as single-color components.
# - Solver assumption: every non-plus connected marker component touching a plus belongs to exactly
#   one side of that plus. Generator guarantee: exactly one seed is attached to each plus and plus
#   neighborhoods are spaced so no marker touches multiple pluses.
# - Solver assumption: rays never need to overwrite any plus cell. Generator guarantee: placement
#   rejects any ray-plus or ray-ray collision.


def detect_big_plus_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> list[dict]:
    """
    Local helper (only used by this solver): collect all non-bg monochrome components whose
    bbox is at least (min_h x min_w). In this task family, those are the plus sprites.
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
    Extend marker tiles outward from each colored '+' shape by repeating them to the grid edge.

    Input structure:
    - bg is the most common color.
    - Several large '+'-like sprites appear in different non-bg colors; each sprite is a separate anchor.
    - Small marker patches touch one arm of each plus, and marker colors avoid every plus color.

    Rule:
    - Detect every large monochrome component as its own plus sprite, then inspect the marker component(s) touching it.
    - For each touching marker tile, infer which side of that plus it is attached to and repeat that tile away from the plus until the boundary.
    - Never overwrite any plus cell, regardless of which plus color it uses.

    Key insight:
    - Shared plus color is no longer a reliable global cue, so the anchor objects must be found by geometry first.
    - Once each large plus component is isolated, the original side-detection and periodic ray extension can be applied independently to every plus.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    pluses = detect_big_plus_components(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if not pluses:
        return out

    all_plus_colors = {int(comp["color"]) for comp in pluses}
    plus_union_mask = np.zeros_like(g, dtype=bool)
    for comp in pluses:
        coords = comp["coords"]
        plus_union_mask[coords[:, 0], coords[:, 1]] = True

    mark_lab, _mn = label_mask((g != int(bg)) & (~plus_union_mask), connectivity=4)

    for plus in pluses:
        plus_coords = plus["coords"]
        if plus_coords.size == 0:
            continue

        plus_mask = np.zeros_like(g, dtype=bool)
        plus_mask[plus_coords[:, 0], plus_coords[:, 1]] = True

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
                protect=set(all_plus_colors),
            )

    return out


# --------------- generator (attempt_1) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)           # grid height [min, max)
_W_RANGE = (13, 23)           # grid width [min, max)
_N_PLUS_RANGE = (2, 4)        # number of plus sprites [min, max)
_SEED_ALONG_CHOICES = [1, 2]  # along-axis dimension options for seed tile
_SEED_MAX_COLORS = 4          # max distinct colors in a seed tile
_SEED_MAX_CROSS = 2           # max nozzle cross-axis span for valid seed
_MIN_RAY_UNITS = 2            # minimum tile repetitions along ray
_PRE_EXT_IDEAL_UNITS = 10     # preferred minimum units for the pre-extended ray
_PLACEMENT_TRIES = 100        # attempts per plus placement
_GEN_RETRIES = 300            # outer retry limit in generate_input

# Pre-computed plus shape templates (LR+UD symmetric, bbox >= 3x3)
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

    return cells, r0, c0


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
                H, W, (r0, c0, r0 + ph - 1, c0 + pw - 1), plus_shape, d, tile
            )
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

            return {
                "plus_color": int(plus_color),
                "tile": tile,
                "cells": cells,
                "seed_r0": sr0,
                "seed_c0": sc0,
                "units": units,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erase non-pre-extended rays from the output to produce the input."""
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

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))
    plus_colors = list(rng.choice(others, size=N, replace=False))
    marker_palette = [c for c in others if c not in plus_colors]
    if not marker_palette:
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
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
