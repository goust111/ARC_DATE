# Variant of 53fb4810
# Dimension: D10 — '+'精灵数量与一致性 (meta)
# Change: 网格中有2-3个'+'精灵，所有精灵形状完全相同，颜色也相同... → 不同形状的精灵

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D10 variant
# a) Concrete behavior:
#    The visible change is that the 2-3 same-colored '+' anchors no longer share one global
#    silhouette. One grid may contain a 3x3 thin cross and a 3x4 fat cross, or a thin cross,
#    a fat cross, and a 5x4 diamond-like cross. Each anchor still owns exactly one marker seed
#    or one already-extended ray. The solver must therefore inspect each connected component's
#    own edge-arm span before deciding whether the adjacent marker is 1 cell wide or 2 cells
#    wide. Example: a 3x3 cross with a right-side seed uses a 1-row nozzle, while a 3x4 fat
#    cross with a right-side seed uses a 2-row nozzle. The extension rule is unchanged, but
#    the alignment check is now explicitly local to each component.
# b) Complexity calibration:
#    We keep the number of shape templates small (three) and keep all '+' sprites the same
#    color. That preserves the original object class while making the mixed-shape condition
#    visible in a few examples. Each grid is forced to use at least two distinct templates so
#    the variant does not silently collapse back to the original task.
# c) Hidden assumptions audit:
#    The original main() already avoided using one global template during extension, but its
#    comments implied a uniform '+' class. The real code dependency to check is the alignment
#    test: it must read the nozzle width from each component's own bbox edge, not from any
#    shared shape choice. We keep component-local nozzle extraction and remove brittle asserts
#    in favor of guarded checks so mixed-shape inputs do not crash if a candidate component is
#    irrelevant noise.
# d) Placement legality:
#    Generator placement remains output-first. Each plus shape is placed as its own connected
#    component, with a one-cell moat against other pluses and no collisions between rays and
#    existing occupied cells. Seed tiles are still edge-adjacent and span exactly the chosen
#    arm tip. The 5x4 diamond template only supports vertical rays because its left/right arm
#    tips are 3 cells wide and the marker vocabulary intentionally remains within the original
#    1x2 / 2x1 / 2x2 family.
# e) Visual observability:
#    The human-visible clue is geometric, not color-id based: same-colored anchors with
#    clearly different outlines emit rays that still obey the same local arm-tip rule. Seeing
#    one thin-nozzle anchor and one fat-nozzle anchor in the same example is enough to suggest
#    that the solver should reason per object, not by one memorized template.
# f) Visual cleanliness:
#    Rays never overlap plus cells, and plus objects stay separated by the original moat rule.
#    Only one ray is pre-expanded in the input, so the grid remains sparse and readable.
# g) Color inference:
#    No new color mapping is introduced. Marker colors still come from the non-bg, non-plus
#    palette and are merely copied periodically along the ray.
# h) Variant metric used after implementation:
#    For sampled generated inputs, count distinct cropped '+' masks among same-colored large
#    components. The variant is active only if almost every sample has at least two distinct
#    masks; otherwise generation has regressed to the original single-template regime.
# i) New solver assumptions after the edit:
#    main() assumes the dominant big-component color is still the shared '+' color, each plus
#    has exactly one adjacent marker component, and the seed bbox matches the relevant arm-tip
#    span. generate_input() explicitly constructs inputs satisfying those assumptions.


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


def _marker_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Return the unique direction from a plus component to its adjacent marker component."""
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
    return next(iter(dirs))


def _edge_span_indices(plus_mask: np.ndarray, direction: tuple[int, int]) -> list[int]:
    """Return the occupied edge positions along the cross-axis for one plus component."""
    coords = np.argwhere(plus_mask)
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [
            int(cc)
            for cc in range(int(pc0), int(pc1) + 1)
            if bool(plus_mask[nozzle_row, cc])
        ]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [
        int(rr)
        for rr in range(int(pr0), int(pr1) + 1)
        if bool(plus_mask[rr, nozzle_col])
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker tile outward from its own '+' object until the grid edge, even when the '+' objects have different shapes.

    Input structure:
    - The background is the most common color.
    - One shared non-background color forms 2-3 separate '+'-like components, but those components may have different silhouettes.
    - Each component has exactly one adjacent marker seed, and one of the components may already show a long pre-expanded periodic ray.

    Rule:
    - For every '+' component, find the touching marker component and infer which side of that specific '+' it is attached to.
    - Read the arm-tip span from that component's own outer edge, require the seed bbox to match that local nozzle width, and then repeat the seed tile away from the '+'.
    - Do not overwrite '+' pixels while extending the ray.

    Key insight:
    - The repeated pattern still comes from the attached marker tile, but the alignment must be computed independently for each '+' component because different components can expose different arm widths.
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
            direction = _marker_direction(pts, plus_mask, H, W)
            if direction is None:
                continue

            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            dr, dc = direction
            nozzle_span = _edge_span_indices(plus_mask, direction)

            if dc == 0:
                seed_span = list(range(int(c0), int(c1) + 1))
            else:
                seed_span = list(range(int(r0), int(r1) + 1))
            if seed_span != nozzle_span:
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


# --------------- generator (attempt_1) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 10
_PLACEMENT_TRIES = 100
_GEN_RETRIES = 300

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
                "shape": plus_shape.copy(),
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
                inp[r, c] = bg
    return inp


def _choose_plus_shapes(rng: np.random.Generator, n_plus: int) -> list[np.ndarray]:
    """Choose 2-3 plus shapes while forcing at least two distinct silhouettes per grid."""
    template_ids = list(range(len(_PLUS_SHAPES)))
    if n_plus == 2:
        chosen_ids = list(rng.choice(template_ids, size=2, replace=False))
    else:
        n_unique = int(rng.integers(2, min(len(_PLUS_SHAPES), n_plus) + 1))
        base_ids = list(rng.choice(template_ids, size=n_unique, replace=False))
        extras = list(rng.choice(base_ids, size=n_plus - n_unique, replace=True))
        chosen_ids = base_ids + extras
        rng.shuffle(chosen_ids)
    return [_PLUS_SHAPES[int(idx)] for idx in chosen_ids]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    N = int(rng.integers(*_N_PLUS_RANGE))
    plus_shapes = _choose_plus_shapes(rng, N)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for plus_shape in plus_shapes:
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

    if len({tuple(p["shape"].flatten()) for p in pluses}) < 2:
        return None

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
