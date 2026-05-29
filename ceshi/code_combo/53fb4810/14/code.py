# Combo variant of 53fb4810 (index 14)
# combo_spec: D10/0+D15/0
# Name: 异形多色精灵

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D10/0 + D15/0 combo
# a) Concrete behavior and trace:
#    The original task groups every '+' anchor by one shared color and one shared silhouette.
#    This combo removes both shortcuts at once: a single grid now contains 2-3 large anchors,
#    each anchor has its own color, and at least two anchors use different masks. Example:
#    a 3x3 thin cross in blue may emit a 1-row rightward seed, while a 3x4 fat cross in red
#    emits a 2-column upward seed. The solver can no longer say "find the dominant color, then
#    reuse one global nozzle width"; it must first isolate every large monochrome component, then
#    read that component's own edge span before extending its adjacent marker.
# b) Complexity calibration:
#    I keep the shape vocabulary small (the three original plus templates), keep one seed per
#    anchor, and keep the ray rule unchanged. That makes the fused change visible but bounded:
#    the new burden is only per-object decomposition by both color and geometry. The generator
#    forces at least two distinct plus shapes and unique plus colors in every sample so the combo
#    does not silently collapse to either single-variant or original behavior.
# c) Hidden-assumption audit of the original main():
#    - detect_dominant_big_component_color(...) assumed one shared plus color. Broken by D15.
#    - plus_lab = label_mask(g == plus_col) assumed one color mask contains every anchor. Broken.
#    - non_mask = (g != bg) & (g != plus_col) assumed every other non-bg cell is marker matter.
#      Broken when other pluses use different colors.
#    - protect={plus_col} assumed one protected color. Broken when pluses are multi-color.
#    - The alignment check was already conceptually local, but the original comments implied a
#      uniform '+' template. D10 requires the nozzle span to be recomputed per component.
#    The fused fix is: detect all large monochrome components, union them into a plus mask,
#    and process touching marker components one anchor at a time with local span checks.
# d) Placement legality:
#    Generation stays output-first. Each plus is placed as its own connected component with a
#    one-cell moat against other pluses, and every full projected ray is collision-checked before
#    commit. Seed tiles remain edge-adjacent to exactly one arm tip and must match that arm tip's
#    cross-axis span. The 5x4 diamond template only allows vertical rays because its lateral tips
#    are 3 cells wide while the marker vocabulary remains within the original max cross-span 2.
# e) Visual observability:
#    The fused rule is visible from a few examples because both signals are on-screen: anchors
#    differ by color and at least two differ by outline. A human can see that each colored blob
#    owns its own nearby seed, and that thin/fat anchors require different seed widths even though
#    the repetition rule is otherwise the same.
# f) Visual cleanliness:
#    Rays never overlap plus cells or each other, only one ray is pre-expanded in the input, and
#    marker tiles stay within the original 1x2 / 2x1 / 2x2 family. Large monochrome components are
#    therefore visually reserved for anchors, while marker/ray material stays narrow and readable.
# g) Color inferability:
#    There is no hidden color arithmetic. The only color constraint is a visible partition:
#    background is most common, every plus has a unique non-bg color, and marker colors avoid the
#    entire plus-color set. This gives repeated evidence that color distinguishes anchors, not rules.
# h) Variant metric used after implementation:
#    Sampled generated inputs should satisfy both metrics simultaneously:
#    - unique_plus_colors == number_of_pluses
#    - distinct_plus_shapes >= 2
#    If either metric collapses, the combo has regressed toward a simpler regime.
# i) New solver assumptions after the edit:
#    main() now assumes every plus is a monochrome connected component with bbox >= 3x3, every
#    marker component touching a plus belongs to exactly one side of exactly one plus, and valid
#    seeds match the touched arm-tip span. generate_input() constructs only inputs meeting those
#    assumptions and rejects any placement whose oracle check fails.


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
    Extend each marker tile outward from its own colored '+' object until the grid edge, even when the '+' objects use different shapes.

    Input structure:
    - The background is the most common color.
    - There are 2-3 separate '+'-like anchors; each anchor is a large monochrome component, and different anchors may use different colors and different silhouettes.
    - Each anchor has one adjacent marker seed, and at most one anchor already shows a long pre-expanded periodic ray.

    Rule:
    - Detect every large monochrome component as its own plus anchor, then find the marker component touching that anchor.
    - Infer the attachment side locally for that anchor, require the seed span to match that anchor's own arm-tip span, and repeat the seed tile away from the anchor to the boundary.
    - Never overwrite any plus cell, regardless of which plus color it belongs to.

    Key insight:
    - Neither one global plus color nor one global plus shape is reliable anymore.
    - The task is solved by decomposing the scene into separate large colored anchors, then applying the original repeat-along-ray rule independently using each anchor's local geometry.
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
                protect=set(all_plus_colors),
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
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "shape": plus_shape.copy(),
                "plus_color": int(plus_color),
                "direction": (int(d[0]), int(d[1])),
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


def _choose_plus_shapes(rng: np.random.Generator, n_plus: int) -> list[np.ndarray]:
    """Choose 2-3 plus shapes while forcing a thin cross plus at least one thicker template."""
    chosen_ids = [0, int(rng.choice([1, 2]))]
    if n_plus == 3:
        chosen_ids.append(int(rng.choice([0, 1, 2])))
    rng.shuffle(chosen_ids)
    return [_PLUS_SHAPES[int(idx)] for idx in chosen_ids]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]

    N = int(rng.integers(*_N_PLUS_RANGE))
    plus_shapes = _choose_plus_shapes(rng, N)
    plus_colors = list(rng.choice(others, size=N, replace=False))
    marker_palette = [c for c in others if c not in plus_colors]
    if not marker_palette:
        return None

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for plus_shape, plus_color in zip(plus_shapes, plus_colors):
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

    if len({int(p["plus_color"]) for p in pluses}) != len(pluses):
        return None
    if len({tuple(p["shape"].flatten()) for p in pluses}) < 2:
        return None
    nozzle_widths = []
    for p in pluses:
        s, e = _nozzle_span(p["shape"], p["direction"])
        nozzle_widths.append(int(e - s + 1))
    if len(set(nozzle_widths)) < 2:
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
