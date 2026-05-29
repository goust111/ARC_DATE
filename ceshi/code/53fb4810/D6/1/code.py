# Variant of 53fb4810
# Dimension: D6 — 终点条件 (dst)
# Change: 射线一直延伸到网格边界为止... → 距离编码停止

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D6 variant
# a) Concrete behavior:
#    - Keep the original visual language: one shared '+' color, one marker tile touching each plus,
#      and exactly one already-finished example ray in the input.
#    - The changed part is only where a ray stops. Instead of running to the border, a ray stops after
#      K tile-units, where K = number of distinct visible colors inside that marker tile.
#    - Example trace: if a 2x2 seed next to a plus contains colors {2,4,7}, then the output contains
#      exactly 3 copies of that 2x2 tile in a row/column, seed included. In the original task this same
#      seed would have kept repeating until the edge; here it stops early and leaves background beyond.
# b) Complexity calibration:
#    - The encoding uses only 2-4 discrete values, which is enough to make lengths differ while staying
#      easy to compare across 2-4 training examples.
#    - To avoid collapsing the rule to a constant, generation only uses plus shapes whose nozzle span is 2;
#      that allows 2x2 seeds and therefore 3- and 4-color markers, not just a fixed 2-color case.
# c) Old assumptions that break:
#    - Original `main()` could treat the pre-extended clue as one huge "tile" because extending to the
#      boundary again was harmless. That is no longer safe: a stopped clue ray must not be repeated past
#      its encoded endpoint. The new solver therefore skips adjacent components whose along-ray span is
#      already larger than the maximum compact seed span (2), treating them as already-complete clues.
#    - Original extension iterated cell-by-cell until boundary. The new helper must extend by an exact
#      number of whole tile copies.
# d) Placement legality:
#    - Generator still works output-first, but now it places only the exact encoded number of copies.
#    - It checks that the whole encoded ray fits in-bounds, stays disjoint from all other pluses/rays,
#      and that at least one placed ray is both long enough and still leaves visible background before
#      the edge, so the input contains an unmistakable stopped clue instead of a border-touching one.
# e) Visual observability:
#    - The encoded quantity is "how many different colors are present", which is directly visible and
#      does not rely on numeric color ids.
#    - Each grid is required to contain at least two different encoded lengths among its pluses, so the
#      output visibly differs across rays instead of looking like one fixed stop distance.
# f) Visual cleanliness:
#    - Rays remain axis-aligned, non-overlapping, and plus pixels are protected, so the grid stays easy
#      to parse; large empty background remains around stopped rays, making the new endpoint rule obvious.
# g) Color inferability:
#    - Colors are not remapped. Their only role is counting distinct hues within the seed tile, which a
#      human can read off from the visible marker itself and confirm on the single pre-extended example ray.


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


def encoded_ray_units(tile: np.ndarray, *, bg: int) -> int:
    """Distance code: total tile-units = number of distinct non-bg colors in the seed tile."""
    cols = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    return max(1, int(len(cols)))


def extend_repeating_tile_exact(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    total_units: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a seed by whole-tile copies until its encoded number of units is reached."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    for unit_idx in range(1, int(total_units)):
        base_r = int(r0 + unit_idx * dr * h)
        base_c = int(c0 + unit_idx * dc * w)
        for i in range(h):
            for j in range(w):
                put(base_r + i, base_c + j, int(tile[i, j]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' shapes, but stop after a seed-encoded distance.

    Input structure:
    - bg = most common color.
    - One dominant non-bg color forms several '+'-shaped anchors.
    - Each plus has one small marker tile touching one arm; exactly one marker is already shown as a
      finished repeated ray that stops before the border.

    Rule:
    - For each compact marker tile touching a '+', infer its outward direction from which arm it touches.
    - Count the marker's distinct non-bg colors; that count is the total number of tile-units the ray
      should occupy, including the seed tile itself.
    - Repeat the tile away from the '+' until that many units are present, never painting over '+' pixels.

    Key insight: The endpoint is not the grid boundary anymore; the marker's own color diversity encodes
    how many copies the ray should contain.
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

            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)
            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                comp_cols = list(range(int(c0), int(c1) + 1))
                assert comp_cols == nozzle_cols
                along_span = int(r1 - r0 + 1)
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                comp_rows = list(range(int(r0), int(r1) + 1))
                assert comp_rows == nozzle_rows
                along_span = int(c1 - c0 + 1)

            # Compact seeds are at most 2 cells long along the ray axis; longer components are the
            # already-finished demonstration ray preserved from the input and should be left unchanged.
            if along_span > 2:
                continue

            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            total_units = encoded_ray_units(tile, bg=int(bg))
            extend_repeating_tile_exact(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                total_units=int(total_units),
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400
_PRECLUE_MIN_ALONG_SPAN = 3

# Keep only nozzle-width-2 shapes so the distance code does not collapse to a constant.
_PLUS_SHAPES = [
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


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int] | None:
    """Build a seed tile and return (tile, encoded_units)."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k < 2:
        return None
    n_colors = int(rng.integers(2, max_k + 1))

    chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
    vals = chosen.copy()
    if n_cells > n_colors:
        vals.extend(list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True))))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(th, tw)
    return tile, int(n_colors)


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    units_total: int,
) -> tuple[list[tuple[int, int, int]], int, int, int, int] | None:
    """Compute the exact encoded ray (seed + copies)."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
        first_r0 = int(r0)
        last_r0 = int(r0 + dr * th * (units_total - 1))
        lo_r = min(first_r0, last_r0)
        hi_r = max(first_r0 + th - 1, last_r0 + th - 1)
        if lo_r < 0 or hi_r >= H or c0 < 0 or c0 + tw > W:
            return None
        along_span = int(th * units_total)
        boundary_gap = int(H - 1 - hi_r) if dr == 1 else int(lo_r)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)
        first_c0 = int(c0)
        last_c0 = int(c0 + dc * tw * (units_total - 1))
        lo_c = min(first_c0, last_c0)
        hi_c = max(first_c0 + tw - 1, last_c0 + tw - 1)
        if lo_c < 0 or hi_c >= W or r0 < 0 or r0 + th > H:
            return None
        along_span = int(tw * units_total)
        boundary_gap = int(W - 1 - hi_c) if dc == 1 else int(lo_c)

    cells: list[tuple[int, int, int]] = []
    for unit_idx in range(int(units_total)):
        base_r = int(r0 + unit_idx * dr * th)
        base_c = int(c0 + unit_idx * dc * tw)
        for i in range(th):
            for j in range(tw):
                cells.append((base_r + i, base_c + j, int(tile[i, j])))

    return cells, int(r0), int(c0), int(along_span), int(boundary_gap)


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
    """Try to place one plus sprite with its exact-length ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) != 0
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
            built = _build_seed_tile(plus_shape, d, palette, rng)
            if built is None:
                continue
            tile, units_total = built
            projected = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                d,
                tile,
                units_total,
            )
            if projected is None:
                continue
            cells, sr0, sc0, along_span, boundary_gap = projected
            if any(int(occ[r, c]) != 0 for r, c, _ in cells):
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
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units_total": int(units_total),
                "along_span": int(along_span),
                "boundary_gap": int(boundary_gap),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Erase all but one completed ray to produce the input clue structure."""
    candidates = [
        i
        for i, p in enumerate(pluses)
        if int(p["along_span"]) >= _PRECLUE_MIN_ALONG_SPAN
        and int(p["units_total"]) >= 3
        and int(p["boundary_gap"]) >= 1
    ]
    if not candidates:
        return None
    pre_idx = max(candidates, key=lambda i: (int(pluses[i]["units_total"]), int(pluses[i]["along_span"])))
    if len(candidates) > 1 and bool(rng.integers(0, 2)):
        pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
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

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(rng, plus_shape, palette, H, W, occ, plus_occ, out, plus_color)
        if info is None:
            return None
        pluses.append(info)

    if len({int(p["units_total"]) for p in pluses}) < 2:
        return None

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
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
