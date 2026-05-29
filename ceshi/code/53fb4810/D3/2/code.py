# Variant of 53fb4810
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: periodic_tiling — 标记瓦片沿射线方向周期性完整重复，形成均匀的平铺图案... → 镜像交替平铺

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D3 variant
# a) Concrete behavior:
#    Each plus still owns one adjacent marker tile. The tile still launches a ray away from the plus,
#    but the repeated units now alternate as original / mirrored / original / mirrored. The mirror axis
#    follows the ray axis: horizontal rays use left-right reflection, vertical rays use top-bottom
#    reflection. Example: a rightward 1x2 seed [2 7] becomes [2 7][7 2][2 7][7 2]...
#    The input keeps exactly one already-expanded clue ray, so a human can see the alternating mirror
#    rhythm and transfer it to the compact seeds near the other pluses.
# b) Complexity calibration:
#    To keep the mirror visually observable, every generated seed has along-ray size 2 and is forced to
#    differ from its own directional mirror. Cross-axis size stays tied to the plus nozzle span (1 or 2),
#    so the task is still the same family of ray projection puzzles rather than a new object system.
# c) Old-code assumption audit:
#    - The original extension helper assumed a single periodic modulo over rows/cols. That fails here
#      because tile parity matters at unit boundaries.
#    - The rest of main() mostly depends only on adjacency, bbox alignment, and protected plus cells;
#      those assumptions remain valid.
#    - The pre-expanded clue is still safe: its whole long ray is one component, and because it already
#      reaches the boundary, re-extending from its bbox does nothing.
# d) Placement legality:
#    Rays keep the same footprint class as the original task: contiguous axis-aligned strips whose
#    cross-axis span matches the nozzle. Generator placement therefore still only needs to avoid overlaps
#    between pluses and projected ray cells. The mirror alternation changes colors inside the footprint,
#    not the occupied coordinates.
# e) Visual observability:
#    The generator forbids symmetric seeds (`tile == mirror(tile)` along the used axis), so mirrored units
#    are always visually different. Because exactly one clue ray is fully present in the input, the
#    normal/mirror alternation is directly visible from examples instead of depending on hidden metadata.
# f) Visual cleanliness:
#    Rays never overlap each other or plus cells. The grid still contains a small number of plus anchors
#    and one strip per plus, so the scene remains legible rather than layered.
# g) Color reasoning:
#    No color-id arithmetic is used. Colors are only copied from the seed tile under geometric mirror
#    transforms, which is directly observable from shape layout alone.
# Main/generator contract after the change:
#    main() now assumes that the intended continuation rule is "alternate mirror by tile unit" and that
#    the seed tile is asymmetric under that mirror. generate_input() guarantees this by construction.


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


def _mirror_tile_for_direction(tile: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """Mirror the tile along the ray axis so horizontal/vertical rays show different stripe parity."""
    dr, dc = map(int, direction)
    if dc != 0:
        return np.fliplr(tile)
    if dr != 0:
        return np.flipud(tile)
    raise ValueError("_mirror_tile_for_direction: zero direction")


def _tile_variant(tile: np.ndarray, direction: tuple[int, int], unit_index: int) -> np.ndarray:
    """Unit 0 is the original seed; odd units are mirrored; even units are original again."""
    return _mirror_tile_for_direction(tile, direction) if unit_index % 2 == 1 else tile


def extend_alternating_mirror_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a ray by repeating tile units that alternate original/mirrored orientation."""
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
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    step_r = dr * h
    step_c = dc * w
    unit = 1
    while True:
        base_r = r0 + unit * step_r
        base_c = c0 + unit * step_c
        if dr == 1 and base_r >= H:
            break
        if dr == -1 and base_r + h <= 0:
            break
        if dc == 1 and base_c >= W:
            break
        if dc == -1 and base_c + w <= 0:
            break

        cur = _tile_variant(tile, (dr, dc), unit)
        for i in range(h):
            for j in range(w):
                put(base_r + i, base_c + j, int(cur[i, j]))
        unit += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' anchors using alternating mirrored repeats.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several identical '+' sprites.
    - Each plus touches one compact marker tile, and exactly one ray in the grid is already shown as a long clue.

    Rule:
    - Read the side on which a marker touches its plus to determine the outward ray direction.
    - Continue the marker to the boundary one tile-unit at a time, alternating original orientation and a mirror along the ray axis.
    - Plus pixels remain protected and are never overwritten by the propagated ray.

    Key insight:
    - The attached marker gives both the launch direction and the base tile; the pre-expanded clue shows that
      repeated units are not plain periodic copies but an original/mirrored stripe pattern.
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

            extend_alternating_mirror_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG = 2
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 3
_PRE_EXT_IDEAL_UNITS = 10
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400

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
    """Return the inclusive cross-axis span of the arm tip that launches the marker."""
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
    """Build a small asymmetric seed whose mirror along the ray axis is visibly different."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None

    dr, dc = direction
    th, tw = (_SEED_ALONG, cross) if dc == 0 else (cross, _SEED_ALONG)
    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k < 2:
        return None

    for _ in range(40):
        n_colors = int(rng.integers(2, max_k + 1))
        chosen = list(rng.choice(palette, size=n_colors, replace=False))
        vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if np.array_equal(tile, _mirror_tile_for_direction(tile, direction)):
            continue
        return tile
    return None


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
    """Count how many full tile units fit from the seed position to the boundary, including the seed."""
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
    """Project the full alternating-mirror ray and return its painted cells plus the seed origin."""
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
        unit = 0
        while (r < H) if dr == 1 else (r + th > 0):
            cur = _tile_variant(tile, direction, unit)
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, c0 + j, int(cur[i, j])))
            r += step
            unit += 1
    else:
        c = c0
        step = tw if dc == 1 else -tw
        unit = 0
        while (c < W) if dc == 1 else (c + tw > 0):
            cur = _tile_variant(tile, direction, unit)
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(cur[i, j])))
            c += step
            unit += 1

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
    """Place one plus and its full output ray without collisions."""
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
                out[r, c] = plus_color
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = v

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": sr0,
                "seed_c0": sc0,
                "units": units,
                "direction": d,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full clue ray; reduce the others back to their compact seeds."""
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
        info = _place_plus_and_ray(
            rng, plus_shape, palette, H, W, occ, plus_occ, out, plus_color
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
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
