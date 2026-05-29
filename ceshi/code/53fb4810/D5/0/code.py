# Variant of 53fb4810
# Dimension: D5 — 标记瓦片结构 (src)
# Change: 标记瓦片为小型矩形（1x2, 2x1, 2x2），包含1-4种非背景非'+'颜色，每个'+'精灵恰好附着一个标记... → 大尺寸种子瓦片

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The unchanged core rule is still "find the seed attached to a '+' and repeat it
#    away from the '+' until the boundary". The visible change is that the seed is no
#    longer a tiny 1x2/2x1/2x2 clue. Each seed is now a filled multicolor rectangle such
#    as 3x2, 2x3, 3x3, 4x2, or 4x3. Example: if a 3x3 seed touches the top arm of a '+',
#    the output repeats that exact 3x3 texture upward tile-by-tile. In the original
#    task the repeated period could visually collapse into a thin stripe; in this variant
#    the repeated period is a visibly larger block with internal structure.
# b) Complexity calibration:
#    To make the change observable, seeds are forced to be at least 6 cells with at least
#    2 colors and non-trivial row/column variation. This avoids degenerate cases that
#    would look like the original narrow stripe or a solid rectangle. Freedom is still
#    controlled: cross-axis span is limited to the nozzle width (2 or 3), while the
#    along-axis span is chosen from a small set (2, 3, 4).
# c) Implicit assumptions in the original solver:
#    - `mark_lab = label_mask(non_mask, connectivity=4)` assumes each seed is one
#      4-connected object. The generator therefore uses fully filled rectangles with no
#      holes or detached islands.
#    - `tile = crop_bbox(...)` assumes the component's bbox is the repeating period.
#      That remains valid only if the marker is itself rectangular, so the generator
#      keeps seeds as filled rectangles.
#    - The alignment check assumes the seed cross-axis span equals the arm-tip nozzle
#      span exactly. Large seeds keep this invariant; only the along-axis grows.
#    - `extend_repeating_tile(...)` already repeats arbitrary rectangle sizes correctly,
#      so no change is needed in the propagation mechanism.
# d) Placement legality:
#    Large seeds need more room, so the generator slightly enlarges grid ranges, uses
#    plus shapes whose nozzle span can be 2 or 3, and rejects placements whose projected
#    rays would collide with other pluses or rays.
# e) Visual observability:
#    The repeated structure is human-visible because the same large multicolor rectangle
#    appears several times along a ray. Exactly one ray remains pre-expanded in the input,
#    giving a direct example of the periodic continuation rule.
# f) Visual cleanliness:
#    Rays are still placed without overlaps. This keeps each seed and each repeated band
#    visually separable despite the richer textures.
# g) Color inference:
#    No hidden color ordering is introduced. Colors are only used as visible tile content;
#    solver decisions rely on geometry and adjacency, not on specific color ids.


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
        raise ValueError("extend_repeating_tile: direction must be axis-aligned")


def _seed_matches_nozzle(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """Check that the seed cross-axis span matches the nozzle edge exactly."""
    r0, c0, r1, c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    pr0 = int(pr0)
    pc0 = int(pc0)
    pr1 = int(pr1)
    pc1 = int(pc1)

    if dc == 0:
        nozzle_row = pr1 if dr == 1 else pr0
        nozzle_cols = sorted(int(cc) for cc in range(pc0, pc1 + 1) if plus_mask[nozzle_row, cc])
        seed_cols = list(range(c0, c1 + 1))
        return seed_cols == nozzle_cols

    nozzle_col = pc1 if dc == 1 else pc0
    nozzle_rows = sorted(int(rr) for rr in range(pr0, pr1 + 1) if plus_mask[rr, nozzle_col])
    seed_rows = list(range(r0, r1 + 1))
    return seed_rows == nozzle_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each attached marker rectangle away from its '+' anchor until the edge.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several plus-like anchor sprites.
    - Each plus has one adjacent rectangular seed marker, and in this variant that seed
      can be a larger multicolor block such as 3x2, 2x3, 3x3, or 4x3.

    Rule:
    - Find every non-plus component touching a plus and infer which side of the plus it
      is attached to.
    - Treat the entire seed bbox as the repeating period and propagate that rectangle
      away from the plus in the same direction until the boundary.
    - Plus pixels stay fixed and are never overwritten by the propagated pattern.

    Key insight:
    - The attachment side still determines the ray direction, but the period is now a
      larger textured rectangle rather than a tiny seed strip, so the whole multi-row /
      multi-column motif must be repeated intact.
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
    mark_lab, _ = label_mask(non_mask, connectivity=4)

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

            direction = next(iter(dirs))
            if not _seed_matches_nozzle(plus_mask, plus_coords, (r0, c0, r1, c1), direction):
                continue

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
                protect={int(plus_col)},
            )

    return out


# --------------- generator ---------------

# Large seeds need a bit more room than the original tiny marker tiles.
_H_RANGE = (18, 31)
_W_RANGE = (18, 28)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [2, 3, 4]
_SEED_MAX_COLORS = 4
_SEED_MIN_COLORS = 2
_SEED_ALLOWED_CROSS = {2, 3}
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 4
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 500

# Symmetric plus-like anchor shapes. Every shape has at least one direction whose nozzle
# span is 2 or 3, which is required for the larger rectangular seeds.
_PLUS_SHAPES = [
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
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span of the chosen nozzle."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])

    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray, direction: tuple[int, int]) -> bool:
    """Reject seeds that visually collapse into the old simple cases."""
    if len(set(map(int, tile.flatten()))) < _SEED_MIN_COLORS:
        return False

    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False

    dr, dc = direction
    if dc == 0:
        if len({tuple(map(int, tile[i, :])) for i in range(tile.shape[0])}) < 2:
            return False
    else:
        if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
            return False

    return True


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a large rectangular seed whose cross-axis matches the nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross not in _SEED_ALLOWED_CROSS:
        return None

    for _ in range(20):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        if cross == 2 and along == 2:
            along = 3

        dr, dc = direction
        th, tw = (along, cross) if dc == 0 else (cross, along)
        n_cells = th * tw
        if n_cells < 6:
            continue

        max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
        if max_k < _SEED_MIN_COLORS:
            return None
        n_colors = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile, direction):
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
    """Compute how many full tile units fit along the ray direction, including the seed."""
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
    """Return all cells of the projected ray plus the seed anchor location."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = pr1 + 1 if dr == 1 else pr0 - th
    else:
        r0 = pr0 + s
        c0 = pc1 + 1 if dc == 1 else pc0 - tw

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


def _plus_touches_existing_ray(
    plus_cells: list[tuple[int, int]],
    occ: np.ndarray,
) -> bool:
    """Reject plus placements that would sit directly next to an existing ray."""
    H, W = occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) == 2:
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
) -> dict | None:
    """Place one plus together with its full output ray."""
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
        if _plus_touches_existing_ray(plus_cells, occ):
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
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
            )
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue
            if any(plus_occ[r, c] == 1 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
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
                "direction": direction,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one long example ray and collapse the rest back to just their seed tiles."""
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
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
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

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng=rng,
            plus_shape=plus_shape,
            palette=palette,
            H=H,
            W=W,
            occ=occ,
            plus_occ=plus_occ,
            out=out,
            plus_color=plus_color,
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
