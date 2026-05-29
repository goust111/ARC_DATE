# Combo variant of 53fb4810 (index 7)
# combo_spec: D3/2+D14/0
# Name: 镜像交替严格截断

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior:
#    The base task attaches one marker tile to each plus and launches a ray away from the plus. This combo
#    changes two linked details at once. First, repeated units alternate as original / mirrored / original /
#    mirrored, where the mirror axis follows the ray axis (horizontal rays use left-right flips, vertical
#    rays use top-bottom flips). Second, propagation stops before any partial unit would spill past the
#    boundary. Example: a rightward seed [[2,7]] becomes [2,7][7,2][2,7] and then stops, even if one extra
#    column remains. The visible consequence is a striped ray that can leave a one-cell gap near an edge.
# b) Complexity calibration:
#    The mirror alternation is only observable if the seed differs from its own directional mirror. The
#    generator therefore fixes the along-ray size at 2 and rejects mirror-symmetric seeds. Cross-axis size
#    still follows the plus nozzle span (1 or 2), so the task stays in the original object family while the
#    new rhythm is easy to read from one long clue ray.
# c) Old-code assumption audit:
#    - Original `extend_repeating_tile` used per-cell modulo and happily painted partial tiles at edges.
#      That violates the strict truncation rule and must be replaced by unit-wise stepping with full-fit
#      checks.
#    - Original `main()` cropped the whole connected component bbox and treated it as the seed. That was safe
#      only because the pre-expanded clue always touched the boundary, so extending the oversized bbox did
#      nothing. Here the clue may stop early, so we must recover the first unit adjacent to the plus instead.
#    - Adjacency detection, dominant plus-color detection, and nozzle-span alignment logic remain valid.
# d) Placement legality:
#    Output is still generated first. Every plus and every projected ray cell is checked against a shared
#    occupancy map, so no rays overlap each other or any plus. Strict truncation simplifies legality because
#    each unit is either fully inside the grid or skipped entirely; there are no clipped edge fragments.
# e) Visual observability:
#    Exactly one ray is kept long in the input, and that clue is chosen from rays with non-zero leftover
#    boundary slack whenever possible. Humans can therefore see both new features in examples: alternating
#    mirror parity and the fact that the strip may stop short of the boundary instead of being clipped.
# f) Visual cleanliness:
#    The scene remains sparse: 2-3 identical plus sprites, one strip per plus, no overlap, no color-based
#    metadata. Leaving at most a one-cell edge gap is visually tidy and keeps the rays readable.
# g) Color reasoning / new assumptions:
#    There is no hidden arithmetic on color ids; colors are copied only through geometric mirroring. The new
#    solver assumes the first unit next to a plus has along-ray size 2, and the generator guarantees that
#    construction. It also guarantees the seed cells adjacent to the nozzle all belong to the same marker
#    component so seed recovery from the first unit is unambiguous.


_SEED_ALONG = 2


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
    """Mirror along the ray axis so alternating units visibly flip as they move away from the plus."""
    dr, dc = map(int, direction)
    if dc != 0:
        return np.fliplr(tile)
    if dr != 0:
        return np.flipud(tile)
    raise ValueError("_mirror_tile_for_direction: zero direction")


def _tile_variant(tile: np.ndarray, direction: tuple[int, int], unit_index: int) -> np.ndarray:
    """Unit 0 is the original seed; odd indices are mirrored; even indices keep the original orientation."""
    return _mirror_tile_for_direction(tile, direction) if unit_index % 2 == 1 else tile


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span of the plus arm tip for the chosen launch direction."""
    dr, dc = map(int, direction)
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _seed_shape_from_direction(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
) -> tuple[int, int]:
    """Seed shape uses fixed along-ray size 2 and cross-axis span equal to the nozzle width."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    dr, dc = map(int, direction)
    return (_SEED_ALONG, cross) if dc == 0 else (cross, _SEED_ALONG)


def _seed_origin_from_plus(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
) -> tuple[int, int]:
    """Compute the top-left corner of the first tile unit adjacent to the plus nozzle."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = _seed_shape_from_direction(plus_mask, direction)
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        return ((pr1 + 1) if dr == 1 else (pr0 - th), pc0 + s)
    return (pr0 + s, (pc1 + 1) if dc == 1 else (pc0 - tw))


def _full_unit_fits(
    H: int,
    W: int,
    base_r: int,
    base_c: int,
    h: int,
    w: int,
) -> bool:
    """Strict truncation: the next unit is drawable only if its entire bbox fits inside the grid."""
    return 0 <= int(base_r) and 0 <= int(base_c) and int(base_r) + int(h) <= int(H) and int(base_c) + int(w) <= int(W)


def extend_alternating_mirror_tile_strict(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend by full tile units only; later units alternate between original and mirrored orientation."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    step_r = dr * h
    step_c = dc * w
    unit = 1
    while True:
        base_r = r0 + unit * step_r
        base_c = c0 + unit * step_c
        if not _full_unit_fits(H, W, base_r, base_c, h, w):
            break
        cur = _tile_variant(tile, (dr, dc), unit)
        for i in range(h):
            for j in range(w):
                v = int(cur[i, j])
                if v == bg:
                    continue
                rr = base_r + i
                cc = base_c + j
                if int(out[rr, cc]) in protect:
                    continue
                out[rr, cc] = v
        unit += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' anchors using alternating mirrored units with strict full-tile stopping.

    Input structure:
    - The most common color is the background.
    - One dominant non-background color forms several identical '+' sprites.
    - Each plus has one attached marker strip; exactly one of those strips is already shown as a long clue,
      and it may stop short of the boundary if another full unit would not fit.

    Rule:
    - Use the side on which the marker touches the plus to determine the launch direction.
    - Recover the first tile unit adjacent to the plus, then continue outward in full tile units only,
      alternating original and mirrored orientation for successive units.
    - Never overwrite the plus color.

    Key insight:
    - The long clue ray teaches two things at once: parity matters because neighboring tile units flip, and
      the strip is quantized into whole units because no clipped fragment appears at the edge.
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
        plus_bbox = bbox_coords(plus_coords)
        pr0, pc0, pr1, pc1 = plus_bbox
        plus_local = plus_mask[pr0:pr1 + 1, pc0:pc1 + 1]

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
            th, tw = _seed_shape_from_direction(plus_local, direction)
            seed_r0, seed_c0 = _seed_origin_from_plus(plus_bbox, plus_local, direction)
            if not _full_unit_fits(H, W, seed_r0, seed_c0, th, tw):
                continue
            if np.any(mark_lab[seed_r0:seed_r0 + th, seed_c0:seed_c0 + tw] != int(cid)):
                continue

            tile = crop_bbox(g, (seed_r0, seed_c0, seed_r0 + th - 1, seed_c0 + tw - 1))

            dr, dc = direction
            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                seed_cols = list(range(int(seed_c0), int(seed_c0 + tw)))
                assert seed_cols == nozzle_cols
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                seed_rows = list(range(int(seed_r0), int(seed_r0 + th)))
                assert seed_rows == nozzle_rows

            extend_alternating_mirror_tile_strict(
                out,
                tile,
                seed_r0,
                seed_c0,
                direction,
                bg=bg,
                protect={int(plus_col)},
            )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 3
_PRE_EXT_IDEAL_UNITS = 8
_PLACEMENT_TRIES = 140
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


def _available_along_length(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> int:
    """Number of cells from the start of the seed unit to the boundary along the ray direction."""
    th, tw = tile.shape
    dr, dc = map(int, direction)
    if dr == 1:
        return int(H - seed_r0)
    if dr == -1:
        return int(seed_r0 + th)
    if dc == 1:
        return int(W - seed_c0)
    return int(seed_c0 + tw)


def _count_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> int:
    """Count how many whole tile units fit, including the seed itself."""
    th, tw = tile.shape
    along = th if direction[0] != 0 else tw
    return _available_along_length(H, W, seed_r0, seed_c0, tile, direction) // along


def _boundary_slack(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
) -> int:
    """Unused cells between the last full unit and the boundary in the ray direction."""
    th, tw = tile.shape
    along = th if direction[0] != 0 else tw
    return _available_along_length(H, W, seed_r0, seed_c0, tile, direction) % along


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a 2-step seed whose mirror along the ray axis is visibly different."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    th, tw = _seed_shape_from_direction(plus_mask, direction)
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


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int, int, int]:
    """Return the strict full-unit ray cells, the seed origin, unit count, and leftover slack."""
    seed_r0, seed_c0 = _seed_origin_from_plus(plus_bbox, plus_mask, direction)
    th, tw = tile.shape
    if not _full_unit_fits(H, W, seed_r0, seed_c0, th, tw):
        return [], seed_r0, seed_c0, 0, 0

    units = _count_full_units(H, W, seed_r0, seed_c0, tile, direction)
    slack = _boundary_slack(H, W, seed_r0, seed_c0, tile, direction)
    dr, dc = map(int, direction)
    step_r = dr * th
    step_c = dc * tw

    cells: list[tuple[int, int, int]] = []
    for unit in range(units):
        base_r = seed_r0 + unit * step_r
        base_c = seed_c0 + unit * step_c
        cur = _tile_variant(tile, direction, unit)
        for i in range(th):
            for j in range(tw):
                cells.append((base_r + i, base_c + j, int(cur[i, j])))

    return cells, seed_r0, seed_c0, units, slack


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    mark_guard: np.ndarray,
    out: np.ndarray,
    plus_color: int,
) -> dict | None:
    """Place one plus and its full strict-truncation output ray without collisions."""
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
        if any(mark_guard[r, c] == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
            if tile is None:
                continue

            cells, seed_r0, seed_c0, units, slack = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
            )
            if units < _MIN_RAY_UNITS:
                continue
            if any(occ[r, c] != 0 for r, c, _ in cells):
                continue
            if any(mark_guard[r, c] == 1 for r, c, _ in cells):
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
                mark_guard[r, c] = 1
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        mark_guard[rr, cc] = 1

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": seed_r0,
                "seed_c0": seed_c0,
                "units": units,
                "slack": slack,
                "direction": direction,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    pre_idx: int,
) -> np.ndarray:
    """Keep one long clue ray and reduce every other ray to its first tile unit."""
    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == int(pre_idx):
            continue
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (r, c) not in seed_set:
                inp[r, c] = int(bg)
    return inp


def _choose_clue_index(pluses: list[dict], rng: np.random.Generator) -> int | None:
    """Prefer long rays with visible slack so the strict-truncation behavior is observable in the input."""
    slackers = [i for i, p in enumerate(pluses) if int(p["slack"]) > 0]
    if not slackers:
        return None

    ideal = [i for i in slackers if int(pluses[i]["units"]) >= _PRE_EXT_IDEAL_UNITS]
    pool = ideal if ideal else slackers
    best_units = max(int(pluses[i]["units"]) for i in pool)
    best = [i for i in pool if int(pluses[i]["units"]) == best_units]
    return int(rng.choice(best))


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
    mark_guard = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            mark_guard,
            out,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    pre_idx = _choose_clue_index(pluses, rng)
    if pre_idx is None:
        return None

    inp = _build_input_from_output(out, pluses, bg, pre_idx)

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
