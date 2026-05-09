# Combo variant of 53fb4810 (index 21)
# combo_spec: D3/1+D6/1+D7/0
# Name: 颜色循环距离编码源消失

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D3/1 + D6/1 + D7/0
#
# a) Concrete behavior trace
#    Base task: a seed touching one arm of a '+' repeats unchanged until the
#    boundary, and the '+' remains visible.
#    Combo task: keep the same anchor geometry, but change three linked stages:
#    1. each whole-tile repetition applies one global color permutation;
#    2. the total number of tile-units equals the number of distinct colors in
#       the seed tile, so the ray stops early instead of running to the edge;
#    3. once all rays are completed, every '+' footprint is erased to bg.
#    Small trace:
#      clue seed [[A, B],       units = 3 distinct colors
#                 [C, A]]
#      output chunks become
#        [[A, B], [B, C], [C, A]]
#        [[C, A], [A, B], [B, C]]
#      and then stop, leaving background after the third chunk; finally the
#      '+' cells disappear. A 2-color seed elsewhere only gets 2 total chunks.
#
# b) Complexity calibration
#    The cycle is kept global and short: exactly three colors shared by all
#    markers in a grid. This avoids an underconstrained color puzzle.
#    Plus shapes are restricted to nozzle-width 2, so a 2x2 clue can expose all
#    three cycle colors at once. Non-clue seeds use 2 or 3 colors, yielding
#    only two distance values; that is enough to make the endpoint rule visible
#    without creating many cases.
#
# c) Hidden assumptions broken from the base solver
#    - A long adjacent component is no longer "the tile": in the clue it is the
#      entire finished ray. The solver therefore reconstructs the compact seed
#      from the nozzle geometry and tries along-axis sizes 1 or 2.
#    - "Extend until boundary" is invalid. The new extension helper paints an
#      exact number of whole chunks derived from the seed's color diversity.
#    - Because the source disappears, the old protection-by-copy behavior from
#      `out = g.copy()` is wrong. The solver must erase all plus-colored source
#      cells only after it has extracted the directional information from input.
#    - The color map must be inferred from the single finished clue ray, so the
#      generator guarantees exactly one adjacent component with along-ray span
#      greater than 2 and makes it internally consistent under one permutation.
#
# d) Placement legality
#    Generation remains output-first. Exact-length rays are projected in-bounds,
#    collision-checked against all existing pluses and rays, and the clue is
#    required to leave a positive gap before the boundary so the new stop rule
#    is visible. Input is derived by erasing every non-clue ray after its seed.
#
# e) Visual observability
#    A human gets two independent clues from the same long example ray: the
#    chunk shape repeats while colors rotate, and the finished ray stops before
#    the border. Across examples, one seed uses 3 colors and another uses 2, so
#    the endpoint difference is readable directly from the compact markers.
#
# f) Visual cleanliness
#    Rays stay axis-aligned and non-overlapping. Only one long component remains
#    in the input, so the color-cycle evidence is localized rather than buried
#    in clutter. Removing the '+' only happens in the output and only on its own
#    cells, so it does not create ambiguous overlaps.
#
# g) Change inferability
#    The generator never relies on numeric color ids. Distance comes from the
#    visible count of distinct marker colors, and the cycle is supported by
#    repeated chunk correspondences. At least one 2-unit ray and one 3-unit ray
#    are present per sample, preventing collapse to a constant endpoint.
#
# h) New solver assumptions audited against the generator
#    - All pluses share one dominant color and one width-2 symmetric shape.
#    - Each plus has exactly one adjacent component.
#    - Compact seeds have along-axis size 1 or 2; the only component with
#      along-axis span > 2 is the finished clue ray.
#    - Seed palettes exclude bg and plus color, and use only the three cycle
#      colors, so inferred recoloring applies to every seed in the grid.
# ---------------------------------------------------------------------------


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-bg color with the most large connected components."""
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


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    src = np.array(tile, dtype=int, copy=False)
    out = src.copy()
    for a, b in color_map.items():
        out[src == int(a)] = int(b)
    return out


def encoded_ray_units(tile: np.ndarray, *, bg: int) -> int:
    cols = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    return max(1, int(len(cols)))


def extend_color_cycle_exact(
    out: np.ndarray,
    seed_tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    total_units: int,
    bg: int,
    color_map: dict[int, int],
) -> None:
    """Paint exactly total_units chunks, advancing the color map per chunk."""
    H, W = out.shape
    h, w = seed_tile.shape
    dr, dc = map(int, direction)
    cur = _apply_color_map(seed_tile, color_map)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if int(v) == int(bg):
            return
        out[rr, cc] = int(v)

    for unit_idx in range(1, int(total_units)):
        base_r = int(r0 + unit_idx * dr * h)
        base_c = int(c0 + unit_idx * dc * w)
        for i in range(h):
            for j in range(w):
                put(int(base_r + i), int(base_c + j), int(cur[i, j]))
        cur = _apply_color_map(cur, color_map)


def _component_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
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


def _nozzle_axis(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> list[int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    if dc == 0:
        row = int(pr1) if dr == 1 else int(pr0)
        return [int(c) for c in range(int(pc0), int(pc1) + 1) if bool(plus_mask[row, c])]
    col = int(pc1) if dc == 1 else int(pc0)
    return [int(r) for r in range(int(pr0), int(pr1) + 1) if bool(plus_mask[r, col])]


def _seed_bbox_from_plus(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
    along: int,
) -> tuple[int, int, int, int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    nozzle = _nozzle_axis(plus_mask, plus_coords, direction)
    along = int(along)

    if dc == 0:
        c0 = int(min(nozzle))
        c1 = int(max(nozzle))
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - along)
        r1 = int(r0 + along - 1)
        return int(r0), int(c0), int(r1), int(c1)

    r0 = int(min(nozzle))
    r1 = int(max(nozzle))
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - along)
    c1 = int(c0 + along - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_in_bounds(b: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = map(int, b)
    return 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W


def _along_extent(pts: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox_coords(pts)
    _dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)


def _extract_chunks(
    grid: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
    count: int,
) -> tuple[list[np.ndarray], set[tuple[int, int]]] | None:
    H, W = grid.shape
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    rr = int(r0)
    cc = int(c0)
    chunks: list[np.ndarray] = []
    coords: set[tuple[int, int]] = set()

    for _ in range(int(count)):
        if not (0 <= rr and rr + h <= H and 0 <= cc and cc + w <= W):
            return None
        chunks.append(np.array(grid[rr : rr + h, cc : cc + w], dtype=int, copy=True))
        for i in range(h):
            for j in range(w):
                coords.add((int(rr + i), int(cc + j)))
        rr += int(dr * h)
        cc += int(dc * w)

    return chunks, coords


def _infer_color_map_from_chunks(
    chunks: list[np.ndarray],
    *,
    bg: int,
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for a, b in zip(chunks, chunks[1:]):
        for va, vb in zip(a.flatten(), b.flatten()):
            va = int(va)
            vb = int(vb)
            if va == int(bg) or vb == int(bg):
                return None
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None
            mapping[int(va)] = int(vb)

    if len(mapping) < 2:
        return None
    if len(set(mapping.values())) != len(mapping):
        return None
    for a, b in zip(chunks, chunks[1:]):
        if not np.array_equal(_apply_color_map(a, mapping), b):
            return None
    return mapping


def _parse_finished_clue(
    grid: np.ndarray,
    pts: np.ndarray,
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
    *,
    bg: int,
    plus_col: int,
) -> dict | None:
    comp_cells = {(int(r), int(c)) for r, c in pts}
    along_extent = _along_extent(pts, direction)
    if along_extent <= 2:
        return None

    for along in (1, 2):
        seed_bbox = _seed_bbox_from_plus(plus_mask, plus_coords, direction, along)
        if not _bbox_in_bounds(seed_bbox, *grid.shape):
            continue
        sr0, sc0, sr1, sc1 = seed_bbox
        tile = crop_bbox(grid, (int(sr0), int(sc0), int(sr1), int(sc1)))
        if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
            continue

        total_units = encoded_ray_units(tile, bg=int(bg))
        if int(along) * int(total_units) != int(along_extent):
            continue

        extracted = _extract_chunks(
            grid,
            int(sr0),
            int(sc0),
            tile.shape,
            direction,
            int(total_units),
        )
        if extracted is None:
            continue
        chunks, coords = extracted
        if coords != comp_cells:
            continue
        color_map = _infer_color_map_from_chunks(chunks, bg=int(bg))
        if color_map is None:
            continue

        return {
            "seed_bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
            "tile": np.array(tile, dtype=int, copy=True),
            "total_units": int(total_units),
            "color_map": color_map,
        }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow marker rays away from '+' anchors using a learned color cycle, stop after a seed-encoded distance, then erase the anchors.

    Input structure:
    - The most common color is the background.
    - One dominant non-background color forms several identical '+'-like anchors.
    - Each anchor has one adjacent marker component; exactly one of those components is already a finished multi-chunk clue ray, while the others are only compact seeds.

    Rule:
    - Use the finished clue ray to infer a global whole-tile color permutation from one chunk to the next.
    - For every anchor, reconstruct its compact seed tile from the nozzle alignment, count that seed's distinct non-background colors, and treat that count as the total number of chunks the ray should contain.
    - Project the seed away from the '+' for exactly that many chunks, applying one more step of the color cycle per chunk, then erase every original '+' cell back to background.

    Key insight:
    The long example ray simultaneously teaches both altered dimensions: its repeated geometry reveals the recoloring cycle, and its early stop shows that ray length is encoded by the seed itself rather than by the grid boundary; the '+' is only a temporary directional scaffold and vanishes in the output.
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

    infos: list[dict] = []
    clue_map: dict[int, int] | None = None

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
            direction = _component_direction(pts, plus_mask, H, W)
            if direction is None:
                continue

            along_extent = _along_extent(pts, direction)
            info = {
                "plus_mask": plus_mask,
                "plus_coords": plus_coords,
                "direction": tuple(map(int, direction)),
                "pts": pts,
                "along_extent": int(along_extent),
                "parsed_clue": None,
            }
            if int(along_extent) > 2:
                parsed = _parse_finished_clue(
                    g,
                    pts,
                    plus_mask,
                    plus_coords,
                    direction,
                    bg=int(bg),
                    plus_col=int(plus_col),
                )
                if parsed is not None and clue_map is None:
                    clue_map = {int(k): int(v) for k, v in parsed["color_map"].items()}
                info["parsed_clue"] = parsed
            infos.append(info)

    if clue_map is None:
        clue_map = {}

    for info in infos:
        direction = tuple(map(int, info["direction"]))
        plus_mask = info["plus_mask"]
        plus_coords = info["plus_coords"]
        parsed = info["parsed_clue"]

        if parsed is not None:
            sr0, sc0, sr1, sc1 = map(int, parsed["seed_bbox"])
            tile = np.array(parsed["tile"], dtype=int, copy=True)
            total_units = int(parsed["total_units"])
        else:
            along = int(info["along_extent"])
            if along not in (1, 2):
                continue
            sr0, sc0, sr1, sc1 = _seed_bbox_from_plus(plus_mask, plus_coords, direction, along)
            if not _bbox_in_bounds((sr0, sc0, sr1, sc1), H, W):
                continue
            tile = crop_bbox(g, (int(sr0), int(sc0), int(sr1), int(sc1)))
            if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
                continue
            total_units = encoded_ray_units(tile, bg=int(bg))

        nozzle = _nozzle_axis(plus_mask, plus_coords, direction)
        if direction[1] == 0:
            assert list(range(int(sc0), int(sc1) + 1)) == nozzle
        else:
            assert list(range(int(sr0), int(sr1) + 1)) == nozzle

        extend_color_cycle_exact(
            out,
            tile,
            int(sr0),
            int(sc0),
            direction,
            total_units=int(total_units),
            bg=int(bg),
            color_map=clue_map,
        )

    out[g == int(plus_col)] = int(bg)
    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 5)
_SEED_ALONG_CHOICES = [1, 2]
_PLACEMENT_TRIES = 140
_GEN_RETRIES = 500
_CYCLE_LEN = 3
_CLUE_UNITS = 3
_CLUE_BOUNDARY_GAP_MIN = 1

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
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
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
    cycle_colors: list[int],
    rng: np.random.Generator,
    *,
    force_clue: bool = False,
    forced_units: int | None = None,
) -> tuple[np.ndarray, int] | None:
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross != 2:
        return None

    dr, dc = map(int, direction)
    if force_clue:
        along = 2
        th, tw = (along, cross) if dc == 0 else (cross, along)
        vals = list(map(int, cycle_colors)) + [int(rng.choice(cycle_colors))]
        rng.shuffle(vals)
        return np.array(vals, dtype=int).reshape(th, tw), int(_CLUE_UNITS)

    valid_alongs = []
    for along in _SEED_ALONG_CHOICES:
        n_cells = int(along * cross)
        if forced_units is not None and int(forced_units) <= n_cells:
            valid_alongs.append(int(along))
        if forced_units is None and n_cells >= 2:
            valid_alongs.append(int(along))
    if not valid_alongs:
        return None

    along = int(rng.choice(valid_alongs))
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)

    if forced_units is None:
        choices = [u for u in (2, 3) if u <= n_cells]
        if not choices:
            return None
        units = int(rng.choice(choices))
    else:
        units = int(forced_units)
        if units > n_cells:
            return None

    chosen = list(map(int, rng.choice(cycle_colors, size=units, replace=False)))
    vals = chosen.copy()
    if n_cells > units:
        vals.extend(list(map(int, rng.choice(chosen, size=n_cells - units, replace=True))))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw), int(units)


def _project_cycle_coords_exact(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
    total_units: int,
) -> tuple[list[tuple[int, int, int]], int, int, int] | None:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - th)
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tw)

    cells: list[tuple[int, int, int]] = []
    rr = int(r0)
    cc = int(c0)
    cur = np.array(tile, dtype=int, copy=True)
    min_r = H
    max_r = -1
    min_c = W
    max_c = -1

    for _ in range(int(total_units)):
        if not (0 <= rr and rr + th <= H and 0 <= cc and cc + tw <= W):
            return None
        for i in range(th):
            for j in range(tw):
                rrr = int(rr + i)
                ccc = int(cc + j)
                cells.append((int(rrr), int(ccc), int(cur[i, j])))
                min_r = min(min_r, int(rrr))
                max_r = max(max_r, int(rrr))
                min_c = min(min_c, int(ccc))
                max_c = max(max_c, int(ccc))
        rr += int(dr * th)
        cc += int(dc * tw)
        cur = _apply_color_map(cur, color_map)

    boundary_gap = (
        int(H - 1 - max_r)
        if dr == 1
        else int(min_r)
        if dr == -1
        else int(W - 1 - max_c)
        if dc == 1
        else int(min_c)
    )

    return cells, int(r0), int(c0), int(boundary_gap)


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    mark_guard: np.ndarray,
    scene: np.ndarray,
    plus_color: int,
    *,
    force_clue: bool = False,
    forced_units: int | None = None,
    forced_direction: tuple[int, int] | None = None,
) -> dict | None:
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue
        if any(int(mark_guard[r, c]) == 1 for r, c in plus_cells):
            continue

        dirs = [tuple(map(int, forced_direction))] if forced_direction is not None else list(_DIRECTIONS)
        if forced_direction is None:
            rng.shuffle(dirs)

        for direction in dirs:
            built = _build_seed_tile(
                plus_shape,
                direction,
                cycle_colors,
                rng,
                force_clue=bool(force_clue),
                forced_units=forced_units,
            )
            if built is None:
                continue
            tile, total_units = built
            projected = _project_cycle_coords_exact(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
                color_map,
                total_units,
            )
            if projected is None:
                continue
            cells, sr0, sc0, boundary_gap = projected
            if force_clue and int(boundary_gap) < _CLUE_BOUNDARY_GAP_MIN:
                continue
            if any(int(occ[r, c]) != 0 for r, c, _v in cells):
                continue
            if any(int(plus_occ[r, c]) == 1 or int(mark_guard[r, c]) == 1 for r, c, _v in cells):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                scene[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                scene[r, c] = int(v)
                mark_guard[r, c] = 1
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        mark_guard[rr, cc] = 1

            return {
                "tile": np.array(tile, dtype=int, copy=True),
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "total_units": int(total_units),
                "plus_cells": plus_cells,
                "boundary_gap": int(boundary_gap),
                "direction": tuple(map(int, direction)),
                "is_clue": bool(force_clue),
            }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    out = np.array(scene, dtype=int, copy=True)
    for p in pluses:
        for r, c in p["plus_cells"]:
            out[int(r), int(c)] = int(bg)
    return out


def _build_input_from_scene(scene: np.ndarray, pluses: list[dict], bg: int, keep_idx: int) -> np.ndarray:
    inp = np.array(scene, dtype=int, copy=True)
    for i, p in enumerate(pluses):
        if int(i) == int(keep_idx):
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _v in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [int(c) for c in range(10) if int(c) != int(bg)]
    plus_color = int(rng.choice(others))
    cycle_colors = list(map(int, rng.choice([c for c in others if c != plus_color], size=_CYCLE_LEN, replace=False)))
    rng.shuffle(cycle_colors)
    color_map = {
        int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN])
        for i in range(_CYCLE_LEN)
    }

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)
    mark_guard = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []

    clue = _place_plus_and_ray(
        rng,
        plus_shape,
        cycle_colors,
        color_map,
        H,
        W,
        occ,
        plus_occ,
        mark_guard,
        scene,
        plus_color,
        force_clue=True,
        forced_units=int(_CLUE_UNITS),
        forced_direction=_DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))],
    )
    if clue is None:
        return None
    pluses.append(clue)

    # Force at least one 2-unit non-clue ray so distance encoding never collapses.
    info = _place_plus_and_ray(
        rng,
        plus_shape,
        cycle_colors,
        color_map,
        H,
        W,
        occ,
        plus_occ,
        mark_guard,
        scene,
        plus_color,
        force_clue=False,
        forced_units=2,
    )
    if info is None:
        return None
    pluses.append(info)

    for _ in range(max(0, N - 2)):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            cycle_colors,
            color_map,
            H,
            W,
            occ,
            plus_occ,
            mark_guard,
            scene,
            plus_color,
            force_clue=False,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_scene(scene, pluses, bg, keep_idx=0)
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
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
