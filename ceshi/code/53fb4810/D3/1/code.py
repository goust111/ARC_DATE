# Variant of 53fb4810
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: periodic_tiling — 标记瓦片沿射线方向周期性完整重复，形成均匀的平铺图案... → 渐变颜色循环

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior trace
#    Original behavior repeats the exact same seed tile forever:
#      seed [[2, 3],        output chunks: [[2, 3]] [[2, 3]] [[2, 3]] ...
#            [4, 2]]
#    The variant keeps the same geometry, but each whole-tile step applies one
#    global color permutation learned from the single long clue ray:
#      color cycle A -> B -> C -> A
#      seed [[A, B],        output chunks: [[A, B]] [[B, C]] [[C, A]] [[A, B]] ...
#            [C, A]]
#    The change therefore lives exactly at the "repeat next tile" stage. Plus
#    detection, nozzle alignment, adjacency, and projection direction stay the
#    same.
#
# b) Complexity calibration
#    A color cycle is only inferable if the clue ray actually shows enough
#    structure. A 1x2 / 2x1 clue with only two visible colors is too weak for a
#    reliable 3-cycle, so the generator keeps the original plus/ray layout but
#    forces the single pre-extended clue ray to use a 2x2 seed containing all
#    three cycle colors. Other seeds retain the original small sizes.
#
# c) Hidden assumptions checked against the original solver
#    - The original `tile = crop_bbox(component_bbox)` only works because every
#      non-clue component is just the seed tile. With a pre-extended clue ray,
#      that bbox spans the whole ray and is no longer the seed. The new solver
#      reconstructs the seed bbox from the plus nozzle and direction instead.
#    - The original `extend_repeating_tile` copies seed colors cellwise by
#      modulo indexing. That bakes in "same colors every period". The variant
#      must instead advance in tile-sized chunks and recolor the whole chunk at
#      each step.
#    - The original code never had to infer any global relation between colors.
#      The variant introduces a new global assumption: exactly one elongated
#      clue component exists in the input, and its adjacent full chunks reveal a
#      consistent color mapping. The generator guarantees this.
#
# d) Placement legality
#    Output is still built first. Rays are generated as whole connected strips,
#    collision-checked before painting, and plus cells remain protected. Input
#    is obtained by erasing every non-clue ray after its compact seed tile, so
#    only one elongated component remains available for cycle inference.
#
# e) Visual observability
#    The cycle is encoded only through visible repeated chunks, never through
#    color-id ordering. The clue ray is long, 2x2, and contains all cycle
#    colors, so humans can compare consecutive chunks and notice a consistent
#    "same shape, shifted colors" pattern.
#
# f) Visual cleanliness
#    Rays still do not collide with plus sprites or each other. Only one ray is
#    long in the input, which avoids clutter and keeps the clue localized.
#
# g) Color-change inferability
#    The generator uses exactly three cycle colors shared by all seeds in one
#    grid. Every seed color therefore appears in the clue ray's cycle, so the
#    solver never has to extrapolate an unseen mapping.
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
    """Apply a whole-tile color permutation once."""
    if not color_map:
        return np.array(tile, dtype=int, copy=True)
    src = np.array(tile, dtype=int, copy=False)
    out = src.copy()
    for a, b in color_map.items():
        out[src == int(a)] = int(b)
    return out


def extend_color_cycle_tile(
    out: np.ndarray,
    seed_tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
    color_map: dict[int, int] | None = None,
) -> None:
    """Extend a seed tile to the edge, recoloring each whole-tile repetition."""
    protect = set() if protect is None else set(map(int, protect))
    color_map = {} if color_map is None else {int(k): int(v) for k, v in color_map.items()}
    H, W = out.shape
    h, w = seed_tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if int(v) == int(bg):
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    rr = int(r0 + dr * h) if dr != 0 else int(r0)
    cc = int(c0 + dc * w) if dc != 0 else int(c0)
    cur = _apply_color_map(seed_tile, color_map)

    while rr < H and rr + h > 0 and cc < W and cc + w > 0:
        for i in range(h):
            for j in range(w):
                put(int(rr + i), int(cc + j), int(cur[i, j]))
        rr += int(dr * h)
        cc += int(dc * w)
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


def _nozzle_cells(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> list[int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [int(c) for c in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, c])]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [int(r) for r in range(int(pr0), int(pr1) + 1) if bool(plus_mask[r, nozzle_col])]


def _along_extent(pts: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox_coords(pts)
    _dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)


def _seed_bbox_from_plus(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
    along: int,
) -> tuple[int, int, int, int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    along = int(along)

    if dc == 0:
        nozzle_cols = _nozzle_cells(plus_mask, plus_coords, direction)
        c0 = int(min(nozzle_cols))
        c1 = int(max(nozzle_cols))
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - along)
        r1 = int(r0 + along - 1)
        return int(r0), int(c0), int(r1), int(c1)

    nozzle_rows = _nozzle_cells(plus_mask, plus_coords, direction)
    r0 = int(min(nozzle_rows))
    r1 = int(max(nozzle_rows))
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - along)
    c1 = int(c0 + along - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_full_chunks(
    grid: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[np.ndarray]:
    H, W = grid.shape
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    rr = int(r0)
    cc = int(c0)
    chunks: list[np.ndarray] = []

    while 0 <= rr and rr + h <= H and 0 <= cc and cc + w <= W:
        chunks.append(np.array(grid[rr : rr + h, cc : cc + w], dtype=int, copy=True))
        rr += int(dr * h)
        cc += int(dc * w)

    return chunks


def _infer_color_map_from_chunks(
    chunks: list[np.ndarray],
    *,
    bg: int,
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    evidence = 0
    for a, b in zip(chunks, chunks[1:]):
        for va, vb in zip(a.flatten(), b.flatten()):
            va = int(va)
            vb = int(vb)
            if va == int(bg) or vb == int(bg):
                continue
            evidence += 1
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None
            mapping[int(va)] = int(vb)
    if evidence == 0 or len(mapping) < 2:
        return None
    return mapping


def _infer_global_color_map(
    grid: np.ndarray,
    infos: list[dict],
    *,
    bg: int,
) -> dict[int, int]:
    for info in infos:
        if int(info["along_extent"]) <= 2:
            continue
        seed_r0, seed_c0, seed_r1, seed_c1 = info["seed_bbox"]
        shape = (int(seed_r1 - seed_r0 + 1), int(seed_c1 - seed_c0 + 1))
        chunks = _extract_full_chunks(
            grid,
            int(seed_r0),
            int(seed_c0),
            shape,
            tuple(map(int, info["direction"])),
        )
        mapping = _infer_color_map_from_chunks(chunks, bg=int(bg))
        if mapping:
            return mapping
    return {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker rays away from '+' anchors, but advance colors through a learned cycle.

    Input structure:
    - The most common color is background.
    - One dominant non-background color forms several identical '+'-like anchors.
    - Each anchor has one adjacent marker tile; exactly one marker is already expanded into a long ray.

    Rule:
    - Use the long ray as the clue: consecutive full tile-chunks keep the same shape but their colors shift by a fixed global permutation.
    - For every anchor, detect the side on which its seed tile attaches, then project that seed away from the '+'.
    - Each new whole-tile repetition applies one more step of the learned color cycle; '+' pixels are never overwritten.

    Key insight:
    The geometry is still ordinary ray projection, but the repeated unit is no longer color-stationary; the single long example ray reveals a global recoloring rule that must be reused on every other seed.
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

            along = _along_extent(pts, direction)
            if along > 2:
                along = 2

            seed_bbox = _seed_bbox_from_plus(plus_mask, plus_coords, direction, along)
            sr0, sc0, sr1, sc1 = seed_bbox
            if not (0 <= sr0 <= sr1 < H and 0 <= sc0 <= sc1 < W):
                continue

            tile = crop_bbox(g, (int(sr0), int(sc0), int(sr1), int(sc1)))
            if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
                continue

            nozzle_axis = _nozzle_cells(plus_mask, plus_coords, direction)
            if direction[1] == 0:
                seed_cols = list(range(int(sc0), int(sc1) + 1))
                assert seed_cols == nozzle_axis
            else:
                seed_rows = list(range(int(sr0), int(sr1) + 1))
                assert seed_rows == nozzle_axis

            infos.append(
                {
                    "direction": tuple(map(int, direction)),
                    "seed_bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                    "tile": np.array(tile, dtype=int, copy=True),
                    "along_extent": int(_along_extent(pts, direction)),
                }
            )

    color_map = _infer_global_color_map(g, infos, bg=int(bg))

    for info in infos:
        sr0, sc0, _sr1, _sc1 = info["seed_bbox"]
        extend_color_cycle_tile(
            out,
            info["tile"],
            int(sr0),
            int(sc0),
            tuple(map(int, info["direction"])),
            bg=int(bg),
            protect={int(plus_col)},
            color_map=color_map,
        )

    return out


# --------------- generator ---------------

# Geometry ranges stay close to the original task.
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_MIN_RAY_UNITS = 2
_CLUE_MIN_FULL_UNITS = 4
_CLUE_IDEAL_UNITS = 4
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400
_CYCLE_LEN = 3

# Only keep plus shapes whose nozzle span is 2. This is the compatibility
# adjustment that allows a 2x2 clue tile to visibly expose a 3-color cycle.
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
) -> np.ndarray | None:
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    dr, dc = map(int, direction)

    if force_clue:
        if cross != 2:
            return None
        vals = list(map(int, cycle_colors)) + [int(rng.choice(cycle_colors))]
        rng.shuffle(vals)
        return np.array(vals, dtype=int).reshape(2, 2)

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)
    max_k = min(len(cycle_colors), n_cells)
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
    vals = list(chosen) + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _compute_full_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    r0: int,
    c0: int,
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> int:
    dr, dc = map(int, direction)
    th, tw = tile.shape
    if dr == 1:
        return int((H - (r0 + ph)) // th)
    if dr == -1:
        return int(r0 // th)
    if dc == 1:
        return int((W - (c0 + pw)) // tw)
    return int(c0 // tw)


def _project_cycle_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
) -> tuple[list[tuple[int, int, int]], int, int]:
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

    while rr < H and rr + th > 0 and cc < W and cc + tw > 0:
        for i in range(th):
            for j in range(tw):
                rrr = int(rr + i)
                ccc = int(cc + j)
                if 0 <= rrr < H and 0 <= ccc < W:
                    cells.append((int(rrr), int(ccc), int(cur[i, j])))
        rr += int(dr * th)
        cc += int(dc * tw)
        cur = _apply_color_map(cur, color_map)

    return cells, int(r0), int(c0)


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    *,
    force_clue: bool = False,
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

        if forced_direction is None:
            dirs = list(_DIRECTIONS)
            rng.shuffle(dirs)
        else:
            dirs = [tuple(map(int, forced_direction))]
        for direction in dirs:
            tile = _build_seed_tile(
                plus_shape,
                direction,
                cycle_colors,
                rng,
                force_clue=bool(force_clue),
            )
            if tile is None:
                continue

            units = _compute_full_units(direction, tile, r0, c0, ph, pw, H, W)
            need = _CLUE_MIN_FULL_UNITS if force_clue else _MIN_RAY_UNITS
            if units < need:
                continue

            cells, sr0, sc0 = _project_cycle_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
                color_map,
            )
            if any(int(occ[r, c]) != 0 for r, c, _v in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "tile": np.array(tile, dtype=int, copy=True),
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(units),
                "is_clue": bool(force_clue),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    keep_idx: int,
) -> np.ndarray:
    inp = np.array(out, dtype=int, copy=True)
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
    color_map = {int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN]) for i in range(_CYCLE_LEN)}

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

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
        out,
        plus_color,
        force_clue=True,
        forced_direction=_DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))],
    )
    if clue is None or int(clue["units"]) < _CLUE_IDEAL_UNITS:
        return None
    pluses.append(clue)

    for _ in range(N - 1):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            cycle_colors,
            color_map,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
            force_clue=False,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, keep_idx=0)

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
