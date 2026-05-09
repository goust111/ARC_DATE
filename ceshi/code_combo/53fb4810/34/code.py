# Combo variant of 53fb4810 (index 34)
# combo_spec: D3/1+D14/1
# Name: 颜色循环末端渐隐

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    Original task:
#      seed [[A, B], [C, A]] -> [[A, B], [C, A]] [[A, B], [C, A]] [[A, B], [C, A]]
#    D3/1 alone:
#      use one global cycle A->B->C->A, so whole chunks become
#      [[A, B], [C, A]] [[B, C], [A, B]] [[C, A], [B, C]] ...
#    Fused combo:
#      keep the chunkwise color cycle, but the final repeated chunks near the
#      boundary are also eaten by bg in a directional order. A right-going ray
#      can therefore look like
#      normal0, normal1, normal2, fade1(3 visible cells), fade2(2 cells),
#      fade3(1 cell).
#    The key difference from plain D3/1 is that the end chunks are not a new
#    color permutation; they are the same cyclic chunk colors with increasing bg
#    replacement layered on top.
#
# b) Complexity calibration
#    The cycle must be inferable before the fade begins. To guarantee that, the
#    generator uses 2x2 seeds, a shared 3-color cycle, and one long clue ray
#    with at least three non-faded chunks before the last three faded chunks.
#    This keeps the visual rule rich enough to be distinct from the base task
#    while avoiding underconstrained color inference.
#
# c) Hidden assumptions checked against the original solver
#    - `crop_bbox(component_bbox)` is no longer valid for the long clue because
#      its bbox spans the whole expanded ray; the seed bbox must be reconstructed
#      from the plus nozzle and the fixed 2x2 seed size.
#    - Modulo-copying colors cellwise is no longer correct; repetition must be
#      chunk-based so that one color-cycle step is applied per whole tile.
#    - Consecutive clue chunks are no longer always fully colored because the
#      tail contains bg holes. Color-map inference therefore ignores positions
#      where either side is bg and relies on the earlier non-faded chunks for
#      full evidence.
#    - A new fused assumption appears: every seed color belongs to the same
#      global 3-cycle, and every ray uses the same terminal fade schedule. The
#      generator enforces both.
#
# d) Placement legality
#    Output is still built first. The generator reserves each full chunk bbox as
#    occupied, even if a terminal fade removes some cells to bg, so rays stay
#    visually separate and pluses are never overwritten.
#
# e) Visual observability
#    A human can compare the clue chunks in order: early chunks show a stable
#    shape with shifting colors, while the last chunks keep following that color
#    rhythm but progressively lose cells to bg. This makes it possible to
#    separate "cycle" from "fade" using only visible structure.
#
# f) Visual cleanliness
#    Restricting the family to nozzle-width-2 pluses and 2x2 seeds keeps all
#    rays aligned, the fade masks readable, and the single long clue localized.
#    Reserving whole chunk corridors prevents overlapping partial tails.
#
# g) Change inferability
#    Only three cycle colors are used per grid, repeated across all markers, so
#    the color mapping has repeated support. The fade uses a fixed directional
#    removal order and exactly the last three repeated chunks, which is simple
#    enough to learn from 2-4 examples and different enough from plain D3/1 to
#    require genuine fusion reasoning.
# ---------------------------------------------------------------------------


_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_SEED_ALONG = 2
_END_FADE_CHUNKS = 3


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


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int], steps: int = 1) -> np.ndarray:
    """Apply the learned whole-tile color permutation `steps` times."""
    out = np.array(tile, dtype=int, copy=True)
    color_map = {int(k): int(v) for k, v in color_map.items()}
    for _ in range(int(max(0, steps))):
        if not color_map:
            break
        src = out.copy()
        for a, b in color_map.items():
            out[src == int(a)] = int(b)
    return out


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
        for dr, dc in _DIRECTIONS:
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


def _seed_bbox_from_plus(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> tuple[int, int, int, int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_cols = _nozzle_cells(plus_mask, plus_coords, direction)
        if len(nozzle_cols) != 2:
            raise ValueError("seed reconstruction expects nozzle width 2")
        c0 = int(min(nozzle_cols))
        c1 = int(max(nozzle_cols))
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - _SEED_ALONG)
        r1 = int(r0 + _SEED_ALONG - 1)
        return int(r0), int(c0), int(r1), int(c1)

    nozzle_rows = _nozzle_cells(plus_mask, plus_coords, direction)
    if len(nozzle_rows) != 2:
        raise ValueError("seed reconstruction expects nozzle width 2")
    r0 = int(min(nozzle_rows))
    r1 = int(max(nozzle_rows))
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - _SEED_ALONG)
    c1 = int(c0 + _SEED_ALONG - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _along_extent(pts: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox_coords(pts)
    _dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)


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


def _infer_color_map_from_chunks(chunks: list[np.ndarray], *, bg: int) -> dict[int, int] | None:
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
    if evidence == 0 or len(mapping) < 3:
        return None
    return mapping


def _fade_order(shape: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    coords = [(i, j) for i in range(h) for j in range(w)]

    def key(pos: tuple[int, int]) -> tuple[int, int]:
        i, j = pos
        if (dr, dc) == (0, 1):
            return (-j, i)
        if (dr, dc) == (0, -1):
            return (j, i)
        if (dr, dc) == (1, 0):
            return (-i, j)
        if (dr, dc) == (-1, 0):
            return (i, j)
        raise ValueError("unsupported direction")

    return sorted(coords, key=key)


def _apply_tail_fade(
    tile: np.ndarray,
    direction: tuple[int, int],
    fade_rank: int,
    fade_count: int,
    *,
    bg: int,
) -> np.ndarray:
    if fade_rank <= 0 or fade_count <= 0:
        return np.array(tile, dtype=int, copy=True)
    out = np.array(tile, dtype=int, copy=True)
    order = _fade_order(out.shape, direction)
    n_cells = int(out.size)
    n_remove = int(np.ceil(int(fade_rank) * n_cells / (int(fade_count) + 1)))
    n_remove = max(0, min(n_cells - 1, n_remove))
    for i, j in order[:n_remove]:
        out[int(i), int(j)] = int(bg)
    return out


def _chunk_origins(
    H: int,
    W: int,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    rr = int(r0)
    cc = int(c0)
    origins: list[tuple[int, int]] = []

    while 0 <= rr and rr + h <= H and 0 <= cc and cc + w <= W:
        origins.append((int(rr), int(cc)))
        rr += int(dr * h)
        cc += int(dc * w)

    return origins


def _paint_variant_ray(
    out: np.ndarray,
    seed_tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
    color_map: dict[int, int] | None = None,
) -> list[tuple[int, int]]:
    protect = set() if protect is None else set(map(int, protect))
    color_map = {} if color_map is None else {int(k): int(v) for k, v in color_map.items()}
    H, W = out.shape
    h, w = seed_tile.shape
    origins = _chunk_origins(H, W, int(r0), int(c0), (h, w), direction)
    repeats = max(0, len(origins) - 1)
    fade_count = min(_END_FADE_CHUNKS, repeats)
    fade_start = len(origins) - fade_count

    def put(rr: int, cc: int, v: int) -> None:
        if int(v) == int(bg):
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    cover: list[tuple[int, int]] = []
    for idx, (rr, cc) in enumerate(origins):
        cur = _apply_color_map(seed_tile, color_map, steps=int(idx))
        if idx > 0 and idx >= fade_start:
            cur = _apply_tail_fade(cur, direction, idx - fade_start + 1, fade_count, bg=int(bg))
        for i in range(h):
            for j in range(w):
                cover.append((int(rr + i), int(cc + j)))
                put(int(rr + i), int(cc + j), int(cur[i, j]))
    return cover


def _collect_infos(grid: np.ndarray, *, bg: int, plus_col: int) -> list[dict]:
    H, W = grid.shape
    plus_lab, pn = label_mask(grid == int(plus_col), connectivity=4)
    non_mask = (grid != int(bg)) & (grid != int(plus_col))
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
            for dr, dc in _DIRECTIONS:
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

            try:
                seed_bbox = _seed_bbox_from_plus(plus_mask, plus_coords, direction)
            except ValueError:
                continue
            sr0, sc0, sr1, sc1 = seed_bbox
            if not (0 <= sr0 <= sr1 < H and 0 <= sc0 <= sc1 < W):
                continue

            nozzle_axis = _nozzle_cells(plus_mask, plus_coords, direction)
            if direction[1] == 0:
                if list(range(int(sc0), int(sc1) + 1)) != nozzle_axis:
                    continue
            else:
                if list(range(int(sr0), int(sr1) + 1)) != nozzle_axis:
                    continue

            tile = crop_bbox(grid, (int(sr0), int(sc0), int(sr1), int(sc1)))
            if tile.shape != (_SEED_ALONG, 2):
                continue
            if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
                continue

            infos.append(
                {
                    "direction": tuple(map(int, direction)),
                    "seed_bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                    "tile": np.array(tile, dtype=int, copy=True),
                    "along_extent": int(_along_extent(pts, direction)),
                }
            )

    return infos


def _infer_global_color_map(grid: np.ndarray, infos: list[dict], *, bg: int) -> dict[int, int]:
    candidates = sorted(infos, key=lambda info: int(info["along_extent"]), reverse=True)
    for info in candidates:
        if int(info["along_extent"]) <= _SEED_ALONG:
            continue
        sr0, sc0, sr1, sc1 = info["seed_bbox"]
        shape = (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1))
        chunks = _extract_full_chunks(grid, int(sr0), int(sc0), shape, tuple(map(int, info["direction"])))
        mapping = _infer_color_map_from_chunks(chunks, bg=int(bg))
        if mapping:
            return mapping
    return {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker rays away from '+' anchors by combining a learned color cycle with a terminal fade.

    Input structure:
    - The most common color is the background.
    - One dominant non-background color forms several identical '+'-like anchors with nozzle width 2.
    - Each anchor has one adjacent 2x2 seed tile, and exactly one anchor already shows a long clue ray.

    Rule:
    - Read the long clue ray chunk by chunk: before the tail, each whole-tile repetition keeps the same geometry but advances through one global color permutation.
    - For every anchor, reconstruct the attached 2x2 seed from the nozzle side and project it away from the '+' by repeatedly applying that color cycle.
    - The final three repeated chunks nearest the boundary are then partially replaced by background in a fixed directional order; '+' pixels remain protected.

    Key insight:
    The clue ray mixes two effects: early chunks reveal the reusable color cycle, while the boundary-end chunks are not a different cycle state but the same cycle with an added fade-to-background mask.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    infos = _collect_infos(g, bg=int(bg), plus_col=int(plus_col))
    color_map = _infer_global_color_map(g, infos, bg=int(bg))

    for info in infos:
        sr0, sc0, _sr1, _sc1 = info["seed_bbox"]
        _paint_variant_ray(
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

_H_RANGE = (18, 29)
_W_RANGE = (18, 23)
_N_PLUS_RANGE = (2, 5)
_MIN_RAY_UNITS = 5
_CLUE_MIN_UNITS = 6
_CLUE_IDEAL_UNITS = 7
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 500
_CYCLE_LEN = 3

_PLUS_SHAPES = [
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=int,
    ),
]


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
    if int(e - s + 1) != 2:
        return None

    if force_clue:
        vals = list(map(int, cycle_colors)) + [int(rng.choice(cycle_colors))]
        rng.shuffle(vals)
        return np.array(vals, dtype=int).reshape(2, 2)

    n_colors = int(rng.integers(2, len(cycle_colors) + 1))
    chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
    vals = list(chosen)
    while len(vals) < 4:
        vals.append(int(rng.choice(chosen)))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(2, 2)


def _compute_units(
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


def _project_variant_cells(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
    *,
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], int, int, int]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - tile.shape[0])
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tile.shape[1])

    origins = _chunk_origins(H, W, int(r0), int(c0), tile.shape, direction)
    repeats = max(0, len(origins) - 1)
    fade_count = min(_END_FADE_CHUNKS, repeats)
    fade_start = len(origins) - fade_count

    paint_cells: list[tuple[int, int, int]] = []
    cover_cells: list[tuple[int, int]] = []
    for idx, (rr, cc) in enumerate(origins):
        cur = _apply_color_map(tile, color_map, steps=int(idx))
        if idx > 0 and idx >= fade_start:
            cur = _apply_tail_fade(cur, direction, idx - fade_start + 1, fade_count, bg=int(bg))
        for i in range(tile.shape[0]):
            for j in range(tile.shape[1]):
                rrr = int(rr + i)
                ccc = int(cc + j)
                cover_cells.append((int(rrr), int(ccc)))
                if int(cur[i, j]) != int(bg):
                    paint_cells.append((int(rrr), int(ccc), int(cur[i, j])))

    return paint_cells, cover_cells, int(r0), int(c0), int(len(origins))


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
    bg: int,
    *,
    force_clue: bool = False,
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

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, cycle_colors, rng, force_clue=bool(force_clue))
            if tile is None:
                continue

            units = _compute_units(direction, tile, r0, c0, ph, pw, H, W)
            need = _CLUE_MIN_UNITS if force_clue else _MIN_RAY_UNITS
            if units < need:
                continue

            paint_cells, cover_cells, sr0, sc0, total_units = _project_variant_cells(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
                color_map,
                bg=int(bg),
            )
            if any(int(occ[r, c]) != 0 for r, c in cover_cells):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c in cover_cells:
                occ[r, c] = 2
            for r, c, v in paint_cells:
                out[r, c] = int(v)

            return {
                "tile": np.array(tile, dtype=int, copy=True),
                "cells": paint_cells,
                "cover": cover_cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(total_units),
                "is_clue": bool(force_clue),
            }

    return None


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int, keep_idx: int) -> np.ndarray:
    inp = np.array(out, dtype=int, copy=True)
    for i, p in enumerate(pluses):
        if int(i) == int(keep_idx):
            continue
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(2) for jj in range(2)}
        for r, c in p["cover"]:
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
        int(bg),
        force_clue=True,
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
            int(bg),
            force_clue=False,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, int(bg), keep_idx=0)

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
