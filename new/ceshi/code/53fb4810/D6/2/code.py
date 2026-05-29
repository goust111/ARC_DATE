# Variant of 53fb4810
# Dimension: D6 — 终点条件 (dst)
# Change: 射线一直延伸到网格边界为止... → 遇到另一个'+'精灵时停止

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D6 variant
# a) Concrete behavior:
#    The original task lets each seed tile keep repeating until the canvas edge.
#    In this variant, the same seed still starts on one side of a '+' sprite and
#    still repeats with the same period, but the repetition halts at the first
#    later '+' encountered on that straight lane. The input still shows one fully
#    expanded example ray, but that example now visibly bridges two '+' sprites
#    instead of disappearing into the border. In a 4-plus rectangle, the top
#    left '+' can emit a rightward striped seed; the output fills the horizontal
#    corridor only until the top right '+' is reached. The old logic would keep
#    going past that second '+', only skipping the protected plus pixels; the new
#    logic must stop just before that target plus.
# b) Complexity calibration:
#    To make the new stop condition obvious, the generator uses four identical
#    pluses placed on the corners of a rectangle and routes four rays around the
#    perimeter. Every sample therefore contains multiple plus-to-plus corridors,
#    repeated evidence of the same rule, and no dependence on color ids or rare
#    special cases.
# c) Hidden assumptions from the old solver:
#    - `extend_repeating_tile(...)` assumed the boundary is always the endpoint.
#      That is invalid here and must be replaced by a "find next plus ahead"
#      search.
#    - The original input never had a marker component touching two pluses.
#      In this variant, the one pre-expanded teaching ray does touch both its
#      source plus and its target plus. If the solver treated every adjacent
#      non-plus component as a seed, the target plus would misread that long ray
#      as its own marker. The fix is to ignore components adjacent to anything
#      other than exactly one plus; compact seeds satisfy that property, the
#      pre-expanded bridge does not.
#    - The original bbox/alignment assertions remain valid for compact seeds, so
#      they are preserved unchanged.
# d) Placement legality:
#    The rectangle layout guarantees disjoint plus sprites, disjoint ray lanes,
#    correct nozzle/seed alignment, and at least one reachable target plus in
#    every ray direction. The output is built first, then non-example rays are
#    erased back to their seed tiles to form the input, and the solver is used as
#    an oracle.
# e) Visual observability:
#    Humans can see that the visible long ray terminates exactly at another plus
#    rather than the border. Because the same interaction appears on all four
#    sides in the output, the change is inferable from only a few examples.
# f) Visual cleanliness:
#    Rays occupy four separate rectangle edges, so no two rays share a corridor
#    and there is no noisy overlap except the intended contact at their target
#    pluses.
# g) Color inference:
#    Colors are only decorative payload colors inside the seed tiles; no sorting
#    or rule branch depends on color ids. Structural cues alone determine the
#    transformation.


_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


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


def _find_stop_before_plus(
    plus_mask: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: tuple[int, int],
) -> int | None:
    """Return the first row/col occupied by a later plus on the ray lane."""
    H, W = plus_mask.shape
    dr, dc = map(int, direction)

    if (dr, dc) == (0, 1):
        for cc in range(int(c0 + w), W):
            if bool(np.any(plus_mask[int(r0) : int(r0 + h), int(cc)])):
                return int(cc)
    elif (dr, dc) == (0, -1):
        for cc in range(int(c0 - 1), -1, -1):
            if bool(np.any(plus_mask[int(r0) : int(r0 + h), int(cc)])):
                return int(cc)
    elif (dr, dc) == (1, 0):
        for rr in range(int(r0 + h), H):
            if bool(np.any(plus_mask[int(rr), int(c0) : int(c0 + w)])):
                return int(rr)
    elif (dr, dc) == (-1, 0):
        for rr in range(int(r0 - 1), -1, -1):
            if bool(np.any(plus_mask[int(rr), int(c0) : int(c0 + w)])):
                return int(rr)
    else:
        raise ValueError("_find_stop_before_plus: direction must be axial")
    return None


def extend_repeating_tile_until_plus(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    plus_mask: np.ndarray,
    protect: set[int] | None = None,
) -> None:
    """Extend a tile pattern until the next plus on that lane, not to the boundary."""
    protect = set() if protect is None else set(map(int, protect))
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    stop = _find_stop_before_plus(plus_mask, r0, c0, int(h), int(w), (dr, dc))
    if stop is None:
        return

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        for k, tc in enumerate(range(int(c0 + w), int(stop))):
            col_idx = k % int(w)
            for i in range(int(h)):
                put(int(r0 + i), int(tc), int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(int(c0 - 1), int(stop), -1), start=1):
            col_idx = (-k) % int(w)
            for i in range(int(h)):
                put(int(r0 + i), int(tc), int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        for k, tr in enumerate(range(int(r0 + h), int(stop))):
            row_idx = k % int(h)
            for j in range(int(w)):
                put(int(tr), int(c0 + j), int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(int(r0 - 1), int(stop), -1), start=1):
            row_idx = (-k) % int(h)
            for j in range(int(w)):
                put(int(tr), int(c0 + j), int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile_until_plus: direction must be axial")


def _adjacent_plus_ids(component_mask: np.ndarray, plus_lab: np.ndarray) -> set[int]:
    ids: set[int] = set()
    H, W = plus_lab.shape
    for r, c in np.argwhere(component_mask):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(plus_lab[rr, cc]) > 0:
                ids.add(int(plus_lab[rr, cc]))
    return ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from '+' shapes by repeating them until they meet another '+'.

    Input structure:
    - bg is the most common color.
    - One dominant non-bg color forms several identical '+'-like sprites.
    - Each plus has one small marker tile on one side, and exactly one marker ray is already fully expanded as an example bridge between two pluses.

    Rule:
    - For each compact marker tile that touches exactly one '+', read which side of that '+' it is attached to.
    - Repeat the tile away from its source '+', but stop as soon as the lane reaches the next '+' on that axis.
    - Existing '+' pixels stay protected, and the already expanded bridge in the input is left as-is.

    Key insight:
    - The marker's attachment side still determines the ray direction, but the endpoint is no longer the canvas edge; it is the first later '+' encountered on the same aligned corridor.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    plus_mask_all = plus_lab > 0
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, mn = label_mask(non_mask, connectivity=4)

    comp_plus_ids: dict[int, set[int]] = {}
    for cid in range(1, int(mn) + 1):
        comp_mask = mark_lab == int(cid)
        comp_plus_ids[int(cid)] = _adjacent_plus_ids(comp_mask, plus_lab)

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
                    cid = int(mark_lab[rr, cc])
                    if comp_plus_ids.get(cid, set()) == {int(pk)}:
                        adj_ids.add(cid)

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
                for dr, dc in _DIRECTIONS:
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

            extend_repeating_tile_until_plus(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                plus_mask=plus_mask_all,
                protect={int(plus_col)},
            )

    return out


# --------------- generator (D6 rectangle-cycle variant) ---------------

_H_RANGE = (17, 30)
_W_RANGE = (17, 30)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_LANE_GAP = 5
_PRE_EXT_IDEAL_LEN = 10
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


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive span of the arm tip along the cross-axis."""
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
    """Build a compact seed tile whose cross-axis matches the chosen arm tip."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)
    dr, dc = direction
    if dc == 0:
        return (int(pr1 + 1) if dr == 1 else int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1) if dc == 1 else int(pc0 - tw))


def _collect_ray_cells(
    plus_mask_all: np.ndarray,
    plus_bbox: tuple[int, int, int, int],
    plus_shape: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int] | None:
    """Return full ray cells (seed + repeated tail) until the next plus."""
    r0, c0 = _seed_origin(plus_bbox, plus_shape, direction, tile)
    th, tw = tile.shape
    stop = _find_stop_before_plus(plus_mask_all, int(r0), int(c0), int(th), int(tw), direction)
    if stop is None:
        return None

    cells: list[tuple[int, int, int]] = []
    for i in range(int(th)):
        for j in range(int(tw)):
            cells.append((int(r0 + i), int(c0 + j), int(tile[i, j])))

    dr, dc = direction
    if (dr, dc) == (0, 1):
        for k, tc in enumerate(range(int(c0 + tw), int(stop))):
            col_idx = k % int(tw)
            for i in range(int(th)):
                cells.append((int(r0 + i), int(tc), int(tile[i, col_idx])))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(int(c0 - 1), int(stop), -1), start=1):
            col_idx = (-k) % int(tw)
            for i in range(int(th)):
                cells.append((int(r0 + i), int(tc), int(tile[i, col_idx])))
    elif (dr, dc) == (1, 0):
        for k, tr in enumerate(range(int(r0 + th), int(stop))):
            row_idx = k % int(th)
            for j in range(int(tw)):
                cells.append((int(tr), int(c0 + j), int(tile[row_idx, j])))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(int(r0 - 1), int(stop), -1), start=1):
            row_idx = (-k) % int(th)
            for j in range(int(tw)):
                cells.append((int(tr), int(c0 + j), int(tile[row_idx, j])))
    else:
        raise ValueError("_collect_ray_cells: direction must be axial")

    return cells, int(r0), int(c0)


def _build_input_from_output(
    out: np.ndarray,
    rays: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep exactly one fully drawn bridge and erase the others back to their seeds."""
    pre_idx = max(range(len(rays)), key=lambda i: len(rays[i]["cells"]))
    candidates = [i for i, ray in enumerate(rays) if len(ray["cells"]) >= _PRE_EXT_IDEAL_LEN]
    if candidates:
        pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, ray in enumerate(rays):
        if i == pre_idx:
            continue
        th, tw = ray["tile"].shape
        sr0, sc0 = ray["seed_r0"], ray["seed_c0"]
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(int(th)) for jj in range(int(tw))}
        for r, c, _ in ray["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    ph, pw = plus_shape.shape

    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    max_gap_y = int(H - 2 * ph - 2)
    max_gap_x = int(W - 2 * pw - 2)
    if max_gap_y < _MIN_LANE_GAP or max_gap_x < _MIN_LANE_GAP:
        return None

    gap_y = int(rng.integers(_MIN_LANE_GAP, max_gap_y + 1))
    gap_x = int(rng.integers(_MIN_LANE_GAP, max_gap_x + 1))

    max_top = int(H - 2 * ph - gap_y - 1)
    max_left = int(W - 2 * pw - gap_x - 1)
    if max_top < 1 or max_left < 1:
        return None

    top = int(rng.integers(1, max_top + 1))
    left = int(rng.integers(1, max_left + 1))
    bottom = int(top + ph + gap_y)
    right = int(left + pw + gap_x)

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    out = np.full((H, W), bg, dtype=int)
    plus_mask_all = np.zeros((H, W), dtype=bool)

    corner_boxes = [
        (top, left, top + ph - 1, left + pw - 1),
        (top, right, top + ph - 1, right + pw - 1),
        (bottom, right, bottom + ph - 1, right + pw - 1),
        (bottom, left, bottom + ph - 1, left + pw - 1),
    ]

    for r0, c0, r1, c1 in corner_boxes:
        for i in range(ph):
            for j in range(pw):
                if int(plus_shape[i, j]) == 1:
                    rr = int(r0 + i)
                    cc = int(c0 + j)
                    plus_mask_all[rr, cc] = True
                    out[rr, cc] = int(plus_color)

    clockwise = bool(rng.integers(0, 2))
    if clockwise:
        edges = [
            (0, 1, (0, 1)),
            (1, 2, (1, 0)),
            (2, 3, (0, -1)),
            (3, 0, (-1, 0)),
        ]
    else:
        edges = [
            (0, 3, (1, 0)),
            (3, 2, (0, 1)),
            (2, 1, (-1, 0)),
            (1, 0, (0, -1)),
        ]

    ray_occ = np.zeros((H, W), dtype=bool)
    rays: list[dict] = []
    for src_idx, dst_idx, direction in edges:
        tile = _build_seed_tile(plus_shape, direction, palette, rng)
        if tile is None:
            return None
        ray_info = _collect_ray_cells(plus_mask_all, corner_boxes[src_idx], plus_shape, direction, tile)
        if ray_info is None:
            return None
        cells, sr0, sc0 = ray_info
        if len(cells) <= int(tile.size):
            return None
        for r, c, _ in cells:
            if plus_mask_all[int(r), int(c)]:
                return None
            if ray_occ[int(r), int(c)]:
                return None
        for r, c, v in cells:
            ray_occ[int(r), int(c)] = True
            out[int(r), int(c)] = int(v)
        rays.append(
            {
                "src_idx": int(src_idx),
                "dst_idx": int(dst_idx),
                "direction": tuple(map(int, direction)),
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
            }
        )

    inp = _build_input_from_output(out, rays, bg, rng)

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
