# Variant of 53fb4810
# Dimension: D2 — 射线方向决定机制 (path)
# Change: 方向由标记瓦片相对于'+'精灵的附着位置决定——标记在'+'的哪一侧，射线就向那一侧的反方向投射... → 由精灵凸出形状决定方向

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    - Each source object is still a plus-like sprite in one shared color, with one marker tile touching it.
#    - The difference is that the plus is now slightly asymmetric: one arm is longer / fatter, and that protruding
#      side indicates the ray direction.
#    - To make the change observable, the marker is attached on a perpendicular side instead of on the emitting side.
#      Example: if the plus points right, its seed may sit above it, and the output extends that seed horizontally
#      to the right. Under the original logic the same seed would have gone upward, so the variant is behaviorally
#      distinct on the same input.
# b) Complexity calibration:
#    - Direction only has four cardinal values, and the asymmetric cue is repeated across multiple objects per grid.
#    - We use two related asymmetric-plus families, both rotated, so humans see a stable "longer arm = ray direction"
#      rule without exploding the shape vocabulary.
# c) Old-assumption audit:
#    - Original `main()` inferred `(dr,dc)` from seed/plus contact. That is invalid here because attachment is forced
#      to be perpendicular to the true ray.
#    - Original alignment asserts matched the seed bbox to the emitting arm tip. Those asserts are also invalid here,
#      because the seed is no longer attached on the emitting side.
#    - `extend_repeating_tile()` itself remains valid: once the seed bbox and true direction are known, periodic
#      extension works unchanged.
# d) Placement legality:
#    - Generator chooses a rotated asymmetric plus, then chooses an attachment side perpendicular to that direction.
#    - Seed bboxes are anchored flush to that side, always fully in bounds, and full rays are rejected if they would
#      overlap or even touch any previously placed plus sprite.
#    - Future pluses are also forbidden from touching existing rays, so solver-side adjacency stays unambiguous.
# e) Visual observability:
#    - The core cue is purely geometric and visible: the only longer / bulkier arm of the plus points the ray.
#    - Because seeds never attach on the emitting side, attachment cannot explain the output; the shape must.
# f) Visual cleanliness:
#    - Objects are separated by occupancy checks, plus halos, and ray-vs-plus non-touch constraints.
#    - Each ray is still a clean axis-aligned repetition of one compact tile.
# g) New solver assumptions and generator support:
#    - Solver assumes every plus crop matches one of the rotated asymmetric templates below.
#    - Solver assumes each plus has exactly one adjacent non-plus component (possibly already pre-extended).
#    - Generator enforces both conditions and verifies them by `main(input) == output`.


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

    best_key: tuple[int, int] | None = None  # (num_big, total_big_area)
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

    if (dr, dc) == (0, 1):  # right
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):  # left
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):  # down
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):  # up
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


_BASE_PLUS_RIGHT = [
    np.array([
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
    ], dtype=int),
    np.array([
        [0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 0],
    ], dtype=int),
]

_ROT_DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # k=0,1,2,3 CCW from canonical RIGHT
_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIR_TO_SIDE = {
    (1, 0): "D",
    (-1, 0): "U",
    (0, 1): "R",
    (0, -1): "L",
}
_PERP_ATTACH_SIDES = {
    (1, 0): ["L", "R"],
    (-1, 0): ["L", "R"],
    (0, 1): ["U", "D"],
    (0, -1): ["U", "D"],
}


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.array(mask, dtype=int))


def _build_shape_lookup() -> dict[tuple[tuple[int, ...], ...], tuple[int, int]]:
    lookup: dict[tuple[tuple[int, ...], ...], tuple[int, int]] = {}
    for base in _BASE_PLUS_RIGHT:
        for k, direction in enumerate(_ROT_DIRS):
            rot = np.rot90(base, k=k)
            lookup[_shape_key(rot)] = direction
    return lookup


_PLUS_SHAPE_TO_DIRECTION = _build_shape_lookup()


def _direction_from_plus_shape(plus_mask: np.ndarray) -> tuple[int, int] | None:
    pts = np.argwhere(plus_mask)
    if pts.size == 0:
        return None
    key = _shape_key(crop_bbox(plus_mask.astype(int), bbox_coords(pts)))
    return _PLUS_SHAPE_TO_DIRECTION.get(key)


def _edge_span(mask: np.ndarray, side: str) -> tuple[int, int]:
    if side == "U":
        xs = np.where(mask[0] != 0)[0]
    elif side == "D":
        xs = np.where(mask[-1] != 0)[0]
    elif side == "L":
        xs = np.where(mask[:, 0] != 0)[0]
    elif side == "R":
        xs = np.where(mask[:, -1] != 0)[0]
    else:
        raise ValueError(side)
    if xs.size == 0:
        raise ValueError(f"edge {side} is empty")
    return int(xs[0]), int(xs[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker tile in the direction encoded by an asymmetric '+' sprite.

    Input structure:
    - bg is the most frequent color.
    - One dominant non-bg color forms several plus-like sprites that are almost symmetric except for one protruding arm.
    - Each plus touches exactly one compact marker tile, and one marker may already be pre-extended as a long periodic strip.

    Rule:
    - For each plus, read the ray direction from the side where its shape sticks out farther.
    - Ignore which side the marker is attached to; use the marker's own bbox as the seed tile.
    - Repeat that tile periodically from its current position in the shape-indicated direction until the grid boundary,
      never painting over the plus color.

    Key insight: The attachment side is now a distractor; the only reliable direction cue is the plus sprite's subtle
    asymmetry, and the repeated pattern follows that geometric cue.
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
        plus_mask = (plus_lab == int(pk))
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        direction = _direction_from_plus_shape(plus_mask)
        if direction is None:
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
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
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


# --------------- generator (variant) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)            # grid height [min, max)
_W_RANGE = (13, 23)            # grid width [min, max)
_N_PLUS_RANGE = (2, 4)         # number of plus sprites [min, max)
_FREE_DIM_CHOICES = [1, 2]     # seed dimension not fixed by the attachment edge span
_SEED_MAX_COLORS = 4           # max distinct colors in a seed tile
_MIN_RAY_UNITS = 2             # minimum tile repetitions beyond the seed
_PRE_EXT_IDEAL_UNITS = 10      # preferred minimum units for the pre-extended ray
_PLACEMENT_TRIES = 140         # attempts per plus placement
_GEN_RETRIES = 400             # outer retry limit in generate_input


def _build_seed_tile(
    side_span: int,
    attach_side: str,
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if attach_side in {"U", "D"}:
        th = int(rng.choice(_FREE_DIM_CHOICES))
        tw = int(side_span)
        if th == 1 and tw == 1:
            th = 2
    else:
        th = int(side_span)
        tw = int(rng.choice(_FREE_DIM_CHOICES))
        if th == 1 and tw == 1:
            tw = 2

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_anchor(
    plus_bbox: tuple[int, int, int, int],
    plus_shape: np.ndarray,
    attach_side: str,
    tile: np.ndarray,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    th, tw = tile.shape
    s, _e = _edge_span(plus_shape, attach_side)

    if attach_side == "U":
        return int(pr0 - th), int(pc0 + s)
    if attach_side == "D":
        return int(pr1 + 1), int(pc0 + s)
    if attach_side == "L":
        return int(pr0 + s), int(pc0 - tw)
    if attach_side == "R":
        return int(pr0 + s), int(pc1 + 1)
    raise ValueError(attach_side)


def _project_tile_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int, int]]:
    th, tw = tile.shape
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []

    if dc != 0:
        c = int(seed_c0)
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                rr = seed_r0 + i
                if not (0 <= rr < H):
                    continue
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((rr, cc, int(tile[i, j])))
            c += step
    else:
        r = int(seed_r0)
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if not (0 <= rr < H):
                    continue
                for j in range(tw):
                    cc = seed_c0 + j
                    if 0 <= cc < W:
                        cells.append((rr, cc, int(tile[i, j])))
            r += step

    return cells


def _compute_ray_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    H: int,
    W: int,
) -> int:
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return (H - (seed_r0 + th)) // th
    if dr == -1:
        return seed_r0 // th
    if dc == 1:
        return (W - (seed_c0 + tw)) // tw
    return seed_c0 // tw


def _plus_can_place(
    plus_cells: list[tuple[int, int]],
    occ: np.ndarray,
    plus_occ: np.ndarray,
) -> bool:
    H, W = occ.shape
    for r, c in plus_cells:
        if occ[r, c] != 0 or plus_occ[r, c] == 1:
            return False
        for dr, dc in _DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] == 2:
                return False
    return True


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
) -> dict | None:
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(0, H - ph + 1))
        c0 = int(rng.integers(0, W - pw + 1))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]
        if not _plus_can_place(plus_cells, occ, plus_occ):
            continue

        attach_sides = list(_PERP_ATTACH_SIDES[direction])
        rng.shuffle(attach_sides)
        for attach_side in attach_sides:
            s, e = _edge_span(plus_shape, attach_side)
            tile = _build_seed_tile(int(e - s + 1), attach_side, palette, rng)
            seed_r0, seed_c0 = _seed_anchor(
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                attach_side,
                tile,
            )
            th, tw = tile.shape
            if seed_r0 < 0 or seed_c0 < 0 or seed_r0 + th > H or seed_c0 + tw > W:
                continue

            units = _compute_ray_units(direction, tile, seed_r0, seed_c0, H, W)
            if units < _MIN_RAY_UNITS:
                continue

            cells = _project_tile_coords(H, W, tile, seed_r0, seed_c0, direction)
            plus_set = set(plus_cells)
            blocked = False
            for r, c, _v in cells:
                if (r, c) in plus_set or occ[r, c] != 0 or plus_occ[r, c] == 1:
                    blocked = True
                    break
            if blocked:
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = plus_color
                plus_occ[r, c] = 1
                for dr2, dc2 in _DIRECTIONS:
                    rr = r + dr2
                    cc = c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = v

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": int(seed_r0),
                "seed_c0": int(seed_c0),
                "units": int(units),
                "direction": tuple(map(int, direction)),
                "attach_side": attach_side,
                "shape_side": _DIR_TO_SIDE[direction],
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)       # 0 bg, 1 plus, 2 ray/seed
    plus_occ = np.zeros((H, W), dtype=int)  # plus cells + 4-neighbor halo

    pluses: list[dict] = []
    for _ in range(N):
        base = _BASE_PLUS_RIGHT[int(rng.integers(0, len(_BASE_PLUS_RIGHT)))]
        k = int(rng.integers(0, 4))
        plus_shape = np.rot90(base, k=k)
        direction = _ROT_DIRS[k]
        info = _place_plus_and_ray(
            rng, plus_shape, direction, palette, H, W, occ, plus_occ, out, plus_color
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
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
