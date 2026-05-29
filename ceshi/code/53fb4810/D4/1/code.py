# Variant of 53fb4810
# Dimension: D4 — 源对象形状 (src)
# Change: '+'精灵为小型中心对称色块（3x3十字、3x4胖十字、5x4胖菱形等），bbox至少3x3，左右+上下对称... → 箭头形精灵

from __future__ import annotations

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior / trace
# - The shared anchor color now forms several small arrow blobs instead of plus-like symmetric blobs.
# - Each arrow is one of a few fixed templates, rotated to point U/D/L/R.
# - Each arrow still owns exactly one colored seed tile and one grid keeps one ray pre-extended.
# - Crucial change: the seed no longer sits on the ray side. It is attached on a rotated side-flank
#   (clockwise from the pointing direction), while the arrow's own facing direction tells us where the
#   ray should go. Example: a right-pointing arrow has its seed directly below it, but the seed repeats
#   to the right. Under the old logic, the attached side would have implied "down"; under the new logic,
#   we must read the arrow shape and project "right".
# b) Complexity calibration
# - If the seed stayed on the tip side, the task would visually collapse back toward the original.
# - Making the attachment side orthogonal to the ray direction forces a genuine D4 change while keeping
#   the rest of the task stable: same output-first generation, same periodic repetition, same protected
#   anchor color, same single pre-extended clue ray.
# - Arrow templates are limited to three canonical shapes so recognition is learnable from 2-4 examples.
# - Each generated grid uses at least two arrow orientations so the orientation cue is actually visible.
# c) Hidden assumptions audit against the original main()
# - detect_dominant_big_component_color still works because arrows remain the only components with both
#   bbox height >= 3 and bbox width >= 3; rays are always 1-cell thick.
# - The original "direction from marker adjacency" block is invalid here: adjacency is intentionally
#   orthogonal to the ray. That logic must be removed.
# - The original nozzle-span assert also depended on a plus arm/tip model and is no longer meaningful.
# - The original "protect anchor color while extending" remains valid and is preserved unchanged.
# d) Placement legality
# - Generator places full output first, then erases all but one long clue ray.
# - Arrows are kept 1 cell away from existing occupied cells so no foreign ray becomes adjacent to the
#   wrong arrow and the solver sees exactly one adjacent non-arrow component per arrow.
# - Full rays are checked for in-bounds seed placement, no overlap with occupied cells, and no entry into
#   existing arrow-neighborhood guard cells.
# e) Visual observability
# - Human-solvable cue: arrows visibly point one way while their seed sits on a different side.
# - The kept pre-extended ray demonstrates that the repeated pattern follows the arrow orientation, not
#   the seed attachment side.
# - At least two orientations per grid prevent degenerate samples where direction-from-shape is unseen.
# f) Visual cleanliness
# - Rays do not overlap arrows and do not touch foreign arrows.
# - Objects share one anchor color and use small multi-color seed tiles, matching the original visual role
#   split between source objects and payloads.
# g) Color-rule inferability
# - No color-id ordering is used. Colors only distinguish bg / anchor / seed-payload.
# - Seed colors are copied periodically with no hidden mapping.
# h) Variant metric to check after implementation
# - For every generated arrow, the unique attachment side must be orthogonal to the ray direction.
# - Per sample, at least two distinct arrow orientations should usually appear.
# i) New solver assumptions and generator support
# - Solver assumes every big anchor component exactly matches one rotated arrow template.
# - Solver assumes each arrow has exactly one adjacent non-anchor component (seed or pre-extended ray).
# - Generator explicitly constructs only those cases and rejects any attempt that violates the oracle.

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


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


_DIR_VECS = {
    "R": (0, 1),
    "U": (-1, 0),
    "L": (0, -1),
    "D": (1, 0),
}

_ARROW_SPECS = [
    {
        "mask": np.array(
            [
                [0, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "launch": (2, 2),  # canonical right-arrow: seed attaches below this cell
    },
    {
        "mask": np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "launch": (4, 2),
    },
    {
        "mask": np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=int,
        ),
        "launch": (4, 2),
    },
]

_CANONICAL_ATTACH_NORMAL = (1, 0)  # right-arrow seeds attach on the clockwise flank: below


def _rotate_point(
    point: tuple[int, int],
    shape: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    """Rotate a canonical right-facing point into the target direction."""
    r, c = map(int, point)
    h, w = map(int, shape)
    if direction == "R":
        return r, c
    if direction == "U":
        return w - 1 - c, r
    if direction == "L":
        return h - 1 - r, w - 1 - c
    if direction == "D":
        return c, h - 1 - r
    raise ValueError(direction)


def _rotate_vector(vec: tuple[int, int], direction: str) -> tuple[int, int]:
    """Rotate a canonical right-facing vector into the target direction."""
    dr, dc = map(int, vec)
    if direction == "R":
        return dr, dc
    if direction == "U":
        return -dc, dr
    if direction == "L":
        return -dr, -dc
    if direction == "D":
        return dc, -dr
    raise ValueError(direction)


def _rotate_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "R":
        return patch.copy()
    if direction == "U":
        return np.rot90(patch, k=1).copy()
    if direction == "L":
        return np.rot90(patch, k=2).copy()
    if direction == "D":
        return np.rot90(patch, k=-1).copy()
    raise ValueError(direction)


def _oriented_arrow_spec(spec: dict, direction: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    patch = _rotate_patch(np.array(spec["mask"], dtype=int), direction)
    launch = _rotate_point(tuple(spec["launch"]), np.array(spec["mask"]).shape, direction)
    normal = _rotate_vector(_CANONICAL_ATTACH_NORMAL, direction)
    return patch, launch, normal


def _match_arrow_direction(mask: np.ndarray) -> tuple[int, int] | None:
    for spec in _ARROW_SPECS:
        base = np.array(spec["mask"], dtype=int)
        for direction, vec in _DIR_VECS.items():
            rotated = _rotate_patch(base, direction)
            if rotated.shape == mask.shape and np.array_equal(rotated != 0, mask != 0):
                return tuple(map(int, vec))
    return None


def _component_cells(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    coords = np.argwhere(mask != 0)
    return [(r0 + int(r), c0 + int(c)) for r, c in coords]


def _seed_origin_from_launch(
    launch_abs: tuple[int, int],
    tile_shape: tuple[int, int],
    normal: tuple[int, int],
) -> tuple[int, int]:
    lr, lc = map(int, launch_abs)
    th, tw = map(int, tile_shape)
    nr, nc = map(int, normal)
    if (nr, nc) == (1, 0):
        return lr + 1, lc
    if (nr, nc) == (-1, 0):
        return lr - th, lc
    if (nr, nc) == (0, 1):
        return lr, lc + 1
    if (nr, nc) == (0, -1):
        return lr, lc - tw
    raise ValueError(normal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles forward from arrow-shaped anchors by repeating them to the grid edge.

    Input structure:
    - bg = most common color.
    - A dominant non-bg color forms several small arrow sprites, possibly in different rotations.
    - Each arrow has exactly one adjacent colored marker component; one of these rays may already be long.

    Rule:
    - For each arrow, recognize which way the arrow points from its shape.
    - Take the adjacent marker tile (or already-extended marker ray) as the repeating payload.
    - Repeat that payload in the arrow's pointing direction until the boundary, without overwriting arrow pixels.

    Key insight: The marker's attachment side is no longer the direction cue; the arrow silhouette itself
    determines where the periodic ray must continue.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    arrow_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if arrow_col is None:
        return out

    arrow_lab, an = label_mask(g == int(arrow_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(arrow_col))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

    for ak in range(1, int(an) + 1):
        arrow_mask = arrow_lab == int(ak)
        arrow_coords = np.argwhere(arrow_mask)
        if arrow_coords.size == 0:
            continue

        ar0, ac0, ar1, ac1 = bbox_coords(arrow_coords)
        local_arrow = g[int(ar0):int(ar1) + 1, int(ac0):int(ac1) + 1] == int(arrow_col)
        direction = _match_arrow_direction(local_arrow)
        if direction is None:
            continue

        adj_ids: set[int] = set()
        for r, c in arrow_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
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
            protect={int(arrow_col)},
        )

    return out


# --------------- generator (D4 arrow variant) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_ARROW_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [2, 3]
_SEED_MAX_COLORS = 4
_MIN_RAY_TILES = 3
_PRE_EXT_IDEAL_TILES = 7
_PLACEMENT_TRIES = 140
_GEN_RETRIES = 400


def _build_seed_tile(
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    along = int(rng.choice(_SEED_ALONG_CHOICES))
    dr, dc = map(int, direction)
    th, tw = (along, 1) if dc == 0 else (1, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _project_tile_coords(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> list[tuple[int, int, int]]:
    th, tw = tile.shape
    dr, dc = map(int, direction)
    cells: list[tuple[int, int, int]] = []

    if (dr, dc) == (0, 1):
        c = int(seed_c0)
        while c < W:
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(seed_r0 + i), int(cc), int(tile[i, j])))
            c += tw
    elif (dr, dc) == (0, -1):
        c = int(seed_c0)
        while c + tw > 0:
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(seed_r0 + i), int(cc), int(tile[i, j])))
            c -= tw
    elif (dr, dc) == (1, 0):
        r = int(seed_r0)
        while r < H:
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((int(rr), int(seed_c0 + j), int(tile[i, j])))
            r += th
    elif (dr, dc) == (-1, 0):
        r = int(seed_r0)
        while r + th > 0:
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((int(rr), int(seed_c0 + j), int(tile[i, j])))
            r -= th
    else:
        raise ValueError(direction)

    return cells


def _count_ray_tiles(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    th, tw = tile.shape
    dr, dc = map(int, direction)
    count = 0

    if (dr, dc) == (0, 1):
        c = int(seed_c0)
        while c < W:
            count += 1
            c += tw
    elif (dr, dc) == (0, -1):
        c = int(seed_c0)
        while c + tw > 0:
            count += 1
            c -= tw
    elif (dr, dc) == (1, 0):
        r = int(seed_r0)
        while r < H:
            count += 1
            r += th
    elif (dr, dc) == (-1, 0):
        r = int(seed_r0)
        while r + th > 0:
            count += 1
            r -= th
    else:
        raise ValueError(direction)

    return int(count)


def _in_bounds_bbox(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= int(r0) and 0 <= int(c0) and int(r0) + int(h) <= H and int(c0) + int(w) <= W


def _cells_have_gap(cells: list[tuple[int, int]], occ: np.ndarray) -> bool:
    H, W = occ.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(occ[r, c]) != 0:
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) != 0:
                return False
    return True


def _mark_arrow_guard(guard: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = guard.shape
    for r, c in cells:
        guard[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                guard[rr, cc] = True


def _place_arrow_and_ray(
    rng: np.random.Generator,
    spec: dict,
    direction_name: str,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    arrow_guard: np.ndarray,
    out: np.ndarray,
    arrow_color: int,
) -> dict | None:
    patch, launch_local, attach_normal = _oriented_arrow_spec(spec, direction_name)
    direction = _DIR_VECS[direction_name]
    tile = _build_seed_tile(direction, palette, rng)
    ph, pw = patch.shape
    th, tw = tile.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(0, H - ph + 1))
        c0 = int(rng.integers(0, W - pw + 1))
        arrow_cells = _component_cells(patch, (r0, c0))

        if not _cells_have_gap(arrow_cells, occ):
            continue

        launch_abs = (r0 + int(launch_local[0]), c0 + int(launch_local[1]))
        seed_r0, seed_c0 = _seed_origin_from_launch(launch_abs, (th, tw), attach_normal)
        if not _in_bounds_bbox(H, W, seed_r0, seed_c0, th, tw):
            continue

        ray_tiles = _count_ray_tiles(H, W, seed_r0, seed_c0, direction, tile)
        if ray_tiles < _MIN_RAY_TILES:
            continue

        ray_cells = _project_tile_coords(H, W, seed_r0, seed_c0, direction, tile)
        blocked = False
        for rr, cc, _v in ray_cells:
            if int(occ[rr, cc]) != 0 or bool(arrow_guard[rr, cc]):
                blocked = True
                break
        if blocked:
            continue

        for rr, cc in arrow_cells:
            occ[rr, cc] = 1
            out[rr, cc] = int(arrow_color)
        _mark_arrow_guard(arrow_guard, arrow_cells)

        for rr, cc, vv in ray_cells:
            occ[rr, cc] = 2
            out[rr, cc] = int(vv)

        return {
            "tile": tile,
            "cells": ray_cells,
            "seed_r0": int(seed_r0),
            "seed_c0": int(seed_c0),
            "tiles": int(ray_tiles),
            "direction": tuple(map(int, direction)),
            "attach_normal": tuple(map(int, attach_normal)),
        }

    return None


def _build_input_from_output(
    out: np.ndarray,
    arrows: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pre_idx = max(range(len(arrows)), key=lambda i: arrows[i]["tiles"])
    if arrows[pre_idx]["tiles"] < _PRE_EXT_IDEAL_TILES:
        candidates = [i for i, meta in enumerate(arrows) if meta["tiles"] >= _PRE_EXT_IDEAL_TILES]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, meta in enumerate(arrows):
        if i == pre_idx:
            continue
        th, tw = meta["tile"].shape
        sr0, sc0 = meta["seed_r0"], meta["seed_c0"]
        seed_set = {(sr0 + rr, sc0 + cc) for rr in range(th) for cc in range(tw)}
        for r, c, _v in meta["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _sample_direction_names(rng: np.random.Generator, n: int) -> list[str]:
    dirs = ["U", "D", "L", "R"]
    while True:
        chosen = [str(rng.choice(dirs)) for _ in range(n)]
        if len(set(chosen)) >= 2:
            return chosen


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    arrow_color = int(rng.choice(others))
    palette = [c for c in others if c != arrow_color]

    spec = _ARROW_SPECS[int(rng.integers(0, len(_ARROW_SPECS)))]
    n = int(rng.integers(*_N_ARROW_RANGE))
    direction_names = _sample_direction_names(rng, n)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    arrow_guard = np.zeros((H, W), dtype=bool)

    arrows: list[dict] = []
    for direction_name in direction_names:
        meta = _place_arrow_and_ray(
            rng,
            spec,
            direction_name,
            palette,
            H,
            W,
            occ,
            arrow_guard,
            out,
            arrow_color,
        )
        if meta is None:
            return None
        arrows.append(meta)

    inp = _build_input_from_output(out, arrows, bg, rng)

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
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
