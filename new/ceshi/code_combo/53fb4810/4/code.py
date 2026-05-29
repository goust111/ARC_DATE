# Combo variant of 53fb4810 (index 4)
# combo_spec: D2/0+D4/1
# Name: 箭头形精灵凸出方向

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
# - Original 53fb4810 uses symmetric '+' anchors, and the seed's attachment side determines the ray direction.
# - This combo replaces the source object with arrow-shaped anchors and simultaneously changes the direction cue:
#   the ray must follow the arrow's protruding head, not the side where the seed happens to touch.
# - To make the fusion behavior visible on the grid, seeds are attached on one of the two perpendicular flanks.
#   Example: a right-facing arrow may have its seed above or below it, but the output still repeats that seed to
#   the right. Under the original task, the same input would have extended upward or downward instead.
# b) Complexity calibration
# - The direction space stays small (four cardinals), but the attachment side is deliberately decoupled from it.
# - Arrow templates are limited to a few reusable families so humans can learn "the protruding tip points the ray"
#   from 2-4 examples instead of memorizing a large shape vocabulary.
# - Each grid uses at least two arrow orientations, so the shape cue is actually demonstrated inside one sample.
# c) Hidden assumptions audit against the original main()
# - `detect_dominant_big_component_color()` still works because arrows remain the only components with bbox >= 3x3.
# - The original adjacency-derived direction logic is invalid: the seed is intentionally placed on a distractor side.
# - The original plus-arm alignment asserts are also invalid, because there is no emitting arm / nozzle model anymore.
# - `extend_repeating_tile()` remains correct once we have the true direction and the seed bbox.
# d) Placement legality
# - The generator is output-first: place full arrows and full rays first, then erase all but one long clue ray.
# - Arrows, rays, and their 4-neighborhood guards are kept separate so two non-arrow components never merge under
#   `label_mask((grid != bg) & (grid != arrow_color))`.
# - Candidate seeds are rejected if they go out of bounds, overlap, or even touch previously committed rays / arrows.
# e) Visual observability
# - The crucial cue is human-visible geometry: each arrow has one protruding tip and that tip points the repetition.
# - Because the seed may appear on either perpendicular flank, the attachment side cannot explain the behavior.
# - The kept pre-extended clue ray shows that the repeated payload follows the arrow orientation rather than contact.
# f) Visual cleanliness
# - One shared color is reserved for all anchors; payload colors are drawn from the remaining palette.
# - No ray is allowed to touch a foreign arrow or another ray, so each object stays visually isolated.
# - The periodic payload remains a small compact tile, preserving the original task's clean ray aesthetic.
# g) Color-rule inferability
# - No ordering on color ids is used. Colors only separate background / arrow anchors / payload tiles.
# - Payload colors are copied verbatim during repetition; there is no hidden remapping to infer.
# h) Non-degeneracy metric to validate after implementation
# - For every generated arrow, the attachment side should be orthogonal to the true ray direction.
# - Across generated samples, the combo output should differ from the original attachment-based output almost always.
# i) New solver assumptions and generator support
# - Solver assumes each big anchor component exactly matches one rotated arrow template below.
# - Solver assumes each arrow has exactly one adjacent non-arrow component (seed or pre-extended ray component).
# - Generator enforces those assumptions with shape templates, flank-only attachment, and no-touch guards.


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
_DIR_ORDER = ["R", "U", "L", "D"]
_PERP_ATTACH_SIDES = {
    "R": ["U", "D"],
    "L": ["U", "D"],
    "U": ["L", "R"],
    "D": ["L", "R"],
}

_ARROW_MASKS = [
    np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.array(mask, dtype=int))


def _rotate_patch(patch: np.ndarray, direction_name: str) -> np.ndarray:
    if direction_name == "R":
        return patch.copy()
    if direction_name == "U":
        return np.rot90(patch, k=1).copy()
    if direction_name == "L":
        return np.rot90(patch, k=2).copy()
    if direction_name == "D":
        return np.rot90(patch, k=-1).copy()
    raise ValueError(direction_name)


def _build_arrow_lookup() -> dict[tuple[tuple[int, ...], ...], tuple[int, int]]:
    lookup: dict[tuple[tuple[int, ...], ...], tuple[int, int]] = {}
    for base in _ARROW_MASKS:
        for direction_name in _DIR_ORDER:
            rot = _rotate_patch(base, direction_name)
            lookup[_shape_key(rot)] = tuple(map(int, _DIR_VECS[direction_name]))
    return lookup


_ARROW_LOOKUP = _build_arrow_lookup()


def _match_arrow_direction(mask: np.ndarray) -> tuple[int, int] | None:
    return _ARROW_LOOKUP.get(_shape_key(mask.astype(int)))


def _side_from_direction(direction: tuple[int, int]) -> str:
    for side, vec in _DIR_VECS.items():
        if tuple(map(int, vec)) == tuple(map(int, direction)):
            return side
    raise ValueError(direction)


def _edge_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    if side == "U":
        cols = np.where(mask[0] != 0)[0]
        return [(0, int(c)) for c in cols]
    if side == "D":
        cols = np.where(mask[-1] != 0)[0]
        return [(mask.shape[0] - 1, int(c)) for c in cols]
    if side == "L":
        rows = np.where(mask[:, 0] != 0)[0]
        return [(int(r), 0) for r in rows]
    if side == "R":
        rows = np.where(mask[:, -1] != 0)[0]
        return [(int(r), mask.shape[1] - 1) for r in rows]
    raise ValueError(side)


def detect_arrow_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
) -> int | None:
    """Choose the color whose big components best match the arrow template family."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        matched_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=int)
            for rr, cc in coords:
                local[int(rr) - int(r0), int(cc) - int(c0)] = 1
            if _match_arrow_direction(local) is not None:
                matched_areas.append(int(coords.shape[0]))
        if not matched_areas:
            continue
        key = (int(len(matched_areas)), int(sum(matched_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


# Extend marker rays from arrow-shaped anchors using the arrow head as the only direction cue.
# Input structure:
# - The most common color is background.
# - One dominant non-background color forms several arrow sprites, all drawn from the same small template family.
# - Each arrow touches exactly one colored marker component, and exactly one of those components may already be a long
#   repeated ray; the others remain compact seeds.
# Rule:
# - Recognize the arrow's facing direction from its protruding tip, not from the marker's contact side.
# - Take the adjacent marker component's bbox as the tile to repeat.
# - Repeat that tile from its current seed position all the way to the boundary in the arrow-indicated direction,
#   while preserving arrow-colored cells.
# Key insight:
# - The marker is deliberately attached on a perpendicular flank, so the old "seed side determines direction" rule
#   fails. Only the arrow silhouette's asymmetric protrusion consistently identifies the ray direction.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    arrow_col = detect_arrow_component_color(g, bg=bg, connectivity=4)
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
        local_arrow = arrow_mask[int(ar0):int(ar1) + 1, int(ac0):int(ac1) + 1]
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


# --------------- generator (combo) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_ARROW_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [2, 3]
_SEED_MAX_COLORS = 4
_MIN_RAY_TILES = 3
_PRE_EXT_IDEAL_TILES = 7
_PLACEMENT_TRIES = 220
_GEN_RETRIES = 600


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


def _choose_launch_cell(mask: np.ndarray, attach_side: str) -> tuple[int, int]:
    cells = _edge_cells(mask, attach_side)
    if not cells:
        raise ValueError(f"empty edge {attach_side}")
    return cells[len(cells) // 2]


def _seed_origin_from_launch(
    launch_abs: tuple[int, int],
    tile_shape: tuple[int, int],
    attach_side: str,
) -> tuple[int, int]:
    lr, lc = map(int, launch_abs)
    th, tw = map(int, tile_shape)
    if attach_side == "U":
        return lr - th, lc
    if attach_side == "D":
        return lr + 1, lc
    if attach_side == "L":
        return lr, lc - tw
    if attach_side == "R":
        return lr, lc + 1
    raise ValueError(attach_side)


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
                rr = int(seed_r0 + i)
                if not (0 <= rr < H):
                    continue
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((rr, int(cc), int(tile[i, j])))
            c += tw
    elif (dr, dc) == (0, -1):
        c = int(seed_c0)
        while c + tw > 0:
            for i in range(th):
                rr = int(seed_r0 + i)
                if not (0 <= rr < H):
                    continue
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((rr, int(cc), int(tile[i, j])))
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


def _component_cells(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    coords = np.argwhere(mask != 0)
    return [(r0 + int(r), c0 + int(c)) for r, c in coords]


def _mark_guard(guard: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = guard.shape
    for r, c in cells:
        guard[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                guard[rr, cc] = True


def _cells_clear_for_arrow(
    cells: list[tuple[int, int]],
    occ: np.ndarray,
    arrow_guard: np.ndarray,
    ray_guard: np.ndarray,
) -> bool:
    for r, c in cells:
        if int(occ[r, c]) != 0 or bool(arrow_guard[r, c]) or bool(ray_guard[r, c]):
            return False
    return True


def _cells_clear_for_ray(
    cells: list[tuple[int, int, int]],
    occ: np.ndarray,
    arrow_guard: np.ndarray,
    ray_guard: np.ndarray,
    own_arrow: set[tuple[int, int]],
) -> bool:
    for r, c, _v in cells:
        if (int(r), int(c)) in own_arrow:
            return False
        if int(occ[r, c]) != 0:
            return False
        if bool(arrow_guard[r, c]) or bool(ray_guard[r, c]):
            return False
    return True


def _place_arrow_and_ray(
    rng: np.random.Generator,
    base_mask: np.ndarray,
    direction_name: str,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    arrow_guard: np.ndarray,
    ray_guard: np.ndarray,
    out: np.ndarray,
    arrow_color: int,
) -> dict | None:
    patch = _rotate_patch(base_mask, direction_name)
    direction = tuple(map(int, _DIR_VECS[direction_name]))
    attach_sides = list(_PERP_ATTACH_SIDES[direction_name])
    ph, pw = patch.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(0, H - ph + 1))
        c0 = int(rng.integers(0, W - pw + 1))
        arrow_cells = _component_cells(patch, (r0, c0))
        if not _cells_clear_for_arrow(arrow_cells, occ, arrow_guard, ray_guard):
            continue

        rng.shuffle(attach_sides)
        for attach_side in attach_sides:
            tile = _build_seed_tile(direction, palette, rng)
            launch_local = _choose_launch_cell(patch, attach_side)
            launch_abs = (r0 + int(launch_local[0]), c0 + int(launch_local[1]))
            seed_r0, seed_c0 = _seed_origin_from_launch(launch_abs, tile.shape, attach_side)
            th, tw = tile.shape
            if not _in_bounds_bbox(H, W, seed_r0, seed_c0, th, tw):
                continue

            ray_tiles = _count_ray_tiles(H, W, seed_r0, seed_c0, direction, tile)
            if ray_tiles < _MIN_RAY_TILES:
                continue

            ray_cells = _project_tile_coords(H, W, seed_r0, seed_c0, direction, tile)
            own_arrow = set((int(r), int(c)) for r, c in arrow_cells)
            if not _cells_clear_for_ray(ray_cells, occ, arrow_guard, ray_guard, own_arrow):
                continue

            for rr, cc in arrow_cells:
                occ[rr, cc] = 1
                out[rr, cc] = int(arrow_color)
            _mark_guard(arrow_guard, arrow_cells)

            for rr, cc, vv in ray_cells:
                occ[rr, cc] = 2
                out[rr, cc] = int(vv)
            _mark_guard(ray_guard, [(int(r), int(c)) for r, c, _v in ray_cells])

            return {
                "tile": tile,
                "cells": ray_cells,
                "seed_r0": int(seed_r0),
                "seed_c0": int(seed_c0),
                "tiles": int(ray_tiles),
                "direction": direction,
                "attach_side": attach_side,
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
    if not (2 <= int(n) <= len(_DIR_ORDER)):
        raise ValueError(f"unsupported arrow count: {n}")

    horizontal = ["L", "R"]
    vertical = ["U", "D"]
    while True:
        chosen = set()
        chosen.add(str(rng.choice(horizontal)))
        chosen.add(str(rng.choice(vertical)))
        remaining = [d for d in _DIR_ORDER if d not in chosen]
        need = int(n) - len(chosen)
        if need > 0:
            extra = list(rng.choice(remaining, size=need, replace=False))
            chosen.update(map(str, extra))
        if len(chosen) == int(n):
            ordered = [d for d in _DIR_ORDER if d in chosen]
            rng.shuffle(ordered)
            return ordered


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    arrow_color = int(rng.choice(others))
    palette = [c for c in others if c != arrow_color]

    base_mask = _ARROW_MASKS[int(rng.integers(0, len(_ARROW_MASKS)))]
    n = int(rng.integers(*_N_ARROW_RANGE))
    direction_names = _sample_direction_names(rng, n)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    arrow_guard = np.zeros((H, W), dtype=bool)
    ray_guard = np.zeros((H, W), dtype=bool)

    arrows: list[dict] = []
    for direction_name in direction_names:
        meta = _place_arrow_and_ray(
            rng,
            base_mask,
            direction_name,
            palette,
            H,
            W,
            occ,
            arrow_guard,
            ray_guard,
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
