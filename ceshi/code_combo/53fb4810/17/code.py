# Combo variant of 53fb4810 (index 17)
# combo_spec: D2/0+D5/0+D9/0
# Name: 形状编码方向大瓦片无线索

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Original task: a compact seed touches one side of a symmetric plus, and that touching side
#      directly tells you which way to extend the repeated ray. One ray is already shown in full.
#    - New fused task: every input keeps only compact seeds, no ray is pre-expanded, and the true
#      ray direction comes from a one-sided protrusion on the plus-like sprite rather than from the
#      seed attachment side. Seeds are also much larger rectangles, such as 3x2, 4x2, 2x3, or 3x3.
#    - Example: suppose a sprite visually protrudes to the right, but its 3x2 seed is attached on
#      the top edge. The original rule would have extended that seed upward because it sits above
#      the plus. The fused rule instead repeats the full 3x2 tile to the right until the boundary.
#      With all rays collapsed in the input, the training pairs must teach both "protrusion means
#      direction" and "repeat the whole large rectangle", not just recover a hidden example ray.
# b) Complexity calibration:
#    - Direction is still limited to the four cardinal directions, and all pluses in one grid come
#      from one base asymmetric family rotated into those directions. This keeps the shape cue
#      learnable rather than exploding into many unrelated silhouettes.
#    - Large seeds are constrained to filled rectangles whose attachment-edge span is only 2 or 3
#      cells, while the free dimension is chosen from {2, 3, 4}. This makes the visual change from
#      the original tiny markers obvious without producing unwieldy blocks.
#    - Attachment is always perpendicular to the encoded direction. That makes the D2 change
#      behaviorally active on every placed object instead of silently matching the original rule.
# c) Implicit-assumption audit of `main()`:
#    - `detect_dominant_big_component_color(...)` still assumes the plus color owns the largest
#      repeated components. The generator keeps pluses large and collapses every ray tail in the
#      input, so seeds never outgrow them as connected components.
#    - `mark_lab = label_mask(non_mask, connectivity=4)` assumes each seed is one connected
#      component. The generator uses filled rectangles only.
#    - `crop_bbox(...)` assumes the bbox of the seed is exactly the tile period. The generator
#      keeps seeds rectangular with no holes, so bbox == tile is valid.
#    - The original solver inferred direction from seed/plus contact and asserted alignment to the
#      emitting arm. Those assumptions are invalid here because seeds attach on perpendicular sides.
#      The new solver removes those checks and reads direction only from the asymmetric plus crop.
#    - The new solver assumes each plus crop matches one of the rotated templates in the lookup and
#      that each plus has exactly one adjacent non-plus component. The generator enforces both.
# d) Placement legality:
#    - Output is built first. Each plus is placed with a 4-neighbor halo, each seed is anchored
#      flush to one perpendicular side, and the full repeated ray is generated before committing.
#    - Placements are rejected if any plus/seed/ray cell would overlap an existing object, touch a
#      protected plus halo, or leave the grid. This keeps adjacency unambiguous for the solver.
# e) Visual observability:
#    - The shape cue is human-visible because one side of every plus family sticks out by one cell.
#      Since seeds never sit on the emitting side, attachment cannot explain the output direction.
#    - Large multicolor tiles create an obvious repeating period in the output, so solvers can see
#      that the whole rectangle, not just a stripe, is copied along the ray.
#    - Removing the pre-expanded clue makes the train pairs carry the whole rule globally, which is
#      exactly the intended D9 inversion.
# f) Visual cleanliness:
#    - Pluses, seeds, and rays are kept disjoint, and pluses are buffered from other rays by a halo.
#      This avoids hard-to-read overlaps even though the seed textures are richer.
#    - All objects in one grid share the same asymmetric family up to rotation, giving repeated
#      support for the "protrusion encodes direction" cue.
# g) Color and rule inferability:
#    - Colors only fill the seed texture. No ordering, ranking, or branch choice depends on color
#      ids, so the rule remains geometric and visible.
#    - The generator also rejects weak seed textures: every seed must have at least two colors and
#      row/column variation, preventing accidental regression to a simple uniform bar.


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


_BASE_PLUS_RIGHT = [
    np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0],
        ],
        dtype=int,
    ),
]

_ROT_DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0)]
_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
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
    Extend each large seed tile in the direction encoded by an asymmetric plus sprite.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several almost-plus sprites with a single protruding side.
    - Each sprite touches exactly one compact rectangular seed, and there is no already-expanded ray anywhere in the input.

    Rule:
    - Read the ray direction from the side where the plus sprite sticks out farther.
    - Ignore the seed's attachment side; use the seed component's full bounding box as the repeating tile.
    - Repeat that large tile periodically from its current position in the shape-indicated direction until the boundary,
      without overwriting plus-colored cells.

    Key insight:
    - The visible asymmetry of the plus, not local attachment geometry, determines the ray direction, and the entire
      multicolor rectangle is the period that must be copied.
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
            protect={int(plus_col)},
        )

    return out


# --------------- generator ---------------

_H_RANGE = (18, 31)
_W_RANGE = (18, 29)
_N_PLUS_RANGE = (3, 5)
_SEED_FREE_DIM_CHOICES = [2, 3, 4]
_SEED_ALLOWED_CROSS = {2, 3}
_SEED_MAX_COLORS = 4
_SEED_MIN_COLORS = 2
_SEED_MIN_AREA = 6
_MIN_RAY_UNITS = 2
_PLACEMENT_TRIES = 180
_GEN_RETRIES = 500


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    if tile.size < _SEED_MIN_AREA:
        return False
    if len(set(map(int, tile.flatten()))) < _SEED_MIN_COLORS:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    return True


def _build_seed_tile(
    plus_shape: np.ndarray,
    attach_side: str,
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    s, e = _edge_span(plus_shape, attach_side)
    cross = int(e - s + 1)
    if cross not in _SEED_ALLOWED_CROSS:
        return None

    for _ in range(20):
        free_dim = int(rng.choice(_SEED_FREE_DIM_CHOICES))
        if attach_side in {"U", "D"}:
            th, tw = free_dim, cross
        else:
            th, tw = cross, free_dim

        n_cells = th * tw
        if n_cells < _SEED_MIN_AREA:
            continue

        max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
        if max_k < _SEED_MIN_COLORS:
            return None
        n_colors = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile):
            return tile

    return None


def _seed_anchor(
    plus_bbox: tuple[int, int, int, int],
    plus_shape: np.ndarray,
    attach_side: str,
    tile: np.ndarray,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    th, tw = tile.shape
    s, _ = _edge_span(plus_shape, attach_side)

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


def _plus_touches_existing_ray(
    plus_cells: list[tuple[int, int]],
    occ: np.ndarray,
) -> bool:
    H, W = occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) == 2:
                return True
    return False


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

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue
        if _plus_touches_existing_ray(plus_cells, occ):
            continue

        attach_sides = list(_PERP_ATTACH_SIDES[direction])
        rng.shuffle(attach_sides)
        for attach_side in attach_sides:
            tile = _build_seed_tile(plus_shape, attach_side, palette, rng)
            if tile is None:
                continue

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
            for r, c, _ in cells:
                if (r, c) in plus_set or int(occ[r, c]) != 0 or int(plus_occ[r, c]) == 1:
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
            }

    return None


def _build_input_from_output(out: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    inp = out.copy()
    for p in pluses:
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _has_direction_attachment_evidence(pluses: list[dict]) -> bool:
    directions = {tuple(map(int, p["direction"])) for p in pluses}
    if len(directions) < 2:
        return False

    has_vertical = any(int(dr) != 0 for dr, _ in directions)
    has_horizontal = any(int(dc) != 0 for _, dc in directions)
    if not (has_vertical and has_horizontal):
        return False

    attach_by_direction: dict[tuple[int, int], set[str]] = {}
    direction_by_attach: dict[str, set[tuple[int, int]]] = {}
    for p in pluses:
        direction = tuple(map(int, p["direction"]))
        attach_side = str(p["attach_side"])
        attach_by_direction.setdefault(direction, set()).add(attach_side)
        direction_by_attach.setdefault(attach_side, set()).add(direction)

    # At least one local contradiction must be visible so the seed's touching side cannot
    # explain the ray direction on its own.
    if any(len(sides) >= 2 for sides in attach_by_direction.values()):
        return True
    if any(len(dirs) >= 2 for dirs in direction_by_attach.values()):
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    base = _BASE_PLUS_RIGHT[int(rng.integers(0, len(_BASE_PLUS_RIGHT)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(n_plus):
        k = int(rng.integers(0, 4))
        plus_shape = np.rot90(base, k=k)
        direction = _ROT_DIRS[k]
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            direction,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    if not _has_direction_attachment_evidence(pluses):
        return None

    inp = _build_input_from_output(out, pluses, bg)

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
