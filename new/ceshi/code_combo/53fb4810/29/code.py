# Combo variant of 53fb4810 (index 29)
# combo_spec: D5/0+D13/0
# Name: 大瓦片居中对齐

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D5/0 + D13/0 combo
# a) Concrete behavior:
#    - Original task: each '+' has one attached seed rectangle, but that seed must match
#      the nozzle span exactly. Example: for a 3x4 plus with a top marker, the original
#      seed above the plus would cover only the two middle columns touched by the top arm.
#    - This combo changes two linked parts at once. D5 makes the seed a large textured
#      rectangle. D13 moves the alignment rule from "match the arm tip" to "center on
#      the whole plus sprite". In a concrete 3x4 example, a 3x4 seed above the plus now
#      covers all four columns of the plus bbox, even though only the two middle columns
#      touch the top arm directly. The ray width is therefore determined by the tile, not
#      by the nozzle.
#    - Input keeps one fully expanded example ray. The other pluses show only their large
#      centered seeds. Output repeats every seed rectangle outward to the boundary.
# b) Complexity calibration:
#    - To keep the fused change visible, the generator forces the seed cross-axis span to
#      equal the whole plus bbox cross-axis span, which is strictly larger than the nozzle
#      span for the chosen shapes. This prevents degeneration back to the original rule.
#    - Along-axis size is limited to {2, 3, 4}; seeds use 2-4 colors and must have row
#      and column variation. That keeps patterns readable while making them visibly richer
#      than the old 1x2 / 2x1 / 2x2 clues.
# c) Implicit assumptions audit:
#    - `label_mask(non_mask, connectivity=4)` still assumes every marker/ray is one
#      4-connected component, so the generator uses fully filled rectangles with no holes.
#    - `crop_bbox(...)` assumes the repeating period is the component bbox. For compact
#      seeds this bbox is the seed itself; for the pre-expanded example ray it becomes the
#      whole already-repeated strip, and extending from that bbox is still safe because it
#      already reaches the boundary.
#    - The original solver's hidden assumption "seed cross-axis == nozzle span" is no
#      longer valid. The replacement invariant is "seed bbox center matches plus bbox
#      center on the cross-axis". The solver must check centering, not nozzle equality.
#    - Direction inference from 4-neighbor contact still works because each seed is placed
#      flush against exactly one side of exactly one plus.
# d) Placement legality:
#    - The generator places full output rays first, rejects any overlap with existing rays
#      or protected cells around other pluses, and only then erases non-example rays back
#      to seeds. This guarantees all compact seeds and expanded rays are in-bounds and
#      non-colliding.
# e) Visual observability:
#    - A human can see the fusion because hidden seeds are large multicolor rectangles and
#      their width/height clearly matches the whole plus bbox rather than just the arm tip.
#      The preserved long example ray shows the exact periodic continuation rule.
# f) Visual cleanliness:
#    - Rays never overlap each other or any plus. Pluses are also kept away from existing
#      rays, so each object remains visually separable despite the thicker bands.
# g) Color inference:
#    - No rule depends on numeric color ids. Colors are only texture content inside the
#      repeated rectangles; all reasoning is geometric.


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


def _seed_is_center_aligned(
    plus_coords: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """Check the centered-alignment rule on the seed/ray cross-axis."""
    r0, c0, r1, c1 = map(int, seed_bbox)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    pr0 = int(pr0)
    pc0 = int(pc0)
    pr1 = int(pr1)
    pc1 = int(pc1)
    dr, dc = map(int, direction)

    if dc == 0:
        return int(c0 + c1) == int(pc0 + pc1)
    if dr == 0:
        return int(r0 + r1) == int(pr0 + pr1)
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each large centered marker rectangle away from its '+' anchor to the edge.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several plus-like anchor sprites.
    - Each plus has one adjacent multicolor rectangular marker, and in this combo the
      marker can be large while being centered on the full plus bbox rather than on the
      narrow arm tip. One marker ray may already be fully pre-expanded as an example.

    Rule:
    - Find every non-plus component that touches a plus and infer which side of the plus
      it is attached to.
    - Keep only centered attachments: for vertical rays the marker/ray bbox must be
      centered in columns with the plus bbox; for horizontal rays it must be centered in
      rows with the plus bbox.
    - Repeat that entire rectangular pattern outward in the same direction until the
      boundary, without overwriting plus pixels.

    Key insight:
    - The side of contact still determines the ray direction, but the repeated band is no
      longer nozzle-width. The whole large rectangle, centered on the plus as a whole, is
      the period that propagates.
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
            if not _seed_is_center_aligned(plus_coords, (int(r0), int(c0), int(r1), int(c1)), direction):
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

_H_RANGE = (19, 33)
_W_RANGE = (18, 30)
_N_PLUS_RANGE = (2, 4)  # 2 or 3
_SEED_ALONG_CHOICES = [2, 3, 4]
_SEED_MIN_COLORS = 2
_SEED_MAX_COLORS = 4
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 4
_PLACEMENT_TRIES = 220
_GEN_RETRIES = 800

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


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive nozzle span along the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])

    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    """Reject visually trivial large rectangles."""
    if len(set(map(int, tile.flatten()))) < _SEED_MIN_COLORS:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    return True


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a large seed centered on the whole plus bbox, not on the nozzle."""
    dr, dc = direction
    ph, pw = plus_mask.shape
    cross = pw if dc == 0 else ph
    s, e = _nozzle_span(plus_mask, direction)
    nozzle = int(e - s + 1)
    if cross <= nozzle:
        return None

    for _ in range(24):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        th, tw = (along, cross) if dc == 0 else (cross, along)
        n_cells = int(th * tw)
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
        if _tile_has_rich_texture(tile):
            return tile

    return None


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int, int]:
    """Return all cells of the projected ray, the seed origin, and number of tile units."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape

    if dc == 0:
        if (pc0 + pc1 - tw + 1) % 2 != 0:
            return [], -1, -1, 0
        c0 = (pc0 + pc1 - tw + 1) // 2
        r0 = pr1 + 1 if dr == 1 else pr0 - th
    else:
        if (pr0 + pr1 - th + 1) % 2 != 0:
            return [], -1, -1, 0
        r0 = (pr0 + pr1 - th + 1) // 2
        c0 = pc1 + 1 if dc == 1 else pc0 - tw

    if r0 < 0 or c0 < 0 or r0 + th > H or c0 + tw > W:
        return [], int(r0), int(c0), 0

    cells: list[tuple[int, int, int]] = []
    units = 0
    if dc == 0:
        r = r0
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            units += 1
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
            units += 1
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(tile[i, j])))
            c += step

    return cells, int(r0), int(c0), int(units)


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
    """Place one plus with a centered large ray."""
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

            cells, sr0, sc0, units = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                direction,
                tile,
            )
            if units < _MIN_RAY_UNITS:
                continue
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
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(units),
                "direction": tuple(map(int, direction)),
                "plus_bbox": (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one long example ray and collapse all others back to their seed rectangles."""
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
