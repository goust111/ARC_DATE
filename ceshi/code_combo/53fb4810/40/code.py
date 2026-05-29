# Combo variant of 53fb4810 (index 40)
# combo_spec: D5/0+D6/1+D15/0
# Name: 大瓦片距离编码多色精灵

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant D5/0 + D6/1 + D15/0
# a) Concrete behavior and trace:
#    - Base task: each plus anchor has one attached seed tile, and the output repeats that
#      tile away from the plus. One ray is already expanded in the input as the clue.
#    - D5 changes the seed from tiny strips into large filled rectangles such as 3x2, 2x3,
#      3x3, 4x2, or 4x3. The repeated period is now a visibly rich block.
#    - D6 changes the endpoint rule: a ray no longer runs to the border. Instead its total
#      number of tile-units equals the number of distinct colors inside the seed tile.
#      Example: a 3x2 seed using colors {2, 4, 7} produces exactly 3 copies of that 3x2
#      block, seed included, then stops and leaves background before the boundary.
#    - D15 changes the anchor colors: every plus uses a different color, and marker/ray
#      colors avoid all plus colors. The solver therefore cannot detect one global plus
#      color first; it must find each large plus-shaped monochrome component separately.
#    - Combined trace: suppose a teal 5x4 plus has a 3x2 seed below it using colors
#      {1, 3, 6}. The input may show that full 3-unit downward ray as the clue, while a
#      yellow plus elsewhere shows only its compact 4x3 seed with colors {2, 5}. The
#      solver must treat the teal clue as already complete, but extend the yellow seed to
#      exactly 2 units rather than to the border.
#
# b) Complexity calibration:
#    - Large seeds are forced to have area at least 6, at least 2 colors, and internal
#      row/column variation, so the D5 change is clearly visible.
#    - The distance code uses only 2-4 units, which is enough to vary lengths while still
#      being easy to infer from a few examples.
#    - Each grid contains 2-3 pluses with unique colors, which makes the D15 change
#      visible without creating too many interacting objects.
#    - All rays are generated to stop before the border, so the D6 change is always
#      observable and never collapses back to the original "extend to edge" behavior.
#
# c) Implicit-assumption audit of the original solver:
#    - "One dominant plus color" is invalid because pluses are now multicolor globally.
#      The solver must detect plus components by local geometry per monochrome component.
#    - "Any non-bg/non-plus-color cell is marker" is invalid because there are many plus
#      colors. The solver instead removes the union of detected plus components.
#    - "A touching component can always be extended again" is invalid because the input
#      clue is already a finished ray with a deliberate early stop. The solver therefore
#      only extends compact touching rectangles whose along-ray span is at most 4.
#    - "Stopping at the boundary is harmless" is invalid. Extension must use the encoded
#      exact number of tile-units, not boundary fill.
#    - "Protect one plus color" is invalid. All detected plus colors must be protected.
#
# d) Placement legality:
#    - Generation is still output-first. Each plus and its exact-length ray are placed only
#      if the whole structure fits in-bounds, stays disjoint from earlier rays, and avoids
#      other pluses plus a small buffer around them.
#    - The compact seed always matches the nozzle span exactly, and the kept clue ray is
#      guaranteed to have along-axis span > 4 so the solver can distinguish it from a seed.
#
# e) Visual observability:
#    - A human can see three simultaneous cues: large textured seed blocks, different plus
#      colors behaving the same way, and rays that stop after a color-count-dependent
#      number of repeats rather than at the border.
#    - Because at least two encoded lengths appear in each grid, the stopping rule is not a
#      constant and can be compared across examples.
#
# f) Visual cleanliness:
#    - Rays do not overlap each other or any plus, and they are kept off the border by at
#      least one cell of background. This leaves clear empty space around the exact stop.
#    - Marker colors are disjoint from plus colors, so anchors and payloads stay visually
#      separable despite the richer textures.
#
# g) New solver assumptions and generator guarantees:
#    - Solver assumption: pluses are monochrome, horizontally and vertically symmetric,
#      non-rectangular components with bbox at least 3x3. Generator guarantee: pluses are
#      chosen from fixed symmetric templates satisfying that geometry.
#    - Solver assumption: compact seeds are filled rectangles with along-axis span <= 4.
#      Generator guarantee: every seed uses along size in {2, 3, 4}, and the preserved
#      clue ray always has along-axis span > 4.
#    - Solver assumption: seed colors encode distance via distinct-color count. Generator
#      guarantee: each seed uses exactly 2-4 distinct colors, and output rays contain that
#      exact number of tile-units.


_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_SOLVER_MAX_COMPACT_ALONG = 4


def _is_contiguous_line(bits: np.ndarray) -> bool:
    idx = np.flatnonzero(bits)
    if idx.size == 0:
        return False
    return int(idx[-1] - idx[0] + 1) == int(idx.size)


def _looks_like_plus_component(mask: np.ndarray) -> bool:
    """Recognize the large plus templates used in this variant by geometry, not color id."""
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if bool(np.all(mask)):
        return False
    if not np.array_equal(mask, mask[::-1, :]):
        return False
    if not np.array_equal(mask, mask[:, ::-1]):
        return False

    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)
    if np.any(row_sums == 0) or np.any(col_sums == 0):
        return False
    if int(row_sums[0]) >= w or int(col_sums[0]) >= h:
        return False
    if int(row_sums.max()) <= int(row_sums[0]) or int(col_sums.max()) <= int(col_sums[0]):
        return False
    if not np.any(row_sums == w):
        return False
    if not np.any(col_sums == h):
        return False

    for i in range(h):
        if not _is_contiguous_line(mask[i, :]):
            return False
    for j in range(w):
        if not _is_contiguous_line(mask[:, j]):
            return False
    return True


def detect_big_plus_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    """
    Collect all large monochrome components that match the plus geometry used here.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps: list[dict] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            crop = crop_bbox(grid == int(col), (int(r0), int(c0), int(r1), int(c1)))
            if not _looks_like_plus_component(crop):
                continue
            comps.append(
                {
                    "color": int(col),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def encoded_ray_units(tile: np.ndarray, *, bg: int) -> int:
    """Distance code: total tile-units equals the number of distinct non-background colors."""
    cols = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    return max(1, int(len(cols)))


def extend_repeating_tile_exact(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    total_units: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a seed by an exact number of whole-tile copies."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
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

    for unit_idx in range(1, int(total_units)):
        base_r = int(r0 + unit_idx * dr * h)
        base_c = int(c0 + unit_idx * dc * w)
        for i in range(h):
            for j in range(w):
                put(base_r + i, base_c + j, int(tile[i, j]))


def _seed_matches_nozzle(
    plus_mask: np.ndarray,
    plus_bbox: tuple[int, int, int, int],
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """Check that the seed cross-axis span matches the plus arm-tip exactly."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    r0, c0, r1, c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_row = pr1 if dr == 1 else pr0
        nozzle_cols = sorted(int(cc) for cc in range(pc0, pc1 + 1) if plus_mask[nozzle_row, cc])
        seed_cols = list(range(c0, c1 + 1))
        return seed_cols == nozzle_cols

    nozzle_col = pc1 if dc == 1 else pc0
    nozzle_rows = sorted(int(rr) for rr in range(pr0, pr1 + 1) if plus_mask[rr, nozzle_col])
    seed_rows = list(range(r0, r1 + 1))
    return seed_rows == nozzle_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each compact marker rectangle from its colored '+' anchor by a seed-encoded distance.

    Input structure:
    - The background is the most common color.
    - Several large plus-like sprites appear, each in its own non-background color.
    - Each plus has one adjacent large rectangular marker tile; exactly one plus keeps its
      full repeated ray in the input as a completed example, while the others show only the
      compact seed rectangles.

    Rule:
    - Detect every large plus-shaped monochrome component as an anchor and remove all plus
      cells from marker detection.
    - For each touching compact rectangle, infer the outward direction from the arm it
      touches, count the tile's distinct colors, and repeat that whole rectangle away from
      the plus until that many tile-units are present.
    - Do not repaint any plus cell, and leave the already-complete long clue ray unchanged.

    Key insight:
    - Three original shortcuts fail at once: there is no shared plus color, the seed period
      is a large textured block rather than a tiny strip, and ray length is encoded by the
      seed's color diversity instead of the border.
    - The clue ray can be recognized because its attached component is longer than any legal
      compact seed, so only short attached rectangles should be expanded.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    pluses = detect_big_plus_components(g, bg=int(bg), connectivity=4)
    if not pluses:
        return out

    all_plus_colors = {int(comp["color"]) for comp in pluses}
    plus_union_mask = np.zeros_like(g, dtype=bool)
    for comp in pluses:
        coords = comp["coords"]
        plus_union_mask[coords[:, 0], coords[:, 1]] = True

    mark_lab, _ = label_mask((g != int(bg)) & (~plus_union_mask), connectivity=4)

    for plus in pluses:
        plus_coords = plus["coords"]
        plus_bbox = plus["bbox"]
        plus_mask = np.zeros_like(g, dtype=bool)
        plus_mask[plus_coords[:, 0], plus_coords[:, 1]] = True

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

            direction = next(iter(dirs))
            if not _seed_matches_nozzle(plus_mask, plus_bbox, (int(r0), int(c0), int(r1), int(c1)), direction):
                continue

            dr, dc = direction
            along_span = int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)

            # Long attached rectangles are the already-finished demonstration ray kept in the input.
            if along_span > _SOLVER_MAX_COMPACT_ALONG:
                continue

            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            total_units = encoded_ray_units(tile, bg=int(bg))
            extend_repeating_tile_exact(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                total_units=int(total_units),
                bg=int(bg),
                protect=set(all_plus_colors),
            )

    return out


# --------------- generator ---------------

_H_RANGE = (22, 35)
_W_RANGE = (20, 30)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [2, 3, 4]
_SEED_ALLOWED_CROSS = {2, 3}
_SEED_MIN_COLORS = 2
_SEED_MAX_COLORS = 4
_SEED_MIN_AREA = 6
_PLACEMENT_TRIES = 240
_GEN_RETRIES = 700
_MIN_BOUNDARY_GAP = 1
_PRECLUE_MIN_ALONG_SPAN = 5

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
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive span of the chosen nozzle along the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])

    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    """Reject degenerate tiles that visually collapse into a uniform stripe or block."""
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
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int] | None:
    """Build a large rectangular seed together with its encoded unit count."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross not in _SEED_ALLOWED_CROSS:
        return None

    for _ in range(30):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        if cross == 2 and along == 2:
            along = 3

        _dr, dc = direction
        th, tw = (along, cross) if dc == 0 else (cross, along)
        if th * tw < _SEED_MIN_AREA:
            continue

        max_k = min(_SEED_MAX_COLORS, th * tw, len(palette))
        if max_k < _SEED_MIN_COLORS:
            return None
        units_total = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(palette, size=units_total, replace=False)))
        vals = chosen.copy()
        if th * tw > units_total:
            vals.extend(list(map(int, rng.choice(chosen, size=th * tw - units_total, replace=True))))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile):
            return tile, int(units_total)

    return None


def _project_exact_ray(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    units_total: int,
) -> tuple[list[tuple[int, int, int]], int, int, int, int] | None:
    """Return all cells of the exact-length ray and metadata for clue selection."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - th)
        first_r0 = int(r0)
        last_r0 = int(r0 + dr * th * (units_total - 1))
        lo_r = min(first_r0, last_r0)
        hi_r = max(first_r0 + th - 1, last_r0 + th - 1)
        if lo_r < 0 or hi_r >= H or c0 < 0 or c0 + tw > W:
            return None
        along_span = int(th * units_total)
        boundary_gap = int(H - 1 - hi_r) if dr == 1 else int(lo_r)
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tw)
        first_c0 = int(c0)
        last_c0 = int(c0 + dc * tw * (units_total - 1))
        lo_c = min(first_c0, last_c0)
        hi_c = max(first_c0 + tw - 1, last_c0 + tw - 1)
        if lo_c < 0 or hi_c >= W or r0 < 0 or r0 + th > H:
            return None
        along_span = int(tw * units_total)
        boundary_gap = int(W - 1 - hi_c) if dc == 1 else int(lo_c)

    if boundary_gap < _MIN_BOUNDARY_GAP:
        return None

    cells: list[tuple[int, int, int]] = []
    for unit_idx in range(int(units_total)):
        base_r = int(r0 + unit_idx * dr * th)
        base_c = int(c0 + unit_idx * dc * tw)
        for i in range(th):
            for j in range(tw):
                cells.append((base_r + i, base_c + j, int(tile[i, j])))

    return cells, int(r0), int(c0), int(along_span), int(boundary_gap)


def _plus_touches_existing_ray(plus_cells: list[tuple[int, int]], occ: np.ndarray) -> bool:
    """Keep new pluses away from already-placed rays for cleaner scenes."""
    H, W = occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) == 2:
                return True
    return False


def _ray_touches_existing_ray(cells: list[tuple[int, int, int]], occ: np.ndarray) -> bool:
    """Reject candidate rays that would become 4-connected to an older ray."""
    H, W = occ.shape
    for r, c, _ in cells:
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
    plus_buffer: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    direction_order: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Place one uniquely colored plus together with its exact-length output ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) != 0
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(bool(plus_buffer[r, c]) for r, c in plus_cells):
            continue
        if _plus_touches_existing_ray(plus_cells, occ):
            continue

        dirs: list[tuple[int, int]] = []
        if direction_order is not None:
            for direction in direction_order:
                direction = tuple(map(int, direction))
                if direction not in dirs:
                    dirs.append(direction)
        remaining_dirs = [direction for direction in _DIRECTIONS if direction not in dirs]
        rng.shuffle(remaining_dirs)
        dirs.extend(remaining_dirs)
        for direction in dirs:
            built = _build_seed_tile(plus_shape, direction, palette, rng)
            if built is None:
                continue
            tile, units_total = built
            projected = _project_exact_ray(
                H,
                W,
                (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                plus_shape,
                direction,
                tile,
                units_total,
            )
            if projected is None:
                continue
            cells, sr0, sc0, along_span, boundary_gap = projected
            if any(int(occ[r, c]) != 0 for r, c, _ in cells):
                continue
            if any(bool(plus_buffer[r, c]) for r, c, _ in cells):
                continue
            if _ray_touches_existing_ray(cells, occ):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = r + dr2
                    cc = c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_buffer[rr, cc] = True
                plus_buffer[r, c] = True
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "plus_color": int(plus_color),
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units_total": int(units_total),
                "along_span": int(along_span),
                "boundary_gap": int(boundary_gap),
                "direction": tuple(map(int, direction)),
            }

    return None


def _pick_direction_targets(rng: np.random.Generator, n_plus: int) -> list[tuple[int, int]]:
    """Choose sample-level direction targets with both axes represented."""
    verticals = [(1, 0), (-1, 0)]
    horizontals = [(0, 1), (0, -1)]
    rng.shuffle(verticals)
    rng.shuffle(horizontals)

    targets = [verticals[0], horizontals[0]]
    remaining = [direction for direction in _DIRECTIONS if direction not in targets]
    rng.shuffle(remaining)
    while len(targets) < int(n_plus):
        targets.append(remaining[len(targets) - 2])
    rng.shuffle(targets)
    return [tuple(map(int, direction)) for direction in targets]


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Keep one finished long clue ray and collapse all other rays back to their compact seed."""
    candidates = [
        i
        for i, p in enumerate(pluses)
        if int(p["along_span"]) > _SOLVER_MAX_COMPACT_ALONG
        and int(p["along_span"]) >= _PRECLUE_MIN_ALONG_SPAN
        and int(p["boundary_gap"]) >= _MIN_BOUNDARY_GAP
    ]
    if not candidates:
        return None

    pre_idx = max(
        candidates,
        key=lambda i: (int(pluses[i]["units_total"]), int(pluses[i]["along_span"])),
    )
    if len(candidates) > 1 and bool(rng.integers(0, 2)):
        pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    plus_colors = list(map(int, rng.choice(others, size=n_plus, replace=False)))
    marker_palette = [c for c in others if c not in plus_colors]
    if len(marker_palette) < _SEED_MIN_COLORS:
        return None
    direction_targets = _pick_direction_targets(rng, n_plus)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_buffer = np.zeros((H, W), dtype=bool)

    pluses: list[dict] = []
    for plus_color, target_direction in zip(plus_colors, direction_targets):
        info = _place_plus_and_ray(
            rng=rng,
            plus_shape=plus_shape,
            palette=marker_palette,
            H=H,
            W=W,
            occ=occ,
            plus_buffer=plus_buffer,
            out=out,
            plus_color=int(plus_color),
            direction_order=[target_direction],
        )
        if info is None:
            return None
        pluses.append(info)

    if len({int(p["units_total"]) for p in pluses}) < 2:
        return None
    if not any(int(p["direction"][0]) != 0 for p in pluses):
        return None
    if not any(int(p["direction"][1]) != 0 for p in pluses):
        return None

    inp = _build_input_from_output(out, pluses, bg, rng)
    if inp is None:
        return None

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
