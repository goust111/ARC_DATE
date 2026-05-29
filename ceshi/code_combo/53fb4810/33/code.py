# Combo variant of 53fb4810 (index 33)
# combo_spec: D10/0+D11/0+D12/0
# Name: 异形多射线碰撞

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and behavior trace
# - Original task: each same-shaped '+' sprite owns one compact seed tile, one ray in the grid
#   is already shown as a long periodic exemplar, and all rays extend independently.
# - Combo task: pluses still share one color, but now different connected components may use
#   different silhouettes, each plus owns 2-4 seeds on different arms, and the rays are no
#   longer independent: later rays stop before the first tile-copy that would hit an already
#   existing ray.
# - Small trace:
#   1. A thin 3x3 plus has an up seed [[2], [4]] and a right seed [6, 7].
#   2. A fat 3x4 plus lower in the grid has a left seed [[3, 5]].
#   3. Rays are processed in row-major order of seed origins. If the up seed is first, it grows
#      upward fully. If the left seed is later and its next copy would enter one of those cells,
#      it stops at the previous copy. The right seed of the thin plus is unaffected if it never
#      reaches an occupied ray cell.
# - Difference from the source task: the solver must recover a local nozzle span per plus shape,
#   recover the base tile from either a compact seed or the one pre-expanded stripe, and then
#   replay all rays in a global order instead of extending every component independently.
#
# b) Complexity calibration
# - Shapes are restricted to the same three symmetric templates used in the original family, but
#   each grid is forced to contain at least two distinct silhouettes so D10 is visible.
# - Marker count stays in the human-readable 2-4 range per plus and directions are limited to the
#   four compass arms, so the combinatorics increase without turning the scene into noise.
# - The ray period along the growth axis is still only 1 or 2, which keeps the long exemplar
#   legible. When along-size is 2, the generator forbids accidental period-1 stripes so the base
#   tile can be inferred from the visible repetition.
#
# c) Hidden assumptions audit against the original main()
# - `detect_dominant_big_component_color(...)` still assumes one shared plus color; the combo
#   variant keeps that invariant.
# - The original `mark_lab = label_mask(non_mask, connectivity=4)` only works if each marker/ray
#   stays a separate 4-connected component touching exactly one plus. Collision-stop is therefore
#   implemented as "stop before an overlapping tile-copy", and the generator also forbids foreign
#   4-neighbor touches so components never merge.
# - The original solver could crop the whole adjacent component and reuse it as the seed tile. In
#   the combo task that fails for the one pre-expanded ray, so the new solver extracts the minimal
#   axis period (1 or 2) from the component while preserving the locally inferred cross-axis span.
# - The old code had no order dependence. The new solver introduces one explicit new assumption:
#   all rays are replayed in row-major order of their seed top-left cells. The generator constructs
#   outputs with exactly that order, so order dependence is visible and reproducible.
#
# d) Placement legality
# - Pluses are placed first with a one-cell moat from foreign structure.
# - For each plus, chosen seed tiles must be in-bounds, must not overlap existing pluses or seed
#   tiles, and must not 4-touch a foreign plus or foreign seed component.
# - Output construction starts from "all pluses + all seeds" and then extends rays in row-major
#   seed order. A ray stops before the first tile-copy that would overlap an occupied ray cell.
#   Samples are rejected if any ray would need to cross a foreign plus or seed.
#
# e) Visual observability
# - D10 is visible because same-colored anchors clearly have different silhouettes in one grid.
# - D11 is visible because a single plus visibly has several attached seed tiles in different arms.
# - D12 is visible because at least one generated output contains a ray that ends early against a
#   previously grown stripe instead of reaching the boundary. The input still contains one long
#   exemplar ray so the periodic-copy rule remains recoverable by eye.
# - The chosen order is geometric, not numeric: seed origins are processed top-to-bottom and then
#   left-to-right. No invisible color-id ordering is used.
#
# f) Visual cleanliness
# - The generator keeps ray components disjoint and avoids foreign 4-neighbor contacts, so there
#   is no ambiguity about which seed belongs to which plus.
# - Only one ray remains pre-expanded in the input. All other rays are collapsed to compact seeds,
#   which keeps the input sparse enough despite the increased marker count.
#
# g) Color inferability and regression metric
# - Colors are copied verbatim from each seed tile; there is no remapping. Palette size per tile
#   stays small, so the only new inference burden is geometry and collision order.
# - Post-implementation metric: over random generated samples, require at least two distinct plus
#   silhouettes, total marker count greater than plus count, and at least one collision-stopped ray
#   in almost every sample. If collision count is often zero or silhouette count drops to one, the
#   generator has regressed toward the easier original task.


_DIRECTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
_DIR_RANK = {d: i for i, d in enumerate(_DIRECTIONS)}


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper: choose the color owning the most large connected components.

    Large component = bbox at least (min_h x min_w).
    Tie-break: larger total foreground area.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {bg}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if int(r1 - r0 + 1) >= int(min_h) and int(c1 - c0 + 1) >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _marker_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Return the unique direction from one plus component to one adjacent marker/ray component."""
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


def _edge_span_indices(plus_mask: np.ndarray, direction: tuple[int, int]) -> list[int]:
    """Return the occupied nozzle coordinates along the cross-axis for one plus component."""
    coords = np.argwhere(plus_mask)
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [
            int(cc)
            for cc in range(int(pc0), int(pc1) + 1)
            if bool(plus_mask[nozzle_row, cc])
        ]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [
        int(rr)
        for rr in range(int(pr0), int(pr1) + 1)
        if bool(plus_mask[rr, nozzle_col])
    ]


def _orient_component(tile: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """Orient a full ray component so axis index 0/1 starts at the seed and moves outward."""
    dr, dc = direction
    if dr == -1:
        return tile[::-1, :]
    if dc == -1:
        return tile[:, ::-1]
    return tile


def _axis_period(tile: np.ndarray, direction: tuple[int, int]) -> int | None:
    """
    Infer the minimal along-axis period (1 or 2) from a seed-only or long pre-expanded component.

    The generator guarantees that any true 2-step tile is not accidentally period-1.
    """
    oriented = _orient_component(tile, direction)
    axis_len = int(oriented.shape[0] if direction[1] == 0 else oriented.shape[1])

    for period in (1, 2):
        if period > axis_len:
            continue
        ok = True
        for idx in range(axis_len):
            if direction[1] == 0:
                if not np.array_equal(oriented[idx, :], oriented[idx % period, :]):
                    ok = False
                    break
            else:
                if not np.array_equal(oriented[:, idx], oriented[:, idx % period]):
                    ok = False
                    break
        if ok:
            return int(period)
    return None


def _extract_seed_tile(
    grid: np.ndarray,
    pts: np.ndarray,
    direction: tuple[int, int],
    nozzle_span: list[int],
) -> tuple[np.ndarray, int, int] | None:
    """Recover the compact seed tile from an adjacent component that may already be a long ray."""
    r0, c0, r1, c1 = bbox_coords(pts)
    full = crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
    dr, dc = direction

    if dc == 0:
        if list(range(int(c0), int(c1) + 1)) != list(nozzle_span):
            return None
    else:
        if list(range(int(r0), int(r1) + 1)) != list(nozzle_span):
            return None

    along = _axis_period(full, direction)
    if along is None:
        return None

    if (dr, dc) == (1, 0):
        tile = full[:along, :]
        return tile, int(r0), int(c0)
    if (dr, dc) == (-1, 0):
        tile = full[-along:, :]
        return tile, int(r1 - along + 1), int(c0)
    if (dr, dc) == (0, 1):
        tile = full[:, :along]
        return tile, int(r0), int(c0)
    if (dr, dc) == (0, -1):
        tile = full[:, -along:]
        return tile, int(r0), int(c1 - along + 1)
    return None


def _paint_tile(
    out: np.ndarray,
    ray_occ: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    *,
    bg: int,
) -> list[tuple[int, int]]:
    """Paint one tile copy and mark its occupied cells."""
    H, W = out.shape
    cells: list[tuple[int, int]] = []
    for i in range(tile.shape[0]):
        for j in range(tile.shape[1]):
            rr = int(r0 + i)
            cc = int(c0 + j)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(tile[i, j])
            if v == int(bg):
                continue
            out[rr, cc] = v
            ray_occ[rr, cc] = True
            cells.append((rr, cc))
    return cells


def _extend_with_collision(
    out: np.ndarray,
    ray_occ: np.ndarray,
    plus_occ: np.ndarray,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    seed_occ: np.ndarray | None = None,
) -> dict:
    """
    Extend a ray by whole tile-copies until the boundary or the next copy would hit an occupied ray.

    Collision checks use the occupancy that already exists when the copy is attempted. This makes
    row-major seed order observable in the final output.
    """
    H, W = out.shape
    dr, dc = map(int, direction)
    step_r = int(dr * tile.shape[0])
    step_c = int(dc * tile.shape[1])
    added_cells: list[tuple[int, int]] = []
    copies = 1
    collided = False
    blocker = "boundary"

    step = 1
    while True:
        r0 = int(seed_r0 + step * step_r)
        c0 = int(seed_c0 + step * step_c)
        in_bounds_any = False
        blocked = False
        pending: list[tuple[int, int, int]] = []

        for i in range(tile.shape[0]):
            for j in range(tile.shape[1]):
                rr = int(r0 + i)
                cc = int(c0 + j)
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                in_bounds_any = True
                if bool(plus_occ[rr, cc]) or bool(ray_occ[rr, cc]):
                    blocked = True
                    if bool(plus_occ[rr, cc]):
                        blocker = "plus"
                    elif seed_occ is not None and bool(seed_occ[rr, cc]):
                        blocker = "seed"
                    else:
                        blocker = "ray"
                    break
                v = int(tile[i, j])
                if v != int(bg):
                    pending.append((rr, cc, v))
            if blocked:
                break

        if not in_bounds_any:
            break
        if blocked:
            collided = True
            break

        for rr, cc, v in pending:
            out[rr, cc] = int(v)
            ray_occ[rr, cc] = True
            added_cells.append((int(rr), int(cc)))
        copies += 1
        step += 1

    return {
        "copies": int(copies),
        "collided": bool(collided),
        "blocker": str(blocker),
        "extra_cells": added_cells,
    }


def _collect_rays(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    plus_count: int,
    plus_color: int,
    bg: int,
) -> list[dict]:
    """Parse all plus-adjacent components into compact seed descriptions."""
    H, W = grid.shape
    non_mask = (grid != int(bg)) & (grid != int(plus_color))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    rays: list[dict] = []

    for pk in range(1, int(plus_count) + 1):
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
            direction = _marker_direction(pts, plus_mask, H, W)
            if direction is None:
                continue
            nozzle_span = _edge_span_indices(plus_mask, direction)
            extracted = _extract_seed_tile(grid, pts, direction, nozzle_span)
            if extracted is None:
                continue
            tile, seed_r0, seed_c0 = extracted
            rays.append(
                {
                    "plus_id": int(pk),
                    "direction": tuple(map(int, direction)),
                    "tile": np.array(tile, dtype=int),
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                }
            )

    rays.sort(key=lambda x: (int(x["seed_r0"]), int(x["seed_c0"]), _DIR_RANK[x["direction"]]))
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild every seed-attached ray from mixed-shape '+' sprites, applying a global collision-stop order.

    Input structure:
    - The background is the most common color.
    - One shared non-background color forms several disconnected '+'-like sprites, and those sprites
      may use different silhouettes in the same grid.
    - Each plus has several adjacent marker tiles on different arms; one marker somewhere in the grid
      may already appear as a long periodic stripe instead of only a compact seed.

    Rule:
    - For each plus-adjacent non-plus component, infer its attachment side from that specific plus,
      recover the compact seed tile by using the local nozzle span and the component's minimal 1-or-2
      step period, and keep only that seed as the ray source.
    - Paint all seeds first, then replay the rays in row-major order of seed origins. Each ray repeats
      its seed tile outward by whole tile-copies until the next copy would hit an existing ray cell.
    - '+' pixels remain protected; generator legality keeps foreign plus collisions out of the scene.

    Key insight:
    - The task now mixes three cues at once: local arm width depends on each plus shape, one plus can
      emit several independent rays, and ray length depends on which seeds were expanded earlier in the
      shared global order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    plus_color = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_color is None:
        return g.copy()

    plus_lab, plus_count = label_mask(g == int(plus_color), connectivity=4)
    rays = _collect_rays(g, plus_lab, plus_count, int(plus_color), int(bg))

    out = np.full((H, W), int(bg), dtype=int)
    plus_occ = g == int(plus_color)
    out[plus_occ] = int(plus_color)
    ray_occ = np.zeros((H, W), dtype=bool)

    for ray in rays:
        _paint_tile(
            out,
            ray_occ,
            np.array(ray["tile"], dtype=int),
            int(ray["seed_r0"]),
            int(ray["seed_c0"]),
            bg=int(bg),
        )

    for ray in rays:
        _extend_with_collision(
            out,
            ray_occ,
            plus_occ,
            np.array(ray["tile"], dtype=int),
            int(ray["seed_r0"]),
            int(ray["seed_c0"]),
            tuple(ray["direction"]),
            bg=int(bg),
        )

    return out


# --------------- generator (combo variant) ---------------

_H_RANGE = (18, 31)
_W_RANGE = (18, 27)
_N_PLUS_RANGE = (2, 4)
_MIN_MARKERS_PER_PLUS = 2
_MAX_MARKERS_PER_PLUS = 4
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_PLACEMENT_TRIES = 220
_SUBSET_TRIES = 24
_TILE_TRIES = 30
_GEN_RETRIES = 700
_PRE_EXT_IDEAL_COPIES = 6

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
    """Return the inclusive cross-axis span of one arm tip."""
    dr, dc = map(int, direction)
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _valid_directions_for_shape(plus_shape: np.ndarray) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for direction in _DIRECTIONS:
        s, e = _nozzle_span(plus_shape, direction)
        if int(e - s + 1) <= _SEED_MAX_CROSS:
            valid.append(tuple(direction))
    return valid


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.flatten()))


def _oriented_tile_has_real_period_two(tile: np.ndarray, direction: tuple[int, int]) -> bool:
    oriented = _orient_component(tile, direction)
    if direction[1] == 0:
        return not np.array_equal(oriented[0, :], oriented[1, :])
    return not np.array_equal(oriented[:, 0], oriented[:, 1])


def _build_seed_tile(
    plus_shape: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build one seed tile whose cross-axis matches the chosen nozzle span."""
    s, e = _nozzle_span(plus_shape, direction)
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
    tile = np.array(vals, dtype=int).reshape(th, tw)

    if along == 2 and not _oriented_tile_has_real_period_two(tile, direction):
        alt_choices = [c for c in palette if c != int(tile.flat[-1])]
        if alt_choices:
            tile.flat[-1] = int(rng.choice(alt_choices))
    if along == 2 and not _oriented_tile_has_real_period_two(tile, direction):
        return None
    return tile


def _seed_origin_from_bbox(
    plus_bbox: tuple[int, int, int, int],
    plus_shape: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    th, tw = map(int, tile.shape)
    s, _ = _nozzle_span(plus_shape, direction)
    dr, dc = map(int, direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - th)
        return int(r0), int(c0)
    r0 = int(pr0 + s)
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tw)
    return int(r0), int(c0)


def _seed_cells(seed_r0: int, seed_c0: int, tile: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(seed_r0 + i), int(seed_c0 + j))
        for i in range(tile.shape[0])
        for j in range(tile.shape[1])
    ]


def _plus_cells_fit(scene_occ: np.ndarray, plus_cells: list[tuple[int, int]]) -> bool:
    """Candidate plus cells must keep a one-cell moat from existing foreground."""
    H, W = scene_occ.shape
    for r, c in plus_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(scene_occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(scene_occ[rr, cc]) != 0:
                return False
    return True


def _seed_component_fits(
    trial_occ: np.ndarray,
    cells: list[tuple[int, int]],
    own_plus_cells: set[tuple[int, int]],
) -> bool:
    """Seed component must stay isolated from foreign pluses and foreign seed components."""
    H, W = trial_occ.shape
    cell_set = {(int(r), int(c)) for r, c in cells}
    if len(cell_set) != len(cells):
        return False
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(trial_occ[r, c]) != 0:
            return False
        for dr, dc in _DIRECTIONS:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in cell_set:
                continue
            if int(trial_occ[rr, cc]) == 2:
                return False
            if int(trial_occ[rr, cc]) == 1 and (rr, cc) not in own_plus_cells:
                return False
    return True


def _choose_plus_shapes(rng: np.random.Generator, n_plus: int) -> list[np.ndarray]:
    """Force at least two distinct silhouettes in each grid."""
    template_ids = list(range(len(_PLUS_SHAPES)))
    if n_plus == 2:
        chosen_ids = list(rng.choice(template_ids, size=2, replace=False))
    else:
        n_unique = int(rng.integers(2, min(len(_PLUS_SHAPES), n_plus) + 1))
        base_ids = list(rng.choice(template_ids, size=n_unique, replace=False))
        extras = list(rng.choice(base_ids, size=n_plus - n_unique, replace=True))
        chosen_ids = base_ids + extras
        rng.shuffle(chosen_ids)
    return [_PLUS_SHAPES[int(idx)] for idx in chosen_ids]


def _sample_direction_subset(
    valid_dirs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    max_k = min(_MAX_MARKERS_PER_PLUS, len(valid_dirs))
    min_k = min(_MIN_MARKERS_PER_PLUS, max_k)
    k = int(rng.integers(min_k, max_k + 1))
    order = valid_dirs.copy()
    rng.shuffle(order)
    return order[:k]


def _place_plus_with_seeds(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    scene_occ: np.ndarray,
) -> dict | None:
    """Place one mixed-shape plus and 2-4 compact seed tiles, but not the long rays yet."""
    ph, pw = plus_shape.shape
    valid_dirs = _valid_directions_for_shape(plus_shape)
    if len(valid_dirs) < _MIN_MARKERS_PER_PLUS:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]
        if not _plus_cells_fit(scene_occ, plus_cells):
            continue

        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        own_plus_cells = set((int(r), int(c)) for r, c in plus_cells)

        for _ in range(_SUBSET_TRIES):
            dirs = _sample_direction_subset(valid_dirs, rng)
            trial_occ = scene_occ.copy()
            for r, c in plus_cells:
                trial_occ[r, c] = 1

            rays: list[dict] = []
            used_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
            ok = True

            for direction in dirs:
                ray_info = None
                for _ in range(_TILE_TRIES):
                    tile = _build_seed_tile(plus_shape, direction, palette, rng)
                    if tile is None:
                        continue
                    sig = _tile_signature(tile)
                    if sig in used_tiles:
                        continue
                    seed_r0, seed_c0 = _seed_origin_from_bbox(plus_bbox, plus_shape, direction, tile)
                    cells = _seed_cells(seed_r0, seed_c0, tile)
                    if not _seed_component_fits(trial_occ, cells, own_plus_cells):
                        continue
                    for rr, cc in cells:
                        trial_occ[rr, cc] = 2
                    ray_info = {
                        "direction": tuple(map(int, direction)),
                        "tile": np.array(tile, dtype=int),
                        "seed_r0": int(seed_r0),
                        "seed_c0": int(seed_c0),
                        "seed_cells": cells,
                    }
                    used_tiles.add(sig)
                    break

                if ray_info is None:
                    ok = False
                    break
                rays.append(ray_info)

            if not ok:
                continue

            for r, c in plus_cells:
                scene_occ[r, c] = 1
            for ray in rays:
                for rr, cc in ray["seed_cells"]:
                    scene_occ[rr, cc] = 2

            return {
                "shape": np.array(plus_shape, dtype=int),
                "plus_bbox": plus_bbox,
                "plus_cells": plus_cells,
                "rays": rays,
            }

    return None


def _simulate_output(
    H: int,
    W: int,
    bg: int,
    plus_color: int,
    pluses: list[dict],
) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    """Build the full output by painting all seeds, then extending rays in row-major order."""
    out = np.full((H, W), int(bg), dtype=int)
    plus_occ = np.zeros((H, W), dtype=bool)
    ray_occ = np.zeros((H, W), dtype=bool)
    ordered_rays: list[dict] = []

    for pi, plus in enumerate(pluses):
        for r, c in plus["plus_cells"]:
            plus_occ[int(r), int(c)] = True
            out[int(r), int(c)] = int(plus_color)
        for ri, ray in enumerate(plus["rays"]):
            ordered_rays.append(
                {
                    "plus_index": int(pi),
                    "ray_index": int(ri),
                    "direction": tuple(ray["direction"]),
                    "tile": np.array(ray["tile"], dtype=int),
                    "seed_r0": int(ray["seed_r0"]),
                    "seed_c0": int(ray["seed_c0"]),
                }
            )

    ordered_rays.sort(key=lambda x: (int(x["seed_r0"]), int(x["seed_c0"]), _DIR_RANK[x["direction"]]))

    for ray in ordered_rays:
        seed_cells = _paint_tile(
            out,
            ray_occ,
            np.array(ray["tile"], dtype=int),
            int(ray["seed_r0"]),
            int(ray["seed_c0"]),
            bg=int(bg),
        )
        ray["seed_cells"] = list(seed_cells)

    seed_occ = np.array(ray_occ, dtype=bool)
    collisions = 0
    for ray in ordered_rays:
        extension = _extend_with_collision(
            out,
            ray_occ,
            plus_occ,
            np.array(ray["tile"], dtype=int),
            int(ray["seed_r0"]),
            int(ray["seed_c0"]),
            tuple(ray["direction"]),
            bg=int(bg),
            seed_occ=seed_occ,
        )
        ray.update(extension)
        if extension["blocker"] in {"plus", "seed"}:
            return None, None
        if extension["blocker"] == "ray":
            collisions += 1

    if collisions < 1:
        return None, None
    return out, ordered_rays


def _build_input_from_output(
    out: np.ndarray,
    rays: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one long exemplar ray; collapse every other ray back to its compact seed tile."""
    exemplar = max(
        range(len(rays)),
        key=lambda i: (int(rays[i]["copies"]), len(rays[i]["extra_cells"])),
    )
    strong = [i for i, ray in enumerate(rays) if int(ray["copies"]) >= _PRE_EXT_IDEAL_COPIES]
    if strong:
        exemplar = int(rng.choice(strong))

    inp = np.array(out, dtype=int)
    for i, ray in enumerate(rays):
        if i == exemplar:
            continue
        for rr, cc in ray["extra_cells"]:
            inp[int(rr), int(cc)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    plus_shapes = _choose_plus_shapes(rng, n_plus)
    scene_occ = np.zeros((H, W), dtype=int)
    pluses: list[dict] = []

    for plus_shape in plus_shapes:
        info = _place_plus_with_seeds(rng, plus_shape, palette, H, W, scene_occ)
        if info is None:
            return None
        pluses.append(info)

    if len({tuple(map(int, plus["shape"].flatten())) for plus in pluses}) < 2:
        return None

    out, ordered_rays = _simulate_output(H, W, bg, plus_color, pluses)
    if out is None or ordered_rays is None:
        return None

    if not any(len(ray["extra_cells"]) > 0 for ray in ordered_rays):
        return None

    inp = _build_input_from_output(out, ordered_rays, bg, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    return inp, out, pluses, ordered_rays


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            inp, _out, _pluses, _rays = attempt
            return inp
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
