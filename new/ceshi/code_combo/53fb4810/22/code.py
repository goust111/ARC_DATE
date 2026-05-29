# Combo variant of 53fb4810 (index 22)
# combo_spec: D11/0+D12/0+D16/0
# Name: 多射线碰撞按距离排序

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, neighbors4


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and pipeline impact
# - Original task: each '+' owns one seed, rays are independent, and a pre-extended clue ray
#   is harmless because the rest of the solver ignores interactions.
# - Combo rule: each '+' now owns 2-4 seeds on distinct arms. The solver must first recover
#   all seeds touching each plus, then sort whole plus sprites by distance to the grid center,
#   and finally project every ray of the nearer pluses before projecting rays of farther ones.
# - Small trace:
#   Near plus A has an upward seed and a rightward seed; farther plus B has a leftward seed.
#   A is processed first, so its upward ray is drawn into the workspace. When B later tries to
#   advance leftward, the next emitted column would touch A's existing ray, so B halts early.
#   Without D11 there would be only one seed per plus; without D12 the later ray would continue
#   through; without D16 reversing A/B order would change which ray survives.
#
# b) Complexity calibration
# - Marker count is capped at 2-4 because the four compass arms are the natural visual slots and
#   this already makes the "one sprite -> several rays" change obvious.
# - Seed tiles stay in the original family: cross-axis width must match the arm nozzle and the
#   along-axis size is only 1 or 2. This keeps the collision/sorting idea visually primary.
# - The generator forces at least one blocked ray and also rejects samples where reversing the
#   plus-order gives the same output, so the fusion does not collapse back to the original task.
#
# c) Implicit assumptions from the original main()
# - `label_mask(non_mask, connectivity=4)` still treats a whole multicolor seed/ray as one
#   component. That only works if distinct rays never 4-touch. The new collision rule therefore
#   stops one strip earlier: a ray halts before the next strip would overlap or 4-neighbor a
#   foreign non-background pixel. This preserves separable components.
# - The old solver could process components in any order. That assumption is false now because
#   earlier rays constrain later ones. The new main() collects all pluses first, sorts them by
#   geometric distance to the grid center, and only then mutates the output grid.
# - A pre-extended clue component cannot be replayed from its full bbox verbatim because that
#   would shift the repetition phase whenever the visible prefix length is not a full period.
#   The solver therefore recovers the smallest consistent along-axis period and re-anchors that
#   tile at the ray's active boundary before continuing projection.
#
# d) Placement legality
# - Initial inputs contain only pluses, compact seeds, and exactly one full exemplar ray from the
#   nearest plus. Seeds may touch only their own plus arm. Distinct seed components cannot touch.
# - During output construction, each candidate strip must stay in bounds and must not overlap or
#   4-touch foreign non-background cells; otherwise that ray stops entirely.
# - The generator plans a near/far pair with a guaranteed potential collision, then validates the
#   whole scene by simulating both near->far and far->near orders.
#
# e) Visual observability
# - The distance key is purely geometric: plus center vs grid center. No color-id ordering or
#   hidden numeric convention is involved.
# - The input shows multiple seeds around each plus and one long exemplar ray on the nearest plus.
#   Across examples, a human can see that farther rays truncate when they approach already visible
#   nearer rays, and that the center-near sprite acts first.
#
# f) Visual cleanliness
# - Grid sizes are kept moderate and plus count stays low. Only one ray is pre-extended in the
#   input, which exposes the repetition rule without flooding the board.
# - The one-cell collision moat prevents merged tangles of multicolor components, so each seed/ray
#   remains visually attributable to one plus arm.
#
# g) Color inferability
# - Colors are copied directly from the visible seed tile. There is no remapping layer.
# - Each plus uses distinct seed tiles across its chosen directions when possible, making the
#   multi-marker structure easy to parse without introducing too many colors at once.


_H_RANGE = (18, 31)
_W_RANGE = (18, 27)
_N_PLUS_CHOICES = [2, 2, 3]
_MIN_MARKERS_PER_PLUS = 2
_MAX_MARKERS_PER_PLUS = 4
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_CLUE_STRIPS = 6
_PLACEMENT_TRIES = 220
_SUBSET_TRIES = 18
_TILE_TRIES = 24
_PAIR_TRIES = 80
_EXTRA_PLUS_TRIES = 120
_GEN_RETRIES = 700

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
_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_ORDER = {(-1, 0): 0, (0, -1): 1, (0, 1): 2, (1, 0): 3}

_LAST_GENERATION_INFO: dict[str, object] = {}


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the color with the most big connected components."""
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
            r0, c0, r1, c1 = bbox_coords(coords)
            if int(r1 - r0 + 1) >= int(min_h) and int(c1 - c0 + 1) >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)
    return int(best_col) if best_col is not None else None


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Inclusive cross-axis span of one arm tip."""
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
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a compact tile whose cross-axis matches the selected arm nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = map(int, direction)
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(th, tw)
    if n_cells > 1 and np.all(tile == tile.flat[0]) and len(palette) >= 2:
        alt = int(rng.choice([c for c in palette if c != int(tile.flat[0])]))
        tile.flat[-1] = alt
    return tile


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.flatten()))


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Origin of the compact seed tile adjacent to one arm."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = map(int, tile.shape)
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        return (int(pr1 + 1), int(pc0 + s)) if dr == 1 else (int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1)) if dc == 1 else (int(pr0 + s), int(pc0 - tw))


def _seed_cells_from_tile(tile: np.ndarray, r0: int, c0: int) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    th, tw = map(int, tile.shape)
    for i in range(th):
        for j in range(tw):
            cells.append((int(r0 + i), int(c0 + j), int(tile[i, j])))
    return cells


def _plus_distance_key(coords: np.ndarray, H: int, W: int) -> tuple[int, int, int]:
    """Squared distance from plus bbox center to grid center, with geometric tie-breaks."""
    r0, c0, r1, c1 = bbox_coords(coords)
    rr = int(r0 + r1)
    cc = int(c0 + c1)
    dist2 = int((rr - (H - 1)) ** 2 + (cc - (W - 1)) ** 2)
    return dist2, rr, cc


def _component_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    dirs: set[tuple[int, int]] = set()
    for pr, pc in pts:
        rr0 = int(pr)
        cc0 = int(pc)
        for dr, dc in _DIRECTIONS:
            rr = rr0 + dr
            cc = cc0 + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None
    return next(iter(dirs))


def _alignment_matches_nozzle(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    """The component bbox must keep exactly the nozzle's cross-axis span."""
    pr0, pc0, pr1, pc1 = bbox_coords(plus_coords)
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, direction)
    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        nozzle_cols = [
            int(cc)
            for cc in range(int(pc0), int(pc1) + 1)
            if bool(plus_mask[nozzle_row, cc])
        ]
        return list(range(int(c0), int(c1) + 1)) == nozzle_cols
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    nozzle_rows = [
        int(rr)
        for rr in range(int(pr0), int(pr1) + 1)
        if bool(plus_mask[rr, nozzle_col])
    ]
    return list(range(int(r0), int(r1) + 1)) == nozzle_rows


def _ray_sort_key(ray: dict) -> tuple[int, int, int]:
    direction = tuple(map(int, ray["direction"]))
    return _DIR_ORDER[direction], int(ray["r0"]), int(ray["c0"])


def _oriented_strips(component: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, ...]]:
    """Return 1-cell strips ordered from the plus side outward."""
    dr, dc = map(int, direction)
    if (dr, dc) == (0, 1):
        return [tuple(map(int, component[:, j])) for j in range(component.shape[1])]
    if (dr, dc) == (0, -1):
        return [tuple(map(int, component[:, j])) for j in range(component.shape[1] - 1, -1, -1)]
    if (dr, dc) == (1, 0):
        return [tuple(map(int, component[i, :])) for i in range(component.shape[0])]
    if (dr, dc) == (-1, 0):
        return [tuple(map(int, component[i, :])) for i in range(component.shape[0] - 1, -1, -1)]
    raise ValueError("_oriented_strips: invalid direction")


def _recover_pattern_tile(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """
    Recover the minimal repeating tile from a visible ray prefix.

    Inputs may already contain one long exemplar ray. We infer the smallest along-axis period
    consistent with the visible component, then anchor that period at the ray's active boundary
    so future strips continue with the correct phase.
    """
    strips = _oriented_strips(component, direction)
    period = len(strips)
    for p in range(1, len(strips) + 1):
        if all(strips[idx] == strips[idx % p] for idx in range(len(strips))):
            period = int(p)
            break

    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, direction)
    if (dr, dc) == (0, 1):
        return component[:, component.shape[1] - period :].copy(), int(r0), int(c1 - period + 1)
    if (dr, dc) == (0, -1):
        return component[:, :period].copy(), int(r0), int(c0)
    if (dr, dc) == (1, 0):
        return component[component.shape[0] - period :, :].copy(), int(r1 - period + 1), int(c0)
    if (dr, dc) == (-1, 0):
        return component[:period, :].copy(), int(r0), int(c0)
    raise ValueError("_recover_pattern_tile: invalid direction")


def _next_strip_cells(
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    offset: int,
) -> list[tuple[int, int, int]]:
    """Cells of the next 1-cell-thick emitted strip at a given offset."""
    h, w = map(int, tile.shape)
    dr, dc = map(int, direction)
    if (dr, dc) == (0, 1):
        tc = int(c0 + w + offset)
        return [(int(r0 + i), tc, int(tile[i, offset % w])) for i in range(h)]
    if (dr, dc) == (0, -1):
        tc = int(c0 - 1 - offset)
        return [(int(r0 + i), tc, int(tile[i, (-(offset + 1)) % w])) for i in range(h)]
    if (dr, dc) == (1, 0):
        tr = int(r0 + h + offset)
        return [(tr, int(c0 + j), int(tile[offset % h, j])) for j in range(w)]
    if (dr, dc) == (-1, 0):
        tr = int(r0 - 1 - offset)
        return [(tr, int(c0 + j), int(tile[(-(offset + 1)) % h, j])) for j in range(w)]
    raise ValueError("_next_strip_cells: invalid direction")


def _extend_ray_component(
    out: np.ndarray,
    ray: dict,
    *,
    bg: int,
    protect: set[int],
) -> tuple[set[tuple[int, int]], bool, int]:
    """
    Extend one component while preserving a 1-cell moat around foreign structures.

    Returns (final_component_cells, blocked, added_strips).
    """
    H, W = out.shape
    tile = np.array(ray["tile"], dtype=int)
    r0 = int(ray["r0"])
    c0 = int(ray["c0"])
    direction = tuple(map(int, ray["direction"]))
    own = set((int(r), int(c)) for r, c in ray["own_cells"])
    offset = 0
    added_strips = 0

    while True:
        strip = _next_strip_cells(tile, r0, c0, direction, offset)
        candidate = [(rr, cc, vv) for rr, cc, vv in strip if int(vv) != int(bg)]
        if not candidate:
            offset += 1
            continue

        candidate_pos = {(int(rr), int(cc)) for rr, cc, _ in candidate}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc, _ in candidate):
            return own, False, added_strips

        blocked = False
        for rr, cc, _ in candidate:
            cell_val = int(out[rr, cc])
            if cell_val in protect:
                blocked = True
                break
            if cell_val != int(bg) and (rr, cc) not in own:
                blocked = True
                break
            for nr, nc in neighbors4(int(rr), int(cc)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in own or (nr, nc) in candidate_pos:
                    continue
                neigh_val = int(out[nr, nc])
                if neigh_val != int(bg):
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            return own, True, added_strips

        for rr, cc, vv in candidate:
            out[rr, cc] = int(vv)
            own.add((int(rr), int(cc)))
        added_strips += 1
        offset += 1


def _project_sorted_pluses(
    out: np.ndarray,
    pluses: list[dict],
    *,
    bg: int,
    plus_color: int,
    reverse: bool = False,
) -> np.ndarray:
    """Project all rays plus-by-plus in distance order."""
    ordered = sorted(pluses, key=lambda p: tuple(p["dist_key"]), reverse=bool(reverse))
    for order_idx, plus in enumerate(ordered):
        plus["order_rank"] = int(order_idx)
        for ray_idx, ray in enumerate(sorted(plus["rays"], key=_ray_sort_key)):
            final_cells, blocked, added = _extend_ray_component(
                out,
                ray,
                bg=int(bg),
                protect={int(plus_color)},
            )
            ray["final_cells"] = set(final_cells)
            ray["blocked"] = bool(blocked)
            ray["added_strips"] = int(added)
            ray["order_rank"] = int(order_idx)
            ray["sorted_index"] = int(ray_idx)
    return out


def _extract_scene_from_grid(grid: np.ndarray) -> tuple[int, int | None, list[dict]]:
    """Parse pluses and attached non-plus components from the current grid."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return bg, None, []

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

    pluses: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            rr0 = int(r)
            cc0 = int(c)
            for dr, dc in _DIRECTIONS:
                rr = rr0 + dr
                cc = cc0 + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        rays: list[dict] = []
        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            direction = _component_direction(pts, plus_mask, H, W)
            if direction is None:
                continue
            bbox = bbox_coords(pts)
            if not _alignment_matches_nozzle(plus_mask, plus_coords, bbox, direction):
                continue
            component = crop_bbox(g, bbox)
            tile, r0, c0 = _recover_pattern_tile(component, bbox, direction)
            rays.append(
                {
                    "direction": tuple(map(int, direction)),
                    "tile": tile,
                    "r0": int(r0),
                    "c0": int(c0),
                    "own_cells": {(int(r), int(c)) for r, c in pts},
                }
            )

        if not rays:
            continue

        pluses.append(
            {
                "bbox": bbox_coords(plus_coords),
                "cells": {(int(r), int(c)) for r, c in plus_coords},
                "coords": plus_coords.copy(),
                "dist_key": _plus_distance_key(plus_coords, H, W),
                "rays": rays,
            }
        )

    return bg, int(plus_col), pluses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every marker component attached to a '+' sprite, but let nearer sprites act first.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several identical '+'-like sprites.
    - Each plus has 2-4 separate marker components on distinct arms, and exactly one ray in the
      whole grid may already be visible as a long periodic exemplar.

    Rule:
    - Recover every non-plus component that is edge-adjacent to a plus, infer its arm direction,
      and treat that component's bbox as the currently visible prefix of one ray.
    - Sort plus sprites by distance to the grid center from near to far. For each plus in that
      order, project all of its rays outward by periodic repetition.
    - A ray stops entirely before the next emitted strip would overlap or 4-neighbor any foreign
      non-background pixel; plus pixels are also protected.

    Key insight: the task is no longer "extend each seed independently". One plus can launch
    several rays, and the visible result depends on which plus is closer to the center because
    earlier rays become obstacles for later ones.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg, plus_col, pluses = _extract_scene_from_grid(g)
    if plus_col is None or not pluses:
        return out
    return _project_sorted_pluses(out, pluses, bg=int(bg), plus_color=int(plus_col), reverse=False)


def _valid_directions_for_shape(plus_shape: np.ndarray) -> list[tuple[int, int]]:
    return [d for d in _DIRECTIONS if _nozzle_span(plus_shape, d)[1] - _nozzle_span(plus_shape, d)[0] + 1 <= _SEED_MAX_CROSS]


def _plus_cells_fit(occ: np.ndarray, plus_cells: list[tuple[int, int]]) -> bool:
    H, W = occ.shape
    for r, c in plus_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(occ[r, c]) != 0:
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < H and 0 <= nc < W and int(occ[nr, nc]) != 0:
                return False
    return True


def _seed_cells_fit(
    trial_occ: np.ndarray,
    cells: list[tuple[int, int, int]],
    own_plus_cells: set[tuple[int, int]],
) -> bool:
    H, W = trial_occ.shape
    cell_pos = {(int(r), int(c)) for r, c, _ in cells}
    if len(cell_pos) != len(cells):
        return False
    for r, c, _ in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(trial_occ[r, c]) != 0:
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in cell_pos or (nr, nc) in own_plus_cells:
                continue
            if int(trial_occ[nr, nc]) != 0:
                return False
    return True


def _sample_direction_subset(
    valid_dirs: list[tuple[int, int]],
    forced_dirs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    forced = [tuple(map(int, d)) for d in forced_dirs]
    rest = [tuple(map(int, d)) for d in valid_dirs if tuple(map(int, d)) not in set(forced)]
    max_k = min(_MAX_MARKERS_PER_PLUS, len(valid_dirs))
    min_k = min(_MIN_MARKERS_PER_PLUS, max_k)
    total_min = max(len(forced), min_k)
    total = int(rng.integers(total_min, max_k + 1))
    rng.shuffle(rest)
    return forced + rest[: max(0, total - len(forced))]


def _plan_bundle_at_position(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    r0: int,
    c0: int,
    *,
    forced_dirs: list[tuple[int, int]] | None = None,
    scene_id: int = 0,
) -> dict | None:
    """Plan one plus sprite with compact seeds only."""
    forced_dirs = [] if forced_dirs is None else [tuple(map(int, d)) for d in forced_dirs]
    ph, pw = plus_shape.shape
    plus_cells = [
        (int(r0 + dr), int(c0 + dc))
        for dr in range(ph)
        for dc in range(pw)
        if int(plus_shape[dr, dc]) != 0
    ]
    if not _plus_cells_fit(occ, plus_cells):
        return None

    plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
    own_plus_cells = set(plus_cells)
    valid_dirs = _valid_directions_for_shape(plus_shape)
    if any(d not in valid_dirs for d in forced_dirs):
        return None

    for _ in range(_SUBSET_TRIES):
        chosen_dirs = _sample_direction_subset(valid_dirs, forced_dirs, rng)
        trial_occ = occ.copy()
        for rr, cc in plus_cells:
            trial_occ[rr, cc] = 1

        used_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
        rays: list[dict] = []
        ok = True
        for direction in chosen_dirs:
            ray = None
            for _ in range(_TILE_TRIES):
                tile = _build_seed_tile(plus_shape, direction, palette, rng)
                if tile is None:
                    break
                sig = _tile_signature(tile)
                if sig in used_tiles:
                    continue
                sr0, sc0 = _seed_origin(plus_bbox, plus_shape, direction, tile)
                cells = _seed_cells_from_tile(tile, sr0, sc0)
                if not _seed_cells_fit(trial_occ, cells, own_plus_cells):
                    continue
                for rr, cc, _vv in cells:
                    trial_occ[rr, cc] = 2
                ray = {
                    "direction": tuple(map(int, direction)),
                    "tile": tile,
                    "r0": int(sr0),
                    "c0": int(sc0),
                    "own_cells": {(int(rr), int(cc)) for rr, cc, _ in cells},
                }
                used_tiles.add(sig)
                break
            if ray is None:
                ok = False
                break
            rays.append(ray)

        if not ok:
            continue

        coords = np.argwhere(plus_shape != 0) + np.array([int(r0), int(c0)])
        return {
            "scene_id": int(scene_id),
            "bbox": plus_bbox,
            "cells": own_plus_cells,
            "coords": coords.astype(int),
            "dist_key": _plus_distance_key(coords, H, W),
            "rays": rays,
            "forced_ray_index": 0 if forced_dirs else None,
        }

    return None


def _commit_bundle(
    base: np.ndarray,
    occ: np.ndarray,
    plus: dict,
    *,
    plus_color: int,
) -> None:
    for r, c in plus["cells"]:
        occ[int(r), int(c)] = 1
        base[int(r), int(c)] = int(plus_color)
    for ray in plus["rays"]:
        for rr, cc in ray["own_cells"]:
            occ[int(rr), int(cc)] = 2
            base[int(rr), int(cc)] = int(ray["tile"][int(rr - ray["r0"]), int(cc - ray["c0"])])


def _bundle_dist2(plus: dict) -> int:
    return int(tuple(plus["dist_key"])[0])


def _central_candidates(H: int, W: int, ph: int, pw: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    cr = (H - ph) // 2
    cc = (W - pw) // 2
    candidates: list[tuple[int, int]] = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r0 = int(cr + dr)
            c0 = int(cc + dc)
            if 1 <= r0 <= H - ph - 2 and 1 <= c0 <= W - pw - 2:
                candidates.append((r0, c0))
    rng.shuffle(candidates)
    return candidates


def _collision_position_candidates(
    plus_a: dict,
    plus_shape: np.ndarray,
    a_dir: tuple[int, int],
    side: str,
    H: int,
    W: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Candidate positions for the farther plus whose forced ray aims at plus A."""
    ph, pw = plus_shape.shape
    ar0, ac0, _ar1, _ac1 = plus_a["bbox"]
    sA, eA = _nozzle_span(plus_shape, a_dir)
    candidates: list[tuple[int, int]] = []

    if a_dir[1] == 0:
        b_dir = (0, 1) if side == "left" else (0, -1)
        sB, _eB = _nozzle_span(plus_shape, b_dir)
        if a_dir[0] == 1:
            row_band = list(range(int(ar0 + ph + 2), int(H - 2)))
        else:
            row_band = list(range(2, int(ar0 - 1)))
        rng.shuffle(row_band)
        barrier_min = int(ac0 + sA)
        barrier_max = int(ac0 + eA)
        for t in row_band[: max(1, min(len(row_band), 18))]:
            br0 = int(t - sB)
            if not (1 <= br0 <= H - ph - 2):
                continue
            if side == "right":
                cands = list(range(int(barrier_max + 4), int(W - pw - 1)))
            else:
                cands = list(range(1, int(barrier_min - pw - 3)))
            rng.shuffle(cands)
            for bc0 in cands[: max(1, min(len(cands), 14))]:
                candidates.append((int(br0), int(bc0)))
    else:
        b_dir = (1, 0) if side == "up" else (-1, 0)
        sB, _eB = _nozzle_span(plus_shape, b_dir)
        if a_dir[1] == 1:
            col_band = list(range(int(ac0 + pw + 2), int(W - 2)))
        else:
            col_band = list(range(2, int(ac0 - 1)))
        rng.shuffle(col_band)
        barrier_min = int(ar0 + sA)
        barrier_max = int(ar0 + eA)
        for t in col_band[: max(1, min(len(col_band), 18))]:
            bc0 = int(t - sB)
            if not (1 <= bc0 <= W - pw - 2):
                continue
            if side == "down":
                rands = list(range(int(barrier_max + 4), int(H - ph - 1)))
            else:
                rands = list(range(1, int(barrier_min - ph - 3)))
            rng.shuffle(rands)
            for br0 in rands[: max(1, min(len(rands), 14))]:
                candidates.append((int(br0), int(bc0)))

    rng.shuffle(candidates)
    return b_dir, candidates


def _forced_sides_for_direction(direction: tuple[int, int]) -> list[str]:
    if direction[1] == 0:
        return ["left", "right"]
    return ["up", "down"]


def _plan_forced_pair(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
) -> tuple[dict, dict] | None:
    """Build a near-center plus and a farther plus whose ray is intended to be blocked."""
    ph, pw = plus_shape.shape
    center_positions = _central_candidates(H, W, ph, pw, rng)
    valid_dirs = _valid_directions_for_shape(plus_shape)

    for _ in range(_PAIR_TRIES):
        if not center_positions:
            break
        a_dir = tuple(map(int, rng.choice(valid_dirs)))
        for ar0, ac0 in center_positions[: max(1, min(len(center_positions), 10))]:
            plus_a = _plan_bundle_at_position(
                rng,
                plus_shape,
                palette,
                H,
                W,
                occ,
                int(ar0),
                int(ac0),
                forced_dirs=[a_dir],
                scene_id=0,
            )
            if plus_a is None:
                continue

            occ_a = occ.copy()
            base_stub = np.zeros((H, W), dtype=int)
            _commit_bundle(base_stub, occ_a, plus_a, plus_color=1)
            for side in rng.permutation(_forced_sides_for_direction(a_dir)):
                b_dir, pos_candidates = _collision_position_candidates(
                    plus_a, plus_shape, a_dir, str(side), H, W, rng
                )
                for br0, bc0 in pos_candidates:
                    plus_b = _plan_bundle_at_position(
                        rng,
                        plus_shape,
                        palette,
                        H,
                        W,
                        occ_a,
                        int(br0),
                        int(bc0),
                        forced_dirs=[tuple(map(int, b_dir))],
                        scene_id=1,
                    )
                    if plus_b is None:
                        continue
                    if _bundle_dist2(plus_b) <= _bundle_dist2(plus_a):
                        continue
                    return plus_a, plus_b
    return None


def _place_extra_plus(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    *,
    center_dist_floor: int,
    scene_id: int,
) -> dict | None:
    ph, pw = plus_shape.shape
    for _ in range(_EXTRA_PLUS_TRIES):
        r0 = int(rng.integers(1, H - ph - 1))
        c0 = int(rng.integers(1, W - pw - 1))
        plus = _plan_bundle_at_position(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            r0,
            c0,
            forced_dirs=[],
            scene_id=scene_id,
        )
        if plus is None:
            continue
        if _bundle_dist2(plus) <= int(center_dist_floor):
            continue
        return plus
    return None


def _clone_scene(pluses: list[dict]) -> list[dict]:
    return copy.deepcopy(pluses)


def _build_input_from_base(
    base: np.ndarray,
    pluses: list[dict],
    out: np.ndarray,
) -> tuple[np.ndarray, dict] | None:
    """Keep one long ray from the nearest plus as the exemplar clue."""
    nearest = min(pluses, key=lambda p: tuple(p["dist_key"]))
    rays = list(nearest["rays"])
    if not rays:
        return None
    clue = max(rays, key=lambda r: (int(r["added_strips"]), len(r["final_cells"])))
    if int(clue["added_strips"]) < _MIN_CLUE_STRIPS:
        return None

    inp = np.array(base, dtype=int).copy()
    for rr, cc in clue["final_cells"]:
        inp[int(rr), int(cc)] = int(out[int(rr), int(cc)])
    return inp, clue


def _scene_order_metrics(base: np.ndarray, pluses: list[dict], bg: int, plus_color: int) -> tuple[np.ndarray, np.ndarray]:
    forward_scene = _clone_scene(pluses)
    reverse_scene = _clone_scene(pluses)
    out_fwd = _project_sorted_pluses(base.copy(), forward_scene, bg=int(bg), plus_color=int(plus_color), reverse=False)
    out_rev = _project_sorted_pluses(base.copy(), reverse_scene, bg=int(bg), plus_color=int(plus_color), reverse=True)
    return out_fwd, out_rev


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]
    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]

    base = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)

    pair = _plan_forced_pair(rng, plus_shape, palette, H, W, occ)
    if pair is None:
        return None

    pluses = [pair[0], pair[1]]
    for plus in pluses:
        _commit_bundle(base, occ, plus, plus_color=plus_color)

    target_n = int(rng.choice(_N_PLUS_CHOICES))
    next_id = 2
    while len(pluses) < target_n:
        extra = _place_extra_plus(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            center_dist_floor=_bundle_dist2(pluses[0]),
            scene_id=next_id,
        )
        if extra is None:
            break
        pluses.append(extra)
        _commit_bundle(base, occ, extra, plus_color=plus_color)
        next_id += 1

    dists = [int(p["dist_key"][0]) for p in pluses]
    if len(set(dists)) != len(dists):
        return None
    if min(pluses, key=lambda p: tuple(p["dist_key"]))["scene_id"] != 0:
        return None

    out, out_rev = _scene_order_metrics(base, pluses, bg, plus_color)
    diff_cells = int(np.sum(out != out_rev))
    if diff_cells <= 0:
        return None

    scene_after = _clone_scene(pluses)
    out = _project_sorted_pluses(base.copy(), scene_after, bg=int(bg), plus_color=int(plus_color), reverse=False)
    plus_a = next(p for p in scene_after if int(p["scene_id"]) == 0)
    plus_b = next(p for p in scene_after if int(p["scene_id"]) == 1)

    blocked_rays = [
        ray for plus in scene_after for ray in plus["rays"] if bool(ray.get("blocked", False))
    ]
    if not blocked_rays:
        return None
    forced_b = plus_b["rays"][int(plus_b["forced_ray_index"])]
    if not bool(forced_b.get("blocked", False)):
        return None
    if int(forced_b.get("added_strips", 0)) < 1:
        return None

    build = _build_input_from_base(base, scene_after, out)
    if build is None:
        return None
    inp, clue = build

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    info = {
        "grid_shape": (int(H), int(W)),
        "plus_count": int(len(scene_after)),
        "marker_counts": [int(len(p["rays"])) for p in sorted(scene_after, key=lambda q: tuple(q["dist_key"]))],
        "blocked_rays": int(len(blocked_rays)),
        "reverse_diff_cells": int(diff_cells),
        "closest_dist2": int(min(dists)),
        "forced_collision_added_strips": int(forced_b["added_strips"]),
        "clue_added_strips": int(clue["added_strips"]),
    }
    return inp, info


def generate_input() -> np.ndarray:
    global _LAST_GENERATION_INFO
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        grid, info = attempt
        _LAST_GENERATION_INFO = dict(info)
        return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
