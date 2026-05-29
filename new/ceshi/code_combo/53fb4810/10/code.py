# Combo variant of 53fb4810 (index 10)
# combo_spec: D6/0+D12/0
# Name: 障碍物与射线碰撞

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for combo D6/0 + D12/0
# ---------------------------------------
# a) Concrete behavior:
#    Original 53fb4810 extended each seed tile away from its '+' until the frame.
#    D6 adds detached obstacle bars, so extension stops at the first non-bg stripe
#    instead of the border. D12 then says newly generated rays are also blockers:
#    once one seed has been expanded, later seeds cannot paint through it.
#    Small trace:
#      clue ray A: plus -> AB AB AB [obstacle]
#      seed  B  : plus -> CD ... [obstacle]
#      seed  C  : plus -> EF ... [obstacle]
#    Fused output order:
#      1. A already exists in the input and remains as the first painted ray.
#      2. Among the remaining seeds, process them in visible reading order of the
#         '+' anchors (top-to-bottom, then left-to-right).
#      3. Each ray repeats stripe-by-stripe and stops on the first occupied stripe,
#         whether that occupancy is an obstacle bar or an earlier ray.
#    The generator makes one seed stop on its own obstacle and another seed stop
#    earlier on that first generated ray, so both mechanisms are simultaneously active.
# b) Complexity calibration:
#    The scene always contains exactly three '+' anchors:
#      - one pre-expanded clue ray,
#      - one seed that grows to an obstacle,
#      - one later seed that would reach its obstacle if processed alone but is cut
#        short by the earlier generated ray.
#    This keeps the fusion visible in every sample without introducing too many
#    interacting rays. Obstacles use one dedicated color that never appears in seeds.
# c) Hidden assumptions from the original solver that break here:
#    - Non-bg connected components can no longer include every non-plus color:
#      obstacle bars touch ray endpoints, so obstacle color must be excluded before
#      marker/ray component labeling.
#    - Extending rays independently is no longer valid. The write surface must be
#      updated after each processed seed so later seeds can collide with earlier rays.
#    - Iteration order over plus components must be explicit and visible; relying on
#      connected-component ids would make the rule depend on scan order rather than
#      a human-observable property.
# d) Placement legality:
#    The generator uses a structured layout. Two seeds are positioned so their
#    projected rectangles cross, but the earlier seed reaches the crossing first and
#    the later seed hits that painted ray before its own obstacle. A third clue ray
#    is placed on a disjoint vertical track. All pluses, seeds, obstacles, and final
#    rays are checked for in-bounds placement and plus-plus separation.
# e) Visual observability:
#    Humans can infer the fused rule because one long clue ray demonstrates the
#    repeat-until-blocked idea, the same obstacle color recurs on every lane, and
#    one late seed visibly stops before reaching its own matching obstacle because a
#    previously drawn ray is in the way. The order cue is visible: the already
#    expanded ray exists first, then the remaining seeds follow top-to-bottom order.
# f) Visual cleanliness:
#    Only one deliberate ray-ray collision is introduced. The clue ray runs on a
#    separate column, while the collision pair uses one vertical lane and one
#    horizontal lane. This avoids dense overlaps and keeps each stopping cause legible.
# g) Color inferability:
#    Seed colors repeat verbatim. Obstacle color is global and purely acts as a
#    blocker. No color ids participate in the ordering rule.
# h) New solver assumptions audited against the generator:
#    - Exactly one adjacent marker component per '+' remains after removing the
#      obstacle color; generator enforces detached obstacle bars.
#    - The only pre-expanded adjacent component has along-axis span > 2; generator
#      keeps the clue ray long enough that it is distinguishable from a compact seed.
#    - Among unexpanded seeds, reading order over the '+' anchors is the intended
#      processing order; generator positions the collision pair so reversing that
#      order changes the output.


_AXIAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
            if int(r1 - r0 + 1) >= int(min_h) and int(c1 - c0 + 1) >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)
    return int(best_col) if best_col is not None else None


def _adjacent_component_ids(mark_lab: np.ndarray, plus_mask: np.ndarray) -> set[int]:
    """Return payload component ids 4-neighbor-adjacent to one '+' component."""
    H, W = mark_lab.shape
    out: set[int] = set()
    for r, c in np.argwhere(plus_mask):
        r = int(r)
        c = int(c)
        for dr, dc in _AXIAL_DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                out.add(int(mark_lab[rr, cc]))
    return out


def _marker_direction_and_alignment(
    pts: np.ndarray,
    plus_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    """Return marker direction and bbox if a payload component is aligned to one arm."""
    if pts.size == 0:
        return None
    H, W = plus_mask.shape
    dirs: set[tuple[int, int]] = set()
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _AXIAL_DIRS:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None

    dr, dc = next(iter(dirs))
    r0, c0, r1, c1 = bbox_coords(pts)
    plus_coords = np.argwhere(plus_mask)
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        nozzle_cols = sorted(
            int(cc) for cc in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, cc])
        )
        seed_cols = list(range(int(c0), int(c1) + 1))
        if seed_cols != nozzle_cols:
            return None
    else:
        nozzle_col = int(pc1) if dc == 1 else int(pc0)
        nozzle_rows = sorted(
            int(rr) for rr in range(int(pr0), int(pr1) + 1) if bool(plus_mask[rr, nozzle_col])
        )
        seed_rows = list(range(int(r0), int(r1) + 1))
        if seed_rows != nozzle_rows:
            return None
    return (int(dr), int(dc)), (int(r0), int(c0), int(r1), int(c1))


def _infer_obstacle_color_and_mark_lab(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    *,
    bg: int,
    plus_col: int,
) -> tuple[int | None, np.ndarray]:
    """
    Identify the dedicated obstacle color by removal.

    The correct obstacle color is the one whose removal leaves every '+' adjacent
    to exactly one structurally valid marker/ray component.
    """
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(plus_col)})
    for obstacle_col in colors:
        non_mask = (grid != int(bg)) & (grid != int(plus_col)) & (grid != int(obstacle_col))
        mark_lab, _ = label_mask(non_mask, connectivity=4)
        ok = True
        for pk in range(1, int(plus_lab.max()) + 1):
            plus_mask = plus_lab == int(pk)
            adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
            if len(adj_ids) != 1:
                ok = False
                break
            cid = next(iter(adj_ids))
            pts = np.argwhere(mark_lab == int(cid))
            if _marker_direction_and_alignment(pts, plus_mask) is None:
                ok = False
                break
        if ok:
            return int(obstacle_col), mark_lab

    non_mask = (grid != int(bg)) & (grid != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)
    return None, mark_lab


def _component_along_span(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    """Return bbox size along the ray axis."""
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)


def _normalize_direction(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    return int(dr), int(dc)


def _iter_ray_stripes(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
):
    """Yield successive 1-cell-thick stripes after the seed tile."""
    h, w = tile.shape
    dr, dc = _normalize_direction(direction)
    r0 = int(r0)
    c0 = int(c0)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            yield [(r0 + i, tc, int(tile[i, k % w])) for i in range(h)]
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            yield [(r0 + i, tc, int(tile[i, (-k) % w])) for i in range(h)]
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            yield [(tr, c0 + j, int(tile[k % h, j])) for j in range(w)]
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            yield [(tr, c0 + j, int(tile[(-k) % h, j])) for j in range(w)]
    else:
        raise ValueError("_iter_ray_stripes: direction must be axial")


def extend_repeating_tile_until_blocked(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Extend a periodic tile until the next stripe would hit a non-bg cell."""
    H, W = out.shape
    bg = int(bg)

    for stripe in _iter_ray_stripes(H, W, tile, int(r0), int(c0), direction):
        active = [(rr, cc, v) for rr, cc, v in stripe if int(v) != int(bg)]
        if any(int(out[rr, cc]) != int(bg) for rr, cc, _ in active):
            break
        for rr, cc, v in active:
            out[rr, cc] = int(v)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker rays from '+' anchors while respecting both blockers and prior rays.

    Input structure:
    - The dominant color is background.
    - One large non-bg color forms several identical '+'-like anchors.
    - Each '+' has one adjacent payload component; exactly one of those payloads is
      already expanded into a long periodic clue ray.
    - A dedicated extra color appears only in detached obstacle bars farther along
      the ray lanes.

    Rule:
    - Recover the obstacle color, then identify the one marker/ray component touching
      each '+' and infer its direction from the side of attachment.
    - Keep the pre-expanded clue ray as already painted. Process the remaining seeds
      in reading order of their '+' anchors.
    - For each processed seed, repeat its tile stripe-by-stripe away from the '+'
      and stop as soon as the next stripe would hit any occupied non-bg cell,
      whether that cell belongs to an obstacle bar or to an earlier ray.

    Key insight: The clue ray is not just an example of the repeating pattern; it is
    also an existing collider. After obstacle color is removed for parsing, the task
    reduces to ordered ray growth on a mutable canvas.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    _obstacle_col, mark_lab = _infer_obstacle_color_and_mark_lab(
        g,
        plus_lab,
        bg=int(bg),
        plus_col=int(plus_col),
    )

    plus_infos: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue
        adj_ids = _adjacent_component_ids(mark_lab, plus_mask)
        if len(adj_ids) != 1:
            continue
        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        parsed = _marker_direction_and_alignment(pts, plus_mask)
        if parsed is None:
            continue
        direction, bbox = parsed
        r0, c0, r1, c1 = map(int, bbox)
        tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
        plus_r0, plus_c0, _, _ = bbox_coords(plus_coords)
        plus_infos.append(
            {
                "plus_order": (int(plus_r0), int(plus_c0)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "direction": (int(direction[0]), int(direction[1])),
                "tile": np.array(tile, dtype=int),
                "expanded": _component_along_span((int(r0), int(c0), int(r1), int(c1)), direction) > 2,
            }
        )

    plus_infos.sort(key=lambda info: (0 if info["expanded"] else 1, info["plus_order"][0], info["plus_order"][1]))

    for info in plus_infos:
        r0, c0, r1, c1 = map(int, info["bbox"])
        tile = np.array(info["tile"], dtype=int)
        direction = tuple(map(int, info["direction"]))
        extend_repeating_tile_until_blocked(
            out,
            tile,
            int(r0),
            int(c0),
            direction,
            bg=int(bg),
        )
    return out


# --------------- generator (structured fused scene) ---------------

_H_RANGE = (18, 29)
_W_RANGE = (18, 23)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_PLACEMENT_TRIES = 200
_GEN_RETRIES = 500
_PRE_EXT_IDEAL_UNITS = 4

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        dtype=int,
    ),
]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive nozzle span on the cross-axis."""
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
    """Build a seed tile whose cross-axis matches the nozzle span."""
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
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Return the top-left corner of the seed tile adjacent to the nozzle."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        return (int(pr1 + 1) if dr == 1 else int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1) if dc == 1 else int(pc0 - tw))


def _compute_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> int:
    """Maximum number of full tile units that fit before the frame."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    if dr == 1:
        return 1 + (H - (int(seed_r0) + th)) // th
    if dr == -1:
        return 1 + int(seed_r0) // th
    if dc == 1:
        return 1 + (W - (int(seed_c0) + tw)) // tw
    return 1 + int(seed_c0) // tw


def _obstacle_cells_after_units(
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    stop_units: int,
    obstacle_color: int,
) -> list[tuple[int, int, int]]:
    """Place a 1-stripe obstacle immediately after `stop_units` full copies."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    cells: list[tuple[int, int, int]] = []
    if dr == 1:
        row = int(seed_r0 + stop_units * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dr == -1:
        row = int(seed_r0 - 1 - (stop_units - 1) * th)
        for j in range(tw):
            cells.append((row, int(seed_c0 + j), int(obstacle_color)))
    elif dc == 1:
        col = int(seed_c0 + stop_units * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    else:
        col = int(seed_c0 - 1 - (stop_units - 1) * tw)
        for i in range(th):
            cells.append((int(seed_r0 + i), col, int(obstacle_color)))
    return cells


def _seed_cells(seed_r0: int, seed_c0: int, tile: np.ndarray) -> list[tuple[int, int, int]]:
    """Return the concrete seed cells."""
    h, w = tile.shape
    return [
        (int(seed_r0 + i), int(seed_c0 + j), int(tile[i, j]))
        for i in range(h)
        for j in range(w)
    ]


def _plus_cells(plus_shape: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    """Return occupied cells of one '+' sprite at (r0,c0)."""
    ph, pw = plus_shape.shape
    return [
        (int(r0 + dr), int(c0 + dc))
        for dr in range(ph)
        for dc in range(pw)
        if int(plus_shape[dr, dc]) == 1
    ]


def _cells_in_bounds(H: int, W: int, cells: list[tuple[int, int, int]]) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c, _ in cells)


def _mark_plus_forbid(forbid: np.ndarray, plus_cells: list[tuple[int, int]]) -> None:
    H, W = forbid.shape
    for r, c in plus_cells:
        for dr, dc in ((0, 0), * _AXIAL_DIRS):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                forbid[rr, cc] = True


def _grow_ray_and_capture(
    out: np.ndarray,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    obstacle_color: int,
) -> tuple[list[tuple[int, int, int]], str]:
    """Paint one ray on `out` and return added cells plus the stopping reason."""
    H, W = out.shape
    bg = int(bg)
    obstacle_color = int(obstacle_color)
    added: list[tuple[int, int, int]] = []

    for stripe in _iter_ray_stripes(H, W, tile, int(seed_r0), int(seed_c0), direction):
        active = [(rr, cc, v) for rr, cc, v in stripe if int(v) != int(bg)]
        blockers = [int(out[rr, cc]) for rr, cc, _ in active if int(out[rr, cc]) != int(bg)]
        if blockers:
            if all(int(v) == int(obstacle_color) for v in blockers):
                return added, "obstacle"
            return added, "ray"
        for rr, cc, v in active:
            out[rr, cc] = int(v)
            added.append((int(rr), int(cc), int(v)))
    return added, "boundary"


def _simulate_no_new_ray_collisions(
    inp: np.ndarray,
    sprites: list[dict],
    *,
    bg: int,
    clue_name: str,
) -> np.ndarray:
    """
    Baseline used for degeneration checks: each non-clue seed sees only the input.
    Earlier generated rays are not treated as blockers.
    """
    out = np.array(inp, dtype=int)
    frozen = np.array(inp, dtype=int)
    H, W = out.shape
    for sprite in sprites:
        if sprite["name"] == clue_name:
            continue
        seed_r0 = int(sprite["seed_r0"])
        seed_c0 = int(sprite["seed_c0"])
        tile = np.array(sprite["tile"], dtype=int)
        direction = tuple(map(int, sprite["direction"]))
        for stripe in _iter_ray_stripes(H, W, tile, seed_r0, seed_c0, direction):
            active = [(rr, cc, v) for rr, cc, v in stripe if int(v) != int(bg)]
            if any(int(frozen[rr, cc]) != int(bg) for rr, cc, _ in active):
                break
            for rr, cc, v in active:
                out[rr, cc] = int(v)
    return out


def _scene_metrics(inp: np.ndarray, out: np.ndarray, sprites: list[dict], *, bg: int, clue_name: str) -> dict:
    """Quantify that both obstacle stops and ray-ray stops are active."""
    no_collision = _simulate_no_new_ray_collisions(inp, sprites, bg=int(bg), clue_name=clue_name)
    stop_counts = {"obstacle": 0, "ray": 0, "boundary": 0}
    for sprite in sprites:
        stop_counts[str(sprite["stop_reason"])] += 1
    return {
        "collision_stop_count": int(stop_counts["ray"]),
        "obstacle_stop_count": int(stop_counts["obstacle"]),
        "collision_delta_cells": int(np.sum(out != no_collision)),
    }


def _transform_direction(direction: tuple[int, int], symmetry: str) -> tuple[int, int]:
    """Map a ray direction through one of the layout-preserving symmetries."""
    dr, dc = map(int, direction)
    if symmetry == "identity":
        return int(dr), int(dc)
    if symmetry == "flip_ud":
        return int(-dr), int(dc)
    if symmetry == "flip_lr":
        return int(dr), int(-dc)
    if symmetry == "rot180":
        return int(-dr), int(-dc)
    raise ValueError(f"unknown symmetry: {symmetry}")


def _transform_cell(r: int, c: int, H: int, W: int, symmetry: str) -> tuple[int, int]:
    """Map one cell through a layout-preserving symmetry."""
    r = int(r)
    c = int(c)
    if symmetry == "identity":
        return int(r), int(c)
    if symmetry == "flip_ud":
        return int(H - 1 - r), int(c)
    if symmetry == "flip_lr":
        return int(r), int(W - 1 - c)
    if symmetry == "rot180":
        return int(H - 1 - r), int(W - 1 - c)
    raise ValueError(f"unknown symmetry: {symmetry}")


def _apply_scene_symmetry(
    inp: np.ndarray,
    out: np.ndarray,
    sprites: list[dict],
    plus_shape: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Apply a safe symmetry without changing the intended top-to-bottom processing order."""
    if symmetry == "identity":
        return np.array(inp, copy=True), np.array(out, copy=True), sprites, np.array(plus_shape, copy=True)
    if symmetry == "flip_ud":
        grid_fn = np.flipud
    elif symmetry == "flip_lr":
        grid_fn = np.fliplr
    elif symmetry == "rot180":
        grid_fn = lambda arr: np.flipud(np.fliplr(arr))
    else:
        raise ValueError(f"unknown symmetry: {symmetry}")

    H, W = inp.shape
    new_inp = np.array(grid_fn(inp), copy=True)
    new_out = np.array(grid_fn(out), copy=True)
    new_plus_shape = np.array(grid_fn(plus_shape), copy=True)
    new_sprites: list[dict] = []

    for sprite in sprites:
        new_sprite = dict(sprite)
        new_sprite["direction"] = _transform_direction(tuple(map(int, sprite["direction"])), symmetry)
        for key in ("seed_cells", "obstacle_cells", "grown_cells"):
            if key in sprite:
                new_sprite[key] = [
                    (*_transform_cell(r, c, H, W, symmetry), int(v))
                    for r, c, v in sprite[key]
                ]

        plus_cells = _plus_cells(plus_shape, int(sprite["plus_r0"]), int(sprite["plus_c0"]))
        new_plus_cells = [_transform_cell(r, c, H, W, symmetry) for r, c in plus_cells]
        plus_rows = [int(r) for r, _ in new_plus_cells]
        plus_cols = [int(c) for _, c in new_plus_cells]
        new_sprite["plus_r0"] = int(min(plus_rows))
        new_sprite["plus_c0"] = int(min(plus_cols))

        seed_rows = [int(r) for r, _, _ in new_sprite["seed_cells"]]
        seed_cols = [int(c) for _, c, _ in new_sprite["seed_cells"]]
        new_sprite["seed_r0"] = int(min(seed_rows))
        new_sprite["seed_c0"] = int(min(seed_cols))
        new_sprites.append(new_sprite)

    return new_inp, new_out, new_sprites, new_plus_shape


def _gen_attempt(
    rng: np.random.Generator,
    *,
    return_meta: bool = False,
) -> np.ndarray | dict | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color, obstacle_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {plus_color, obstacle_color}]
    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]

    ph, pw = plus_shape.shape
    down = (1, 0)
    right = (0, 1)

    tile_clue = _build_seed_tile(plus_shape, down, palette, rng)
    tile_early = _build_seed_tile(plus_shape, down, palette, rng)
    tile_late = _build_seed_tile(plus_shape, right, palette, rng)
    if tile_clue is None or tile_early is None or tile_late is None:
        return None

    th_clue, tw_clue = tile_clue.shape
    th_early, tw_early = tile_early.shape
    th_late, tw_late = tile_late.shape
    s_down, _ = _nozzle_span(plus_shape, down)
    s_right, _ = _nozzle_span(plus_shape, right)

    for _ in range(_PLACEMENT_TRIES):
        early_r0 = int(rng.integers(1, max(2, H // 4)))
        early_c0_min = max(4, pw + 4)
        early_c0_max = W - pw - 6
        if early_c0_min > early_c0_max:
            continue
        early_c0 = int(rng.integers(early_c0_min, early_c0_max + 1))
        seed_early_r0, seed_early_c0 = _seed_origin(
            (early_r0, early_c0, early_r0 + ph - 1, early_c0 + pw - 1), plus_shape, down, tile_early
        )

        gap_rows = int(rng.integers(max(2, th_early), max(5, th_early + 4)))
        gap_cols = int(rng.integers(tw_late + 1, tw_late + 5))

        late_r0 = int(seed_early_r0 + gap_rows - s_right)
        late_c0 = int(seed_early_c0 - pw - gap_cols)
        if late_r0 < 1 or late_c0 < 1:
            continue
        if late_r0 + ph >= H - 1 or late_c0 + pw >= W - 1:
            continue
        seed_late_r0, seed_late_c0 = _seed_origin(
            (late_r0, late_c0, late_r0 + ph - 1, late_c0 + pw - 1), plus_shape, right, tile_late
        )
        if seed_late_r0 != seed_early_r0 + gap_rows or seed_late_c0 != seed_early_c0 - gap_cols:
            continue

        clue_c0 = W - pw - 2
        clue_r0_max = max(1, min(H // 4, late_r0 - ph - 2))
        if clue_r0_max < 1:
            continue
        clue_r0 = int(rng.integers(1, clue_r0_max + 1))
        if clue_r0 + ph >= H - th_clue - 2:
            continue
        seed_clue_r0, seed_clue_c0 = _seed_origin(
            (clue_r0, clue_c0, clue_r0 + ph - 1, clue_c0 + pw - 1), plus_shape, down, tile_clue
        )

        full_early = _compute_full_units(H, W, seed_early_r0, seed_early_c0, down, tile_early)
        full_late = _compute_full_units(H, W, seed_late_r0, seed_late_c0, right, tile_late)
        full_clue = _compute_full_units(H, W, seed_clue_r0, seed_clue_c0, down, tile_clue)
        min_stop_early = 1 + (gap_rows + th_late - 1) // th_early
        min_stop_late = 1 + (gap_cols + tw_early - 1) // tw_late
        min_stop_clue = max(_PRE_EXT_IDEAL_UNITS, 3)
        if full_early <= min_stop_early or full_late <= min_stop_late or full_clue <= min_stop_clue:
            continue

        stop_early = int(rng.integers(min_stop_early, full_early))
        stop_late = int(rng.integers(min_stop_late, full_late))
        stop_clue = int(rng.integers(min_stop_clue, full_clue))

        obstacle_early = _obstacle_cells_after_units(seed_early_r0, seed_early_c0, down, tile_early, stop_early, obstacle_color)
        obstacle_late = _obstacle_cells_after_units(seed_late_r0, seed_late_c0, right, tile_late, stop_late, obstacle_color)
        obstacle_clue = _obstacle_cells_after_units(seed_clue_r0, seed_clue_c0, down, tile_clue, stop_clue, obstacle_color)
        if not all(_cells_in_bounds(H, W, cells) for cells in (obstacle_early, obstacle_late, obstacle_clue)):
            continue

        plus_forbid = np.zeros((H, W), dtype=bool)
        placements = [
            ("clue", clue_r0, clue_c0),
            ("early", early_r0, early_c0),
            ("late", late_r0, late_c0),
        ]
        ok = True
        for _name, pr0, pc0 in placements:
            cells = _plus_cells(plus_shape, int(pr0), int(pc0))
            if any(plus_forbid[r, c] for r, c in cells):
                ok = False
                break
            _mark_plus_forbid(plus_forbid, cells)
        if not ok:
            continue

        sprites = [
            {
                "name": "clue",
                "plus_r0": int(clue_r0),
                "plus_c0": int(clue_c0),
                "direction": down,
                "tile": np.array(tile_clue, dtype=int),
                "seed_r0": int(seed_clue_r0),
                "seed_c0": int(seed_clue_c0),
                "seed_cells": _seed_cells(seed_clue_r0, seed_clue_c0, tile_clue),
                "obstacle_cells": obstacle_clue,
            },
            {
                "name": "early",
                "plus_r0": int(early_r0),
                "plus_c0": int(early_c0),
                "direction": down,
                "tile": np.array(tile_early, dtype=int),
                "seed_r0": int(seed_early_r0),
                "seed_c0": int(seed_early_c0),
                "seed_cells": _seed_cells(seed_early_r0, seed_early_c0, tile_early),
                "obstacle_cells": obstacle_early,
            },
            {
                "name": "late",
                "plus_r0": int(late_r0),
                "plus_c0": int(late_c0),
                "direction": right,
                "tile": np.array(tile_late, dtype=int),
                "seed_r0": int(seed_late_r0),
                "seed_c0": int(seed_late_c0),
                "seed_cells": _seed_cells(seed_late_r0, seed_late_c0, tile_late),
                "obstacle_cells": obstacle_late,
            },
        ]

        out = np.full((H, W), int(bg), dtype=int)
        occupied: set[tuple[int, int]] = set()
        for sprite in sprites:
            plus_cells = _plus_cells(plus_shape, sprite["plus_r0"], sprite["plus_c0"])
            static_cells = [(r, c, int(plus_color)) for r, c in plus_cells] + sprite["seed_cells"] + sprite["obstacle_cells"]
            if any((int(r), int(c)) in occupied for r, c, _ in static_cells):
                ok = False
                break
            for r, c, v in static_cells:
                occupied.add((int(r), int(c)))
                out[int(r), int(c)] = int(v)
        if not ok:
            continue

        # Paint clue first, then the seed-only rays in visible reading order.
        sprite_by_name = {sprite["name"]: sprite for sprite in sprites}
        order = ["clue", "early", "late"]
        for name in order:
            sprite = sprite_by_name[name]
            grown_cells, stop_reason = _grow_ray_and_capture(
                out,
                sprite["tile"],
                sprite["seed_r0"],
                sprite["seed_c0"],
                sprite["direction"],
                bg=int(bg),
                obstacle_color=int(obstacle_color),
            )
            sprite["grown_cells"] = grown_cells
            sprite["stop_reason"] = stop_reason

        if sprite_by_name["clue"]["stop_reason"] != "obstacle":
            continue
        if sprite_by_name["early"]["stop_reason"] != "obstacle":
            continue
        if sprite_by_name["late"]["stop_reason"] != "ray":
            continue
        if not sprite_by_name["late"]["grown_cells"]:
            continue

        inp = np.array(out, copy=True)
        for name in ("early", "late"):
            seed_set = {(int(r), int(c)) for r, c, _ in sprite_by_name[name]["seed_cells"]}
            for r, c, _ in sprite_by_name[name]["grown_cells"]:
                if (int(r), int(c)) not in seed_set:
                    inp[int(r), int(c)] = int(bg)

        metrics = _scene_metrics(inp, out, sprites, bg=int(bg), clue_name="clue")
        if metrics["collision_stop_count"] < 1:
            continue
        if metrics["obstacle_stop_count"] < 2:
            continue
        if metrics["collision_delta_cells"] <= 0:
            continue

        try:
            produced = main(inp)
        except Exception:
            continue
        if not np.array_equal(produced, out):
            continue

        # Only left-right mirroring preserves the collision pair's reading-order
        # dependency; vertical symmetries would swap which seed is processed first.
        symmetry = str(rng.choice(["identity", "flip_lr"]))
        inp, out, sprites, plus_shape = _apply_scene_symmetry(inp, out, sprites, plus_shape, symmetry)

        try:
            produced = main(inp)
        except Exception:
            continue
        if not np.array_equal(produced, out):
            continue

        if return_meta:
            return {
                "input": inp,
                "output": out,
                "bg": int(bg),
                "plus_color": int(plus_color),
                "obstacle_color": int(obstacle_color),
                "plus_shape": np.array(plus_shape, dtype=int),
                "sprites": sprites,
                "metrics": metrics,
                "symmetry": symmetry,
            }
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
