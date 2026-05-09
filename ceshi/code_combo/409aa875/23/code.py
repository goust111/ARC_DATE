# Combo variant of 409aa875 (index 23)
# combo_spec: D2/2+D13/2
# Name: 射线到碰撞+传送门

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# a) Concrete behavior and trace
#    - Triangles stay identical to the source task: each 3-pixel monochrome component is either an
#      axis-aligned L or a diagonal V, and its geometry still determines one of the 8 firing directions.
#    - Portals are singleton pixels. Two singleton pixels of the same non-background color form one
#      paired portal. This combo uses two portal pairs so a ray can visibly traverse more than one warp.
#    - The original "apex + 5 * dir" rule is replaced by true ray casting: starting from apex+dir, the
#      projectile advances until the first triangle hit or until the next step would leave the grid.
#      When a step lands on a portal pixel, the projectile immediately reappears at the paired portal and
#      keeps the same direction; portal cells themselves are transit-only, not painted outputs.
#    - Example trace: a down-right L could visit bg cells (4,4),(5,5), step onto portal A at (6,6),
#      emerge at A'=(11,3), continue to (12,4), then step onto portal B at (13,5), emerge at B'=(3,12),
#      and finally hit a distant triangle at (4,13). In the source task the visible effect would have
#      been only at apex+5*dir; in this combo the relevant endpoint is the first triangle reached after
#      the chained teleports.
#
# b) Complexity calibration
#    - Two portal pairs are enough to make multi-warp paths possible without cluttering the 16x16 board.
#    - The generator forces at least one blue collision created by portal-using rays and at least one ray
#      with two or more portal hops. That keeps the portal mechanic central rather than cosmetic.
#    - Sprite colors and portal colors are drawn from a small shared palette excluding output-only colors
#      1 and 9, so the human-visible categories remain easy to separate.
#
# c) Hidden assumption audit
#    - Original solver assumption "every non-bg component is a triangle" is broken by singleton portals,
#      so parsing now separates 3-pixel triangles from 1-pixel portal components.
#    - Original destination calculation `apex + 5*dir` is invalid under D2/2 and D13/2 simultaneously:
#      the path length is no longer fixed and the route can jump non-locally mid-flight.
#    - Original hit test `dest in cell_to_comp` is still usable only after destination becomes "first
#      triangle encountered on the traced ray" or "last background cell before exiting the board".
#    - The generator can no longer validate a placement by looking at a single endpoint. It must protect
#      the whole traversed background prefix of already-committed rays so later sprites do not silently
#      introduce earlier collisions.
#
# d) Placement legality
#    - Triangles remain pairwise 8-disconnected. Portal cells are treated as occupied seeds, so triangle
#      halos cannot touch them either.
#    - Portal cells are sampled from the interior and kept pairwise well separated, which reduces both
#      accidental ambiguity and degenerate immediate-exit teleports.
#    - Each committed bg event protects its full background route; each committed hit event protects only
#      the prefix before the intended first hit. This guarantees output-first consistency.
#
# e) Visual observability
#    - Repeated singleton colors expose portal pairing directly.
#    - The generator rejects cases where portal-aware ray casting produces the same output as plain ray
#      casting without portals or as the original fixed-5 solver. Humans therefore see genuine fusion
#      behavior, not a silent copy of a simpler rule.
#    - At least one sample ray per grid uses multiple portal hops, so the "can pass through multiple
#      portals before hitting a far sprite" part of the combo is actually present in the data.
#
# f) Visual cleanliness
#    - Only two portal colors are introduced, and output still paints only final impact cells or whole hit
#      sprites. No full ray trails are drawn.
#    - Portal cells are kept away from sprite bodies and from each other, and output destinations are
#      never allowed to coincide with portal cells in generated examples.
#
# g) Change inferability
#    - Colors do not encode hidden ordering. Portal pairing is inferred only from same-color singleton
#      repetition, while 9 and 1 keep the same structural meanings as before: lone/hit -> 9, shared bg
#      destination -> 1.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr_sum = d1[0] + d2[0]
    dc_sum = d1[1] + d2[1]
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _find_triangles_and_portals(
    g: np.ndarray, bg: int
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    triangles: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    singleton_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            tri_info = _triangle_apex_and_dir(comp) if len(comp) == 3 else None
            if tri_info is not None:
                cid = len(triangles)
                triangles.append(comp)
                for rc in comp:
                    cell_to_comp[rc] = cid
            elif len(comp) == 1:
                singleton_by_color[col].append(comp[0])

    portal_map: dict[tuple[int, int], tuple[int, int]] = {}
    for cells in singleton_by_color.values():
        if len(cells) == 2:
            a, b = cells
            portal_map[a] = b
            portal_map[b] = a

    return triangles, cell_to_comp, portal_map


def _trace_ray_empty(
    apex: tuple[int, int],
    direction: tuple[int, int],
    portal_map: dict[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    h, w = shape
    dr, dc = direction
    r, c = apex
    seen_states: set[tuple[int, int]] = set()
    bg_path: list[tuple[int, int]] = []
    portal_entries: list[tuple[int, int]] = []
    portal_hops = 0
    last_kind = "start"

    while True:
        state = (int(r), int(c))
        if state in seen_states:
            return None
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return {
                "bg_path": tuple(bg_path),
                "bg_set": frozenset(bg_path),
                "dest": bg_path[-1] if bg_path and last_kind == "bg" else None,
                "portal_hops": int(portal_hops),
                "portal_entries": tuple(portal_entries),
            }

        cell = (int(nr), int(nc))
        if cell in portal_map:
            portal_entries.append(cell)
            portal_hops += 1
            r, c = portal_map[cell]
            last_kind = "portal"
            continue

        bg_path.append(cell)
        r, c = cell
        last_kind = "bg"


def _trace_ray_on_grid(
    apex: tuple[int, int],
    direction: tuple[int, int],
    portal_map: dict[tuple[int, int], tuple[int, int]],
    triangle_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    h, w = shape
    dr, dc = direction
    r, c = apex
    seen_states: set[tuple[int, int]] = set()
    last_bg: tuple[int, int] | None = None
    portal_hops = 0

    while True:
        state = (int(r), int(c))
        if state in seen_states:
            return None
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            if last_bg is None:
                return None
            return {"dest": last_bg, "hit": False, "portal_hops": int(portal_hops)}

        cell = (int(nr), int(nc))
        if cell in triangle_cells:
            return {"dest": cell, "hit": True, "portal_hops": int(portal_hops)}
        if cell in portal_map:
            portal_hops += 1
            r, c = portal_map[cell]
            continue

        last_bg = cell
        r, c = cell


def _collect_ray_events(grid: np.ndarray, *, use_portals: bool) -> tuple[dict[tuple[int, int], list[int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, cell_to_comp, portal_map = _find_triangles_and_portals(g, bg)
    triangle_cells = set(cell_to_comp)
    if not use_portals:
        portal_map = {}

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    traces: list[dict] = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        trace = _trace_ray_on_grid(ad[0], ad[1], portal_map, triangle_cells, g.shape)
        if trace is None:
            continue
        traces.append({"cid": cid, **trace})
        dest_to_shooters[trace["dest"]].append(cid)
    return dest_to_shooters, traces


def _collect_fixed5_events(grid: np.ndarray) -> dict[tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps, _cell_to_comp, _portal_map = _find_triangles_and_portals(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(int(rr), int(cc))].append(cid)
    return dest_to_shooters


def _apply_events(
    grid: np.ndarray,
    triangles: list[list[tuple[int, int]]],
    cell_to_comp: dict[tuple[int, int], int],
    dest_to_shooters: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}
    for cid in hit_components:
        for r, c in triangles[cid]:
            out[r, c] = 9

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = 1 if len(shooters) >= 2 else 9

    return out


def _solve_ray(grid: np.ndarray, *, use_portals: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, cell_to_comp, _portal_map = _find_triangles_and_portals(g, bg)
    dest_to_shooters, _traces = _collect_ray_events(g, use_portals=use_portals)
    return _apply_events(g, triangles, cell_to_comp, dest_to_shooters)


def _solve_fixed5(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, cell_to_comp, _portal_map = _find_triangles_and_portals(g, bg)
    dest_to_shooters = _collect_fixed5_events(g)
    return _apply_events(g, triangles, cell_to_comp, dest_to_shooters)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast triangle rays until collision or boundary, letting same-color portal pairs warp the route.

    Input structure:
    - Background is the most common color.
    - Triangle sprites are still 3-pixel monochrome L-shapes or diagonal V-shapes.
    - Portal pairs are shown as isolated singleton pixels, with each pair sharing one non-background color.

    Rule:
    - Each triangle still supplies one apex and one firing direction from its local shape.
    - The projectile no longer stops after 5 cells. It advances cell by cell until it first reaches a
      triangle or until the next step would leave the grid.
    - If a step lands on a portal pixel, the projectile instantly moves to the paired portal and keeps
      flying in the same direction. A single ray may pass through multiple portals before it stops.
    - Hitting a triangle recolors that entire triangle to maroon (9). Reaching an empty background stop
      paints that cell maroon (9), except that shared empty stops from two or more rays become blue (1).

    Key insight: Sprite geometry still determines the direction, but the visible result now comes from
    tracing the whole ray through portal warps and taking the first real collision, not from reading a
    fixed offset.
    """
    return _solve_ray(np.asarray(grid, dtype=int), use_portals=True)


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]
PORTAL_COLORS = [0, 2, 3, 4, 5, 6, 8]
SPRITE_COUNT_RANGE = (4, 9)
PORTAL_PAIR_COUNT = 2
MAX_HIT_EVENTS = 2
MAX_EXTRA_COLLISIONS = 1
MAX_ATTEMPTS = 500
PORTAL_INTERIOR = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_portal_placements(
    portal_map: dict[tuple[int, int], tuple[int, int]]
) -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_bg_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    portal_cells = set(portal_map)

    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                if set(cells) & portal_cells:
                    continue
                route = _trace_ray_empty((ar, ac), (dr, dc), portal_map, (H, W))
                if route is None or not route["bg_path"]:
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "bg_path": route["bg_path"],
                    "bg_set": route["bg_set"],
                    "dest": route["dest"],
                    "portal_hops": route["portal_hops"],
                    "portal_entries": route["portal_entries"],
                }
                placements.append(placement)
                if placement["dest"] is not None:
                    by_dest[placement["dest"]].append(placement)
                for cell in placement["bg_path"]:
                    by_bg_cell[cell].append(placement)

    return placements, by_dest, by_bg_cell


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    return (
        not (placement["halo"] & occupied)
        and not (placement["cell_set"] & protected)
        and not (placement["cell_set"] & reserved_bg)
    )


def _route_clear(placement: dict, occupied: set[tuple[int, int]]) -> bool:
    return not (placement["bg_set"] & occupied)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _add_sprite_state(
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    sprite: dict,
) -> None:
    idx = len(sprites)
    sprites.append(sprite)
    occupied.update(sprite["placement"]["cells"])
    protected.update(sprite["protected"])
    for cell in sprite["placement"]["cells"]:
        cell_to_sprite[cell] = idx


def _first_existing_hit(
    placement: dict, cell_to_sprite: dict[tuple[int, int], int]
) -> tuple[int, tuple[int, int], int] | None:
    for idx, cell in enumerate(placement["bg_path"]):
        if cell in cell_to_sprite:
            return idx, cell, int(cell_to_sprite[cell])
    return None


def _choose_collision_pair(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    require_portal: bool,
    require_multi: bool,
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(placements_by_dest)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = placements_by_dest[dest][:]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            if require_portal and first["portal_hops"] < 1:
                continue
            if not _fits(first, occupied, protected, reserved_bg):
                continue
            if not _route_clear(first, occupied):
                continue
            occ2 = occupied | set(first["cells"])
            prot2 = protected | set(first["bg_path"])
            for second in candidates[i + 1 :]:
                if require_portal and second["portal_hops"] < 1:
                    continue
                if require_multi and max(first["portal_hops"], second["portal_hops"]) < 2:
                    continue
                if not _fits(second, occ2, prot2, reserved_bg):
                    continue
                if not _route_clear(second, occ2):
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    prefer_portal: bool,
    prefer_multi: bool,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    primary: list[dict] = []
    secondary: list[dict] = []
    fallback: list[dict] = []

    for placement in candidates:
        if placement["dest"] is None:
            continue
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if not _fits(placement, occupied, protected, reserved_bg):
            continue
        if not _route_clear(placement, occupied):
            continue
        if prefer_multi and placement["portal_hops"] >= 2:
            primary.append(placement)
        elif prefer_portal and placement["portal_hops"] >= 1:
            secondary.append(placement)
        else:
            fallback.append(placement)

    if primary:
        return primary[0]
    if secondary:
        return secondary[0]
    return fallback[0] if fallback else None


def _choose_hit(
    rng: random.Random,
    target_idx: int,
    sprites: list[dict],
    placements_by_bg_cell: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    prefer_portal: bool,
    prefer_multi: bool,
) -> tuple[dict, int, tuple[int, int]] | None:
    target_cells = list(sprites[target_idx]["placement"]["cells"])
    rng.shuffle(target_cells)

    seen_ids: set[int] = set()
    multi_candidates: list[dict] = []
    portal_candidates: list[dict] = []
    plain_candidates: list[dict] = []

    for cell in target_cells:
        for placement in placements_by_bg_cell[cell]:
            pid = id(placement)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if not _fits(placement, occupied, protected, reserved_bg):
                continue
            hit = _first_existing_hit(placement, cell_to_sprite)
            if hit is None:
                continue
            stop_idx, stop_cell, hit_idx = hit
            if hit_idx != target_idx:
                continue
            if placement["portal_hops"] >= 2:
                multi_candidates.append({"placement": placement, "stop_idx": stop_idx, "stop_cell": stop_cell})
            elif placement["portal_hops"] >= 1:
                portal_candidates.append({"placement": placement, "stop_idx": stop_idx, "stop_cell": stop_cell})
            else:
                plain_candidates.append({"placement": placement, "stop_idx": stop_idx, "stop_cell": stop_cell})

    if prefer_multi and multi_candidates:
        choice = rng.choice(multi_candidates)
        return choice["placement"], choice["stop_idx"], choice["stop_cell"]
    if prefer_portal and portal_candidates:
        choice = rng.choice(portal_candidates)
        return choice["placement"], choice["stop_idx"], choice["stop_cell"]
    for bucket in (multi_candidates, portal_candidates, plain_candidates):
        if bucket:
            choice = rng.choice(bucket)
            return choice["placement"], choice["stop_idx"], choice["stop_cell"]
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False

    tri_count = 0
    portal_groups: list[int] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {BG}):
        singletons = 0
        for comp in _cc8(grid == col):
            tri_info = _triangle_apex_and_dir(comp) if len(comp) == 3 else None
            if tri_info is not None:
                tri_count += 1
            elif len(comp) == 1:
                singletons += 1
            else:
                return False
        if singletons:
            portal_groups.append(singletons)
    return tri_count == n_sprites and sorted(portal_groups) == [2, 2]


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for extra_coll in range(0, min(MAX_EXTRA_COLLISIONS, (n_sprites - 2 - e_hit) // 2) + 1):
            unique = n_sprites - 2 - e_hit - 2 * extra_coll
            if unique < 0:
                continue
            if e_hit + unique < 1:
                continue
            # Bias toward at least one hit event so 2-4 training examples are more likely
            # to demonstrate both "hit sprite" and "stop on background" branches.
            weight = 1
            if e_hit == 1:
                weight = 3
            elif e_hit >= 2:
                weight = 2
            options.extend([(e_hit, extra_coll, unique)] * weight)
    return rng.choice(options) if options else None


def _place_collision_event(
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    dest: tuple[int, int],
    pair: list[dict],
) -> None:
    reserved_bg.add(dest)
    for placement in pair:
        _add_sprite_state(
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "bg",
                "target": dest,
                "used_portal": placement["portal_hops"] > 0,
                "protected": frozenset(placement["bg_path"]),
            },
        )


def _place_unique_landings(
    rng: random.Random,
    count: int,
    placements: list[dict],
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for i in range(count):
        placement = _choose_unique(
            rng,
            placements,
            occupied,
            protected,
            reserved_bg,
            prefer_portal=(i == 0),
            prefer_multi=False,
        )
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        _add_sprite_state(
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "bg",
                "target": placement["dest"],
                "used_portal": placement["portal_hops"] > 0,
                "protected": frozenset(placement["bg_path"]),
            },
        )
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    sprites: list[dict],
    placements_by_bg_cell: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if e_hit == 0:
        return True
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for i, idx in enumerate(hit_targets):
        choice = _choose_hit(
            rng,
            idx,
            sprites,
            placements_by_bg_cell,
            occupied,
            protected,
            reserved_bg,
            cell_to_sprite,
            prefer_portal=(i == 0),
            prefer_multi=(i == 0),
        )
        if choice is None:
            return False
        placement, stop_idx, stop_cell = choice
        _add_sprite_state(
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "hit",
                "target": stop_cell,
                "target_sprite": idx,
                "used_portal": placement["portal_hops"] > 0,
                "protected": frozenset(placement["bg_path"][:stop_idx]),
            },
        )
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _random_portal_pairs(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    while True:
        cells = rng.sample(PORTAL_INTERIOR, 2 * PORTAL_PAIR_COUNT)
        if any(_chebyshev(a, b) < 4 for i, a in enumerate(cells) for b in cells[i + 1 :]):
            continue
        return [(cells[0], cells[1]), (cells[2], cells[3])]


def _event_summary(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, cell_to_comp, portal_map = _find_triangles_and_portals(g, bg)
    triangle_cells = set(cell_to_comp)

    portal_traces: list[dict] = []
    plain_traces: list[dict] = []
    for cid, comp in enumerate(triangles):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        trace_portal = _trace_ray_on_grid(ad[0], ad[1], portal_map, triangle_cells, g.shape)
        trace_plain = _trace_ray_on_grid(ad[0], ad[1], {}, triangle_cells, g.shape)
        if trace_portal is not None:
            portal_traces.append({"cid": cid, **trace_portal})
        if trace_plain is not None:
            plain_traces.append({"cid": cid, **trace_plain})

    portal_by_id = {t["cid"]: t for t in portal_traces}
    plain_by_id = {t["cid"]: t for t in plain_traces}
    shared_ids = sorted(set(portal_by_id) & set(plain_by_id))

    portal_bg = Counter(t["dest"] for t in portal_traces if t["dest"] not in triangle_cells)
    plain_bg = Counter(t["dest"] for t in plain_traces if t["dest"] not in triangle_cells)
    portal_hits = {cell_to_comp[t["dest"]] for t in portal_traces if t["dest"] in triangle_cells}

    return {
        "portal_users": sum(1 for t in portal_traces if t["portal_hops"] >= 1),
        "multi_portal_users": sum(1 for t in portal_traces if t["portal_hops"] >= 2),
        "max_portal_hops": max((t["portal_hops"] for t in portal_traces), default=0),
        "portal_deviations": sum(1 for cid in shared_ids if portal_by_id[cid]["dest"] != plain_by_id[cid]["dest"]),
        "collision_points": sum(1 for count in portal_bg.values() if count >= 2),
        "portal_only_collision_points": sum(
            1 for dest, count in portal_bg.items() if count >= 2 and plain_bg.get(dest, 0) < 2
        ),
        "hit_components": len(portal_hits),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    portal_color_choices = [c for c in PORTAL_COLORS if c != sprite_color]
    portal_colors = rng.sample(portal_color_choices, PORTAL_PAIR_COUNT)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, extra_coll, unique_count = counts

    for _ in range(80):
        portal_pairs = _random_portal_pairs(rng)
        portal_map: dict[tuple[int, int], tuple[int, int]] = {}
        for a, b in portal_pairs:
            portal_map[a] = b
            portal_map[b] = a

        placements, placements_by_dest, placements_by_bg_cell = _build_portal_placements(portal_map)
        if sum(1 for p in placements if p["portal_hops"] >= 1) < 20:
            continue
        if sum(1 for p in placements if p["portal_hops"] >= 2) < 6:
            continue

        sprites: list[dict] = []
        portal_cells = set(portal_map)
        occupied: set[tuple[int, int]] = set(portal_cells)
        protected: set[tuple[int, int]] = set()
        cell_to_sprite: dict[tuple[int, int], int] = {}
        reserved_bg: set[tuple[int, int]] = set()

        mandatory = _choose_collision_pair(
            rng,
            placements_by_dest,
            occupied,
            protected,
            reserved_bg,
            require_portal=True,
            require_multi=True,
        )
        if mandatory is None:
            continue
        mandatory_dest, mandatory_pair = mandatory
        _place_collision_event(
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            reserved_bg,
            mandatory_dest,
            mandatory_pair,
        )

        for _extra in range(extra_coll):
            choice = _choose_collision_pair(
                rng,
                placements_by_dest,
                occupied,
                protected,
                reserved_bg,
                require_portal=False,
                require_multi=False,
            )
            if choice is None:
                break
            dest, pair = choice
            _place_collision_event(
                sprites,
                occupied,
                protected,
                cell_to_sprite,
                reserved_bg,
                dest,
                pair,
            )
        if len(sprites) != 2 + 2 * extra_coll:
            continue

        base_needed = max(0, e_hit - len(sprites))
        if base_needed > unique_count:
            continue
        if not _place_unique_landings(
            rng,
            base_needed,
            placements,
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            reserved_bg,
        ):
            continue

        if not _place_hits(
            rng,
            e_hit,
            sprites,
            placements_by_bg_cell,
            occupied,
            protected,
            cell_to_sprite,
            reserved_bg,
        ):
            continue

        remaining = n_sprites - len(sprites)
        if remaining < 0:
            continue
        if not _place_unique_landings(
            rng,
            remaining,
            placements,
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            reserved_bg,
        ):
            continue
        if len(sprites) != n_sprites:
            continue

        inp = np.full((H, W), BG, dtype=int)
        for color, (a, b) in zip(portal_colors, portal_pairs):
            inp[a] = color
            inp[b] = color
        for sprite in sprites:
            _paint_sprite(inp, sprite["placement"], sprite_color)

        if not _validate_input(inp, n_sprites):
            continue

        out = _render_output(inp, sprites)
        if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
            continue
        if not np.any(out == HIT_COLOR):
            continue
        if not np.any(out == COLLISION_COLOR):
            continue
        if np.array_equal(inp, out):
            continue

        try:
            solved = main(inp)
            plain_ray = _solve_ray(inp, use_portals=False)
            fixed5 = _solve_fixed5(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        if np.array_equal(solved, plain_ray):
            continue
        if np.array_equal(solved, fixed5):
            continue

        summary = _event_summary(inp)
        if summary["portal_users"] < 2:
            continue
        if summary["multi_portal_users"] < 1:
            continue
        if summary["portal_deviations"] < 2:
            continue
        if summary["collision_points"] < 1:
            continue
        if summary["portal_only_collision_points"] < 1:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
