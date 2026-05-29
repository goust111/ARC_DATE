# Combo variant of 409aa875 (index 5)
# combo_spec: D2/2+D13/0
# Name: 射线到碰撞/边界+静态墙壁

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# a) Concrete behavior and trace
#    - Keep the original 16x16 orange background (7) and the same 3-pixel triangle sprites.
#    - Add fixed grey wall pixels (5). Walls are not sprites and never recolor.
#    - A sprite still determines its apex and shooting direction from geometry, but the shot no longer
#      uses a fixed endpoint. It now scans the whole ray from apex+dir onward and stops at the first
#      obstruction: sprite, wall, or boundary exit. Hitting a sprite recolors that whole sprite to 9.
#      Hitting a wall means the visible landing cell is the last free cell before that wall. Reaching
#      the edge with no obstruction lands on the final in-bounds boundary cell.
#    - Small trace: an L with apex=(10,10) firing up-left used to mark only (5,5). In the combo
#      variant it scans (9,9),(8,8),... . If the first wall sits at (6,6), the mark moves to (7,7).
#      If no wall exists and the first sprite appears at (3,3), that sprite is recolored instead.
#      If neither exists, the landing becomes the boundary corner (0,0).
#
# b) Complexity calibration
#    - Only one new object type is introduced: isolated wall pixels in one fixed color. The direction
#      vocabulary stays the original 8 orientations, so the rule is still inferable from a few samples.
#    - The generator forces both dimensions to matter in every sample: at least one shot must be cut by
#      a wall, and at least one shot must travel beyond 5 steps to a later stop. It also rejects cases
#      whose output matches either single-dimension ablation (`ray without walls` or `5-step with walls`).
#
# c) Hidden assumption audit
#    - Original code assumed every non-bg component was a sprite. That breaks once walls are added, so
#      component extraction must exclude wall color.
#    - Original code also assumed a single arithmetic jump `apex + 5*dir`. That breaks under ray-cast
#      stopping, so destination calculation must become a stepwise scan.
#    - The old hit test `dest in cell_to_comp` is still valid only if `dest` is redefined as the first
#      sprite cell actually reached on the ray; wall cells themselves are never hit targets.
#    - Generator-side, the original "one precomputed endpoint per placement" assumption fails because
#      later sprites or walls can alter earlier shots. The combo generator therefore protects every ray
#      prefix that must stay empty and checks that each newly added wall preserves all earlier plans.
#
# d) Placement legality
#    - Sprites remain fully in bounds and pairwise separated by Chebyshev distance >= 2 via halo tests.
#    - Walls are single pixels, kept off sprite halos and off other wall halos so they stay visually
#      separate from sprites and from one another.
#    - Unblocked background shots reserve their full ray to the boundary. Hit shots reserve only the
#      prefix before their intended hit. Blocked shots reserve the prefix before their wall and claim
#      one specific wall cell; candidate walls are rejected if they would shorten any existing shot.
#
# e) Visual observability
#    - The combo should be readable by eye: some marks sit immediately before visible grey walls, while
#      other shots clearly pass much farther than 5 cells or recolor sprites deeper along the ray.
#    - Because the generator requires output differences versus both ablations, a solver cannot succeed
#      by learning only wall-shortening or only boundary ray-casting in isolation.
#
# f) Visual cleanliness
#    - Only sprite bodies, wall pixels, and final stopping cells appear. No ray trails are painted.
#    - Wall isolation and sprite halo spacing keep the grid from turning into ambiguous clumps where a
#      human cannot tell whether a stop came from a wall, a sprite collision, or adjacency noise.
#
# g) Change inferability
#    - Colors stay minimal and structured: sprite color is a single shared non-special color; walls are
#      always grey (5); lone landings and hit sprites are maroon (9); shared bg landings are blue (1).
#    - Sprite colors exclude 1, 5, 7, and 9 so the new wall color and the output semantics remain easy
#      to distinguish from the background and from the original sprite bodies.


H = W = 16
BG = 7
WALL_COLOR = 5
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 6, 8]

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_BLOCKED_UNIQUES = 2
MAX_ATTEMPTS = 800


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
        dr = -dr_sum
        dc = -dc_sum
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
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


def _extract_components(g: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg, WALL_COLOR}):
        for comp in _cc8(g == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _trace_shot(
    apex: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    cell_to_comp: dict[tuple[int, int], int],
    walls: set[tuple[int, int]],
    max_steps: int | None,
) -> dict | None:
    h, w = shape
    dr, dc = direction
    r = apex[0] + dr
    c = apex[1] + dc
    if not (0 <= r < h and 0 <= c < w):
        return None

    prev = apex
    steps = 0
    while 0 <= r < h and 0 <= c < w and (max_steps is None or steps < max_steps):
        steps += 1
        cell = (int(r), int(c))
        if cell in walls:
            return {
                "dest": prev,
                "steps": steps - 1,
                "stop": "wall",
                "cell": cell,
                "hit_comp": None,
            }
        if cell in cell_to_comp:
            return {
                "dest": cell,
                "steps": steps,
                "stop": "sprite",
                "cell": cell,
                "hit_comp": int(cell_to_comp[cell]),
            }
        prev = cell
        r += dr
        c += dc

    if prev == apex:
        return None
    if max_steps is not None and steps == max_steps and 0 <= r < h and 0 <= c < w:
        stop = "limit"
    else:
        stop = "boundary"
    return {"dest": prev, "steps": steps, "stop": stop, "cell": prev, "hit_comp": None}


def _collect_shot_events(grid: np.ndarray, *, use_walls: bool, max_steps: int | None) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    walls = {(int(r), int(c)) for r, c in zip(*np.where(g == WALL_COLOR))} if use_walls else set()
    comps, cell_to_comp = _extract_components(g, bg)

    events: list[dict] = []
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        trace = _trace_shot(ad[0], ad[1], g.shape, cell_to_comp, walls, max_steps)
        if trace is None:
            continue
        event = {
            "shooter": cid,
            "dest": trace["dest"],
            "steps": int(trace["steps"]),
            "stop": trace["stop"],
            "hit_comp": trace["hit_comp"],
        }
        events.append(event)
        dest_to_shooters[event["dest"]].append(cid)

    return {
        "grid": g,
        "bg": bg,
        "walls": walls,
        "comps": comps,
        "cell_to_comp": cell_to_comp,
        "events": events,
        "dest_to_shooters": dest_to_shooters,
    }


def _solve_grid(grid: np.ndarray, *, use_walls: bool, max_steps: int | None) -> np.ndarray:
    info = _collect_shot_events(grid, use_walls=use_walls, max_steps=max_steps)
    g = info["grid"]
    out = g.copy()

    hit_components = {event["hit_comp"] for event in info["events"] if event["hit_comp"] is not None}
    for cid in hit_components:
        for r, c in info["comps"][cid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in info["dest_to_shooters"].items():
        if dest in info["cell_to_comp"]:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _behavior_metrics(grid: np.ndarray) -> dict:
    combo = _collect_shot_events(grid, use_walls=True, max_steps=None)
    ray_only = _collect_shot_events(grid, use_walls=False, max_steps=None)
    wall_only = _collect_shot_events(grid, use_walls=True, max_steps=5)
    combo_out = _solve_grid(grid, use_walls=True, max_steps=None)
    ray_out = _solve_grid(grid, use_walls=False, max_steps=None)
    wall_out = _solve_grid(grid, use_walls=True, max_steps=5)

    return {
        "wall_stops": sum(event["stop"] == "wall" for event in combo["events"]),
        "boundary_stops": sum(event["stop"] == "boundary" for event in combo["events"]),
        "sprite_stops": sum(event["stop"] == "sprite" for event in combo["events"]),
        "non5_paths": sum(event["steps"] != 5 for event in combo["events"]),
        "long_paths": sum(event["steps"] > 5 for event in combo["events"]),
        "hit_components": len({event["hit_comp"] for event in combo["events"] if event["hit_comp"] is not None}),
        "collision_points": sum(
            1 for dest, shooters in combo["dest_to_shooters"].items() if dest not in combo["cell_to_comp"] and len(shooters) >= 2
        ),
        "combo_vs_ray_cells": int(np.count_nonzero(combo_out != ray_out)),
        "combo_vs_wall_cells": int(np.count_nonzero(combo_out != wall_out)),
        "ray_only_wall_stops": sum(event["stop"] == "wall" for event in ray_only["events"]),
        "wall_only_limit_stops": sum(event["stop"] == "limit" for event in wall_only["events"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast sprite-defined rays through the grid and stop them at sprites, walls, or the boundary.

    Input structure:
    - The background is the most common color.
    - Non-background, non-grey triangle sprites are still 3-pixel monochrome L-shapes or diagonal V-shapes.
    - Grey cells (color 5) are static walls: they are neither shooters nor targets, but they block rays.

    Rule:
    - Each triangle still determines an apex and emission direction from its geometry: L-shapes fire diagonally away from the missing 2x2 corner, and V-shapes fire along the axis opposite the opening.
    - From the apex, advance one cell at a time along that direction until the ray first reaches another sprite, a wall, or the grid edge.
    - If the first obstruction is a sprite, recolor the entire struck sprite to maroon (9).
    - If the first obstruction is a wall, the projectile lands on the cell immediately before that wall; if no obstruction appears before leaving the grid, it lands on the last in-bounds boundary cell. Shared empty landing cells become blue (1), while lone empty landing cells become maroon (9).

    Key insight: Orientation still tells you which way each triangle fires, but the visible result now depends on the first thing encountered anywhere on that full ray, with grey walls shortening some shots before they can reach distant sprites or the boundary.
    """
    return _solve_grid(np.asarray(grid, dtype=int), use_walls=True, max_steps=None)


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


def _cell_halo(cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r, c = cell
    return frozenset(
        (rr, cc)
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _ray_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    r = ar + dr
    c = ac + dc
    cells = []
    while 0 <= r < H and 0 <= c < W:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return tuple(cells)


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[tuple[int, int], list[dict]], list[dict]]:
    placements: list[dict] = []
    by_endpoint: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_ray_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    blocked_options: list[dict] = []
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                ray = _ray_cells((ar, ac), (dr, dc))
                if not ray:
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "ray": ray,
                    "endpoint": ray[-1],
                }
                placements.append(placement)
                by_endpoint[placement["endpoint"]].append(placement)
                for cell in ray:
                    by_ray_cell[cell].append(placement)
                for wall_idx in range(1, len(ray)):
                    blocked_options.append(
                        {
                            "placement": placement,
                            "dest": ray[wall_idx - 1],
                            "wall": ray[wall_idx],
                            "wall_idx": wall_idx,
                            "wall_halo": _cell_halo(ray[wall_idx]),
                        }
                    )
    return placements, by_endpoint, by_ray_cell, blocked_options


PLACEMENTS, PLACEMENTS_BY_ENDPOINT, PLACEMENTS_BY_RAY_CELL, BLOCKED_OPTIONS = _build_placements()


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    blocked = occupied | walls
    forbidden = protected | reserved_bg | walls
    return not (placement["halo"] & blocked) and not (placement["cell_set"] & forbidden)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _first_obstruction(
    placement: dict,
    cell_to_sprite: dict[tuple[int, int], int],
    walls: set[tuple[int, int]],
) -> dict:
    for idx, cell in enumerate(placement["ray"]):
        if cell in walls:
            dest = placement["apex"] if idx == 0 else placement["ray"][idx - 1]
            return {"stop": "wall", "idx": idx, "cell": cell, "dest": dest}
        if cell in cell_to_sprite:
            return {
                "stop": "sprite",
                "idx": idx,
                "cell": cell,
                "dest": cell,
                "sprite": int(cell_to_sprite[cell]),
            }
    return {"stop": "boundary", "idx": len(placement["ray"]) - 1, "cell": placement["endpoint"], "dest": placement["endpoint"]}


def _add_shot_state(
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    shot: dict,
) -> None:
    idx = len(shots)
    shots.append(shot)
    occupied.update(shot["placement"]["cells"])
    protected.update(shot["protected"])
    for cell in shot["placement"]["cells"]:
        cell_to_sprite[cell] = idx


def _all_shots_stable(
    shots: list[dict],
    cell_to_sprite: dict[tuple[int, int], int],
    walls: set[tuple[int, int]],
) -> bool:
    for shot in shots:
        if _first_obstruction(shot["placement"], cell_to_sprite, walls)["dest"] != shot["target"]:
            return False
    return True


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    endpoints = list(PLACEMENTS_BY_ENDPOINT)
    rng.shuffle(endpoints)
    for endpoint in endpoints:
        if endpoint in reserved_bg or endpoint in occupied or endpoint in walls:
            continue
        candidates = PLACEMENTS_BY_ENDPOINT[endpoint][:]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            if not _fits(first, occupied, protected, reserved_bg, walls):
                continue
            if _first_obstruction(first, cell_to_sprite, walls)["stop"] != "boundary":
                continue
            occ2 = occupied | set(first["cells"])
            prot2 = protected | set(first["ray"])
            cell_to_sprite2 = dict(cell_to_sprite)
            for cell in first["cells"]:
                cell_to_sprite2[cell] = -1
            for second in candidates[i + 1 :]:
                if not _fits(second, occ2, prot2, reserved_bg, walls):
                    continue
                if _first_obstruction(second, cell_to_sprite2, walls)["stop"] != "boundary":
                    continue
                return endpoint, [first, second]
    return None


def _choose_unblocked_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        endpoint = placement["endpoint"]
        if endpoint in reserved_bg or endpoint in occupied or endpoint in walls:
            continue
        if not _fits(placement, occupied, protected, reserved_bg, walls):
            continue
        if _first_obstruction(placement, cell_to_sprite, walls)["stop"] == "boundary":
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_idx: int,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, int, tuple[int, int]] | None:
    target_cells = list(shots[target_idx]["placement"]["cells"])
    rng.shuffle(target_cells)

    seen: set[int] = set()
    candidates: list[dict] = []
    for cell in target_cells:
        for placement in PLACEMENTS_BY_RAY_CELL[cell]:
            pid = id(placement)
            if pid in seen:
                continue
            seen.add(pid)
            candidates.append(placement)
    rng.shuffle(candidates)

    for placement in candidates:
        if not _fits(placement, occupied, protected, reserved_bg, walls):
            continue
        obstruction = _first_obstruction(placement, cell_to_sprite, walls)
        if obstruction["stop"] != "sprite":
            continue
        if obstruction["sprite"] != target_idx or obstruction["cell"] in used_target_cells:
            continue
        return placement, obstruction["idx"], obstruction["cell"]
    return None


def _choose_blocked_unique(
    rng: random.Random,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> dict | None:
    candidates = BLOCKED_OPTIONS[:]
    rng.shuffle(candidates)
    for option in candidates:
        placement = option["placement"]
        dest = option["dest"]
        wall = option["wall"]
        if dest in reserved_bg or dest in occupied or dest in walls:
            continue
        if wall in occupied or wall in reserved_bg or wall in walls or wall in protected:
            continue
        if option["wall_halo"] & occupied:
            continue
        if option["wall_halo"] & walls:
            continue
        if not _fits(placement, occupied, protected, reserved_bg, walls):
            continue
        trial_walls = set(walls)
        trial_walls.add(wall)
        obstruction = _first_obstruction(placement, cell_to_sprite, trial_walls)
        if obstruction["stop"] != "wall" or obstruction["cell"] != wall or obstruction["dest"] != dest:
            continue
        if not _all_shots_stable(shots, cell_to_sprite, trial_walls):
            continue
        return {
            "placement": placement,
            "target_kind": "bg",
            "target": dest,
            "blocked": True,
            "wall": wall,
            "protected": frozenset(placement["ray"][: option["wall_idx"]]),
        }
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, n_walls: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    if np.count_nonzero(grid == WALL_COLOR) != n_walls:
        return False
    sprite_colors = sorted(set(int(v) for v in np.unique(grid)) - {BG, WALL_COLOR})
    if len(sprite_colors) != 1 or sprite_colors[0] not in SPRITE_COLORS:
        return False
    comps: list[list[tuple[int, int]]] = []
    for col in sprite_colors:
        comps.extend(_cc8(grid == col))
    if len(comps) != n_sprites:
        return False
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            for blocked_unique in range(1, min(MAX_BLOCKED_UNIQUES, n_sprites - 1) + 1):
                unblocked_unique = n_sprites - e_hit - 2 * e_coll - blocked_unique
                if unblocked_unique < 1:
                    continue
                if e_hit + e_coll < 1:
                    continue
                options.append((e_hit, e_coll, blocked_unique, unblocked_unique))
    return rng.choice(options) if options else None


def _place_collisions(
    rng: random.Random,
    e_coll: int,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, protected, cell_to_sprite, reserved_bg, walls)
        if choice is None:
            return False
        endpoint, pair = choice
        reserved_bg.add(endpoint)
        for placement in pair:
            _add_shot_state(
                shots,
                occupied,
                protected,
                cell_to_sprite,
                {
                    "placement": placement,
                    "target_kind": "bg",
                    "target": endpoint,
                    "blocked": False,
                    "wall": None,
                    "protected": frozenset(placement["ray"]),
                },
            )
    return True


def _place_unblocked_uniques(
    rng: random.Random,
    count: int,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        placement = _choose_unblocked_unique(rng, occupied, protected, cell_to_sprite, reserved_bg, walls)
        if placement is None:
            return False
        reserved_bg.add(placement["endpoint"])
        _add_shot_state(
            shots,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "bg",
                "target": placement["endpoint"],
                "blocked": False,
                "wall": None,
                "protected": frozenset(placement["ray"]),
            },
        )
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(shots)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(
            rng,
            idx,
            shots,
            occupied,
            protected,
            cell_to_sprite,
            reserved_bg,
            walls,
            used_target_cells,
        )
        if choice is None:
            return False
        placement, stop_idx, stop_cell = choice
        _add_shot_state(
            shots,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "hit",
                "target_sprite": idx,
                "target": stop_cell,
                "blocked": False,
                "wall": None,
                "protected": frozenset(placement["ray"][:stop_idx]),
            },
        )
        used_target_cells.add(stop_cell)
    return True


def _place_blocked_uniques(
    rng: random.Random,
    count: int,
    shots: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        shot = _choose_blocked_unique(rng, shots, occupied, protected, cell_to_sprite, reserved_bg, walls)
        if shot is None:
            return False
        reserved_bg.add(shot["target"])
        walls.add(shot["wall"])
        _add_shot_state(shots, occupied, protected, cell_to_sprite, shot)
    return True


def _render_output(inp: np.ndarray, shots: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(shot["target"] for shot in shots if shot["target_kind"] == "bg")
    hit_sprite_ids = {shot["target_sprite"] for shot in shots if shot["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, shots[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, blocked_unique, unblocked_unique = counts

    shots: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    reserved_bg: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, shots, occupied, protected, cell_to_sprite, reserved_bg, walls):
        return None

    base_needed = max(0, e_hit - len(shots))
    if base_needed > unblocked_unique:
        return None
    if not _place_unblocked_uniques(
        rng, base_needed, shots, occupied, protected, cell_to_sprite, reserved_bg, walls
    ):
        return None

    if not _place_hits(rng, e_hit, shots, occupied, protected, cell_to_sprite, reserved_bg, walls):
        return None

    if not _place_blocked_uniques(
        rng, blocked_unique, shots, occupied, protected, cell_to_sprite, reserved_bg, walls
    ):
        return None

    remaining_unblocked = unblocked_unique - base_needed
    if remaining_unblocked < 0:
        return None
    if not _place_unblocked_uniques(
        rng, remaining_unblocked, shots, occupied, protected, cell_to_sprite, reserved_bg, walls
    ):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for r, c in walls:
        inp[r, c] = WALL_COLOR
    for shot in shots:
        _paint_sprite(inp, shot["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, blocked_unique):
        return None

    out = _render_output(inp, shots)
    combo_out = main(inp)
    ray_only_out = _solve_grid(inp, use_walls=False, max_steps=None)
    wall_only_out = _solve_grid(inp, use_walls=True, max_steps=5)
    metrics = _behavior_metrics(inp)

    if not np.array_equal(combo_out, out):
        return None
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(combo_out, ray_only_out):
        return None
    if np.array_equal(combo_out, wall_only_out):
        return None
    if metrics["wall_stops"] < 1:
        return None
    if metrics["boundary_stops"] < 1:
        return None
    if metrics["long_paths"] < 1:
        return None
    if metrics["hit_components"] + metrics["collision_points"] < 1:
        return None
    if metrics["combo_vs_ray_cells"] < 1 or metrics["combo_vs_wall_cells"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
