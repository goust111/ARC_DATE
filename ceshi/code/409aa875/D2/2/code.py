# Variant of 409aa875
# Dimension: D2 — 投射距离 (path)
# Change: 固定距离5格（apex + 5 * direction）... → 射线持续到碰撞或边界（无固定距离）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# a) Concrete behavior:
#    The input still contains only 3-pixel monochrome triangle sprites on a 16x16 grid with bg=7.
#    L-shapes still fire diagonally; V-shapes still fire axis-aligned. The only changed dimension is
#    how far the projectile travels. Instead of "apex + 5*dir", the projectile now steps cell by cell
#    from apex+dir until one of two stopping conditions:
#    - the first non-bg sprite cell on that ray is reached -> that whole sprite is recolored to 9
#    - no sprite is reached before leaving the grid -> the projectile ends on the last in-bounds bg cell
#      on that ray, which is always a boundary cell; lone landings paint 9, shared landings paint 1
#    Example: an up-left L with apex=(9,9) used to mark only (4,4). In the variant it scans
#    (8,8)->(7,7)->... . If a sprite cell appears first at (6,6), the hit sprite is recolored there.
#    If the whole diagonal is empty, the landing shifts all the way to the boundary, e.g. (0,0).
#
# b) Complexity calibration:
#    The variant must be visibly different from the original fixed-distance task, so the generator
#    rejects inputs whose ray-cast output equals the fixed-5 output. Rays are still limited to the same
#    8 direction classes, so the task keeps a small rule set and does not introduce new object types.
#
# c) Hidden assumption audit:
#    - `_triangle_apex_and_dir(...)` depends only on local 3-pixel geometry and remains valid.
#    - The original `rr = ar + 5*dr`, `cc = ac + 5*dc` hard-coded the old D2 value; this must be
#      replaced with a full scan along the ray.
#    - The old "dest in cell_to_comp" hit test is still correct, but only after the new destination is
#      defined as the first sprite cell on the ray or the last in-bounds bg cell.
#    - The original generator precomputed one destination per placement. That assumption breaks entirely:
#      validity now depends on every cell on the ray, not just one endpoint. The new generator therefore
#      tracks a protected prefix for every placed shooter so later sprites cannot accidentally block it.
#
# d) Placement legality:
#    Sprites still obey Chebyshev>=2 separation by halo checks, stay fully in bounds, and all generated
#    rays start with at least one in-bounds step. For bg-boundary events, the entire ray must remain free
#    of sprites. For sprite-hit events, the ray prefix before the intended hit must remain free.
#
# e) Visual observability:
#    Boundary markers make the path-length change readable: some outputs place marks at edges/corners or
#    recolor sprites much sooner/later than 5 cells. The generator enforces at least one non-5 path and
#    also enforces `ray_output != fixed5_output` on the same input, so the variant is not a silent clone.
#
# f) Visual cleanliness:
#    Only sprite bodies and final impact cells are painted; no full ray trails are drawn. The old clean
#    visual style is preserved, and the new placement logic avoids dense overlaps that would make the
#    stopping rule unreadable.
#
# g) Color inferability:
#    Colors are unchanged from the source task: 9 marks lone impacts and hit sprites, 1 marks shared bg
#    landings. No invisible ordering by color id is introduced.


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
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _trace_ray(g: np.ndarray, bg: int, apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], int] | None:
    h, w = g.shape
    dr, dc = direction
    r = apex[0] + dr
    c = apex[1] + dc
    if not (0 <= r < h and 0 <= c < w):
        return None

    steps = 0
    last = None
    while 0 <= r < h and 0 <= c < w:
        steps += 1
        last = (int(r), int(c))
        if int(g[r, c]) != int(bg):
            return last, steps
        r += dr
        c += dc
    return last, steps


def _solve(grid: np.ndarray, *, ray_cast: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps, cell_to_comp = _extract_components(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        if ray_cast:
            traced = _trace_ray(g, bg, (ar, ac), (dr, dc))
            if traced is None:
                continue
            dest, _steps = traced
        else:
            rr = ar + 5 * dr
            cc = ac + 5 * dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            dest = (int(rr), int(cc))
        dest_to_shooters[dest].append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


def _event_summary(grid: np.ndarray, *, ray_cast: bool) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, cell_to_comp = _extract_components(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    path_lengths: list[int] = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        if ray_cast:
            traced = _trace_ray(g, bg, (ar, ac), (dr, dc))
            if traced is None:
                continue
            dest, steps = traced
            path_lengths.append(int(steps))
        else:
            rr = ar + 5 * dr
            cc = ac + 5 * dc
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            dest = (int(rr), int(cc))
            path_lengths.append(5)
        dest_to_shooters[dest].append(cid)

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}
    bg_counter = Counter(
        dest for dest, shooters in dest_to_shooters.items() if dest not in cell_to_comp for _ in shooters
    )
    return {
        "path_lengths": path_lengths,
        "non5_paths": sum(1 for d in path_lengths if d != 5),
        "hit_components": len(hit_components),
        "bg_landings": sum(bg_counter.values()),
        "collision_points": sum(1 for count in bg_counter.values() if count >= 2),
        "destinations": dict(dest_to_shooters),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from 3-pixel triangle sprites and color only their stopping cells or hit sprites.

    Input structure:
    - bg = most common color. Non-bg cells form 3-pixel monochrome sprites, each either an axis-aligned L or a diagonal V.

    Rule:
    - Each sprite still determines an apex and an emission direction from its geometry: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite their opening.
    - Instead of stopping 5 cells away, the projectile now advances cell by cell from the apex until it either meets the first sprite on that ray or reaches the grid boundary.
    - A ray that hits a sprite recolors that entire struck sprite to maroon (9).
    - A ray that reaches the boundary without hitting a sprite lands on the last in-bounds background cell; shared empty landing cells become blue (1), while lone empty landings become maroon (9).

    Key insight: The sprite orientation still tells you the shooting direction, but the final marked cell now depends on the first obstruction along the full ray, not on a fixed offset.
    """
    return _solve(np.asarray(grid, dtype=int), ray_cast=True)


# --------------- generator ---------------

H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 400


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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_endpoint: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_ray_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
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
                    "ray_set": frozenset(ray),
                    "endpoint": ray[-1],
                }
                placements.append(placement)
                by_endpoint[placement["endpoint"]].append(placement)
                for cell in ray:
                    by_ray_cell[cell].append(placement)
    return placements, by_endpoint, by_ray_cell


PLACEMENTS, PLACEMENTS_BY_ENDPOINT, PLACEMENTS_BY_RAY_CELL = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], protected: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & protected)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _ray_clear_to_boundary(placement: dict, occupied: set[tuple[int, int]]) -> bool:
    return not (placement["ray_set"] & occupied)


def _first_hit_on_ray(
    placement: dict, cell_to_sprite: dict[tuple[int, int], int]
) -> tuple[int, tuple[int, int], int] | None:
    for idx, cell in enumerate(placement["ray"]):
        if cell in cell_to_sprite:
            return idx, cell, int(cell_to_sprite[cell])
    return None


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


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    endpoints = list(PLACEMENTS_BY_ENDPOINT)
    rng.shuffle(endpoints)
    for endpoint in endpoints:
        if endpoint in reserved_bg or endpoint in occupied:
            continue
        candidates = PLACEMENTS_BY_ENDPOINT[endpoint][:]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            if not _fits(first, occupied, protected):
                continue
            if not _ray_clear_to_boundary(first, occupied):
                continue
            occ2 = occupied | set(first["cells"])
            prot2 = protected | set(first["ray"])
            for second in candidates[i + 1 :]:
                if not _fits(second, occ2, prot2):
                    continue
                if not _ray_clear_to_boundary(second, occ2):
                    continue
                return endpoint, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["endpoint"] in reserved_bg or placement["endpoint"] in occupied:
            continue
        if not _fits(placement, occupied, protected):
            continue
        if _ray_clear_to_boundary(placement, occupied):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_idx: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, int, tuple[int, int]] | None:
    target_cells = list(sprites[target_idx]["placement"]["cells"])
    rng.shuffle(target_cells)

    seen_ids: set[int] = set()
    candidates: list[dict] = []
    for cell in target_cells:
        for placement in PLACEMENTS_BY_RAY_CELL[cell]:
            pid = id(placement)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            candidates.append(placement)
    rng.shuffle(candidates)

    for placement in candidates:
        if not _fits(placement, occupied, protected):
            continue
        hit = _first_hit_on_ray(placement, cell_to_sprite)
        if hit is None:
            continue
        stop_idx, stop_cell, hit_idx = hit
        if hit_idx != target_idx or stop_cell in used_target_cells:
            continue
        return placement, stop_idx, stop_cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(
    rng: random.Random,
    e_coll: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, protected, reserved_bg)
        if choice is None:
            return False
        endpoint, pair = choice
        reserved_bg.add(endpoint)
        for placement in pair:
            _add_sprite_state(
                sprites,
                occupied,
                protected,
                cell_to_sprite,
                {
                    "placement": placement,
                    "target_kind": "bg",
                    "target": endpoint,
                    "protected": frozenset(placement["ray"]),
                },
            )
    return True


def _place_unique_landings(
    rng: random.Random,
    count: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        placement = _choose_unique(rng, occupied, protected, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["endpoint"])
        _add_sprite_state(
            sprites,
            occupied,
            protected,
            cell_to_sprite,
            {
                "placement": placement,
                "target_kind": "bg",
                "target": placement["endpoint"],
                "protected": frozenset(placement["ray"]),
            },
        )
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    cell_to_sprite: dict[tuple[int, int], int],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, idx, sprites, occupied, protected, cell_to_sprite, used_target_cells)
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
                "target_sprite": idx,
                "target": stop_cell,
                "protected": frozenset(placement["ray"][:stop_idx]),
            },
        )
        used_target_cells.add(stop_cell)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, protected, cell_to_sprite, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, protected, cell_to_sprite, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, protected, cell_to_sprite):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(rng, remaining, sprites, occupied, protected, cell_to_sprite, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    try:
        ray_out = main(inp)
        fixed_out = _solve(inp, ray_cast=False)
    except Exception:
        return None

    if np.array_equal(inp, ray_out):
        return None
    if np.count_nonzero(ray_out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not (np.any(ray_out == HIT_COLOR) or np.any(ray_out == COLLISION_COLOR)):
        return None
    if np.array_equal(ray_out, fixed_out):
        return None

    summary = _event_summary(inp, ray_cast=True)
    if summary["non5_paths"] < 1:
        return None
    if summary["bg_landings"] < 1:
        return None
    if summary["hit_components"] + summary["collision_points"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
