# Variant of 409aa875
# Dimension: D5 — 碰撞规则（投射物-投射物） (transform)
# Change: 两枚以上投射物落在同一空背景格时涂蓝(1)，单枚涂栗色(9)... → 碰撞点不着色，但两枚投射物各自偏转90度继续飞行

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   concretization:
#     - The 16x16 canvas, bg=7, and the 3-pixel triangle grammar remain unchanged.
#     - Every sprite still emits toward apex + 5*dir, so the first interaction point is exactly
#       the same cell the original task would have used.
#     - The only changed case is: multiple projectiles reaching the same EMPTY bg cell at that
#       first landing time. That cell is now a turning hub rather than a painted mark.
#     - To make the variant deterministic and learnable, every projectile in such a hub rotates
#       90 degrees clockwise. After that, it advances one cell per tick until it exits the grid
#       or enters a sprite. Unique bounced steps paint bg as 9; same-tick bounced collisions are
#       handled again in the same way (cell stays bg, directions rotate clockwise again).
#   behavior_trace:
#     - Original local trace:
#         shooter A -> X, shooter B -> X, X is bg  => output[X] = 1 and both projectiles stop.
#     - Variant local trace:
#         shooter A -> X, shooter B -> X, X is bg  => output[X] stays 7,
#         A turns clockwise, B turns clockwise, then both keep moving.
#         Example: A arrives from up with dir=(1,0) so it leaves right with (0,-1) in row/col
#         coordinates? No: clockwise on (dr,dc) is (dc,-dr), so (1,0)->(0,-1), (-1,0)->(0,1),
#         (0,1)->(1,0), (1,1)->(1,-1), etc. The visible change is the maroon trail after X.
#   complexity_calibration:
#     - A fixed clockwise rule avoids an underconstrained "left or right?" ambiguity.
#     - The generator always includes at least one collision hub and requires visible bounce
#       trails, so the new mechanism is observable rather than degenerating to the old task.
#     - Sprite color 9 is excluded in generation; otherwise recolored sprites and bounce trails
#       could become invisible and the D5 change would be harder to infer from 2-4 examples.
#   hidden_assumptions_review:
#     - The original main() grouped destinations once and terminated every projectile there.
#       That relies on "collision means final paint event"; it is false in this variant.
#     - The original "if len(shooters) >= 2: out[dest] = 1" would now be wrong twice:
#       it paints the hub, and it discards the future motion state.
#     - The original one-pass destination accounting ignored time. Here, post-collision motion
#       must be simulated synchronously so that later same-tick meetings can bounce again.
#   placement_legality:
#     - The generator still uses in-bounds apex+dir placements and Chebyshev>=2 sprite spacing.
#     - For every forced collision pair, it also reserves the future clockwise rays as bg cells,
#       so later sprites cannot occupy those cells and accidentally hide the bounce evidence.
#     - The solver assumes components are still 3-pixel monochrome triangles; the generator
#       guarantees that exactly as in the base task.
#   visual_observability:
#     - Every generated grid has at least one collision hub, and each bounced projectile must
#       contribute at least two post-turn steps. This makes "hub stays blank, trail appears"
#       visible by inspection.
#     - The generator keeps the rest of the task grammar familiar: some samples still include
#       lone maroon landing dots or sprite recolors, so the user can isolate the changed branch.
#   cleanliness_and_color_inference:
#     - Only one new output color role remains: 9 marks lone landings, hit sprites, and bounce
#       trails. No hidden color ordering or id-based rule is introduced.
#     - Collision hubs are intentionally left in bg, which visually separates "meeting points"
#       from "painted path" and reduces output clutter.


H = W = 16
BG = 7
HIT_COLOR = 9
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MIN_BOUNCE_LEN = 2
MIN_TOTAL_BOUNCE_CELLS = 4
MAX_ATTEMPTS = 400


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


def _rotate_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], h: int = H, w: int = W) -> tuple[tuple[int, int], ...]:
    r, c = start
    dr, dc = direction
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
    return tuple(cells)


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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                bounce_dir = _rotate_cw((dr, dc))
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "bounce_dir": bounce_dir,
                    "bounce_path": _ray_cells(dest, bounce_dir),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _simulate_variant(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    shooters = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), direction = ad
        rr = ar + PROJECTILE_DIST * direction[0]
        cc = ac + PROJECTILE_DIST * direction[1]
        if 0 <= rr < h and 0 <= cc < w:
            shooters.append({"cid": cid, "dir": direction, "dest": (rr, cc)})

    out = g.copy()
    hit_components: set[int] = set()
    initial_bg_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for shooter in shooters:
        dest = shooter["dest"]
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])
        else:
            initial_bg_groups[dest].append(shooter)

    active: list[dict] = []
    initial_collision_cells: list[tuple[int, int]] = []
    lone_bg_landings = 0
    initial_bounced_projectiles = 0
    for dest, group in initial_bg_groups.items():
        if len(group) == 1:
            out[dest] = HIT_COLOR
            lone_bg_landings += 1
        else:
            initial_collision_cells.append(dest)
            initial_bounced_projectiles += len(group)
            for shooter in group:
                active.append({"pos": dest, "dir": _rotate_cw(shooter["dir"])})

    bounce_trail_cells: set[tuple[int, int]] = set()
    secondary_collision_cells: list[tuple[int, int]] = []
    bounce_sprite_hits = 0
    seen_states: set[tuple[tuple[int, int, int, int], ...]] = set()

    for _ in range(h * w * 8):
        if not active:
            break
        state = tuple(sorted((p["pos"][0], p["pos"][1], p["dir"][0], p["dir"][1]) for p in active))
        if state in seen_states:
            break
        seen_states.add(state)

        arrivals: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for proj in active:
            nr = proj["pos"][0] + proj["dir"][0]
            nc = proj["pos"][1] + proj["dir"][1]
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            pos = (nr, nc)
            if pos in cell_to_comp:
                hit_components.add(cell_to_comp[pos])
                bounce_sprite_hits += 1
                continue
            arrivals[pos].append({"pos": pos, "dir": proj["dir"]})

        active = []
        for pos, group in arrivals.items():
            if len(group) == 1:
                out[pos] = HIT_COLOR
                bounce_trail_cells.add(pos)
                active.append(group[0])
            else:
                secondary_collision_cells.append(pos)
                for proj in group:
                    active.append({"pos": pos, "dir": _rotate_cw(proj["dir"])})

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    metrics = {
        "sprite_count": len(comps),
        "shooter_count": len(shooters),
        "initial_collision_cells": tuple(initial_collision_cells),
        "initial_collision_count": len(initial_collision_cells),
        "bounced_projectiles": initial_bounced_projectiles,
        "bounce_trail_cells": len(bounce_trail_cells),
        "secondary_collision_count": len(secondary_collision_cells),
        "lone_bg_landings": lone_bg_landings,
        "hit_component_count": len(hit_components),
        "bounce_sprite_hits": bounce_sprite_hits,
    }
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangle sprites still aim at a cell 5 steps away, but shared empty landings now redirect motion instead of painting a blue collision dot.

    Input structure:
    - The background is the most common color. Every non-background 8-connected component is a 3-pixel monochrome triangle sprite.
    - L-shapes use their 4-neighbor corner as the apex and fire diagonally away from the missing corner; diagonal V-shapes use their 8-neighbor apex and fire along the axis opposite the opening.

    Rule:
    - Each sprite first sends one projectile to apex + 5*dir, exactly as in the base task.
    - If that landing cell is on a sprite, the whole hit sprite is recolored to 9. If exactly one projectile lands on empty background, only that landing cell becomes 9.
    - If multiple projectiles land on the same empty background cell at that first landing time, the hub cell stays background. Each of those projectiles rotates 90 degrees clockwise and then advances one cell per tick, painting visited background cells 9 until it exits the grid or strikes a sprite.
    - If bounced projectiles later reach the same background cell in the same tick, that cell also remains background and the same clockwise-turn rule applies again.

    Key insight: Sprite orientation still determines the first rendezvous point, but a shared empty landing is no longer the final output mark; it is a right-angle redirection hub whose outgoing maroon trail reveals the new rule.
    """
    return _simulate_variant(grid)[0]


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict], set[tuple[int, int]]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied or dest in reserved_bg:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            trail1 = set(first["bounce_path"])
            if len(trail1) < MIN_BOUNCE_LEN:
                continue
            if trail1 & (occupied | reserved_bg | set(first["cells"])):
                continue
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1:]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                trail2 = set(second["bounce_path"])
                if len(trail2) < MIN_BOUNCE_LEN:
                    continue
                if trail2 & (occupied | reserved_bg | occ2 | set(second["cells"])):
                    continue
                if trail1 & trail2:
                    continue
                return dest, [first, second], trail1 | trail2
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in occupied or placement["dest"] in reserved_bg:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for placement in candidates:
            return placement, cell
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
    for e_coll in range(1, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
        max_hits = min(MAX_HIT_EVENTS, n_sprites - 2 * e_coll)
        for e_hit in range(max_hits + 1):
            unique = n_sprites - 2 * e_coll - e_hit
            if unique < 0:
                continue
            if unique + e_hit < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair, trail_cells = choice
        reserved_bg.add(dest)
        reserved_bg.update(trail_cells)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "collision"})
            occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    if e_hit > len(sprites):
        return False
    target_ids = rng.sample(range(len(sprites)), e_hit)
    for idx in target_ids:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
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
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining != unique_total:
        return None
    if not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out, metrics = _simulate_variant(inp)
    collision_cells = metrics["initial_collision_cells"]
    if metrics["initial_collision_count"] < 1:
        return None
    if metrics["bounce_trail_cells"] < MIN_TOTAL_BOUNCE_CELLS:
        return None
    if any(out[cell] != BG for cell in collision_cells):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not any(out[dest] == BG for dest in collision_cells):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
