# Variant of 409aa875
# Dimension: D13 — 障碍物/干扰项 (obstacle)
# Change: 无障碍物或干扰项；网格中只有精灵和背景... → 静态墙壁阻挡投射物

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8

# Design analysis for the D13 variant
# a) Concrete behavior and trace
#    - Keep the original 16x16 orange background (7) and 3-pixel triangle sprites.
#    - Add a second fixed input color: grey wall pixels (5). Walls are static obstacles, not sprites.
#    - A triangle still defines an apex and one of 8 firing directions. The original landing cell is
#      apex + 5*dir, but the new rule scans the ray step-by-step. If the first wall appears at step k,
#      the projectile stops on step k-1 instead of step 5.
#    - Small trace: if a V-shape fires upward from apex (10, 6), the old landing cell is (5, 6).
#      With a wall at (7, 6), the new landing cell becomes (8, 6). The hit/collision logic then runs
#      on (8, 6), not on (5, 6).
# b) Complexity calibration
#    - Wall color is fixed and visually distinct from sprite color. Sprite colors exclude 1, 5, 7, 9,
#      so blue collision marks, maroon hit marks, orange background, and grey walls all stay legible.
#    - Every generated grid contains at least one blocked projectile and at least one unblocked one,
#      so the solver must infer that only some rays are shortened.
# c) Implicit assumptions that break from the original main()
#    - Original component scan treated every non-bg component as a possible sprite. That is no longer
#      safe because walls are also non-bg; the solver must skip wall color when building triangles.
#    - Original landing code used a single arithmetic jump `apex + 5*dir`. That assumes empty rays.
#      The variant invalidates this and requires a stepwise wall check.
#    - Original hit lookup allowed any non-bg destination component. In the variant, walls must not be
#      considered hitable sprites, so `cell_to_comp` is built from sprite cells only.
# d) Placement legality
#    - Generator keeps sprite-sprite Chebyshev separation >= 2 as before.
#    - Walls are single pixels placed off all sprite halos, so they do not merge visually with sprites.
#    - A blocked shot owns one wall pixel on its ray. Before accepting a new wall, the generator checks
#      that it does not accidentally shorten previously placed shots.
# e) Visual observability
#    - Blocked shots create output marks immediately before visible grey walls. Unblocked shots still
#      land at the full 5-cell distance, so a human can compare both cases within the same sample.
# f) Visual cleanliness
#    - Walls are isolated from sprites and from one another by at least one cell, preventing clutter.
#    - No extra decoy mechanics are introduced; the only new visual element is the wall color.
# g) Color-change inference
#    - Output colors keep the original semantics: maroon (9) for a lone bg landing or a hit sprite,
#      blue (1) for >=2 projectiles on the same empty bg cell. Grey (5) never changes role.


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


def _landing_cell(
    apex: tuple[int, int],
    direction: tuple[int, int],
    walls: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    ar, ac = apex
    dr, dc = direction
    prev = (ar, ac)
    for step in range(1, 6):
        cell = (ar + step * dr, ac + step * dc)
        if not (0 <= cell[0] < h and 0 <= cell[1] < w):
            return None
        if cell in walls:
            return prev
        prev = cell
    return prev


def _solve_grid(grid: np.ndarray, use_walls: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    walls = set(zip(*np.where(g == WALL_COLOR))) if use_walls else set()
    walls = {(int(r), int(c)) for r, c in walls}

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg, WALL_COLOR}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        dest = _landing_cell(ad[0], ad[1], walls, (h, w))
        if dest is not None:
            dest_to_shooters.setdefault(dest, []).append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites, but let grey walls shorten some rays.

    Input structure:
    - Background is the most common color.
    - Triangle sprites are still 3-pixel monochrome L-shapes or diagonal V-shapes.
    - Grey cells (color 5) are walls: they are not sprites and never move or recolor.

    Rule:
    - Each triangle still determines an apex and an emission direction from its shape.
    - Follow that ray step-by-step for up to 5 cells. Without a wall, the projectile lands on step 5.
    - If the first wall on the ray appears earlier, the projectile lands on the cell immediately before that wall.
    - After the actual landing cell is known, the original effects remain: hitting a sprite recolors the whole sprite to maroon (9), two projectiles on the same empty bg cell paint blue (1), and a lone bg landing paints maroon (9).

    Key insight: The triangles still encode directions exactly as before; the only new inference is that grey wall pixels truncate the 5-cell shot, so the visible mark appears just in front of the first wall on that ray.
    """
    return _solve_grid(grid, use_walls=True)


# ---- canvas / palette ----
H = W = 16
BG = 7
WALL_COLOR = 5
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_BLOCKED_UNIQUES = 2

# ---- generation retries ----
MAX_ATTEMPTS = 600


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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], list[dict]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    blocked_options: list[dict] = []
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                path = tuple((ar + step * dr, ac + step * dc) for step in range(1, PROJECTILE_DIST + 1))
                dest = path[-1]
                if not all(0 <= r < H and 0 <= c < W for r, c in path):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "path": path,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                for block_step in range(2, PROJECTILE_DIST + 1):
                    wall = path[block_step - 1]
                    blocked_options.append(
                        {
                            "placement": placement,
                            "dest": path[block_step - 2],
                            "wall": wall,
                            "wall_halo": _cell_halo(wall),
                        }
                    )
    return placements, by_dest, blocked_options


PLACEMENTS, PLACEMENTS_BY_DEST, BLOCKED_OPTIONS = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _effective_dest(placement: dict, walls: set[tuple[int, int]]) -> tuple[int, int]:
    prev = placement["apex"]
    for cell in placement["path"]:
        if cell in walls:
            return prev
        prev = cell
    return prev


def _placement_fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> bool:
    blocked_cells = occupied | walls
    return not (placement["halo"] & blocked_cells) and not (placement["cell_set"] & reserved_bg)


def _all_plans_stable(shots: list[dict], walls: set[tuple[int, int]]) -> bool:
    for shot in shots:
        if _effective_dest(shot["placement"], walls) != shot["dest"]:
            return False
    return True


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied or dest in walls:
            continue
        candidates = [
            p
            for p in PLACEMENTS_BY_DEST[dest]
            if _placement_fits(p, occupied, reserved_bg, walls) and _effective_dest(p, walls) == dest
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if not _placement_fits(second, occ2, reserved_bg, walls):
                    continue
                return dest, [first, second]
    return None


def _choose_unblocked_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        dest = placement["dest"]
        if dest in reserved_bg or dest in occupied or dest in walls:
            continue
        if not _placement_fits(placement, occupied, reserved_bg, walls):
            continue
        if _effective_dest(placement, walls) == dest:
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [
            p
            for p in PLACEMENTS_BY_DEST[cell]
            if _placement_fits(p, occupied, reserved_bg, walls) and _effective_dest(p, walls) == cell
        ]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _choose_blocked_unique(
    rng: random.Random,
    shots: list[dict],
    occupied: set[tuple[int, int]],
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
        if wall in occupied or wall in reserved_bg or wall in walls:
            continue
        if option["wall_halo"] & occupied:
            continue
        if option["wall_halo"] & walls:
            continue
        if not _placement_fits(placement, occupied, reserved_bg, walls):
            continue
        trial_walls = set(walls)
        trial_walls.add(wall)
        shot = {"placement": placement, "dest": dest, "target_kind": "bg", "blocked": True, "wall": wall}
        if _effective_dest(placement, trial_walls) != dest:
            continue
        if _all_plans_stable(shots + [shot], trial_walls):
            return shot
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, n_walls: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    if np.count_nonzero(grid == WALL_COLOR) != n_walls:
        return False
    sprite_colors = sorted(set(int(v) for v in np.unique(grid)) - {BG, WALL_COLOR})
    if len(sprite_colors) != 1:
        return False
    comps = []
    for col in sprite_colors:
        comps.extend(_cc8(grid == col))
    if len(comps) != n_sprites:
        return False
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            for blocked_unique in range(1, min(MAX_BLOCKED_UNIQUES, n_sprites) + 1):
                unblocked_unique = n_sprites - e_hit - 2 * e_coll - blocked_unique
                if unblocked_unique < 1:
                    continue
                if e_hit + e_coll < 1:
                    continue
                options.append((e_hit, e_coll, blocked_unique, unblocked_unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, shots, occupied, reserved_bg, walls) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg, walls)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            shots.append({"placement": placement, "dest": dest, "target_kind": "bg", "blocked": False, "wall": None})
            occupied.update(placement["cells"])
    return True


def _place_unblocked_uniques(rng, n, shots, occupied, reserved_bg, walls) -> bool:
    for _ in range(n):
        placement = _choose_unblocked_unique(rng, occupied, reserved_bg, walls)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        shots.append(
            {"placement": placement, "dest": placement["dest"], "target_kind": "bg", "blocked": False, "wall": None}
        )
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, shots, occupied, reserved_bg, walls) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(shots)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, shots[idx], occupied, reserved_bg, walls, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        shots.append(
            {
                "placement": placement,
                "dest": cell,
                "target_kind": "hit",
                "target_sprite": idx,
                "blocked": False,
                "wall": None,
            }
        )
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _place_blocked_uniques(rng, n, shots, occupied, reserved_bg, walls) -> bool:
    for _ in range(n):
        shot = _choose_blocked_unique(rng, shots, occupied, reserved_bg, walls)
        if shot is None:
            return False
        reserved_bg.add(shot["dest"])
        shots.append(shot)
        occupied.update(shot["placement"]["cells"])
        walls.add(shot["wall"])
    return True


def _render_output(inp: np.ndarray, shots: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["dest"] for s in shots if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in shots if s["target_kind"] == "hit"}
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
    reserved_bg: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, shots, occupied, reserved_bg, walls):
        return None

    base_needed = max(0, e_hit - len(shots))
    if base_needed > unblocked_unique:
        return None
    if not _place_unblocked_uniques(rng, base_needed, shots, occupied, reserved_bg, walls):
        return None

    if not _place_hits(rng, e_hit, shots, occupied, reserved_bg, walls):
        return None

    if not _place_blocked_uniques(rng, blocked_unique, shots, occupied, reserved_bg, walls):
        return None

    remaining_unblocked = unblocked_unique - base_needed
    if remaining_unblocked < 0:
        return None
    if not _place_unblocked_uniques(rng, remaining_unblocked, shots, occupied, reserved_bg, walls):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for r, c in walls:
        inp[r, c] = WALL_COLOR
    for shot in shots:
        _paint_sprite(inp, shot["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, blocked_unique):
        return None

    out = _render_output(inp, shots)

    if not any(shot["blocked"] for shot in shots):
        return None
    if not any((not shot["blocked"]) and shot["target_kind"] == "bg" for shot in shots):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
        old_style = _solve_grid(inp, use_walls=False)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(old_style, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
