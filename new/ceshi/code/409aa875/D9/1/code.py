# Variant of 409aa875
# Dimension: D9 — 精灵颜色策略 (meta)
# Change: 所有精灵共享同一随机颜色（从9种非背景色中选1种）... → 两组颜色（红队 vs 蓝队）

# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Grid geometry stays the same as the source task: 16x16, bg=7, every sprite is a 3-pixel
#      L or V triangle whose orientation defines an invisible projectile landing 5 cells away.
#    - The visible change is now driven by sprite color groups. Inputs use exactly two sprite colors:
#      red team = 2 and blue team = 8.
#    - Projectile color inherits the shooter's team.
#    - Empty-cell landing:
#        * 1 projectile -> maroon 9 (unchanged from the source task)
#        * >=2 projectiles, all same team -> blue 1
#        * >=2 projectiles, mixed teams -> maroon 9
#    - Sprite landing:
#        * an enemy-team projectile recolors the whole hit sprite to maroon 9
#        * a same-team projectile does nothing
#    - Small trace:
#        * Original: red sprite A and red sprite B both land on one empty cell => blue, because count>=2.
#        * Variant: still blue, but now because both shooters are same team.
#        * Original: red sprite A and blue sprite C land on one empty cell => blue.
#        * Variant: now maroon, because the collision is cross-team.
#        * Original: red sprite A hits red sprite B => B turns maroon.
#        * Variant: same-team hit is harmless, so B stays red.
#
# b) Complexity calibration
#    - Only two team colors are used, which keeps the color logic learnable.
#    - Every generated grid contains both:
#        * one same-team background collision
#        * one cross-team background collision
#      This forces a visible contrast between blue and maroon collision markers.
#    - Additional hit events are optional for variety, but the collision contrast alone already
#      prevents the task from collapsing back to the original "count-only" rule.
#
# c) Implicit assumption review of the original main()
#    - Original component collection by monochrome color still works, because every sprite remains
#      monochrome and separated.
#    - Original `hit_components.add(cell_to_comp[dest])` assumed any hit recolors the target.
#      This is no longer valid; we must compare shooter color against target sprite color.
#    - Original bg rule `if len(shooters) >= 2: blue else maroon` assumed collision outcome depends
#      only on count. This is no longer valid; team-color agreement must be checked.
#    - Original unconditional recolor loop to 9 remains valid only after the new enemy-hit test.
#
# d) Placement legality
#    - The original placement machinery is retained: every sprite cell must be in bounds, every
#      landing cell must be in bounds, and sprites stay Chebyshev-separated by at least 2.
#    - Collision landing cells are reserved so they remain empty background in the input.
#    - Hit shooters are added only after base collision sprites exist, and they target only those
#      pre-existing sprites, avoiding dependency chains.
#
# e) Visual observability
#    - The solver can infer team membership directly from visible sprite colors 2 and 8.
#    - Same-team and cross-team collisions are both guaranteed in each generated sample, so the
#      blue-vs-maroon difference can be compared within a single grid.
#    - Some samples also include sprite hits, letting the viewer notice that only opposite-color
#      hits recolor a target sprite.
#
# f) Visual cleanliness
#    - Sprites keep the original spacing rule, so colored teams do not merge into ambiguous blobs.
#    - Output markers remain single cells on empty background, separate from the 3-pixel sprites.
#
# g) Color-rule inferability
#    - Only four visible non-bg colors appear in generated examples:
#        * team colors 2 and 8 in the input
#        * result colors 1 and 9 in the output
#    - The mapping is structural rather than numeric: "same team" vs "mixed teams" explains 1/9,
#      and "enemy hit" explains sprite recoloring. No hidden ordering or color-id arithmetic is used.
#
# New main() assumptions and generator guarantees
#    - main() assumes the meaningful projectile sources are 3-pixel monochrome triangle components.
#      generate_input() guarantees every non-bg component in the input has exactly 3 cells.
#    - main() assumes team identity is encoded by sprite color; generate_input() guarantees both
#      colors 2 and 8 are present in every grid.
#    - main() assumes mixed-team background collisions should be distinguishable from same-team ones;
#      generate_input() guarantees at least one event of each kind per sample.

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8
from arc_utils import get_bg


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; resolve landings using team colors.

    Input structure:
    - bg is the most common color.
    - Non-bg 3-pixel sprites are axis-aligned L triangles or diagonal V triangles.
    - Sprites belong to one of two visible teams, encoded by their input color.

    Rule:
    - Each triangle emits one projectile. L-shapes fire diagonally away from the missing 2x2 corner; V-shapes fire axis-aligned away from the opening.
    - The projectile lands exactly 5 cells from the apex in that emission direction.
    - If a projectile lands on a sprite of the opposite team, that whole target sprite becomes maroon (9); same-team hits do nothing.
    - If projectiles land on an empty background cell, a same-team collision paints blue (1), while a mixed-team collision paints maroon (9); a lone projectile also paints maroon (9).

    Key insight:
    - Orientation still determines where each projectile goes, but color now determines what a shared landing means: same team cooperates into blue, opposite teams clash into maroon, and only enemy fire damages a sprite.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    comp_colors: list[int] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            comp_colors.append(col)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    hit_components: set[int] = set()
    for dest, shooters in dest_to_shooters.items():
        target_cid = cell_to_comp.get(dest)
        if target_cid is None:
            continue
        target_color = comp_colors[target_cid]
        if any(comp_colors[shooter] != target_color for shooter in shooters):
            hit_components.add(target_cid)

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        teams = {comp_colors[shooter] for shooter in shooters}
        if len(shooters) >= 2 and len(teams) == 1:
            out[r, c] = 1
        else:
            out[r, c] = 9

    return out


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
TEAM_COLORS = (2, 8)

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_ENEMY_HITS = 2
MAX_FRIENDLY_HITS = 1

# ---- generation retries ----
MAX_ATTEMPTS = 400


def _other_team(color: int) -> int:
    return TEAM_COLORS[1] if color == TEAM_COLORS[0] else TEAM_COLORS[0]


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
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
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
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    colors = set(int(v) for v in np.unique(grid)) - {BG}
    if colors != set(TEAM_COLORS):
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    base = 4  # one same-team collision pair + one cross-team collision pair
    for enemy_hits in range(0, min(MAX_ENEMY_HITS, n_sprites - base) + 1):
        for friendly_hits in range(0, min(MAX_FRIENDLY_HITS, n_sprites - base - enemy_hits) + 1):
            unique = n_sprites - base - enemy_hits - friendly_hits
            if unique < 0:
                continue
            options.append((enemy_hits, friendly_hits, unique))
    return rng.choice(options) if options else None


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hit_sprite_ids: set[int] = set()
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]].append(sprite["color"])
        else:
            target_idx = sprite["target_sprite"]
            if sprite["color"] != sprites[target_idx]["color"]:
                hit_sprite_ids.add(target_idx)
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, colors in bg_targets.items():
        out[dest] = COLLISION_COLOR if len(colors) >= 2 and len(set(colors)) == 1 else HIT_COLOR
    return out


def _add_collision_event(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    same_team: bool,
) -> bool:
    choice = _choose_collision_pair(rng, occupied, reserved_bg)
    if choice is None:
        return False
    dest, pair = choice
    reserved_bg.add(dest)
    if same_team:
        color = rng.choice(TEAM_COLORS)
        colors = [color, color]
    else:
        colors = list(TEAM_COLORS)
        rng.shuffle(colors)
    for placement, color in zip(pair, colors):
        sprites.append(
            {
                "placement": placement,
                "target": dest,
                "target_kind": "bg",
                "color": color,
                "event_kind": "same_collision" if same_team else "cross_collision",
            }
        )
        occupied.update(placement["cells"])
    return True


def _place_hits(
    rng: random.Random,
    enemy_hits: int,
    friendly_hits: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    target_ids: list[int],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    relations = ["enemy"] * enemy_hits + ["friendly"] * friendly_hits
    rng.shuffle(relations)
    for relation in relations:
        candidate_ids = target_ids[:]
        rng.shuffle(candidate_ids)
        placed = False
        for idx in candidate_ids:
            choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
            if choice is None:
                continue
            placement, cell = choice
            target_color = sprites[idx]["color"]
            color = _other_team(target_color) if relation == "enemy" else target_color
            sprites.append(
                {
                    "placement": placement,
                    "target": cell,
                    "target_kind": "hit",
                    "target_sprite": idx,
                    "color": color,
                    "event_kind": f"{relation}_hit",
                }
            )
            occupied.update(placement["cells"])
            used_target_cells.add(cell)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_unique_landings(
    rng: random.Random,
    n: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append(
            {
                "placement": placement,
                "target": placement["dest"],
                "target_kind": "bg",
                "color": rng.choice(TEAM_COLORS),
                "event_kind": "unique_bg",
            }
        )
        occupied.update(placement["cells"])
    return True


def _variant_stats(sprites: list[dict]) -> dict[str, int]:
    colors = Counter(sprite["color"] for sprite in sprites)
    return {
        "same_collision_shooters": sum(sprite["event_kind"] == "same_collision" for sprite in sprites),
        "cross_collision_shooters": sum(sprite["event_kind"] == "cross_collision" for sprite in sprites),
        "enemy_hits": sum(sprite["event_kind"] == "enemy_hit" for sprite in sprites),
        "friendly_hits": sum(sprite["event_kind"] == "friendly_hit" for sprite in sprites),
        "unique_bg": sum(sprite["event_kind"] == "unique_bg" for sprite in sprites),
        "red_sprites": colors[TEAM_COLORS[0]],
        "blue_sprites": colors[TEAM_COLORS[1]],
    }


def _gen_case() -> dict | None:
    rng = random.Random()
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    enemy_hits, friendly_hits, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _add_collision_event(rng, sprites, occupied, reserved_bg, same_team=True):
        return None
    if not _add_collision_event(rng, sprites, occupied, reserved_bg, same_team=False):
        return None

    target_ids = list(range(len(sprites)))
    if not _place_hits(rng, enemy_hits, friendly_hits, sprites, occupied, reserved_bg, target_ids):
        return None
    if not _place_unique_landings(rng, unique_total, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    stats = _variant_stats(sprites)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if stats["cross_collision_shooters"] != 2 or stats["same_collision_shooters"] != 2:
        return None
    if stats["red_sprites"] == 0 or stats["blue_sprites"] == 0:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    return {"input": inp, "output": out, "sprites": sprites, "stats": stats}


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _gen_case()
        if case is not None:
            return case["input"]
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
