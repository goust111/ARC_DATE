# Combo variant of 409aa875 (index 9)
# combo_spec: D5/1+D9/1
# Name: 红蓝阵营+碰撞计数颜色

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        if str(_parent / "solvers") not in sys.path:
            sys.path.append(str(_parent / "solvers"))
        break

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and trace
#    - Geometry stays identical to the source task: every non-background object is a 3-cell triangle,
#      either an axis-aligned L or a diagonal V, and its shape determines a landing cell exactly
#      5 steps away from the apex.
#    - The fusion splits sprites into two visible teams, red (2) and blue (8). Team identity affects
#      only sprite hits: enemy fire recolors the whole target sprite to maroon (9), while friendly fire
#      is harmless.
#    - Empty background landings ignore team agreement and instead encode the exact number of shooters:
#        1 -> 9, 2 -> 1, 3 -> 2, 4 -> 3, then continue through visible non-background colors while
#        skipping background 7.
#    - Small trace:
#        * source task: a red sprite and a blue sprite both land on the same empty cell -> blue(1).
#        * D9 alone:   that same mixed-team pair would become maroon(9).
#        * D5 alone:   three same-color shooters on one empty cell would become red(2), but sprite hits
#          would still ignore team.
#        * combo:      mixed/same team does not matter for the empty-cell marker, so 2 shooters still
#          give blue(1) and 3 shooters give red(2); however, a same-team sprite hit does nothing while
#          an enemy-team hit recolors the whole target to 9.
#      The solver therefore must keep two facts at once: team labels for sprite-hit damage, and shooter
#      multiplicity for empty-cell markers.
#
# b) Complexity calibration
#    - Inputs use exactly two team colors, so the extra meta-logic stays learnable instead of exploding
#      into many palette cases.
#    - The generator always includes:
#        * one friendly hit and one enemy hit, so team reasoning visibly matters;
#        * one 2-way collision and one 3-way or 4-way collision on empty background, so count reasoning
#          visibly matters and cannot collapse back to the source task's coarse >=2 rule.
#    - Collision team composition is also varied (one same-team, one mixed-team) so the examples do not
#      accidentally suggest that empty-cell color depends on allegiance.
#
# c) Implicit assumption review
#    - Component extraction by monochrome connected components is still valid because every sprite remains
#      monochrome and the generator preserves Chebyshev separation.
#    - The source task assumed every sprite hit recolors the target. That assumption is broken here:
#      `main()` must compare shooter color with target color before deciding whether to recolor.
#    - The D9 single variant assumed empty-cell collisions depend on team agreement. That assumption is
#      also broken here: `main()` must keep the full shooter list so it can count arrivals even when the
#      shooters come from different teams.
#    - Writing one output color per landing coordinate remains safe because the generator never mixes an
#      empty landing and a sprite landing at the same destination.
#
# d) Placement legality
#    - All sprite placements are precomputed from (apex, direction) pairs whose sprite cells and landing
#      cells are in bounds.
#    - Halo separation is enforced, so sprites never merge into a larger component and `main()` can keep
#      assuming that each visible object is exactly one 3-cell triangle.
#    - Collision groups are chosen by backtracking over compatible placements that share a landing cell
#      without touching each other or any previously placed sprite.
#
# e) Visual observability
#    - Team identity is directly readable from the input colors 2 and 8.
#    - Empty-cell markers are single pixels while teams are 3-pixel sprites, so the reused red color 2
#      for the 3-way marker is still structurally distinguishable.
#    - Every sample contains both a harmless friendly hit and a damaging enemy hit, plus at least two
#      different collision multiplicities, which is enough evidence for a human to infer the fusion rule
#      from a few examples.
#
# f) Visual cleanliness
#    - Output markers remain on background cells only; they never overwrite partial sprite geometry.
#    - The generator keeps the original spacing discipline and limits itself to two team colors plus a
#      small set of result colors, so the board stays readable instead of becoming a dense color cloud.
#
# g) Change inferability
#    - The color changes are tied to visible structure:
#        * whole-sprite recolor depends on whether the incoming shooter is from the other team;
#        * single-cell marker color depends on how many triangles aim at that empty cell.
#    - No hidden sorting by color id or ordering of components is used.
#
# New `main()` assumptions and generator guarantees
#    - `main()` assumes each non-background component is a 3-cell monochrome triangle with one of the
#      canonical firing orientations; `generate_input()` constructs only such components.
#    - `main()` assumes team identity is represented by color; `generate_input()` guarantees both 2 and 8
#      are present in every sample.
#    - `main()` assumes friendly hits can be ignored and enemy hits can be unioned at the sprite level;
#      `generate_input()` keeps hit destinations unambiguous and ensures friendly-hit targets are distinct
#      from enemy-hit targets, so the contrast is observable.


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


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
TEAM_COLORS = (2, 8)
COLLISION_PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (7, 9)
COLLISION_SIZE_OPTIONS = ((2, 3), (2, 4))

# ---- generation retries ----
MAX_ATTEMPTS = 800


def _collision_color(shooter_count: int) -> int:
    if shooter_count <= 1:
        return HIT_COLOR
    idx = min(shooter_count - 2, len(COLLISION_PALETTE) - 1)
    return COLLISION_PALETTE[idx]


def _other_team(color: int) -> int:
    return TEAM_COLORS[1] if color == TEAM_COLORS[0] else TEAM_COLORS[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangle sprites still fire 5-cell projectiles, but sprite damage uses teams while empty collisions use counts.

    Input structure:
    - The background is the most frequent color.
    - Every non-background object is a 3-pixel monochrome triangle, either an axis-aligned L or a diagonal V.
    - Sprites belong to one of two visible teams, encoded by their input color.

    Rule:
    - Read each triangle's apex and firing direction from its shape: L-shapes shoot diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their opening.
    - Each projectile lands exactly 5 cells from the apex along that direction.
    - If a landing cell lies on a sprite, only enemy-team shooters matter: if at least one incoming projectile comes from the opposite team, recolor the whole target sprite to maroon (9); friendly fire leaves it unchanged.
    - If a landing cell is empty background, paint that one cell by the number of projectiles that arrived there: 1 -> maroon (9), 2 -> blue (1), 3 -> red (2), 4 -> green (3), then continue through the visible non-background palette while skipping the background color 7.

    Key insight:
    - Orientation still tells you where every projectile lands, but two different aggregations are now needed afterward: compare shooter team against the hit sprite's team for sprite damage, and count how many shooters share an empty landing cell for the collision marker color.
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
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
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
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = _collision_color(len(shooters))

    return out


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


def _pick_compatible_group(
    rng: random.Random,
    candidates: list[dict],
    need: int,
    occupied: set[tuple[int, int]],
) -> list[dict] | None:
    order = candidates[:]
    rng.shuffle(order)

    def search(start: int, blocked: set[tuple[int, int]], chosen: list[dict]) -> list[dict] | None:
        if len(chosen) == need:
            return chosen
        if len(order) - start < need - len(chosen):
            return None
        for idx in range(start, len(order)):
            placement = order[idx]
            if placement["halo"] & blocked:
                continue
            found = search(idx + 1, blocked | set(placement["cell_set"]), chosen + [placement])
            if found is not None:
                return found
        return None

    return search(0, set(occupied), [])


def _choose_collision_group(
    rng: random.Random,
    size: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        if len(candidates) < size:
            continue
        group = _pick_compatible_group(rng, candidates, size, occupied)
        if group is not None:
            return dest, group
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
        if not candidates:
            continue
        rng.shuffle(candidates)
        for placement in candidates:
            return placement, cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    colors = set(int(v) for v in np.unique(grid)) - {BG}
    if colors != set(TEAM_COLORS):
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(len(comp) == 3 and _triangle_apex_and_dir(comp) is not None for comp in comps)


def _collision_team_colors(rng: random.Random, size: int, same_team: bool) -> list[int]:
    if same_team:
        color = rng.choice(TEAM_COLORS)
        return [color] * size
    colors = [TEAM_COLORS[0], TEAM_COLORS[1]]
    while len(colors) < size:
        colors.append(rng.choice(TEAM_COLORS))
    rng.shuffle(colors)
    return colors


def _render_fused_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: Counter = Counter()
    hit_sprite_ids: set[int] = set()
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]] += 1
            continue
        target_idx = sprite["target_sprite"]
        if sprite["color"] != sprites[target_idx]["color"]:
            hit_sprite_ids.add(target_idx)
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = _collision_color(count)
    return out


def _render_count_only_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: Counter = Counter()
    hit_sprite_ids: set[int] = set()
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]] += 1
        else:
            hit_sprite_ids.add(sprite["target_sprite"])
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = _collision_color(count)
    return out


def _render_team_only_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hit_sprite_ids: set[int] = set()
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]].append(sprite["color"])
            continue
        target_idx = sprite["target_sprite"]
        if sprite["color"] != sprites[target_idx]["color"]:
            hit_sprite_ids.add(target_idx)
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, colors in bg_targets.items():
        out[dest] = 1 if len(colors) >= 2 and len(set(colors)) == 1 else HIT_COLOR
    return out


def _render_original_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: Counter = Counter()
    hit_sprite_ids: set[int] = set()
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]] += 1
        else:
            hit_sprite_ids.add(sprite["target_sprite"])
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = 1 if count >= 2 else HIT_COLOR
    return out


def _add_collision_event(
    rng: random.Random,
    size: int,
    same_team: bool,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    choice = _choose_collision_group(rng, size, occupied, reserved_bg)
    if choice is None:
        return False
    dest, group = choice
    colors = _collision_team_colors(rng, size, same_team)
    reserved_bg.add(dest)
    for placement, color in zip(group, colors):
        sprites.append(
            {
                "placement": placement,
                "target": dest,
                "target_kind": "bg",
                "color": color,
                "event_kind": "same_collision" if same_team else "mixed_collision",
                "collision_size": size,
            }
        )
        occupied.update(placement["cells"])
    return True


def _place_hits(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    target_ids: list[int],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    used_target_sprites: set[int] = set()
    relations = ["enemy", "friendly"]
    rng.shuffle(relations)
    for relation in relations:
        candidate_ids = [idx for idx in target_ids if idx not in used_target_sprites]
        rng.shuffle(candidate_ids)
        placed = False
        for idx in candidate_ids:
            choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
            if choice is None:
                continue
            placement, cell = choice
            target_color = sprites[idx]["color"]
            shooter_color = _other_team(target_color) if relation == "enemy" else target_color
            sprites.append(
                {
                    "placement": placement,
                    "target": cell,
                    "target_kind": "hit",
                    "target_sprite": idx,
                    "color": shooter_color,
                    "event_kind": f"{relation}_hit",
                }
            )
            occupied.update(placement["cells"])
            used_target_cells.add(cell)
            used_target_sprites.add(idx)
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
                "collision_size": 1,
            }
        )
        occupied.update(placement["cells"])
    return True


def _variant_stats(sprites: list[dict], out: np.ndarray, base_outs: dict[str, np.ndarray]) -> dict[str, object]:
    colors = Counter(sprite["color"] for sprite in sprites)
    bg_counts = Counter(sprite["target"] for sprite in sprites if sprite["target_kind"] == "bg")
    bg_team_modes = []
    for dest, count in bg_counts.items():
        dest_colors = {sprite["color"] for sprite in sprites if sprite["target_kind"] == "bg" and sprite["target"] == dest}
        bg_team_modes.append((count, "same" if len(dest_colors) == 1 else "mixed"))
    return {
        "enemy_hits": sum(sprite["event_kind"] == "enemy_hit" for sprite in sprites),
        "friendly_hits": sum(sprite["event_kind"] == "friendly_hit" for sprite in sprites),
        "unique_bg": sum(sprite["event_kind"] == "unique_bg" for sprite in sprites),
        "red_sprites": colors[TEAM_COLORS[0]],
        "blue_sprites": colors[TEAM_COLORS[1]],
        "bg_team_modes": sorted(bg_team_modes),
        "max_bg_collision": max(bg_counts.values()),
        "diff_vs_count_only": not np.array_equal(out, base_outs["count_only"]),
        "diff_vs_team_only": not np.array_equal(out, base_outs["team_only"]),
        "diff_vs_original": not np.array_equal(out, base_outs["original"]),
    }


def _gen_case() -> dict | None:
    rng = random.Random()
    collision_sizes = rng.choice(COLLISION_SIZE_OPTIONS)
    min_sprites = max(SPRITE_COUNT_RANGE[0], sum(collision_sizes) + 2)
    n_sprites = rng.randint(min_sprites, SPRITE_COUNT_RANGE[1])
    unique_total = n_sprites - sum(collision_sizes) - 2

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    relation_order = [True, False]
    rng.shuffle(relation_order)
    for size, same_team in zip(collision_sizes, relation_order):
        if not _add_collision_event(rng, size, same_team, sprites, occupied, reserved_bg):
            return None

    target_ids = list(range(len(sprites)))
    if not _place_hits(rng, sprites, occupied, reserved_bg, target_ids):
        return None
    if not _place_unique_landings(rng, unique_total, sprites, occupied, reserved_bg):
        return None
    if len(sprites) != n_sprites:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_fused_output(inp, sprites)
    base_outs = {
        "count_only": _render_count_only_output(inp, sprites),
        "team_only": _render_team_only_output(inp, sprites),
        "original": _render_original_output(inp, sprites),
    }
    stats = _variant_stats(sprites, out, base_outs)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if stats["enemy_hits"] != 1 or stats["friendly_hits"] != 1:
        return None
    if stats["red_sprites"] == 0 or stats["blue_sprites"] == 0:
        return None
    if stats["max_bg_collision"] < 3:
        return None
    if not any(mode == (2, "same") or mode == (2, "mixed") for mode in stats["bg_team_modes"]):
        return None
    if not any(count >= 3 for count, _mode in stats["bg_team_modes"]):
        return None
    if not any(mode == "same" for _count, mode in stats["bg_team_modes"]):
        return None
    if not any(mode == "mixed" for _count, mode in stats["bg_team_modes"]):
        return None
    if not stats["diff_vs_count_only"]:
        return None
    if not stats["diff_vs_team_only"]:
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(out == 1):
        return None
    if not np.any(np.isin(out, [2, 3])):
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
