# Combo variant of 409aa875 (index 22)
# combo_spec: D4/0+D19/2
# Name: 命中消失+未命中加外框

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for combo D4/0 + D19/2
# ---------------------------------------
# fused_rule_summary:
#   - Keep the original geometry intact:
#     every 3-cell monochrome triangle sprite still decodes an apex and a
#     firing direction, and every projectile still lands at apex + 5 * dir.
#   - Change the two outcome branches jointly:
#     1. If a projectile lands on a sprite, the whole struck sprite disappears.
#     2. If a sprite is not struck by any projectile, it remains in place and
#        gains a 1-pixel border on surrounding background cells.
#   - Empty-background impacts remain the original ones:
#     lone bg hit -> 9, shared bg hit -> 1.
# concrete_trace:
#   - Suppose sprite A fires into sprite B, while sprite C fires into an empty
#     cell and sprite D is not targeted by anyone.
#   - Original task:
#       B becomes solid maroon, C paints one maroon marker, D stays unchanged.
#   - Fused task:
#       B is erased to background, C still paints one maroon marker, D keeps its
#       original fill but gets a teal border.
#   - The visual contrast is therefore binary for sprites:
#       hit => absent, unhit => more salient via framing.
# complexity_calibration:
#   - The variant is intentionally high-contrast but still low-branch:
#     only two sprite states exist in the output, "gone" and "outlined".
#   - The border color is fixed to 8, so the learner does not need to infer a
#     per-color mapping. This keeps the new freedom small and repeatable.
#   - To keep the change observable in 2-4 examples, generation must force:
#       * at least one hit sprite that disappears,
#       * at least one unhit sprite that survives with a border,
#       * at least one background impact marker so projectile landing behavior
#         is still directly visible.
# hidden_assumption_audit:
#   - Component detection on the input still assumes isolated monochrome 3-cell
#     8-connected sprites; the generator preserves that.
#   - Hit detection must be computed from the input occupancy map before any
#     erasure. If we erased early, later hit checks could misclassify a second
#     projectile as a background hit.
#   - The border pass can no longer be "paint every halo cell blindly":
#     it must run after erasures and bg markers, and it may only paint cells
#     that are still background to avoid overwriting landing marks.
#   - Bordering only unhit sprites assumes we can tell hit vs unhit per whole
#     component, not per cell. The projectile rule still targets whole sprites,
#     so this assumption remains valid.
# placement_legality:
#   - The D19-style border introduces a new spatial object: each sprite's
#     radius-1 halo. If halos of different sprites touch, the outlines merge and
#     the output gets messy.
#   - Therefore the generator requires each placed sprite's full halo to be
#     in-bounds and disjoint from every other sprite halo.
#   - Background landing cells are also forbidden inside any sprite halo, so
#     border cells never compete with lone-hit/collision markers.
# visual_observability:
#   - Every sample contains both classes of sprite outcomes:
#       * at least one erased target sprite,
#       * at least one outlined survivor.
#   - Because halo cells are reserved and disjoint, the teal frame reads as a
#     clean ring around each survivor rather than a tangled shared region.
#   - Sprite colors exclude {1, 8, 9} so blue collision marks, teal borders,
#     and maroon bg hits remain unambiguous to a human viewer.
# inferability_check:
#   - A solver looking only at examples can trace orientation -> landing cell
#     from unchanged projectile geometry, then separate outcomes by whether the
#     landing cell falls on a sprite.
#   - The same condition that previously meant recolor now means disappearance,
#     while the complement set of sprites becomes outlined. This is a compact
#     conditional rule with repeated support across samples, not a one-off case.


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


def _frame_cells(comp: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    s = set(comp)
    frame: set[tuple[int, int]] = set()
    for r, c in comp:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in s:
                    frame.add((rr, cc))
    return frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; hit targets vanish while untouched sprites gain a border.

    Input structure:
    - The background is the most common color, and every non-background object is an isolated monochrome 3-pixel sprite shaped as either an axis-aligned L or a diagonal V.

    Rule:
    - Each triangle's shape determines an apex and firing direction: L-shapes shoot diagonally away from the missing corner of their 2x2 box, while V-shapes shoot along the axis opposite their opening.
    - The projectile lands exactly 5 cells away from that apex along the decoded direction.
    - If the landing cell lies on a sprite, erase the entire struck sprite to the background; if it lies on empty background, paint that cell maroon (9) for a single projectile or blue (1) for a shared collision.
    - After resolving hits and empty-cell markers, every sprite that was never struck keeps its fill color and receives a 1-pixel teal border (8) on surrounding background cells.

    Key insight:
    - The direction-decoding logic is unchanged from the base task, but the output now partitions sprites into two extremes based on whether any projectile lands on them: victims disappear completely, while survivors become more conspicuous via an added halo.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    lone_bg_color = 9
    collision_color = 1
    border_color = 8

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
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
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = bg

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = collision_color if len(shooters) >= 2 else lone_bg_color

    for cid, comp in enumerate(comps):
        if cid in hit_components:
            continue
        for r, c in _frame_cells(comp, h, w):
            if out[r, c] == bg:
                out[r, c] = border_color

    return out


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
LONE_BG_COLOR = 9
COLLISION_COLOR = 1
BORDER_COLOR = 8
SPRITE_COLORS = [0, 2, 3, 4, 5, 6]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 8)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 700


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _zone(cells: tuple[tuple[int, int], ...], radius: int) -> frozenset[tuple[int, int]] | None:
    zone: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if not (0 <= rr < H and 0 <= cc < W):
                    return None
                zone.add((rr, cc))
    return frozenset(zone)


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
                halo = _zone(cells, 1)
                if halo is None:
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": halo,
                    "frame": halo - frozenset(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied_halo: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied_halo) and not (placement["halo"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied_halo:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied_halo, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            halo2 = occupied_halo | set(first["halo"])
            for second in candidates[i + 1:]:
                if second["halo"] & halo2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied_halo:
            continue
        if _fits(placement, occupied_halo, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied_halo, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(1, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied_halo, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied_halo, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied_halo.update(placement["halo"])
    return True


def _place_unique_landings(rng, n, sprites, occupied_halo, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied_halo, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied_halo.update(placement["halo"])
    return True


def _place_hits(rng, e_hit, sprites, occupied_halo, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied_halo, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied_halo.update(placement["halo"])
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}

    for idx in hit_sprite_ids:
        for r, c in sprites[idx]["placement"]["cells"]:
            out[r, c] = BG

    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else LONE_BG_COLOR

    for idx, sprite in enumerate(sprites):
        if idx in hit_sprite_ids:
            continue
        for r, c in sprite["placement"]["frame"]:
            if out[r, c] == BG:
                out[r, c] = BORDER_COLOR

    return out


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    erased = 0
    survivors = 0
    framed = 0
    for comp in _cc8(inp != BG):
        comp_set = set(comp)
        if all(int(out[r, c]) == BG for r, c in comp):
            erased += 1
            continue
        survivors += 1
        frame = _frame_cells(comp, *inp.shape)
        if frame and all(int(out[r, c]) == BORDER_COLOR for r, c in frame):
            framed += 1
        elif frame and any(int(out[r, c]) == BORDER_COLOR for r, c in frame):
            framed += 1

    return {
        "erased_sprites": erased,
        "survivor_sprites": survivors,
        "framed_survivors": framed,
        "lone_bg_marks": int(np.count_nonzero(out == LONE_BG_COLOR)),
        "collision_marks": int(np.count_nonzero(out == COLLISION_COLOR)),
        "border_pixels": int(np.count_nonzero(out == BORDER_COLOR)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied_halo: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied_halo, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied_halo, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied_halo, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied_halo, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    metrics = _behavior_metrics(inp, out)

    if metrics["erased_sprites"] < 1:
        return None
    if metrics["survivor_sprites"] < 1 or metrics["framed_survivors"] < 1:
        return None
    if metrics["lone_bg_marks"] + metrics["collision_marks"] < 1:
        return None
    if metrics["border_pixels"] < 8:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
