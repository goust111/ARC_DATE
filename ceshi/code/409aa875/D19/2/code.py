# Variant of 409aa875
# Dimension: D19 — 未命中精灵的保留策略 (transform)
# Change: 未被命中的精灵保持原色不变... → 未命中精灵在输出中用外框标记
#
# design_analysis:
#   variant_goal:
#     - Keep the original projectile geometry exactly the same.
#     - Change only the "preserve unhit sprite" branch: an unhit 3-cell triangle stays its
#       input color but gains a 1-pixel Chebyshev surround in a fixed border color.
#     - Hit sprites still recolor solid maroon; empty landing cells still become maroon/blue.
#   concrete_behavior:
#     - Example trace:
#       - Sprite A is an L-shape whose apex+5*dir lands on sprite B.
#       - Sprite C lands on an empty background cell.
#       - Original task: B becomes 9, C's landing cell becomes 9, A stays unchanged.
#       - This variant: B still becomes 9, C's landing cell still becomes 9, but A and every
#         other unhit sprite are outlined on neighboring background cells with color 8.
#     - The border is the 8-neighborhood halo around the sprite, excluding the sprite's own
#       three cells. The border only paints untouched background, so it never overwrites a
#       hit recolor or a projectile marker.
#   complexity_calibration:
#     - A fixed border color is sufficient; per-sprite custom border colors would add
#       unnecessary freedom and would be hard to infer from 2-4 examples.
#     - To keep the border visually obvious, the generator excludes sprite colors {1,8,9}:
#       1 is reserved for multi-hit bg collisions, 8 is the new border color, 9 is hit color.
#     - Every generated grid is forced to contain at least one hit target and at least one
#       unhit shooter, so the learner can compare "solid recolor" vs "surrounded preserve".
#   hidden_assumption_audit:
#     - Original `main()` assumed "unhit sprites need no output work"; that is no longer true.
#     - The hit-detection loop itself is still valid because projectile direction and landing
#       distance do not change.
#     - The old output phase could stop after recoloring hit components and marking bg cells;
#       the new variant needs a third phase for unhit components.
#     - A naive border pass could overwrite projectile markers on nearby background cells.
#       This implementation only paints cells that are still background after previous phases.
#   placement_legality:
#     - If sprites remain only Chebyshev-2 apart as in the original generator, two halos can
#       merge and the new border becomes messy or ambiguous.
#     - Therefore the generator now requires each sprite's radius-1 halo to be disjoint from
#       all previously placed sprite halos, which implies at least one empty-cell gap between
#       any two future borders.
#     - Each sprite placement must also keep its full radius-1 halo in-bounds, so the border
#       is never clipped by the frame.
#     - Background landing cells are forbidden from falling inside any sprite halo, so marker
#       cells and border cells never compete for the same location.
#   visual_observability:
#     - Output grids visibly separate the two sprite classes:
#       - hit target: whole 3-pixel object changes to 9
#       - unhit sprite: original 3-pixel object remains, ring of 8 appears around it
#     - Because both classes appear in every generated example, the D19 change is learnable
#       from a small number of demonstrations rather than being an optional corner case.
#   main_new_assumptions:
#     - `main()` assumes inputs still contain isolated monochrome 3-pixel L/V components.
#     - The generator enforces that with bounded placements and exact component checks.
#     - `main()` also assumes the new border color is absent from the input for human clarity;
#       the generator guarantees that by excluding sprite color 8.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Returns (apex_rc, dir_rc). Direction is one of 4-neigh (axis) or diagonal.
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    # L-shape (axis-aligned): one pixel has two 4-neighbors (the corner).
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

    # Diagonal V-shape: one pixel has two 8-neighbors (diagonal legs).
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
    Fire projectiles from 3-pixel triangle sprites; hit targets turn maroon while survivors get a border.

    Input structure:
    - bg = most common color. Non-bg cells form isolated monochrome 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each triangle has an apex and emission direction determined by its shape: L-shapes fire diagonally away from the missing 2x2 corner; V-shapes fire along the axis opposite their opening.
    - A projectile lands exactly 5 cells from the apex in that direction.
    - If a landing cell lies on a sprite, the entire hit sprite becomes maroon (9). If the landing cell is empty bg, then a lone projectile paints maroon (9) and a multi-projectile collision paints blue (1).
    - Any sprite that is not hit keeps its original fill color and gains a 1-pixel border of color 8 on surrounding background cells.

    Key insight: Orientation still determines where each projectile goes, but now the output splits sprites into two visible classes—solid-maroon victims versus original-color survivors outlined by a fixed halo.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    hit_color = 9
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
            out[r, c] = hit_color

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = collision_color if len(shooters) >= 2 else hit_color

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
HIT_COLOR = 9
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
MAX_ATTEMPTS = 600


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
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    for idx, sprite in enumerate(sprites):
        if idx in hit_sprite_ids:
            continue
        for r, c in sprite["placement"]["frame"]:
            if out[r, c] == BG:
                out[r, c] = BORDER_COLOR
    return out


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
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}

    if len(hit_sprite_ids) < 1:
        return None
    if len(sprites) - len(hit_sprite_ids) < 1:
        return None
    if np.count_nonzero(out == BORDER_COLOR) == 0:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
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
