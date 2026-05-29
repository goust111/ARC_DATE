# Variant of 409aa875
# Dimension: D3 — 精灵形状与方向编码 (src)
# Change: 两种3像素三角形：L形（轴对齐直角，占2x2的3个角）和V形（顶点+两条对角臂），形状决定发射方向（L对角线/V轴方向）... → 箭头形精灵（凸部指向发射方向）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg

_bg_color = get_bg


# Design analysis for the D3 variant
# ----------------------------------
# spec_variant:
#   preserved:
#     grid: 16x16, bg=7
#     sprite_count: 4..9
#     sprite_color: one shared non-bg color
#     projectile_rule: tip + 5*dir
#     transform:
#       hit_sprite -> recolor whole target sprite to 9
#       lone_hit_bg -> 9
#       multi_hit_bg -> 1
#     separation: keep Chebyshev >= 2 between sprites
#   changed_core:
#     old_encoding:
#       diagonal directions used L triominoes and axis directions used V triominoes;
#       direction had to be inferred from "missing corner" or "open side".
#     new_encoding:
#       every direction is represented by a 3-pixel arrow whose pointed tip is the emitter.
#       axis directions use compact chevrons (^, >, v, <).
#       diagonal directions use 2x2 wedge arrows whose single corner tip points diagonally.
#       the solver now matches one of 8 explicit arrow templates instead of classifying
#       components as L vs V.
#   concrete_example:
#     old:
#       cells {(0,0),(0,1),(1,0)} were an L; old code read the occupied corner (0,0)
#       and fired to (-1,-1) because the missing 2x2 corner was (1,1).
#     new:
#       cells {(0,0),(1,0),(1,1)} are a NW arrow; the visible lone corner (0,0) is the tip,
#       so the projectile also goes to (-1,-1), but by direct tip semantics instead of by
#       "missing corner" logic. The old solver would misread this exact shape as firing SW,
#       so template replacement is necessary and observable.
#   complexity_calibration:
#     - Keep only 8 discrete templates, one per compass direction, so matching stays simple.
#     - Force each generated grid to contain at least two diagonal arrows and at least one
#       axis arrow; otherwise the input could look too similar to the original task.
#     - Do not add extra colors, labels, or new interaction types.
#   implicit_assumption_audit:
#     - Original `_triangle_apex_and_dir()` assumed every valid component had either:
#       one 4-neighbor corner (L) or one 8-neighbor apex (V). That is no longer true.
#     - Original generator assumed `_sprite_cells()` could be derived from direction using
#       the old L/V formulas. This must be replaced by explicit arrow templates.
#     - The recoloring code is still safe because it depends only on projectile destinations,
#       not on the old geometry family.
#   placement_legality:
#     - Every template remains a 3-pixel 8-connected monochrome component.
#     - Precomputed placements require both the sprite cells and tip+5*dir landing cell
#       to be in-bounds.
#     - Halo-based spacing is preserved so distinct sprites do not merge under 8-connectivity.
#   visual_observability:
#     - The diagonal sprites now look like literal diagonal arrowheads instead of old "L with
#       a missing-corner rule". Because each grid is required to include diagonal arrows, the
#       D3 change is visible in the input alone.
#   visual_cleanliness:
#     - Output markers stay single-cell and sprites remain sparse, so arrows and landing marks
#       do not visually stack into unreadable clusters.
#   color_inference:
#     - No new color mapping is introduced; colors still mean only {original sprite, maroon hit,
#       blue multi-hit}, which is already supported by spatial structure.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

PROJECTILE_DIST = 5
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = AXIS_DIRS + DIAG_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 300
MIN_DIAGONAL_SPRITES = 2
MIN_AXIS_SPRITES = 1


ARROW_OFFSETS_BY_DIR: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (-1, 0): ((0, 0), (1, -1), (1, 1)),   # ^
    (-1, 1): ((0, 0), (1, -1), (1, 0)),   # ↗
    (0, 1): ((0, 0), (-1, -1), (1, -1)),  # >
    (1, 1): ((0, 0), (-1, -1), (0, -1)),  # ↘
    (1, 0): ((0, 0), (-1, -1), (-1, 1)),  # v
    (1, -1): ((0, 0), (-1, 0), (-1, 1)),  # ↙
    (0, -1): ((0, 0), (-1, 1), (1, 1)),   # <
    (-1, -1): ((0, 0), (0, 1), (1, 1)),   # ↖
}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _arrow_tip_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return (tip_rc, dir_rc) for a 3-pixel arrow sprite, else None."""
    if len(comp) != 3:
        return None
    cell_set = set(comp)
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for tip in comp:
        tr, tc = tip
        for direction, offsets in ARROW_OFFSETS_BY_DIR.items():
            shaped = {(tr + dr, tc + dc) for dr, dc in offsets}
            if shaped == cell_set:
                matches.append((tip, direction))
    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel arrow sprites; mark collisions with maroon or blue.

    Input structure:
    - bg = most common color. Non-bg 3-pixel sprites are monochrome arrows in one of 8 compass directions.
    - Each arrow has a single pointed tip cell; that tip indicates the firing direction directly.

    Rule:
    - Match each 3-pixel component to one of the 8 arrow templates and use the pointed tip as the emitter.
    - A projectile lands exactly 5 cells from the tip in the arrow's direction.
    - If the landing cell lies on another sprite, recolor that entire hit sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, paint it blue (1); otherwise a lone empty-cell landing paints maroon (9).

    Key insight: The arrowhead itself now encodes direction, so the only job is to read each tip,
    jump 5 cells forward, and then apply the same hit-vs-collision coloring as before.
    """
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

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        td = _arrow_tip_and_dir(comp)
        if td is None:
            continue
        (tr, tc), (dr, dc) = td
        rr = tr + PROJECTILE_DIST * dr
        cc = tc + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

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


def _sprite_cells(tip: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    tr, tc = tip
    return tuple((tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS_BY_DIR[direction])


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
    for tr in range(H):
        for tc in range(W):
            for dr, dc in ALL_DIRS:
                dest = (tr + PROJECTILE_DIST * dr, tc + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((tr, tc), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "tip": (tr, tc),
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
            for second in candidates[i + 1:]:
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
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    for comp in comps:
        if comp.shape != (3, 2):
            return False
        coords = [(int(r), int(c)) for r, c in comp]
        if _arrow_tip_and_dir(coords) is None:
            return False
    return True


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


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
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


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
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
    return out


def _direction_mix_ok(sprites: list[dict]) -> bool:
    diag = sum(1 for s in sprites if abs(s["placement"]["dir"][0]) + abs(s["placement"]["dir"][1]) == 2)
    axis = len(sprites) - diag
    return diag >= MIN_DIAGONAL_SPRITES and axis >= MIN_AXIS_SPRITES


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

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    if not _direction_mix_ok(sprites):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)

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
