# Combo variant of 409aa875 (index 11)
# combo_spec: D2/0+D6/2
# Name: 颜色编码距离+渐变轨迹

# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   core_change:
#     old:
#       - Every triangle fired exactly 5 cells.
#       - Only the landing site was painted, with a special blue branch for multi-hit bg cells.
#     new:
#       - Triangle color encodes total firing distance: red(2)->3, green(3)->5, teal(8)->7.
#       - The whole ray is now visible, and the trail color encodes flight phase rather than
#         collision type: early steps use 4, middle steps use 6, late steps use 9.
#       - The three phase bands are normalized by each sprite's own distance, so a 3-step ray
#         becomes 4/6/9, a 5-step ray becomes 4/4/6/6/9, and a 7-step ray becomes 4/4/4/6/6/9/9.
#   concrete_behavior:
#     input:
#       - Same 16x16 grid with background 7.
#       - Same 3-pixel triangle geometry and same orientation decoding.
#       - Unlike the base task, sprites no longer share one color; their own color now matters.
#     output:
#       - Each sprite emits a straight ray from its apex in the inferred direction.
#       - Ray length depends on the sprite color.
#       - Each ray is painted cell-by-cell with the fixed phase palette 4, 6, 9.
#       - If the final cell lands on another sprite, that whole target sprite is recolored to 9.
#   behavior_trace:
#     local_example:
#       - Suppose a red L-sprite and a teal V-sprite point in different directions.
#       - Base task: both would still travel 5 cells and only endpoints would change.
#       - Combo task:
#         * the red sprite travels only 3 cells, producing colors [4, 6, 9];
#         * the teal sprite travels 7 cells, producing colors [4, 4, 4, 6, 6, 9, 9].
#       - The same orientation logic survives, but color now decides how far the ray extends,
#         and the trail itself exposes that distance through repeated phase colors.
#   complexity_calibration:
#     - Only three sprite colors are used for distance coding, so the mapping is small enough
#       to infer from 2-4 examples.
#     - Only three trail colors are used for phase coding, and they are disjoint from sprite
#       colors and background, so the new visual signal is strong.
#     - Every generated grid contains at least one sprite of each coded color, ensuring that
#       all three distances are visible within a single example instead of requiring many samples.
#   implicit_assumptions_review:
#     - Original main() assumed sprite color was irrelevant. That assumption is removed; the
#       solver must preserve component color and translate it into a distance before tracing a ray.
#     - Original main() only needed landing cells. That assumption is removed; the full path must
#       be generated and colored step-by-step.
#     - Original multi-projectile blue collision logic would conflict with phase-colored trails,
#       so the generator avoids path overlap entirely and the solver no longer reasons about counts.
#     - Recoloring a hit sprite to 9 is still valid because generation guarantees that only the
#       endpoint can touch another sprite; interior path cells always stay on background.
#   placement_legality:
#     - Sprites still keep Chebyshev separation >= 2 via halo checks.
#     - The generator forbids path/path overlap and forbids any interior path/sprite contact.
#     - Hit events are built by selecting a target sprite cell first, then finding a shooter of
#       the requested color whose color-coded distance lands on that exact cell.
#   visual_observability:
#     - A human can count trail lengths directly in the output, then tie those lengths back to
#       the input sprite colors because all three coded colors appear repeatedly.
#     - The 3/5/7 distance spread is large enough to be visually distinct, and the phase palette
#       repeats in predictable blocks, making the normalization rule learnable.
#   cleanliness:
#     - The generator keeps rays disjoint, so no ambiguous overpainting hides whether a color came
#       from a short ray's endpoint or a long ray's late-stage segment.
#     - Because each grid mixes multiple orientations and multiple encoded distances, the task is
#       recognizably different from both the base task and the trajectory-only single variant.

from __future__ import annotations

import random
import sys
from collections import defaultdict

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4: dict[tuple[int, int], int] = {}
    deg8: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms: list[tuple[int, int]] = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
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
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


H = W = 16
BG = 7

COLOR_TO_DIST = {
    2: 3,
    3: 5,
    8: 7,
}
SPRITE_COLORS = sorted(COLOR_TO_DIST)
DISTANCES = sorted(set(COLOR_TO_DIST.values()))

EARLY_COLOR = 4
MID_COLOR = 6
LATE_COLOR = 9
TRAIL_COLORS = {EARLY_COLOR, MID_COLOR, LATE_COLOR}

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (5, 8)
MAX_HIT_EVENTS = 2
MAX_ATTEMPTS = 500


def _path_cells(apex: tuple[int, int], direction: tuple[int, int], dist: int) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, dist + 1))


def _phase_bounds(dist: int) -> tuple[int, int]:
    first = (dist + 2) // 3
    second = (2 * dist + 2) // 3
    return first, second


def _color_for_step(dist: int, step: int) -> int:
    first, second = _phase_bounds(dist)
    if step <= first:
        return EARLY_COLOR
    if step <= second:
        return MID_COLOR
    return LATE_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire distance-coded projectiles from 3-pixel triangle sprites and paint normalized gradients.

    Input structure:
    - The background is the most common color.
    - Every relevant non-background 8-connected 3-pixel component is a monochrome triangle sprite,
      either an axis-aligned L or a diagonal V.
    - Sprite color is meaningful: red(2), green(3), and teal(8) encode different total travel
      distances.

    Rule:
    - Decode each triangle's apex and firing direction exactly as in the base task:
      L-shapes fire diagonally away from the missing 2x2 corner, and V-shapes fire along the
      axis opposite their opening.
    - Convert sprite color into total ray length using the fixed mapping 2->3, 3->5, 8->7.
    - Paint every traversed cell, not just the endpoint. For a ray of length d, steps
      1..ceil(d/3) use color 4, the next phase up to ceil(2d/3) uses color 6, and the rest
      uses color 9.
    - If the endpoint lands on another sprite, recolor that entire hit sprite to 9.

    Key insight: Shape still gives direction, but color now gives total distance, and the
    visible trail expresses progress through that color-coded distance using a 4/6/9 gradient.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

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

    rays: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    hit_components: set[int] = set()
    for cid, comp in enumerate(comps):
        sprite_color = comp_colors[cid]
        dist = COLOR_TO_DIST.get(sprite_color)
        if dist is None:
            continue
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        apex, direction = ad
        path = _path_cells(apex, direction, dist)
        rr, cc = path[-1]
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        rays.append((cid, dist, path))
        if path[-1] in cell_to_comp:
            hit_components.add(cell_to_comp[path[-1]])

    out = g.copy()
    for _cid, dist, path in rays:
        for step, (r, c) in enumerate(path, start=1):
            out[r, c] = _color_for_step(dist, step)

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = LATE_COLOR

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


def _build_placements() -> tuple[dict[int, list[dict]], dict[int, dict[tuple[int, int], list[dict]]]]:
    placements_by_dist: dict[int, list[dict]] = {dist: [] for dist in DISTANCES}
    by_dist_dest: dict[int, dict[tuple[int, int], list[dict]]] = {
        dist: defaultdict(list) for dist in DISTANCES
    }
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                for dist in DISTANCES:
                    path = _path_cells((ar, ac), (dr, dc), dist)
                    if not _in_bounds_cells(path):
                        continue
                    placement = {
                        "apex": (ar, ac),
                        "dir": (dr, dc),
                        "dist": dist,
                        "dest": path[-1],
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                        "path": path,
                        "path_set": frozenset(path),
                        "interior": path[:-1],
                        "interior_set": frozenset(path[:-1]),
                    }
                    placements_by_dist[dist].append(placement)
                    by_dist_dest[dist][path[-1]].append(placement)
    return placements_by_dist, by_dist_dest


PLACEMENTS_BY_DIST, PLACEMENTS_BY_DIST_DEST = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _sprite_fits(
    placement: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> bool:
    if placement["halo"] & occupied_sprites:
        return False
    if placement["cell_set"] & occupied_paths:
        return False
    return True


def _placement_fits_bg(
    placement: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if not _sprite_fits(placement, occupied_sprites, occupied_paths):
        return False
    if placement["path_set"] & occupied_paths:
        return False
    if placement["path_set"] & occupied_sprites:
        return False
    if placement["path_set"] & reserved_bg:
        return False
    return True


def _placement_fits_hit(
    placement: dict,
    target_cell: tuple[int, int],
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if placement["dest"] != target_cell:
        return False
    if not _sprite_fits(placement, occupied_sprites, occupied_paths):
        return False
    if placement["interior_set"] & occupied_paths:
        return False
    if placement["interior_set"] & occupied_sprites:
        return False
    if placement["interior_set"] & reserved_bg:
        return False
    if target_cell in occupied_paths:
        return False
    return True


def _choose_unique(
    rng: random.Random,
    sprite_color: int,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    dist = COLOR_TO_DIST[sprite_color]
    candidates = PLACEMENTS_BY_DIST[dist][:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg:
            continue
        if _placement_fits_bg(placement, occupied_sprites, occupied_paths, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    sprite_color: int,
    target_sprite: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    dist = COLOR_TO_DIST[sprite_color]
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = PLACEMENTS_BY_DIST_DEST[dist][cell][:]
        rng.shuffle(candidates)
        for placement in candidates:
            if _placement_fits_hit(placement, cell, occupied_sprites, occupied_paths, reserved_bg):
                return placement, cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    colors = set(int(v) for v in np.unique(grid)) - {BG}
    if not colors <= set(SPRITE_COLORS):
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites:
        dist = COLOR_TO_DIST[sprite["color"]]
        for step, (r, c) in enumerate(sprite["placement"]["path"], start=1):
            out[r, c] = _color_for_step(dist, step)
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], LATE_COLOR)
    return out


def _choose_counts(rng: random.Random, n_sprites: int) -> tuple[int, list[int], list[int]]:
    e_hit = rng.randint(1, min(MAX_HIT_EVENTS, n_sprites - len(SPRITE_COLORS)))
    base_count = n_sprites - e_hit
    extras = [rng.choice(SPRITE_COLORS) for _ in range(n_sprites - len(SPRITE_COLORS))]
    rng.shuffle(extras)
    hit_colors = extras[:e_hit]
    base_colors = SPRITE_COLORS[:] + extras[e_hit:]
    rng.shuffle(base_colors)
    rng.shuffle(hit_colors)
    assert len(base_colors) == base_count
    return e_hit, base_colors, hit_colors


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    e_hit, base_colors, hit_colors = _choose_counts(rng, n_sprites)

    sprites: list[dict] = []
    occupied_sprites: set[tuple[int, int]] = set()
    occupied_paths: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    for color in base_colors:
        placement = _choose_unique(rng, color, occupied_sprites, occupied_paths, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "color": color, "target": placement["dest"], "target_kind": "bg"})
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        reserved_bg.add(placement["dest"])

    used_target_cells: set[tuple[int, int]] = set()
    target_indices = rng.sample(range(len(base_colors)), e_hit)
    rng.shuffle(target_indices)
    for color, idx in zip(hit_colors, target_indices):
        choice = _choose_hit(
            rng,
            color,
            sprites[idx],
            occupied_sprites,
            occupied_paths,
            reserved_bg,
            used_target_cells,
        )
        if choice is None:
            return None
        placement, cell = choice
        sprites.append(
            {
                "placement": placement,
                "color": color,
                "target": cell,
                "target_kind": "hit",
                "target_sprite": idx,
            }
        )
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        used_target_cells.add(cell)

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None
    if set(SPRITE_COLORS) - set(int(v) for v in np.unique(inp)):
        return None

    out = _render_output(inp, sprites)

    if np.array_equal(inp, out):
        return None
    if not TRAIL_COLORS <= set(int(v) for v in np.unique(out)):
        return None
    if np.count_nonzero(out == LATE_COLOR) < 3:
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
