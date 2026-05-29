# Variant of 409aa875
# Dimension: D12 — 方向确定规则（形状语义） (path)
# Change: L形：对角线方向远离缺失的2x2角；V形：轴方向远离开口侧。方向完全由像素排列的几何结构决定... → 方向由精灵与最近边缘的关系决定

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D12 variant
# a) Concrete behavior and trace:
#    The visible sprites stay exactly the same as in the original task: every object is still a
#    monochrome 3-pixel triangle, either an axis-aligned L or a diagonal V, and projectiles still
#    land 5 cells away from the apex. The change is only in how the direction is chosen.
#    Original flow on one L example:
#      cells {(6,6),(6,7),(7,6)} -> corner/apex=(6,6) -> old dir=(-1,-1) from shape ->
#      landing cell (1,1).
#    Variant flow on the same pixels inside a 16x16 grid:
#      bbox distances are top=6, left=6, bottom=8, right=8 -> nearest edges are tied, so this
#      placement would be ambiguous and is excluded by the generator.
#      For a nearby legal example with cells {(4,6),(4,7),(5,6)}:
#      top=4, left=6, bottom=10, right=8 -> uniquely nearest edge is TOP ->
#      the shot goes AWAY from that edge, i.e. downward -> landing cell is apex+(5,0).
#    So the local shape still reveals the launch vertex, but no longer reveals the path.
#
# b) Complexity calibration:
#    The new rule has only four direction classes (away from top/bottom/left/right), which keeps
#    the hypothesis space small enough for ARC. To avoid making the puzzle too weak or visually
#    close to the original, the generator enforces both shape kinds in one grid and requires at
#    least one nearest-edge category to contain both an L and a V. That gives direct evidence that
#    shape no longer determines direction.
#
# c) Hidden assumptions from the original main():
#    - `_triangle_apex_and_dir(comp)` bundled two jobs: recovering the apex and inferring the
#      firing direction from the local geometry. In the variant, only the apex logic remains valid.
#    - `dest = apex + 5*dir` still holds unchanged, so downstream hit/collision logic can stay.
#    - The original solver tolerated every recognized shape because each valid shape implied a
#      unique direction. The variant needs an extra assumption: the nearest edge must be unique.
#      The solver returns `None` for ties, and the generator constructs only strict minima.
#    - `cell_to_comp[dest]` is still safe because all sprites keep the original non-touching
#      placement rule, so every occupied destination belongs to exactly one component.
#
# d) Placement legality:
#    The generator precomputes every in-bounds 3-pixel L/V placement, then keeps only those whose
#    bbox has a unique nearest edge and whose 5-step projectile landing cell remains in bounds.
#    Sprites are still separated by a one-cell halo, so components never merge and the solver's
#    8-connected component parsing remains correct.
#
# e) Visual observability:
#    Because the output only shows landing effects, the generator rejects degenerate samples where
#    the variant output equals the original shape-driven output on the exact same input. This makes
#    the D12 change observable rather than merely latent. It also enforces repeated support: at
#    least one edge class contains both L and V sprites, so the solver cannot rely on shape.
#
# f) Visual cleanliness:
#    The task keeps the original sparse 16x16 layout, small object count, and simple marker colors.
#    Collision markers remain single pixels, and sprite-hit recolors remain whole-component fills,
#    so there are never stacked overlays that obscure object identity.
#
# g) Color change inferability:
#    Colors are unchanged from the original task. Only two special output colors carry meaning:
#    maroon (9) for lone background landings and hit sprites, blue (1) for multi-projectile
#    collisions on empty background. No ordering or branching depends on color ids themselves.
#
# h) Variant metric used after implementation:
#    A generated sample is accepted only if `main(inp)` matches the constructed output AND the
#    original shape-driven solver would produce a different output. This is a direct anti-regression
#    check for the D12 behavior. Additional diagnostics later count how many sprites have
#    `new_dir != old_dir` and how edge classes are distributed.
#
# i) New solver assumptions to audit:
#    - Every non-bg component is still exactly one valid 3-pixel L or V.
#    - Every valid sprite has a unique apex under the old geometry rules.
#    - Every generated sprite has a unique nearest edge; ties are excluded in generation.
#    - Projectile destinations stay in bounds by construction.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex(comp: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {(r, c): sum(1 for n in neighbors4(r, c) if n in s) for r, c in comp}
    corners = [p for p, d in deg4.items() if d == 2]
    if len(corners) == 1:
        return corners[0]

    deg8 = {(r, c): sum(1 for n in neighbors8(r, c) if n in s) for r, c in comp}
    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) == 1:
        return apexes[0]
    return None


def _old_triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
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


def _nearest_edge_dir(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], h: int, w: int) -> tuple[str, tuple[int, int]] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    distances = {
        "top": min(rows),
        "bottom": (h - 1) - max(rows),
        "left": min(cols),
        "right": (w - 1) - max(cols),
    }
    best = min(distances.values())
    closest = [edge for edge, dist in distances.items() if dist == best]
    if len(closest) != 1:
        return None
    edge = closest[0]
    away = {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }
    return edge, away[edge]


def _triangle_apex_and_edge_dir(comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    apex = _triangle_apex(comp)
    if apex is None:
        return None
    edge_dir = _nearest_edge_dir(comp, h, w)
    if edge_dir is None:
        return None
    _edge, direction = edge_dir
    return apex, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites, but choose the path from global border proximity.

    Input structure:
    - The background is the most common color.
    - Every non-background 8-connected component is a monochrome 3-pixel triangle sprite, either an L or a V.
    - Local geometry still lets you recover the sprite's apex pixel, but it no longer determines the projectile direction.

    Rule:
    - Find each sprite's apex from its 3-pixel shape.
    - Measure the sprite's distance to the four outer grid edges using its occupied cells; the projectile travels AXIS-ALIGNED away from the uniquely nearest edge.
    - The landing cell is exactly 5 steps from the apex in that direction.
    - A landing on another sprite recolors that entire sprite to maroon (9); multiple projectiles sharing one empty background cell paint it blue (1); a lone empty-background landing paints maroon (9).

    Key insight: The triangle shape only provides the launch vertex. The path is controlled by the sprite's global position inside the frame, not by whether it looks like an L or a V.
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
        ad = _triangle_apex_and_edge_dir(comp, h, w)
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
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATIONS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 400


def _sprite_cells(apex: tuple[int, int], orientation: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = orientation
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
            for orientation in ORIENTATIONS:
                cells = _sprite_cells((ar, ac), orientation)
                if not _in_bounds_cells(cells):
                    continue
                edge_dir = _nearest_edge_dir(cells, H, W)
                if edge_dir is None:
                    continue
                edge, new_dir = edge_dir
                dest = (ar + PROJECTILE_DIST * new_dir[0], ac + PROJECTILE_DIST * new_dir[1])
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "shape": "L" if abs(orientation[0]) + abs(orientation[1]) == 2 else "V",
                    "orientation": orientation,
                    "old_dir": orientation,
                    "dir": new_dir,
                    "edge": edge,
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    targetable = frozenset(by_dest)
    for placement in placements:
        placement["targetable_cells"] = tuple(cell for cell in placement["cells"] if cell in targetable)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
TARGETABLE_CELLS = frozenset(PLACEMENTS_BY_DEST)


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
    *,
    require_targetable: bool = False,
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if require_targetable and not placement["targetable_cells"]:
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
    cells = list(target_sprite["placement"]["targetable_cells"])
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
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    if not all(comp.shape == (3, 2) for comp in comps):
        return False
    return all(_triangle_apex_and_edge_dir([(int(r), int(c)) for r, c in comp], H, W) is not None for comp in comps)


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


def _count_targetable_sprites(sprites: list[dict]) -> int:
    return sum(1 for sprite in sprites if sprite["placement"]["targetable_cells"])


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg, *, require_targetable=False) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg, require_targetable=require_targetable)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    candidate_ids = [idx for idx, sprite in enumerate(sprites) if sprite["placement"]["targetable_cells"]]
    if len(candidate_ids) < e_hit:
        return False
    hit_targets = rng.sample(candidate_ids, e_hit)
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
    bg_targets = Counter(sprite["target"] for sprite in sprites if sprite["target_kind"] == "bg")
    hit_sprite_ids = {sprite["target_sprite"] for sprite in sprites if sprite["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _old_main(grid: np.ndarray) -> np.ndarray:
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
        ad = _old_triangle_apex_and_dir(comp)
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
            out[r, c] = HIT_COLOR
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR
    return out


def _supports_variant(sprites: list[dict], inp: np.ndarray, out: np.ndarray) -> bool:
    shapes = {sprite["placement"]["shape"] for sprite in sprites}
    if shapes != {"L", "V"}:
        return False

    edge_to_shapes: dict[str, set[str]] = defaultdict(set)
    changed_dirs = 0
    for sprite in sprites:
        placement = sprite["placement"]
        edge_to_shapes[placement["edge"]].add(placement["shape"])
        if placement["dir"] != placement["old_dir"]:
            changed_dirs += 1

    if changed_dirs < 2:
        return False
    if not any(len(shapes_here) == 2 for shapes_here in edge_to_shapes.values()):
        return False
    if np.array_equal(_old_main(inp), out):
        return False
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

    base_needed = max(0, e_hit - _count_targetable_sprites(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg, require_targetable=True):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
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
    if not _supports_variant(sprites, inp, out):
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
