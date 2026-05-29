# Variant of 409aa875
# Dimension: D16 — 投射物起始偏移 (path)
# Change: 投射物从顶点出发，落在顶点+5*方向处（含从顶点本身计数）... → 从精灵质心发射

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D16 variant:
# a) Concrete behavior:
#    - Objects stay the same: 16x16 grid, orange background 7, monochrome 3-pixel triangle sprites.
#    - Direction inference also stays the same: L-shapes fire diagonally, V-shapes fire orthogonally.
#    - The changed step is only the projectile start cell.
#      Old rule: landing = apex + 5*dir.
#      New rule: landing = round(centroid(sprite)) + 5*dir.
#      For L-shapes the rounded centroid usually stays on the corner pixel, so they often behave as before.
#      For V-shapes the rounded centroid is the empty center between the two diagonal legs, one cell behind the apex
#      along the firing axis; their landing cell shifts by exactly one grid step compared with the old rule.
#    - Small trace:
#      cells {(2,3),(3,2),(3,4)} form an up-firing V.
#      Old: apex=(2,3), dir=(-1,0), landing=(2,3)+5*(-1,0)=(-3,3) if extended; in-grid version shifts similarly.
#      New: centroid=(8/3,3), rounded start=(3,3), landing=(3,3)+5*(-1,0)=(-2,3). The destination moves down by 1.
# b) Complexity calibration:
#    - Only two shape families remain, so the variant is still human-trackable from a few examples.
#    - To keep the D16 change visible, the generator forces at least one V-shape whose centroid-based landing differs
#      from the apex-based landing and keeps the old landing cell empty; this guarantees that the variant is observable.
# c) Implicit assumptions audit for main():
#    - Component discovery still assumes sprites are 8-connected monochrome 3-pixel pieces; generator preserves that.
#    - Shape classification still assumes exactly one valid L-corner or one valid V-apex; generator only places canonical
#      shapes, so direction inference remains stable.
#    - The old code assumed "apex is the launch origin". That is the only invalid assumption under D16.
#      Replacing just the landing computation is safe because recoloring/collision logic depends only on destinations.
#    - Rounding must be deterministic. The code uses nearest-integer rounding with half-up semantics; for these 3-pixel
#      centroids only .0/.333/.667 occur, so there is no tie ambiguity in practice.
# d) Placement legality:
#    - Every placement is precomputed from apex+direction, then converted to centroid-start+direction.
#    - Generator checks: sprite cells in bounds, centroid-based destination in bounds, no 8-neighbor touching,
#      protected empty cells (future bg markers and the forced old-rule landing cell) stay empty.
# e) Visual observability:
#    - At least one V-shape is mandatory, and its new landing cell differs from the old-rule landing cell on the same
#      input. The generator rejects any sample where the new output equals the old apex-based output.
# f) Visual cleanliness:
#    - Sprites keep Chebyshev distance >= 2, so the new centroid-start idea does not create merged or unreadable shapes.
#    - Only the usual output markers appear: maroon 9 and optional blue 1.
# g) Color inference:
#    - No new color mapping is introduced. Colors only indicate hit/collision outcomes exactly as in the original task.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def _rounded_centroid(comp: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    n = len(comp)
    rr = sum(r for r, _ in comp) / n
    cc = sum(c for _, c in comp) / n
    return _round_half_up(rr), _round_half_up(cc)


def _triangle_apex_and_dir(comp: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[int, int]] | None:
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


def _solve(grid: np.ndarray, *, centroid_start: bool) -> np.ndarray:
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
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        apex, direction = ad
        start = _rounded_centroid(comp) if centroid_start else apex
        rr = start[0] + 5 * direction[0]
        cc = start[1] + 5 * direction[1]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangle sprites shoot from their rounded geometric centers; impacts tint sprites or background cells.

    Input structure:
    - The background is the most frequent color, and every foreground object is a 3-pixel monochrome triangle sprite.
    - Sprites are either axis-aligned L-shapes or diagonal V-shapes, exactly as in the original task.

    Rule:
    - Each sprite still determines its firing direction from shape alone: L-shapes shoot diagonally away from the missing
      corner, while V-shapes shoot along the axis opposite the open side.
    - The launch point is no longer the apex pixel. Compute the sprite's 3-pixel centroid, round it to the nearest grid
      cell, and place the projectile landing point at rounded-centroid + 5*direction.
    - A projectile landing on a sprite recolors that entire sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, that cell becomes blue (1); a single landing on an
      empty background cell paints it maroon (9).

    Key insight:
    - Orientation still gives the direction, but the path origin now depends on the whole 3-pixel geometry. V-shapes
      therefore land one step differently from the old apex-based rule, while L-shapes often do not.
    """
    return _solve(grid, centroid_start=True)


# --------------- generator (D16 centroid-start variant) ---------------

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
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 400


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3 pixel coords of a triangle sprite with given apex and firing direction."""
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
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                start = _rounded_centroid(cells)
                dest = (start[0] + PROJECTILE_DIST * dr, start[1] + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                old_dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "kind": "L" if abs(dr) + abs(dc) == 2 else "V",
                    "start": start,
                    "dest": dest,
                    "old_dest": old_dest if 0 <= old_dest[0] < H and 0 <= old_dest[1] < W else None,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
SENSITIVE_V_PLACEMENTS = [
    p
    for p in PLACEMENTS
    if p["kind"] == "V" and p["old_dest"] is not None and p["old_dest"] != p["dest"]
]


def _fits(placement: dict, occupied: set[tuple[int, int]], protected_empty: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & protected_empty)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_sensitive_v_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> dict | None:
    candidates = SENSITIVE_V_PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        old_dest = placement["old_dest"]
        if old_dest is None:
            continue
        if placement["dest"] in occupied or placement["dest"] in protected_empty:
            continue
        if old_dest in occupied or old_dest in protected_empty:
            continue
        if _fits(placement, occupied, protected_empty):
            return placement
    return None


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in protected_empty or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, protected_empty)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                if second["cell_set"] & protected_empty:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in protected_empty or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, protected_empty):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, protected_empty)]
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
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_remaining(rng: random.Random, n_other: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_other - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_other // 2) + 1):
            unique = n_other - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if e_hit + e_coll < 1:
                continue
            if unique + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, protected_empty) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, protected_empty)
        if choice is None:
            return False
        dest, pair = choice
        protected_empty.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, protected_empty) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, protected_empty)
        if placement is None:
            return False
        protected_empty.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, protected_empty) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, protected_empty, used_target_cells)
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


def _count_v_sprites(inp: np.ndarray) -> int:
    count = 0
    bg = _bg_color(inp)
    for col in sorted(set(int(v) for v in np.unique(inp)) - {bg}):
        for comp in _cc8(inp == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is not None and abs(ad[1][0]) + abs(ad[1][1]) == 1:
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    if n_sprites < 4:
        return None
    counts = _counts_for_remaining(rng, n_sprites - 1)
    if counts is None:
        return None
    e_hit, e_coll, _unique_other = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    protected_empty: set[tuple[int, int]] = set()

    anchor = _choose_sensitive_v_unique(rng, occupied, protected_empty)
    if anchor is None:
        return None
    sprites.append({"placement": anchor, "target": anchor["dest"], "target_kind": "bg"})
    occupied.update(anchor["cells"])
    protected_empty.add(anchor["dest"])
    protected_empty.add(anchor["old_dest"])

    if not _place_collisions(rng, e_coll, sprites, occupied, protected_empty):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if not _place_unique_landings(rng, base_needed, sprites, occupied, protected_empty):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, protected_empty):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, protected_empty):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None
    if _count_v_sprites(inp) < 1:
        return None

    out = _render_output(inp, sprites)
    old_out = _solve(inp, centroid_start=False)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, old_out):
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
