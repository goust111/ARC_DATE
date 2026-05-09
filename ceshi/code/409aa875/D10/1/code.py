# Variant of 409aa875
# Dimension: D10 — 背景色与特殊标记色 (meta)
# Change: 背景固定为橙色(7)，命中色为栗色(9)，碰撞色为蓝色(1)... → 碰撞色由碰撞方向决定
#
# Design analysis
# spec:
#   variant_focus: |
#     Keep the original triangle-sprite geometry and the 5-cell projectile rule, but
#     replace the single "collision on empty background = blue" marker with a
#     direction-coded palette:
#       horizontal collision -> blue(1)
#       vertical collision -> red(2)
#       diagonal collision -> green(3)
#   concretization_and_trace:
#     input:
#       - 16x16 grid with fixed orange background 7.
#       - 4-9 monochrome 3-pixel sprites, all sharing one non-7 color.
#       - Sprites are still only the original two canonical shapes:
#         L-shapes emit diagonally, V-shapes emit horizontally or vertically.
#     output:
#       - A sprite that is hit by any projectile is recolored to maroon 9.
#       - A lone projectile on empty background still paints maroon 9.
#       - A background cell targeted by 2+ projectiles is no longer uniformly blue.
#         Its color now reveals the shared direction family of those projectiles.
#     trace_example: |
#       Suppose two V-shapes fire upward and downward into the same empty cell.
#       Original rule: that cell becomes blue(1) just because two projectiles meet.
#       Variant rule: the same cell becomes red(2) because the shared travel family
#       is vertical. If two diagonal L-shape shots meet instead, the cell becomes
#       green(3). Horizontal V-shape collisions remain blue(1).
#   complexity_calibration:
#     - Only three collision colors are introduced, matching three visually distinct
#       direction families. This keeps the mapping learnable from a few examples.
#     - The generator forces at least one collision per sample and at least one
#       non-horizontal collision, so the variant does not collapse back to the
#       original all-blue-collision behavior.
#     - Mixed-family collisions are excluded from generation because they would make
#       the color mapping ambiguous to a human solver.
#   implicit_assumption_audit:
#     - _triangle_apex_and_dir still assumes every relevant component has exactly
#       three pixels in one of the two canonical shapes. That remains valid.
#     - main() originally stored only shooter ids per landing cell because collision
#       color was constant. The variant needs each shooter's direction too; otherwise
#       len(shooters) >= 2 is no longer enough information.
#     - The original unconditional assignment out[r, c] = 1 for background collisions
#       depended on a single global collision color. In the variant that would erase
#       the direction signal, so collision color must be derived per destination.
#     - main() now assumes every generated background collision has a single shared
#       direction family. generate_input() guarantees this by constructing each
#       collision from placements pre-filtered by family.
#   placement_legality:
#     - Sprite cells must stay in-bounds and remain Chebyshev-separated by at least 2.
#     - Every projectile landing cell must stay in-bounds.
#     - Background collision cells are reserved so no sprite occupies them and no
#       second event reuses them.
#     - Hit events still target an existing sprite pixel; collision events target an
#       empty background cell; lone landings target a fresh empty background cell.
#   visual_observability:
#     - Direction families are already visible in sprite orientation, so changing the
#       collision marker color adds an observable second channel rather than a hidden
#       metadata trick.
#     - Requiring at least one vertical or diagonal collision per sample ensures the
#       output visibly differs from the source task on the same input distribution.
#   visual_cleanliness:
#     - Collision markers remain single isolated cells on the orange background.
#     - Sprite spacing is unchanged, so recolored sprites and colored markers do not
#       merge into larger confusing components.
#   color_inference:
#     - Only three collision colors are used, and each is tied to one geometric
#       family repeated across samples. Humans can compare sprite orientation with the
#       colored collision point to infer the mapping.
#   solver_generator_contract:
#     - main() treats same-family multi-hit background cells as directed collisions.
#     - generate_input() guarantees at least one such event per sample and validates
#       the result by re-running main() on the generated input.

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _dir_kind(direction: tuple[int, int]) -> str:
    dr, dc = direction
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diagonal"


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
        # Projectile goes in the diagonal direction opposite the missing 2x2 corner.
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
    # The direction is the axis direction opposite the average of the two diagonal legs.
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _collision_color_from_dirs(directions: list[tuple[int, int]]) -> int:
    kinds = {_dir_kind(direction) for direction in directions}
    if len(kinds) == 1:
        return COLLISION_COLORS[next(iter(kinds))]
    return HIT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; encode empty-space collisions by direction family.

    Input structure:
    - bg = most common color, fixed to orange in generated tasks. Non-bg objects are monochrome 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each sprite has an apex and an emission direction determined by shape: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite their opening.
    - A projectile lands exactly 5 cells from the apex in that direction.
    - If the landing cell lies on a sprite, that entire hit sprite is recolored to maroon (9).
    - If a single projectile lands on empty background, that background cell becomes maroon (9).
    - If multiple projectiles land on the same empty background cell, the marker color reveals their shared direction family: horizontal -> blue (1), vertical -> red (2), diagonal -> green (3).

    Key insight: Triangle orientation still determines where each projectile lands, but now multi-projectile background collisions must be grouped by direction family before coloring the landing cell.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    # Find all monochrome components for each non-bg color.
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    # Each triangle emits a projectile 5 cells from its apex; collisions need shooter directions.
    dest_to_shots: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), direction = ad
        rr = ar + 5 * direction[0]
        cc = ac + 5 * direction[1]
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shots[(rr, cc)].append((cid, direction))

    hit_components: set[int] = set()
    for dest in dest_to_shots:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    # Projectile hitting a sprite recolors the entire sprite to maroon (9).
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    # Projectile collision locations on empty background.
    for (r, c), shots in dest_to_shots.items():
        if (r, c) in cell_to_comp:
            continue
        if len(shots) >= 2:
            out[r, c] = _collision_color_from_dirs([direction for _, direction in shots])
        else:
            out[r, c] = HIT_COLOR

    return out


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLORS = {
    "horizontal": 1,
    "vertical": 2,
    "diagonal": 3,
}
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
COLLISION_KINDS = ("horizontal", "vertical", "diagonal")
NON_HORIZONTAL_COLLISION_KINDS = ("vertical", "diagonal")

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MIN_COLL_EVENTS = 1
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 300


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3-pixel coords of a triangle sprite with the given apex and firing direction."""
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
    """Cell + 8-neighborhood, clipped to the grid. Used to enforce Chebyshev >= 2 separation."""
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[
    list[dict],
    dict[tuple[int, int], list[dict]],
    dict[tuple[tuple[int, int], str], list[dict]],
    dict[str, list[tuple[int, int]]],
]:
    """Precompute all placements whose sprite cells and landing cell are in-bounds."""
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_dest_kind: dict[tuple[tuple[int, int], str], list[dict]] = defaultdict(list)
    dests_by_kind: dict[str, set[tuple[int, int]]] = {kind: set() for kind in COLLISION_KINDS}
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                dir_kind = _dir_kind((dr, dc))
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dir_kind": dir_kind,
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                by_dest_kind[(dest, dir_kind)].append(placement)
                dests_by_kind[dir_kind].add(dest)
    return placements, by_dest, by_dest_kind, {kind: list(dests) for kind, dests in dests_by_kind.items()}


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_DEST_KIND, DESTS_BY_KIND = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    """Placement is OK if its halo avoids occupied cells and its cells avoid reserved bg markers."""
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    collision_kind: str,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    """Pick an empty bg cell and two compatible shooters from one direction family."""
    dests = DESTS_BY_KIND[collision_kind][:]
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [
            placement
            for placement in PLACEMENTS_BY_DEST_KIND[(dest, collision_kind)]
            if _fits(placement, occupied, reserved_bg)
        ]
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
    """Pick a shooter whose landing cell is a fresh, empty bg cell."""
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
    """Pick a shooter that lands on one of target_sprite's 3 pixels."""
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [placement for placement in PLACEMENTS_BY_DEST[cell] if _fits(placement, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    """Sanity check: correct shape, bg, and exactly n_sprites 3-pixel components."""
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    """Pick (e_hit, e_coll, unique) event counts with at least one collision and one maroon effect."""
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(MIN_COLL_EVENTS, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_hit < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _plan_collision_kinds(rng: random.Random, e_coll: int) -> list[str]:
    """Guarantee at least one non-horizontal collision so the variant is visibly active."""
    kinds = [rng.choice(NON_HORIZONTAL_COLLISION_KINDS)]
    while len(kinds) < e_coll:
        unused = [kind for kind in COLLISION_KINDS if kind not in kinds]
        if unused:
            kinds.append(rng.choice(unused))
        else:
            kinds.append(rng.choice(COLLISION_KINDS))
    rng.shuffle(kinds)
    return kinds


def _place_collisions(rng, collision_kinds, sprites, occupied, reserved_bg) -> bool:
    for collision_kind in collision_kinds:
        choice = _choose_collision_pair(rng, collision_kind, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append(
                {
                    "placement": placement,
                    "target": dest,
                    "target_kind": "bg",
                    "dir_kind": placement["dir_kind"],
                }
            )
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
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
                "dir_kind": placement["dir_kind"],
            }
        )
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    """Add e_hit shooters, each targeting an existing sprite's pixels."""
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
    """Apply the transform rule to produce the expected output grid."""
    out = inp.copy()
    bg_targets: dict[tuple[int, int], list[str]] = defaultdict(list)
    hit_sprite_ids = {sprite["target_sprite"] for sprite in sprites if sprite["target_kind"] == "hit"}
    for sprite in sprites:
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]].append(sprite["dir_kind"])
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, kinds in bg_targets.items():
        if len(kinds) >= 2 and len(set(kinds)) == 1:
            out[dest] = COLLISION_COLORS[kinds[0]]
        else:
            out[dest] = HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts
    collision_kinds = _plan_collision_kinds(rng, e_coll)

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    # 1. Directed collision events first; they are the most constrained placements.
    if not _place_collisions(rng, collision_kinds, sprites, occupied, reserved_bg):
        return None

    # 2. Base sprites that hit-shooters will target. Need enough existing sprites for e_hit picks.
    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg):
        return None

    # 3. Hit-shooters aimed at existing sprites.
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    # 4. Remaining shooters with fresh bg landing cells.
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
    if not np.any((inp == BG) & np.isin(out, [COLLISION_COLORS["vertical"], COLLISION_COLORS["diagonal"]])):
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
