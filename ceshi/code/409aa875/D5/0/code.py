# Variant of 409aa875
# Dimension: D5 — 碰撞规则（投射物-投射物） (transform)
# Change: 两枚以上投射物落在同一空背景格时涂蓝(1)，单枚涂栗色(9)... → 多投射物碰撞产生爆炸扩散

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The input is unchanged: 16x16, bg=7, and every non-bg object is still a 3-pixel
#    triangle sprite (axis-aligned L or diagonal V). Each sprite still emits exactly one
#    projectile to apex + 5 * dir. The only changed output branch is the empty-background
#    multi-projectile collision: instead of coloring only the landing cell blue, the output
#    now shows a 5-cell blue cross centered at that landing cell. Example: if two shooters
#    land on empty (8,8), the old rule painted only (8,8)=1; the new rule paints
#    {(8,8),(7,8),(9,8),(8,7),(8,9)}=1. Sprite hits and lone bg landings stay maroon.
# b) Complexity calibration:
#    The change is visually meaningful only if collision samples actually exist and the full
#    cross is visible. The generator therefore requires at least one collision event and at
#    least one lone bg landing, so each sample contains both a blue cross and a maroon
#    singleton/hit cue for contrast. The collision multiplicity remains ">=2"; generation
#    still uses pairs, matching the original generator's difficulty level.
# c) Hidden assumptions in the original solver:
#    - `if len(shooters) >= 2: out[r, c] = 1` assumed a collision marker occupied one cell.
#      This no longer holds; the variant must paint a structured footprint.
#    - The original code could ignore cells adjacent to a collision point. Now those cells
#      become semantically important, so the generator must reserve them before placing
#      later sprites/markers.
#    - Unconditional recoloring of a single cell was harmless before. With a 5-cell cross,
#      careless painting could overwrite sprites or other markers. The generator therefore
#      guarantees every collision footprint is fully empty in the input and disjoint from
#      other reserved marker cells.
# d) Placement legality:
#    Every collision center is chosen away from the border so its 4-neighbor cross is fully
#    in-bounds. The generator reserves the whole cross footprint for each collision, and no
#    sprite cell may occupy any reserved marker cell. This keeps solver assumptions aligned
#    with generated inputs.
# e) Visual observability:
#    The blue cross is strictly more salient than the old blue dot, and the presence of at
#    least one maroon lone landing in the same sample makes the branching rule inferable:
#    one projectile on empty bg -> maroon point, multiple projectiles on empty bg -> blue
#    cross. No color-id arithmetic or invisible ordering is used.
# f) Visual cleanliness:
#    Collision footprints are kept disjoint from sprites and from other marker cells, so the
#    output is readable rather than layered. The rest of the original spacing constraints
#    between sprites are preserved.
# g) Color inferability:
#    Only two effect colors remain relevant: maroon for lone bg hits / hit sprites, blue for
#    multi-projectile bg collisions. The new behavior changes only the collision shape, not
#    the color vocabulary, so the rule can be learned from a few examples.
# h) Post-change solver assumptions checked against generation:
#    - Every non-bg component is still a valid 3-pixel triangle: enforced by placement and
#      `_validate_input`.
#    - Every collision footprint is obstacle-free in the input: enforced by reserved marker
#      cells during generation.
#    - Collision centers do not need edge clipping in generated data: enforced by choosing
#      only full in-bounds crosses, although the solver still clips safely for robustness.


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


def _collision_cross(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _paint_collision_cross(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = out.shape
    for r, c in _collision_cross(center):
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; mark hits, lone landings, and explosive collisions.

    Input structure:
    - The background is the most common color, and every foreground object is a 3-cell monochrome triangle sprite: either an axis-aligned L or a diagonal V.

    Rule:
    - Each triangle determines its apex and firing direction from its geometry: L-shapes shoot diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their open side.
    - Every projectile lands exactly 5 cells from the apex in that direction.
    - A landing on an existing sprite recolors the whole struck sprite to maroon (9), and a lone landing on empty background paints only that landing cell maroon (9).
    - If multiple projectiles land on the same empty background cell, that collision creates a blue (1) explosion cross centered on the landing cell, extending 1 step in the four cardinal directions.

    Key insight: Triangle orientation still reveals the hidden landing cell, but now repeated landings on the same empty spot are identified by a 5-cell blue cross rather than a single blue pixel.
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

    # Each triangle emits a projectile whose destination is 5 cells away from its apex.
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

    # Projectile hitting a sprite recolors the entire sprite to maroon (9).
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    # Projectile landings on empty background: lone impacts are maroon, repeated impacts explode into a blue cross.
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        if len(shooters) >= 2:
            _paint_collision_cross(out, dest, 1)
        else:
            r, c = dest
            out[r, c] = 9

    return out


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
    """Cell + 8-neighborhood, clipped to grid. Used to enforce Chebyshev>=2 separation."""
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _cross_is_full_in_bounds(center: tuple[int, int]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in _collision_cross(center))


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    """Precompute all (apex, dir) placements whose sprite and landing cell are in-bounds."""
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


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_marks: set[tuple[int, int]]) -> bool:
    """Placement is OK if its halo doesn't touch any occupied cell and its cells avoid reserved marker cells."""
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_marks)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _collision_footprint_clear(
    dest: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> bool:
    footprint = set(_collision_cross(dest))
    return _cross_is_full_in_bounds(dest) and not (footprint & occupied) and not (footprint & reserved_marks)


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    """Pick an empty bg cell and two compatible shooters whose projectiles both land on it."""
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_marks or dest in occupied:
            continue
        if not _collision_footprint_clear(dest, occupied, reserved_marks):
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_marks)]
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
    reserved_marks: set[tuple[int, int]],
) -> dict | None:
    """Pick a shooter whose landing cell is a fresh, empty bg cell."""
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_marks or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_marks):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    """Pick a shooter that lands on one of target_sprite's 3 pixels."""
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_marks)]
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
    """Pick (e_hit, e_coll, unique) counts with at least one visible collision and one lone bg landing."""
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(1, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_marks) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_marks)
        if choice is None:
            return False
        dest, pair = choice
        reserved_marks.update(_collision_cross(dest))
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_marks) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_marks)
        if placement is None:
            return False
        reserved_marks.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_marks) -> bool:
    """Add e_hit shooters, each targeting an existing sprite's pixels."""
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_marks, used_target_cells)
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
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        if count >= 2:
            _paint_collision_cross(out, dest, COLLISION_COLOR)
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

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_marks: set[tuple[int, int]] = set()

    # 1. Collision events first: their 5-cell output footprint is the tightest constraint.
    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_marks):
        return None

    # 2. Base sprites that hit-shooters may target.
    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_marks):
        return None

    # 3. Hit-shooters aimed at existing sprites.
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_marks):
        return None

    # 4. Remaining shooters with fresh bg landing cells.
    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_marks):
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
    if np.sum(out == COLLISION_COLOR) < 5:
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
