# Variant of 409aa875
# Dimension: D10 — 背景色与特殊标记色 (meta)
# Change: 背景固定为橙色(7)，命中色为栗色(9)，碰撞色为蓝色(1)... → 背景色不固定（由示例推断）

"""
Design analysis for the D10 variant.

a) Concrete behavior and trace:
   - The geometric rule is unchanged: each 3-pixel triangle sprite fires to apex + 5*dir.
   - The visible change is that the large uniform canvas color is no longer always 7. Each sample
     chooses one background color from a small non-special pool, while hit markers still use 9 and
     collision markers still use 1.
   - Example trace: in the original generator a grid with background 7 and a shooter landing on an
     empty cell produced a new 9 at that cell. In the variant the same geometry may live on a
     background-3 grid; the solver must first infer that 3 is the background, then the same landing
     produces a 9 on top of 3. The projectile logic is identical; only the preprocessing assumption
     "background is orange" is no longer valid.

b) Complexity calibration:
   - Only one new degree of freedom is introduced: per-example background color.
   - The pool is intentionally small and excludes the two special output colors {1, 9}; this keeps
     the visual difference obvious without turning color choice into an unrelated combinatorial task.
   - Consecutive generator calls rotate through shuffled background choices so small sets of examples
     are likely to exhibit the variation instead of accidentally repeating 7.

c) Hidden-assumption audit of main():
   - `bg = get_bg(grid)` assumes the background is the global majority color. This still holds because
     the generator fills the whole 16x16 canvas with one color and places only 4-9 tiny sprites.
   - Iterating over `unique(g) - {bg}` assumes every non-background component is a sprite candidate.
     That remains safe because inputs contain only the uniform background plus one shared sprite color.
   - Writing `out[r, c] = 1/9` assumes those colors are distinguishable from the background. The new
     generator therefore never selects background 1 or 9.

d) Placement legality:
   - Sprite geometry, spacing, in-bounds checks, and event construction remain the original output-
     first design, so no new overlap mode is introduced by the D10 change.
   - The generator still validates that all foreground components are 3-pixel 8-connected sprites and
     that every target cell is either an intended background landing or an intended sprite hit.

e) Visual observability:
   - Humans can still infer the projectile rule from shape/orientation and output markers.
   - The variant itself is directly visible because identical marker semantics appear on different
     canvas colors across examples; the only stable way to solve is "find the majority background
     color first", not "treat 7 as empty".

f) Visual cleanliness:
   - Background and sprite colors are both chosen outside {1, 9}, so blue collision marks and maroon
     hit marks stay visually isolated.
   - The original minimum Chebyshev separation is preserved, preventing merged sprites or cluttered
     target regions.

g) Color-change inferability:
   - The output color mapping is still simple and repeated across all samples: empty single-hit -> 9,
     empty multi-hit -> 1, hit sprite -> 9. Only the background identity varies.
"""

import random
from collections import Counter, defaultdict

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites on a variable-color background.

    Input structure:
    - The background is the most common color and is no longer fixed to orange; non-background 3-pixel
      components are monochrome triangle sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each triangle contributes an apex and a firing direction from its orientation: L-shapes shoot
      diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their
      opening.
    - The projectile lands exactly 5 cells from the apex in that direction.
    - Landing on a sprite recolors the whole hit sprite to maroon (9).
    - Landing on an empty background cell paints maroon (9) for a single shot, or blue (1) if two or
      more projectiles share that same empty destination.

    Key insight:
    - The geometric emission rule is unchanged, but the empty canvas color must be inferred from the
      example itself rather than assumed to be 7 before sprites are segmented.
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
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


# ---- canvas / palette ----
H = W = 16
HIT_COLOR = 9
COLLISION_COLOR = 1
NON_SPECIAL_COLORS = [0, 2, 3, 4, 5, 6, 7, 8]

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
MAX_ATTEMPTS = 300

_BG_CYCLE: list[int] = []
LAST_GENERATION_INFO: dict[str, int] = {}


def _next_bg_color() -> int:
    global _BG_CYCLE
    if not _BG_CYCLE:
        _BG_CYCLE = NON_SPECIAL_COLORS[:]
        random.shuffle(_BG_CYCLE)
    return _BG_CYCLE.pop()


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


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    """Placement is OK if its halo doesn't touch any occupied cell and its cells are free."""
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    """Pick a bg cell and two compatible shooters whose projectiles both land on it."""
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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, bg: int, sprite_color: int) -> bool:
    """Sanity check: correct shape, inferred bg, shared sprite color, and 3-pixel components."""
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != bg:
        return False
    non_bg_colors = sorted(set(int(v) for v in np.unique(grid)) - {bg})
    if non_bg_colors != [sprite_color]:
        return False
    comps = components_mask(grid != bg, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    """Pick (e_hit, e_coll, unique) event counts. Requires at least one hit or collision."""
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
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _gen_attempt(bg: int) -> np.ndarray | None:
    global LAST_GENERATION_INFO

    rng = random.Random()
    sprite_color = rng.choice([c for c in NON_SPECIAL_COLORS if c != bg])
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

    inp = np.full((H, W), bg, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, bg, sprite_color):
        return None

    out = _render_output(inp, sprites)

    if np.count_nonzero(out != bg) <= np.count_nonzero(inp != bg):
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

    LAST_GENERATION_INFO = {
        "bg": int(bg),
        "sprite_color": int(sprite_color),
        "n_sprites": int(n_sprites),
        "e_hit": int(e_hit),
        "e_coll": int(e_coll),
        "e_unique": int(remaining + base_needed),
    }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(len(NON_SPECIAL_COLORS)):
        bg = _next_bg_color()
        for _ in range(MAX_ATTEMPTS):
            grid = _gen_attempt(bg)
            if grid is not None:
                return grid
    raise RuntimeError(f"Failed to generate valid input after {len(NON_SPECIAL_COLORS) * MAX_ATTEMPTS} attempts")
