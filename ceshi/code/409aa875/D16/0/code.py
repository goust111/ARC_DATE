# Variant of 409aa875
# Dimension: D16 — 投射物起始偏移 (path)
# Change: 投射物从顶点出发，落在顶点+5*方向处（含从顶点本身计数）... → 从精灵外1格开始计数

# Design analysis for the D16 variant
# - Source-task check: this variant has no source_task, so there is no extra solver to inspect.
# - a) Concrete behavior:
#   Every non-background component is still a 3-pixel monochrome triangle sprite, either an L
#   or a diagonal V. Shape-to-direction inference is unchanged. The only changed stage is the
#   landing-cell calculation: instead of using apex + 5*dir directly, first walk from the apex
#   along dir until leaving the shooter sprite, then move 5 more cells. For these canonical
#   3-pixel shapes the ray exits immediately, so the visible destination becomes apex + 6*dir.
#   Example: an up-left-firing L with apex at (8,8) used to mark/hit (3,3); now it marks/hits
#   (2,2). The recolor / collision rules downstream stay identical.
# - b) Complexity calibration:
#   The puzzle remains a single shared-color scene with 4-9 sprites and the same two triangle
#   families, so recognition complexity does not increase. To keep D16 observable, the generator
#   rejects any sample where the old and new solvers produce the same output on the same input.
#   This prevents accidental degeneration into layouts where the +1 shift is visually irrelevant.
# - c) Hidden assumptions in the original main():
#   `_triangle_apex_and_dir` only assumes canonical 3-pixel shapes; still valid.
#   Component gathering by color still works because sprites remain monochrome.
#   The line `rr = ar + 5*dr; cc = ac + 5*dc` depended on the old path origin and must change.
#   The later `dest in cell_to_comp` hit test still remains valid once the new destination is
#   computed, because hit semantics did not change.
# - d) Placement legality:
#   The generator must precompute placements with the new landing cell in bounds. Since sprites
#   remain separated by Chebyshev distance >= 2, the first cell just outside a sprite in its
#   firing direction is guaranteed not to belong to that sprite or touch another sprite. The
#   generator still reserves empty landing cells for bg events and only permits hit events whose
#   new landing cell lies on an already placed target sprite.
# - e) Visual observability:
#   The only latent change is path origin, so samples must make the one-cell shift matter. The
#   generator enforces `variant_output != legacy_output`, which guarantees a human would see at
#   least one changed marked/hit location caused specifically by the D16 offset.
# - f) Visual cleanliness:
#   No extra symbols are introduced. Separation, fixed-size sprites, and sparse landing markers
#   keep outputs readable; collisions remain a single blue cell and hits remain whole-sprite
#   recolors, so overlap depth is still low.
# - g) Color inference:
#   Colors keep the original semantics: input sprites share one non-bg color, output uses maroon
#   (9) for lone bg hits and hit sprites, blue (1) for multi-projectile bg collisions. No hidden
#   color ordering or color-id-dependent behavior is introduced.
# - 9) New assumptions audit:
#   The new solver assumes the firing ray leaves the shooter after finitely many steps; the
#   generator guarantees this by only using the same canonical 3-pixel L/V sprites. The solver
#   also assumes landing effects depend only on the final cell, not on intermediate ray cells;
#   the generator follows that same contract when constructing hit and collision events.

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


PATH_STEPS = 5
LEGACY_APEX_DIST = 5


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


def _landing_after_exit(
    cell_set: set[tuple[int, int]],
    apex: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    """Walk to the first non-sprite cell on the ray, then move PATH_STEPS more cells."""
    rr, cc = apex
    dr, dc = direction
    while (rr, cc) in cell_set:
        rr += dr
        cc += dc
    return rr + PATH_STEPS * dr, cc + PATH_STEPS * dc


def _landing_legacy(_cell_set: set[tuple[int, int]], apex: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return apex[0] + LEGACY_APEX_DIST * dr, apex[1] + LEGACY_APEX_DIST * dc


def _solve_with_landing(
    grid: np.ndarray,
    landing_fn,
) -> np.ndarray:
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
        rr, cc = landing_fn(set(comp), apex, direction)
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
    Fire projectiles from 3-pixel triangle sprites; mark the shifted landing sites and hits.

    Input structure:
    - The background is the most common color. Every foreground component is a 3-cell monochrome
      triangle sprite, either an axis-aligned L or a diagonal V.

    Rule:
    - Infer each sprite's apex and firing direction from its shape: L-sprites shoot diagonally
      away from the missing 2x2 corner, while V-sprites shoot along the axis opposite the opening.
    - Do not count from the apex itself. First step from the apex along the firing ray until you
      reach the first cell outside that sprite; from that outside cell, advance 5 more cells to
      get the landing site.
    - If that landing site lies on a sprite, recolor the entire hit sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, paint that cell blue (1);
      otherwise a lone projectile on empty background paints maroon (9).

    Key insight: Orientation still determines the ray, but D16 changes where the 5-step count
    begins: the shot starts just outside the shooter, not on the apex, so every landing site is
    shifted one cell farther along the same direction.
    """
    return _solve_with_landing(grid, _landing_after_exit)


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 500


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3-pixel sprite whose inferred firing direction is `direction`."""
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


def _variant_dest(apex: tuple[int, int], direction: tuple[int, int], cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return _landing_after_exit(set(cells), apex, direction)


def _legacy_dest(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return apex[0] + LEGACY_APEX_DIST * dr, apex[1] + LEGACY_APEX_DIST * dc


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    """Precompute all placements whose sprite and D16 landing cell are in-bounds."""
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                apex = (ar, ac)
                direction = (dr, dc)
                cells = _sprite_cells(apex, direction)
                if not _in_bounds_cells(cells):
                    continue
                dest = _variant_dest(apex, direction, cells)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                placement = {
                    "apex": apex,
                    "dir": direction,
                    "dest": dest,
                    "legacy_dest": _legacy_dest(apex, direction),
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
    """Pick a bg cell and two compatible shooters whose D16 projectiles both land on it."""
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
    """Pick a shooter whose D16 landing cell is a fresh, empty bg cell."""
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
    """Pick a shooter whose D16 landing cell lands on one target sprite pixel."""
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
    """Add e_hit shooters, each targeting an existing sprite pixel under the D16 rule."""
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
        solved_variant = main(inp)
        solved_legacy = _solve_with_landing(inp, _landing_legacy)
    except Exception:
        return None
    if not np.array_equal(solved_variant, out):
        return None
    if np.array_equal(solved_variant, solved_legacy):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
