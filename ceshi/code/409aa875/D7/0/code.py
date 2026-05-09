# Variant of 409aa875
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出与输入尺寸相同（16x16），原地修改... → 裁剪到活动区域
#
# Design analysis
# a) Concrete behavior:
#    - Input stays a 16x16 orange-background field containing several 3-pixel triangle
#      sprites. Each sprite is still either an axis-aligned L or a diagonal V.
#    - The inner transformation is unchanged: every sprite fires to apex + 5 * dir,
#      hits recolor whole sprites to 9, repeated bg landings become 1, lone bg landings
#      become 9.
#    - The D7 change happens after all painting is complete. Instead of returning the
#      full 16x16 canvas, we keep only the smallest rectangle containing every non-bg
#      output pixel.
#    - Example trace: if the original logic would place a maroon landing mark at
#      row 3, col 12 and the leftmost sprite pixel is at row 5, col 4, the old output
#      kept all outer background margins; the new output crops to rows[min..max] and
#      cols[min..max] over all sprite pixels plus landing marks, so those margins vanish.
# b) Complexity calibration:
#    - Only one semantic degree of freedom changed: output shape. Projectile logic,
#      colors, sprite vocabulary, and event types are preserved, which keeps the task
#      recognizable instead of creating a second independent puzzle.
#    - To make the crop visually obvious, the generator only uses placements whose
#      sprite cells and landing cells stay at least one cell away from the border.
#      This ensures the final bbox is strictly smaller than 16x16 on successful samples.
# c) Old-code assumptions audited:
#    - `h, w = g.shape` remains valid because input is still 16x16.
#    - Component discovery by monochrome 8-connected sets still works because input
#      sprites are unchanged.
#    - Landing-cell painting in the old code wrote directly into a same-sized copy.
#      That assumption no longer holds for the final return value, so the new solver
#      must crop after all recoloring/marking is finished; cropping earlier would
#      change projectile coordinates and be wrong.
#    - The old generator compared `main(inp)` with an uncropped expected output.
#      That would now fail even when the transform is correct, so expected outputs are
#      cropped before verification.
# d) Placement legality:
#    - Sprites keep Chebyshev separation >= 2 via halo checks, so components never
#      merge accidentally.
#    - Every placement keeps both the sprite and its landing cell in bounds, and the
#      generator additionally requires them to lie inside a one-cell border margin so
#      the crop is nontrivial.
#    - Bg landing cells are reserved to avoid accidental reuse except when a deliberate
#      collision pair is being created.
# e) Visual observability:
#    - Across examples, users still see the same shot-resolution rule, but outputs are
#      now compact rectangles tightly hugging all active pixels. That difference is
#      visible immediately from 16x16 inputs versus smaller outputs.
# f) Visual cleanliness:
#    - No extra symbols are introduced. The crop only removes empty border area, so the
#      interior visual language stays sparse and readable.
# g) Color inference:
#    - No new color mapping is introduced. The original maroon/blue semantics remain,
#      avoiding any hidden dependence on color ids.
# h) Reference-task note:
#    - The prompt cites `e87109e9` as the tight-crop inspiration. In this workspace the
#      official JSON is present, but no readable solver `.py` for that task was found,
#      so the crop mechanism is implemented directly as a minimal-bbox post-process.
# i) New solver assumptions and generator guarantees:
#    - `main()` now assumes the final output contains at least one non-bg pixel if a
#      strict crop is desired. The generator guarantees many sprites, hence non-empty
#      activity; the helper still safely returns the full array if ever given an empty
#      case.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color, neighbors4, neighbors8


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


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return grid.copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites, then return only the active output window.

    Input structure:
    - The input is still a 16x16 grid whose most common color is background 7.
    - Every non-background 8-connected component is a monochrome 3-pixel triangle sprite,
      either an L-shape or a diagonal V-shape.

    Rule:
    - Each triangle's shape determines its apex and firing direction exactly as in the
      base task, and the projectile lands 5 cells away from that apex.
    - A landing on an existing sprite recolors the entire hit sprite to maroon (9).
    - A landing on empty background paints that cell maroon (9), unless at least two
      projectiles share the same empty landing cell, in which case that cell becomes blue (1).
    - After all marks are applied, discard every all-background border row and column:
      the output is the minimal rectangle containing all non-background pixels.

    Key insight:
    - Solving still requires reading each triangle's orientation to infer the hidden
      landing cells, but the returned canvas is no longer the original board; it is the
      tight crop of the fully resolved result.
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

    # Each triangle emits a projectile to apex + 5 * direction.
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

    # Projectile collision locations on empty background.
    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return _tight_crop_non_bg(out, bg)


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
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
ACTIVE_MARGIN = 1

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


def _cell_in_active_window(cell: tuple[int, int], margin: int = ACTIVE_MARGIN) -> bool:
    r, c = cell
    return margin <= r < H - margin and margin <= c < W - margin


def _cells_in_active_window(cells: tuple[tuple[int, int], ...], margin: int = ACTIVE_MARGIN) -> bool:
    return all(_cell_in_active_window(cell, margin) for cell in cells)


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
    """Precompute placements whose sprite and landing cell stay inside the crop-safe interior."""
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
                if not _cell_in_active_window(dest):
                    continue
                if not _cells_in_active_window(cells):
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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    """Sanity check: correct shape, bg, and exactly n_sprites 3-pixel components."""
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
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


def _render_output_full(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    """Apply the transform rule on the full 16x16 canvas."""
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _active_bbox_shape(grid: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return grid.shape
    r0, c0, r1, c1 = bbox_coords(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_touches_border(grid: np.ndarray, bg: int) -> bool:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1


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

    # 1. Collision events first (most constrained).
    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
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

    out_full = _render_output_full(inp, sprites)
    if _bbox_touches_border(out_full, BG):
        return None
    out = _tight_crop_non_bg(out_full, BG)

    if np.count_nonzero(out_full != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out_full == HIT_COLOR):
        return None
    if np.array_equal(inp, out_full):
        return None
    if out.shape == (H, W):
        return None
    if _active_bbox_shape(out_full, BG) == (H, W):
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
