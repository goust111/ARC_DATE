# Variant of 409aa875
# Dimension: D1 — 发射路径类型（对角线 vs 轴对齐） (path)
# Change: L形精灵沿对角线发射，V形精灵沿轴方向发射；方向由形状的缺口/开口方向决定... → 统一为直线射线（仅轴对齐）
#
# Design analysis
# ---------------
# variant_spec:
#   preserved:
#     - 16x16 fixed canvas, bg=7
#     - 4-9 monochrome 3-pixel sprites, all sharing one non-bg color
#     - the same two sprite families: axis-aligned L and diagonal V
#     - landing cell = apex + 5 * direction
#     - hit sprite => recolor entire sprite to 9
#     - >=2 projectiles on the same empty bg cell => 1, otherwise lone bg hit => 9
#   changed_core:
#     - every projectile path is now axis-aligned
#     - V-shapes keep their original "away from opening" axis direction
#     - L-shapes still derive an underlying orientation from the missing 2x2 corner,
#       but the old diagonal ray is snapped into a straight ray by a fixed 45-degree
#       clockwise turn: UL->U, UR->R, DR->D, DL->L
# behavior_trace:
#   original_local_example:
#     - L cells {(5,5),(6,5),(5,6)} have apex=(5,5), old direction=(-1,-1), landing=(0,0)
#   variant_local_example:
#     - the same L still has apex=(5,5), but (-1,-1) is snapped to (-1,0), so landing=(0,5)
#   observable_difference:
#     - the input sprite is unchanged, yet the output marker / hit location moves from a
#       diagonal target to a straight vertical target, which is visible from examples
# complexity_calibration:
#   - The task is simpler than the source because solvers no longer need to infer two path
#     families (diagonal vs axis) from the shape kind.
#   - It is not degenerate: both L and V remain present, and L orientations still map to
#     four distinct cardinal directions.
# implicit_assumptions_review:
#   - Solver assumption: every foreground component is a 3-pixel monochrome 8-connected
#     sprite. Generator preserves this with halo separation and component-count checks.
#   - Solver assumption: an L has exactly one pixel of 4-degree 2. Generator uses only the
#     canonical 2x2-minus-one shapes, so this remains valid.
#   - Solver assumption: a V has exactly one pixel with 8-degree 2 and yields an axis
#     direction from the average of its diagonal legs. Generator keeps the original V shapes.
#   - New solver assumption: L diagonal orientation is collapsed with a fixed clockwise
#     snap. Generator uses the exact inverse mapping when constructing L placements.
# placement_legality:
#   - Placements are precomputed only when sprite cells and landing cell are in bounds.
#   - Halo exclusion keeps Chebyshev distance >= 2 so components never merge.
#   - Reserved landing cells prevent accidental reuse except for deliberate collision events.
# visual_observability:
#   - Each generated grid must contain at least two L-shapes and at least one V-shape, so
#     examples actually demonstrate that both sprite families now shoot along straight rays.
#   - At least one hit/collision event is still required, so the output is richer than a set
#     of isolated single markers.
# cleanliness:
#   - Collisions are limited to at most two bg cells and hits to at most two sprites, keeping
#     the outputs legible.

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _l_diag_to_axis(diag_dir: tuple[int, int]) -> tuple[int, int] | None:
    """Collapse a diagonal L direction into a cardinal direction via a fixed clockwise snap."""
    dr, dc = diag_dir
    if abs(dr) != 1 or abs(dc) != 1:
        return None
    if dr == dc:
        return dr, 0
    return 0, dc


def _axis_to_l_diag(direction: tuple[int, int]) -> tuple[int, int]:
    mapping = {
        (-1, 0): (-1, -1),
        (0, 1): (-1, 1),
        (1, 0): (1, 1),
        (0, -1): (1, -1),
    }
    return mapping[direction]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Returns (apex_rc, dir_rc). Direction is always one of the 4 axis directions.
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4: dict[tuple[int, int], int] = {}
    deg8: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    # L-shape: infer the original diagonal-away vector from the missing 2x2 corner,
    # then snap that diagonal to a cardinal ray with a fixed clockwise turn.
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
        diag_dir = (-_sign(arms[0][0] + arms[1][0]), -_sign(arms[0][1] + arms[1][1]))
        axis_dir = _l_diag_to_axis(diag_dir)
        if axis_dir is None:
            return None
        return corner, axis_dir

    # V-shape: unchanged from the original task, already axis-aligned.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire straight projectiles from 3-pixel triangle sprites; mark hits in maroon or blue.

    Input structure:
    - bg = most common color. Non-bg 3-pixel sprites are monochrome L-shapes or diagonal V-shapes.

    Rule:
    - Every sprite emits along a cardinal ray and lands exactly 5 cells from its apex.
    - V-shapes still fire away from their opening; L-shapes use the missing-corner orientation too, but their former diagonal ray is collapsed into a straight axis direction.
    - A projectile that lands on a sprite recolors that entire sprite to maroon (9).
    - A projectile that lands on empty background paints maroon (9), except that two or more projectiles sharing the same empty landing cell paint blue (1).

    Key insight: Shape orientation still encodes the target, but both shape families now resolve to the same path type: a straight horizontal or vertical ray.
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


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KINDS = ("L", "V")

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MIN_L_SPRITES = 2
MIN_V_SPRITES = 1

# ---- generation retries ----
MAX_ATTEMPTS = 400


def _sprite_cells(kind: str, apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3 cells of an L or V sprite whose projectile goes in the given axis direction."""
    ar, ac = apex
    dr, dc = direction
    if kind == "L":
        ldr, ldc = _axis_to_l_diag(direction)
        return ((ar, ac), (ar - ldr, ac), (ar, ac - ldc))
    if kind == "V":
        if dr != 0:
            return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
        return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))
    raise ValueError(f"Unknown kind: {kind}")


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
    """Precompute all legal (kind, apex, dir) placements with in-bounds landing cells."""
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for kind in KINDS:
        for ar in range(H):
            for ac in range(W):
                for dr, dc in AXIS_DIRS:
                    dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                        continue
                    cells = _sprite_cells(kind, (ar, ac), (dr, dc))
                    if not _in_bounds_cells(cells):
                        continue
                    placement = {
                        "kind": kind,
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
    return all(comp.shape == (3, 2) for comp in comps)


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


def _kinds_ok(sprites: list[dict]) -> bool:
    counts = Counter(s["placement"]["kind"] for s in sprites)
    return counts["L"] >= MIN_L_SPRITES and counts["V"] >= MIN_V_SPRITES


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

    if not _kinds_ok(sprites):
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
