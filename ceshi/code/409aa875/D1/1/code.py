# Variant of 409aa875
# Dimension: D1 — 发射路径类型（对角线 vs 轴对齐） (path)
# Change: L形精灵沿对角线发射，V形精灵沿轴方向发射；方向由形状的缺口/开口方向决定... → 统一为对角线射线（仅对角线）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D1 variant
# a) Concrete behavior
#    - Input objects stay the same: monochrome 3-cell L shapes and 3-cell V shapes on a 16x16 bg=7 grid.
#    - L sprites keep the original rule: the missing corner of the 2x2 box defines one diagonal shot.
#    - V sprites are the only changed case. Their opening is still read as up/right/down/left, but the shot is now
#      diagonal-only via a fixed orientation lookup:
#        up-opening -> up-left, right-opening -> up-right, down-opening -> down-right, left-opening -> down-left.
#    - Example trace: the V {(5,5),(6,4),(6,6)} originally opened upward and shot to (0,5); now it lands at (0,0).
#      The rest of the transform is unchanged: sprite hits recolor whole targets to 9, shared empty landings become 1,
#      lone empty landings become 9.
# b) Complexity calibration
#    - The change is visible because V landings move from axis rays to dispersed diagonal corners.
#    - The lookup has only four cases and repeats across samples, so it is inferable without introducing hidden state.
# c) Old implicit assumptions that must change
#    - The original V branch in the solver assumed the averaged leg direction always collapsed to a 4-neighbor vector.
#      That is no longer the final answer; we now compute that axis opening first and then map it to a diagonal.
#    - The original generator encoded V placements through axis directions. That would silently keep producing old-path
#      examples, so placement enumeration must become kind-aware with diagonal destinations for both shape families.
# d) Placement legality
#    - Every sprite is still 8-connected, size 3, in bounds, and Chebyshev-separated by at least 2 via halo checks.
#    - Every projectile destination is prevalidated to stay in bounds, and hit/collision planning still uses output-first
#      construction followed by `main(input)` verification.
# e) Visual observability
#    - Each accepted grid must contain both L and V sprites and the new output must differ from the original-path output.
#      This prevents degeneration into all-L examples or accidental cases where the V-path change is invisible.
# f) Visual cleanliness
#    - We keep the original sparse counts, single sprite color, and one-cell landing markers so the diagonal spread stays
#      easy to read rather than cluttered.
# g) Color inference
#    - No new color logic is introduced. Only 9 and 1 are written by the transform, exactly as in the base task.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _v_axis_to_diag(axis_dir: tuple[int, int]) -> tuple[int, int]:
    dr, dc = axis_dir
    if abs(dr) + abs(dc) != 1:
        raise ValueError(f"Expected axis direction, got {axis_dir}")
    if dr != 0:
        return dr, dr
    return -dc, dc


DIAG_TO_V_AXIS = {
    (-1, -1): (-1, 0),
    (-1, 1): (0, 1),
    (1, 1): (1, 0),
    (1, -1): (0, -1),
}


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Returns (apex_rc, dir_rc). In this variant both L and V sprites emit along diagonals.
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    # L-shape: same as the base task, shooting away from the missing 2x2 corner.
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
        dr = -_sign(arms[0][0] + arms[1][0])
        dc = -_sign(arms[0][1] + arms[1][1])
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        return corner, (dr, dc)

    # V-shape: read the opening axis from the two diagonal legs, then convert that orientation to a diagonal shot.
    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    if any(abs(dr) != 1 or abs(dc) != 1 for dr, dc in (d1, d2)):
        return None
    axis_dir = (-_sign(d1[0] + d2[0]), -_sign(d1[1] + d2[1]))
    if abs(axis_dir[0]) + abs(axis_dir[1]) != 1:
        return None
    return apex, _v_axis_to_diag(axis_dir)


def _triangle_apex_and_dir_original(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Base-task path logic kept only for generator-side regression checks.
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
        dr = -_sign(arms[0][0] + arms[1][0])
        dc = -_sign(arms[0][1] + arms[1][1])
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    axis_dir = (-_sign(d1[0] + d2[0]), -_sign(d1[1] + d2[1]))
    if abs(axis_dir[0]) + abs(axis_dir[1]) != 1:
        return None
    return apex, axis_dir


def _solve_with_rule(
    grid: np.ndarray,
    detector,
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
        ad = detector(comp)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; mark impacts with maroon or blue.

    Input structure:
    - The background is the most common color. Every foreground component is a monochrome 3-cell triangle sprite, either
      an axis-aligned L or a diagonal-legged V.

    Rule:
    - Each sprite has an apex and a shooting direction derived from its geometry.
    - L sprites still fire diagonally away from the missing corner of their 2x2 box.
    - V sprites now also fire diagonally: first read the opening as up/right/down/left, then map those four cases to
      up-left / up-right / down-right / down-left respectively.
    - The projectile lands exactly 5 cells from the apex. Hitting a sprite recolors that whole sprite to 9; multiple
      projectiles on the same empty background cell paint 1; a lone empty-cell landing paints 9.

    Key insight:
    - Shape orientation still encodes the target, but in this variant every valid ray is diagonal, so V sprites must be
      interpreted through a fixed opening-to-diagonal lookup instead of the original axis shot.
    """
    return _solve_with_rule(grid, _triangle_apex_and_dir)


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SHAPE_KINDS = ["L", "V"]

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 400


def _sprite_cells(
    apex: tuple[int, int],
    direction: tuple[int, int],
    kind: str,
) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if (dr, dc) not in DIAG_TO_V_AXIS:
        raise ValueError(f"Expected diagonal direction, got {direction}")
    if kind == "L":
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if kind == "V":
        axis_dr, axis_dc = DIAG_TO_V_AXIS[(dr, dc)]
        if axis_dr != 0:
            return ((ar, ac), (ar - axis_dr, ac - 1), (ar - axis_dr, ac + 1))
        return ((ar, ac), (ar - 1, ac - axis_dc), (ar + 1, ac - axis_dc))
    raise ValueError(f"Unknown sprite kind: {kind}")


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
            for direction in DIAG_DIRS:
                dest = (ar + PROJECTILE_DIST * direction[0], ac + PROJECTILE_DIST * direction[1])
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                for kind in SHAPE_KINDS:
                    cells = _sprite_cells((ar, ac), direction, kind)
                    if not _in_bounds_cells(cells):
                        continue
                    placement = {
                        "apex": (ar, ac),
                        "dir": direction,
                        "kind": kind,
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
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(len(comp) == 3 and _triangle_apex_and_dir(comp) is not None for comp in comps)


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


def _kind_counts(sprites: list[dict]) -> Counter:
    return Counter(sprite["placement"]["kind"] for sprite in sprites)


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

    kinds = _kind_counts(sprites)
    if kinds["L"] == 0 or kinds["V"] == 0:
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
        original_out = _solve_with_rule(inp, _triangle_apex_and_dir_original)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(original_out, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
