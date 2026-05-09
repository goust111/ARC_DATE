# Variant of 409aa875
# Dimension: D18 — 终止/目标条件 (dst)
# Change: 无显式目标；投射物飞行固定距离后落地，效果取决于落点内容... → 到达边界停止

# Design analysis for the D18 variant:
# - Concrete behavior:
#   The sprites stay exactly the same as the original task: each non-bg object is a
#   monochrome 3-pixel triangle, either an axis-aligned L or a diagonal V. The
#   orientation still determines one of eight directions. The only rule change is
#   the stopping condition: instead of landing at apex + 5*dir, the projectile keeps
#   moving in that direction until the last in-bounds cell on that ray, which is
#   necessarily a border cell. The output therefore shows effects only on border
#   cells or on border-touching sprites that get hit there.
# - Behavior trace:
#   Example: an L-sprite with apex (7,7) firing (-1,-1) used to land at (2,2). In
#   this variant it lands at (0,0), because the ray keeps going until the boundary.
#   If another shooter also points to (0,0), the old task would produce two separate
#   mid-grid landings, while the variant paints a single blue corner cell. This is
#   the exact D18 change.
# - Complexity calibration:
#   Keeping the original shape vocabulary preserves the object-recognition burden.
#   Restricting the visible endpoints to the border makes the change legible without
#   adding new object types. To keep the new rule visible, the generator guarantees
#   at least one non-collision singleton border landing and at least one projectile
#   whose boundary travel distance is >5, so the examples do not collapse back to
#   the old fixed-distance behavior.
# - Hidden assumptions audit:
#   The original solver assumed that a single arithmetic expression (`apex + 5*dir`)
#   fully determines the destination. That assumption is invalid now; every place
#   that depended on it must instead compute the maximal in-bounds step count along
#   the ray. The hit/paint logic remains valid once `dest` is replaced. A subtle
#   generator assumption also changes: hit events can now only target border cells,
#   so target sprites must expose at least one border pixel.
# - Placement legality:
#   The generator precomputes only placements whose 3-pixel sprite is in-bounds and
#   whose boundary travel distance is at least 2, avoiding zero/one-step degenerate
#   self-hits. Sprites still obey Chebyshev separation >= 2. Background landing
#   cells are reserved so later sprites cannot overwrite them, and hit-shooters are
#   only chosen for existing border pixels.
# - Visual observability and cleanliness:
#   Initial sprite colors exclude 1 and 9 so new markers/recolors remain visible.
#   Border landings are sparse (4-9 sprites total, at most two shared endpoints),
#   so the output reads as a set of projections from interior triangles to the edge
#   rather than a noisy border fill.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


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
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
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


def _boundary_dest(
    apex: tuple[int, int], direction: tuple[int, int], h: int, w: int
) -> tuple[tuple[int, int], int]:
    ar, ac = apex
    dr, dc = direction
    row_steps = h + w
    col_steps = h + w
    if dr > 0:
        row_steps = h - 1 - ar
    elif dr < 0:
        row_steps = ar
    if dc > 0:
        col_steps = w - 1 - ac
    elif dc < 0:
        col_steps = ac
    steps = min(row_steps, col_steps)
    return (ar + steps * dr, ac + steps * dc), int(steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project triangle emissions to the border; color the border endpoint or struck sprite.

    Input structure:
    - bg is the most common color.
    - Every non-bg 8-connected component is a monochrome 3-pixel triangle: either an L-shape or a diagonal V-shape.

    Rule:
    - Each triangle's geometry determines an apex and one emission direction exactly as in the original task.
    - The projectile no longer stops after 5 cells; it keeps moving until the last in-bounds cell on that ray, which is a border cell.
    - If that border endpoint lies on a sprite, the entire hit sprite is recolored to maroon (9).
    - If multiple projectiles share the same empty border cell, that cell becomes blue (1); a lone empty endpoint becomes maroon (9).

    Key insight: Shape still encodes direction, but the visible target is now the ray's boundary endpoint rather than a fixed-radius landing cell.
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
        dest, _ = _boundary_dest(ad[0], ad[1], h, w)
        dest_to_shooters.setdefault(dest, []).append(cid)

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
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
MIN_TRAVEL = 2
LONG_TRAVEL_MIN = 6

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 500


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


def _edge_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, c) for r, c in cells if r in (0, H - 1) or c in (0, W - 1))


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                dest, dist = _boundary_dest((ar, ac), (dr, dc), H, W)
                if dist < MIN_TRAVEL:
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "dist": dist,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "edge_cells": _edge_cells(cells),
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
    *,
    require_targetable: bool = False,
    require_long: bool = False,
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if require_targetable and not placement["edge_cells"]:
            continue
        if require_long and placement["dist"] < LONG_TRAVEL_MIN:
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
    cells = list(target_sprite["placement"]["edge_cells"])
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
            if unique < 1:
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


def _place_targetable_bases(rng, need, sprites, occupied, reserved_bg, need_long) -> tuple[bool, bool, int]:
    used = 0
    for _ in range(need):
        placement = _choose_unique(
            rng,
            occupied,
            reserved_bg,
            require_targetable=True,
            require_long=need_long,
        )
        if placement is None and need_long:
            placement = _choose_unique(
                rng,
                occupied,
                reserved_bg,
                require_targetable=True,
                require_long=False,
            )
        if placement is None:
            return False, need_long, used
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
        need_long = need_long and placement["dist"] < LONG_TRAVEL_MIN
        used += 1
    return True, need_long, used


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    eligible = [idx for idx, s in enumerate(sprites) if s["placement"]["edge_cells"]]
    if len(eligible) < e_hit:
        return False
    hit_targets = rng.sample(eligible, e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg, *, force_long_first: bool = False) -> bool:
    need_long = force_long_first
    for _ in range(n):
        placement = _choose_unique(
            rng,
            occupied,
            reserved_bg,
            require_long=need_long,
        )
        if placement is None and need_long:
            return False
        if placement is None:
            placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
        need_long = False
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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    long_shots = 0
    edge_touching = 0
    total = 0
    boundary_diff_from_fixed5 = 0
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            (ar, ac), (dr, dc) = ad
            dest, dist = _boundary_dest((ar, ac), (dr, dc), *g.shape)
            total += 1
            if dist >= LONG_TRAVEL_MIN:
                long_shots += 1
            if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in comp):
                edge_touching += 1
            fixed5 = (ar + 5 * dr, ac + 5 * dc)
            if not (0 <= fixed5[0] < g.shape[0] and 0 <= fixed5[1] < g.shape[1]) or fixed5 != dest:
                boundary_diff_from_fixed5 += 1
    return {
        "sprites": total,
        "long_shots": long_shots,
        "edge_touching_sprites": edge_touching,
        "boundary_diff_from_fixed5": boundary_diff_from_fixed5,
    }


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

    need_long = not any(s["placement"]["dist"] >= LONG_TRAVEL_MIN for s in sprites)
    current_targetable = sum(1 for s in sprites if s["placement"]["edge_cells"])
    need_targetable = max(0, e_hit - current_targetable)
    if need_targetable > unique_total:
        return None

    ok, need_long, used_targetable = _place_targetable_bases(
        rng, need_targetable, sprites, occupied, reserved_bg, need_long
    )
    if not ok:
        return None
    unique_left = unique_total - used_targetable

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    if unique_left <= 0:
        return None
    if not _place_unique_landings(rng, unique_left, sprites, occupied, reserved_bg, force_long_first=need_long):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    metrics = _variant_metrics(inp)
    if metrics["sprites"] != n_sprites:
        return None
    if metrics["long_shots"] < 1:
        return None
    if metrics["boundary_diff_from_fixed5"] < 1:
        return None

    out = _render_output(inp, sprites)
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(out[0, :] != BG) and not np.any(out[-1, :] != BG) and not np.any(out[:, 0] != BG) and not np.any(out[:, -1] != BG):
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
