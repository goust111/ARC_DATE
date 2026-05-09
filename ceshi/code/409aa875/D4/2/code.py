# Variant of 409aa875
# Dimension: D4 — 碰撞规则（投射物-精灵） (transform)
# Change: 投射物命中精灵时，整个被命中精灵重着色为栗色(9)... → 命中精灵被投射物颜色覆盖并向发射方向平移1格

# Design analysis
# - Source task: none, so the design is derived from the original 409aa875 solver/spec only.
# - Concrete behavior:
#   Every 3-cell triangle still emits exactly one projectile to apex + 5*dir. Empty-bg landings are unchanged:
#   one projectile paints 9, two or more on the same empty cell paint 1. The D4 change only affects sprite hits:
#   instead of repainting the struck sprite in place, the whole 3-cell struck sprite is removed from its old
#   location, recolored with the shooter's own color, and translated by 1 cell in the shooter's firing direction.
# - Small trace:
#   Original: shooter color 2 lands on a color-6 target cell -> all 3 target cells become 9 at the same place.
#   Variant: the same hit makes the 3 target cells become color 2 and reappear one step farther along the shot.
#   This changes both color and coordinates, so the solver must infer a translation, not just a recolor mask.
# - Complexity calibration:
#   If all sprites kept one shared color, "covered by projectile color" would be invisible. To make D4 observable
#   without changing the shape grammar, the generator uses exactly two sprite colors per grid from a small palette
#   that excludes the special marker colors 1, 7, 9. Every hit pair is color-contrasted, so the moved target
#   visibly adopts the shooter's color. The number of hit events stays small (1-2) to keep the rule legible.
# - Hidden assumptions from the old main():
#   1. Old code could mark a hit by recoloring in place; new code cannot, so hit targets must be cleared first and
#      then repainted at translated coordinates.
#   2. Old code only needed the set of hit components. New code needs per-hit shooter metadata: shooter color and
#      shooter direction. Therefore dest_to_shooters stores structured records instead of bare component ids.
#   3. Old code was harmlessly insensitive to sprite colors because all sprites shared one color. The variant makes
#      shooter color semantically important, so the solver now extracts each monochrome component independently.
#   4. New motion introduces an in-bounds / non-overlap assumption for translated targets. The generator enforces
#      that every pushed sprite can move one cell without leaving the grid or colliding with other final objects.
# - Placement legality:
#   Generation still places valid 3-cell L/V sprites with in-bounds projectile landings, and still enforces input
#   Chebyshev separation. In addition, when a sprite is selected as a hit target, the generator reserves its shifted
#   output cells so later placements cannot overlap or visually merge with the moved target.
# - Visual observability:
#   Every generated grid contains at least one sprite-hit event. Because hit targets move and switch to the
#   shooter's color, the output differs from the original task even when bg markers are also present. Two colors
#   repeat across the grid, so the recolor rule is inferable from 2-4 examples rather than looking arbitrary.
# - Visual cleanliness:
#   The generator forbids moved targets from covering bg impact markers or other sprites, and keeps output sprites
#   separated so the moved component is still seen as one clean 3-cell triangle rather than as a merged cluster.
# - New solver assumptions review:
#   The hit-resolution loop assumes at most one projectile is chosen per target sprite and that a translated target
#   remains in-bounds. The generator guarantees both by sampling distinct target sprites and validating shifted
#   cells before accepting a hit shooter. Empty-cell marker painting also assumes those cells stay empty in the
#   final output; the generator reserves moved-target output cells away from reserved bg impact cells.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
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


def _shift_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _cells_in_bounds(shape: tuple[int, int], cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _extract_sprites(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    sprites: list[dict] = []
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for comp_arr in components_mask(grid != bg, connectivity=8):
        comp = tuple((int(r), int(c)) for r, c in comp_arr)
        cols = {int(grid[r, c]) for r, c in comp}
        if len(cols) != 1:
            continue
        sid = len(sprites)
        color = next(iter(cols))
        sprites.append({"cells": comp, "color": color})
        for rc in comp:
            cell_to_sprite[rc] = sid
    return sprites, cell_to_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace triangle shots; empty impacts still paint markers, but sprite impacts now shove the target.

    Input structure:
    - The majority color is background.
    - Every foreground object is a monochrome 3-cell triangle, either an axis L or a diagonal V; different sprites may use different colors.

    Rule:
    - Each triangle's shape identifies its apex and firing direction.
    - The projectile lands exactly 5 cells from that apex.
    - If the landing cell is empty background, paint that cell maroon (9), or blue (1) when two or more projectiles share that same empty cell.
    - If the landing cell belongs to another sprite, recolor the whole struck sprite with the shooter's color and translate that entire 3-cell sprite by 1 cell in the same firing direction.

    Key insight: The orientation still determines where each shot lands, but hit sprites no longer stay in place; the output must preserve the struck triangle's shape while moving it one step farther along the projectile path in the shooter's color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sprites, cell_to_sprite = _extract_sprites(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sid, sprite in enumerate(sprites):
        ad = _triangle_apex_and_dir(list(sprite["cells"]))
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(
                {"sid": sid, "dir": (dr, dc), "color": sprite["color"]}
            )

    hit_specs: dict[int, dict] = {}
    for dest, shooters in dest_to_shooters.items():
        target_sid = cell_to_sprite.get(dest)
        if target_sid is None or target_sid in hit_specs:
            continue
        shooter = sorted(shooters, key=lambda item: item["sid"])[0]
        shifted = _shift_cells(sprites[target_sid]["cells"], *shooter["dir"])
        if _cells_in_bounds((h, w), shifted):
            hit_specs[target_sid] = {
                "color": shooter["color"],
                "dir": shooter["dir"],
                "shifted": shifted,
            }

    out = g.copy()

    for target_sid in hit_specs:
        for r, c in sprites[target_sid]["cells"]:
            out[r, c] = bg

    moved_cells: set[tuple[int, int]] = set()
    for target_sid, spec in hit_specs.items():
        for r, c in spec["shifted"]:
            out[r, c] = spec["color"]
            moved_cells.add((r, c))

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite or dest in moved_cells:
            continue
        out[dest] = 1 if len(shooters) >= 2 else 9

    return out


H = W = 16
BG = 7
HIT_MARK_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 600


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return _cells_in_bounds((H, W), cells)


def _halo(cells: tuple[tuple[int, int], ...] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
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


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    input_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    shifted_halo: set[tuple[int, int]],
    shifted_cells: set[tuple[int, int]],
    *,
    allow_occupied_dest: bool = False,
) -> bool:
    if placement["dest"] in reserved_bg or placement["dest"] in shifted_cells:
        return False
    if not allow_occupied_dest and placement["dest"] in occupied:
        return False
    if placement["cell_set"] & reserved_bg:
        return False
    if placement["cell_set"] & input_halo:
        return False
    if placement["cell_set"] & shifted_halo:
        return False
    return True


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    input_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    shifted_halo: set[tuple[int, int]],
    shifted_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied or dest in shifted_cells:
            continue
        candidates = [
            p
            for p in PLACEMENTS_BY_DEST[dest]
            if _fits(p, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells)
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            halo2 = input_halo | set(first["halo"])
            for second in candidates[i + 1 :]:
                if not _fits(second, occ2, halo2, reserved_bg, shifted_halo, shifted_cells):
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    input_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    shifted_halo: set[tuple[int, int]],
    shifted_cells: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits(placement, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    input_halo: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    shifted_halo: set[tuple[int, int]],
    shifted_cells: set[tuple[int, int]],
    used_targets: set[int],
) -> tuple[int, tuple[int, int], dict, tuple[tuple[int, int], ...]] | None:
    target_ids = list(range(len(sprites)))
    rng.shuffle(target_ids)
    for idx in target_ids:
        if idx in used_targets:
            continue
        target = sprites[idx]
        target_cells = target["placement"]["cells"]
        target_cell_set = set(target_cells)
        other_occupied = occupied - target_cell_set
        cells = list(target_cells)
        rng.shuffle(cells)
        for cell in cells:
            candidates = PLACEMENTS_BY_DEST[cell][:]
            rng.shuffle(candidates)
            for placement in candidates:
                if not _fits(
                    placement,
                    occupied,
                    input_halo,
                    reserved_bg,
                    shifted_halo,
                    shifted_cells,
                    allow_occupied_dest=True,
                ):
                    continue
                shifted = _shift_cells(target_cells, *placement["dir"])
                shifted_set = set(shifted)
                if not _in_bounds_cells(shifted):
                    continue
                if shifted_set & reserved_bg:
                    continue
                if shifted_set & shifted_cells:
                    continue
                if shifted_set & set(placement["cells"]):
                    continue
                if shifted_set & other_occupied:
                    continue
                if shifted_set & shifted_halo:
                    continue
                if _halo(shifted) & other_occupied:
                    continue
                if _halo(shifted) & set(placement["cells"]):
                    continue
                return idx, cell, placement, shifted
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    for comp in comps:
        if comp.shape != (3, 2):
            return False
        coords = [(int(r), int(c)) for r, c in comp]
        if len({int(grid[r, c]) for r, c in coords}) != 1:
            return False
        if _triangle_apex_and_dir(coords) is None:
            return False
    return True


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(1, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
            input_halo.update(placement["halo"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
        input_halo.update(placement["halo"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells) -> bool:
    used_targets: set[int] = set()
    for _ in range(e_hit):
        choice = _choose_hit(
            rng,
            sprites,
            occupied,
            input_halo,
            reserved_bg,
            shifted_halo,
            shifted_cells,
            used_targets,
        )
        if choice is None:
            return False
        target_idx, hit_cell, placement, shifted = choice
        shooter_idx = len(sprites)
        sprites.append(
            {
                "placement": placement,
                "target": hit_cell,
                "target_kind": "hit",
                "target_sprite": target_idx,
            }
        )
        sprites[target_idx]["hit_by"] = shooter_idx
        sprites[target_idx]["shifted_cells"] = shifted
        occupied.update(placement["cells"])
        input_halo.update(placement["halo"])
        shifted_cells.update(shifted)
        shifted_halo.update(_halo(shifted))
        used_targets.add(target_idx)
    return True


def _assign_colors(rng: random.Random, sprites: list[dict]) -> tuple[int, int]:
    color_a, color_b = rng.sample(SPRITE_COLORS, 2)
    for idx, sprite in enumerate(sprites):
        sprite["color"] = color_a if idx % 2 == 0 else color_b
    hit_shooters = [idx for idx, sprite in enumerate(sprites) if sprite["target_kind"] == "hit"]
    for order, shooter_idx in enumerate(hit_shooters):
        target_idx = sprites[shooter_idx]["target_sprite"]
        if order % 2 == 0:
            sprites[target_idx]["color"] = color_a
            sprites[shooter_idx]["color"] = color_b
        else:
            sprites[target_idx]["color"] = color_b
            sprites[shooter_idx]["color"] = color_a
    return color_a, color_b


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    moved_cells: set[tuple[int, int]] = set()
    bg_targets = Counter(sprite["target"] for sprite in sprites if sprite["target_kind"] == "bg")

    for sprite in sprites:
        if "hit_by" not in sprite:
            continue
        for r, c in sprite["placement"]["cells"]:
            out[r, c] = BG

    for sprite in sprites:
        shooter_idx = sprite.get("hit_by")
        if shooter_idx is None:
            continue
        shooter_color = sprites[shooter_idx]["color"]
        shifted = sprite["shifted_cells"]
        for r, c in shifted:
            out[r, c] = shooter_color
            moved_cells.add((r, c))

    for dest, count in bg_targets.items():
        if dest in moved_cells:
            continue
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_MARK_COLOR

    return out


def _event_profile(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    sprites, cell_to_sprite = _extract_sprites(g, bg)
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sid, sprite in enumerate(sprites):
        ad = _triangle_apex_and_dir(list(sprite["cells"]))
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(sid)

    hit_targets = 0
    lone_bg = 0
    multi_bg = 0
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite:
            hit_targets += 1
        elif len(shooters) >= 2:
            multi_bg += 1
        else:
            lone_bg += 1
    return {"hit_targets": hit_targets, "lone_bg": lone_bg, "multi_bg": multi_bg}


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    input_halo: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    shifted_cells: set[tuple[int, int]] = set()
    shifted_halo: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(
        rng, base_needed, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells
    ):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(
        rng, remaining, sprites, occupied, input_halo, reserved_bg, shifted_halo, shifted_cells
    ):
        return None

    color_a, color_b = _assign_colors(rng, sprites)
    if color_a == color_b:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_cells(inp, sprite["placement"]["cells"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None
    if len({int(v) for v in inp[inp != BG]}) < 2:
        return None

    out = _render_output(inp, sprites)
    if np.array_equal(inp, out):
        return None
    if not np.any(out == HIT_MARK_COLOR):
        return None

    hit_events = [sprite for sprite in sprites if "hit_by" in sprite]
    if len(hit_events) != e_hit:
        return None
    if not all(inp[sprite["placement"]["cells"][0]] != out[sprite["shifted_cells"][0]] for sprite in hit_events):
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
