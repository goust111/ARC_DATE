# Variant of 409aa875
# Dimension: D17 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码或图例；所有规则由精灵形状隐式编码... → 网格边缘图例定义方向-颜色映射

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8

# -----------------------------------------------------------------------------
# Design analysis for the D17 variant
#
# 0. Reference-task note
#    - The prompt names `e87109e9` as the symbol-instruction inspiration.
#    - In this workspace the JSON examples for `e87109e9` exist, but no readable
#      solver `.py` file was present, so its solver code could not be inspected.
#      The variant therefore borrows only the observable presentation pattern:
#      a small edge legend must be read first, and that legend changes the
#      meaning of the main objects without changing their geometry.
#
# 1. Concrete behavior and flow trace
#    - Original task: every 3-cell triangle already tells both "where it shoots"
#      and "what visible effect happens" because all non-collision results are a
#      fixed maroon 9.
#    - New task: triangle shape still determines the direction exactly as before,
#      but the color of a normal landing is no longer fixed. The top edge now
#      contains two tiny legend components:
#        * a 2-pixel diagonal symbol -> the color used by diagonal shots
#        * a 2-pixel orthogonal domino -> the color used by axis-aligned shots
#      Overlap on an empty background cell still uses the original special
#      collision color 1, so only the ordinary single-projectile effects move
#      behind the legend.
#    - Small trace:
#      original: an L-shape emits diagonally, lands on empty bg, output paints 9.
#      variant : the same L-shape emits diagonally, solver first reads the top
#      legend, sees that the diagonal symbol is color 4, and therefore paints 4
#      at that landing cell instead of 9.
#    - For sprite hits the same substitution happens at the whole hit sprite:
#      the projectile family (diagonal vs axis) chooses the recolor from legend.
#
# 2. Complexity calibration
#    - Only two legend categories are introduced: diagonal-family shots and
#      axis-family shots. This is enough to make the legend necessary, while
#      avoiding 8 separate direction->color entries that would be too sparse for
#      ARC's 2-4 examples.
#    - The generator always includes at least one visible diagonal-family result
#      and one visible axis-family result, so both legend rows are supported by
#      repeated evidence in each sample.
#
# 3. Review of original `main()` assumptions
#    - Original assumption: every non-bg component is a sprite candidate.
#      Not valid anymore because the new input has two non-bg legend components.
#      Fix: parse the legend from the top edge first and exclude those cells from
#      component discovery.
#    - Original assumption: every non-collision effect uses fixed color 9.
#      Not valid anymore. Fix: derive the recolor from the projectile family and
#      the legend color.
#    - Original assumption: all in-bounds landing cells are part of the play
#      area. Not valid anymore because the top edge is reserved for the legend.
#      Fix: the solver ignores landings above `PLAY_TOP`, and the generator never
#      creates them.
#
# 4. Placement legality
#    - Rows 0-1 are the legend strip; row 2 is kept blank as a visual spacer.
#      Sprites and landing cells are restricted to rows 3-15.
#    - This keeps legend cells disjoint from sprites, prevents accidental 8-
#      connected merges, and guarantees that output painting never overwrites the
#      legend.
#
# 5. Visual observability
#    - The input/output difference is visible in two ways:
#        * the same geometry still produces the original landing pattern
#        * but the two families of shots recolor to two legend-defined colors
#    - The legend colors vary across generated samples, so a solver cannot rely
#      on fixed ids like 9 for all ordinary hits.
#
# 6. Visual cleanliness
#    - The blank spacer row keeps the legend from touching the playfield.
#    - Legend symbols are tiny 2-cell components and are fixed far apart on the
#      top edge, so they read as instructions instead of as extra sprites.
#
# 7. Color inferability
#    - Sprite color is chosen from a pool that excludes the fixed collision color
#      and the two legend colors. This avoids ambiguous "recoloring" that would
#      visually look unchanged.
#    - Only two legend-driven colors appear, which is small enough for a human to
#      infer from few examples.
#
# 8. New solver assumptions and generator guarantees
#    - Assumption: exactly one diagonal legend component and one axis legend
#      component exist in the top legend strip, each monochrome.
#      Guarantee: generator paints those fixed symbols and leaves row 2 blank.
#    - Assumption: no projectile landing should target the legend strip.
#      Guarantee: precomputed placements require destination row >= PLAY_TOP.
#    - Assumption: if multiple projectiles hit the same sprite, color precedence
#      would be ambiguous. Guarantee: generator allows at most one hit per target
#      sprite, matching the original output-first discipline.
# -----------------------------------------------------------------------------

H = W = 16
BG = 7
COLLISION_COLOR = 1
PROJECTILE_DIST = 5

LEGEND_ROWS = 2
PLAY_TOP = 3  # rows 0-1 legend, row 2 blank spacer

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

LEGEND_DIAG_CELLS = ((0, 1), (1, 2))
LEGEND_AXIS_CELLS = ((0, 13), (1, 13))
LEGEND_KINDS = ("diag", "axis")

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 600

SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]
LEGEND_COLOR_POOL = [0, 2, 3, 4, 5, 6, 8, 9]


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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


def _legend_kind(comp: list[tuple[int, int]]) -> str | None:
    if len(comp) != 2:
        return None
    (r1, c1), (r2, c2) = sorted(comp)
    if max(r1, r2) >= LEGEND_ROWS:
        return None
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    if dr + dc == 1:
        return "axis"
    if dr == 1 and dc == 1:
        return "diag"
    return None


def _read_legend(grid: np.ndarray, bg: int) -> tuple[dict[str, int], set[tuple[int, int]]]:
    top_mask = np.zeros_like(grid, dtype=bool)
    top_mask[:LEGEND_ROWS, :] = grid[:LEGEND_ROWS, :] != bg
    comps = _cc8(top_mask)
    legend_colors: dict[str, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    for comp in comps:
        kind = _legend_kind(comp)
        colors = {int(grid[r, c]) for r, c in comp}
        if kind is None or len(colors) != 1:
            raise ValueError("Malformed legend")
        legend_colors[kind] = colors.pop()
        legend_cells.update(comp)
    if set(legend_colors) != set(LEGEND_KINDS):
        raise ValueError("Legend must define both diagonal and axis colors")
    return legend_colors, legend_cells


def _is_play_dest(rc: tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < H and 0 <= c < W and r >= PLAY_TOP


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; use an edge legend to choose ordinary landing colors.

    Input structure:
    - bg = most common color. Rows 0-1 contain a small legend: one 2-pixel diagonal symbol and one 2-pixel orthogonal domino, whose colors define the diagonal-shot and axis-shot paint colors. Row 2 is blank. The playfield below contains 3-pixel triangle sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each triangle still determines its apex and emission direction from shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
    - A projectile lands exactly 5 cells from the apex in that direction, but only inside the playfield below the legend.
    - If the landing cell hits an existing sprite, the whole hit sprite is recolored with the legend color for that projectile family (diagonal family or axis family).
    - If two or more projectiles land on the same empty playfield cell, that overlap cell becomes blue (1); otherwise a lone landing on empty background uses the legend color for that projectile family.

    Key insight: Triangle geometry still tells you where each projectile goes, but the top-edge symbols now tell you what color a normal diagonal shot or axis shot should paint.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend_colors, legend_cells = _read_legend(g, bg)

    play = g.copy()
    for r, c in legend_cells:
        play[r, c] = bg

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(play)) - {bg}):
        for comp in _cc8(play == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
        if not _is_play_dest(dest):
            continue
        family = "diag" if abs(dr) + abs(dc) == 2 else "axis"
        dest_to_shooters.setdefault(dest, []).append((cid, family))

    out = g.copy()

    hit_comp_to_color: dict[int, int] = {}
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            family = shooters[0][1]
            hit_comp_to_color.setdefault(cell_to_comp[dest], legend_colors[family])

    for cid, color in hit_comp_to_color.items():
        for r, c in comps[cid]:
            out[r, c] = color

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        if len(shooters) >= 2:
            out[r, c] = COLLISION_COLOR
        else:
            out[r, c] = legend_colors[shooters[0][1]]

    return out


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W and r >= PLAY_TOP for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[str, list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not (_is_play_dest(dest) and _in_bounds_cells(cells)):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "family": "diag" if abs(dr) + abs(dc) == 2 else "axis",
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                by_family[placement["family"]].append(placement)
    return placements, by_dest, by_family


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_FAMILY = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    _paint_cells(grid, placement["cells"], color)


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
    family: str | None = None,
) -> dict | None:
    candidates = (PLACEMENTS_BY_FAMILY[family] if family is not None else PLACEMENTS)[:]
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
    if np.any(grid[2, :] != BG):
        return False
    try:
        _, legend_cells = _read_legend(grid, BG)
    except Exception:
        return False
    play_mask = grid != BG
    for r, c in legend_cells:
        play_mask[r, c] = False
    comps = components_mask(play_mask, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, (n_sprites - 2) // 2) + 1):
            unique_extra = n_sprites - 2 - e_hit - 2 * e_coll
            if unique_extra < 0:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique_extra))
    return rng.choice(options) if options else None


def _place_collision_events(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
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


def _place_unique_shots(rng, n, sprites, occupied, reserved_bg) -> bool:
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


def _render_output(inp: np.ndarray, sprites: list[dict], legend_colors: dict[str, int]) -> np.ndarray:
    out = inp.copy()

    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    bg_single_family = {
        s["target"]: s["placement"]["family"]
        for s in sprites
        if s["target_kind"] == "bg" and bg_targets[s["target"]] == 1
    }
    hit_sprite_colors = {
        s["target_sprite"]: legend_colors[s["placement"]["family"]]
        for s in sprites
        if s["target_kind"] == "hit"
    }

    for idx, color in hit_sprite_colors.items():
        _paint_sprite(out, sprites[idx]["placement"], color)

    for dest, count in bg_targets.items():
        if count >= 2:
            out[dest] = COLLISION_COLOR
        else:
            out[dest] = legend_colors[bg_single_family[dest]]

    return out


def _visible_family_colors(out: np.ndarray, legend_colors: dict[str, int]) -> set[str]:
    visible = set()
    for family, color in legend_colors.items():
        for r in range(PLAY_TOP, H):
            for c in range(W):
                if out[r, c] == color:
                    visible.add(family)
                    break
            if family in visible:
                break
    return visible


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    legend_choices = [c for c in LEGEND_COLOR_POOL if c != sprite_color and c != COLLISION_COLOR]
    diag_color, axis_color = rng.sample(legend_choices, 2)
    legend_colors = {"diag": diag_color, "axis": axis_color}

    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_extra = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    for family in LEGEND_KINDS:
        placement = _choose_unique(rng, occupied, reserved_bg, family=family)
        if placement is None:
            return None
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])

    if not _place_collision_events(rng, e_coll, sprites, occupied, reserved_bg):
        return None
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None
    if not _place_unique_shots(rng, unique_extra, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    _paint_cells(inp, LEGEND_DIAG_CELLS, diag_color)
    _paint_cells(inp, LEGEND_AXIS_CELLS, axis_color)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites, legend_colors)
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if _visible_family_colors(out, legend_colors) != set(LEGEND_KINDS):
        return None
    if not np.any(out == COLLISION_COLOR) and e_coll == 0 and e_hit == 0:
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
