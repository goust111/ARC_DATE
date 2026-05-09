# Variant of 409aa875
# Dimension: D17 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码或图例；所有规则由精灵形状隐式编码... → 网格角落的2x2色块作为碰撞规则图例

# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete realization and behavior trace
#    - The original 16x16 board keeps the same 3-pixel triangle sprites and the same
#      projectile geometry: apex + 5 * direction.
#    - The new visible object is a 2x2 legend block placed in one grid corner. Its
#      four local positions have fixed semantics:
#        top-left     -> lone projectile landing on empty background
#        top-right    -> multiple projectiles landing on the same empty background cell
#        bottom-left  -> projectile hitting the target sprite's apex ("front")
#        bottom-right -> projectile hitting one of the target sprite's two non-apex cells ("side")
#    - Small trace: suppose a target V-sprite has apex at (6, 8). In the original
#      task, any projectile landing on any of that sprite's three cells recolors the
#      whole sprite to 9. In the variant, a landing on (6, 8) recolors the sprite to
#      the legend's bottom-left color, while a landing on either diagonal leg recolors
#      the sprite to the legend's bottom-right color. Background landings likewise use
#      the legend's top row instead of fixed 9/1.
# b) Complexity calibration
#    - The legend introduces exactly four categories, no more. The palette is random
#      per sample, so the legend is required, but the category count stays small enough
#      to be inferable from 2-4 examples.
#    - Every generated sample is forced to exhibit all four categories at least once,
#      so the legend is not decorative.
# c) Old assumptions that break
#    - Original `main()` treated every non-background monochrome component as a sprite.
#      That fails once a multicolor legend is present, so the solver must detect and
#      mask out the corner legend before collecting components.
#    - Original `main()` used unconditional constants 9 and 1 for all outcomes. That
#      is no longer valid because each semantic now maps to an instance-specific legend
#      color.
#    - Original sprite-hit logic did not care which cell of the target was struck.
#      The variant requires distinguishing apex hits from non-apex hits.
# d) Placement legality
#    - The generator reserves a 4x4 zone around the legend corner, keeping sprites and
#      projectile destinations away from the legend for visual clarity.
#    - Sprite-to-sprite Chebyshev separation is kept via the same halo test as the
#      original generator. Projectile destinations reserved as background markers are
#      also blocked from later sprite placement.
# e) Visual observability
#    - The top legend row governs empty-cell outcomes; the bottom row governs sprite-hit
#      outcomes. This positional grouping makes the mapping easier to read visually.
#    - Front vs side is visible because the target cell is a different pixel within an
#      asymmetric 3-pixel triangle; the apex is the pointed pixel already used by the
#      firing rule.
# f) Visual cleanliness
#    - The legend is isolated in a corner with padding, so it does not fuse with the
#      triangle field or with new output markers.
#    - The output only adds isolated marker cells and full-sprite recolors, matching the
#      original task's clean look.
# g) Color inference
#    - Legend colors are distinct from each other, from the background, and from the
#      common sprite color. This prevents invisible events and avoids accidental support
#      for a wrong constant-color hypothesis.
# h) Non-degeneracy target
#    - Each sample contains at least one lone background hit, one multi-hit background
#      collision, one front hit, and one side hit. Because front and side colors are
#      distinct, the new behavior cannot collapse back to the original fixed-color rule.
# i) New solver assumptions checked against the generator
#    - Solver assumes exactly one corner contains a full 2x2 all-non-bg legend with
#      four distinct colors. Generator guarantees that.
#    - Solver assumes hit classification only needs apex vs non-apex. Generator places
#      hit events on valid sprite cells and keeps target sprites distinct, while the
#      solver still handles repeated hits conservatively by prioritizing front over side.

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


CORNERS = ("tl", "tr", "bl", "br")
LEGEND_KEYS = ("single_bg", "multi_bg", "front_hit", "side_hit")


def _corner_cells(h: int, w: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if corner == "tr":
        return [(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)]
    if corner == "bl":
        return [(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)]
    return [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)]


def _legend_zone(h: int, w: int, corner: str, pad: int = 2) -> set[tuple[int, int]]:
    side = 2 + pad
    if corner == "tl":
        rows = range(0, min(h, side))
        cols = range(0, min(w, side))
    elif corner == "tr":
        rows = range(0, min(h, side))
        cols = range(max(0, w - side), w)
    elif corner == "bl":
        rows = range(max(0, h - side), h)
        cols = range(0, min(w, side))
    else:
        rows = range(max(0, h - side), h)
        cols = range(max(0, w - side), w)
    return {(r, c) for r in rows for c in cols}


def _find_legend(grid: np.ndarray, bg: int) -> tuple[str, dict[str, int], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for corner in CORNERS:
        cells = _corner_cells(h, w, corner)
        colors = [int(grid[r, c]) for r, c in cells]
        if all(col != bg for col in colors) and len(set(colors)) == 4:
            mapping = dict(zip(LEGEND_KEYS, colors))
            candidates.append((corner, mapping, set(cells), _legend_zone(h, w, corner)))
    if len(candidates) != 1:
        raise ValueError("expected exactly one 2x2 corner legend")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; use the corner legend to color each collision type.

    Input structure:
    - bg = most common color. A 2x2 block in one grid corner is a legend with four distinct non-bg colors; all other non-bg cells form monochrome 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
    - A projectile lands exactly 5 cells from the apex in the emission direction.
    - The legend's top-left / top-right colors mean lone-hit background / multi-hit background. The bottom-left / bottom-right colors mean hitting a sprite on its apex ("front") / on one of its other two cells ("side").
    - If multiple hits affect one target sprite, an apex hit takes precedence over side-only hits.

    Key insight: The firing geometry is still encoded by the triangle shapes, but the output colors are no longer fixed constants; the 2x2 corner legend is the symbol key that decodes all four collision outcomes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    _, legend, legend_cells, _ = _find_legend(g, bg)

    sprite_mask = g != bg
    for r, c in legend_cells:
        sprite_mask[r, c] = False

    comps: list[list[tuple[int, int]]] = []
    comp_apex: list[tuple[int, int]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for comp in _cc8(sprite_mask):
        if len({int(g[r, c]) for r, c in comp}) != 1:
            continue
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        cid = len(comps)
        comps.append(comp)
        comp_apex.append(ad[0])
        for rc in comp:
            cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        (ar, ac), (dr, dc) = _triangle_apex_and_dir(comp)
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    out = g.copy()
    hit_kind_by_comp: dict[int, str] = {}
    for dest in dest_to_shooters:
        if dest in legend_cells:
            continue
        if dest not in cell_to_comp:
            continue
        cid = cell_to_comp[dest]
        kind = "front" if dest == comp_apex[cid] else "side"
        prev = hit_kind_by_comp.get(cid)
        if prev != "front":
            hit_kind_by_comp[cid] = kind

    for cid, kind in hit_kind_by_comp.items():
        color = legend["front_hit"] if kind == "front" else legend["side_hit"]
        for r, c in comps[cid]:
            out[r, c] = color

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in legend_cells or (r, c) in cell_to_comp:
            continue
        out[r, c] = legend["multi_bg"] if len(shooters) >= 2 else legend["single_bg"]

    return out


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
PROJECTILE_DIST = 5

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (5, 9)
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 400


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


def _fits(placement: dict, occupied: set[tuple[int, int]], blocked_cells: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & blocked_cells)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _paint_legend(grid: np.ndarray, corner: str, legend: dict[str, int]) -> None:
    cells = _corner_cells(*grid.shape, corner)
    for (r, c), key in zip(cells, LEGEND_KEYS):
        grid[r, c] = legend[key]


def _choose_legend(rng: random.Random, sprite_color: int) -> dict[str, int]:
    legend_palette = [c for c in SPRITE_COLORS if c != sprite_color]
    chosen = rng.sample(legend_palette, 4)
    return dict(zip(LEGEND_KEYS, chosen))


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int]:
    options = []
    for e_coll in range(1, MAX_COLL_EVENTS + 1):
        unique_total = n_sprites - 2 * e_coll - 2  # +1 front-hit shooter +1 side-hit shooter
        if unique_total >= 1:
            options.append((e_coll, unique_total))
    if not options:
        raise ValueError("no valid event counts")
    return rng.choice(options)


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    blocked_cells = forbidden_cells | reserved_bg
    for dest in dests:
        if dest in reserved_bg or dest in occupied or dest in forbidden_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, blocked_cells)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            blocked2 = blocked_cells | {dest}
            for second in candidates[i + 1:]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                if second["cell_set"] & blocked2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    blocked_cells = forbidden_cells | reserved_bg
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied or placement["dest"] in forbidden_cells:
            continue
        if _fits(placement, occupied, blocked_cells):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    base_sprites: list[dict],
    target_indices: list[int],
    desired_kind: str,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> tuple[int, dict, tuple[int, int]] | None:
    blocked_cells = forbidden_cells | reserved_bg
    rng.shuffle(target_indices)
    for idx in target_indices:
        target = base_sprites[idx]["placement"]
        apex = target["apex"]
        cells = [apex] if desired_kind == "front" else [cell for cell in target["cells"] if cell != apex]
        rng.shuffle(cells)
        for cell in cells:
            candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, blocked_cells)]
            rng.shuffle(candidates)
            for placement in candidates:
                return idx, placement, cell
    return None


def _validate_input(
    grid: np.ndarray,
    n_sprites: int,
    legend_corner: str,
    legend: dict[str, int],
) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    try:
        found_corner, found_legend, legend_cells, _ = _find_legend(grid, BG)
    except ValueError:
        return False
    if found_corner != legend_corner or found_legend != legend:
        return False
    mask = grid != BG
    for r, c in legend_cells:
        mask[r, c] = False
    comps = _cc8(mask)
    if len(comps) != n_sprites:
        return False
    return all(len(comp) == 3 and len({int(grid[r, c]) for r, c in comp}) == 1 and _triangle_apex_and_dir(comp) is not None for comp in comps)


def _render_output(inp: np.ndarray, sprites: list[dict], legend: dict[str, int]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_kind_by_target: dict[int, str] = {}
    for s in sprites:
        if s["target_kind"] != "hit":
            continue
        idx = s["target_sprite"]
        prev = hit_kind_by_target.get(idx)
        if prev != "front":
            hit_kind_by_target[idx] = s["hit_kind"]
    for idx, kind in hit_kind_by_target.items():
        color = legend["front_hit"] if kind == "front" else legend["side_hit"]
        _paint_sprite(out, sprites[idx]["placement"], color)
    for dest, count in bg_targets.items():
        out[dest] = legend["multi_bg"] if count >= 2 else legend["single_bg"]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    legend_corner = rng.choice(CORNERS)
    legend = _choose_legend(rng, sprite_color)
    legend_cells = set(_corner_cells(H, W, legend_corner))
    forbidden_cells = _legend_zone(H, W, legend_corner)

    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    try:
        e_coll, unique_total = _counts_for_n(rng, n_sprites)
    except ValueError:
        return None

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set(forbidden_cells)

    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg, forbidden_cells)
        if choice is None:
            return None
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])

    for _ in range(unique_total):
        placement = _choose_unique(rng, occupied, reserved_bg, forbidden_cells)
        if placement is None:
            return None
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])

    base_count = len(sprites)
    if base_count < 2:
        return None

    front_choice = _choose_hit(
        rng,
        sprites,
        list(range(base_count)),
        "front",
        occupied,
        reserved_bg,
        forbidden_cells,
    )
    if front_choice is None:
        return None
    front_idx, front_placement, front_cell = front_choice
    sprites.append(
        {
            "placement": front_placement,
            "target": front_cell,
            "target_kind": "hit",
            "target_sprite": front_idx,
            "hit_kind": "front",
        }
    )
    occupied.update(front_placement["cells"])

    remaining_targets = [idx for idx in range(base_count) if idx != front_idx]
    side_choice = _choose_hit(
        rng,
        sprites[:base_count],
        remaining_targets,
        "side",
        occupied,
        reserved_bg,
        forbidden_cells,
    )
    if side_choice is None:
        return None
    side_idx, side_placement, side_cell = side_choice
    sprites.append(
        {
            "placement": side_placement,
            "target": side_cell,
            "target_kind": "hit",
            "target_sprite": side_idx,
            "hit_kind": "side",
        }
    )
    occupied.update(side_placement["cells"])

    inp = np.full((H, W), BG, dtype=int)
    _paint_legend(inp, legend_corner, legend)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if legend_cells & occupied:
        return None
    if not _validate_input(inp, n_sprites, legend_corner, legend):
        return None

    out = _render_output(inp, sprites, legend)

    if np.array_equal(inp, out):
        return None
    if not all(np.any(out == legend[key]) for key in LEGEND_KEYS):
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
