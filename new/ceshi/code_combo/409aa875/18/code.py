# Combo variant of 409aa875 (index 18)
# combo_spec: D2/0+D17/1
# Name: 颜色编码距离+控制像素图例

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# combo_variant:
#   concreteization:
#     legend:
#       - The grid adds exactly three singleton control pixels on fixed top-edge slots:
#         (0,2), (0,7), (0,12).
#       - Those three slots are an explicit short/medium/long legend, read left-to-right as
#         distances 3, 5, and 7.
#       - The control pixel's color is not the projectile marker color; it is the sprite-color key.
#         A sprite fires the distance of the legend slot whose pixel has the same color.
#     sprites:
#       - Real emitters remain interior monochrome 3-cell triangles, but the grid is no longer
#         single-colored: multiple sprite colors coexist because color now carries the legend lookup key.
#       - Geometry is unchanged from the base task: L-shapes fire diagonally, V-shapes fire axially.
#     behavior_trace:
#       - Example legend: top slots contain colors [5, 2, 8], so 5->3, 2->5, 8->7.
#       - A color-5 L with apex (8,8) and direction (-1,-1) lands at (5,5); the original task
#         would have used fixed distance 5 and landed at (3,3).
#       - A color-8 V with apex (10,6) and direction (0,1) lands at (10,13); the same geometry in
#         the original task would have stopped at (10,11).
#       - The fusion is therefore not "nearest control decides distance"; border pixels publish the
#         color->distance table and each sprite consults the entry matching its own color.
#   complexity_calibration:
#     - Distances are restricted to exactly three reusable values: 3 / 5 / 7.
#     - Every sample uses all three legend entries, and at least two sprite colors repeat.
#       This gives enough support for a human to notice that same-color sprites share a range.
#     - Legend colors are re-permuted across samples, so the task cannot collapse to a hardcoded
#       global palette mapping; the border legend must be read.
#   implicit_assumption_review:
#     - Original assumption: all non-bg components are candidate sprites.
#       Broken by the new border singletons, so the solver must filter to interior 3-cell triangles.
#     - Original assumption: all sprites share one color.
#       Broken because color now selects the legend entry; the solver must preserve component color.
#     - Original assumption: distance is always 5.
#       Replaced by a color lookup through the legend slots.
#     - D17-only assumption: proximity to a control pixel chooses distance.
#       Rejected here; control pixels are a visible legend, not local attractors.
#   placement_legality:
#     - Legend pixels are isolated on the top border and are the only border non-bg cells.
#     - Sprites stay fully inside the field with Chebyshev separation >= 2, so components never merge.
#     - Projectile destinations stay off the border, so output markers never overwrite the legend.
#   visual_observability:
#     - Humans can compare two same-shape sprites of different colors and see that the endpoint changes.
#     - They can also compare two same-color sprites at different positions and see that the distance
#       stays tied to color rather than to the nearest legend pixel.
#     - The generator rejects cases where the fused rule accidentally matches either the fixed-5 base
#       task or the D17 nearest-control variant.
#   visual_cleanliness:
#     - The instruction layer is only three border pixels in a straight line.
#     - Output markers remain in the interior, so the border legend stays readable in both input/output.
#   color_inference:
#     - Legend/sprite colors are chosen from a small pool excluding bg=7 and output colors {1,9}.
#     - Each active color appears in both the legend and multiple interior sprites across samples,
#       making the color->distance association recoverable from few demonstrations.
# main_assumption_audit:
#   - `_legend_controls` assumes the only meaningful border instruction pixels are at the three fixed
#     top slots. The generator guarantees this.
#   - `_triangle_components` assumes every real emitter is an interior 3-cell monochrome triangle.
#     The generator guarantees that and keeps legend pixels as isolated singletons.
#   - `_color_lookup_destinations` assumes each sprite color appears at most once in the legend.
#     The generator uses three distinct legend colors.
#   - `_apply_projectiles` assumes only background cells should receive new lone-hit/collision markers.
#     The generator keeps all projectile destinations either on sprite cells or on interior bg cells.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1

DISTANCES = (3, 5, 7)
LEGEND_SLOTS = ((0, 2), (0, 7), (0, 12))  # left->right = short/medium/long
LEGEND_SLOT_TO_DIST = dict(zip(LEGEND_SLOTS, DISTANCES))

SPRITE_COLOR_POOL = [0, 2, 3, 4, 5, 6, 8]

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (6, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 500


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


def _legend_controls(g: np.ndarray, bg: int) -> list[dict]:
    controls = []
    for pos in LEGEND_SLOTS:
        color = int(g[pos])
        if color == bg:
            continue
        controls.append({"pos": pos, "color": color, "dist": LEGEND_SLOT_TO_DIST[pos]})
    return controls


def _legend_color_to_dist(g: np.ndarray, bg: int) -> dict[int, int]:
    return {ctl["color"]: ctl["dist"] for ctl in _legend_controls(g, bg)}


def _nearest_control(
    apex: tuple[int, int],
    controls: list[dict],
    min_margin: int = 0,
) -> dict | None:
    if not controls:
        return None
    ar, ac = apex
    ordered = sorted(
        (abs(ar - ctl["pos"][0]) + abs(ac - ctl["pos"][1]), idx)
        for idx, ctl in enumerate(controls)
    )
    best_dist, best_idx = ordered[0]
    if len(ordered) >= 2 and ordered[1][0] - best_dist < min_margin:
        return None
    return controls[best_idx]


def _triangle_components(g: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = g.shape
    triangles: list[dict] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            if len(comp) != 3:
                continue
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
                continue
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            apex, direction = ad
            cid = len(triangles)
            triangles.append({
                "cells": tuple(comp),
                "apex": apex,
                "dir": direction,
                "color": col,
            })
            for rc in comp:
                cell_to_comp[rc] = cid
    return triangles, cell_to_comp


def _projectile_dest(
    apex: tuple[int, int],
    direction: tuple[int, int],
    distance: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    rr = apex[0] + distance * direction[0]
    cc = apex[1] + distance * direction[1]
    if 0 <= rr < h and 0 <= cc < w:
        return rr, cc
    return None


def _color_lookup_destinations(
    triangles: list[dict],
    color_to_dist: dict[int, int],
    h: int,
    w: int,
) -> dict[int, tuple[int, int] | None]:
    dests: dict[int, tuple[int, int] | None] = {}
    for cid, tri in enumerate(triangles):
        dist = color_to_dist.get(tri["color"])
        if dist is None:
            dests[cid] = None
            continue
        dests[cid] = _projectile_dest(tri["apex"], tri["dir"], dist, h, w)
    return dests


def _nearest_control_destinations(
    triangles: list[dict],
    controls: list[dict],
    h: int,
    w: int,
) -> dict[int, tuple[int, int] | None]:
    dests: dict[int, tuple[int, int] | None] = {}
    for cid, tri in enumerate(triangles):
        ctl = _nearest_control(tri["apex"], controls, min_margin=0)
        if ctl is None:
            dests[cid] = None
            continue
        dests[cid] = _projectile_dest(tri["apex"], tri["dir"], ctl["dist"], h, w)
    return dests


def _fixed_distance_destinations(
    triangles: list[dict],
    distance: int,
    h: int,
    w: int,
) -> dict[int, tuple[int, int] | None]:
    return {
        cid: _projectile_dest(tri["apex"], tri["dir"], distance, h, w)
        for cid, tri in enumerate(triangles)
    }


def _dest_to_shooters(destinations: dict[int, tuple[int, int] | None]) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, dest in destinations.items():
        if dest is not None:
            out[dest].append(cid)
    return dict(out)


def _apply_projectiles(
    g: np.ndarray,
    bg: int,
    triangles: list[dict],
    cell_to_comp: dict[tuple[int, int], int],
    dest_to_shooters: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    out = g.copy()
    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}

    for cid in hit_components:
        for r, c in triangles[cid]["cells"]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        if int(g[r, c]) != bg:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR
    return out


def _solve_fixed5_blind(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, cell_to_comp = _triangle_components(g, bg)
    dests = _fixed_distance_destinations(triangles, 5, *g.shape)
    return _apply_projectiles(g, bg, triangles, cell_to_comp, _dest_to_shooters(dests))


def _solve_nearest_control_blind(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    controls = _legend_controls(g, bg)
    triangles, cell_to_comp = _triangle_components(g, bg)
    dests = _nearest_control_destinations(triangles, controls, *g.shape)
    return _apply_projectiles(g, bg, triangles, cell_to_comp, _dest_to_shooters(dests))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from multicolored 3-pixel triangles; a border legend maps sprite colors to shot length.

    Input structure:
    - bg = most common color. Interior non-bg 3-pixel components are monochrome triangle sprites shaped as axis-aligned L's or diagonal V's.
    - Three singleton control pixels sit in fixed top-edge legend slots. Reading those slots left-to-right gives the three distances 3, 5, and 7; the pixel color in each slot says which sprite color uses that distance.

    Rule:
    - Each triangle still derives its apex and firing direction from geometry: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite the opening.
    - A sprite looks up its own color in the top legend, takes that slot's distance, and lands at apex + distance * direction.
    - If the landing cell lies on another sprite, recolor that entire hit sprite to maroon (9).
    - If two projectiles land on the same empty bg cell, paint it blue (1); a lone projectile on empty bg paints maroon (9). The border legend itself is only an instruction layer and does not change.

    Key insight: Shape still determines direction, but distance is now an explicit color-keyed legend lookup: same-color sprites share a range even when they are far from the same control pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, cell_to_comp = _triangle_components(g, bg)
    color_to_dist = _legend_color_to_dist(g, bg)
    dests = _color_lookup_destinations(triangles, color_to_dist, *g.shape)
    return _apply_projectiles(g, bg, triangles, cell_to_comp, _dest_to_shooters(dests))


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


def _choose_controls(rng: random.Random) -> list[dict]:
    colors = rng.sample(SPRITE_COLOR_POOL, len(DISTANCES))
    return [
        {"pos": pos, "color": color, "dist": dist}
        for pos, dist, color in zip(LEGEND_SLOTS, DISTANCES, colors)
    ]


def _build_placements(color_to_dist: dict[int, int]) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for color, dist in color_to_dist.items():
        for ar in range(1, H - 1):
            for ac in range(1, W - 1):
                for dr, dc in ALL_DIRS:
                    cells = _sprite_cells((ar, ac), (dr, dc))
                    if not _in_bounds_cells(cells):
                        continue
                    if any(r <= 1 or c == 0 or r == H - 1 or c == W - 1 for r, c in cells):
                        continue
                    dest = _projectile_dest((ar, ac), (dr, dc), dist, H, W)
                    if dest is None:
                        continue
                    if not (0 < dest[0] < H - 1 and 0 < dest[1] < W - 1):
                        continue
                    placement = {
                        "apex": (ar, ac),
                        "dir": (dr, dc),
                        "distance": dist,
                        "color": color,
                        "dest": dest,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                    }
                    placements.append(placement)
                    by_dest[dest].append(placement)
    return placements, by_dest


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = placement["color"]


def _choose_collision_pair(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(placements_by_dest)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in placements_by_dest[dest] if _fits(p, occupied, reserved_bg)]
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
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    allowed_colors: set[int] | None = None,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if allowed_colors is not None and placement["color"] not in allowed_colors:
            continue
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in placements_by_dest[cell] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
    return None


def _validate_input(grid: np.ndarray, controls: list[dict], n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False

    seen = {(ctl["pos"], ctl["color"]) for ctl in controls}
    for ctl in controls:
        if int(grid[ctl["pos"]]) != ctl["color"]:
            return False

    for r in range(H):
        for c in range(W):
            if r not in (0, H - 1) and c not in (0, W - 1):
                continue
            v = int(grid[r, c])
            if v == BG:
                continue
            if ((r, c), v) not in seen:
                return False

    triangles, _ = _triangle_components(grid, BG)
    if len(triangles) != n_sprites:
        return False
    color_to_dist = _legend_color_to_dist(grid, BG)
    if len(color_to_dist) != len(DISTANCES):
        return False
    return all(tri["color"] in color_to_dist for tri in triangles)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(1, min(MAX_HIT_EVENTS, n_sprites - 3) + 1):
        for e_coll in range(1, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < len(DISTANCES):
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(
    rng: random.Random,
    e_coll: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    placements_by_dest: dict[tuple[int, int], list[dict]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, placements_by_dest, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(
    rng: random.Random,
    count: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    placements: list[dict],
    allowed_colors: set[int] | None = None,
) -> bool:
    for _ in range(count):
        placement = _choose_unique(rng, placements, occupied, reserved_bg, allowed_colors)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    placements_by_dest: dict[tuple[int, int], list[dict]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], placements_by_dest, occupied, reserved_bg, used_target_cells)
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
        for r, c in sprites[idx]["placement"]["cells"]:
            out[r, c] = HIT_COLOR
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _destination_change_count(
    inp: np.ndarray,
    color_to_dist: dict[int, int],
) -> tuple[int, int]:
    triangles, _ = _triangle_components(inp, BG)
    controls = _legend_controls(inp, BG)
    actual = _color_lookup_destinations(triangles, color_to_dist, H, W)
    fixed = _fixed_distance_destinations(triangles, 5, H, W)
    nearest = _nearest_control_destinations(triangles, controls, H, W)
    changed_vs_fixed = sum(actual[cid] != fixed[cid] for cid in actual)
    changed_vs_nearest = sum(actual[cid] != nearest[cid] for cid in actual)
    return changed_vs_fixed, changed_vs_nearest


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    controls = _choose_controls(rng)
    color_to_dist = {ctl["color"]: ctl["dist"] for ctl in controls}
    placements, placements_by_dest = _build_placements(color_to_dist)
    if len(placements) < 150:
        return None

    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg, placements_by_dest):
        return None

    legend_colors = [ctl["color"] for ctl in controls]
    if not _place_unique_landings(
        rng,
        len(legend_colors),
        sprites,
        occupied,
        reserved_bg,
        placements,
        set(legend_colors),
    ):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg, placements_by_dest):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg, placements):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for ctl in controls:
        inp[ctl["pos"]] = ctl["color"]
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"])

    if not _validate_input(inp, controls, n_sprites):
        return None

    color_counts = Counter(sprite["placement"]["color"] for sprite in sprites)
    if set(color_counts) != set(legend_colors):
        return None
    if not all(count >= 2 for count in color_counts.values()):
        return None

    out = _render_output(inp, sprites)
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    changed_vs_fixed, changed_vs_nearest = _destination_change_count(inp, color_to_dist)
    if changed_vs_fixed < 2 or changed_vs_nearest < 2:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(_solve_fixed5_blind(inp), out):
        return None
    if np.array_equal(_solve_nearest_control_blind(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
