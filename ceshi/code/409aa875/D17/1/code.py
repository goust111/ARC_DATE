# Variant of 409aa875
# Dimension: D17 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码或图例；所有规则由精灵形状隐式编码... → 控制像素编码投射距离

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# variant_spec:
#   concreteization:
#     controls:
#       - The grid now contains exactly three border singleton pixels used as a legend.
#       - Their colors are fixed small categories: 2 -> distance 3, 4 -> distance 5, 6 -> distance 7.
#       - A triangle does not read distance from its own color or shape; it reads the Manhattan-nearest
#         control pixel to its apex. Direction still comes only from the 3-pixel triangle geometry.
#       - The generator enforces a unique nearest control with a visibility margin, so the intended control
#         is visually clear instead of depending on hidden tie-breaks.
#     preserved_from_original:
#       - Every sprite is still a monochrome 3-pixel component.
#       - L-shapes still fire diagonally and V-shapes still fire axially.
#       - Hit-sprite recolor and bg-collision coloring stay unchanged.
#   behavior_trace:
#     example:
#       - Suppose an L sprite has apex (8, 8) and direction (-1, -1).
#       - Original task: fixed range 5, so it lands at (3, 3).
#       - Variant: if its nearest border control is color 2, the range is 3, so it lands at (5, 5).
#       - Same sprite, same direction, different endpoint solely because the edge legend changed.
#   complexity_calibration:
#     - Only three range categories are used: 3 / 5 / 7.
#     - Each generated grid uses at least two control colors, and at least one used distance is not 5.
#     - This keeps the rule observable while avoiding too many arbitrary cases.
#   implicit_assumption_review:
#     - Original solver assumption: every non-bg component is a candidate sprite.
#       This is no longer true because border legend pixels are also non-bg; the new solver filters to
#       interior 3-cell triangle components only.
#     - Original solver assumption: every projectile travels exactly 5 cells.
#       This is replaced by a nearest-control lookup per sprite.
#     - Original solver assumption: any occupied target cell belongs to a hittable sprite.
#       This would wrongly let legend pixels be "hit", so the new solver only registers triangle cells as
#       hittable targets and ignores non-bg, non-sprite cells.
#   placement_legality:
#     - Controls are isolated edge singletons on well-spaced anchor slots.
#     - Sprites are kept fully in the interior, away from the border legend and from each other.
#     - Landing cells are also kept in the interior so output markers do not overlap the legend.
#   visual_observability:
#     - A human can compare nearby sprites and see that those nearest one legend shoot shorter/farther than
#       those nearest another, while shape still explains direction.
#     - The generator rejects samples that collapse back to the fixed-range behavior.
#   visual_cleanliness:
#     - The legend occupies only three isolated border cells.
#     - All projectile effects stay inside the field, so the border instruction layer remains readable.
#   color_inference:
#     - Only three instruction colors participate in the new rule.
#     - Sprite colors avoid these legend colors and avoid the output colors 1/9, so the legend remains easy
#       to separate from both the emitters and the result markers.
# main_assumption_audit:
#   - `_triangle_components` assumes every real emitter is an interior 3-cell monochrome triangle.
#     The generator guarantees that.
#   - `_nearest_control` assumes a unique closest legend pixel for every sprite apex.
#     The generator guarantees a margin of at least 2 Manhattan units between the closest and second-closest
#     controls for every placed sprite.
#   - Output painting assumes projectile destinations never need to overwrite legend pixels.
#     The generator guarantees all landing cells are interior.


# ---- shared canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1

# Control legend: color -> projectile distance
CONTROL_TO_DIST = {2: 3, 4: 5, 6: 7}
CONTROL_COLORS = tuple(sorted(CONTROL_TO_DIST))

# Keep sprite colors visually separate from control/output colors.
SPRITE_COLORS = [0, 3, 5, 8]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
FIXED_DIST_BASELINE = 5

# ---- generation counts ----
SPRITE_COUNT_RANGE = (5, 8)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 400

# ---- legend placement ----
EDGE_SLOTS = [
    (0, 2), (0, 8), (0, 13),
    (15, 2), (15, 8), (15, 13),
    (2, 0), (8, 0), (13, 0),
    (2, 15), (8, 15), (13, 15),
]


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


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
        coords.append((h - 1, c))
    for r in range(1, h - 1):
        coords.append((r, 0))
        coords.append((r, w - 1))
    return coords


def _edge_controls(g: np.ndarray, bg: int) -> list[dict]:
    controls = []
    for r, c in _border_coords(*g.shape):
        col = int(g[r, c])
        if col == bg or col not in CONTROL_TO_DIST:
            continue
        controls.append({"pos": (r, c), "color": col, "dist": CONTROL_TO_DIST[col]})
    return controls


def _nearest_control(
    apex: tuple[int, int],
    controls: list[dict],
    min_margin: int = 1,
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
    comps: list[dict] = []
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
            cid = len(comps)
            comps.append({
                "cells": tuple(comp),
                "apex": apex,
                "dir": direction,
                "color": col,
            })
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _dest_to_shooters(
    triangles: list[dict],
    controls: list[dict],
    h: int,
    w: int,
    fixed_dist: int | None = None,
) -> dict[tuple[int, int], list[int]]:
    dests: dict[tuple[int, int], list[int]] = {}
    for cid, tri in enumerate(triangles):
        dist = fixed_dist
        if dist is None:
            ctl = _nearest_control(tri["apex"], controls, min_margin=1)
            if ctl is None:
                continue
            dist = ctl["dist"]
        ar, ac = tri["apex"]
        dr, dc = tri["dir"]
        rr = ar + dist * dr
        cc = ac + dist * dc
        if 0 <= rr < h and 0 <= cc < w:
            dests.setdefault((rr, cc), []).append(cid)
    return dests


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
    dest_to_shooters = _dest_to_shooters(triangles, [], *g.shape, fixed_dist=FIXED_DIST_BASELINE)
    return _apply_projectiles(g, bg, triangles, cell_to_comp, dest_to_shooters)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; the border legend decides how far they travel.

    Input structure:
    - bg = most common color. Interior non-bg 3-pixel components are triangle sprites shaped as axis-aligned L's or diagonal V's.
    - Three singleton control pixels sit on the border; their colors encode projectile ranges 3, 5, and 7.

    Rule:
    - Each triangle still derives its apex and firing direction from its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis opposite their opening.
    - The sprite looks at its apex, finds the nearest border control pixel, and uses that control color's range for the shot.
    - If the landing cell hits another sprite, recolor that whole sprite to maroon (9).
    - If two projectiles land on the same empty bg cell, paint it blue (1); a lone projectile on empty bg paints maroon (9). Legend pixels themselves stay unchanged.

    Key insight: Shape still explains direction, but the edge legend now externalizes the range, so nearby sprites can shoot different distances without changing their geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    controls = _edge_controls(g, bg)
    triangles, cell_to_comp = _triangle_components(g, bg)
    dest_to_shooters = _dest_to_shooters(triangles, controls, *g.shape)
    return _apply_projectiles(g, bg, triangles, cell_to_comp, dest_to_shooters)


# --------------- generator ---------------


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


def _buffer(points: set[tuple[int, int]], radius: int = 1) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in points
        for rr in range(max(0, r - radius), min(H, r + radius + 1))
        for cc in range(max(0, c - radius), min(W, c + radius + 1))
    )


def _choose_controls(rng: random.Random) -> list[dict]:
    for _ in range(200):
        slots = EDGE_SLOTS[:]
        rng.shuffle(slots)
        chosen: list[tuple[int, int]] = []
        for pos in slots:
            if all(abs(pos[0] - r) + abs(pos[1] - c) >= 8 for r, c in chosen):
                chosen.append(pos)
            if len(chosen) == len(CONTROL_COLORS):
                colors = list(CONTROL_COLORS)
                rng.shuffle(colors)
                return [
                    {"pos": pos, "color": col, "dist": CONTROL_TO_DIST[col]}
                    for pos, col in zip(chosen, colors)
                ]
    raise RuntimeError("failed to place border controls")


def _build_placements(controls: list[dict]) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    control_positions = {ctl["pos"] for ctl in controls}
    control_buffer = _buffer(control_positions, radius=1)

    for ar in range(1, H - 1):
        for ac in range(1, W - 1):
            ctl = _nearest_control((ar, ac), controls, min_margin=2)
            if ctl is None:
                continue
            dist = ctl["dist"]
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                if any(r == 0 or c == 0 or r == H - 1 or c == W - 1 for r, c in cells):
                    continue
                if any(cell in control_buffer for cell in cells):
                    continue
                dest = (ar + dist * dr, ac + dist * dc)
                if not (0 < dest[0] < H - 1 and 0 < dest[1] < W - 1):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "distance": dist,
                    "control_color": ctl["color"],
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


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


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
    allowed_control_colors: set[int] | None = None,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if allowed_control_colors is not None and placement["control_color"] not in allowed_control_colors:
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
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, controls: list[dict]) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False

    seen_controls = {(ctl["pos"], ctl["color"]) for ctl in controls}
    for pos, color in seen_controls:
        if int(grid[pos]) != color:
            return False

    for r, c in _border_coords(H, W):
        v = int(grid[r, c])
        if v == BG:
            continue
        if ((r, c), v) not in seen_controls:
            return False

    triangles, _ = _triangle_components(grid, BG)
    if len(triangles) != n_sprites:
        return False
    return all(_nearest_control(tri["apex"], controls, min_margin=2) is not None for tri in triangles)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 2:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg, placements_by_dest) -> bool:
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
    rng,
    n,
    sprites,
    occupied,
    reserved_bg,
    placements,
    allowed_control_colors: set[int] | None = None,
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, placements, occupied, reserved_bg, allowed_control_colors)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_mandatory_unique_landings(
    rng,
    colors: list[int],
    sprites,
    occupied,
    reserved_bg,
    placements,
) -> bool:
    for color in colors:
        placement = _choose_unique(rng, placements, occupied, reserved_bg, {color})
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg, placements_by_dest) -> bool:
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
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _select_mandatory_control_colors(rng: random.Random, unique_total: int) -> list[int]:
    first = rng.choice([2, 6])
    remaining = [c for c in CONTROL_COLORS if c != first]
    second = rng.choice(remaining)
    chosen = [first, second]
    last = [c for c in CONTROL_COLORS if c not in chosen]
    if unique_total >= 3 and last and rng.random() < 0.6:
        chosen.append(last[0])
    return chosen


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    controls = _choose_controls(rng)
    placements, placements_by_dest = _build_placements(controls)
    if len(placements) < 80:
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

    mandatory_colors = _select_mandatory_control_colors(rng, unique_total)
    if not _place_mandatory_unique_landings(
        rng,
        mandatory_colors,
        sprites,
        occupied,
        reserved_bg,
        placements,
    ):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if len(mandatory_colors) + base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg, placements):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg, placements_by_dest):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg, placements):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for ctl in controls:
        inp[ctl["pos"]] = ctl["color"]
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, controls):
        return None

    used_distances = {sprite["placement"]["distance"] for sprite in sprites}
    if len(used_distances) < 2 or used_distances == {FIXED_DIST_BASELINE}:
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

    if np.array_equal(_solve_fixed5_blind(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
