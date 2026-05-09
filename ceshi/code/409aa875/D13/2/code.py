# Variant of 409aa875
# Dimension: D13 — 障碍物/干扰项 (obstacle)
# Change: 无障碍物或干扰项；网格中只有精灵和背景... → 传送门（成对出现，入一个出另一个）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D13 portal variant
# a) Concretization and behavior trace
#    - Triangles stay exactly as in the base task: monochrome 3-pixel L or V sprites.
#    - A portal is a single non-background pixel. Exactly two portal pixels share one color,
#      which makes the pairing visible directly in the input.
#    - A projectile still advances for exactly 5 unit steps along the direction encoded by the
#      triangle. The only new interaction is: after each step, if the projectile lands on a
#      portal pixel, it is moved immediately to the paired portal and the same step is considered
#      spent; the projectile then continues with any remaining steps in the original direction.
#    - Small trace: suppose an L-sprite fires down-right from apex (2,2). In the base task its
#      landing cell is (7,7). In the variant, if step 3 reaches portal A at (5,5) and portal B
#      is at (10,3), then after step 3 the projectile is at (10,3); steps 4 and 5 continue to
#      (11,4) and (12,5). The visible effect is therefore at (12,5), not at the straight-line
#      cell (7,7).
# b) Complexity calibration
#    - The portal mechanism must be obvious, so the generator always places exactly one portal pair
#      and forces at least two shooters to use it. Those two shooters collide on one empty cell,
#      producing a blue marker that cannot be explained by straight 5-step flight.
#    - Portal colors are chosen from a small palette distinct from the sprite color and distinct
#      from the output-only colors 1 and 9. This avoids ambiguous "was that already recolored?"
#      readings.
# c) Hidden assumption audit from the original solver
#    - Original component scan assumed every non-bg component was a sprite. That is no longer
#      true, so the solver now separates 3-pixel triangles from 1-pixel portals.
#    - Original destination code `apex + 5*dir` assumed no mid-path interaction. That is the core
#      broken assumption under D13, so it is replaced with step-by-step tracing.
#    - Original hit lookup over `cell_to_comp` still works, but only if portal cells are excluded
#      from that map. The new solver only maps triangle cells.
#    - Original output painting was harmless because destinations were only bg or sprite cells.
#      The generator now guarantees that valid final destinations are never portal cells, so the
#      solver does not need extra overwrite rules for portals.
# d) Placement legality
#    - Sprites must remain 8-disconnected from each other and from the portal pixels, so the
#      generator treats portal cells as occupied seeds when enforcing sprite halos.
#    - Portal pixels are chosen away from each other (Chebyshev distance >= 4) and away from the
#      border interior enough to allow several legal teleported trajectories.
#    - Every generated example is validated by running `main(input)` and comparing to the
#      generator's output-first reconstruction.
# e) Visual observability
#    - The repeated portal color is the only repeated non-sprite color in the input, so humans can
#      infer that the two singleton pixels are linked.
#    - Because at least two projectiles use the same portal pair and create one blue collision, the
#      new mechanic is not a rare corner case hidden among ordinary straight shots.
# f) Visual cleanliness
#    - There is only one portal pair per grid, reducing clutter.
#    - Portals do not touch sprites, and output markers are kept off portal cells.
# g) Color inference
#    - No hidden ordering or color-id arithmetic is used. Portal pairing is purely by visible color
#      equality; output recolors remain the same structural rule as the base task.


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


def _find_triangles_and_portals(
    g: np.ndarray, bg: int
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    triangles: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    singleton_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            if len(comp) == 3 and _triangle_apex_and_dir(comp) is not None:
                cid = len(triangles)
                triangles.append(comp)
                for rc in comp:
                    cell_to_comp[rc] = cid
            elif len(comp) == 1:
                singleton_by_color[col].append(comp[0])

    portal_map: dict[tuple[int, int], tuple[int, int]] = {}
    for cells in singleton_by_color.values():
        if len(cells) == 2:
            a, b = cells
            portal_map[a] = b
            portal_map[b] = a

    return triangles, cell_to_comp, portal_map


def _trace_projectile(
    apex: tuple[int, int],
    direction: tuple[int, int],
    portal_map: dict[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
    steps: int = 5,
) -> tuple[tuple[int, int], bool] | None:
    h, w = shape
    r, c = apex
    dr, dc = direction
    used_portal = False
    for _ in range(steps):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in portal_map:
            r, c = portal_map[(r, c)]
            used_portal = True
    return (r, c), used_portal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle projectiles for 5 steps, with same-color singleton portals warping the path.

    Input structure:
    - Background is the most common color.
    - Triangle sprites are still monochrome 3-pixel L or V components.
    - A portal pair is shown as two isolated single pixels of the same non-background color.

    Rule:
    - Each triangle determines its apex and firing direction exactly as in the base task.
    - The projectile advances one cell at a time for 5 steps; whenever a step lands on one portal
      pixel, the projectile is moved immediately to the paired portal and then continues in the
      same direction for any remaining steps.
    - After the full 5-step path is resolved, hitting a triangle recolors that whole triangle to
      maroon (9); multiple projectiles sharing one empty background destination paint it blue (1);
      a lone projectile on empty background paints maroon (9).

    Key insight: The triangle still tells you the direction, but the final effect depends on
    tracing the projectile through the paired singleton portal rather than reading only apex+5*dir.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps, cell_to_comp, portal_map = _find_triangles_and_portals(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        trace = _trace_projectile(ad[0], ad[1], portal_map, g.shape)
        if trace is None:
            continue
        dest, _used_portal = trace
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
        if len(shooters) >= 2:
            out[r, c] = 1
        else:
            out[r, c] = 9

    return out


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]
PORTAL_COLORS = [0, 2, 3, 4, 5, 6, 8]
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_ATTEMPTS = 300
PORTAL_INTERIOR = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
EXTRA_PORTAL_BIAS = 0.35


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_base_placements() -> list[dict]:
    placements: list[dict] = []
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                direct_dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= direct_dest[0] < H and 0 <= direct_dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not all(0 <= r < H and 0 <= c < W for r, c in cells):
                    continue
                placements.append(
                    {
                        "apex": (ar, ac),
                        "dir": (dr, dc),
                        "direct_dest": direct_dest,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                    }
                )
    return placements


BASE_PLACEMENTS = _build_base_placements()


def _build_portal_placements(
    portal_map: dict[tuple[int, int], tuple[int, int]]
) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    portal_cells = set(portal_map)
    for base in BASE_PLACEMENTS:
        if base["cell_set"] & portal_cells:
            continue
        trace = _trace_projectile(base["apex"], base["dir"], portal_map, (H, W))
        if trace is None:
            continue
        dest, used_portal = trace
        if dest in portal_cells:
            continue
        placement = dict(base)
        placement["dest"] = dest
        placement["used_portal"] = used_portal
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
    require_portal: bool,
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(placements_by_dest)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [
            p
            for p in placements_by_dest[dest]
            if _fits(p, occupied, reserved_bg) and (p["used_portal"] or not require_portal)
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                if require_portal and not second["used_portal"]:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    prefer_portal: bool,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    primary = []
    fallback = []
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if not _fits(placement, occupied, reserved_bg):
            continue
        if placement["used_portal"]:
            primary.append(placement)
        else:
            fallback.append(placement)
    if prefer_portal and primary:
        return primary[0]
    if not prefer_portal and fallback:
        return fallback[0]
    if primary:
        return primary[0]
    return fallback[0] if fallback else None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
    prefer_portal: bool,
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [
            p
            for p in placements_by_dest[cell]
            if _fits(p, occupied, reserved_bg) and not (p["cell_set"] & set(target_sprite["placement"]["cells"]))
        ]
        if not candidates:
            continue
        portal_candidates = [p for p in candidates if p["used_portal"]]
        plain_candidates = [p for p in candidates if not p["used_portal"]]
        if prefer_portal and portal_candidates:
            return rng.choice(portal_candidates), cell
        if not prefer_portal and plain_candidates:
            return rng.choice(plain_candidates), cell
        return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False

    tri_count = 0
    portal_counts: list[int] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {BG}):
        singletons = 0
        for comp in _cc8(grid == col):
            if len(comp) == 3 and _triangle_apex_and_dir(comp) is not None:
                tri_count += 1
            elif len(comp) == 1:
                singletons += 1
            else:
                return False
        if singletons:
            portal_counts.append(singletons)

    return tri_count == n_sprites and portal_counts == [2]


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 3) + 1):
        unique = n_sprites - 2 - e_hit
        if unique >= 1:
            options.append((e_hit, unique))
    return rng.choice(options) if options else None


def _place_unique_landings(
    rng: random.Random,
    n: int,
    placements: list[dict],
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, placements, occupied, reserved_bg, prefer_portal=rng.random() < EXTRA_PORTAL_BIAS)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append(
            {
                "placement": placement,
                "target": placement["dest"],
                "target_kind": "bg",
                "used_portal": placement["used_portal"],
            }
        )
        occupied.update(placement["cells"])
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(
            rng,
            sprites[idx],
            placements_by_dest,
            occupied,
            reserved_bg,
            used_target_cells,
            prefer_portal=rng.random() < EXTRA_PORTAL_BIAS,
        )
        if choice is None:
            return False
        placement, cell = choice
        sprites.append(
            {
                "placement": placement,
                "target": cell,
                "target_kind": "hit",
                "target_sprite": idx,
                "used_portal": placement["used_portal"],
            }
        )
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


def _random_portal_pair(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        a, b = rng.sample(PORTAL_INTERIOR, 2)
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 4:
            return a, b


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    portal_color_choices = [c for c in PORTAL_COLORS if c != sprite_color]
    portal_color = rng.choice(portal_color_choices)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, unique_count = counts

    for _ in range(40):
        p1, p2 = _random_portal_pair(rng)
        portal_map = {p1: p2, p2: p1}
        placements, placements_by_dest = _build_portal_placements(portal_map)
        used_portal_count = sum(1 for p in placements if p["used_portal"])
        if used_portal_count < 12:
            continue

        sprites: list[dict] = []
        occupied: set[tuple[int, int]] = {p1, p2}
        reserved_bg: set[tuple[int, int]] = set()

        mandatory = _choose_collision_pair(
            rng,
            placements_by_dest,
            occupied,
            reserved_bg,
            require_portal=True,
        )
        if mandatory is None:
            continue
        dest, pair = mandatory
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append(
                {
                    "placement": placement,
                    "target": dest,
                    "target_kind": "bg",
                    "used_portal": placement["used_portal"],
                }
            )
            occupied.update(placement["cells"])

        if not _place_hits(rng, e_hit, placements_by_dest, sprites, occupied, reserved_bg):
            continue
        if not _place_unique_landings(rng, unique_count, placements, sprites, occupied, reserved_bg):
            continue
        if len(sprites) != n_sprites:
            continue

        inp = np.full((H, W), BG, dtype=int)
        inp[p1] = portal_color
        inp[p2] = portal_color
        for sprite in sprites:
            _paint_sprite(inp, sprite["placement"], sprite_color)

        if not _validate_input(inp, n_sprites):
            continue

        out = _render_output(inp, sprites)
        if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
            continue
        if not np.any(out == HIT_COLOR):
            continue
        if not np.any(out == COLLISION_COLOR):
            continue
        if np.array_equal(inp, out):
            continue
        if sum(1 for s in sprites if s["used_portal"]) < 2:
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
