# Variant of 409aa875
# Dimension: D13 — 障碍物/干扰项 (obstacle)
# Change: 无障碍物或干扰项；网格中只有精灵和背景... → 镜面障碍物反射投射物

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# variant_spec:
#   concrete_objects:
#     - Triangle sprites stay exactly as in the source task: monochrome 3-pixel L or V shapes.
#     - New obstacles are 2-pixel diagonal mirror components in a fixed color.
#       "\" occupies a 2x2 main diagonal; "/" occupies a 2x2 anti-diagonal.
#     - Output still only adds maroon/blue impact marks and recolors hit sprites; mirrors stay unchanged.
#   behavior_trace:
#     - Original flow: read triangle orientation -> compute apex + 5*dir directly -> inspect that one landing cell.
#     - Variant flow: read orientation -> walk 5 unit steps -> whenever the current step enters a mirror cell,
#       rotate the direction by 90 degrees according to the mirror orientation -> inspect the final step cell.
#     - Example: a V-shape firing right would originally land at (r,c+5). With a "/" mirror on step 2, it goes
#       right,right, then turns downward and finishes at the cell reached after the remaining 3 downward steps.
#   complexity_calibration:
#     - Only one new object family is added, with two visible geometric states ("/" and "\").
#     - Generator allows at most one reflection per projectile. This keeps the polyline effect obvious while
#       avoiding deep path-search complexity or visually noisy multi-bounce layouts.
#   hidden_assumptions_review:
#     - The source solver assumed every non-bg component was a triangle; that no longer holds, so components must
#       be classified as triangle vs mirror by shape before emitting projectiles.
#     - The source solver assumed the destination is a single affine expression apex + 5*dir; with mirrors this
#       is invalid, so stepwise tracing replaces direct arithmetic.
#     - The source solver safely wrote bg impact colors because only sprites occupied non-bg cells. Now mirrors also
#       occupy non-bg cells, so final impacts landing on mirrors must be ignored rather than overwritten.
#   placement_legality:
#     - Sprites and mirrors are kept Chebyshev-separated so mirror components do not merge with each other and stay
#       visually distinct from sprites.
#     - Generator reserves all non-destination path cells as required background, preventing later object placement
#       from creating accidental extra reflections or unintended early sprite crossings.
#   observability_and_cleanliness:
#     - Every generated instance contains at least one real reflected projectile and at least one mirror.
#     - Mirror color is fixed and different from sprite/output colors so the obstacle layer is easy to isolate.
#     - The rule still has repeated support across samples because the old hit/collision outcomes are preserved,
#       but now some of those outcomes arise only after a visible bend.
#   color_inference:
#     - No color ids drive control logic except the fixed mirror color; mirror orientation is geometric, not color-
#       dependent. Output recoloring remains the same structured maroon/blue scheme as the source task.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
MIRROR_COLOR = 8
SPRITE_COLORS = [0, 2, 3, 4, 5, 6]
PROJECTILE_STEPS = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
MIRROR_KINDS = ["/", "\\"]
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 500


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    pts = set(comp)
    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in pts)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in pts)

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
        dr = arms[0][0] + arms[1][0]
        dc = arms[0][1] + arms[1][1]
        return corner, (-1 if dr > 0 else 1, -1 if dc > 0 else 1)

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


def _mirror_kind(comp: list[tuple[int, int]]) -> str | None:
    if len(comp) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(comp)
    if r1 != r0 + 1:
        return None
    if c1 == c0 + 1:
        return "\\"
    if c1 == c0 - 1:
        return "/"
    return None


def _turn_on_mirror(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind == "/":
        return dc, -dr
    return -dc, dr


def _trace_projectile(
    start: tuple[int, int],
    direction: tuple[int, int],
    mirror_by_cell: dict[tuple[int, int], str],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], int, tuple[tuple[int, int], ...]] | None:
    h, w = shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = map(int, direction)
    reflections = 0
    path: list[tuple[int, int]] = []
    for _ in range(PROJECTILE_STEPS):
        nr = cur[0] + dr
        nc = cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        cur = (int(nr), int(nc))
        path.append(cur)
        kind = mirror_by_cell.get(cur)
        if kind is not None:
            dr, dc = _turn_on_mirror((dr, dc), kind)
            reflections += 1
    return cur, reflections, tuple(path)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire 5-step projectiles from 3-pixel triangle sprites; mirrors can bend the flight path.

    Input structure:
    - bg is the most common color. Non-bg components are either 3-pixel triangle sprites (L or V) or 2-pixel diagonal mirrors ("/" or "\").

    Rule:
    - Each triangle determines an apex and launch direction exactly as in the source task.
    - The projectile now moves cell-by-cell for 5 steps; entering a mirror cell rotates its direction by 90 degrees and it keeps flying with the remaining steps.
    - After the 5th step, landing on a sprite recolors that whole sprite to maroon (9); landing on empty bg paints maroon (9), or blue (1) if multiple projectiles share that bg landing cell.
    - Mirrors are obstacles only: they deflect projectiles but are never recolored.

    Key insight: Triangle orientation still chooses the initial ray, but the visible 2-pixel diagonal mirrors reroute some shots, so the decisive target is the endpoint of a bent 5-step path rather than a straight-line offset.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sprites: list[list[tuple[int, int]]] = []
    sprite_cell_to_id: dict[tuple[int, int], int] = {}
    mirror_by_cell: dict[tuple[int, int], str] = {}

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            tri = _triangle_apex_and_dir(comp)
            if tri is not None:
                sid = len(sprites)
                sprites.append(comp)
                for cell in comp:
                    sprite_cell_to_id[cell] = sid
                continue
            kind = _mirror_kind(comp)
            if kind is not None:
                for cell in comp:
                    mirror_by_cell[cell] = kind

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sid, comp in enumerate(sprites):
        info = _triangle_apex_and_dir(comp)
        if info is None:
            continue
        apex, direction = info
        traced = _trace_projectile(apex, direction, mirror_by_cell, (h, w))
        if traced is None:
            continue
        dest, _reflections, _path = traced
        if dest in mirror_by_cell:
            continue
        dest_to_shooters[dest].append(sid)

    hit_sprites = {sprite_cell_to_id[dest] for dest in dest_to_shooters if dest in sprite_cell_to_id}
    out = g.copy()

    for sid in hit_sprites:
        for r, c in sprites[sid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in dest_to_shooters.items():
        if dest in sprite_cell_to_id or dest in mirror_by_cell:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

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
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _mirror_variants_through(cell: tuple[int, int], kind: str) -> list[dict]:
    r, c = cell
    if kind == "\\":
        raw = [((r, c), (r + 1, c + 1)), ((r - 1, c - 1), (r, c))]
    else:
        raw = [((r, c), (r + 1, c - 1)), ((r - 1, c + 1), (r, c))]
    variants = []
    seen = set()
    for cells in raw:
        if not _in_bounds_cells(cells):
            continue
        key = frozenset(cells)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            {
                "kind": kind,
                "cells": cells,
                "cell_set": key,
                "halo": _halo(cells),
            }
        )
    return variants


def _build_placements() -> list[dict]:
    placements: list[dict] = []
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                cells = _sprite_cells((ar, ac), direction)
                if not _in_bounds_cells(cells):
                    continue
                placements.append(
                    {
                        "apex": (ar, ac),
                        "dir": direction,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                    }
                )
    return placements


def _build_plans() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    plans: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)

    def register(placement: dict, mirrors: tuple[dict, ...], path: tuple[tuple[int, int], ...], reflections: int) -> None:
        mirror_cells = frozenset(cell for mirror in mirrors for cell in mirror["cells"])
        dest = path[-1]
        if dest in mirror_cells:
            return
        if len(set(path)) != len(path):
            return
        if any(mirror["cell_set"] & placement["halo"] for mirror in mirrors):
            return
        if any(placement["cell_set"] & mirror["halo"] for mirror in mirrors):
            return
        path_prefix_bg = frozenset(set(path[:-1]) - mirror_cells)
        plan = {
            "id": len(plans),
            "placement": placement,
            "mirrors": mirrors,
            "dest": dest,
            "path": path,
            "path_prefix_bg": path_prefix_bg,
            "visible_cells": frozenset(set(placement["cells"]) | set(mirror_cells)),
            "reflected": reflections > 0,
        }
        plans.append(plan)
        by_dest[dest].append(plan)

    for placement in _build_placements():
        traced = _trace_projectile(placement["apex"], placement["dir"], {}, (H, W))
        if traced is not None:
            _dest, reflections, path = traced
            if reflections == 0:
                register(placement, tuple(), path, reflections)

        straight = _trace_projectile(placement["apex"], placement["dir"], {}, (H, W))
        if straight is None:
            continue
        _dest0, _ref0, path0 = straight
        for hit_idx in range(1, PROJECTILE_STEPS):
            hit_cell = path0[hit_idx - 1]
            for kind in MIRROR_KINDS:
                for mirror in _mirror_variants_through(hit_cell, kind):
                    mirror_by_cell = {cell: kind for cell in mirror["cells"]}
                    traced = _trace_projectile(placement["apex"], placement["dir"], mirror_by_cell, (H, W))
                    if traced is None:
                        continue
                    _dest, reflections, path = traced
                    if reflections != 1:
                        continue
                    if path[hit_idx - 1] != hit_cell:
                        continue
                    register(placement, (mirror,), path, reflections)

    return plans, by_dest


PLANS, PLANS_BY_DEST = _build_plans()


def _reflection_ok(plan: dict, require_reflection: bool | None) -> bool:
    if require_reflection is None:
        return True
    return bool(plan["reflected"]) == bool(require_reflection)


def _plan_fits(
    plan: dict,
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
    *,
    allow_dest_occupied: bool,
) -> bool:
    placement = plan["placement"]
    if placement["halo"] & occupied_visible:
        return False
    if placement["cell_set"] & reserved_bg:
        return False
    if placement["cell_set"] & reserved_path_bg:
        return False
    for mirror in plan["mirrors"]:
        if mirror["halo"] & occupied_visible:
            return False
        if mirror["cell_set"] & reserved_bg:
            return False
        if mirror["cell_set"] & reserved_path_bg:
            return False
    if plan["path_prefix_bg"] & (occupied_visible | reserved_bg):
        return False
    if plan["dest"] in reserved_bg or plan["dest"] in reserved_path_bg:
        return False
    if not allow_dest_occupied and plan["dest"] in occupied_visible:
        return False
    return True


def _commit_plan(
    sprites: list[dict],
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
    plan: dict,
    *,
    target_kind: str,
    target_sprite: int | None = None,
) -> None:
    occupied_visible.update(plan["visible_cells"])
    reserved_path_bg.update(plan["path_prefix_bg"])
    if target_kind == "bg":
        reserved_bg.add(plan["dest"])
        sprites.append({"plan": plan, "target": plan["dest"], "target_kind": "bg"})
    else:
        sprites.append(
            {
                "plan": plan,
                "target": plan["dest"],
                "target_kind": "hit",
                "target_sprite": int(target_sprite),
            }
        )


def _choose_collision_pair(
    rng: random.Random,
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
    *,
    require_reflection: bool = False,
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLANS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied_visible or dest in reserved_bg or dest in reserved_path_bg:
            continue
        candidates = [
            plan
            for plan in PLANS_BY_DEST[dest]
            if _plan_fits(plan, occupied_visible, reserved_bg, reserved_path_bg, allow_dest_occupied=False)
        ]
        straight = [plan for plan in candidates if not plan["reflected"]]
        reflected = [plan for plan in candidates if plan["reflected"]]
        groups = []
        if require_reflection:
            groups = [reflected + straight]
        else:
            if len(straight) >= 2:
                groups.append(straight)
            groups.append(candidates)
        for group in groups:
            pool = group[:]
            rng.shuffle(pool)
            for i, first in enumerate(pool):
                occ2 = occupied_visible | set(first["visible_cells"])
                path2 = reserved_path_bg | set(first["path_prefix_bg"])
                for second in pool[i + 1 :]:
                    if not _plan_fits(second, occ2, reserved_bg, path2, allow_dest_occupied=False):
                        continue
                    if require_reflection and not (first["reflected"] or second["reflected"]):
                        continue
                    return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
    *,
    require_reflection: bool | None = None,
) -> dict | None:
    dests = list(PLANS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied_visible or dest in reserved_bg or dest in reserved_path_bg:
            continue
        candidates = [
            plan
            for plan in PLANS_BY_DEST[dest]
            if _reflection_ok(plan, require_reflection)
            and _plan_fits(plan, occupied_visible, reserved_bg, reserved_path_bg, allow_dest_occupied=False)
        ]
        if not candidates:
            continue
        if require_reflection is None:
            straight = [plan for plan in candidates if not plan["reflected"]]
            if straight:
                return rng.choice(straight)
        return rng.choice(candidates)
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["plan"]["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells or cell in reserved_bg or cell in reserved_path_bg:
            continue
        candidates = [
            plan
            for plan in PLANS_BY_DEST[cell]
            if _plan_fits(plan, occupied_visible, reserved_bg, reserved_path_bg, allow_dest_occupied=True)
        ]
        if candidates:
            straight = [plan for plan in candidates if not plan["reflected"]]
            return (rng.choice(straight if straight else candidates), cell)
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    sprite_count = 0
    mirror_count = 0
    for col in sorted(set(int(v) for v in np.unique(grid)) - {BG}):
        for comp in _cc8(grid == col):
            if _triangle_apex_and_dir(comp) is not None:
                sprite_count += 1
            elif _mirror_kind(comp) is not None:
                mirror_count += 1
            else:
                return False
    return sprite_count == n_sprites and mirror_count >= 1


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


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _paint_mirror(grid: np.ndarray, mirror: dict) -> None:
    for r, c in mirror["cells"]:
        grid[r, c] = MIRROR_COLOR


def _place_collisions(
    rng: random.Random,
    e_coll: int,
    sprites: list[dict],
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied_visible, reserved_bg, reserved_path_bg)
        if choice is None:
            return False
        _dest, pair = choice
        for plan in pair:
            _commit_plan(sprites, occupied_visible, reserved_bg, reserved_path_bg, plan, target_kind="bg")
    return True


def _place_unique_landings(
    rng: random.Random,
    count: int,
    sprites: list[dict],
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        plan = _choose_unique(rng, occupied_visible, reserved_bg, reserved_path_bg)
        if plan is None:
            return False
        _commit_plan(sprites, occupied_visible, reserved_bg, reserved_path_bg, plan, target_kind="bg")
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    sprites: list[dict],
    occupied_visible: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    reserved_path_bg: set[tuple[int, int]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied_visible, reserved_bg, reserved_path_bg, used_target_cells)
        if choice is None:
            return False
        plan, cell = choice
        if plan["dest"] != cell:
            return False
        _commit_plan(
            sprites,
            occupied_visible,
            reserved_bg,
            reserved_path_bg,
            plan,
            target_kind="hit",
            target_sprite=idx,
        )
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(entry["target"] for entry in sprites if entry["target_kind"] == "bg")
    hit_sprite_ids = {entry["target_sprite"] for entry in sprites if entry["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["plan"]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied_visible: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    reserved_path_bg: set[tuple[int, int]] = set()
    consumed_unique = 0

    if unique_total > 0:
        plan = _choose_unique(
            rng,
            occupied_visible,
            reserved_bg,
            reserved_path_bg,
            require_reflection=True,
        )
        if plan is None:
            return None
        _commit_plan(sprites, occupied_visible, reserved_bg, reserved_path_bg, plan, target_kind="bg")
        consumed_unique = 1
    else:
        choice = _choose_collision_pair(
            rng,
            occupied_visible,
            reserved_bg,
            reserved_path_bg,
            require_reflection=True,
        )
        if choice is None:
            return None
        _dest, pair = choice
        for plan in pair:
            _commit_plan(sprites, occupied_visible, reserved_bg, reserved_path_bg, plan, target_kind="bg")
        e_coll -= 1

    reflected_count = sum(1 for entry in sprites if entry["plan"]["reflected"])
    if reflected_count < 1:
        return None

    base_needed = max(0, e_hit - len(sprites))
    remaining_unique_before_hits = unique_total - consumed_unique
    if base_needed > remaining_unique_before_hits:
        return None

    if not _place_collisions(rng, e_coll, sprites, occupied_visible, reserved_bg, reserved_path_bg):
        return None
    if not _place_unique_landings(
        rng,
        base_needed,
        sprites,
        occupied_visible,
        reserved_bg,
        reserved_path_bg,
    ):
        return None
    if not _place_hits(rng, e_hit, sprites, occupied_visible, reserved_bg, reserved_path_bg):
        return None

    remaining = n_sprites - len(sprites)
    expected_remaining = remaining_unique_before_hits - base_needed
    if remaining != expected_remaining:
        return None
    if remaining < 0 or not _place_unique_landings(
        rng,
        remaining,
        sprites,
        occupied_visible,
        reserved_bg,
        reserved_path_bg,
    ):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for entry in sprites:
        _paint_sprite(inp, entry["plan"]["placement"], sprite_color)
    painted_mirrors: set[frozenset[tuple[int, int]]] = set()
    for entry in sprites:
        for mirror in entry["plan"]["mirrors"]:
            if mirror["cell_set"] in painted_mirrors:
                continue
            painted_mirrors.add(mirror["cell_set"])
            _paint_mirror(inp, mirror)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(inp == MIRROR_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
