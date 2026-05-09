# Combo variant of 409aa875 (index 19)
# combo_spec: D1/3+D6/0
# Name: 重力抛物线路径+连续轨迹
#
# Design analysis
# ===============
# a) Concretization and behavior trace
# - Input keeps the original object language: a 16x16 grid with bg=7 and 4-9
#   isolated monochrome 3-cell triangle sprites. L-shapes still encode diagonal
#   launch directions; V-shapes still encode axis launch directions.
# - The fusion changes both "where the projectile goes" and "what becomes visible".
# - I model the gravity arc as a 5-cell discrete flight:
#   step 1 follows the sprite's original launch direction exactly once,
#   then steps 2..5 move by (down 1 row, same horizontal drift sign).
# - Example with apex=(6, 7) and launch_dir=(-1, -1):
#   original task path:      (5,6),(4,5),(3,4),(2,3),(1,2)
#   D1-only gravity-flow:    (7,6),(8,5),(9,4),(10,3),(11,2)
#   fused combo path here:   (5,6),(6,5),(7,4),(8,3),(9,2)
#   The first cell reveals the original up-left launch, then gravity bends the
#   remaining trail downward while preserving leftward drift.
# - Output paints every in-flight background cell on that fused path in maroon 9.
#   The landing cell still keeps the old endpoint semantics: sprite hit => recolor
#   whole target sprite to 9; shared empty landing => 1; lone empty landing => 9.
#
# b) Complexity calibration
# - The path family still has only 8 launch categories inherited from the source
#   sprites, so examples repeat enough structure for humans to infer the rule.
# - The visible change is strong: compared with the original hidden teleport, each
#   accepted sample shows multiple explicit trail cells. Compared with straight
#   trails, at least one shooter in every accepted sample has a bent path.
# - The bend is simple enough to read: "one launch kick, then gravity takes over".
#
# c) Hidden assumptions from the original main()
# - Original code assumed `apex + 5*dir` was both the full motion model and the
#   only visible effect. Both assumptions break here.
# - Component parsing and apex/direction inference remain valid because inputs are
#   still trajectory-free and sprites remain isolated 3-cell objects.
# - Write order now matters: trail cells must be painted before landing overrides,
#   so a shared landing can turn blue even if the last step would otherwise be 9.
# - Hit-sprite detection must be based on the input sprite occupancy, not on the
#   partially painted output, otherwise trails could be mistaken for new objects.
#
# d) Placement legality
# - Every sprite, every trail cell, and every landing cell must stay in bounds.
# - Generator forbids any intermediate trail cell from touching any sprite cell,
#   including the shooter's own body; this avoids ambiguous overlap and removes the
#   awkward "trail immediately re-enters the source" cases.
# - Different sprites still keep Chebyshev separation >= 2 via halo checks.
# - Distinct trails may share only a deliberately shared landing cell in a bg
#   collision event; all non-endpoint trail cells are reserved as background-only.
#
# e) Visual observability
# - A solver can now see both dimensions in the pixel output:
#   the first step identifies the original launch orientation, while the remaining
#   descending trail exposes the gravity direction.
# - Accepted samples are rejected if the combo output matches either the old task,
#   a no-trail arc solver, or a straight-trail solver. This prevents visual
#   degeneration into "only D1 changed" or "only D6 changed".
#
# f) Visual cleanliness
# - Trails use only one extra color (9), and collisions keep the original blue 1.
# - The generator reserves mid-trail cells so the board does not turn into an
#   unreadable mesh of overlapping arcs.
# - Event counts remain small (0-2 hit sprites, 0-2 shared-bg collisions).
#
# g) Color inference
# - Color semantics stay tightly structured:
#   9 = visible flight cells, lone bg landing, or hit sprite recolor;
#   1 = shared bg landing;
#   sprite source color remains the original single non-bg color.
# - No rule depends on numeric ordering of colors; only geometry and multiplicity
#   determine recoloring.
#
# Post-change assumptions audited against generate_input()
# - main() assumes each non-bg component is still one isolated 3-cell triangle:
#   generate_input() only paints source sprites into the input.
# - main() assumes all visible mid-trail cells are background in the input:
#   generator reserves them before adding later sprites.
# - main() assumes path-shape differences are observable:
#   generator requires at least one bent shooter and enough visible trail gain.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, get_bg, neighbors4, neighbors8


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


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _straight_path(apex: tuple[int, int], launch_dir: tuple[int, int], steps: int = 5) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = launch_dir
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, steps + 1))


def _arc_path(apex: tuple[int, int], launch_dir: tuple[int, int], steps: int = 5) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = launch_dir
    horiz = _sign(dc)
    r, c = ar, ac
    path: list[tuple[int, int]] = []
    for step in range(steps):
        if step == 0:
            r += dr
            c += dc
        else:
            r += 1
            c += horiz
        path.append((r, c))
    return tuple(path)


def _all_in_bounds(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _solve(grid: np.ndarray, path_builder, *, draw_trail: bool) -> np.ndarray:
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

    shots: list[tuple[tuple[int, int], ...]] = []
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        path = path_builder(*ad)
        if not path or not _all_in_bounds(path, h, w):
            continue
        sid = len(shots)
        shots.append(path)
        dest_to_shooters[path[-1]].append(sid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    if draw_trail:
        for path in shots:
            for r, c in path[:-1]:
                if g[r, c] == bg:
                    out[r, c] = TRAIL_COLOR

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangle sprites fire projectiles along a visible gravity-bent arc.

    Input structure:
    - The background is the most frequent color.
    - Every non-background component is a monochrome 3-cell triangle sprite, either an axis-aligned L or a diagonal V.

    Rule:
    - Infer each sprite's apex and original launch direction from its geometry exactly as in the base task.
    - The projectile now follows a 5-cell discrete gravity arc: the first step uses the original launch direction, then the next four steps each move one row downward while keeping the launch direction's horizontal sign.
    - Paint every non-endpoint background cell on that arc maroon (9).
    - At the final cell, keep the original endpoint semantics: a hit sprite is recolored entirely to maroon (9), a shared empty landing cell becomes blue (1), and a lone empty landing cell becomes maroon (9).

    Key insight:
    - The sprite shape still tells you the initial launch vector, but the output also reveals gravity directly: each trail starts with one launch-kick cell and then bends into a downward drift, while endpoint color still depends on collision multiplicity.
    """
    return _solve(np.asarray(grid, dtype=int), _arc_path, draw_trail=True)


def _main_original_straight(grid: np.ndarray) -> np.ndarray:
    return _solve(np.asarray(grid, dtype=int), _straight_path, draw_trail=False)


def _main_arc_no_trail(grid: np.ndarray) -> np.ndarray:
    return _solve(np.asarray(grid, dtype=int), _arc_path, draw_trail=False)


def _main_straight_trail(grid: np.ndarray) -> np.ndarray:
    return _solve(np.asarray(grid, dtype=int), _straight_path, draw_trail=True)


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
TRAIL_COLOR = 9
# Keep source sprites off the semantic output colors so recolors and shared
# landings stay visually attributable to the rule rather than the input palette.
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_STEPS = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 900
SCENARIO_CYCLE = (
    "hit_only",
    "collision_only",
    "hit_collision",
    "full_mix",
)
_GENERATE_CALL_COUNT = 0


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _is_bent(direction: tuple[int, int]) -> bool:
    dr, dc = direction
    return (dr, dc) != (1, _sign(dc))


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            apex = (ar, ac)
            for direction in ALL_DIRS:
                cells = _sprite_cells(apex, direction)
                if not _in_bounds_cells(cells):
                    continue
                path = _arc_path(apex, direction, PROJECTILE_STEPS)
                straight_path = _straight_path(apex, direction, PROJECTILE_STEPS)
                if not _in_bounds_cells(path):
                    continue
                cell_set = frozenset(cells)
                trail_mid = path[:-1]
                trail_mid_set = frozenset(trail_mid)
                dest = path[-1]
                if dest in cell_set:
                    continue
                if trail_mid_set & cell_set:
                    continue
                placement = {
                    "apex": apex,
                    "dir": direction,
                    "dest": dest,
                    "path": path,
                    "straight_path": straight_path,
                    "cells": cells,
                    "cell_set": cell_set,
                    "halo": _halo(cells),
                    "trail_mid": trail_mid,
                    "trail_mid_set": trail_mid_set,
                    "trail_set": frozenset(path),
                    "changed_dest": dest != straight_path[-1],
                    "bent": _is_bent(direction),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _sprite_fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _trail_clear(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["trail_mid_set"] & occupied) and not (placement["trail_mid_set"] & reserved_bg)


def _trail_fits_bg(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if not _trail_clear(placement, occupied, reserved_bg):
        return False
    return placement["dest"] not in occupied and placement["dest"] not in reserved_bg


def _trail_fits_shared_dest(
    placement: dict,
    dest: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if placement["dest"] != dest:
        return False
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    return _trail_clear(placement, occupied, reserved_bg)


def _trail_fits_hit(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    return _trail_clear(placement, occupied, reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _reserve_sprite(sprite: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> None:
    placement = sprite["placement"]
    occupied.update(placement["cells"])
    reserved_bg.update(placement["trail_mid"])
    if sprite["target_kind"] == "bg":
        reserved_bg.add(sprite["target"])


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied or dest in reserved_bg:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _trail_fits_bg(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            res2 = reserved_bg | set(first["trail_mid"]) | {dest}
            for second in candidates[i + 1 :]:
                if _trail_fits_shared_dest(second, dest, occ2, res2):
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
        if _trail_fits_bg(placement, occupied, reserved_bg):
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
        candidates = [
            p
            for p in PLACEMENTS_BY_DEST[cell]
            if cell not in p["cell_set"] and _trail_fits_hit(p, occupied, reserved_bg)
        ]
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


def _counts_match_mode(mode: str | None, e_hit: int, e_coll: int, unique: int) -> bool:
    if mode is None:
        return True
    if mode == "hit_only":
        return e_hit >= 1 and e_coll == 0 and unique >= 1
    if mode == "collision_only":
        return e_hit == 0 and e_coll >= 1 and unique >= 1
    if mode == "hit_collision":
        return e_hit >= 1 and e_coll >= 1
    if mode == "full_mix":
        return e_hit >= 1 and e_coll >= 1 and unique >= 1
    return False


def _counts_for_n(rng: random.Random, n_sprites: int, mode: str | None = None) -> tuple[int, int, int] | None:
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
            if not _counts_match_mode(mode, e_hit, e_coll, unique):
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        for placement in pair:
            sprite = {"placement": placement, "target": dest, "target_kind": "bg"}
            sprites.append(sprite)
            _reserve_sprite(sprite, occupied, reserved_bg)
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprite = {"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}

    for sprite in sprites:
        for r, c in sprite["placement"]["trail_mid"]:
            out[r, c] = TRAIL_COLOR

    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)

    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR

    return out


def _variant_metrics(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> dict[str, int]:
    old_out = _main_original_straight(inp)
    arc_no_trail = _main_arc_no_trail(inp)
    straight_trail = _main_straight_trail(inp)
    return {
        "bent_shooters": sum(1 for s in sprites if s["placement"]["bent"]),
        "changed_dest_shooters": sum(1 for s in sprites if s["placement"]["changed_dest"]),
        "trail_gain_vs_arc_no_trail": int(np.count_nonzero((out != arc_no_trail) & (inp == BG))),
        "trail_gain_vs_original": int(np.count_nonzero((out != old_out) & (inp == BG))),
        "path_gain_vs_straight_trail": int(np.count_nonzero((out != straight_trail) & (inp == BG))),
    }


def _gen_attempt(mode: str | None = None) -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites, mode)
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

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    old_out = _main_original_straight(inp)
    arc_no_trail = _main_arc_no_trail(inp)
    straight_trail = _main_straight_trail(inp)
    metrics = _variant_metrics(inp, out, sprites)

    if metrics["bent_shooters"] < 1:
        return None
    if metrics["changed_dest_shooters"] < 1:
        return None
    if metrics["trail_gain_vs_arc_no_trail"] < 4:
        return None
    if metrics["path_gain_vs_straight_trail"] < 2:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, old_out) or np.array_equal(out, arc_no_trail) or np.array_equal(out, straight_trail):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT
    mode = SCENARIO_CYCLE[_GENERATE_CALL_COUNT % len(SCENARIO_CYCLE)]
    _GENERATE_CALL_COUNT += 1
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    # Fall back to unconstrained sampling if the targeted branch mix is unusually hard
    # to realize under the current random placements.
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
