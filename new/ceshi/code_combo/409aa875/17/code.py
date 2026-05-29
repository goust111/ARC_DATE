# Combo variant of 409aa875 (index 17)
# combo_spec: D5/0+D16/2
# Name: 碰撞爆炸扩散+臂端双发射

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for combo D5/0 + D16/2
# a) Concrete behavior and trace:
#    - The input remains a 16x16 grid with bg=7 and monochrome 3-pixel triangle sprites.
#      The two valid shapes are unchanged: axis-aligned L and diagonal V.
#    - Geometry still determines one direction per sprite, but the firing origin flips from
#      the apex to the two arm endpoints. Each sprite therefore contributes two landings.
#    - The D5 part changes only the empty-bg multi-hit branch: if >=2 projectiles land on
#      one empty cell, the visible result is a 5-cell blue cross centered there.
#    - Example: an L sprite at {(5,5),(6,5),(5,6)} points to (-1,-1). The original task
#      used only apex (5,5) and landed at (0,0). This combo instead fires from (6,5) and
#      (5,6), so the two landings become (1,0) and (0,1). If another sprite also sends a
#      projectile to (1,0), the old D16-only variant would show a blue dot at (1,0); this
#      combo shows a full blue cross on {(1,0),(0,0),(2,0),(1,1)} plus the in-bounds arm.
# b) Complexity calibration:
#    - The two changes reinforce each other rather than acting independently: dual firing
#      raises collision frequency, and the explosion cross makes those extra collisions
#      visually obvious. To keep that visible in every sample, generation requires at least
#      two collision centers, leaves the other arm-based landings as maroon singleton cues,
#      and includes one clean sprite-hit event so every solver branch is observable.
#    - The rule still depends on a small set of stable categories: L vs V, 8 directions,
#      two arm launch points, and the binary empty-bg count split (1 vs >=2).
# c) Hidden assumptions from the original solver:
#    - The original `dest_to_shooters` loop assumed one destination per component. That is
#      invalid here; each component must contribute two arm-based destinations.
#    - The original empty-bg collision branch recolored only the center cell. With D5/0, the
#      adjacent cardinal cells also become semantically meaningful and must be painted too.
#    - The original generator could reserve only landing centers. In this combo, a collision
#      consumes a full cross footprint, so later sprites and later markers must avoid all
#      five cells, not just the center.
# d) Placement legality:
#    - Every precomputed placement requires both sprite cells and both arm-based landing
#      cells to stay in bounds.
#    - Sprites still obey Chebyshev distance >= 2 so connected-component parsing stays stable.
#    - For collision events, the generator reserves the entire blue-cross footprint plus the
#      non-colliding secondary arm landings from the same shooters. This prevents overlaps
#      between sprites, blue crosses, and maroon singleton marks.
# e) Visual observability:
#    - A human can infer the combo from a few samples because two regularities repeat: each
#      triangle causes two shifted impacts, and repeated impacts on empty bg create a cross
#      instead of a point. Samples contain multiple blue crosses, multiple maroon singleton
#      landings, and one recolored hit sprite, so all branches are visible together.
# f) Visual cleanliness:
#    - Output markers are forced to be disjoint: blue cross footprints do not overlap sprites,
#      other crosses, or singleton maroon cells. This keeps the scene legible despite the
#      doubled projectile count.
# g) Color inferability:
#    - Color semantics stay minimal and structural: blue marks only empty-bg multiplicity,
#      maroon marks lone bg impacts. No hidden color ordering or arithmetic is used.
# h) New solver assumptions audited against generation:
#    - `main()` now assumes any valid sprite yields exactly two arm endpoints; `_triangle_info`
#      enforces that for both L and V shapes.
#    - `main()` allows collision crosses to clip at borders for robustness, but the generator
#      always chooses full in-bounds crosses so training samples show the intended shape.
#    - `main()` treats hits on sprites and bg landings independently from projectile order.
#      The generator respects that by keeping all planned bg events on genuinely empty cells.


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (6, 9)
MIN_COLLISION_EVENTS = 2
MAX_COLLISION_EVENTS = 3
HIT_EVENT_COUNT = 1
MAX_ATTEMPTS = 600


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_info(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None:
    """Return (apex, direction, arm_pixels) for a valid 3-pixel L or V sprite."""
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4: dict[tuple[int, int], int] = {}
    deg8: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if len(corners) == 1:
        apex = corners[0]
        ar, ac = apex
        arms: list[tuple[int, int]] = []
        for r, c in comp:
            if (r, c) == apex:
                continue
            if abs(r - ar) + abs(c - ac) != 1:
                return None
            arms.append((r, c))
        if len(arms) != 2:
            return None
        dr_sum = sum(r - ar for r, _ in arms)
        dc_sum = sum(c - ac for _, c in arms)
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        return apex, (dr, dc), tuple(sorted(arms))

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    arms = [p for p in comp if p != apex]
    if len(arms) != 2:
        return None
    deltas = [(r - ar, c - ac) for r, c in arms]
    if not all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in deltas):
        return None
    dr = -((deltas[0][0] + deltas[1][0]) // 2)
    dc = -((deltas[0][1] + deltas[1][1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc), tuple(sorted(arms))


def _collision_cross(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _paint_collision_cross(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = out.shape
    for r, c in _collision_cross(center):
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color


def _collect_components(g: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _render_rule(grid: np.ndarray, arm_based: bool, explosive_collision: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps, cell_to_comp = _collect_components(g, bg)

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        info = _triangle_info(comp)
        if info is None:
            continue
        (ar, ac), (dr, dc), arms = info
        sources = arms if arm_based else ((ar, ac),)
        for sr, sc in sources:
            rr = sr + PROJECTILE_DIST * dr
            cc = sc + PROJECTILE_DIST * dc
            if 0 <= rr < h and 0 <= cc < w:
                dest_to_shooters[(rr, cc)].append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        if len(shooters) >= 2 and explosive_collision:
            _paint_collision_cross(out, dest, COLLISION_COLOR)
        elif len(shooters) >= 2:
            r, c = dest
            out[r, c] = COLLISION_COLOR
        else:
            r, c = dest
            out[r, c] = HIT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve dual arm-fired projectiles from 3-pixel triangle sprites and paint explosive empty-bg collisions.

    Input structure:
    - The background is the dominant color, and every foreground component is a monochrome 3-pixel triangle sprite: either an axis-aligned L or a diagonal V.

    Rule:
    - Each sprite still determines one firing direction from its geometry, but it fires from the two non-apex arm endpoints rather than from the apex.
    - Each arm projectile lands exactly 5 cells away in that shared direction.
    - If a projectile lands on a sprite, the entire struck sprite is recolored maroon (9).
    - If multiple projectiles land on the same empty background cell, that collision becomes a blue (1) cross centered there; a lone empty-bg landing becomes a single maroon (9) cell.

    Key insight:
    - The orientation of each tiny triangle still reveals the direction, but the fused variant must be read as "two shifted shots per sprite" plus "repeated empty-bg landings expand into a blue cross", so the output contains denser and more structured impact patterns than the original task.
    """
    return _render_rule(grid, arm_based=True, explosive_collision=True)


def _d5_only_main(grid: np.ndarray) -> np.ndarray:
    return _render_rule(grid, arm_based=False, explosive_collision=True)


def _d16_only_main(grid: np.ndarray) -> np.ndarray:
    return _render_rule(grid, arm_based=True, explosive_collision=False)


def _original_main(grid: np.ndarray) -> np.ndarray:
    return _render_rule(grid, arm_based=False, explosive_collision=False)


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


def _cross_is_full_in_bounds(center: tuple[int, int]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in _collision_cross(center))


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_any_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                arms = (cells[1], cells[2])
                dests = tuple((r + PROJECTILE_DIST * dr, c + PROJECTILE_DIST * dc) for r, c in arms)
                if not _in_bounds_cells(dests):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "arms": arms,
                    "dests": dests,
                    "legacy_dest": (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc),
                }
                placements.append(placement)
                for dest in dests:
                    by_any_dest[dest].append(placement)
    return placements, by_any_dest


PLACEMENTS, PLACEMENTS_BY_ANY_DEST = _build_placements()
COLLISION_CENTERS = [dest for dest in PLACEMENTS_BY_ANY_DEST if _cross_is_full_in_bounds(dest)]


def _fits_sprite(placement: dict, occupied: set[tuple[int, int]], reserved_marks: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_marks)


def _dests_are_fresh(
    dests: tuple[tuple[int, int], ...],
    sprite_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> bool:
    return len(set(dests)) == len(dests) and not (
        set(dests) & sprite_cells or set(dests) & occupied or set(dests) & reserved_marks
    )


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict], tuple[tuple[int, int], ...]] | None:
    centers = COLLISION_CENTERS[:]
    rng.shuffle(centers)
    for center in centers:
        footprint = set(_collision_cross(center))
        if footprint & occupied or footprint & reserved_marks:
            continue
        candidates = [p for p in PLACEMENTS_BY_ANY_DEST[center] if _fits_sprite(p, occupied, reserved_marks)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ1 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                combined_cells = set(first["cells"]) | set(second["cells"])
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ1:
                    continue
                if center in combined_cells:
                    continue
                other_dests: list[tuple[int, int]] = []
                ok = True
                for placement in (first, second):
                    if center not in placement["dests"]:
                        ok = False
                        break
                    for dest in placement["dests"]:
                        if dest == center:
                            continue
                        if dest in footprint or dest in occupied or dest in reserved_marks or dest in combined_cells:
                            ok = False
                            break
                        other_dests.append(dest)
                    if not ok:
                        break
                if not ok or len(set(other_dests)) != len(other_dests):
                    continue
                return center, [first, second], tuple(other_dests)
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if not _fits_sprite(placement, occupied, reserved_marks):
            continue
        if _dests_are_fresh(placement["dests"], set(placement["cells"]), occupied, reserved_marks):
            return placement
    return None


def _choose_hit_event(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> tuple[dict, dict, tuple[int, int]] | None:
    targets = PLACEMENTS[:]
    rng.shuffle(targets)
    for target in targets:
        if not _fits_sprite(target, occupied, reserved_marks):
            continue
        target_cells = set(target["cells"])
        occ1 = occupied | target_cells

        shooters = PLACEMENTS[:]
        rng.shuffle(shooters)
        for shooter in shooters:
            if not _fits_sprite(shooter, occ1, reserved_marks):
                continue
            shooter_cells = set(shooter["cells"])
            if shooter_cells & target_cells:
                continue

            hit_dests = [dest for dest in shooter["dests"] if dest in target_cells]
            if len(hit_dests) != 1:
                continue

            other_dests = [dest for dest in shooter["dests"] if dest not in target_cells]
            if len(other_dests) != 1:
                continue
            other_dest = other_dests[0]
            if other_dest in occupied or other_dest in reserved_marks or other_dest in target_cells or other_dest in shooter_cells:
                continue
            return target, shooter, other_dest
    return None


def _event_data(placements: list[dict]) -> tuple[dict[tuple[int, int], int], set[int], Counter]:
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_sprite[cell] = idx

    hit_sprites: set[int] = set()
    bg_counts: Counter = Counter()
    for placement in placements:
        for dest in placement["dests"]:
            if dest in cell_to_sprite:
                hit_sprites.add(cell_to_sprite[dest])
            else:
                bg_counts[dest] += 1
    return cell_to_sprite, hit_sprites, bg_counts


def _render_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    _cell_to_sprite, hit_sprites, bg_counts = _event_data(placements)
    for idx in hit_sprites:
        _paint_sprite(out, placements[idx], HIT_COLOR)
    for dest, count in bg_counts.items():
        if count >= 2:
            _paint_collision_cross(out, dest, COLLISION_COLOR)
        else:
            out[dest] = HIT_COLOR
    return out


def _output_stats(inp: np.ndarray, placements: list[dict], out: np.ndarray) -> dict[str, int | bool]:
    cell_to_sprite, hit_sprites, bg_counts = _event_data(placements)
    collision_centers = [dest for dest, count in bg_counts.items() if count >= 2]
    lone_bg = sum(1 for count in bg_counts.values() if count == 1)

    marker_cells: set[tuple[int, int]] = set()
    clean_markers = True
    for dest, count in bg_counts.items():
        footprint = set(_collision_cross(dest)) if count >= 2 else {dest}
        if footprint & set(cell_to_sprite):
            clean_markers = False
        if marker_cells & footprint:
            clean_markers = False
        marker_cells.update(footprint)

    return {
        "projectiles": 2 * len(placements),
        "hit_sprites": len(hit_sprites),
        "collision_centers": len(collision_centers),
        "blue_pixels": int(np.sum(out == COLLISION_COLOR)),
        "lone_bg": lone_bg,
        "marker_cells": len(marker_cells),
        "clean_markers": clean_markers,
        "d5_diff_cells": int(np.count_nonzero(out != _d5_only_main(inp))),
        "d16_diff_cells": int(np.count_nonzero(out != _d16_only_main(inp))),
        "original_diff_cells": int(np.count_nonzero(out != _original_main(inp))),
    }


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_coll in range(MIN_COLLISION_EVENTS, min(MAX_COLLISION_EVENTS, n_sprites // 2) + 1):
        unique = n_sprites - 2 * e_coll - 2 * HIT_EVENT_COUNT
        if unique < 0:
            continue
        options.append((e_coll, HIT_EVENT_COUNT, unique))
    return rng.choice(options) if options else None


def _place_collision_pairs(
    rng: random.Random,
    e_coll: int,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_marks)
        if choice is None:
            return False
        center, pair, other_dests = choice
        reserved_marks.update(_collision_cross(center))
        reserved_marks.update(other_dests)
        for placement in pair:
            placements.append(placement)
            occupied.update(placement["cells"])
    return True


def _place_hit_events(
    rng: random.Random,
    e_hit: int,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> bool:
    for _ in range(e_hit):
        choice = _choose_hit_event(rng, occupied, reserved_marks)
        if choice is None:
            return False
        target, shooter, other_dest = choice
        placements.extend([target, shooter])
        occupied.update(target["cells"])
        occupied.update(shooter["cells"])
        reserved_marks.add(other_dest)
    return True


def _place_unique_landings(
    rng: random.Random,
    n: int,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_marks: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_marks)
        if placement is None:
            return False
        placements.append(placement)
        occupied.update(placement["cells"])
        reserved_marks.update(placement["dests"])
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_coll, e_hit, unique_count = counts

    placements: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_marks: set[tuple[int, int]] = set()

    if not _place_collision_pairs(rng, e_coll, placements, occupied, reserved_marks):
        return None
    if not _place_hit_events(rng, e_hit, placements, occupied, reserved_marks):
        return None
    if not _place_unique_landings(rng, unique_count, placements, occupied, reserved_marks):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for placement in placements:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, placements)
    stats = _output_stats(inp, placements, out)

    if stats["collision_centers"] < MIN_COLLISION_EVENTS:
        return None
    if stats["blue_pixels"] < 5 * MIN_COLLISION_EVENTS:
        return None
    if stats["hit_sprites"] < HIT_EVENT_COUNT:
        return None
    if stats["lone_bg"] < 2:
        return None
    if not stats["clean_markers"]:
        return None
    if stats["d5_diff_cells"] < 2:
        return None
    if stats["d16_diff_cells"] < 4:
        return None
    if stats["original_diff_cells"] < 4:
        return None
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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
