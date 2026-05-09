# Combo variant of 409aa875 (index 16)
# combo_spec: D3/1+D13/3
# Name: 大小可变三角精灵+干扰精灵
#
# Design analysis
# ===============
# concrete_behavior:
#   - Keep the original eight headings and the original projectile distance of 5.
#   - Expand the visual vocabulary so the grid may contain triangle-family components
#     of size 3, 4, or 5, plus smaller 2-pixel near-miss pieces.
#   - Only exact 3-pixel canonical triangles emit. Larger 4/5-pixel triangles are
#     genuine members of the same L/V families, but they are inert; 2-pixel pieces
#     are also inert. Any component, emitter or decoy, can still be hit and recolored.
#   - Trace on one component family:
#       original: every visible triangle component is 3 pixels, so "recognize shape"
#       is enough to decide it emits.
#       fused: first recognize whether the component belongs to the triangle family,
#       then apply a strict size==3 gate before firing. A 5-pixel L can expose the
#       same diagonal heading as a 3-pixel L, but it must stay silent.
# complexity_calibration:
#   - The size palette is intentionally small: emitters are only size 3, large decoys
#     are only size 4 or 5, small decoys are only size 2.
#   - Every generated grid contains at least one 4/5-pixel triangle-family decoy and
#     at least one 2-pixel decoy, so the stricter size filter is visible within a
#     single sample instead of being implied only across a dataset.
#   - Large decoys are restricted to the same arm-length family as D3/1 (arm lengths
#     1 or 2), preventing ambiguous oversized shapes.
# implicit_assumptions_review:
#   - Original assumption "all non-bg components are emitters" is invalid; the solver
#     must index all components for hit detection but only let size-3 signatures shoot.
#   - D3/1's size-recognition helper is still useful, but in this combo it is used for
#     classification, not for variable projectile distance. Distance stays fixed at 5
#     because the fusion description changes the source filter, not the downstream hit
#     geometry.
#   - Hit logic must remain component-level for decoys too: a projectile may recolor a
#     2-pixel near-miss or a 4/5-pixel triangle-family decoy.
# placement_legality:
#   - All placed components obey a one-cell halo, so same-color objects never merge
#     under 8-connectivity.
#   - Every emitter placement guarantees its landing cell apex+5*dir is in bounds.
#   - Decoys are placed after emitter events are fixed and may not occupy reserved empty
#     landing cells, so the output-first event plan remains valid.
# visual_observability:
#   - All input objects share one foreground color, forcing the solver to use geometry
#     and pixel count rather than color identity.
#   - Large decoys are chosen from the same triangle family as emitters, so a naive
#     "all triangles fire" hypothesis is tempting; 2-pixel near-misses reinforce that
#     exact size, not vague resemblance, controls firing.
# visual_cleanliness:
#   - Distractor count stays small and halo separation is preserved, keeping component
#     boundaries readable.
#   - Marker colors remain structurally simple: maroon (9) for hits/lone landings and
#     blue (1) for shared empty-cell landings.
# fusion_observability_guard:
#   - The generator rejects any sample whose output would be unchanged under a relaxed
#     solver that incorrectly lets 4/5-pixel triangle-family decoys emit too. This
#     prevents degeneration back into a dataset where the new size filter is irrelevant.
# post_change_solver_assumptions:
#   - Exact emitters are canonical 3-pixel triangles from the L/V families.
#   - Larger triangle-family decoys, when present, still have unique apex+direction
#     signatures under the same family matcher.
#   - generate_input enforces both assumptions before accepting a sample.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _l_arm_vectors(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (-dr, 0), (0, -dc)


def _v_arm_vectors(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (-dr, -1), (-dr, 1)
    return (-1, -dc), (1, -dc)


def _ray_len(cells: set[tuple[int, int]], apex: tuple[int, int], step: tuple[int, int]) -> int:
    ar, ac = apex
    sr, sc = step
    if (ar + sr, ac + sc) not in cells:
        return 0
    if (ar + 2 * sr, ac + 2 * sc) in cells:
        return 2
    return 1


def _match_component(
    comp_set: set[tuple[int, int]],
    apex: tuple[int, int],
    step1: tuple[int, int],
    step2: tuple[int, int],
) -> tuple[int, int] | None:
    a = _ray_len(comp_set, apex, step1)
    b = _ray_len(comp_set, apex, step2)
    if a not in (1, 2) or b not in (1, 2):
        return None

    ar, ac = apex
    expected = {apex}
    for i in range(1, a + 1):
        expected.add((ar + i * step1[0], ac + i * step1[1]))
    for i in range(1, b + 1):
        expected.add((ar + i * step2[0], ac + i * step2[1]))
    if expected != comp_set:
        return None
    return a, b


def _triangle_signature(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    if len(comp) not in (3, 4, 5):
        return None

    comp_set = set(comp)
    matches: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for apex in comp:
        for direction in DIAG_DIRS:
            arm_lens = _match_component(comp_set, apex, *_l_arm_vectors(direction))
            if arm_lens is not None:
                size = 1 + arm_lens[0] + arm_lens[1]
                matches.append((apex, direction, size))
        for direction in AXIS_DIRS:
            arm_lens = _match_component(comp_set, apex, *_v_arm_vectors(direction))
            if arm_lens is not None:
                size = 1 + arm_lens[0] + arm_lens[1]
                matches.append((apex, direction, size))

    if len(matches) != 1:
        return None
    return matches[0]


def _apply_projectiles(
    g: np.ndarray,
    comps: list[list[tuple[int, int]]],
    cell_to_comp: dict[tuple[int, int], int],
    emitters: list[tuple[int, tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    h, w = g.shape
    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, (ar, ac), (dr, dc) in emitters:
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    out = g.copy()
    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _solve_relaxed_all_triangles(g: np.ndarray) -> np.ndarray:
    bg = _bg_color(g)
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    emitters: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
            sig = _triangle_signature(comp)
            if sig is not None:
                apex, direction, _size = sig
                emitters.append((cid, apex, direction))
    return _apply_projectiles(g, comps, cell_to_comp, emitters)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire only exact 3-pixel triangles; larger same-family triangles and smaller lookalikes stay inert.

    Input structure:
    - The background is the most common color.
    - All non-background components share one foreground color.
    - Components may be exact 3-pixel L/V triangles, larger 4/5-pixel triangles from the same
      L/V families, or 2-pixel near-miss fragments.

    Rule:
    - Recognize triangle-family geometry first, but only components with exactly 3 pixels emit.
    - A 3-pixel L fires diagonally away from its missing 2x2 corner; a 3-pixel V fires along
      the axis opposite its opening. Every valid emitter lands at apex + 5 * direction.
    - If a projectile lands on any component, recolor that whole hit component to maroon (9),
      even when the hit object is a non-emitting decoy.
    - If a projectile lands on empty background, paint that cell maroon (9), except when two or
      more emitters share the same empty landing cell, in which case it becomes blue (1).

    Key insight:
    - The hard part is not just reading direction; it is refusing to fire from components that
      look like the same triangle family but have the wrong pixel count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    emitters: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
            sig = _triangle_signature(comp)
            if sig is None:
                continue
            apex, direction, size = sig
            if size == 3:
                emitters.append((cid, apex, direction))

    return _apply_projectiles(g, comps, cell_to_comp, emitters)


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
LARGE_ARM_LENGTHS = [(1, 2), (2, 1), (2, 2)]

EMITTER_COUNT_RANGE = (4, 8)
DISTRACTOR_COUNT_RANGE = (2, 3)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 800


def _triangle_cells(
    apex: tuple[int, int],
    direction: tuple[int, int],
    arm_lens: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    steps = _l_arm_vectors(direction) if abs(direction[0]) + abs(direction[1]) == 2 else _v_arm_vectors(direction)
    cells = [apex]
    for step, length in zip(steps, arm_lens):
        for i in range(1, length + 1):
            cells.append((ar + i * step[0], ac + i * step[1]))
    return tuple(cells)


def _small_distractor_variants(
    apex: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[tuple[int, int], ...]]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return [((ar, ac), (ar - dr, ac)), ((ar, ac), (ar, ac - dc))]
    if dr != 0:
        return [((ar, ac), (ar - dr, ac - 1)), ((ar, ac), (ar - dr, ac + 1))]
    return [((ar, ac), (ar - 1, ac - dc)), ((ar, ac), (ar + 1, ac - dc))]


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_emitter_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * direction[0], ac + PROJECTILE_DIST * direction[1])
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _triangle_cells((ar, ac), direction, (1, 1))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": direction,
                    "size": 3,
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


def _build_large_distractor_placements() -> dict[int, list[dict]]:
    seen: dict[frozenset[tuple[int, int]], dict] = {}
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * direction[0], ac + PROJECTILE_DIST * direction[1])
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                for arm_lens in LARGE_ARM_LENGTHS:
                    cells = _triangle_cells((ar, ac), direction, arm_lens)
                    if not _in_bounds_cells(cells):
                        continue
                    sig = _triangle_signature(list(cells))
                    if sig is None or sig[2] not in (4, 5):
                        continue
                    cell_set = frozenset(cells)
                    if cell_set in seen:
                        continue
                    seen[cell_set] = {
                        "kind": "large",
                        "apex": (ar, ac),
                        "dir": direction,
                        "size": sig[2],
                        "dest": dest,
                        "cells": cells,
                        "cell_set": cell_set,
                        "halo": _halo(cells),
                    }
    by_size: dict[int, list[dict]] = {4: [], 5: []}
    for placement in seen.values():
        by_size[placement["size"]].append(placement)
    return by_size


def _build_small_distractor_placements() -> list[dict]:
    seen: dict[frozenset[tuple[int, int]], dict] = {}
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                for cells in _small_distractor_variants((ar, ac), direction):
                    if not _in_bounds_cells(cells):
                        continue
                    cell_set = frozenset(cells)
                    if cell_set in seen:
                        continue
                    seen[cell_set] = {
                        "kind": "small",
                        "size": 2,
                        "cells": cells,
                        "cell_set": cell_set,
                        "halo": _halo(cells),
                    }
    return list(seen.values())


EMITTER_PLACEMENTS, EMITTERS_BY_DEST = _build_emitter_placements()
LARGE_DISTRACTOR_PLACEMENTS = _build_large_distractor_placements()
SMALL_DISTRACTOR_PLACEMENTS = _build_small_distractor_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & blocked)


def _paint_component(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(EMITTERS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in EMITTERS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
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
) -> dict | None:
    candidates = EMITTER_PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_emitter: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_emitter["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in EMITTERS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _choose_large_distractor(
    rng: random.Random,
    size: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = LARGE_DISTRACTOR_PLACEMENTS[size][:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_small_distractor(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = SMALL_DISTRACTOR_PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_covering_distractor(
    rng: random.Random,
    kind: str,
    size: int,
    target_cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    if kind == "small":
        candidates = SMALL_DISTRACTOR_PLACEMENTS[:]
    else:
        candidates = LARGE_DISTRACTOR_PLACEMENTS[size][:]
    rng.shuffle(candidates)
    for placement in candidates:
        if target_cell not in placement["cell_set"]:
            continue
        if placement["halo"] & occupied:
            continue
        blocked_hits = placement["cell_set"] & reserved_bg
        if blocked_hits - {target_cell}:
            continue
        return placement
    return None


def _validate_input(grid: np.ndarray, n_emitters: int, n_distractors: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False

    non_bg_colors = set(int(v) for v in np.unique(grid)) - {BG}
    if len(non_bg_colors) != 1:
        return False

    comps = _cc8(grid != BG)
    if len(comps) != n_emitters + n_distractors:
        return False

    emitter_count = 0
    small_count = 0
    large_count = 0
    for comp in comps:
        sig = _triangle_signature(comp)
        if sig is not None and sig[2] == 3:
            emitter_count += 1
        elif sig is not None and sig[2] in (4, 5):
            large_count += 1
        elif len(comp) == 2:
            small_count += 1
        else:
            return False

    return emitter_count == n_emitters and emitter_count + large_count + small_count == len(comps) and small_count >= 1 and large_count >= 1


def _counts_for_n(rng: random.Random, n_emitters: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_emitters - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_emitters // 2) + 1):
            unique = n_emitters - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, emitters, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            emitters.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, emitters, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        emitters.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, emitters, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(emitters)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, emitters[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        emitters.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _distractor_plan(rng: random.Random) -> list[tuple[str, int]]:
    total = rng.randint(*DISTRACTOR_COUNT_RANGE)
    plan: list[tuple[str, int]] = [("small", 2), ("large", rng.choice([4, 5]))]
    while len(plan) < total:
        if rng.random() < 0.5:
            plan.append(("small", 2))
        else:
            plan.append(("large", rng.choice([4, 5])))
    rng.shuffle(plan)
    return plan


def _place_distractors(
    rng: random.Random,
    plan: list[tuple[str, int]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    cover_request: tuple[int, tuple[int, int]] | None = None,
) -> list[dict] | None:
    distractors: list[dict] = []
    for idx, (kind, size) in enumerate(plan):
        if cover_request is not None and idx == cover_request[0]:
            placement = _choose_covering_distractor(
                rng, kind, size, cover_request[1], occupied, reserved_bg
            )
        elif kind == "small":
            placement = _choose_small_distractor(rng, occupied, reserved_bg)
        else:
            placement = _choose_large_distractor(rng, size, occupied, reserved_bg)
        if placement is None:
            return None
        distractors.append(placement)
        occupied.update(placement["cells"])
    return distractors


def _component_stats(grid: np.ndarray) -> dict[str, int]:
    comps = _cc8(grid != BG)
    stats = {"emitters": 0, "large_decoys": 0, "small_decoys": 0}
    for comp in comps:
        sig = _triangle_signature(comp)
        if sig is not None and sig[2] == 3:
            stats["emitters"] += 1
        elif sig is not None and sig[2] in (4, 5):
            stats["large_decoys"] += 1
        elif len(comp) == 2:
            stats["small_decoys"] += 1
    return stats


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_emitters = rng.randint(*EMITTER_COUNT_RANGE)
    distractor_plan = _distractor_plan(rng)
    counts = _counts_for_n(rng, n_emitters)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    emitters: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, emitters, occupied, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(emitters))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, emitters, occupied, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, emitters, occupied, reserved_bg):
        return None

    remaining = n_emitters - len(emitters)
    if remaining < 0 or not _place_unique_landings(rng, remaining, emitters, occupied, reserved_bg):
        return None

    bg_target_counts = Counter(s["target"] for s in emitters if s["target_kind"] == "bg")
    unique_bg_targets = [dest for dest, count in bg_target_counts.items() if count == 1]
    cover_request = None
    if len(unique_bg_targets) >= 2 and rng.random() < 0.6:
        target_cell = rng.choice(unique_bg_targets)
        candidate_slots = [
            (idx, kind, size)
            for idx, (kind, size) in enumerate(distractor_plan)
            if kind == "small" or size in (4, 5)
        ]
        rng.shuffle(candidate_slots)
        for idx, kind, size in candidate_slots:
            if _choose_covering_distractor(rng, kind, size, target_cell, occupied, reserved_bg) is not None:
                cover_request = (idx, target_cell)
                break

    distractors = _place_distractors(rng, distractor_plan, occupied, reserved_bg, cover_request)
    if distractors is None:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for emitter in emitters:
        _paint_component(inp, emitter["placement"]["cells"], sprite_color)
    for distractor in distractors:
        _paint_component(inp, distractor["cells"], sprite_color)

    if not _validate_input(inp, n_emitters, len(distractors)):
        return None

    stats = _component_stats(inp)
    if stats["large_decoys"] < 1 or stats["small_decoys"] < 1:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(_solve_relaxed_all_triangles(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
