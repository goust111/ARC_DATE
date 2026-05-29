# Variant of 409aa875
# Dimension: D13 — 障碍物/干扰项 (obstacle)
# Change: 无障碍物或干扰项；网格中只有精灵和背景... → 干扰精灵（形状相似但不发射）
#
# Design analysis
# ===============
# spec:
#   variant_focus:
#     - Keep the original "triangle emits a projectile 5 cells away" mechanic.
#     - Add same-color decoy components that look like underfilled or overfilled versions of
#       the real triangles.
#     - Only exact 3-pixel L/V components are emitters. The decoys are inert.
#   concrete_objects:
#     - Real emitters remain the original 3-pixel monochrome L and V components.
#     - Decoys use the same non-background color and stay 8-connected, but have size 2 or 4.
#     - 2-pixel decoys are subsets of a real triangle: one arm of an L, or apex+one leg of a V.
#     - 4-pixel decoys are supersets of a real triangle: a filled 2x2 square for the L family,
#       or a V with its middle cell filled in for the V family.
#   behavior_trace:
#     - Original flow on a component: collect component -> infer apex+direction -> fire.
#     - Variant flow on a component: collect component -> first test "len(component) == 3" ->
#       only then infer apex+direction -> fire.
#     - Example: a 4-pixel 2x2 square at rows 4-5, cols 7-8 resembles an L with one extra cell.
#       In the original dataset it never appeared, so the solver could get away with assuming
#       every non-bg component was a candidate sprite. In the variant it must reject this square
#       before any direction logic; otherwise a mistaken heuristic would invent a projectile.
#   complexity_calibration:
#     - Decoys share the emitters' color so color cannot be used as a shortcut.
#     - Only two decoy sizes are used (2 and 4), which is enough to make the change visible
#       without introducing too many categories.
#     - Each generated grid contains at least two decoys; when possible, one is size 2 and one
#       is size 4 so the size filter is reinforced across a single example.
#   implicit_assumptions_review:
#     - Component extraction by color still works because decoys are monochrome components too.
#     - The old "iterate every component and call _triangle_apex_and_dir" was accidentally safe
#       because _triangle_apex_and_dir returned None for non-3-pixel shapes. The new solver makes
#       this filtering explicit so the task logic visibly matches the variant statement.
#     - Hit detection must still index all components, not just emitters, because a projectile
#       can land on a decoy and should recolor that whole decoy component like any other object.
#     - Empty-cell collision coloring still depends only on emitters; decoys must never contribute
#       extra shots.
#   placement_legality:
#     - Emitters and decoys both obey Chebyshev separation >= 2, so same-color components never
#       merge under 8-connectivity.
#     - Every emitter keeps its original in-bounds landing cell constraint.
#     - Decoys are placed after emitter events are fixed, and may not occupy any reserved empty
#       landing cell. This guarantees the planned output remains consistent with the solver.
#   visual_observability:
#     - All non-bg objects share one color, so the human must rely on exact pixel count/shape.
#     - Inputs contain several near-miss shapes; outputs add marks only from exact 3-pixel ones,
#       making the inert decoys visible as unchanged same-color clutter.
#   visual_cleanliness:
#     - Decoy count is capped at 2-3 and the original halo rule is reused, preventing crowded
#       clusters that would hide component boundaries.
#   color_change_inference:
#     - Base object color is chosen from colors excluding 1 and 9, so blue collision cells and
#       maroon hit marks remain visually attributable to the projectile rule rather than palette
#       collisions.
#   post_change_solver_assumptions:
#     - Every emitting component is exactly one of the canonical 3-pixel L/V patterns.
#     - No reserved empty landing cell is occupied by a decoy.
#     - generate_input enforces both assumptions before validating with main().

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


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
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
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
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles only from exact 3-pixel triangles; same-color lookalikes stay inert.

    Input structure:
    - The background is the most frequent color.
    - All non-background components use one shared foreground color.
    - Some components are exact 3-pixel L/V triangles, while extra 2-pixel or 4-pixel near-miss
      shapes of the same color act as decoys.

    Rule:
    - Only components with exactly 3 pixels are tested for the triangle orientation rule.
    - An L emits diagonally away from its missing 2x2 corner; a V emits along the axis away from
      its open side, and the landing cell is always apex + 5 * direction.
    - If a landing cell lies on any non-background component, recolor that entire hit component
      to maroon (9), even if the hit component is a decoy.
    - If a landing cell is empty background, paint it maroon (9), except when multiple emitters
      share the same empty landing cell, in which case that cell becomes blue (1).

    Key insight:
    - Orientation matters only after a strict size filter: exact 3-pixel components shoot, while
      smaller and larger same-color lookalikes can be hit but never fire.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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
            if len(comp) != 3:
                continue
            apex_and_dir = _triangle_apex_and_dir(comp)
            if apex_and_dir is not None:
                apex, direction = apex_and_dir
                emitters.append((cid, apex, direction))

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, (ar, ac), (dr, dc) in emitters:
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

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
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

EMITTER_COUNT_RANGE = (4, 8)
DISTRACTOR_COUNT_RANGE = (2, 3)
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


def _distractor_variants(
    apex: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return [
            (2, ((ar, ac), (ar - dr, ac))),
            (2, ((ar, ac), (ar, ac - dc))),
            (4, ((ar, ac), (ar - dr, ac), (ar, ac - dc), (ar - dr, ac - dc))),
        ]
    if dr != 0:
        return [
            (2, ((ar, ac), (ar - dr, ac - 1))),
            (2, ((ar, ac), (ar - dr, ac + 1))),
            (4, ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac), (ar - dr, ac + 1))),
        ]
    return [
        (2, ((ar, ac), (ar - 1, ac - dc))),
        (2, ((ar, ac), (ar + 1, ac - dc))),
        (4, ((ar, ac), (ar - 1, ac - dc), (ar, ac - dc), (ar + 1, ac - dc))),
    ]


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


def _build_distractor_placements() -> dict[int, list[dict]]:
    seen: dict[frozenset[tuple[int, int]], dict] = {}
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                for size, cells in _distractor_variants((ar, ac), direction):
                    if not _in_bounds_cells(cells):
                        continue
                    cell_set = frozenset(cells)
                    if cell_set in seen:
                        continue
                    seen[cell_set] = {
                        "size": size,
                        "cells": cells,
                        "cell_set": cell_set,
                        "halo": _halo(cells),
                    }
    by_size: dict[int, list[dict]] = {2: [], 4: []}
    for placement in seen.values():
        by_size[placement["size"]].append(placement)
    return by_size


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
DISTRACTOR_PLACEMENTS = _build_distractor_placements()


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
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
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
    candidates = PLACEMENTS[:]
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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _choose_distractor(
    rng: random.Random,
    size: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = DISTRACTOR_PLACEMENTS[size][:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits(placement, occupied, reserved_bg):
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
    distractor_count = 0
    for comp in comps:
        if _triangle_apex_and_dir(comp) is not None:
            emitter_count += 1
        elif len(comp) in {2, 4}:
            distractor_count += 1
        else:
            return False
    return emitter_count == n_emitters and distractor_count == n_distractors


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


def _place_distractors(rng, distractor_sizes, occupied, reserved_bg) -> list[dict] | None:
    distractors: list[dict] = []
    for size in distractor_sizes:
        placement = _choose_distractor(rng, size, occupied, reserved_bg)
        if placement is None:
            return None
        distractors.append(placement)
        occupied.update(placement["cells"])
    return distractors


def _render_output(inp: np.ndarray, emitters: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in emitters if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in emitters if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_component(out, emitters[idx]["placement"]["cells"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _distractor_size_plan(rng: random.Random) -> list[int]:
    n_distractors = rng.randint(*DISTRACTOR_COUNT_RANGE)
    if n_distractors == 1:
        return [rng.choice([2, 4])]
    sizes = [2, 4]
    while len(sizes) < n_distractors:
        sizes.append(rng.choice([2, 4]))
    rng.shuffle(sizes)
    return sizes


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_emitters = rng.randint(*EMITTER_COUNT_RANGE)
    distractor_sizes = _distractor_size_plan(rng)
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

    distractors = _place_distractors(rng, distractor_sizes, occupied, reserved_bg)
    if distractors is None:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for emitter in emitters:
        _paint_component(inp, emitter["placement"]["cells"], sprite_color)
    for distractor in distractors:
        _paint_component(inp, distractor["cells"], sprite_color)

    if not _validate_input(inp, n_emitters, len(distractors)):
        return None

    out = _render_output(inp, emitters)
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
