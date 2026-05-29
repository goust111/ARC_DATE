# Variant of 409aa875
# Dimension: D14 — 精灵间最小间距 (meta)
# Change: 任意两个精灵之间 Chebyshev 距离 >= 2（不可8-连通相邻）... → 允许精灵8-连通相邻（间距>=1）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior:
#    The visible objects stay the same: every sprite is still a 3-cell monochrome
#    triangle, either an axis-aligned L or a diagonal V, and each emits one shot
#    that lands 5 steps away from its apex. The variant changes only spacing:
#    two same-color sprites may now touch at an edge or a corner. Example:
#    the original solver treats two edge-touching L-shapes as one 6-cell blob,
#    so `_triangle_apex_and_dir` never runs on either sprite. In the variant we
#    must split that blob back into two legal 3-cell triangles before tracing
#    the shots. The input construction, solver parsing, and generator validation
#    are all affected by this one change.
# b) Complexity calibration:
#    Letting every touching arrangement through would create many ambiguous blobs.
#    To keep the puzzle inferable, the generator still uses a single sprite color
#    and the same 3-cell shape vocabulary, but it forces only a small number of
#    tight contacts inside otherwise sparse scenes. The solver therefore needs a
#    local decomposition mechanism, not a new visual language.
# c) Hidden assumptions in the original `main()`:
#    - `_cc8(g == col)` assumed one 8-connected same-color component = one sprite.
#      This fails immediately once touching is allowed.
#    - `if len(comp) != 3: return None` inside `_triangle_apex_and_dir` was safe
#      only because separation guaranteed every component had size 3.
#    - `cell_to_comp` depended on the component list already being the sprite list.
#      In the variant we must build sprite IDs only after decomposition.
# d) Placement legality:
#    The generator now forbids only cell overlap, not Chebyshev-1 contact. It
#    still keeps every landing cell in bounds, prevents bg landing markers from
#    later being occupied by sprites, and validates that each same-color cluster
#    has exactly one exact-cover decomposition into legal 3-cell triangles.
# e) Visual observability:
#    Every generated grid contains at least one pair of sprites with minimum
#    Chebyshev distance exactly 1, so some examples visibly contain touching
#    triangles. The output rule is unchanged, which isolates the spacing change.
# f) Visual cleanliness:
#    Tight pairs are required, but full overlap is still impossible and dense
#    tangles are rejected by the unique-decomposition check. This keeps blobs
#    readable rather than collapsing into noisy masses.
# g) Color inference:
#    No new color mapping is introduced. The original semantics remain: 9 marks
#    sprite hits or single empty-cell impacts, and 1 marks multi-shot collisions.


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 400


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
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        if abs(dr) + abs(dc) != 2:
            return None
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


def _touch_kind(cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]) -> str | None:
    kind = None
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            if max(dr, dc) != 1:
                continue
            if dr + dc == 1:
                return "edge"
            kind = "diag"
    return kind


def _placement_sort_key(placement: dict) -> tuple:
    return placement["apex"], placement["dir"], placement["cells"]


def _make_placement(apex: tuple[int, int], direction: tuple[int, int], pid: int) -> dict | None:
    dest = (apex[0] + PROJECTILE_DIST * direction[0], apex[1] + PROJECTILE_DIST * direction[1])
    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
        return None
    cells = tuple(sorted(_sprite_cells(apex, direction)))
    if not _in_bounds_cells(cells):
        return None
    ad = _triangle_apex_and_dir(list(cells))
    if ad != (apex, direction):
        return None
    return {
        "id": pid,
        "apex": apex,
        "dir": direction,
        "dest": dest,
        "cells": cells,
        "cell_set": frozenset(cells),
    }


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[int, list[tuple[int, str]]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pid = 0
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                placement = _make_placement((ar, ac), direction, pid)
                if placement is None:
                    continue
                placements.append(placement)
                by_dest[placement["dest"]].append(placement)
                pid += 1

    touching: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i, first in enumerate(placements):
        for second in placements[i + 1 :]:
            if first["cell_set"] & second["cell_set"]:
                continue
            kind = _touch_kind(first["cells"], second["cells"])
            if kind is None:
                continue
            touching[first["id"]].append((second["id"], kind))
            touching[second["id"]].append((first["id"], kind))
    return placements, by_dest, touching


PLACEMENTS, PLACEMENTS_BY_DEST, TOUCHING_BY_ID = _build_placements()
PLACEMENT_BY_ID = {placement["id"]: placement for placement in PLACEMENTS}


def _cluster_candidates(cells: list[tuple[int, int]]) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    cell_set = frozenset(cells)
    candidates = [p for p in PLACEMENTS if p["cell_set"] <= cell_set]
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for placement in candidates:
        for cell in placement["cells"]:
            by_cell[cell].append(placement)
    for cell in by_cell:
        by_cell[cell].sort(key=_placement_sort_key)
    return candidates, by_cell


def _solve_cluster_cover(cells: list[tuple[int, int]], limit: int = 1) -> tuple[int, list[dict] | None]:
    if len(cells) % 3 != 0:
        return 0, None

    remaining0 = frozenset(cells)
    _candidates, by_cell = _cluster_candidates(cells)
    for cell in remaining0:
        if cell not in by_cell:
            return 0, None

    count = 0
    first_solution: list[dict] | None = None
    stack: list[dict] = []

    def dfs(remaining: frozenset[tuple[int, int]]) -> None:
        nonlocal count, first_solution
        if count >= limit:
            return
        if not remaining:
            count += 1
            if first_solution is None:
                first_solution = stack.copy()
            return

        pivot = None
        pivot_options = None
        best_len = None
        for cell in sorted(remaining):
            options = [placement for placement in by_cell[cell] if placement["cell_set"] <= remaining]
            if not options:
                return
            option_len = len(options)
            if best_len is None or option_len < best_len:
                best_len = option_len
                pivot = cell
                pivot_options = options
                if best_len == 1:
                    break

        assert pivot is not None and pivot_options is not None
        for placement in pivot_options:
            stack.append(placement)
            dfs(remaining - placement["cell_set"])
            stack.pop()
            if count >= limit:
                return

    dfs(remaining0)
    return count, first_solution


def _extract_sprites(grid: np.ndarray, require_unique: bool = False) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]] | None:
    bg = _bg_color(grid)
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}

    for color in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for cluster in components_mask(grid == color, connectivity=8):
            cells = [(int(r), int(c)) for r, c in cluster]
            count, solution = _solve_cluster_cover(cells, limit=2 if require_unique else 1)
            if solution is None or count == 0:
                return None
            if require_unique and count != 1:
                return None
            assert solution is not None
            for placement in sorted(solution, key=_placement_sort_key):
                cid = len(comps)
                comp = list(placement["cells"])
                comps.append(comp)
                for rc in comp:
                    cell_to_comp[rc] = cid

    return comps, cell_to_comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split touching 3-pixel triangle sprites, then trace their 5-step projectiles.

    Input structure:
    - The background is the most common color. Every non-background pixel belongs to a 3-cell monochrome triangle sprite, but same-color sprites may now touch by an edge or a corner and form larger blobs.

    Rule:
    - First decompose each same-color blob into legal L-shapes or diagonal V-shapes.
    - Each triangle fires from its apex: L-shapes shoot diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their opening.
    - The projectile lands exactly 5 cells from the apex. Landing on a sprite recolors that whole sprite to maroon (9); multiple projectiles on the same empty background cell paint it blue (1); a single projectile on empty background paints it maroon (9).

    Key insight:
    - The firing rule is unchanged; the only new challenge is that touching sprites must be disentangled into separate 3-cell triangles before any shot can be followed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    extracted = _extract_sprites(g, require_unique=False)
    if extracted is None:
        raise ValueError("Grid cannot be decomposed into legal triangle sprites")
    comps, cell_to_comp = extracted

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["cell_set"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_touching_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[list[dict], str] | None:
    candidate_ids = [placement["id"] for placement in PLACEMENTS]
    rng.shuffle(candidate_ids)
    for pid in candidate_ids:
        first = PLACEMENT_BY_ID[pid]
        if first["dest"] in reserved_bg or first["dest"] in occupied:
            continue
        if not _fits(first, occupied, reserved_bg):
            continue
        partners = TOUCHING_BY_ID.get(pid, [])[:]
        rng.shuffle(partners)
        occ2 = occupied | set(first["cells"])
        for qid, kind in partners:
            second = PLACEMENT_BY_ID[qid]
            if second["dest"] in reserved_bg or second["dest"] in occ2:
                continue
            if first["dest"] == second["dest"]:
                continue
            if first["dest"] in second["cell_set"]:
                continue
            if not _fits(second, occ2, reserved_bg):
                continue
            return [first, second], kind
    return None


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
                if second["dest"] != dest:
                    continue
                if second["cell_set"] & reserved_bg:
                    continue
                if second["cell_set"] & occ2:
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


def _count_touching_pairs(placements: list[dict]) -> tuple[int, Counter]:
    count = 0
    kinds: Counter = Counter()
    for i, first in enumerate(placements):
        for second in placements[i + 1 :]:
            kind = _touch_kind(first["cells"], second["cells"])
            if kind is None:
                continue
            count += 1
            kinds[kind] += 1
    return count, kinds


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    extracted = _extract_sprites(grid, require_unique=True)
    if extracted is None:
        return False
    comps, _cell_to_comp = extracted
    if len(comps) != n_sprites:
        return False
    touches, _kinds = _count_touching_pairs([{"cells": tuple(sorted(comp))} for comp in comps])
    return touches >= 1


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 2:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
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


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
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


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
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
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    touching_choice = _choose_touching_pair(rng, occupied, reserved_bg)
    if touching_choice is None:
        return None
    touching_pair, _touch_kind_name = touching_choice
    for placement in touching_pair:
        if placement["dest"] in occupied:
            return None
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    remaining_unique = unique_total - 2
    if base_needed > remaining_unique:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg):
        return None
    remaining_unique -= base_needed

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining != remaining_unique:
        return None
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
