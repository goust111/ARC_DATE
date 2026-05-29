# Variant of 409aa875
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 单轮同时处理：所有精灵同时发射，落点同时计算碰撞结果... → L形优先，V形后发

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8

# Design analysis
# yaml:
#   a_concretization_and_trace:
#     input_view:
#       - Grid stays 16x16 with fixed background 7.
#       - Every foreground object is still a 3-cell monochrome triangle sprite:
#         axis-aligned L-shapes fire diagonally, diagonal V-shapes fire orthogonally.
#       - The visible D11 change is not in geometry; it is in when marks become
#         part of the board state. After all L-shots resolve, the board already
#         contains their 9/1 results before any V-shot is checked.
#     trace_example:
#       - Suppose one L-sprite and one V-sprite both target the same empty cell X.
#       - Original simultaneous rule: X receives two projectiles at once, so X=1.
#       - New phased rule: L phase sees X as empty and paints X=9. V phase then
#         evaluates against the updated board, so X is no longer empty background;
#         the V-shot does not create a blue collision there. Final X stays 9.
#       - This is the main generator-enforced witness that the task is no longer
#         explainable by a single simultaneous pass.
#   b_complexity_calibration:
#     - The rule change would be visually weak if samples only contained unique
#       landings, because simultaneous and phased outputs would match.
#     - To make D11 observable, every generated grid contains:
#       - at least one mixed L/V shared destination, whose final color differs
#         from the original simultaneous solver, and
#       - at least one same-phase shared destination (L+L or V+V), which still
#         becomes blue and shows that collisions remain possible inside a phase.
#   c_implicit_assumptions_from_original_main:
#     - Original code assumes every landing decision can be computed from the
#       untouched input grid. That is false here: V-shots must inspect the grid
#       after L-shots have already painted 9/1 cells.
#     - Original code collapses all shooters into one dest->shooters map. That
#       erases the phase boundary, so it must be replaced by two batch updates.
#     - Original hit detection only mattered for original sprites, because the
#       board had no intermediate marks. In the phased variant, intermediate
#       marks exist. The solver therefore recomputes current monochrome
#       components before each phase and treats any current non-bg component as
#       occupied for that phase's hit-vs-empty decision.
#   d_placement_legality:
#     - Sprite geometry, in-bounds landing, and Chebyshev>=2 sprite separation
#       from the original task are preserved.
#     - Generator additionally keeps distinct background landing cells separated
#       from each other and from sprite halos so that phase-created 9/1 markers
#       remain isolated single cells rather than merging into larger components.
#   e_visual_observability:
#     - Ordering never depends on hidden properties such as color id ordering.
#       It depends only on visible shape category: L before V.
#     - Sprite colors deliberately exclude 1 and 9 so output marks stand out and
#       the order-sensitive mixed destination is not camouflaged.
#   f_visual_cleanliness:
#     - Isolated marks avoid messy merged 9-components that would obscure which
#       pixel came from which landing event.
#     - Mandatory mixed and same-phase collision witnesses give the viewer both
#       contrast cases in a small number of samples.
#   g_color_inference:
#     - No new color mapping is introduced. 9 remains the hit/lone marker color
#       and 1 remains the same-phase multi-hit background color.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_dir_kind(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
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
        return corner, (dr, dc), "L"

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
    return apex, (dr, dc), "V"


def _current_components(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    sprites: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            info = _triangle_apex_dir_kind(comp)
            if info is None:
                continue
            apex, direction, kind = info
            ar, ac = apex
            dr, dc = direction
            dest = (ar + 5 * dr, ac + 5 * dc)
            if not (0 <= dest[0] < h and 0 <= dest[1] < w):
                continue
            sprites.append(
                {
                    "cells": tuple(comp),
                    "kind": kind,
                    "apex": apex,
                    "dir": direction,
                    "dest": dest,
                }
            )
    return sprites


def _apply_batches(grid: np.ndarray, batches: list[list[dict]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = _bg_color(out)
    for shooters in batches:
        if not shooters:
            continue
        comps, cell_to_comp = _current_components(out, bg)
        dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
        for sid, sprite in enumerate(shooters):
            dest_to_shooters[sprite["dest"]].append(sid)

        hit_components: set[int] = set()
        for dest in dest_to_shooters:
            if dest in cell_to_comp:
                hit_components.add(cell_to_comp[dest])

        for cid in hit_components:
            for r, c in comps[cid]:
                out[r, c] = 9

        for (r, c), batch_shooters in dest_to_shooters.items():
            if (r, c) in cell_to_comp:
                continue
            out[r, c] = 1 if len(batch_shooters) >= 2 else 9
    return out


def _solve_simultaneous(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g, _bg_color(g))
    return _apply_batches(g, [sprites])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites in two shape-ordered phases.

    Input structure:
    - bg = most common color. Foreground objects are monochrome 3-cell triangles of two visible kinds: axis-aligned L's and diagonal V's.

    Rule:
    - Every triangle still has the same apex and direction logic as the original task, and every projectile still lands exactly 5 cells from that apex.
    - However, all L-shapes resolve first as one simultaneous batch; only after their sprite hits and background marks are written does the V-shape batch resolve on the updated board.
    - In any phase, landing on an already non-background cell recolors that whole current component to maroon (9); landing on empty background paints maroon (9) for one shooter or blue (1) for 2+ shooters in that same phase.

    Key insight: The crucial pattern is not just where each triangle points, but that shape type determines time order: mixed L/V overlaps no longer behave like same-phase overlaps because V-shots see L-phase marks that were absent in the original simultaneous rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sprites = _extract_sprites(g, bg)
    l_batch = [sprite for sprite in sprites if sprite["kind"] == "L"]
    v_batch = [sprite for sprite in sprites if sprite["kind"] == "V"]
    return _apply_batches(g, [l_batch, v_batch])


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
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 500


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


def _cell_halo(cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r, c = cell
    return frozenset(
        (rr, cc)
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
                    "kind": "L" if abs(dr) + abs(dc) == 2 else "V",
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg_halo: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg_halo)


def _bg_dest_is_clear(
    dest: tuple[int, int],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> bool:
    return dest not in occupied_halo and dest not in reserved_bg_halo


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _place_sprite(sprites: list[dict], placement: dict, occupied: set[tuple[int, int]], occupied_halo: set[tuple[int, int]]) -> None:
    sprites.append({"placement": placement})
    occupied.update(placement["cells"])
    occupied_halo.update(placement["halo"])


def _reserve_bg_dest(dest: tuple[int, int], reserved_bg: set[tuple[int, int]], reserved_bg_halo: set[tuple[int, int]]) -> None:
    reserved_bg.add(dest)
    reserved_bg_halo.update(_cell_halo(dest))


def _choose_mixed_collision(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if not _bg_dest_is_clear(dest, occupied_halo, reserved_bg_halo):
            continue
        l_candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == "L" and _fits(p, occupied, reserved_bg_halo)]
        v_candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == "V" and _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(l_candidates)
        rng.shuffle(v_candidates)
        for first in l_candidates:
            occ2 = occupied | set(first["cells"])
            for second in v_candidates:
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_same_kind_collision(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
    kind: str,
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if not _bg_dest_is_clear(dest, occupied_halo, reserved_bg_halo):
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == kind and _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique_bg(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if not _bg_dest_is_clear(placement["dest"], occupied_halo, reserved_bg_halo):
            continue
        if _fits(placement, occupied, reserved_bg_halo):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    kinds = set()
    for comp in comps:
        if comp.shape != (3, 2):
            return False
        info = _triangle_apex_dir_kind([(int(r), int(c)) for r, c in comp])
        if info is None:
            return False
        kinds.add(info[2])
    return kinds == {"L", "V"} or kinds == {"V", "L"}


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    hit_events = rng.randint(0, min(MAX_HIT_EVENTS, n_sprites - 4))
    pair_budget = (n_sprites - 4 - hit_events) // 2
    extra_pair_events = rng.randint(0, min(2, pair_budget))
    extra_pair_kinds = [rng.choice(["mixed", "L", "V"]) for _ in range(extra_pair_events)]

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    occupied_halo: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    reserved_bg_halo: set[tuple[int, int]] = set()

    mixed = _choose_mixed_collision(rng, occupied, occupied_halo, reserved_bg_halo)
    if mixed is None:
        return None
    mixed_dest, mixed_pair = mixed
    _reserve_bg_dest(mixed_dest, reserved_bg, reserved_bg_halo)
    for placement in mixed_pair:
        _place_sprite(sprites, placement, occupied, occupied_halo)

    coll_kind = rng.choice(["L", "V"])
    same_phase = _choose_same_kind_collision(rng, occupied, occupied_halo, reserved_bg_halo, coll_kind)
    if same_phase is None:
        return None
    same_phase_dest, same_phase_pair = same_phase
    _reserve_bg_dest(same_phase_dest, reserved_bg, reserved_bg_halo)
    for placement in same_phase_pair:
        _place_sprite(sprites, placement, occupied, occupied_halo)

    for pair_kind in extra_pair_kinds:
        if pair_kind == "mixed":
            extra = _choose_mixed_collision(rng, occupied, occupied_halo, reserved_bg_halo)
        else:
            extra = _choose_same_kind_collision(rng, occupied, occupied_halo, reserved_bg_halo, pair_kind)
        if extra is None:
            return None
        extra_dest, extra_pair = extra
        _reserve_bg_dest(extra_dest, reserved_bg, reserved_bg_halo)
        for placement in extra_pair:
            _place_sprite(sprites, placement, occupied, occupied_halo)

    used_target_cells: set[tuple[int, int]] = set()
    for _ in range(hit_events):
        target_idx = rng.randrange(len(sprites))
        choice = _choose_hit(rng, sprites[target_idx], occupied, reserved_bg_halo, used_target_cells)
        if choice is None:
            return None
        placement, cell = choice
        _place_sprite(sprites, placement, occupied, occupied_halo)
        used_target_cells.add(cell)

    while len(sprites) < n_sprites:
        placement = _choose_unique_bg(rng, occupied, occupied_halo, reserved_bg_halo)
        if placement is None:
            return None
        _reserve_bg_dest(placement["dest"], reserved_bg, reserved_bg_halo)
        _place_sprite(sprites, placement, occupied, occupied_halo)

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = main(inp)
    simultaneous = _solve_simultaneous(inp)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, simultaneous):
        return None
    if out[mixed_dest] != HIT_COLOR:
        return None
    if out[same_phase_dest] != COLLISION_COLOR:
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
