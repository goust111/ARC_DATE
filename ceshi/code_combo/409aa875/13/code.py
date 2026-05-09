# Combo variant of 409aa875 (index 13)
# combo_spec: D5/3+D8/0+D15/2
# Name: 极简版：小网格+2精灵+无碰撞特殊色

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# variant_spec:
#   task_type: move_along_path
#   grid:
#     shape: 8 rows x 8 cols
#     bg: 7
#   sprite:
#     count: exactly 2 triangular sprites
#     color: one shared non-bg color sampled from {2,3,4,5,6,8}
#       Rationale: excluding 1 and 9 keeps the transformed maroon marks visible,
#       and excluding 0 avoids relying on black for human interpretation.
#     size: exactly 3 cells per sprite, still one 8-connected component each
#     shapes: unchanged geometry decoder
#       - L-shape fires diagonally away from the missing corner
#       - V-shape fires axially away from the open side
#     separation: keep Chebyshev distance >= 2 so the two sprites remain distinct
#   projectile:
#     target_cell: apex + 3 * direction
#   transform:
#     hit_sprite: recolor the entire struck sprite to 9
#     empty_bg_hit: recolor the struck empty cell to 9
#     removed_rule: even if both projectiles land on the same empty bg cell, that
#       cell is still 9; blue (1) is never produced by the fused rule
#
# behavior_trace:
#   Shared-bg witness for the removed blue rule:
#   - Sprite A: apex=(1,1), dir=(1,1), target=(4,4).
#   - Sprite B: apex=(1,4), dir=(1,0), target=(4,4).
#   - Parent 8x8-short-range variant would paint (4,4)=1 because two projectiles
#     meet on one empty cell.
#   - Fused combo paints (4,4)=9 instead, with no sprite recolor.
#   Mutual-hit witness for the two-sprite minimal core:
#   - Sprite A above can instead pair with sprite C: apex=(3,4), dir=(-1,-1).
#   - Then A lands on one cell of C and C lands on one cell of A, so both
#     3-cell sprites recolor to 9 and no background marker appears.
#
# complexity_calibration:
#   - A literal reading of "exactly 2 sprites" + "must mutually hit" would make
#     D5/3 invisible, because empty-cell double impacts could never occur.
#   - To keep the fused change observable, generation stays at exactly 2 sprites
#     but alternates between two legal pairwise interactions on the same 8x8 / 3-step
#     mechanics: mutual-hit pairs and shared-bg pairs.
#   - This is still minimal: there are never more than two objects, never more than
#     one landing site per projectile, and the solver logic remains one-pass.
#
# implicit_assumption_audit:
#   - _triangle_apex_and_dir assumes every non-bg component has exactly 3 cells and
#     belongs to one of the canonical L/V families. _validate_input enforces that.
#   - main() builds cell_to_comp from the input before repainting; this is still
#     valid because projectile destinations are read from the original geometry only.
#   - Shrinking to 8x8 does not change the decoding logic, only PROJECTILE_DIST.
#   - The original len(shooters)>=2 branch on empty background is no longer valid
#     after D5/3; the fused solver must collapse that branch to one maroon outcome.
#   - The generator must not use sprite color 9, otherwise mutual-hit examples could
#     become visually inert even though the solver is correct.
#
# generation_legality:
#   - Placements are precomputed only when both the 3-cell sprite and its 3-step
#     landing cell fit inside the 8x8 canvas.
#   - Pair catalogs keep only placements whose sprites are Chebyshev-separated, so
#     the two components never merge under 8-connectivity.
#   - Shared-bg pairs additionally require the common landing site to be empty in
#     the input; mutual-hit pairs require each destination to lie on the other sprite.
#
# observability_check:
#   - The module cycles through pair modes, so successive generated examples expose
#     both visible outcomes needed for human inference: whole-sprite recolor and
#     same-cell double landing without blue.
#   - Because there is a single sprite color and a single output color, the human
#     cue comes from structure (where the projectile lands), not hidden color ranking.
#
# cleanliness_check:
#   - Two sprites on an 8x8 board keep the scene sparse enough that the apex, shape,
#     and landing result remain separable by eye.
#   - Shared-bg examples add only one marker, and mutual-hit examples recolor whole
#     sprites; there is no multi-layer clutter in the same local patch.


H = W = 8
BG = 7
HIT_COLOR = 9
SPRITE_COLORS = [2, 3, 4, 5, 6, 8]
PROJECTILE_DIST = 3

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT = 2
MAX_ATTEMPTS = 200
MODE_SEQUENCE = ["mutual_hit", "shared_bg", "shared_bg", "mutual_hit"]
_generation_index = 0


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve two-sprite short-range projectile interactions on a compact 8x8 board.

    Input structure:
    - The intended canvas is 8x8 with background given by the most common color.
    - Every non-background object is a monochrome 3-cell triangle, either an L-corner or a diagonal V.
    - In the fused generator there are exactly two such sprites, although the solver logic remains generic.

    Rule:
    - Decode each sprite's apex and firing direction from its shape: L-sprites shoot diagonally away from the missing corner, while V-sprites shoot axially away from the open side.
    - Each projectile lands exactly 3 cells from the apex in that direction.
    - If a landing cell lies on a sprite, repaint that entire struck sprite maroon (9).
    - If a landing cell is empty background, paint that cell maroon (9), even when both projectiles share that same empty cell.

    Key insight: The only semantic change from the parent family is the removal of the blue collision exception; on this 8x8 compressed board, both single empty impacts and double empty impacts collapse to the same maroon marker, while shape-based direction decoding stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        apex_and_dir = _triangle_apex_and_dir(comp)
        if apex_and_dir is None:
            continue
        (ar, ac), (dr, dc) = apex_and_dir
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components: set[int] = set()
    bg_dest_counts: Counter[tuple[int, int]] = Counter()
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])
        else:
            bg_dest_counts[dest] += len(shooters)

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for r, c in bg_dest_counts:
        out[r, c] = HIT_COLOR

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


def _pair_mode(first: dict, second: dict) -> str | None:
    if first["halo"] & second["cell_set"]:
        return None

    first_hits_second = first["dest"] in second["cell_set"]
    second_hits_first = second["dest"] in first["cell_set"]
    shared_bg = (
        first["dest"] == second["dest"]
        and first["dest"] not in first["cell_set"]
        and first["dest"] not in second["cell_set"]
    )

    if first_hits_second and second_hits_first:
        return "mutual_hit"
    if shared_bg:
        return "shared_bg"
    return None


def _build_pair_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = {"mutual_hit": [], "shared_bg": []}
    for i, first in enumerate(PLACEMENTS):
        for second in PLACEMENTS[i + 1 :]:
            mode = _pair_mode(first, second)
            if mode is None:
                continue
            catalog[mode].append({"mode": mode, "placements": (first, second)})
    return catalog


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _validate_input(grid: np.ndarray) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    non_bg_colors = {int(v) for v in np.unique(grid) if int(v) != BG}
    if len(non_bg_colors) != 1:
        return False
    sprite_color = next(iter(non_bg_colors))
    if sprite_color not in SPRITE_COLORS:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != SPRITE_COUNT:
        return False
    if not all(comp.shape == (3, 2) for comp in comps):
        return False
    return all(_triangle_apex_and_dir([(int(r), int(c)) for r, c in comp]) is not None for comp in comps)


def _render_output(inp: np.ndarray, placements: tuple[dict, dict]) -> np.ndarray:
    out = inp.copy()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_sprite[cell] = idx

    hit_sprites: set[int] = set()
    bg_dest_counts: Counter[tuple[int, int]] = Counter()
    for placement in placements:
        dest = placement["dest"]
        if dest in cell_to_sprite:
            hit_sprites.add(cell_to_sprite[dest])
        else:
            bg_dest_counts[dest] += 1

    for idx in hit_sprites:
        _paint_sprite(out, placements[idx], HIT_COLOR)
    for dest in bg_dest_counts:
        out[dest] = HIT_COLOR
    return out


def _classify_pair_interaction(placements: tuple[dict, dict]) -> str:
    first, second = placements
    first_hits_second = first["dest"] in second["cell_set"]
    second_hits_first = second["dest"] in first["cell_set"]
    if first_hits_second and second_hits_first:
        return "mutual_hit"
    if first["dest"] == second["dest"] and first["dest"] not in first["cell_set"] and first["dest"] not in second["cell_set"]:
        return "shared_bg"
    return "other"


def _has_clean_observability(inp: np.ndarray, out: np.ndarray, placements: tuple[dict, dict], mode: str) -> bool:
    input_cells = {cell for placement in placements for cell in placement["cells"]}
    changed_cells = {
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(inp[r, c]) != int(out[r, c])
    }
    recolored_sprite_cells = {cell for cell in input_cells if int(out[cell]) == HIT_COLOR and int(inp[cell]) != HIT_COLOR}
    new_bg_marks = changed_cells - input_cells

    if mode == "mutual_hit":
        return recolored_sprite_cells == input_cells and not new_bg_marks
    if mode == "shared_bg":
        return not recolored_sprite_cells and len(new_bg_marks) == 1 and all(int(out[cell]) == HIT_COLOR for cell in new_bg_marks)
    return False


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()
PAIR_CATALOG = _build_pair_catalog()


def _gen_attempt(mode: str) -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    choices = PAIR_CATALOG[mode]
    if not choices:
        return None

    pair = rng.choice(choices)
    inp = np.full((H, W), BG, dtype=int)
    for placement in pair["placements"]:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp):
        return None

    if _classify_pair_interaction(pair["placements"]) != mode:
        return None

    out = _render_output(inp, pair["placements"])
    if np.array_equal(inp, out):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if mode == "shared_bg" and np.any(out == 1):
        return None
    if not _has_clean_observability(inp, out, pair["placements"], mode):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _generation_index

    mode = MODE_SEQUENCE[_generation_index % len(MODE_SEQUENCE)]
    _generation_index += 1

    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid {mode} sample after {MAX_ATTEMPTS} attempts")
