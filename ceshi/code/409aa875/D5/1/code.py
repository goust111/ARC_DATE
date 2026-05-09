# Variant of 409aa875
# Dimension: D5 — 碰撞规则（投射物-投射物） (transform)
# Change: 两枚以上投射物落在同一空背景格时涂蓝(1)，单枚涂栗色(9)... → 碰撞颜色由参与者数量决定

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# ---------------
# analysis:
#   concretization:
#     - Grid geometry, sprite shapes, firing directions, projectile distance, and sprite-hit behavior stay the
#       same as the original task. The only semantic change is at EMPTY background landing cells.
#     - A background collision point is still "one landing coordinate shared by several sprites", but now its
#       output color must encode the exact number of shooters instead of the coarse predicate ">=2".
#     - Because the fixed background color is 7, the direct numeric continuation 2->1, 3->2, ..., 8->7 would
#       make an 8-way collision invisible. To keep D5 visually observable, the count palette uses increasing
#       visible non-background colors while skipping 7:
#         2->1, 3->2, 4->3, 5->4, 6->5, 7->6, 8->8, 9->9.
#     - Small trace:
#       original: three sprites land on the same empty cell -> len(shooters)=3 -> ">=2" -> blue(1)
#       variant:  three sprites land on the same empty cell -> len(shooters)=3 -> red(2)
#       difference: the solver must preserve multiplicity, not just detect that a collision happened.
#   complexity_calibration:
#     - The solver supports counts up to the task maximum, but the generator concentrates on 2/3/4-way
#       collisions so the rule is learnable from a few ARC examples instead of spreading evidence across too
#       many color cases.
#     - Every generated sample contains at least one 3-way or 4-way background collision, so it cannot collapse
#       back to the original task where all multi-collisions look blue.
#     - Sprite colors are restricted to {0,4,5,6,8}. This avoids accidental reuse of the key collision colors
#       1/2/3 and keeps maroon(9) recolors visibly meaningful.
#   implicit_assumptions_review:
#     - The component parsing logic still assumes each non-background object is a monochrome 3-pixel triangle.
#       That remains valid because the generator still builds only canonical L and V sprites.
#     - The old output loop relied on a harmless threshold: len(shooters)>=2 always wrote color 1. In the
#       variant, that threshold is no longer sufficient; exact counts must survive until the final write.
#     - Writing a single color per destination is still safe because each background destination is processed
#       once from the grouped dictionary, and sprite-hit destinations are skipped before any background paint.
#   placement_legality:
#     - Placements are still precomputed from (apex, direction) pairs whose sprite cells and landing cell are
#       in bounds.
#     - Halo separation is preserved so sprites remain distinct 8-connected components, which keeps solver-side
#       component extraction reliable.
#     - Collision groups are chosen by backtracking over compatible shooters that share one destination but whose
#       sprite bodies remain mutually separated.
#   visual_observability:
#     - Samples always include at least one red(2) or green(3) collision marker, guaranteeing that the new
#       count-sensitive rule actually appears in the dataset.
#     - Many samples also include a lone maroon landing or a maroon sprite-hit, so the viewer can contrast
#       "single projectile" vs "multi-projectile with count-coded color" directly.
#   visual_cleanliness:
#     - Collision markers stay on empty cells only, never on top of sprites.
#     - Sprite spacing and the limited collision-count set keep the outputs readable rather than filling the
#       board with many unrelated marker colors.
#   color_inference:
#     - The color change is tied to a visible structural cue: the number of triangles aiming at the same empty
#       cell. No hidden ordering, palette arithmetic on sprite colors, or color-id sorting is used.
#   solver_assumptions_after_change:
#     - main() assumes every relevant collision destination can be summarized by the number of shooters only.
#       generate_input() guarantees this by never mixing sprite-hit and empty-cell semantics at one coordinate.


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


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
SPRITE_COLORS = [0, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
COLLISION_PATTERNS = (
    (3,),
    (4,),
    (2, 3),
    (2, 4),
    (3, 3),
    (3, 4),
)

# ---- generation retries ----
MAX_ATTEMPTS = 500


def _collision_color(shooter_count: int) -> int:
    if shooter_count <= 1:
        return HIT_COLOR
    idx = min(shooter_count - 2, len(COLLISION_PALETTE) - 1)
    return COLLISION_PALETTE[idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangular sprites launch 5-cell projectiles; sprite hits stay maroon while empty-cell collisions are count-colored.

    Input structure:
    - The background is the most frequent color, and every non-background object is a 3-pixel monochrome triangle.
    - Triangles appear only in the original two canonical forms: an axis-aligned L or a diagonal V.

    Rule:
    - Read each triangle's apex and firing direction from its shape: L-shapes shoot diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their opening.
    - Every projectile lands exactly 5 cells from the apex along that direction.
    - If a landing cell lies on a sprite, recolor the entire hit sprite to maroon (9).
    - If a landing cell is empty background, paint that one cell according to how many projectiles arrived there: 1 projectile -> maroon (9), 2 -> blue (1), 3 -> red (2), 4 -> green (3), then continue through the visible non-background palette while skipping the background color 7.

    Key insight:
    - The geometry is unchanged from the original task; the new work is to group projectiles by shared empty landing cell and keep the exact group size, because different collision multiplicities now produce different output colors.
    """
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

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = _collision_color(len(shooters))

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


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _pick_compatible_group(
    rng: random.Random,
    candidates: list[dict],
    need: int,
    occupied: set[tuple[int, int]],
) -> list[dict] | None:
    order = candidates[:]
    rng.shuffle(order)

    def search(start: int, blocked: set[tuple[int, int]], chosen: list[dict]) -> list[dict] | None:
        if len(chosen) == need:
            return chosen
        remaining_need = need - len(chosen)
        if len(order) - start < remaining_need:
            return None
        for idx in range(start, len(order)):
            placement = order[idx]
            if placement["halo"] & blocked:
                continue
            found = search(
                idx + 1,
                blocked | set(placement["cell_set"]),
                chosen + [placement],
            )
            if found is not None:
                return found
        return None

    return search(0, set(occupied), [])


def _choose_collision_group(
    rng: random.Random,
    size: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        if len(candidates) < size:
            continue
        group = _pick_compatible_group(rng, candidates, size, occupied)
        if group is not None:
            return dest, group
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
        if not candidates:
            continue
        rng.shuffle(candidates)
        for placement in candidates:
            return placement, cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(len(comp) == 3 and _triangle_apex_and_dir(comp) is not None for comp in comps)


def _event_options(n_sprites: int) -> list[tuple[int, tuple[int, ...], int]]:
    options: list[tuple[int, tuple[int, ...], int]] = []
    for coll_sizes in COLLISION_PATTERNS:
        used = sum(coll_sizes)
        if used > n_sprites:
            continue
        max_hits = min(MAX_HIT_EVENTS, n_sprites - used)
        for e_hit in range(max_hits + 1):
            unique = n_sprites - used - e_hit
            if unique < 0:
                continue
            if unique + e_hit < 1:
                continue
            options.append((e_hit, coll_sizes, unique))
    return options


def _place_collisions(rng, coll_sizes, sprites, occupied, reserved_bg) -> bool:
    for size in coll_sizes:
        choice = _choose_collision_group(rng, size, occupied, reserved_bg)
        if choice is None:
            return False
        dest, group = choice
        reserved_bg.add(dest)
        for placement in group:
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
    if e_hit > len(sprites):
        return False
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


def _bg_target_counts(sprites: list[dict]) -> Counter:
    return Counter(s["target"] for s in sprites if s["target_kind"] == "bg")


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = _bg_target_counts(sprites)
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = _collision_color(count)
    return out


def _render_original_style_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = _bg_target_counts(sprites)
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = 1 if count >= 2 else HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    options = _event_options(n_sprites)
    if not options:
        return None
    e_hit, coll_sizes, unique_total = rng.choice(options)

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, coll_sizes, sprites, occupied, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = unique_total
    if not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    if len(sprites) != n_sprites:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    bg_targets = _bg_target_counts(sprites)
    if not any(count >= 3 for count in bg_targets.values()):
        return None

    out = _render_output(inp, sprites)
    old_out = _render_original_style_output(inp, sprites)

    if np.array_equal(out, old_out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(np.isin(out, [2, 3])):
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
