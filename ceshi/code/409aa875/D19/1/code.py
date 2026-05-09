# Variant of 409aa875
# Dimension: D19 — 未命中精灵的保留策略 (transform)
# Change: 未被命中的精灵保持原色不变... → 所有精灵在输出中消失，仅保留投射物标记

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D19 variant
# -----------------------------------
# variant_spec:
#   unchanged_geometry:
#     - grid is fixed 16x16 with bg=7
#     - each sprite is still a monochrome 3-pixel L or V component
#     - each sprite still emits exactly one projectile
#     - landing cell is still apex + 5 * direction
#     - a shared empty-bg landing still becomes blue(1); a lone landing is maroon(9)
#   changed_transform:
#     - original: start from the input grid, keep all untouched sprites, recolor a hit sprite as a 3-pixel maroon object
#     - variant: erase every sprite in the output, regardless of hit status; only keep point markers at projectile landing cells
#
# a) Concrete behavior trace:
#   Example 1: a lone shooter lands on empty bg at (4, 10).
#     - Original output kept every triangle and added a single 9 at (4, 10).
#     - Variant output is bg everywhere except a single 9 at (4, 10).
#   Example 2: a shooter lands on one cell of another triangle at (9, 3).
#     - Original output recolored the entire 3-cell target triangle to 9.
#     - Variant output erases both triangles and keeps only one 9 at the struck cell (9, 3).
#   The change happens only in the final rendering stage; direction decoding and landing-cell computation stay identical.
#
# b) Complexity calibration:
#   - The input side keeps the original 4-9 sprite density and the full mix of 8 possible firing directions, so the inverse problem stays rich.
#   - The output is intentionally sparse, but not trivial: blue collision markers are still generated often, and occupied-cell hits remain possible even though they now collapse to single dots.
#   - To avoid accidental ambiguity where output dots touch and resemble a new sprite, the generator enforces Chebyshev>=2 spacing between distinct landing markers.
#
# c) Implicit-assumption audit against the original solver:
#   - "out = g.copy()" depended on sprites surviving into the output. That premise is now false; the variant must start from an all-bg canvas.
#   - Recoloring every cell of a hit component depended on "hit sprite remains visible as an object". That is the exact property being removed, so the recolor loop must become a single landing-point mark.
#   - The original bg-collision rule remains valid because it depends only on how many projectiles share an empty landing cell, not on sprite preservation.
#   - Building connected components from the input is still valid: the solver still needs the original sprite cells only to decode directions and detect whether a landing point was occupied.
#
# d) Placement legality in the generator:
#   - Sprite placement still enforces in-bounds cells, in-bounds landing cells, and Chebyshev>=2 separation between sprites so 8-connected components never merge.
#   - Empty-bg landing cells are reserved so later sprites cannot occupy them.
#   - Distinct visible landing markers are kept Chebyshev-separated, which guarantees the output contains isolated dots only.
#
# e) Visual observability:
#   - Across samples, the striking visual signature is that every 3-pixel triangle disappears completely and the output consists only of sparse 1/9 points.
#   - Because n_sprites>=4 and hit events are capped at 2, each generated puzzle necessarily contains multiple sprites that would have survived in the original task but are erased here.
#
# f) Visual cleanliness:
#   - The marker-spacing rule avoids clutter and prevents the sparse output from accidentally forming new multi-pixel objects.
#
# g) Color inference:
#   - The variant uses only the original two projectile colors in the output: maroon(9) for lone landings / sprite hits, blue(1) for multi-projectile bg collisions.
#   - No invisible property such as color-id ordering is introduced.


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
    Decode the triangle shots and output only the projectile landing markers.

    Input structure:
    - bg = most common color. Every non-bg 8-connected component is a 3-pixel triangle sprite:
      either an axis-aligned L or a diagonal V.

    Rule:
    - Infer each sprite's apex and firing direction from its 3-pixel geometry.
    - Fire one projectile per sprite; its landing cell is exactly 5 steps from the apex.
    - Erase all sprites from the output. Then mark only the landing cells:
      a hit on any sprite cell becomes maroon (9), a lone empty-bg landing becomes maroon (9),
      and a shared empty-bg landing by 2+ projectiles becomes blue (1).

    Key insight: The geometry-to-direction decoding is unchanged from the original task; the
    variant changes only the render policy, collapsing every interaction to sparse landing dots.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

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
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    out = np.full_like(g, bg)

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            out[r, c] = 9
        elif len(shooters) >= 2:
            out[r, c] = 1
        else:
            out[r, c] = 9

    return out


# --------------- generator ---------------

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
MAX_ATTEMPTS = 300


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


def _point_halo(point: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r, c = point
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


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    blocked_targets: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied or dest in blocked_targets:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1:]:
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
    blocked_targets: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied or placement["dest"] in blocked_targets:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    blocked_targets: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in blocked_targets:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
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


def _validate_output(grid: np.ndarray) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    if not set(int(v) for v in np.unique(grid)).issubset({BG, HIT_COLOR, COLLISION_COLOR}):
        return False
    comps = components_mask(grid != BG, connectivity=8)
    return all(comp.shape[0] == 1 for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 2:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    if not options:
        return None
    with_collision = [opt for opt in options if opt[1] >= 1]
    pool = with_collision if with_collision and rng.random() < 0.7 else options
    return rng.choice(pool)


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg, blocked_targets) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg, blocked_targets)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        blocked_targets.update(_point_halo(dest))
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg, blocked_targets) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg, blocked_targets)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        blocked_targets.update(_point_halo(placement["dest"]))
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg, blocked_targets) -> bool:
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, blocked_targets)
        if choice is None:
            return False
        placement, cell = choice
        blocked_targets.update(_point_halo(cell))
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = np.full_like(inp, BG)
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    for sprite in sprites:
        if sprite["target_kind"] == "hit":
            out[sprite["target"]] = HIT_COLOR
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
    blocked_targets: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg, blocked_targets):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg, blocked_targets):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg, blocked_targets):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg, blocked_targets):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)

    if np.count_nonzero(out != BG) >= np.count_nonzero(inp != BG):
        return None
    if not _validate_output(out):
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
