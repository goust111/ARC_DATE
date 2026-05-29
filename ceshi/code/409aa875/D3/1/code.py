# Variant of 409aa875
# Dimension: D3 — 精灵形状与方向编码 (src)
# Change: 两种3像素三角形：L形（轴对齐直角，占2x2的3个角）和V形（顶点+两条对角臂），形状决定发射方向（L对角线/V轴方向）... → 大小可变的三角精灵（3-5像素）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color


# Design analysis for the D3 variant
# ----------------------------------
# concrete_behavior:
#   - Keep the original eight emission headings, but enlarge the source sprites.
#   - Diagonal shots still come from L-family sprites: apex at the elbow, with one
#     horizontal arm and one vertical arm trailing behind the apex. Each arm has
#     length 1 or 2, so the component has 1+a+b pixels in {3,4,5}.
#   - Axis-aligned shots still come from V-family sprites: apex at the point,
#     with two diagonal arms trailing behind it. Again each arm has length 1 or 2.
#   - The visible D3 change is not just "bigger objects exist": pixel count now
#     changes the projectile range. A 3/4/5-pixel sprite shoots 4/5/6 cells.
#   - Example trace:
#       original: a 3-pixel L at apex (5,5) firing up-left lands at (0,0).
#       variant: a 5-pixel L with the same apex and heading would land at (-1,-1)
#       because its range is now 6, so the generator must move such a sprite inward
#       to keep the landing cell in bounds. The hit/collision coloring rule downstream
#       is unchanged; only source decoding and range differ.
# complexity_calibration:
#   - Arm lengths are restricted to {1,2}. That gives only four templates per
#     heading ((1,1),(1,2),(2,1),(2,2)), enough visible variety without making
#     recognition ambiguous or requiring many hidden sub-cases.
# implicit_assumptions_to_remove:
#   - `len(comp) == 3` is no longer valid; recognition must infer size from rays.
#   - `dest = apex + 5*dir` is no longer valid; distance depends on component size.
#   - Generator validation can no longer require each component to have shape (3,2).
#   - Placement precomputation must enumerate the expanded templates, not just one
#     canonical 3-pixel mask per direction.
# placement_legality:
#   - Sprites remain pairwise separated by a 1-cell halo so components never merge.
#   - Every generated placement guarantees both sprite cells and its derived landing
#     cell are in bounds. Hit-events target an existing sprite cell; collision-events
#     target an empty background cell shared by multiple shooters.
# visual_observability_and_cleanliness:
#   - Every generated grid must contain at least one sprite larger than 3 pixels,
#     and at least two distinct sizes overall, so the new size→range rule is
#     observable within a single sample rather than only across a dataset.
#   - Colors still play the same structural role as the original task: all input
#     sprites share one non-background color, while outputs only introduce maroon
#     (9) and blue (1) as result markers. No invisible color ordering is used.


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
) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    """
    Return (apex_rc, dir_rc, pixel_count, projectile_dist) for a valid variant sprite.
    """
    if len(comp) not in (3, 4, 5):
        return None

    comp_set = set(comp)
    matches: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []

    for apex in comp:
        for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            arm_lens = _match_component(comp_set, apex, *_l_arm_vectors(direction))
            if arm_lens is not None:
                size = 1 + arm_lens[0] + arm_lens[1]
                matches.append((apex, direction, size, size + 1))
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            arm_lens = _match_component(comp_set, apex, *_v_arm_vectors(direction))
            if arm_lens is not None:
                size = 1 + arm_lens[0] + arm_lens[1]
                matches.append((apex, direction, size, size + 1))

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from variable-size triangle sprites; larger triangles shoot farther.

    Input structure:
    - The background is the most common color.
    - Every non-background monochrome component is a 3-5 pixel triangle sprite.
    - Diagonal-firing sprites are L-shaped elbows with arm lengths 1 or 2; axis-firing sprites are V-shaped wedges with diagonal arm lengths 1 or 2.

    Rule:
    - Decode each component's apex and firing direction from its two rays.
    - The projectile distance is the component's pixel count plus 1, so 3/4/5-pixel sprites land 4/5/6 cells away.
    - A projectile that lands on a sprite recolors the whole struck sprite to maroon (9).
    - A projectile landing on empty background paints maroon (9), except that two or more projectiles sharing that empty cell paint it blue (1).

    Key insight:
    - Orientation still chooses the heading, but the component's total size now controls range, so the solver must read both shape family and pixel count before applying the unchanged hit/collision coloring rule.
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
        sig = _triangle_signature(comp)
        if sig is None:
            continue
        (ar, ac), (dr, dc), _size, dist = sig
        rr = ar + dist * dr
        cc = ac + dist * dc
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


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
ARM_LENGTH_PAIRS = [(1, 1), (1, 2), (2, 1), (2, 2)]

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 600


def _sprite_cells(
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
            for direction in ALL_DIRS:
                for arm_lens in ARM_LENGTH_PAIRS:
                    size = 1 + arm_lens[0] + arm_lens[1]
                    dist = size + 1
                    dest = (ar + dist * direction[0], ac + dist * direction[1])
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                        continue
                    cells = _sprite_cells((ar, ac), direction, arm_lens)
                    if not _in_bounds_cells(cells):
                        continue
                    placement = {
                        "apex": (ar, ac),
                        "dir": direction,
                        "arm_lens": arm_lens,
                        "size": size,
                        "dist": dist,
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


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    signatures = [_triangle_signature(comp) for comp in comps]
    return all(sig is not None for sig in signatures)


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


def _size_profile(sprites: list[dict]) -> list[int]:
    return [int(sprite["placement"]["size"]) for sprite in sprites]


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

    sizes = _size_profile(sprites)
    if max(sizes) < 4 or len(set(sizes)) < 2:
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
