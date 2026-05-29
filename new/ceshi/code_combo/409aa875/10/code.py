# Combo variant of 409aa875 (index 10)
# combo_spec: D3/0+D12/0
# Name: 箭头精灵+方向由最近边缘决定

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg

_bg_color = get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The visible sprites are now 3-pixel arrows in 8 compass facings, so the input suggests
#    "follow the arrow tip". The fusion rule breaks that expectation: the tip still marks the
#    launch cell, but the projectile direction is chosen from the sprite's position in the frame.
#    Example inside a 16x16 grid:
#      cells {(3,6),(4,5),(4,6)} form a NE-looking arrow with tip (3,6).
#      D3-only reading would fire NE to ( -2,11 ), which is invalid/out of bounds here.
#      Combo reading measures edge distances from the whole sprite:
#        top=3, bottom=11, left=5, right=9 -> uniquely nearest edge is TOP.
#      Therefore the real projectile goes TOWARD that edge, i.e. upward, to (3,6)+5*(-1,0)=( -2,6 ),
#      so this exact placement is excluded by generation. A nearby legal case at rows 6..7 instead
#      fires upward to an in-bounds cell even though the arrow still visually points NE.
#      The changed step is therefore: "read tip from shape" stays, "read direction from shape"
#      is replaced by "read nearest border from global position".
#
# b) Complexity calibration:
#    - Keep exactly 8 arrow templates and exactly 4 actual firing directions (toward top/bottom/
#      left/right), so the hypothesis space stays small.
#    - Require both diagonal-looking and axis-looking arrows in each generated grid; otherwise the
#      input could collapse into a trivial all-axis layout.
#    - Require repeated evidence that arrows are misleading: several sprites must fire in a
#      different direction than they visibly face, and at least one nearest-edge class must contain
#      multiple visual facings.
#
# c) Implicit assumption audit against the original code:
#    - Original `_triangle_apex_and_dir()` bundled emitter recovery and path inference. In the
#      combo, only emitter recovery survives, and it must be replaced by arrow-template matching.
#    - `dest = emitter + 5*dir` still holds, so the downstream hit/collision logic is unchanged.
#    - Direction is no longer one of 8 local shape directions; it is one of 4 border-driven axis
#      directions. This adds a new assumption: each sprite must have a unique nearest edge.
#    - `cell_to_comp[dest]` remains safe because generation still enforces a one-cell halo, so
#      components never merge under 8-connectivity.
#
# d) Placement legality:
#    - Generation precomputes all arrow placements whose 3 cells are in bounds, whose nearest edge
#      is unique, and whose tip+5*toward_edge destination is also in bounds.
#    - Halo spacing preserves Chebyshev distance >= 2, preventing component merges and ambiguous
#      hits.
#    - Under this toward-edge geometry, legal 5-step destinations stay in narrow border bands.
#      In this placement space that makes sprite-hit events and same-cell collision pairs
#      effectively unavailable, so generated examples focus on distinct empty-background landings.
#      The solver would still handle a legal hit or collision if one appeared.
#
# e) Visual observability:
#    - The crucial observable is disagreement between arrow facing and actual shot direction.
#      Generated samples are rejected unless the combo output differs from a D3-only "follow the
#      arrow" solver on the same input.
#    - Repeated support is added by forcing multiple misleading sprites and by mixing different
#      arrow facings within at least one shared nearest-edge class.
#
# f) Visual cleanliness:
#    - Sprites stay sparse on a fixed 16x16 grid, landing markers stay single-cell, and sprite hits
#      recolor whole 3-pixel components. No extra overlays, legends, or multi-color coding are
#      introduced.
#
# g) Color inferability:
#    - Colors keep the original semantics: original sprite color is arbitrary but uniform; maroon 9
#      marks lone background landings and hit sprites; blue 1 marks multi-projectile collisions on
#      empty background. No rule depends on numeric color ordering.
#
# h) Fusion metric used after implementation:
#    - For each sprite, compare `visual_dir` from the arrow template with `actual_dir` from the
#      nearest-edge rule. The main metric is `misleading_count`, the number of sprites where these
#      differ. Accepted samples require several mismatches and must also satisfy
#      `main(inp) != _visual_dir_main(inp)`.
#
# i) New solver assumptions to audit:
#    - Every non-bg component is exactly one 3-pixel arrow matching a unique template.
#    - Every generated arrow has a unique nearest edge; ties are excluded.
#    - The projectile always starts from the arrow tip, not from any other pixel.
#    - Projectile destinations remain in bounds by construction.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

PROJECTILE_DIST = 5
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = AXIS_DIRS + DIAG_DIRS

SPRITE_COUNT_RANGE = (4, 9)
# Under the nearest-edge rule plus Chebyshev>=2 spacing, legal 5-step shots cannot
# simultaneously realize sprite-hit or shared-destination collision events on this 16x16 canvas.
MAX_HIT_EVENTS = 0
MAX_COLL_EVENTS = 0
MAX_ATTEMPTS = 500

MIN_DIAGONAL_ARROWS = 2
MIN_AXIS_ARROWS = 1
MIN_MISLEADING_ARROWS = 3
# Every accepted grid must expose all four border-driven direction branches.
MIN_EDGE_CLASSES = 4


ARROW_OFFSETS_BY_DIR: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (-1, 0): ((0, 0), (1, -1), (1, 1)),
    (-1, 1): ((0, 0), (1, -1), (1, 0)),
    (0, 1): ((0, 0), (-1, -1), (1, -1)),
    (1, 1): ((0, 0), (-1, -1), (0, -1)),
    (1, 0): ((0, 0), (-1, -1), (-1, 1)),
    (1, -1): ((0, 0), (-1, 0), (-1, 1)),
    (0, -1): ((0, 0), (-1, 1), (1, 1)),
    (-1, -1): ((0, 0), (0, 1), (1, 1)),
}

TOWARD_EDGE_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _arrow_tip_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    cell_set = set(comp)
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for tip in comp:
        tr, tc = tip
        for direction, offsets in ARROW_OFFSETS_BY_DIR.items():
            shaped = {(tr + dr, tc + dc) for dr, dc in offsets}
            if shaped == cell_set:
                matches.append((tip, direction))
    if len(matches) != 1:
        return None
    return matches[0]


def _nearest_edge_dir(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> tuple[str, tuple[int, int]] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    distances = {
        "top": min(rows),
        "bottom": (h - 1) - max(rows),
        "left": min(cols),
        "right": (w - 1) - max(cols),
    }
    best = min(distances.values())
    closest = [edge for edge, dist in distances.items() if dist == best]
    if len(closest) != 1:
        return None
    edge = closest[0]
    return edge, TOWARD_EDGE_DIR[edge]


def _arrow_tip_and_edge_dir(
    comp: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    tip_data = _arrow_tip_and_dir(comp)
    if tip_data is None:
        return None
    tip, _visual_dir = tip_data
    edge_data = _nearest_edge_dir(comp, h, w)
    if edge_data is None:
        return None
    _edge, actual_dir = edge_data
    return tip, actual_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel arrow sprites, but aim them by border proximity instead of arrow facing.

    Input structure:
    - The background is the most common color.
    - Every non-background 8-connected component is a monochrome 3-pixel arrow in one of 8 visual facings.
    - The arrow tip still identifies the launch cell, but the arrow's pointing direction is not the movement rule.

    Rule:
    - Match each component to an arrow template to recover its tip.
    - Compare the sprite's occupied cells to the four grid edges; the projectile travels 5 cells from the tip TOWARD the uniquely nearest edge, always axis-aligned.
    - If that landing cell lies on another sprite, recolor the entire hit sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, paint it blue (1); otherwise a lone empty-background landing paints maroon (9).

    Key insight: The arrowhead is a red herring. Shape tells you where the shot starts, but the shot direction comes from the sprite's global position relative to the border.
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
        td = _arrow_tip_and_edge_dir(comp, h, w)
        if td is None:
            continue
        (tr, tc), (dr, dc) = td
        rr = tr + PROJECTILE_DIST * dr
        cc = tc + PROJECTILE_DIST * dc
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
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _visual_dir_main(grid: np.ndarray) -> np.ndarray:
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
        td = _arrow_tip_and_dir(comp)
        if td is None:
            continue
        (tr, tc), (dr, dc) = td
        rr = tr + PROJECTILE_DIST * dr
        cc = tc + PROJECTILE_DIST * dc
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
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR
    return out


def _sprite_cells(tip: tuple[int, int], visual_dir: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    tr, tc = tip
    return tuple((tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS_BY_DIR[visual_dir])


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
    for tr in range(H):
        for tc in range(W):
            for visual_dir in ALL_DIRS:
                cells = _sprite_cells((tr, tc), visual_dir)
                if not _in_bounds_cells(cells):
                    continue
                edge_data = _nearest_edge_dir(cells, H, W)
                if edge_data is None:
                    continue
                edge, actual_dir = edge_data
                dest = (tr + PROJECTILE_DIST * actual_dir[0], tc + PROJECTILE_DIST * actual_dir[1])
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                placement = {
                    "tip": (tr, tc),
                    "visual_dir": visual_dir,
                    "actual_dir": actual_dir,
                    "edge": edge,
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "looks_axis": abs(visual_dir[0]) + abs(visual_dir[1]) == 1,
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    targetable = frozenset(by_dest)
    for placement in placements:
        placement["targetable_cells"] = tuple(cell for cell in placement["cells"] if cell in targetable)
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
    *,
    require_targetable: bool = False,
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if require_targetable and not placement["targetable_cells"]:
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
    cells = list(target_sprite["placement"]["targetable_cells"])
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
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    for comp in comps:
        if comp.shape != (3, 2):
            return False
        coords = [(int(r), int(c)) for r, c in comp]
        if _arrow_tip_and_edge_dir(coords, H, W) is None:
            return False
    return True


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
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


def _count_targetable_sprites(sprites: list[dict]) -> int:
    return sum(1 for sprite in sprites if sprite["placement"]["targetable_cells"])


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg, *, require_targetable=False) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg, require_targetable=require_targetable)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    candidate_ids = [idx for idx, sprite in enumerate(sprites) if sprite["placement"]["targetable_cells"]]
    if len(candidate_ids) < e_hit:
        return False
    hit_targets = rng.sample(candidate_ids, e_hit)
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
    bg_targets = Counter(sprite["target"] for sprite in sprites if sprite["target_kind"] == "bg")
    hit_sprite_ids = {sprite["target_sprite"] for sprite in sprites if sprite["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _supports_variant(sprites: list[dict], inp: np.ndarray, out: np.ndarray) -> bool:
    diag_count = sum(1 for sprite in sprites if not sprite["placement"]["looks_axis"])
    axis_count = len(sprites) - diag_count
    if diag_count < MIN_DIAGONAL_ARROWS or axis_count < MIN_AXIS_ARROWS:
        return False

    edge_classes = {sprite["placement"]["edge"] for sprite in sprites}
    if len(edge_classes) < MIN_EDGE_CLASSES:
        return False

    misleading_count = sum(
        1
        for sprite in sprites
        if sprite["placement"]["visual_dir"] != sprite["placement"]["actual_dir"]
    )
    if misleading_count < MIN_MISLEADING_ARROWS:
        return False

    edge_to_visuals: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for sprite in sprites:
        placement = sprite["placement"]
        edge_to_visuals[placement["edge"]].add(placement["visual_dir"])
    if not any(len(visuals) >= 2 for visuals in edge_to_visuals.values()):
        return False

    if np.array_equal(_visual_dir_main(inp), out):
        return False
    return True


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

    base_needed = max(0, e_hit - _count_targetable_sprites(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg, require_targetable=True):
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

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if not _supports_variant(sprites, inp, out):
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
