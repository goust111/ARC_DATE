from __future__ import annotations

# Variant of 409aa875
# Dimension: D3 — 精灵形状与方向编码 (src)
# Change: 两种3像素三角形：L形（轴对齐直角，占2x2的3个角）和V形（顶点+两条对角臂），形状决定发射方向（L对角线/V轴方向）... → 单像素源 + 边缘位置编码方向
#
# Design analysis:
#   concretization_and_trace:
#     - Emitters are no longer 3-pixel interior sprites. Each emitter is exactly one pixel placed on
#       a non-corner border cell. The border itself encodes direction:
#         top -> down, bottom -> up, left -> right, right -> left.
#     - A shot still lands exactly 5 cells away. Therefore top/bottom emitters land on rows 5/10,
#       and left/right emitters land on cols 5/10.
#     - Pure border-only single-pixel inputs would make "hit a sprite and recolor the whole sprite"
#       impossible, because a 5-cell inward shot cannot reach another legal border emitter. To keep
#       that original branch alive without changing the projectile rule, the grid also contains a few
#       passive interior targets: straight 3-pixel bars. They do not encode direction; they only act
#       as hitable objects whose whole component turns maroon.
#     - Small trace:
#         original: an L/V triomino is first classified by local shape, then apex+dir is inferred,
#         then apex+5*dir gives the landing cell.
#         variant: a source at (0, 8) is already a complete emitter. Its edge fixes dir=(1, 0), so
#         the landing cell is (5, 8). If a vertical 3-pixel bar covers (5, 8), the whole bar turns 9.
#         If instead a left-edge source at (5, 0) also exists, its landing cell is (5, 5); two such
#         inward lanes meeting on the same empty cell paint blue just as in the original task.
#   complexity_calibration:
#     - The new control freedom is intentionally small and visible: four edge classes, one fixed
#       inward distance, and passive targets with only two shapes (horizontal/vertical bars).
#     - This avoids two failure modes:
#         1. Over-simplification: if there were only border dots and no hitable interior objects,
#            the "whole sprite recolor" branch would disappear.
#         2. Over-complication: adding many target shapes or color-coded direction tables would make
#            2-4 ARC examples too hard to read.
#     - Collision geometry is also structured rather than noisy: shared empty-cell collisions can
#       only happen where vertical and horizontal inward lanes cross, so blue points look deliberate.
#   hidden_assumptions_review:
#     - Original `_triangle_apex_and_dir(comp)` assumed every emitter is a size-3 component whose
#       local 4/8-neighbor degrees reveal direction. That assumption is false here and must be
#       replaced by an edge-position decoder on size-1 border components.
#     - Original main loop treated every component as a shooter. That is now wrong: only border
#       singleton components emit. Interior 3-pixel bars are passive targets.
#     - Original `cell_to_comp` remained valid for hit detection, but only if the generator keeps
#       emitters and targets as separated 8-connected components. The generator therefore enforces
#       gaps so nothing merges accidentally.
#     - Because direction comes from edge membership, corner cells would be ambiguous (two edges at
#       once). The solver ignores corners, and the generator never places emitters there.
#   placement_legality:
#     - Emitters use border coordinates 2..13 rather than hugging the corners. This avoids diagonal
#       8-connectivity between sources on adjacent edges and keeps each source a size-1 component.
#     - Emitters are spaced by a Chebyshev halo of 1, so no two border sources merge.
#     - Targets stay at least one full cell away from the border and are also halo-separated, so they
#       never merge with one another or with border sources.
#     - Background landing cells reserved for lone-hit/collision events are kept free of target cells.
#   visual_observability:
#     - The D3 change is directly visible: border dots, not interior triangle orientation, determine
#       where marks appear. Every generated grid uses both a row-edge family and a col-edge family so
#       the inward-direction rule is repeatedly demonstrated.
#     - Passive targets are straight bars, not old L/V triangles, so the old shape decoder cannot be
#       mentally reused from the original task.
#   visual_tidiness:
#     - Sources live on the perimeter; targets live inside; landing marks appear on four clean lanes
#       (rows 5/10 and cols 5/10). This keeps the scene readable instead of stacking many object
#       types in one crowded area.
#   color_inference:
#     - Output colors keep the original semantics: maroon(9) = lone bg hit or hit target, blue(1) =
#       multi-projectile bg collision. To make those changes reliably visible, generated object colors
#       exclude 1 and 9.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _edge_source_dir(r: int, c: int, h: int, w: int) -> tuple[int, int] | None:
    edges = []
    if r == 0:
        edges.append((1, 0))
    if r == h - 1:
        edges.append((-1, 0))
    if c == 0:
        edges.append((0, 1))
    if c == w - 1:
        edges.append((0, -1))
    if len(edges) != 1:
        return None
    return edges[0]


def _is_straight_triomino(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 3:
        return False
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1 and len(cols) == 3:
        return cols[-1] - cols[0] == 2
    if len(cols) == 1 and len(rows) == 3:
        return rows[-1] - rows[0] == 2
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire inward projectiles from border control pixels; mark hits with maroon or blue.

    Input structure:
    - bg = most common color. Shooting sources are single non-bg pixels on non-corner border cells.
      Interior non-bg components are passive 3-pixel straight bars that can be hit but do not shoot.

    Rule:
    - Each border singleton fires inward according to its edge: top/down, bottom/up, left/right,
      right/left.
    - A projectile lands exactly 5 cells from the source in that inward direction.
    - If it lands on an existing non-bg component, that entire component is recolored to maroon (9).
    - If multiple projectiles land on the same empty background cell, that cell becomes blue (1);
      a lone projectile on empty bg paints maroon (9).

    Key insight: Direction is no longer encoded by interior shape at all; the source's border
    location alone determines the shot, while interior bars are only passive targets.
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

    emitters: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for cid, comp in enumerate(comps):
        if len(comp) != 1:
            continue
        r, c = comp[0]
        direction = _edge_source_dir(r, c, h, w)
        if direction is not None:
            emitters.append((cid, (r, c), direction))

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, (r, c), (dr, dc) in emitters:
        rr = r + 5 * dr
        cc = c + 5 * dc
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


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
OBJECT_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- structural counts ----
SOURCE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_DECOY_TARGETS = 2
MAX_ATTEMPTS = 400
PROJECTILE_DIST = 5


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_source_candidates() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for c in range(2, W - 2):
        for edge, cell, direction in (
            ("top", (0, c), (1, 0)),
            ("bottom", (H - 1, c), (-1, 0)),
        ):
            dest = (cell[0] + PROJECTILE_DIST * direction[0], cell[1] + PROJECTILE_DIST * direction[1])
            placement = {
                "edge": edge,
                "cell": cell,
                "dir": direction,
                "dest": dest,
                "cells": (cell,),
                "cell_set": frozenset((cell,)),
                "halo": _halo((cell,)),
            }
            placements.append(placement)
            by_dest[dest].append(placement)
    for r in range(2, H - 2):
        for edge, cell, direction in (
            ("left", (r, 0), (0, 1)),
            ("right", (r, W - 1), (0, -1)),
        ):
            dest = (cell[0] + PROJECTILE_DIST * direction[0], cell[1] + PROJECTILE_DIST * direction[1])
            placement = {
                "edge": edge,
                "cell": cell,
                "dir": direction,
                "dest": dest,
                "cells": (cell,),
                "cell_set": frozenset((cell,)),
                "halo": _halo((cell,)),
            }
            placements.append(placement)
            by_dest[dest].append(placement)
    return placements, by_dest


def _build_target_candidates() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for r in range(2, H - 2):
        for c in range(2, W - 2):
            cells_h = ((r, c - 1), (r, c), (r, c + 1))
            placement_h = {
                "shape": "hbar",
                "cells": cells_h,
                "cell_set": frozenset(cells_h),
                "halo": _halo(cells_h),
            }
            placements.append(placement_h)
            for cell in cells_h:
                by_cell[cell].append(placement_h)

            cells_v = ((r - 1, c), (r, c), (r + 1, c))
            placement_v = {
                "shape": "vbar",
                "cells": cells_v,
                "cell_set": frozenset(cells_v),
                "halo": _halo(cells_v),
            }
            placements.append(placement_v)
            for cell in cells_v:
                by_cell[cell].append(placement_v)

    return placements, by_cell


SOURCE_CANDIDATES, SOURCES_BY_DEST = _build_source_candidates()
TARGET_CANDIDATES, TARGETS_BY_CELL = _build_target_candidates()
COLLISION_DESTS = [dest for dest, placements in SOURCES_BY_DEST.items() if len(placements) >= 2]


def _fits_source(placement: dict, occupied: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied)


def _fits_target(
    placement: dict,
    occupied: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & blocked_cells)


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = COLLISION_DESTS[:]
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in used_dests or dest in occupied:
            continue
        candidates = [p for p in SOURCES_BY_DEST[dest] if _fits_source(p, occupied)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if _fits_source(second, occ2):
                    return dest, [first, second]
    return None


def _choose_hit_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
) -> tuple[dict, dict] | None:
    candidates = SOURCE_CANDIDATES[:]
    rng.shuffle(candidates)
    for source in candidates:
        dest = source["dest"]
        if dest in reserved_bg or dest in used_dests:
            continue
        if not _fits_source(source, occupied):
            continue
        blocked = set(reserved_bg) | set(used_dests)
        target_candidates = [
            target
            for target in TARGETS_BY_CELL[dest]
            if _fits_target(target, occupied | set(source["cells"]), blocked)
        ]
        rng.shuffle(target_candidates)
        if target_candidates:
            return source, target_candidates[0]
    return None


def _choose_unique_bg_source(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
) -> dict | None:
    candidates = SOURCE_CANDIDATES[:]
    rng.shuffle(candidates)
    for source in candidates:
        dest = source["dest"]
        if dest in reserved_bg or dest in used_dests or dest in occupied:
            continue
        if _fits_source(source, occupied):
            return source
    return None


def _choose_decoy_target(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> dict | None:
    candidates = TARGET_CANDIDATES[:]
    rng.shuffle(candidates)
    for target in candidates:
        if _fits_target(target, occupied, blocked_cells):
            return target
    return None


def _source_event_counts(rng: random.Random, n_sources: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sources) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sources // 2) + 1):
            unique_bg = n_sources - e_hit - 2 * e_coll
            if unique_bg < 0:
                continue
            if unique_bg + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique_bg))
    return rng.choice(options) if options else None


def _validate_input(grid: np.ndarray, n_sources: int, n_targets: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False

    source_count = 0
    target_count = 0
    for color in sorted(set(int(v) for v in np.unique(grid)) - {BG}):
        for comp in _cc8(grid == color):
            if len(comp) == 1:
                r, c = comp[0]
                if _edge_source_dir(r, c, H, W) is None:
                    return False
                source_count += 1
            else:
                if not _is_straight_triomino(comp):
                    return False
                if any(r in (0, H - 1) or c in (0, W - 1) for r, c in comp):
                    return False
                target_count += 1
    return source_count == n_sources and target_count == n_targets


def _render_output(inp: np.ndarray, sources: list[dict], targets: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["dest"] for s in sources if s["target_kind"] == "bg")
    hit_target_ids = {s["target_id"] for s in sources if s["target_kind"] == "hit"}
    for tid in hit_target_ids:
        _paint_cells(out, targets[tid]["cells"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    source_edges: Counter[str] = Counter()
    target_count = 0

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
            if len(comp) == 1:
                r, c = comp[0]
                direction = _edge_source_dir(r, c, h, w)
                if direction is not None:
                    if r == 0:
                        source_edges["top"] += 1
                    elif r == h - 1:
                        source_edges["bottom"] += 1
                    elif c == 0:
                        source_edges["left"] += 1
                    elif c == w - 1:
                        source_edges["right"] += 1
            elif _is_straight_triomino(comp):
                target_count += 1

    dest_to_shooters: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        if len(comp) != 1:
            continue
        r, c = comp[0]
        direction = _edge_source_dir(r, c, h, w)
        if direction is None:
            continue
        dr, dc = direction
        dest = (r + PROJECTILE_DIST * dr, c + PROJECTILE_DIST * dc)
        if 0 <= dest[0] < h and 0 <= dest[1] < w:
            dest_to_shooters[dest].append((r, c))

    bg_collision_cells = 0
    bg_single_cells = 0
    hit_components: set[int] = set()
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])
        elif len(shooters) >= 2:
            bg_collision_cells += 1
        else:
            bg_single_cells += 1

    return {
        "source_count": int(sum(source_edges.values())),
        "target_count": int(target_count),
        "edge_counts": dict(source_edges),
        "hit_component_count": int(len(hit_components)),
        "bg_collision_cells": int(bg_collision_cells),
        "bg_single_cells": int(bg_single_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    object_color = rng.choice(OBJECT_COLORS)
    n_sources = rng.randint(*SOURCE_COUNT_RANGE)
    counts = _source_event_counts(rng, n_sources)
    if counts is None:
        return None
    e_hit, e_coll, unique_bg = counts
    decoy_targets = rng.randint(0, MAX_DECOY_TARGETS if e_hit > 0 else 1)

    sources: list[dict] = []
    targets: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    used_dests: set[tuple[int, int]] = set()

    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg, used_dests)
        if choice is None:
            return None
        dest, pair = choice
        reserved_bg.add(dest)
        used_dests.add(dest)
        for source in pair:
            sources.append({"source": source, "dest": dest, "target_kind": "bg"})
            occupied.update(source["cells"])

    for _ in range(e_hit):
        choice = _choose_hit_pair(rng, occupied, reserved_bg, used_dests)
        if choice is None:
            return None
        source, target = choice
        target_id = len(targets)
        targets.append(target)
        sources.append({"source": source, "dest": source["dest"], "target_kind": "hit", "target_id": target_id})
        used_dests.add(source["dest"])
        occupied.update(target["cells"])
        occupied.update(source["cells"])

    for _ in range(unique_bg):
        source = _choose_unique_bg_source(rng, occupied, reserved_bg, used_dests)
        if source is None:
            return None
        sources.append({"source": source, "dest": source["dest"], "target_kind": "bg"})
        reserved_bg.add(source["dest"])
        used_dests.add(source["dest"])
        occupied.update(source["cells"])

    for _ in range(decoy_targets):
        target = _choose_decoy_target(rng, occupied, reserved_bg | used_dests)
        if target is None:
            break
        targets.append(target)
        occupied.update(target["cells"])

    inp = np.full((H, W), BG, dtype=int)
    for source in sources:
        _paint_cells(inp, source["source"]["cells"], object_color)
    for target in targets:
        _paint_cells(inp, target["cells"], object_color)

    if not _validate_input(inp, len(sources), len(targets)):
        return None

    analysis = _analyze_input(inp)
    edges = set(analysis["edge_counts"])
    if not ({"top", "bottom"} & edges and {"left", "right"} & edges):
        return None
    if analysis["hit_component_count"] < e_hit:
        return None
    if analysis["bg_collision_cells"] < e_coll:
        return None
    if analysis["bg_single_cells"] < 1:
        return None

    out = _render_output(inp, sources, targets)
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
