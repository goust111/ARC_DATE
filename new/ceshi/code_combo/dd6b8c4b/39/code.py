# Combo variant of dd6b8c4b (index 39)
# combo_spec: D2/1+D18/2
# Name: 不可达像素向开口移动+墙壁有通道口

# Design analysis
# ---
# spec:
#   reference_baseline:
#     original_dd6b8c4b:
#       - reachable maroons are selected by BFS distance from the central 3x3 block.
#       - maroons cut off by magenta walls are ignored and remain where they are.
#     fused_change:
#       - D18/2 changes those blocked maroons into active movers that travel along the wall network.
#       - D2/1 changes the wall layout so the network exposes several visible corridor mouths, some 1 cell wide and some 2 cells wide.
#   concrete_fusion_layout:
#     - Keep the original 11x11 canvas and the fixed green+red destination at the center.
#     - Regularize the walls into an inner ring at Chebyshev radius 2 and an outer frame at radius 4.
#     - Put blocked maroons in 1-cell side pockets between the two rings; each pocket touches the wall graph through two short connectors.
#     - Open 2 inner wall mouths and 1-2 outer wall mouths. Each mouth stays on a side-center slot so humans can compare sides directly, but some mouths widen to 2 cells.
#   behavior_trace:
#     old_rule_example:
#       - a top-pocket maroon cannot be reached from the center because the top inner wall cell is closed.
#       - in the seed task it stays in that pocket forever.
#     fused_rule_example:
#       - the same top-pocket maroon first attaches to adjacent magenta wall cells,
#       - then follows the connected wall graph to whichever visible mouth is nearest,
#       - if that mouth is an inner opening on another side, the maroon becomes reachable and may join the central fill,
#       - if that mouth is an outer opening on its own side, it stops there and remains outside.
#     difference:
#       - the new wall mouths are not cosmetic: they change both the nearest target of blocked maroons and the final ranking pool for the center.
#   complexity_calibration:
#     - Openings are limited to the four side centers of the inner ring and outer frame, so there are only a few repeated target categories.
#     - Width only has two values, 1 or 2, which is enough to make D2 visible without creating many special cases.
#     - The generator always includes both outcomes: one blocked maroon that reaches an inner opening and one that stops at an outer opening.
#   implicit_assumption_audit:
#     original_solver_dependencies_broken:
#       - unreachable maroons can no longer be ignored before ranking.
#       - clearing only originally reachable sources is no longer sufficient because blocked maroons relocate first.
#     new_solver_dependencies:
#       - blocked maroons must touch the wall graph; the generator guarantees this with side pockets.
#       - openings must be recognizable from geometry alone, so each mouth keeps wall flanks on both sides.
#       - two blocked maroons should not compete for one target opening; the generator probes the layout and rejects collisions.
#       - pocket sides cannot have an inner opening on the same side, otherwise the pocket maroon would already be reachable.
#   placement_legality:
#     - Destination, wall rings, pocket cells, connectors, and opening cells are all fixed relative to the center and stay inside the 11x11 board.
#     - Reachable source maroons are sampled only from floor cells that are not pockets and not any potential opening cell.
#   visual_observability_and_cleanliness:
#     - The wall network is a clean square-with-frame motif, so widened mouths and moved maroons stand out immediately.
#     - Motion is always between a pocket and a side-center mouth, avoiding cluttered diagonal ambiguities.
#     - All ordering remains spatial: BFS distance, then row-major tie-breaking. No hidden color-value ordering is introduced.

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9
INF = 10**9

SIDES = ("top", "right", "bottom", "left")
SIDE_INFO = {
    "top": {
        "pocket": (-3, 0),
        "connectors": [(-3, -1), (-3, 1)],
        "inner": {
            "target": (-2, 0),
            "variants": {
                "narrow": {"segment": [(-2, 0)], "flanks": [(-2, -1), (-2, 1)], "width": 1},
                "wide_left": {"segment": [(-2, -1), (-2, 0)], "flanks": [(-2, -2), (-2, 1)], "width": 2},
                "wide_right": {"segment": [(-2, 0), (-2, 1)], "flanks": [(-2, -1), (-2, 2)], "width": 2},
            },
        },
        "outer": {
            "target": (-4, 0),
            "variants": {
                "narrow": {"segment": [(-4, 0)], "flanks": [(-4, -1), (-4, 1)], "width": 1},
                "wide_left": {"segment": [(-4, -1), (-4, 0)], "flanks": [(-4, -2), (-4, 1)], "width": 2},
                "wide_right": {"segment": [(-4, 0), (-4, 1)], "flanks": [(-4, -1), (-4, 2)], "width": 2},
            },
        },
    },
    "right": {
        "pocket": (0, 3),
        "connectors": [(-1, 3), (1, 3)],
        "inner": {
            "target": (0, 2),
            "variants": {
                "narrow": {"segment": [(0, 2)], "flanks": [(-1, 2), (1, 2)], "width": 1},
                "wide_up": {"segment": [(-1, 2), (0, 2)], "flanks": [(-2, 2), (1, 2)], "width": 2},
                "wide_down": {"segment": [(0, 2), (1, 2)], "flanks": [(-1, 2), (2, 2)], "width": 2},
            },
        },
        "outer": {
            "target": (0, 4),
            "variants": {
                "narrow": {"segment": [(0, 4)], "flanks": [(-1, 4), (1, 4)], "width": 1},
                "wide_up": {"segment": [(-1, 4), (0, 4)], "flanks": [(-2, 4), (1, 4)], "width": 2},
                "wide_down": {"segment": [(0, 4), (1, 4)], "flanks": [(-1, 4), (2, 4)], "width": 2},
            },
        },
    },
    "bottom": {
        "pocket": (3, 0),
        "connectors": [(3, -1), (3, 1)],
        "inner": {
            "target": (2, 0),
            "variants": {
                "narrow": {"segment": [(2, 0)], "flanks": [(2, -1), (2, 1)], "width": 1},
                "wide_left": {"segment": [(2, -1), (2, 0)], "flanks": [(2, -2), (2, 1)], "width": 2},
                "wide_right": {"segment": [(2, 0), (2, 1)], "flanks": [(2, -1), (2, 2)], "width": 2},
            },
        },
        "outer": {
            "target": (4, 0),
            "variants": {
                "narrow": {"segment": [(4, 0)], "flanks": [(4, -1), (4, 1)], "width": 1},
                "wide_left": {"segment": [(4, -1), (4, 0)], "flanks": [(4, -2), (4, 1)], "width": 2},
                "wide_right": {"segment": [(4, 0), (4, 1)], "flanks": [(4, -1), (4, 2)], "width": 2},
            },
        },
    },
    "left": {
        "pocket": (0, -3),
        "connectors": [(-1, -3), (1, -3)],
        "inner": {
            "target": (0, -2),
            "variants": {
                "narrow": {"segment": [(0, -2)], "flanks": [(-1, -2), (1, -2)], "width": 1},
                "wide_up": {"segment": [(-1, -2), (0, -2)], "flanks": [(-2, -2), (1, -2)], "width": 2},
                "wide_down": {"segment": [(0, -2), (1, -2)], "flanks": [(-1, -2), (2, -2)], "width": 2},
            },
        },
        "outer": {
            "target": (0, -4),
            "variants": {
                "narrow": {"segment": [(0, -4)], "flanks": [(-1, -4), (1, -4)], "width": 1},
                "wide_up": {"segment": [(-1, -4), (0, -4)], "flanks": [(-2, -4), (1, -4)], "width": 2},
                "wide_down": {"segment": [(0, -4), (1, -4)], "flanks": [(-1, -4), (2, -4)], "width": 2},
            },
        },
    },
}

REACHABLE_SOURCE_RANGE = (2, 6)
SOURCE_POOL_CAP = 12
GEN_RETRIES = 500


def _add(center_r: int, center_c: int, delta: tuple[int, int]) -> tuple[int, int]:
    dr, dc = delta
    return center_r + dr, center_c + dc


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _inner_ring_cells(cr: int, cc: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(cr - 2, cr + 3):
        for c in range(cc - 2, cc + 3):
            if max(abs(r - cr), abs(c - cc)) == 2:
                cells.add((r, c))
    return cells


def _outer_frame_cells(cr: int, cc: int) -> set[tuple[int, int]]:
    d = 4
    cells = set()
    for r in range(cr - d, cr + d + 1):
        cells.add((r, cc - d))
        cells.add((r, cc + d))
    for c in range(cc - d, cc + d + 1):
        cells.add((cr - d, c))
        cells.add((cr + d, c))
    return cells


def _opening_variants(side: str, kind: str) -> dict[str, dict[str, object]]:
    return SIDE_INFO[side][kind]["variants"]


def _pick_variant(side: str, kind: str, prefer_wide: bool) -> str:
    variants = _opening_variants(side, kind)
    if prefer_wide:
        wide = [name for name, spec in variants.items() if int(spec["width"]) == 2]
        return random.choice(wide)
    return random.choice(list(variants))


def _slot_cells(
    cr: int,
    cc: int,
    side: str,
    kind: str,
    variant_name: str,
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], int]:
    info = SIDE_INFO[side][kind]
    variant = info["variants"][variant_name]
    target = _add(cr, cc, info["target"])
    segment = [_add(cr, cc, delta) for delta in variant["segment"]]
    flanks = [_add(cr, cc, delta) for delta in variant["flanks"]]
    return target, segment, flanks, int(variant["width"])


def _all_slot_cells(cr: int, cc: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for side in SIDES:
        for kind in ("inner", "outer"):
            for variant_name in _opening_variants(side, kind):
                target, segment, _, _ = _slot_cells(cr, cc, side, kind, variant_name)
                cells.add(target)
                cells.update(segment)
    return cells


def _detect_open_slots(
    grid: np.ndarray,
    cr: int,
    cc: int,
) -> dict[tuple[int, int], dict[str, object]]:
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == WALL_COLOR)}
    slots: dict[tuple[int, int], dict[str, object]] = {}
    for side in SIDES:
        for kind in ("inner", "outer"):
            for variant_name in _opening_variants(side, kind):
                target, segment, flanks, width = _slot_cells(cr, cc, side, kind, variant_name)
                if not (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]):
                    continue
                if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in segment + flanks):
                    continue
                if all(int(grid[r, c]) != WALL_COLOR for r, c in segment) and all(cell in wall_cells for cell in flanks):
                    slots[target] = {
                        "kind": kind,
                        "side": side,
                        "variant": variant_name,
                        "width": width,
                        "target": target,
                        "segment": segment,
                        "flanks": flanks,
                    }
                    break
    return slots


def _reachable_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]], floor: int) -> np.ndarray:
    h, w = grid.shape
    passable = {floor, MAROON}
    dist = np.full((h, w), INF, dtype=int)
    queue: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and (nr, nc) not in seen:
                seen.add((nr, nc))
                dist[nr, nc] = 0
                queue.append((nr, nc))
    assert queue, "destination must have at least one passable adjacent cell"

    qi = 0
    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and int(dist[nr, nc]) == INF:
                dist[nr, nc] = int(dist[r, c]) + 1
                queue.append((nr, nc))
    return dist


def _wall_bfs_dist(wall_cells: set[tuple[int, int]], starts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for cell in starts:
        if cell in wall_cells and cell not in dist:
            dist[cell] = 0
            q.append(cell)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in wall_cells and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _move_blocked_maroons(
    grid: np.ndarray,
    cr: int,
    cc: int,
    dest_cells: list[tuple[int, int]],
    floor: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    dist = _reachable_dist(grid, dest_cells, floor)
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == WALL_COLOR)}
    open_slots = _detect_open_slots(grid, cr, cc)

    moved: list[dict[str, object]] = []
    reserved_targets: set[tuple[int, int]] = set()
    maroons = [(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons.sort()
    for r, c in maroons:
        if int(dist[r, c]) < INF:
            continue
        starts = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in wall_cells]
        if not starts:
            continue
        wall_dist = _wall_bfs_dist(wall_cells, starts)
        candidates: list[tuple[int, int, int, dict[str, object]]] = []
        for target, info in open_slots.items():
            if target in reserved_targets:
                continue
            flank_dists = [int(wall_dist[cell]) for cell in info["flanks"] if cell in wall_dist]
            if not flank_dists:
                continue
            best = min(flank_dists)
            candidates.append(
                (
                    best,
                    target[0],
                    target[1],
                    {
                        "from": (r, c),
                        "to": target,
                        "kind": info["kind"],
                        "side": info["side"],
                        "variant": info["variant"],
                        "width": info["width"],
                    },
                )
            )
        if not candidates:
            continue
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        record = candidates[0][3]
        moved.append(record)
        reserved_targets.add(record["to"])

    out = grid.copy()
    for record in moved:
        sr, sc = record["from"]
        out[sr, sc] = floor
    for record in moved:
        tr, tc = record["to"]
        out[tr, tc] = MAROON
    return out, moved


def _reachable_floor_cells(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    dist = _reachable_dist(grid, dest_cells, FLOOR_COLOR)
    cells = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == FLOOR_COLOR and int(dist[r, c]) < INF:
                cells.add((r, c))
    return cells


def _build_layout(
    inner_openings: dict[str, str],
    outer_openings: dict[str, str],
    pocket_sides: set[str],
) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W):
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_cells = _inner_ring_cells(CENTER_R, CENTER_C) | _outer_frame_cells(CENTER_R, CENTER_C)
    for side in pocket_sides:
        for delta in SIDE_INFO[side]["connectors"]:
            wall_cells.add(_add(CENTER_R, CENTER_C, delta))

    for side, variant_name in inner_openings.items():
        _, segment, _, _ = _slot_cells(CENTER_R, CENTER_C, side, "inner", variant_name)
        for cell in segment:
            wall_cells.discard(cell)
    for side, variant_name in outer_openings.items():
        _, segment, _, _ = _slot_cells(CENTER_R, CENTER_C, side, "outer", variant_name)
        for cell in segment:
            wall_cells.discard(cell)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR
    return grid


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    open_slots = _detect_open_slots(np.asarray(grid, dtype=int), CENTER_R, CENTER_C)
    staged, moved = _move_blocked_maroons(np.asarray(grid, dtype=int), CENTER_R, CENTER_C, dest_cells, FLOOR_COLOR)
    dist = _reachable_dist(staged, dest_cells, FLOOR_COLOR)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(staged == MAROON)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()
    chosen = {(r, c) for _, r, c in maroons[: min(len(dest_cells), len(maroons))]}
    return {
        "inner_open_count": sum(1 for info in open_slots.values() if info["kind"] == "inner"),
        "outer_open_count": sum(1 for info in open_slots.values() if info["kind"] == "outer"),
        "wide_open_count": sum(1 for info in open_slots.values() if int(info["width"]) == 2),
        "moved_total": len(moved),
        "moved_to_inner": sum(1 for rec in moved if rec["kind"] == "inner"),
        "moved_to_outer": sum(1 for rec in moved if rec["kind"] == "outer"),
        "moved_to_wide": sum(1 for rec in moved if int(rec["width"]) == 2),
        "moved_then_reachable": sum(1 for rec in moved if int(dist[rec["to"]]) < INF),
        "moved_then_filled": sum(1 for rec in moved if rec["to"] in chosen),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route blocked maroon pixels to the nearest wall mouth, then fill the center block with the closest reachable maroons.

    Input structure:
    - Orange floor surrounds a fixed 3x3 destination centered on the unique red cell, with green on the other eight cells.
    - Magenta walls form an inner ring and an outer frame, with visible side openings of width 1 or 2 and a few maroon pixels trapped in side pockets.

    Rule:
    - First, flood-fill reachability from the destination through floor and maroon cells.
    - Any maroon that is still blocked travels along adjacent magenta wall cells to the nearest open wall mouth; inner mouths make it newly reachable, outer mouths only relocate it.
    - After those relocations, recompute reachability, rank reachable maroons by distance then row-major tie-break, and move the closest ones into the 3x3 destination in row-major order.

    Key insight:
    - The wall openings matter twice: they are both pass-through corridors for the center flood-fill and the landing targets for previously unreachable maroons, so extra or wider mouths can change which maroons ultimately enter the block.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    assert (h, w) == (GRID_H, GRID_W), "grid must stay 11x11"

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])
    dest_cells = _destination_cells(cr, cc, h, w)

    staged, _ = _move_blocked_maroons(grid, cr, cc, dest_cells, FLOOR_COLOR)
    dist = _reachable_dist(staged, dest_cells, FLOOR_COLOR)

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(staged == MAROON)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    out = staged.copy()
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)

    inner_open_sides = set(random.sample(SIDES, 2))
    blocked_sides = [side for side in SIDES if side not in inner_open_sides]
    random.shuffle(blocked_sides)
    to_dest_side, stop_side = blocked_sides
    pocket_sides = {to_dest_side, stop_side}

    inner_openings: dict[str, str] = {}
    wide_inner_side = random.choice(list(inner_open_sides))
    for side in inner_open_sides:
        inner_openings[side] = _pick_variant(side, "inner", prefer_wide=(side == wide_inner_side))

    outer_openings: dict[str, str] = {
        stop_side: _pick_variant(stop_side, "outer", prefer_wide=True),
    }
    if random.random() < 0.5:
        extra_outer_side = random.choice(list(inner_open_sides))
        outer_openings[extra_outer_side] = _pick_variant(extra_outer_side, "outer", prefer_wide=(random.random() < 0.5))

    base = _build_layout(inner_openings, outer_openings, pocket_sides)

    probe = base.copy()
    pocket_cells_by_side = {side: _add(CENTER_R, CENTER_C, SIDE_INFO[side]["pocket"]) for side in pocket_sides}
    for cell in pocket_cells_by_side.values():
        probe[cell] = MAROON

    staged_probe, moved_probe = _move_blocked_maroons(probe, CENTER_R, CENTER_C, dest_cells, FLOOR_COLOR)
    if len(moved_probe) != len(pocket_sides):
        return None
    probe_map = {tuple(rec["from"]): rec for rec in moved_probe}
    to_dest_from = pocket_cells_by_side[to_dest_side]
    stop_from = pocket_cells_by_side[stop_side]
    if to_dest_from not in probe_map or stop_from not in probe_map:
        return None
    if probe_map[to_dest_from]["kind"] != "inner":
        return None
    if probe_map[stop_from]["kind"] != "outer":
        return None
    expected_stop = _add(CENTER_R, CENTER_C, SIDE_INFO[stop_side]["outer"]["target"])
    if probe_map[stop_from]["to"] != expected_stop:
        return None
    if len({tuple(rec["to"]) for rec in moved_probe}) != len(moved_probe):
        return None

    metrics_probe = _variant_metrics(probe)
    if metrics_probe["inner_open_count"] < 2 or metrics_probe["outer_open_count"] < 1:
        return None
    if metrics_probe["wide_open_count"] < 1 or metrics_probe["moved_to_wide"] < 1:
        return None
    if metrics_probe["moved_then_reachable"] < 1:
        return None

    reachable_floor = _reachable_floor_cells(base, dest_cells)
    dist_base = _reachable_dist(base, dest_cells, FLOOR_COLOR)
    banned = _all_slot_cells(CENTER_R, CENTER_C) | set(pocket_cells_by_side.values())
    source_pool = [cell for cell in reachable_floor if cell not in banned]
    source_pool.sort(key=lambda cell: (-int(dist_base[cell]), cell[0], cell[1]))
    if len(source_pool) < REACHABLE_SOURCE_RANGE[0]:
        return None

    n_sources = random.randint(REACHABLE_SOURCE_RANGE[0], min(REACHABLE_SOURCE_RANGE[1], len(source_pool)))
    biased_pool = source_pool[: max(n_sources, min(len(source_pool), SOURCE_POOL_CAP))]
    source_cells = random.sample(biased_pool, n_sources)

    inp = base.copy()
    for cell in pocket_cells_by_side.values():
        inp[cell] = MAROON
    for r, c in source_cells:
        inp[r, c] = MAROON

    metrics = _variant_metrics(inp)
    if metrics["moved_total"] != len(pocket_sides):
        return None
    if metrics["moved_to_inner"] < 1 or metrics["moved_to_outer"] < 1:
        return None
    if metrics["moved_to_wide"] < 1:
        return None
    if metrics["moved_then_reachable"] < 1 or metrics["moved_then_filled"] < 1:
        return None

    try:
        _ = main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
