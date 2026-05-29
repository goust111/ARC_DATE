# Variant of dd6b8c4b
# Dimension: D18 — 不可达像素的处理 (transform)
# Change: 被品红墙壁阻隔的不可达栗色像素在输出中保持不变（仍为栗色）... → 不可达像素向最近开口移动

# Design analysis
# ---
# spec:
#   reference_baseline:
#     original_dd6b8c4b: reachable maroon cells are ranked by floor BFS distance; unreachable maroons stay put.
#     source_task_88e364bc: movement is constrained by enclosing structure rather than free global motion.
#   concrete_variant_layout:
#     grid: still 11x11 with the same 3x3 green+red destination at the center.
#     walls: keep magenta as the only blocker, but regularize the generator into one connected wall network:
#       - inner ring around the destination
#       - outer frame at radius 4 from the red center
#       - 1-cell side pockets between the two, created by short wall connectors
#     openings: restricted to 8 visually obvious side-center slots:
#       - inner slots at offsets (±2, 0), (0, ±2) from the red center
#       - outer slots at offsets (±4, 0), (0, ±4) from the red center
#     blocked maroons: placed in side pockets so they are clearly unreachable by floor BFS but adjacent to the wall network.
#   behavior_trace:
#     old_rule:
#       - a maroon in the top pocket is unreachable from the destination, so it simply remains in the top pocket.
#     new_rule:
#       - the same maroon first attaches to the adjacent magenta wall cells,
#       - then follows the connected wall graph to the closest open slot,
#       - if that slot is an inner opening next to the destination, the maroon becomes reachable and can be selected into the 3x3 block,
#       - if that slot is an outer opening on the outer frame, the maroon stops there and remains visible in the output.
#     difference:
#       - D18 changes only the handling of originally unreachable maroons; reachable-source ranking and destination filling stay as in the seed task.
#   complexity_calibration:
#     - openings are limited to 8 fixed side-center positions, which keeps the motion legible in 2-4 examples.
#     - pockets are only one cell each, so each moved maroon has a single unambiguous origin.
#     - generator enforces both branches: at least one blocked maroon moves to an inner opening and at least one moves to an outer opening.
#   implicit_assumption_audit:
#     original_solver_dependencies:
#       - "unreachable maroons can be ignored for selection" is no longer safe.
#       - clearing only chosen reachable maroons is no longer sufficient because blocked maroons now relocate before ranking.
#       - BFS only on floor+maroon still works for final selection, but it must run after the wall-motion stage.
#     new_solver_dependencies:
#       - blocked maroons must touch at least one wall cell; generator guarantees this by using pocket cells.
#       - open slots must have intact flanking wall cells; generator only opens side-center slots on the inner ring / outer frame.
#       - no two blocked maroons may need the same opening; generator validates distinct nearest targets.
#   placement_legality:
#     - destination, inner ring, outer frame, pocket connectors, openings, and maroons are all placed on fixed non-overlapping cells.
#     - reachable source maroons are sampled only from floor cells that are not pockets and not opening slots.
#   visual_observability_and_cleanliness:
#     - the only new visual event is "sealed-pocket maroons jump to wall gaps".
#     - inner-gap moves are close to the square and outer-gap moves remain on the frame, so the two outcomes are easy to distinguish.
#     - no color-id-based ordering is introduced; all ranking remains spatial.

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

SIDES = ("top", "right", "bottom", "left")
SIDE_INFO = {
    "top": {
        "inner_gap": (-2, 0),
        "outer_gap": (-4, 0),
        "pocket": (-3, 0),
        "inner_flanks": [(-2, -1), (-2, 1)],
        "outer_flanks": [(-4, -1), (-4, 1)],
        "connectors": [(-3, -1), (-3, 1)],
    },
    "right": {
        "inner_gap": (0, 2),
        "outer_gap": (0, 4),
        "pocket": (0, 3),
        "inner_flanks": [(-1, 2), (1, 2)],
        "outer_flanks": [(-1, 4), (1, 4)],
        "connectors": [(-1, 3), (1, 3)],
    },
    "bottom": {
        "inner_gap": (2, 0),
        "outer_gap": (4, 0),
        "pocket": (3, 0),
        "inner_flanks": [(2, -1), (2, 1)],
        "outer_flanks": [(4, -1), (4, 1)],
        "connectors": [(3, -1), (3, 1)],
    },
    "left": {
        "inner_gap": (0, -2),
        "outer_gap": (0, -4),
        "pocket": (0, -3),
        "inner_flanks": [(-1, -2), (1, -2)],
        "outer_flanks": [(-1, -4), (1, -4)],
        "connectors": [(-1, -3), (1, -3)],
    },
}

INNER_OPEN_RANGE = (1, 2)
REACHABLE_SOURCE_RANGE = (2, 7)
GEN_RETRIES = 400


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
            if max(abs(r - cr), abs(c - cc)) != 2:
                continue
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


def _opening_slots(cr: int, cc: int) -> dict[tuple[int, int], dict[str, object]]:
    slots: dict[tuple[int, int], dict[str, object]] = {}
    for side in SIDES:
        info = SIDE_INFO[side]
        inner = _add(cr, cc, info["inner_gap"])
        outer = _add(cr, cc, info["outer_gap"])
        slots[inner] = {
            "kind": "inner",
            "side": side,
            "flanks": [_add(cr, cc, d) for d in info["inner_flanks"]],
        }
        slots[outer] = {
            "kind": "outer",
            "side": side,
            "flanks": [_add(cr, cc, d) for d in info["outer_flanks"]],
        }
    return slots


def _reachable_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]], floor: int) -> np.ndarray:
    h, w = grid.shape
    passable = {floor, MAROON}
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
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
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _wall_bfs_dist(wall_cells: set[tuple[int, int]], starts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for cell in starts:
        if cell not in wall_cells or cell in dist:
            continue
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
    slot_defs = _opening_slots(cr, cc)
    open_slots = {
        pos: info
        for pos, info in slot_defs.items()
        if 0 <= pos[0] < grid.shape[0]
        and 0 <= pos[1] < grid.shape[1]
        and int(grid[pos]) != WALL_COLOR
        and all(flank in wall_cells for flank in info["flanks"])
    }

    moved: list[dict[str, object]] = []
    reserved_targets: set[tuple[int, int]] = set()
    maroons = [(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons.sort()
    for r, c in maroons:
        if int(dist[r, c]) < 10**9:
            continue
        starts = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in wall_cells]
        if not starts:
            continue
        wall_dist = _wall_bfs_dist(wall_cells, starts)
        candidates: list[tuple[int, int, int, dict[str, object]]] = []
        for pos, info in open_slots.items():
            if pos in reserved_targets:
                continue
            flank_dists = [wall_dist[flank] for flank in info["flanks"] if flank in wall_dist]
            if not flank_dists:
                continue
            best = min(flank_dists)
            candidates.append((best, pos[0], pos[1], {"from": (r, c), "to": pos, "kind": info["kind"], "side": info["side"]}))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move blocked maroon pixels to the nearest wall opening, then fill the center block with the closest reachable maroons.

    Input structure:
    - Orange floor surrounds a fixed 3x3 destination centered on the unique red pixel, with green on the other eight cells.
    - Magenta walls form an inner ring, an outer frame, and small sealed side pockets containing some maroon pixels.

    Rule:
    - First, any maroon already reachable from the destination through floor/maroon cells stays in place.
    - Any maroon trapped behind the magenta walls instead travels along the connected wall framework to the nearest open side-center slot.
    - After that relocation, recompute reachability and move the closest reachable maroons into the 3x3 destination in row-major order, erasing them from their source cells.

    Key insight:
    - The critical change is not how reachable maroons are ranked, but that blocked maroons are no longer static distractors: the wall network itself routes them to visible openings, and inner openings can turn them into new candidates.
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
    maroons = [t for t in maroons if t[0] < 10**9]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    out = staged.copy()
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _build_layout(inner_open_sides: set[str], outer_open_sides: set[str], pocket_sides: set[str]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W):
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_cells = _inner_ring_cells(CENTER_R, CENTER_C) | _outer_frame_cells(CENTER_R, CENTER_C)
    for side in inner_open_sides:
        wall_cells.discard(_add(CENTER_R, CENTER_C, SIDE_INFO[side]["inner_gap"]))
    for side in outer_open_sides:
        wall_cells.discard(_add(CENTER_R, CENTER_C, SIDE_INFO[side]["outer_gap"]))
    for side in pocket_sides:
        for delta in SIDE_INFO[side]["connectors"]:
            wall_cells.add(_add(CENTER_R, CENTER_C, delta))

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR
    return grid


def _reachable_floor_cells(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    dist = _reachable_dist(grid, dest_cells, FLOOR_COLOR)
    cells = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == FLOOR_COLOR and int(dist[r, c]) < 10**9:
                cells.add((r, c))
    return cells


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    staged, moved = _move_blocked_maroons(np.asarray(grid, dtype=int), CENTER_R, CENTER_C, dest_cells, FLOOR_COLOR)
    dist = _reachable_dist(staged, dest_cells, FLOOR_COLOR)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(staged == MAROON)]
    maroons = [t for t in maroons if t[0] < 10**9]
    maroons.sort()
    chosen = {(r, c) for _, r, c in maroons[: min(len(dest_cells), len(maroons))]}
    return {
        "moved_total": len(moved),
        "moved_to_inner": sum(1 for rec in moved if rec["kind"] == "inner"),
        "moved_to_outer": sum(1 for rec in moved if rec["kind"] == "outer"),
        "moved_then_filled": sum(1 for rec in moved if rec["to"] in chosen),
    }


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)

    inner_open_sides = set(random.sample(SIDES, random.randint(*INNER_OPEN_RANGE)))
    blocked_sides = [side for side in SIDES if side not in inner_open_sides]
    if len(blocked_sides) < 2:
        return None

    random.shuffle(blocked_sides)
    to_dest_side = blocked_sides[0]
    stop_side = blocked_sides[1]
    pocket_sides = {to_dest_side, stop_side}
    outer_open_sides = {stop_side}

    if len(blocked_sides) >= 3 and random.random() < 0.35:
        extra_side = blocked_sides[2]
        pocket_sides.add(extra_side)
        if random.random() < 0.5:
            outer_open_sides.add(extra_side)

    base = _build_layout(inner_open_sides, outer_open_sides, pocket_sides)

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
    expected_stop = _add(CENTER_R, CENTER_C, SIDE_INFO[stop_side]["outer_gap"])
    if probe_map[stop_from]["to"] != expected_stop:
        return None
    targets = [tuple(rec["to"]) for rec in moved_probe]
    if len(set(targets)) != len(targets):
        return None

    reachable_floor = _reachable_floor_cells(base, dest_cells)
    banned = set(pocket_cells_by_side.values()) | set(_opening_slots(CENTER_R, CENTER_C).keys())
    source_pool = sorted(cell for cell in reachable_floor if cell not in banned)
    if len(source_pool) < REACHABLE_SOURCE_RANGE[0]:
        return None

    n_sources = random.randint(REACHABLE_SOURCE_RANGE[0], min(REACHABLE_SOURCE_RANGE[1], len(source_pool)))
    source_cells = random.sample(source_pool, n_sources)

    inp = base.copy()
    for cell in pocket_cells_by_side.values():
        inp[cell] = MAROON
    for r, c in source_cells:
        inp[r, c] = MAROON

    staged, moved = _move_blocked_maroons(inp, CENTER_R, CENTER_C, dest_cells, FLOOR_COLOR)
    moved_map = {tuple(rec["from"]): rec for rec in moved}
    if any(cell not in moved_map for cell in pocket_cells_by_side.values()):
        return None
    if moved_map[to_dest_from]["kind"] != "inner":
        return None
    if moved_map[stop_from]["to"] != expected_stop:
        return None

    metrics = _variant_metrics(inp)
    if metrics["moved_total"] < 2 or metrics["moved_to_inner"] < 1 or metrics["moved_to_outer"] < 1:
        return None
    if metrics["moved_then_filled"] < 1:
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
