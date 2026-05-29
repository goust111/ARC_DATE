# Combo variant of dd6b8c4b (index 51)
# combo_spec: D14/2+D18/2
# Name: 框架与方块分离+不可达像素向开口移动

# Design analysis
# ---
# a) Concrete behavior trace
# - Baseline dd6b8c4b: the magenta barrier hugs the 3x3 green/red destination. Reachable maroons are found by
#   floor BFS from cells immediately outside the block; maroons trapped behind the wall are ignored and stay put.
# - D14/2 change: the magenta structure is no longer the tight ring around the block. Here it becomes a detached
#   border frame on the grid perimeter, while the destination block stays fixed at the center. The frame and the
#   block are separated by a wide orange field, so the wall is visually an independent structure rather than the
#   block's immediate boundary.
# - D18/2 change: blocked maroons are not static distractors. Each trapped maroon sits in a sealed 1-cell pocket
#   near the detached frame. It first "attaches" to neighboring wall cells, walks along the connected wall graph,
#   and exits at the nearest side-center opening on the border frame. Only after this wall-guided relocation do we
#   recompute which maroons are reachable from the center block and fill the block by distance.
# - Example trace: a maroon in the left-upper pocket at (3,1) is surrounded by wall on all four sides, so the
#   baseline solver cannot reach it. In this combo variant, if the left side is closed and the top border opening
#   is open, the maroon follows the frame around the top-left corner to the top opening at (0,5). From that new
#   floor position it becomes reachable through the large central orange space and may then be selected into the
#   3x3 block. The visible difference is both the detached frame geometry and the extra relocation step.
#
# b) Complexity calibration
# - The layout uses only four candidate openings (the side centers of the border frame) and eight pocket templates.
#   This keeps the movement rule legible while still allowing multiple route shapes.
# - Pockets are single-cell chambers, so each moved maroon has a unique origin. Openings are one cell wide and are
#   visually special because their flanking border-frame cells remain magenta.
# - The generator enforces 2-3 moving maroons, at least one moved maroon that later fills the center block, at
#   least one moved maroon that remains visible at an opening, and a wall-path length threshold. This prevents the
#   combo from collapsing into "same as original, but farther away".
#
# c) Implicit assumption audit against the original solver
# - Original assumption: walls are adjacent to the destination, so BFS can infer the relevant ring locally.
#   Broken by D14/2. The new solver must identify the fixed 3x3 destination independently of any nearby wall ring.
# - Original assumption: unreachable maroons can be ignored forever. Broken by D18/2. Now they may become reachable
#   after wall motion, so selection must happen after a relocation stage.
# - Original assumption: clearing only the chosen reachable source cells is sufficient. Broken because moved maroons
#   must first be removed from their pocket cells and written to opening cells before ranking.
# - New assumptions introduced here:
#   1. Every trapped maroon is adjacent to at least one wall cell.
#   2. Every usable opening is one of the four side-center border cells and keeps both flanking border cells intact.
#   3. Different trapped maroons are assigned distinct nearest openings.
#   The generator explicitly constructs and validates all three.
#
# d) Placement legality
# - Grid size remains 11x11. The destination stays centered at (5,5), so the eight green cells and one red center
#   never overlap with the frame or pockets.
# - The border frame uses only perimeter cells. Pocket walls use small local templates inside the perimeter, and
#   pocket cells themselves are disjoint from the destination, openings, and sampled reachable source cells.
# - Reachable source maroons are sampled only from orange floor cells that are not pockets and not opening slots.
#
# e) Visual observability
# - Across 2-4 examples, a human can see the large orange gap between the center block and the magenta frame, which
#   is the D14 part. They can also see maroons disappear from sealed side pockets and reappear on border openings,
#   which is the D18 part.
# - The ranking rule remains purely spatial: BFS distance to the center block, then row-major destination filling.
#   No hidden color arithmetic or non-visual IDs are introduced.
#
# f) Visual cleanliness
# - The scene is separated into three readable zones: the central destination block, the detached border frame, and
#   a handful of isolated pocket/source maroons. No region carries multiple kinds of markup on top of each other.
# - Because pockets are tiny and sparse, the output does not become cluttered even when one moved maroon remains on
#   the frame while other maroons fill the center block.
#
# g) Fusion-behavior check
# - A sample is accepted only if the combo logic changes the outcome relative to the baseline "no relocation"
#   behavior, if at least two blocked maroons actually move along the frame, and if the moved maroons exhibit both
#   downstream fates: one contributes to the fill and one stays at an opening.

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
GEN_RETRIES = 600

SIDE_ORDER = ("top", "right", "bottom", "left")
OPENING_INFO = {
    "top": {"pos": (0, 5), "flanks": ((0, 4), (0, 6))},
    "right": {"pos": (5, 10), "flanks": ((4, 10), (6, 10))},
    "bottom": {"pos": (10, 5), "flanks": ((10, 4), (10, 6))},
    "left": {"pos": (5, 0), "flanks": ((4, 0), (6, 0))},
}

POCKET_INFO = {
    "top_left": {"cell": (1, 3), "walls": ((0, 3), (1, 2), (1, 4), (2, 3)), "side": "top"},
    "top_right": {"cell": (1, 7), "walls": ((0, 7), (1, 6), (1, 8), (2, 7)), "side": "top"},
    "right_top": {"cell": (3, 9), "walls": ((3, 10), (2, 9), (4, 9), (3, 8)), "side": "right"},
    "right_bottom": {"cell": (7, 9), "walls": ((7, 10), (6, 9), (8, 9), (7, 8)), "side": "right"},
    "bottom_left": {"cell": (9, 3), "walls": ((10, 3), (9, 2), (9, 4), (8, 3)), "side": "bottom"},
    "bottom_right": {"cell": (9, 7), "walls": ((10, 7), (9, 6), (9, 8), (8, 7)), "side": "bottom"},
    "left_top": {"cell": (3, 1), "walls": ((3, 0), (2, 1), (4, 1), (3, 2)), "side": "left"},
    "left_bottom": {"cell": (7, 1), "walls": ((7, 0), (6, 1), (8, 1), (7, 2)), "side": "left"},
}

ROUTES = (
    {"name": "top_from_left", "pocket": "left_top", "source_side": "left", "target_side": "top"},
    {"name": "top_from_right", "pocket": "right_top", "source_side": "right", "target_side": "top"},
    {"name": "right_from_top", "pocket": "top_right", "source_side": "top", "target_side": "right"},
    {"name": "right_from_bottom", "pocket": "bottom_right", "source_side": "bottom", "target_side": "right"},
    {"name": "bottom_from_left", "pocket": "left_bottom", "source_side": "left", "target_side": "bottom"},
    {"name": "bottom_from_right", "pocket": "right_bottom", "source_side": "right", "target_side": "bottom"},
    {"name": "left_from_top", "pocket": "top_left", "source_side": "top", "target_side": "left"},
    {"name": "left_from_bottom", "pocket": "bottom_left", "source_side": "bottom", "target_side": "left"},
)


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


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
    assert queue, "destination must have at least one adjacent floor cell"

    qi = 0
    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _wall_bfs_dist(
    wall_cells: set[tuple[int, int]],
    starts: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
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


def _open_slots(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    slots: dict[tuple[int, int], dict[str, object]] = {}
    for side in SIDE_ORDER:
        pos = OPENING_INFO[side]["pos"]
        flanks = OPENING_INFO[side]["flanks"]
        if int(grid[pos]) == WALL_COLOR:
            continue
        if not all(int(grid[fr, fc]) == WALL_COLOR for fr, fc in flanks):
            continue
        slots[pos] = {"side": side, "flanks": flanks}
    return slots


def _move_blocked_maroons(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    floor: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    dist = _reachable_dist(grid, dest_cells, floor)
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == WALL_COLOR)}
    open_slots = _open_slots(grid)

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
        for pos, info in open_slots.items():
            if pos in reserved_targets:
                continue
            flank_hits = [wall_dist[flank] for flank in info["flanks"] if flank in wall_dist]
            if not flank_hits:
                continue
            best = min(flank_hits)
            candidates.append(
                (
                    best,
                    pos[0],
                    pos[1],
                    {"from": (r, c), "to": pos, "side": info["side"], "wall_steps": best},
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


def _select_reachable_maroons(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    dist = _reachable_dist(grid, dest_cells, FLOOR_COLOR)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()
    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}
    return maroons, chosen


def _baseline_fill_only(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> np.ndarray:
    maroons, chosen = _select_reachable_maroons(grid, dest_cells)
    _ = maroons
    out = np.asarray(grid, dtype=int).copy()
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route trapped maroons along a detached border frame, then use the closest reachable maroons to fill the center block.

    Input structure:
    - The destination is the usual 3x3 block centered on the unique red pixel, with green on the other eight cells.
    - Magenta no longer hugs that block: it forms a detached border frame on the perimeter, leaving a wide orange gap.
    - Some maroons are ordinary reachable floor tokens, while others sit inside tiny sealed pockets attached to the frame.

    Rule:
    - First, detect maroons that are unreachable from the center through orange/maroon floor travel.
    - Each such blocked maroon moves along the connected magenta frame to the nearest valid side-center opening.
    - After all blocked maroons relocate, recompute reachability and move the closest reachable maroons into the 3x3 block in row-major order, erasing the selected source positions.

    Key insight:
    - The detached frame is not the destination boundary anymore; it is an independent transport network. The puzzle now requires two stages: wall-guided relocation from sealed pockets, then the original distance-based fill from the center.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    assert (h, w) == (GRID_H, GRID_W), "grid must stay 11x11"

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])
    dest_cells = _destination_cells(cr, cc, h, w)

    staged, _ = _move_blocked_maroons(grid, dest_cells, FLOOR_COLOR)
    _maroons, chosen = _select_reachable_maroons(staged, dest_cells)

    out = staged.copy()
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = MAROON
    return out


def _build_base_grid(open_sides: set[str], pocket_names: list[str]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_cells: set[tuple[int, int]] = set()
    for r in range(GRID_H):
        for c in range(GRID_W):
            if r not in (0, GRID_H - 1) and c not in (0, GRID_W - 1):
                continue
            wall_cells.add((r, c))
    for side in open_sides:
        wall_cells.discard(OPENING_INFO[side]["pos"])

    for name in pocket_names:
        for cell in POCKET_INFO[name]["walls"]:
            wall_cells.add(cell)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR
    return grid


def _reachable_floor_cells(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    dist = _reachable_dist(grid, dest_cells, FLOOR_COLOR)
    cells = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == FLOOR_COLOR and int(dist[r, c]) < INF:
                cells.add((r, c))
    return cells


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    staged, moved = _move_blocked_maroons(np.asarray(grid, dtype=int), dest_cells, FLOOR_COLOR)
    _maroons, chosen = _select_reachable_maroons(staged, dest_cells)
    combo_out = main(grid)
    base_out = _baseline_fill_only(np.asarray(grid, dtype=int), dest_cells)
    return {
        "moved_total": len(moved),
        "moved_selected": sum(1 for rec in moved if rec["to"] in chosen),
        "moved_left_visible": sum(1 for rec in moved if rec["to"] not in chosen),
        "max_wall_steps": max((int(rec["wall_steps"]) for rec in moved), default=0),
        "baseline_diff": int(not np.array_equal(combo_out, base_out)),
    }


def _candidate_routes(closed_sides: set[str], open_sides: set[str]) -> list[dict[str, str]]:
    return [
        route
        for route in ROUTES
        if route["source_side"] in closed_sides and route["target_side"] in open_sides
    ]


def _choose_routes(closed_sides: set[str], open_sides: set[str]) -> list[dict[str, str]] | None:
    candidates = _candidate_routes(closed_sides, open_sides)
    by_target: dict[str, list[dict[str, str]]] = {}
    for route in candidates:
        by_target.setdefault(route["target_side"], []).append(route)
    viable_targets = [side for side in SIDE_ORDER if side in by_target]
    if len(viable_targets) < 2:
        return None

    target_count = random.randint(2, min(3, len(viable_targets)))
    chosen_targets = random.sample(viable_targets, target_count)
    routes = [random.choice(by_target[target]) for target in chosen_targets]
    return routes


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)

    closed_count = random.choice((1, 2))
    closed_sides = set(random.sample(SIDE_ORDER, closed_count))
    open_sides = set(SIDE_ORDER) - closed_sides
    if len(open_sides) < 2:
        return None

    routes = _choose_routes(closed_sides, open_sides)
    if routes is None:
        return None

    pocket_names = [route["pocket"] for route in routes]
    if len(set(pocket_names)) != len(pocket_names):
        return None

    base = _build_base_grid(open_sides, pocket_names)

    probe = base.copy()
    for name in pocket_names:
        pr, pc = POCKET_INFO[name]["cell"]
        probe[pr, pc] = MAROON
    staged_probe, moved_probe = _move_blocked_maroons(probe, dest_cells, FLOOR_COLOR)
    _ = staged_probe
    if len(moved_probe) != len(pocket_names):
        return None

    open_by_side = {side: OPENING_INFO[side]["pos"] for side in open_sides}
    probe_map = {tuple(rec["from"]): rec for rec in moved_probe}
    for route in routes:
        cell = POCKET_INFO[route["pocket"]]["cell"]
        if cell not in probe_map:
            return None
        if tuple(probe_map[cell]["to"]) != tuple(open_by_side[route["target_side"]]):
            return None
    if len({tuple(rec["to"]) for rec in moved_probe}) != len(moved_probe):
        return None
    if max(int(rec["wall_steps"]) for rec in moved_probe) < 3:
        return None

    reachable_floor = _reachable_floor_cells(base, dest_cells)
    banned = set(OPENING_INFO[side]["pos"] for side in SIDE_ORDER)
    banned |= {POCKET_INFO[name]["cell"] for name in POCKET_INFO}
    source_pool = [cell for cell in reachable_floor if cell not in banned]
    if len(source_pool) < 10:
        return None

    dist = _reachable_dist(base, dest_cells, FLOOR_COLOR)
    source_pool.sort(key=lambda rc: (int(dist[rc]), rc[0], rc[1]))
    near_pool = source_pool[: min(16, len(source_pool))]
    if len(near_pool) < 7:
        return None
    source_count = random.choice((7, 8))
    source_cells = random.sample(near_pool, source_count)

    inp = base.copy()
    for name in pocket_names:
        pr, pc = POCKET_INFO[name]["cell"]
        inp[pr, pc] = MAROON
    for r, c in source_cells:
        inp[r, c] = MAROON

    metrics = _variant_metrics(inp)
    if metrics["moved_total"] < 2:
        return None
    if metrics["moved_selected"] < 1:
        return None
    if metrics["moved_left_visible"] < 1:
        return None
    if metrics["max_wall_steps"] < 3:
        return None
    if metrics["baseline_diff"] != 1:
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
    raise RuntimeError(f"failed to generate valid combo sample after {GEN_RETRIES} attempts")
