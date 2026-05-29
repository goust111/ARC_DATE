# Combo variant of dd6b8c4b (index 19)
# combo_spec: D2/3+D15/0
# Name: 非背景色均为障碍+逐步收集

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis for the fused rule
# a) Concrete behavior trace:
#    Example corridor: dest -> orange -> maroon A -> maroon B -> orange -> maroon C.
#    Original dd6b8c4b treats maroon as passable, so A/B/C are all reachable in one BFS.
#    This combo does not: only orange is flood-filled. First A is reachable, then A disappears
#    and becomes orange, which exposes B; after B disappears, the flood reaches the next orange
#    cell and finally exposes C. The candidate set therefore changes after each pick.
# b) Complexity calibration:
#    The generator uses 2-3 thin corridors plus 0-2 isolated distractors. That is enough to
#    make chained unlocking visible, but still visually clean and easy to parse by eye.
# c) Implicit assumptions broken from the original solver:
#    - `passable = {floor, 9}` is no longer valid: maroon must block flood-fill.
#    - A single distance map over the input grid is no longer sufficient: the grid changes after
#      every collected maroon, so distances must be recomputed iteratively.
#    - "There is a passable start next to the destination" is no longer guaranteed: a maroon can
#      sit directly at the opening, so the solver must allow direct-adjacent maroon targets even
#      when no orange start cell exists yet.
# d) Placement legality:
#    Corridors and rooms are carved from a wall field by fixed templates, so cells never overlap
#    the 3x3 destination, never leave the 11x11 board, and never create accidental bypass paths.
# e) Visual observability:
#    The only active colors remain the original ones. Orange marks the traversable tunnels, magenta
#    marks hard obstacles, and maroon chains visibly sit inside the tunnels they block.
# f) Visual cleanliness:
#    Tokens are only placed on corridor/room cells. Distractors live in isolated corner pockets, so
#    the central area stays readable instead of mixing many object types in the same region.
# g) Inferability:
#    The destination fill is still row-major, but the input layouts make the new rule visible
#    because some moved maroons are not initially exposed; they appear to come from behind others.
# h) Important audit result:
#    This combo changes the reachability process, but not necessarily the final distance order versus
#    the original "floor+maroon are both passable" ranking. The generator therefore measures the
#    non-degenerate effect against a one-shot floor-only ablation: collected count must increase after
#    iterative unlocking, otherwise the sample would not actually display the fused behavior.

GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR = 7
WALL = 6
DEST = 3
RED = 2
MAROON = 9
GEN_RETRIES = 400

DEST_CELLS = [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]
DEST_SET = set(DEST_CELLS)

DIR_TEMPLATES = {
    "L": [
        {"path": [(5, 3), (5, 2), (5, 1), (5, 0)], "room": [(4, 0), (6, 0)]},
        {"path": [(5, 3), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2)], "room": [(0, 2), (1, 1)]},
        {"path": [(5, 3), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)], "room": [(10, 2), (9, 1)]},
    ],
    "R": [
        {"path": [(5, 7), (5, 8), (5, 9), (5, 10)], "room": [(4, 10), (6, 10)]},
        {"path": [(5, 7), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8)], "room": [(0, 8), (1, 9)]},
        {"path": [(5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)], "room": [(10, 8), (9, 9)]},
    ],
    "U": [
        {"path": [(3, 5), (2, 5), (1, 5), (0, 5)], "room": [(0, 4), (0, 6)]},
        {"path": [(3, 5), (2, 5), (1, 5), (1, 4), (1, 3)], "room": [(0, 3), (1, 2)]},
        {"path": [(3, 5), (2, 5), (1, 5), (1, 6), (1, 7)], "room": [(0, 7), (1, 8)]},
    ],
    "D": [
        {"path": [(7, 5), (8, 5), (9, 5), (10, 5)], "room": [(10, 4), (10, 6)]},
        {"path": [(7, 5), (8, 5), (9, 5), (9, 4), (9, 3)], "room": [(10, 3), (9, 2)]},
        {"path": [(7, 5), (8, 5), (9, 5), (9, 6), (9, 7)], "room": [(10, 7), (9, 8)]},
    ],
}

DISTRACTOR_POCKETS = [[(0, 0)], [(0, 10)], [(10, 0)], [(10, 10)]]


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GRID_H and 0 <= c < GRID_W


def _dest_cells_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    centers = np.argwhere(np.asarray(grid) == RED)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                cells.append((r, c))
    cells.sort()
    return cells, set(cells)


def _maroon_candidates(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR and dist[nr, nc] < 0:
                dist[nr, nc] = 0
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    candidates = []
    for r, c in np.argwhere(grid == MAROON):
        r = int(r)
        c = int(c)
        best = None
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in dest_set:
                best = 0 if best is None else min(best, 0)
            elif 0 <= nr < h and 0 <= nc < w and dist[nr, nc] >= 0:
                cand = int(dist[nr, nc]) + 1
                best = cand if best is None else min(best, cand)
        if best is not None:
            candidates.append((int(best), r, c))
    candidates.sort()
    return candidates


def _collect_positions(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int]]:
    work = np.array(grid, dtype=int, copy=True)
    dest_cells, dest_set = _dest_cells_from_grid(work)
    max_take = len(dest_cells) if limit is None else min(int(limit), len(dest_cells))
    collected: list[tuple[int, int]] = []

    while len(collected) < max_take:
        candidates = _maroon_candidates(work, dest_cells, dest_set)
        if not candidates:
            break
        _dist, r, c = candidates[0]
        collected.append((r, c))
        work[r, c] = FLOOR

    return collected


def _one_shot_positions(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int]]:
    dest_cells, dest_set = _dest_cells_from_grid(grid)
    max_take = len(dest_cells) if limit is None else min(int(limit), len(dest_cells))
    return [(r, c) for _dist, r, c in _maroon_candidates(np.asarray(grid, dtype=int), dest_cells, dest_set)[:max_take]]


def _legacy_positions(grid: np.ndarray, limit: int | None = None) -> list[tuple[int, int]]:
    """Reference helper for the original dd6b8c4b connectivity model: floor+maroon are passable."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dest_cells, dest_set = _dest_cells_from_grid(g)
    max_take = len(dest_cells) if limit is None else min(int(limit), len(dest_cells))

    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and g[nr, nc] in (FLOOR, MAROON) and dist[nr, nc] < 0:
                dist[nr, nc] = 0
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and g[nr, nc] in (FLOOR, MAROON) and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    ranked = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(g == MAROON) if dist[int(r), int(c)] >= 0]
    ranked.sort()
    return [(r, c) for _dist, r, c in ranked[:max_take]]


def _paint_output(grid: np.ndarray, collected: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    dest_cells, _dest_set = _dest_cells_from_grid(out)
    for r, c in collected:
        out[r, c] = FLOOR
    for r, c in dest_cells[: len(collected)]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect maroon tokens into the 3x3 target one at a time, recomputing reachability after each pick.

    Input structure:
    - Orange is the only traversable background; magenta walls and maroon tokens are both solid blockers.
    - A unique red cell marks the center of the fixed 3x3 destination block whose other cells are green.

    Rule:
    - Flood-fill only through orange cells that touch the destination boundary.
    - At each step, consider maroon tokens touching that flood region (or touching the destination directly),
      take exactly the nearest one using distance / row / column ordering, erase it to orange, and repeat.
    - The collected tokens fill the destination cells in row-major order until either no token is reachable
      or the 3x3 block is full.

    Key insight:
    - Maroon tokens are no longer part of the traversable corridor; they are movable plugs. Removing a front
      token can expose new orange floor and unlock deeper tokens, so a one-shot BFS is no longer enough.
    """

    grid = np.asarray(grid, dtype=int)
    collected = _collect_positions(grid)
    return _paint_output(grid, collected)


def _carve_cells(grid: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> None:
    for r, c in cells:
        if _in_bounds(r, c) and (r, c) not in DEST_SET:
            grid[r, c] = FLOOR


def _build_base_grid() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), WALL, dtype=int)
    for r, c in DEST_CELLS:
        grid[r, c] = DEST
    grid[CENTER_R, CENTER_C] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _build_base_grid()

    active_dirs = random.sample(["L", "R", "U", "D"], random.randint(2, 3))
    layouts = {}
    carved = set()
    for d in active_dirs:
        layout = random.choice(DIR_TEMPLATES[d])
        cells = tuple(layout["path"] + layout["room"])
        if any(cell in carved for cell in cells):
            return None
        carved.update(cells)
        layouts[d] = layout
        _carve_cells(grid, layout["path"])
        _carve_cells(grid, layout["room"])

    chain_dir = random.choice(active_dirs)
    token_cells: set[tuple[int, int]] = set()

    chain_path = list(layouts[chain_dir]["path"])
    chain_room = list(layouts[chain_dir]["room"])
    start_idx = random.randint(0, 1 if len(chain_path) >= 5 else 0)
    max_chain = min(4, len(chain_path) - start_idx)
    if max_chain < 2:
        return None
    chain_len = random.randint(2, max_chain)
    chain_segment = chain_path[start_idx : start_idx + chain_len]
    token_cells.update(chain_segment)
    token_cells.add(random.choice(chain_room))

    for d in active_dirs:
        if d == chain_dir:
            continue
        room = list(layouts[d]["room"])
        count = random.randint(1, len(room))
        token_cells.update(random.sample(room, count))
        if random.random() < 0.35:
            tail = layouts[d]["path"][-1]
            if tail not in token_cells:
                token_cells.add(tail)

    reachable_cap = random.randint(4, 8)
    ordered_tokens = sorted(token_cells)
    if len(ordered_tokens) > reachable_cap:
        protected = set(chain_segment)
        protected.add(next(iter(set(chain_room) & token_cells)))
        removable = [cell for cell in ordered_tokens if cell not in protected]
        random.shuffle(removable)
        while len(token_cells) > reachable_cap and removable:
            token_cells.remove(removable.pop())

    distractor_count = random.randint(0, 2)
    for pocket in random.sample(DISTRACTOR_POCKETS, distractor_count):
        _carve_cells(grid, pocket)
        token_cells.add(pocket[0])

    for r, c in token_cells:
        grid[r, c] = MAROON

    collected = _collect_positions(grid)
    one_shot = _one_shot_positions(grid)
    if not (4 <= len(collected) <= 9):
        return None
    if len(one_shot) < 2:
        return None
    if len(collected) <= len(one_shot):
        return None

    expected = _paint_output(grid, collected)
    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
