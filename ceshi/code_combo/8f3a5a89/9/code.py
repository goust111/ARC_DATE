# Combo variant of 8f3a5a89 (index 9)
# combo_spec: D1/2+D11/2
# Name: 通道宽度约束洪填+迷宫墙壁

from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization / behavior trace
#    - The original task floods single cells from the corner marker, so any
#      1-cell slit is traversable. Structural blue components merely partition
#      space, often as a few edge-attached blocks or lines.
#    - This combo replaces that layout with one connected blue maze wall
#      network. The open space is a tree of rooms joined by doorways. Some
#      doorways are 2 cells wide, others are only 1 cell wide.
#    - Example trace: marker at bottom-left enters the maze vestibule. Under
#      plain flood, the search passes through both 2-wide and 1-wide doors, so
#      it reaches every room. Under the fused rule, reachability is computed by
#      moving a 2x2 probe; a 1-wide doorway is therefore blocked, so any branch
#      behind that chokepoint becomes unreachable. Orange is painted only along
#      the inner wall of the width-reachable sub-maze, and blue decorations in
#      the blocked branch are erased.
#
# b) Complexity calibration
#    - The maze is still simple enough to parse visually: one connected wall
#      component, orthogonal rooms, and a small number of narrow chokepoints.
#    - The generator uses only 2 room-grid sizes (2x2, 2x3, 3x2, 3x3) and only
#      2 doorway widths (1 or 2), so the key concept repeats across samples
#      instead of exploding into too many cases.
#    - At least one entire room-subtree is plain-reachable but width-blocked,
#      which makes the fusion effect visible rather than a subtle corner case.
#
# c) Implicit-assumption review of the original main()
#    - Original reachability assumed cell-wise 4-connectivity. That assumption
#      is invalid here and must be replaced by 2x2-placement reachability.
#    - Original structural-wall rediscovery depends on the first reach set. In
#      this combo, that set must already respect width constraints; otherwise a
#      narrow doorway would be mistaken for a real passage and the maze would
#      collapse back to the original behavior.
#    - Decoration handling still assumes non-structural blue pixels can be
#      traversed during the second pass. The generator therefore places hidden
#      decorations as isolated interior blobs that never 4-touch the maze wall,
#      so they remain non-structural and are safely erased when unreachable.
#
# d) Placement legality
#    - The maze is rendered from non-overlapping rectangular rooms and one-cell
#      wall bands; all coordinates are derived from the room lattice, so there
#      is no overlap or out-of-bounds placement.
#    - The bottom-left corner contains a carved vestibule large enough to host a
#      valid 2x2 seed placement covering the marker, guaranteeing width search
#      can actually start.
#    - Hidden decorations are placed only on interior room cells of blocked
#      branches, so they do not touch the border and do not merge into the
#      structural wall component.
#
# e) Visual observability
#    - A human can compare 1-cell and 2-cell doors directly in the samples.
#      Rooms behind 1-cell doors remain visible in the input but lose both their
#      orange boundary and their blue decorations in the output.
#    - Because the blocked region is an entire maze branch rather than a tiny
#      slit artifact, the width constraint is inferable from a few examples.
#
# f) Visual cleanliness
#    - The scene uses only four semantic layers: azure bg, one connected blue
#      maze wall, one magenta marker, and orange inner-border paint.
#    - Decorations live at room interiors, away from walls and doorways, so they
#      do not visually merge with the structural maze.
#
# g) Change inferability
#    - No extra color mapping is introduced. Blue always denotes wall/decor,
#      magenta is the start, azure is empty space, and orange is the reachable
#      inner boundary. The new inference burden is purely geometric: doorway
#      width decides reachability.
#
# New solver assumptions introduced here and enforced by the generator:
#    - The start corner participates in at least one feasible 2x2 passable
#      placement.
#    - All structural walls belong to one border-touching 4-connected maze
#      component adjacent to the width-reachable side.
#    - Every generated sample has a non-empty region that plain flood reaches
#      but width-constrained flood does not, and this changes the final output.
# ---------------------------------------------------------------------------


BG = 8
WALL = 1
MARKER = 6
FILL = 7

ROOM = 5
STEP = ROOM + 1

ROOM_ROWS_CHOICES = (2, 3)
ROOM_COLS_CHOICES = (2, 3)
REACH_RATIO_RANGE = (0.22, 0.82)
DIFF_RATIO_RANGE = (0.08, 0.55)
HIDDEN_DECOR_RANGE = (1, 3)
VISIBLE_DECOR_PROB = 0.45
GEN_ATTEMPTS = 500


def _plain_reach_mask(start: tuple[int, int], passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    sr, sc = start
    out = np.zeros((h, w), dtype=bool)
    if not passable[sr, sc]:
        return out
    out[sr, sc] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _width_reach_mask(start: tuple[int, int], passable: np.ndarray) -> np.ndarray:
    """
    Width>=2 reachability modeled as movement of a 2x2 passable probe.

    A cell is reachable iff it belongs to some reachable 2x2 placement. If the
    marker corner has no feasible seed placement, only the marker cell remains
    reachable.
    """
    h, w = passable.shape
    sr, sc = start
    out = np.zeros((h, w), dtype=bool)
    if not (0 <= sr < h and 0 <= sc < w) or not passable[sr, sc]:
        return out
    out[sr, sc] = True
    if h < 2 or w < 2:
        return out

    feasible = (
        passable[:-1, :-1]
        & passable[1:, :-1]
        & passable[:-1, 1:]
        & passable[1:, 1:]
    )
    ph, pw = feasible.shape

    seeds: list[tuple[int, int]] = []
    for r0 in (sr - 1, sr):
        for c0 in (sc - 1, sc):
            if 0 <= r0 < ph and 0 <= c0 < pw and feasible[r0, c0]:
                seeds.append((r0, c0))
    if not seeds:
        return out

    place_reach = np.zeros((ph, pw), dtype=bool)
    q = deque()
    for seed in sorted(set(seeds)):
        place_reach[seed] = True
        q.append(seed)

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and feasible[nr, nc] and not place_reach[nr, nc]:
                place_reach[nr, nc] = True
                q.append((nr, nc))

    for r0, c0 in np.argwhere(place_reach):
        out[r0 : r0 + 2, c0 : c0 + 2] = True
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _solve_core(grid: np.ndarray, *, use_width_constraint: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach_fn = _width_reach_mask if use_width_constraint else _plain_reach_mask

    passable0 = (g == bg) | (g == marker_col)
    reach0 = reach_fn((mr, mc), passable0)

    walls_mask = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        touches_border = bool(
            np.any(comp_arr[:, 0] == 0)
            or np.any(comp_arr[:, 0] == h - 1)
            or np.any(comp_arr[:, 1] == 0)
            or np.any(comp_arr[:, 1] == w - 1)
        )
        if not touches_border:
            continue
        adjacent_to_reach0 = any(
            0 <= nr < h and 0 <= nc < w and reach0[nr, nc]
            for r, c in comp_arr
            for nr, nc in neighbors4(int(r), int(c))
        )
        if adjacent_to_reach0:
            walls_mask[comp_arr[:, 0], comp_arr[:, 1]] = True

    passable = (g == bg) | (g == marker_col) | ((g == wall_col) & ~walls_mask)
    reach = reach_fn((mr, mc), passable)

    out = g.copy()
    out[(out == wall_col) & (~walls_mask) & (~reach)] = bg

    for r, c in np.argwhere(reach):
        r = int(r)
        c = int(c)
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = FILL
            continue
        if any(
            0 <= nr < h and 0 <= nc < w and walls_mask[nr, nc]
            for nr, nc in neighbors8(r, c)
        ):
            out[r, c] = FILL
            continue
        if any(
            not (0 <= nr < h and 0 <= nc < w and reach[nr, nc])
            for nr, nc in neighbors4(r, c)
        ):
            out[r, c] = FILL

    return out, reach, walls_mask


def _plain_main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, use_width_constraint=False)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the inner wall of the maze region reachable from the corner marker.

    Input structure:
    - Azure is the dominant background, blue pixels form one connected maze-wall
      network, and a unique magenta corner pixel marks the starting side.
    - The maze consists of rooms and corridors; some doors are two cells wide,
      while others are only one cell wide.
    - Small interior blue blobs may appear as decorations inside maze rooms.

    Rule:
    - Reconstruct the structural maze wall from border-touching blue components,
      but determine reachability from the marker using a width-constrained flood:
      motion is allowed only where a 2x2 passable probe can move.
    - A one-cell doorway therefore blocks access even though ordinary cell-wise
      flood could squeeze through it. Non-structural blue decorations are
      traversable during the second pass but erased if they lie in the blocked
      side of the maze.
    - Paint orange on every reachable background cell that lies on the maze's
      inner boundary: on the grid edge, next to a structural wall, or next to
      a non-reachable cell.

    Key insight: The maze geometry alone is not enough; you must evaluate door
    width. Rooms behind a one-cell chokepoint belong to the plain maze, but not
    to the width-valid reachable region that receives the orange lining.
    """
    return _solve_core(grid, use_width_constraint=True)[0]


def _room_top_left(room: tuple[int, int]) -> tuple[int, int]:
    rr, cc = room
    return 1 + rr * STEP, 1 + cc * STEP


def _room_center(room: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = _room_top_left(room)
    return r0 + ROOM // 2, c0 + ROOM // 2


def _room_inner_spots(room: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = _room_top_left(room)
    return [
        (r, c)
        for r in range(r0 + 1, r0 + ROOM - 1)
        for c in range(c0 + 1, c0 + ROOM - 1)
    ]


def _room_neighbors(room: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    rr, cc = room
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _build_tree(rows: int, cols: int, start: tuple[int, int], rng: random.Random) -> dict[tuple[int, int], tuple[int, int] | None]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    stack = [start]
    while stack:
        cur = stack[-1]
        unvisited = [nb for nb in _room_neighbors(cur, rows, cols) if nb not in parent]
        if not unvisited:
            stack.pop()
            continue
        nxt = rng.choice(unvisited)
        parent[nxt] = cur
        stack.append(nxt)
    return parent


def _tree_children(
    parent: dict[tuple[int, int], tuple[int, int] | None]
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    children = {node: [] for node in parent}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)
    return children


def _subtree_sets(
    children: dict[tuple[int, int], list[tuple[int, int]]],
    root: tuple[int, int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    memo: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def dfs(node: tuple[int, int]) -> set[tuple[int, int]]:
        acc = {node}
        for ch in children[node]:
            acc |= dfs(ch)
        memo[node] = acc
        return acc

    dfs(root)
    return memo


def _pick_blocked_roots(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    children: dict[tuple[int, int], list[tuple[int, int]]],
    root: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    subtree = _subtree_sets(children, root)
    total = len(parent)
    cands = [node for node in parent if node != root]
    combos: list[tuple[tuple[int, int], ...]] = []
    for k in (1, 2):
        for nodes in combinations(cands, k):
            blocked: set[tuple[int, int]] = set()
            valid = True
            for node in nodes:
                sub = subtree[node]
                if blocked & sub:
                    valid = False
                    break
                blocked |= sub
            if not valid:
                continue
            blocked_count = len(blocked)
            reach_count = total - blocked_count
            if blocked_count < 2 or reach_count < 2:
                continue
            blocked_ratio = blocked_count / total
            if not (0.20 <= blocked_ratio <= 0.60):
                continue
            combos.append(nodes)
    if not combos:
        return None
    return list(rng.choice(combos))


def _door_offset(width: int, rng: random.Random) -> int:
    lo = 1
    hi = ROOM - width - 1
    if hi < lo:
        lo = 0
        hi = ROOM - width
    return rng.randint(lo, hi)


def _carve_door(
    structural: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    width: int,
    rng: random.Random,
) -> None:
    ar, ac = _room_top_left(a)
    br, bc = _room_top_left(b)
    if ar == br:
        left_c = min(ac, bc)
        offset = _door_offset(width, rng)
        structural[ar + offset : ar + offset + width, left_c + ROOM] = False
    else:
        top_r = min(ar, br)
        offset = _door_offset(width, rng)
        structural[top_r + ROOM, ac + offset : ac + offset + width] = False


def _render_structural(
    room_rows: int,
    room_cols: int,
    parent: dict[tuple[int, int], tuple[int, int] | None],
    blocked_roots: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    h = room_rows * STEP + 1
    w = room_cols * STEP + 1
    structural = np.ones((h, w), dtype=bool)

    for rr in range(room_rows):
        for cc in range(room_cols):
            r0, c0 = _room_top_left((rr, cc))
            structural[r0 : r0 + ROOM, c0 : c0 + ROOM] = False

    for node, par in parent.items():
        if par is None:
            continue
        width = 1 if node in blocked_roots else 2
        _carve_door(structural, node, par, width, rng)

    structural[h - STEP : h, 0:STEP] = False
    return structural


def _reachable_rooms(
    root: tuple[int, int],
    children: dict[tuple[int, int], list[tuple[int, int]]],
    blocked_roots: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    q = deque([root])
    out.add(root)
    while q:
        node = q.popleft()
        for ch in children[node]:
            if ch in blocked_roots or ch in out:
                continue
            out.add(ch)
            q.append(ch)
    return out


def _build_base_grid(structural: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    grid = np.full(structural.shape, BG, dtype=int)
    grid[structural] = WALL
    grid[marker_rc] = MARKER
    return grid


def _place_hidden_decor(
    grid: np.ndarray,
    blocked_rooms: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    hidden = np.zeros_like(grid, dtype=bool)
    count = min(len(blocked_rooms), rng.randint(*HIDDEN_DECOR_RANGE))
    for room in rng.sample(blocked_rooms, count):
        r, c = _room_center(room)
        hidden[r, c] = True
        grid[r, c] = WALL
    return hidden


def _place_visible_decor(
    grid: np.ndarray,
    reachable_rooms: list[tuple[int, int]],
    marker_room: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    visible = np.zeros_like(grid, dtype=bool)
    if rng.random() >= VISIBLE_DECOR_PROB:
        return visible

    candidates = [room for room in reachable_rooms if room != marker_room]
    rng.shuffle(candidates)
    for room in candidates:
        for r, c in rng.sample(_room_inner_spots(room), len(_room_inner_spots(room))):
            if grid[r, c] != BG:
                continue
            visible[r, c] = True
            grid[r, c] = WALL
            return visible
    return visible


def _metrics_from_input(grid: np.ndarray) -> dict[str, float]:
    g = np.asarray(grid, dtype=int)
    marker_rc = (g.shape[0] - 1, 0)
    passable = g != WALL
    plain = _plain_reach_mask(marker_rc, passable)
    width = _width_reach_mask(marker_rc, passable)
    diff = plain & ~width
    return {
        "reach_ratio": float(width.mean()),
        "diff_ratio": float(diff.mean()),
        "plain_only_cells": float(diff.sum()),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.choice(ROOM_ROWS_CHOICES)
    room_cols = rng.choice(ROOM_COLS_CHOICES)
    h = room_rows * STEP + 1
    w = room_cols * STEP + 1
    marker_rc = (h - 1, 0)
    marker_room = (room_rows - 1, 0)

    parent = _build_tree(room_rows, room_cols, marker_room, rng)
    children = _tree_children(parent)
    blocked_roots = _pick_blocked_roots(parent, children, marker_room, rng)
    if blocked_roots is None:
        return None
    blocked_root_set = set(blocked_roots)
    subtree = _subtree_sets(children, marker_room)
    blocked_rooms_set: set[tuple[int, int]] = set()
    for node in blocked_roots:
        blocked_rooms_set |= subtree[node]
    reachable_rooms_set = _reachable_rooms(marker_room, children, blocked_root_set)

    structural = _render_structural(room_rows, room_cols, parent, blocked_root_set, rng)
    if structural[marker_rc]:
        return None
    if len(components_mask(structural, connectivity=4)) != 1:
        return None

    grid = _build_base_grid(structural, marker_rc)
    hidden = _place_hidden_decor(grid, sorted(blocked_rooms_set), rng)
    if not hidden.any():
        return None
    _place_visible_decor(grid, sorted(reachable_rooms_set), marker_room, rng)

    metrics = _metrics_from_input(grid)
    if not (REACH_RATIO_RANGE[0] <= metrics["reach_ratio"] <= REACH_RATIO_RANGE[1]):
        return None
    if not (DIFF_RATIO_RANGE[0] <= metrics["diff_ratio"] <= DIFF_RATIO_RANGE[1]):
        return None
    if metrics["plain_only_cells"] < ROOM * ROOM:
        return None

    out = main(grid)
    plain_out = _plain_main(grid)
    if np.array_equal(out, plain_out):
        return None

    if np.any(hidden & (out == WALL)):
        return None
    if not np.any(out == FILL):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant")
