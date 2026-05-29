# Combo variant of dd6b8c4b (index 8)
# combo_spec: D3/3+D5/1
# Name: 多色待收集像素+按颜色分组排序

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Keep the original 11x11 scene: fixed green 3x3 destination around one red center,
#      orange floor, optional magenta walls, and scattered single-pixel source tokens.
#    - Change 1 (D5/1): source tokens can use several visible colors instead of only maroon.
#    - Change 2 (D3/3): reachable tokens are no longer globally sorted just by distance.
#      They are grouped by color first, using one fixed visible palette order
#      blue -> yellow -> teal -> maroon, and within each color group they are sorted by
#      flood-fill distance, then row/column as a stable tiebreaker.
#    - Small trace:
#      suppose reachable tokens are
#      [(d=0, blue), (d=1, maroon), (d=2, yellow), (d=3, blue)].
#      Original task / D5-only variant would place colors [blue, maroon, yellow, blue].
#      This combo places [blue, blue, yellow, maroon] because both blue tokens must be
#      consumed before any yellow, and yellow before maroon.
# b) Complexity calibration:
#    - Restrict token colors to a compact palette {blue=1, yellow=4, teal=8, maroon=9}.
#      Each sample activates only 2-3 of them. This keeps the new grouping rule visible
#      without turning the board into color noise.
#    - Generator prefers 10-14 reachable tokens so the destination usually fills all 9 cells.
#      This matters because the ninth slot cuts through one color group, making the inner
#      distance sort affect which same-colored source tokens disappear.
# c) Implicit assumptions broken from the original solver:
#    - `passable = {floor, 9}` no longer holds; all active token colors must be traversable.
#    - `np.argwhere(grid == 9)` no longer finds all candidates; candidates are every
#      non-structural token color in the allowed palette.
#    - `maroons.sort()` was enough before. Now the key must be
#      `(color_group_order, distance, row, col)`.
#    - Writing `9` into the destination is wrong now; each chosen token keeps its own color.
#    - Original floor inference relied on token color being unique. This combo keeps the
#      structural colors fixed, so the solver can use explicit constants safely.
# d) Placement legality:
#    - Tokens are only placed on floor cells outside the destination block.
#    - Reachable tokens come from the floor component connected to the destination openings;
#      distractors come from unreachable floor cells; sampling uses disjoint cell sets.
#    - Wall generation stays inside bounds and never overwrites the destination block.
# e) Visual observability:
#    - Every accepted sample has at least two reachable colors, and the color sequence produced
#      by combo sorting differs from the old distance-only sequence.
#    - The same visible color order repeats across samples, so humans can infer a stable
#      group precedence from examples instead of relying on hidden numeric ids.
#    - Generator also forces one color group to be partially selected at the 9-cell cutoff,
#      so "within-group by distance" changes which same-colored source tokens vanish.
# f) Visual cleanliness:
#    - Structural colors remain unchanged from the base task.
#    - Tokens stay as isolated single pixels on the floor, so walls, destination, and tokens
#      remain visually separable.
# g) Color-change inferability:
#    - There is no recoloring. Tokens preserve their source color when moved.
#    - The only new abstraction is the repeated cross-example group order among visible colors.
# h) New solver assumptions and generator guarantees:
#    - Assumption: movable token colors belong to the fixed palette order below.
#      Generator enforces this exactly.
#    - Assumption: there is at least one passable cell adjacent to the destination.
#      Generator rejects blocked layouts.
#    - Assumption: at least two token colors are reachable and the combo ordering changes the
#      destination relative to distance-only sorting. Generator checks this before accepting.
# i) Review of new hidden requirements:
#    - The solver loops over the fixed palette order to recover color groups; therefore the
#      input must never introduce an out-of-palette movable color.
#    - The BFS uses floor plus active token colors as passable terrain; generator guarantees
#      tokens are only placed on floor, so this matches the intended geometry.
#    - The erasure logic assumes chosen tokens occupy unique cells; generator samples unique
#      source locations, so no overlap can invalidate that assumption.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
SOURCE_PALETTE = (1, 4, 8, 9)
COLOR_RANK = {color: idx for idx, color in enumerate(SOURCE_PALETTE)}

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

DEST_CAPACITY = 9
REACHABLE_TOKEN_RANGE = (10, 14)
DISTRACTOR_MAX = 4
TOTAL_TOKEN_RANGE = (10, 16)
MIN_REACHABLE_FLOOR = REACHABLE_TOKEN_RANGE[0]
GEN_RETRIES = 400
INF = 10**9


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    passable: set[int],
) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    dist = np.full((h, w), INF, dtype=int)
    queue = deque()

    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                if int(grid[rr, cc]) in passable and dist[rr, cc] == INF:
                    dist[rr, cc] = 0
                    queue.append((rr, cc))

    assert queue, "there must be at least one passable cell adjacent to the destination block"

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == INF:
                if int(grid[rr, cc]) in passable:
                    dist[rr, cc] = int(dist[r, c]) + 1
                    queue.append((rr, cc))

    return dist


def _active_token_colors(grid: np.ndarray) -> list[int]:
    present = {int(color) for color in np.unique(grid)}
    colors = [color for color in SOURCE_PALETTE if color in present]
    assert colors, "grid must contain at least one movable token color"
    return colors


def _combo_sorted_tokens(
    grid: np.ndarray,
    dist: np.ndarray,
    token_colors: list[int],
) -> list[tuple[int, int, int, int]]:
    tokens = []
    token_set = set(token_colors)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in token_set and int(dist[r, c]) < INF:
                tokens.append((COLOR_RANK[color], int(dist[r, c]), int(r), int(c), color))
    tokens.sort()
    return [(d, r, c, color) for _, d, r, c, color in tokens]


def _distance_sorted_tokens(
    grid: np.ndarray,
    dist: np.ndarray,
    token_colors: list[int],
) -> list[tuple[int, int, int, int]]:
    tokens = []
    token_set = set(token_colors)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in token_set and int(dist[r, c]) < INF:
                tokens.append((int(dist[r, c]), int(r), int(c), color))
    tokens.sort()
    return tokens


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable colored pixels into the 3x3 block around the unique red center, but group them by color before using distance.

    Input structure:
    - The board keeps the original fixed destination: a green 3x3 block with one red center, placed on orange floor and optionally fenced by magenta walls.
    - Source tokens are scattered single pixels on floor cells outside the destination, and they can use several visible colors from a small palette.

    Rule:
    - Flood-fill outward from passable cells touching the destination, traversing floor cells and token cells while walls remain blocking.
    - Collect every reachable token, sort them first by the fixed visible color-group order blue -> yellow -> teal -> maroon, then within each color by flood-fill distance and row/column, and place the chosen colors into the destination in row-major order.
    - Erase exactly the moved source tokens back to floor, while unreachable tokens and any unselected overflow tokens remain in place.

    Key insight: Reachability still decides which tokens are even eligible, but once eligibility is known the decisive ordering is no longer purely geometric; color groups dominate globally, and distance only breaks ties inside a color group.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    token_colors = _active_token_colors(grid)
    passable = {FLOOR_COLOR, *token_colors}
    dist = _distance_map(grid, dest_cells, passable)
    tokens = _combo_sorted_tokens(grid, dist, token_colors)

    k = min(len(dest_cells), len(tokens))
    chosen = tokens[:k]

    for _, r, c, _ in chosen:
        out[r, c] = FLOOR_COLOR

    for (r, c), (_, _, _, color) in zip(dest_cells[:k], chosen):
        out[r, c] = color

    return out


def _bfs_reachable_floor(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], np.ndarray]:
    dist = _distance_map(grid, dest_cells, {FLOOR_COLOR})
    dest_set = set(dest_cells)
    reachable = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if (r, c) not in dest_set and grid[r, c] == FLOOR_COLOR and int(dist[r, c]) < INF
    }
    return reachable, dist


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    gap_cells = set(random.sample(ring, random.randint(*RING_GAP_RANGE)))
    wall_ring = [cell for cell in ring if cell not in gap_cells]
    wall_cells = set(wall_ring)
    style = random.choice(WALL_STYLES)

    if style == "arms":
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        dist2 = random.randint(*FRAME_DIST_RANGE)
        for r in range(CENTER_R - dist2, CENTER_R + dist2 + 1):
            for c in range(CENTER_C - dist2, CENTER_C + dist2 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (CENTER_R - dist2, CENTER_R + dist2) or c in (CENTER_C - dist2, CENTER_C + dist2):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr += dr
                    nc += dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for rc in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _random_composition(total: int, parts: int, min_each: int) -> list[int]:
    remainder = total - parts * min_each
    assert remainder >= 0
    cuts = sorted(random.sample(range(remainder + parts - 1), parts - 1)) if parts > 1 else []
    values = []
    prev = -1
    for cut in cuts + [remainder + parts - 1]:
        values.append(cut - prev - 1 + min_each)
        prev = cut
    return values


def _choose_active_colors(kinds: int) -> list[int]:
    chosen = set(random.sample(list(SOURCE_PALETTE), kinds))
    return [color for color in SOURCE_PALETTE if color in chosen]


def _sample_group_counts(active_count: int, total_reachable: int) -> list[int]:
    min_each = 2
    prefix_min = min_each * (active_count - 1)
    prefix_max = min(DEST_CAPACITY - 1, total_reachable - min_each)
    if prefix_min > prefix_max:
        raise ValueError("cannot fit visible color groups into the destination cutoff")
    prefix_total = random.randint(prefix_min, prefix_max)
    early_counts = _random_composition(prefix_total, active_count - 1, min_each) if active_count > 1 else []
    last_count = total_reachable - prefix_total
    return early_counts + [last_count]


def _build_color_multiset(active_colors: list[int], counts: list[int]) -> list[int]:
    colors = []
    for color, count in zip(active_colors, counts):
        colors.extend([color] * count)
    return colors


def _combo_preview(
    source_cells: list[tuple[int, int]],
    source_colors: list[int],
    dist: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    records = [(COLOR_RANK[color], int(dist[r, c]), int(r), int(c), color) for (r, c), color in zip(source_cells, source_colors)]
    records.sort()
    return [(d, r, c, color) for _, d, r, c, color in records]


def _distance_preview(
    source_cells: list[tuple[int, int]],
    source_colors: list[int],
    dist: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    records = [(int(dist[r, c]), int(r), int(c), color) for (r, c), color in zip(source_cells, source_colors)]
    records.sort()
    return records


def _accept_behavior(
    source_cells: list[tuple[int, int]],
    source_colors: list[int],
    dist: np.ndarray,
) -> bool:
    combo = _combo_preview(source_cells, source_colors, dist)[:DEST_CAPACITY]
    distance_only = _distance_preview(source_cells, source_colors, dist)[:DEST_CAPACITY]
    combo_colors = [color for _, _, _, color in combo]
    distance_colors = [color for _, _, _, color in distance_only]
    if combo_colors == distance_colors:
        return False

    prefix_by_color: dict[int, int] = {}
    for _, _, _, color in combo:
        prefix_by_color[color] = prefix_by_color.get(color, 0) + 1
    total_by_color: dict[int, int] = {}
    for color in source_colors:
        total_by_color[color] = total_by_color.get(color, 0) + 1

    # Require at least one color group to be split by the 9-cell capacity:
    # some tokens of that color move and some same-colored tokens remain.
    return any(0 < prefix_by_color.get(color, 0) < total for color, total in total_by_color.items())


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor, dist = _bfs_reachable_floor(base, dest_cells)
    if len(reachable_floor) < MIN_REACHABLE_FLOOR:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_reachable_tokens = min(REACHABLE_TOKEN_RANGE[1], len(reachable_floor))
    if max_reachable_tokens < REACHABLE_TOKEN_RANGE[0]:
        return None
    reachable_token_count = random.randint(REACHABLE_TOKEN_RANGE[0], max_reachable_tokens)

    active_count = random.randint(2, min(3, len(SOURCE_PALETTE)))
    active_colors = _choose_active_colors(active_count)
    counts = _sample_group_counts(active_count, reachable_token_count)
    source_colors = _build_color_multiset(active_colors, counts)

    max_total = TOTAL_TOKEN_RANGE[1]
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), max_total - reachable_token_count)
    min_total = TOTAL_TOKEN_RANGE[0]
    min_needed = max(0, min_total - reachable_token_count)
    if min_needed > max_distractors:
        return None
    distractor_count = random.randint(min_needed, max_distractors)

    source_cells = random.sample(list(reachable_floor), reachable_token_count)
    random.shuffle(source_colors)
    if not _accept_behavior(source_cells, source_colors, dist):
        return None

    distractor_cells = random.sample(list(unreachable_floor), distractor_count)
    distractor_colors = [random.choice(active_colors) for _ in range(distractor_count)]

    inp = base.copy()
    for (r, c), color in zip(source_cells, source_colors):
        inp[r, c] = color
    for (r, c), color in zip(distractor_cells, distractor_colors):
        inp[r, c] = color

    ordered_sources = _combo_preview(source_cells, source_colors, dist)

    out = base.copy()
    for (r, c), color in zip(distractor_cells, distractor_colors):
        out[r, c] = color
    for _, r, c, _ in ordered_sources[:DEST_CAPACITY]:
        out[r, c] = FLOOR_COLOR
    for (r, c), (_, _, _, color) in zip(dest_cells[:DEST_CAPACITY], ordered_sources[:DEST_CAPACITY]):
        out[r, c] = color
    for _, r, c, color in ordered_sources[DEST_CAPACITY:]:
        out[r, c] = color

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
