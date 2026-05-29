# Combo variant of dd6b8c4b (index 45)
# combo_spec: D5/1+D10/2
# Name: 同距离合并标记+多色像素

import random
from collections import defaultdict, deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The original task sorted every reachable maroon token individually by
#      (distance, row, col) and copied the first k of them into the 3x3 block.
#    - This combo keeps the same 11x11 scene, fixed destination block, optional
#      walls, and reachability test, but changes the collected objects to
#      multiple colors and changes the destination fill unit from "one token" to
#      "one distance layer summary".
#    - Reachable tokens are first partitioned by BFS distance from the block.
#      Each distinct distance contributes exactly one destination cell:
#      monochrome layer -> that token color; mixed-color layer -> a fixed merge
#      marker (grey 5). Example:
#      distance 0: [(3,5,blue), (5,3,yellow)] -> output cell grey
#      distance 1: [(6,5,maroon)] -> output cell maroon
#      distance 2: [(7,5,blue), (5,7,blue)] -> output cell blue
#      The destination begins [grey, maroon, blue, ...] in row-major order.
#      The original task would instead have serialized all five tokens one by
#      one and could not produce the compressed grey summary.
# b) Complexity calibration:
#    - Token colors are restricted to a small palette {blue, yellow, teal,
#      maroon}. Each sample uses only 2-3 of them, so mixed layers are visible
#      without turning the board into noise.
#    - The merge marker is a single fixed color (grey 5). This keeps the new
#      logic observable: humans only need to infer "same distance + multiple
#      colors => grey summary" rather than learn an arbitrary recoloring table.
#    - Generator forces at least two reachable distance layers and at least one
#      mixed-color merged layer, so the fusion behavior is present in accepted
#      samples rather than being an occasional edge case.
# c) Implicit assumptions broken from the original solver:
#    - `passable = {floor, 9}` no longer holds because movable tokens can have
#      several colors; the new solver treats every non-structural token color as
#      passable.
#    - `np.argwhere(grid == 9)` no longer finds all candidates; now candidates
#      are all non-structural token colors.
#    - Filling the destination with one copy per token is no longer valid.
#      Output length now depends on the number of occupied distance layers, not
#      the number of reachable tokens.
#    - Conservation of token count is intentionally broken by D10/2: several
#      reachable tokens may collapse into one summary cell. The generator and
#      solver must agree on this compression.
#    - Multi-color layers introduce a new assumption: grey 5 is reserved for the
#      merge summary and never used as a source token color. The generator
#      enforces that by excluding 5 from the token palette.
# d) Placement legality:
#    - Source and distractor tokens are sampled only from floor cells outside
#      the destination block, so they never overlap walls or the block itself.
#    - Grouping is done on precomputed distance layers of reachable floor cells,
#      so every chosen source location is in bounds and already known to be
#      reachable or unreachable as intended.
#    - The number of chosen distance layers never exceeds the 9 destination
#      cells, so the summarized output always fits in the block.
# e) Visual observability:
#    - Humans can see two simultaneous regularities: wall-restricted
#      reachability still matters, and the output count now matches the number of
#      occupied distance rings rather than the number of tokens.
#    - Mixed layers are visible because multiple nearby token colors disappear
#      together and a single grey cell appears in the destination. Monochrome
#      layers remain colored, providing contrast that makes the grey rule
#      inferable from a few examples.
#    - No ordering depends on raw color ids. Distances are geometric, and the
#      summary color depends only on whether the layer's token colors all match.
# f) Visual cleanliness:
#    - The board still uses the original sparse single-pixel token layout and
#      the same structural colors. Only one extra color (grey) is introduced,
#      and it appears only inside the destination as a merge summary.
#    - Because the destination stores one summary per distance layer, it stays
#      readable instead of becoming a dense multi-token collage.
# g) Change inferability:
#    - There is no hidden color mapping. A monochrome layer keeps its visible
#      token color; a mixed layer turns into the same fixed grey marker in every
#      sample. This makes the color rule structurally grounded and learnable.
# h) New solver assumptions and generator guarantees:
#    - Assumption: structural colors remain fixed at floor=7, wall=6, block=3,
#      center=2, merge marker=5. Generator enforces exactly that.
#    - Assumption: at least one passable cell touches the destination, so BFS
#      has a valid start. Generator rejects layouts without reachable floor.
#    - Assumption: at least one occupied distance layer contains >=2 distinct
#      token colors, otherwise the combo would collapse to a simpler D5-only
#      variant. Generator explicitly constructs such a layer.
# i) Fusion-specific calibration:
#    - The two dimensions are not independent patches. Distance-layer merging is
#      defined in terms of the color composition of each layer, so the presence
#      of multiple token colors directly changes the compressed representation.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MERGE_MARK = 5
SOURCE_PALETTE = (1, 4, 8, 9)

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

DIST_GROUP_RANGE = (2, 6)
GROUP_SIZE_RANGE = (1, 3)
DISTRACTOR_MAX = 10
TOTAL_TOKEN_RANGE = (4, 16)
MIN_REACHABLE = 4
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


def _reachable_token_groups(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    token_colors: set[int],
) -> list[tuple[int, list[tuple[int, int, int]]]]:
    dist = _distance_map(grid, dest_cells, {FLOOR_COLOR, *token_colors})
    groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in token_colors and int(dist[r, c]) < INF:
                groups[int(dist[r, c])].append((int(r), int(c), color))

    ordered = []
    for d in sorted(groups):
        tokens = sorted(groups[d], key=lambda x: (x[0], x[1], x[2]))
        ordered.append((d, tokens))
    return ordered


def _group_summary_color(tokens: list[tuple[int, int, int]]) -> int:
    colors = {color for _, _, color in tokens}
    return next(iter(colors)) if len(colors) == 1 else MERGE_MARK


def _reachable_floor_and_dist(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], np.ndarray]:
    dist = _distance_map(grid, dest_cells, {FLOOR_COLOR})
    dest_set = set(dest_cells)
    reachable = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r, c) not in dest_set and grid[r, c] == FLOOR_COLOR and int(dist[r, c]) < INF
    }
    return reachable, dist


def _distance_buckets(cells: set[tuple[int, int]], dist: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in cells:
        buckets[int(dist[r, c])].append((r, c))
    for d in buckets:
        buckets[d].sort()
    return buckets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize reachable colored token layers inside the 3x3 block around the unique red center.

    Input structure:
    - Orange floor and optional magenta walls surround a fixed green 3x3 destination whose center is red.
    - Single-pixel source tokens of several non-structural colors are scattered on floor cells outside that block.

    Rule:
    - Flood-fill outward from passable cells touching the destination, moving through floor or token cells, and collect all reachable source tokens.
    - Group the reachable tokens by BFS distance from the destination. Each occupied distance contributes one destination cell in row-major order: if every token in that layer has the same color, write that color; otherwise write the grey merge marker.
    - Erase every reachable source token back to floor. Unreachable tokens stay in place, and any unused destination cells keep their original green/red values.

    Key insight: The output is no longer a per-token transfer. The solver must first detect wall-limited reachability, then compress each distance layer into a single summary whose color depends on whether that layer is monochrome or mixed.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, grid.shape[0], grid.shape[1])

    token_colors = {
        int(color)
        for color in np.unique(grid)
        if int(color) not in {FLOOR_COLOR, WALL_COLOR, DEST_COLOR, RED_CENTER, MERGE_MARK}
    }
    assert token_colors, "grid must contain at least one non-structural source color"

    groups = _reachable_token_groups(grid, dest_cells, token_colors)
    k = min(len(dest_cells), len(groups))

    for _, tokens in groups:
        for r, c, _ in tokens:
            out[r, c] = FLOOR_COLOR

    for (r, c), (_, tokens) in zip(dest_cells[:k], groups[:k]):
        out[r, c] = _group_summary_color(tokens)

    return out


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


def _assign_group_colors(size: int, active_colors: list[int], mixed: bool) -> list[int]:
    if mixed:
        distinct = random.randint(2, min(len(active_colors), size, 3))
        chosen = random.sample(active_colors, distinct)
        colors = list(chosen)
        while len(colors) < size:
            colors.append(random.choice(chosen))
        random.shuffle(colors)
        return colors
    color = random.choice(active_colors)
    return [color] * size


def _choose_group_sizes(chosen_distances: list[int], buckets: dict[int, list[tuple[int, int]]], mixed_distance: int) -> dict[int, int] | None:
    sizes: dict[int, int] = {}
    for d in chosen_distances:
        capacity = min(GROUP_SIZE_RANGE[1], len(buckets[d]))
        if d == mixed_distance:
            if capacity < 2:
                return None
            sizes[d] = random.randint(2, capacity)
        else:
            if capacity >= 2 and random.random() < 0.35:
                sizes[d] = random.randint(2, capacity)
            else:
                sizes[d] = 1

    total = sum(sizes.values())
    expandable = [d for d in chosen_distances if sizes[d] < min(GROUP_SIZE_RANGE[1], len(buckets[d]))]
    while total < TOTAL_TOKEN_RANGE[0] and expandable:
        d = random.choice(expandable)
        sizes[d] += 1
        total += 1
        if sizes[d] >= min(GROUP_SIZE_RANGE[1], len(buckets[d])):
            expandable.remove(d)

    if total < TOTAL_TOKEN_RANGE[0] or total > TOTAL_TOKEN_RANGE[1]:
        return None
    return sizes


def _expected_output(
    base: np.ndarray,
    dest_cells: list[tuple[int, int]],
    source_specs: list[tuple[int, int, int, int]],
    distractor_specs: list[tuple[int, int, int]],
) -> np.ndarray:
    out = base.copy()
    by_distance: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for d, r, c, color in source_specs:
        by_distance[d].append((r, c, color))
    for r, c, color in distractor_specs:
        out[r, c] = color
    for idx, d in enumerate(sorted(by_distance)):
        out[dest_cells[idx]] = _group_summary_color(sorted(by_distance[d]))
    return out


def _fusion_signature_from_specs(source_specs: list[tuple[int, int, int, int]]) -> dict[str, int]:
    by_distance: dict[int, list[int]] = defaultdict(list)
    for d, _, _, color in source_specs:
        by_distance[d].append(color)
    merged_layers = sum(1 for colors in by_distance.values() if len(colors) > 1)
    mixed_layers = sum(1 for colors in by_distance.values() if len(set(colors)) > 1)
    monochrome_merged_layers = sum(1 for colors in by_distance.values() if len(colors) > 1 and len(set(colors)) == 1)
    return {
        "distance_layers": len(by_distance),
        "merged_layers": merged_layers,
        "mixed_layers": mixed_layers,
        "monochrome_merged_layers": monochrome_merged_layers,
        "tokens": len(source_specs),
    }


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor, dist = _reachable_floor_and_dist(base, dest_cells)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    buckets = _distance_buckets(reachable_floor, dist)
    available_distances = sorted(buckets)
    multi_cell_distances = [d for d in available_distances if len(buckets[d]) >= 2]
    if len(available_distances) < DIST_GROUP_RANGE[0] or not multi_cell_distances:
        return None

    max_groups = min(DIST_GROUP_RANGE[1], len(available_distances), len(dest_cells))
    min_groups = min(DIST_GROUP_RANGE[0], max_groups)
    if max_groups < DIST_GROUP_RANGE[0]:
        return None
    num_groups = random.randint(min_groups, max_groups)

    mixed_distance = random.choice(multi_cell_distances)
    other_distances = [d for d in available_distances if d != mixed_distance]
    if len(other_distances) < num_groups - 1:
        return None
    chosen_distances = sorted([mixed_distance] + random.sample(other_distances, num_groups - 1))

    sizes = _choose_group_sizes(chosen_distances, buckets, mixed_distance)
    if sizes is None:
        return None

    active_count = random.randint(2, 3)
    active_colors = random.sample(list(SOURCE_PALETTE), active_count)

    source_specs: list[tuple[int, int, int, int]] = []
    saw_monochrome_color_layer = False
    for d in chosen_distances:
        cells = random.sample(buckets[d], sizes[d])
        if d == mixed_distance:
            colors = _assign_group_colors(sizes[d], active_colors, mixed=True)
        else:
            make_mixed = sizes[d] >= 2 and random.random() < 0.2
            colors = _assign_group_colors(sizes[d], active_colors, mixed=make_mixed)
        if len(set(colors)) == 1:
            saw_monochrome_color_layer = True
        for (r, c), color in zip(cells, colors):
            source_specs.append((d, r, c, color))

    if not saw_monochrome_color_layer:
        return None

    signature = _fusion_signature_from_specs(source_specs)
    if signature["mixed_layers"] < 1 or signature["distance_layers"] < 2 or signature["merged_layers"] < 1:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = sorted(all_floor - reachable_floor)
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor))
    total_tokens = len(source_specs)
    min_needed = max(0, TOTAL_TOKEN_RANGE[0] - total_tokens)
    if max_distractors < min_needed:
        return None
    max_allowed = min(max_distractors, TOTAL_TOKEN_RANGE[1] - total_tokens)
    num_distractors = random.randint(min_needed, max_allowed) if max_allowed > 0 else 0

    distractor_cells = random.sample(unreachable_floor, num_distractors)
    distractor_specs = [(r, c, random.choice(active_colors)) for r, c in distractor_cells]

    inp = base.copy()
    for _, r, c, color in source_specs:
        inp[r, c] = color
    for r, c, color in distractor_specs:
        inp[r, c] = color

    out = _expected_output(base, dest_cells, source_specs, distractor_specs)

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
