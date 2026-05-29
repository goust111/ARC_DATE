# Combo variant of dd6b8c4b (index 3)
# combo_spec: D2/1+D9/0
# Name: 墙壁有通道口+规则矩形框架

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task already moves reachable maroon tokens into the 3x3 destination by BFS distance.
#    Here the destination stays fixed, but the obstacle pattern becomes two clean magenta rectangles:
#    an inner 5x5 frame around the destination and an outer 9x9 frame. An "effective doorway" needs
#    three aligned floor strips on one side: inner-frame gap, 1-cell corridor between frames, and
#    outer-frame gap. Inner-only gaps are visible decoys: they look like entrances near the green block
#    but do not reach the outside ring where most source maroons live.
# b) Complexity calibration:
#    The geometry is regular enough to read by eye, but not trivial: each side may contain 0-2 openings,
#    and each opening has width 1 or 2. Only a subset of the inner gaps align all the way through both
#    rectangles, so the solver must inspect connectivity rather than nearest distance alone.
# c) Broken implicit assumptions from the original code:
#    - "A passable neighbor of the destination means access to the main floor" is no longer true.
#      Some destination-adjacent gap cells are dead ends.
#    - "The wall ring is irregular, so local openings are obvious" is no longer true. The structure is
#      globally rectangular; the relevant cue is whether aligned gaps exist across both frames.
#    The BFS logic itself still works, but the generator must now guarantee both true and false gaps.
# d) Placement legality:
#    All walls and gaps live on fixed rectangles: outer rows/cols 1..9, inner rows/cols 3..7, destination
#    rows/cols 4..6. Gap spans only use non-corner side cells, so no overlaps or out-of-bounds openings
#    occur. Distractors are placed in isolated annulus pockets that stay disconnected from every tunnel.
# e) Visual observability:
#    Humans can compare 2-4 examples and spot the rule because the only moving parts are maroon tokens and
#    the few missing cells on the two rectangular borders. The important difference is visible: some inner
#    frame holes line up to the outside border and some do not.
# f) Cleanliness:
#    The nested rectangles keep object roles separated: center target, inner frame, corridor strip, outer
#    frame, outside ring. No region carries multiple competing marker systems.
# g) Inferability:
#    No hidden ordering tricks are introduced. Sorting is still by visible path length through floor cells,
#    and width-2 doorways are literal 2-pixel holes rather than a color-coded convention.
# h) New solver assumptions and generator guarantees:
#    - There is exactly one red center. Generator guarantees it.
#    - At least one inner gap exists, so BFS always has a legal start.
#    - At least one full tunnel reaches the exterior, so reachable maroons can always be placed outside.
#    - Distractor pockets are sealed by construction, so unreachable maroons remain unreachable even after
#      successful tunnels make the outside ring reachable.

GRID_H = GRID_W = 11
CENTER_R = CENTER_C = 5
FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

INNER_R0, INNER_C0, INNER_R1, INNER_C1 = 3, 3, 7, 7
OUTER_R0, OUTER_C0, OUTER_R1, OUTER_C1 = 1, 1, 9, 9

GEN_RETRIES = 300
REACHABLE_TOKEN_RANGE = (3, 14)
DISTRACTOR_MAX = 4
TOTAL_MAROON_RANGE = (4, 16)

SIDE_PATTERNS = [
    [],
    [[4]],
    [[5]],
    [[6]],
    [[4, 5]],
    [[5, 6]],
    [[4], [6]],
]
SIDES = ("top", "bottom", "left", "right")
ISOLATED_POCKETS = [(2, 2), (2, 8), (8, 2), (8, 8)]


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _bfs_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    dest_set = set(dest_cells)
    passable = {FLOOR_COLOR, MAROON}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and (rr, cc) not in seen:
                seen.add((rr, cc))
                starts.append((rr, cc))
    assert starts, "the rectangular frame must expose at least one doorway next to the destination"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and int(dist[rr, cc]) == inf:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

    return dist, starts


def _sorted_reachable_maroons(grid: np.ndarray, dist: np.ndarray) -> list[tuple[int, int, int]]:
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < 10**9]
    maroons.sort()
    return maroons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the reachable maroon tokens through the true frame openings into the 3x3 target block.

    Input structure:
    - Orange floor surrounds two nested magenta rectangular frames centered on the unique red pixel.
    - The red pixel sits inside a 3x3 green destination block.
    - Maroon tokens lie on floor cells; some are outside the outer frame, some may be trapped in sealed pockets.
    - The rectangular borders contain 1-2 pixel gaps, but not every inner-frame gap actually connects outward.

    Rule:
    - Start a flood-fill from passable cells touching the destination block and move through orange or maroon cells.
    - Only maroon tokens in the same 4-connected component as a real doorway sequence are candidates.
    - Sort reachable maroons by BFS distance, then by row and column.
    - Replace the chosen source cells with orange floor and write the chosen maroons into the destination in row-major order.

    Key insight:
    - The inner rectangle can show decoy holes that terminate before the outer rectangle, so a visible opening
      near the green block is not enough; the path must cross both frames to access the outside maroons.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    dist, _starts = _bfs_distances(grid, dest_cells)
    maroons = _sorted_reachable_maroons(grid, dist)

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _door_cells(side: str, coords: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    if side == "top":
        inner = [(INNER_R0, c) for c in coords]
        mid = [(INNER_R0 - 1, c) for c in coords]
        outer = [(OUTER_R0, c) for c in coords]
    elif side == "bottom":
        inner = [(INNER_R1, c) for c in coords]
        mid = [(INNER_R1 + 1, c) for c in coords]
        outer = [(OUTER_R1, c) for c in coords]
    elif side == "left":
        inner = [(r, INNER_C0) for r in coords]
        mid = [(r, INNER_C0 - 1) for r in coords]
        outer = [(r, OUTER_C0) for r in coords]
    else:
        inner = [(r, INNER_C1) for r in coords]
        mid = [(r, INNER_C1 + 1) for r in coords]
        outer = [(r, OUTER_C1) for r in coords]
    return inner, mid, outer


def _sample_gap_layout() -> list[tuple[str, list[int], bool]]:
    for _ in range(200):
        spans: list[tuple[str, list[int]]] = []
        for side in SIDES:
            pattern = random.choice(SIDE_PATTERNS)
            for seg in pattern:
                spans.append((side, list(seg)))

        if not (3 <= len(spans) <= 6):
            continue

        success_min = 2 if len(spans) >= 4 else 1
        success_count = random.randint(success_min, len(spans) - 1)
        random.shuffle(spans)
        success_idx = set(range(success_count))
        layout = [(side, coords, idx in success_idx) for idx, (side, coords) in enumerate(spans)]

        if not any(len(coords) == 2 for _, coords, _ in layout):
            continue
        if not any(flag for _, _, flag in layout):
            continue
        if not any(not flag for _, _, flag in layout):
            continue
        return layout
    raise RuntimeError("failed to sample a non-degenerate doorway layout")


def _build_base_grid() -> tuple[np.ndarray, list[tuple[int, int]]]:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    for r in range(OUTER_R0, OUTER_R1 + 1):
        for c in range(OUTER_C0, OUTER_C1 + 1):
            if (r, c) not in dest_set:
                grid[r, c] = WALL_COLOR

    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid, dest_cells


def _carve_layout(
    grid: np.ndarray, layout: list[tuple[str, list[int], bool]]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    inner_gap_cells: set[tuple[int, int]] = set()
    full_tunnel_cells: set[tuple[int, int]] = set()
    decoy_cells: set[tuple[int, int]] = set()

    for side, coords, is_success in layout:
        inner, mid, outer = _door_cells(side, coords)
        for cell in inner + mid:
            grid[cell] = FLOOR_COLOR
        inner_gap_cells.update(inner)

        if is_success:
            for cell in outer:
                grid[cell] = FLOOR_COLOR
            full_tunnel_cells.update(inner + mid + outer)
        else:
            decoy_cells.update(inner + mid)

    return inner_gap_cells, full_tunnel_cells, decoy_cells


def _open_isolated_pockets(grid: np.ndarray, count: int) -> list[tuple[int, int]]:
    cells = random.sample(ISOLATED_POCKETS, count)
    for r, c in cells:
        grid[r, c] = FLOOR_COLOR
    return cells


def _reachable_floor_cells(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], np.ndarray]:
    dist, _starts = _bfs_distances(grid, dest_cells)
    reachable = [(int(r), int(c)) for r in range(GRID_H) for c in range(GRID_W) if int(grid[r, c]) == FLOOR_COLOR and int(dist[r, c]) < 10**9]
    return reachable, dist


def _expected_output(inp: np.ndarray, dest_cells: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    dist, _starts = _bfs_distances(inp, dest_cells)
    maroons = _sorted_reachable_maroons(inp, dist)
    k = min(len(dest_cells), len(maroons))
    for _, r, c in maroons[:k]:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _pick_reachable_sources(reachable: list[tuple[int, int]], dist: np.ndarray, n_tokens: int) -> list[tuple[int, int]]:
    ranked = sorted(reachable, key=lambda rc: (int(dist[rc]), rc[0], rc[1]))
    assert len(ranked) >= n_tokens

    if n_tokens <= 9:
        window = ranked[: min(len(ranked), max(12, n_tokens + 3))]
        return sorted(random.sample(window, n_tokens), key=lambda rc: (int(dist[rc]), rc[0], rc[1]))

    near = ranked[:9]
    far_pool = ranked[12:] if len(ranked) > 12 else ranked[9:]
    need_far = n_tokens - 9
    if len(far_pool) < need_far:
        far_pool = ranked[9:]
    extras = random.sample(far_pool, need_far)
    return near + sorted(extras, key=lambda rc: (int(dist[rc]), rc[0], rc[1]))


def _analyze_variant(
    layout: list[tuple[str, list[int], bool]], reachable_sources: list[tuple[int, int]], distractors: list[tuple[int, int]]
) -> dict[str, int]:
    return {
        "total_gaps": len(layout),
        "successful_gaps": sum(1 for _, _, ok in layout if ok),
        "decoy_gaps": sum(1 for _, _, ok in layout if not ok),
        "wide_gaps": sum(1 for _, coords, _ in layout if len(coords) == 2),
        "reachable_tokens": len(reachable_sources),
        "distractors": len(distractors),
    }


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    grid, dest_cells = _build_base_grid()
    layout = _sample_gap_layout()
    inner_gap_cells, _full_tunnel_cells, _decoy_cells = _carve_layout(grid, layout)

    assert inner_gap_cells, "generator must create at least one destination-adjacent gap"

    max_distractors = min(DISTRACTOR_MAX, len(ISOLATED_POCKETS))
    n_distractors = random.randint(0, max_distractors)
    pocket_cells = _open_isolated_pockets(grid, n_distractors)

    reachable, dist = _reachable_floor_cells(grid, dest_cells)
    reachable = [rc for rc in reachable if rc not in inner_gap_cells]
    if len(reachable) < REACHABLE_TOKEN_RANGE[0]:
        return None

    min_reach, max_reach = REACHABLE_TOKEN_RANGE
    n_reachable = random.randint(min_reach, min(max_reach, len(reachable)))

    total = n_reachable + n_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        extra = TOTAL_MAROON_RANGE[0] - total
        if n_reachable + extra <= min(max_reach, len(reachable)):
            n_reachable += extra
        elif n_distractors + extra <= max_distractors:
            extra_cells = [cell for cell in ISOLATED_POCKETS if cell not in pocket_cells]
            if len(extra_cells) < extra:
                return None
            for r, c in extra_cells[:extra]:
                grid[r, c] = FLOOR_COLOR
            pocket_cells.extend(extra_cells[:extra])
            n_distractors += extra
        else:
            return None

    if n_reachable + n_distractors > TOTAL_MAROON_RANGE[1]:
        overflow = n_reachable + n_distractors - TOTAL_MAROON_RANGE[1]
        if n_distractors >= overflow:
            n_distractors -= overflow
            pocket_cells = pocket_cells[:n_distractors]
        else:
            n_reachable -= overflow - n_distractors
            n_distractors = 0
            pocket_cells = []
        if n_reachable < REACHABLE_TOKEN_RANGE[0]:
            return None

    reachable_sources = _pick_reachable_sources(reachable, dist, n_reachable)
    distractor_cells = pocket_cells[:n_distractors]

    inp = grid.copy()
    for r, c in reachable_sources + distractor_cells:
        inp[r, c] = MAROON

    expected = _expected_output(inp, dest_cells)
    result = main(inp)
    if not np.array_equal(result, expected):
        return None

    metrics = _analyze_variant(layout, reachable_sources, distractor_cells)
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _metrics = result
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
