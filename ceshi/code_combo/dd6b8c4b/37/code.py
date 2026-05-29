# Combo variant of dd6b8c4b (index 37)
# combo_spec: D5/1+D12/2
# Name: 一源填多目标+多色像素

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Replace the original "many scattered maroons -> one center block" layout with one shared
#      lower source chamber that contains mixed single-pixel tokens of two colors.
#    - The grid now has two separate 3x3 green destination blocks near the top. Each destination
#      has a colored center, and that center color names which source-token color belongs there.
#    - For a concrete trace, suppose the left center is blue(1) and the right center is yellow(4).
#      The chamber contains blue/yellow tokens interleaved in one connected orange region.
#      Old logic would have looked for one red center and treated all source tokens as one pool.
#      New logic runs per target: left block collects only reachable blue tokens, right block only
#      reachable yellow tokens; each target still uses its own BFS distance ordering and row-major
#      fill inside the block.
# b) Complexity calibration:
#    - Keep exactly two targets and exactly two active token colors per sample. This is enough to
#      make the source->multiple-target split visible without creating a hard-to-read color soup.
#    - The chamber is shared, so the two colors are spatially mixed. This forces the solver to use
#      the center-color mapping rather than a simple geometric partition of the source area.
# c) Original implicit assumptions that break:
#    - `len(centers) == 1` no longer holds: there are multiple destinations, and their centers are
#      not fixed to red.
#    - `np.argwhere(grid == 9)` assumed a single source color; now movable tokens are the two
#      distinct destination-center colors.
#    - One global distance ranking is no longer correct. Distances must be computed separately for
#      each destination block because one source region feeds multiple targets.
#    - Writing a uniform color into one destination block is wrong. Each destination is filled only
#      with tokens matching that block's center color.
# d) Placement legality:
#    - The generator uses fixed non-overlapping target positions and a fixed framed source chamber.
#    - Tokens are sampled only from chamber interior cells, so they never overlap walls or targets.
#    - Optional chamber pillars are single wall cells chosen from a safe interior subset, so the
#      chamber stays connected and remains large enough for all tokens.
# e) Visual observability:
#    - The chamber visibly contains a mixed-color token pool, while the two target centers provide
#      an explicit visible key for the color->target mapping.
#    - Because the two colors are interleaved inside one chamber, the mapping cannot be reduced to
#      "left side tokens go left, right side tokens go right"; examples visibly require color use.
# f) Visual cleanliness:
#    - Structure is layered cleanly: green target blocks at the top, magenta walls framing them and
#      the source chamber, orange floor as corridors, and only two token colors at once.
# g) Color inference:
#    - The mapping is not hidden: a destination's center color is the same color that fills that
#      destination in the output. The generator rejects samples where this mapping would be
#      redundant with a color-blind nearest-token baseline.
# h) New solver assumptions and generator guarantees:
#    - Assumption: every destination is a 3x3 block with a full green ring and a unique non-green
#      center color. Generator uses two fixed, separated targets and distinct center colors.
#    - Assumption: all movable tokens use only those center colors. Generator enforces this.
#    - Assumption: each target has at least one passable adjacent start cell and at least two
#      matching reachable tokens. The fixed corridor layout guarantees both.
#    - Assumption: targets do not share a color, so token ownership is unambiguous. Generator
#      samples two distinct colors each time.
# i) Fusion-specific anti-degeneracy check:
#    - A sample is accepted only if the fused solver output differs from an ablated solver that
#      ignores the center-color mapping and lets each target take its nearest reachable tokens of
#      any color. This guarantees that D5/1 and D12/2 interact materially rather than decoratively.


GRID_H, GRID_W = 11, 11
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_RING = 3
SOURCE_PALETTE = (1, 4, 8, 9)
INF = 10**9

TARGET_CENTERS = [(2, 2), (2, 8)]
TARGET_COUNT = 2
TARGET_TOKEN_RANGE = (2, 7)
TOTAL_TOKEN_RANGE = (6, 14)
GEN_RETRIES = 300

CHAMBER_TOP, CHAMBER_LEFT = 6, 1
CHAMBER_BOTTOM, CHAMBER_RIGHT = 10, 9
CHAMBER_OPENINGS = {(6, 2), (6, 8)}
PILLAR_CANDIDATES = [(7, 4), (7, 6), (8, 5), (9, 4), (9, 6)]
MAX_PILLARS = 2


def _block_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _scan_targets(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    targets = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            ring = patch.copy()
            ring[1, 1] = DEST_RING
            if np.all(ring == DEST_RING):
                center_color = int(patch[1, 1])
                if center_color not in {FLOOR_COLOR, WALL_COLOR, DEST_RING}:
                    targets.append(
                        {
                            "center": (int(r), int(c)),
                            "color": center_color,
                            "cells": _block_cells(int(r), int(c), h, w),
                        }
                    )
    targets.sort(key=lambda t: (t["center"][0], t["center"][1]))
    return targets


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
                if int(grid[rr, cc]) in passable and int(dist[rr, cc]) == INF:
                    dist[rr, cc] = 0
                    queue.append((rr, cc))

    assert queue, "each destination needs at least one adjacent passable cell"

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(dist[rr, cc]) == INF:
                if int(grid[rr, cc]) in passable:
                    dist[rr, cc] = int(dist[r, c]) + 1
                    queue.append((rr, cc))

    return dist


def _sorted_tokens(
    grid: np.ndarray,
    dist: np.ndarray,
    *,
    allowed_colors: set[int],
) -> list[tuple[int, int, int, int]]:
    tokens = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in allowed_colors and int(dist[r, c]) < INF:
                tokens.append((int(dist[r, c]), int(r), int(c), color))
    tokens.sort()
    return tokens


def _apply_rule(grid: np.ndarray, *, use_color_mapping: bool) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    targets = _scan_targets(grid)
    assert len(targets) >= 2, "grid must contain at least two destination blocks"

    center_colors = [int(t["color"]) for t in targets]
    assert len(center_colors) == len(set(center_colors)), "destination center colors must be unique"

    token_colors = set(center_colors)
    passable = {FLOOR_COLOR, *token_colors}

    for target in targets:
        dist = _distance_map(grid, target["cells"], passable)
        allowed = {int(target["color"])} if use_color_mapping else token_colors
        candidates = _sorted_tokens(grid, dist, allowed_colors=allowed)
        chosen = candidates[: len(target["cells"])]

        for _, r, c, _ in chosen:
            out[r, c] = FLOOR_COLOR
            grid[r, c] = FLOOR_COLOR

        for (r, c), (_, _, _, color) in zip(target["cells"], chosen):
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move color-matched reachable pixels from one shared source chamber into multiple green target blocks.

    Input structure:
    - Two green 3x3 destination blocks sit near the top; each has a distinct colored center.
    - A single walled orange source chamber below contains scattered single-pixel tokens using exactly those center colors.
    - Magenta walls leave corridor openings from each destination into the shared chamber.

    Rule:
    - For each destination block separately, flood-fill from passable cells touching that block, moving through orange floor or token cells.
    - Keep only reachable tokens whose color matches that block's center color, sort them by distance to that block and then by row/column, and place them into the block in row-major order.
    - Erase the moved source tokens back to orange floor; tokens of the other color stay available for their own destination.

    Key insight: The source is shared, so geometry alone is not enough; the solver must combine wall-constrained reachability with the visible center-color mapping to decide which tokens belong to which target.
    """

    return _apply_rule(np.array(grid, dtype=int), use_color_mapping=True)


def _build_base(center_colors: list[int]) -> tuple[np.ndarray, list[dict], list[tuple[int, int]]]:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    targets = []

    for (cr, cc), color in zip(TARGET_CENTERS, center_colors):
        cells = _block_cells(cr, cc, GRID_H, GRID_W)
        for r, c in cells:
            grid[r, c] = DEST_RING
        grid[cr, cc] = int(color)
        targets.append({"center": (cr, cc), "color": int(color), "cells": cells})

    for cr, cc in TARGET_CENTERS:
        opening = (cr + 2, cc)
        for r in range(cr - 2, cr + 3):
            for c in range(cc - 2, cc + 3):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in set(_block_cells(cr, cc, GRID_H, GRID_W)):
                    if (r, c) != opening:
                        grid[r, c] = WALL_COLOR

    for r in range(CHAMBER_TOP, CHAMBER_BOTTOM + 1):
        for c in range(CHAMBER_LEFT, CHAMBER_RIGHT + 1):
            is_border = r in {CHAMBER_TOP, CHAMBER_BOTTOM} or c in {CHAMBER_LEFT, CHAMBER_RIGHT}
            if is_border and (r, c) not in CHAMBER_OPENINGS:
                grid[r, c] = WALL_COLOR

    for r, c in [(5, 1), (5, 3), (5, 7), (5, 9)]:
        grid[r, c] = WALL_COLOR

    pillars = random.sample(PILLAR_CANDIDATES, random.randint(0, MAX_PILLARS))
    for r, c in pillars:
        grid[r, c] = WALL_COLOR

    source_cells = [
        (r, c)
        for r in range(CHAMBER_TOP + 1, CHAMBER_BOTTOM)
        for c in range(CHAMBER_LEFT + 1, CHAMBER_RIGHT)
        if int(grid[r, c]) == FLOOR_COLOR
    ]

    return grid, targets, source_cells


def _sample_color_counts(capacity: int) -> list[int] | None:
    for _ in range(80):
        counts = [random.randint(*TARGET_TOKEN_RANGE) for _ in range(TARGET_COUNT)]
        total = sum(counts)
        if total <= capacity and TOTAL_TOKEN_RANGE[0] <= total <= TOTAL_TOKEN_RANGE[1]:
            return counts
    return None


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _ablated_output(inp: np.ndarray) -> np.ndarray:
    return _apply_rule(np.array(inp, dtype=int), use_color_mapping=False)


def _gen_attempt() -> np.ndarray | None:
    center_colors = random.sample(list(SOURCE_PALETTE), TARGET_COUNT)
    base, targets, source_cells = _build_base(center_colors)

    counts = _sample_color_counts(len(source_cells))
    if counts is None:
        return None

    chosen_cells = random.sample(source_cells, sum(counts))
    assignments: list[tuple[tuple[int, int], int]] = []
    idx = 0
    for target, count in zip(targets, counts):
        for _ in range(count):
            assignments.append((chosen_cells[idx], int(target["color"])))
            idx += 1
    random.shuffle(assignments)

    inp = base.copy()
    for (r, c), color in assignments:
        inp[r, c] = int(color)

    out = _expected_output(inp)

    for target in targets:
        moved = sum(int(out[r, c] == target["color"]) for r, c in target["cells"])
        if moved < 2:
            return None

    if np.array_equal(out, _ablated_output(inp)):
        return None

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
