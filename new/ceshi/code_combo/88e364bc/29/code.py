# Combo variant of 88e364bc (index 29)
# combo_spec: D12/1+D19/1
# Name: 边缘中部图例+3-4个图例

"""
Design analysis for the combo variant
=====================================

spec_variant:
  grid: 32x32, bg=0
  legends:
    count: 3-4
    placement: fixed side-middle slots on the outer border; never in corners
      - top legend at the upper edge midpoint
      - bottom legend at the lower edge midpoint
      - left legend at the left edge midpoint
      - right legend at the right edge midpoint
    structure:
      - all legends use the source-task 5x5 three-color rectangle
      - fill color C matches exactly one region-wall color
      - blue 1 is the center mass (4 cells), red 2 is the indicator mass (2 cells)
      - direction is still sign(centroid(red) - centroid(blue))
    direction inventory:
      - top legend points down
      - bottom legend points up
      - left legend points right
      - right legend points left
      - each sample uses 3 or 4 of these side/direction pairs
  regions:
    count: exactly one enclosed chamber per legend color
    placement: one chamber near the corresponding side, with large gaps between chambers
    shape: clean rectangular rooms with 1-cell-thick colored walls
    wall colors: 3-4 unique colors from {3,5,6,7,8,9}
  yellow_pixels:
    1-2 per region
    generation is output-first: choose final stacked cells against the side-specific far wall,
    then reverse-slide 1-3 steps to obtain the input

Behavior trace
--------------
Fusion-rule clarification:
  - The raw D12/1 single-variant reference moved legends inside regions, but the combo description
    explicitly overrides placement to edge-middle legends. This implementation follows the combo
    description because that is the actual contract for D12/1+D19/1.

Concrete example:
  - Suppose there is a green 5x5 legend centered on the top edge, and its red pixels sit below the
    blue pixels, so the encoded direction is downward.
  - The matching green chamber sits below it in the upper half of the grid. If a yellow starts near
    the chamber's top interior row, the source solver would already understand "move down", but the
    visual scene is now different: there are several side-mounted legends rather than one corner cue.
  - The yellow ends on the chamber's bottom interior row, while the chambers controlled by the left,
    right, and bottom legends simultaneously move in their own directions.

Difference from the source task:
  - Source task: 1-2 legends, usually corner-biased, so the puzzle is mainly "which color uses which
    arrow".
  - Combo task: 3-4 legends distributed across the four sides, so the solver must extract multiple
    color-direction bindings from a side-balanced global layout.

Complexity calibration
----------------------
  - The legends keep the original shape language, so the added complexity comes from quantity and
    layout rather than from inventing a new symbol vocabulary.
  - Using 3-4 distinct wall colors avoids ambiguity: every legend color has repeated evidence because
    its chamber uses the same color, yet the total number of colors remains small enough to parse.
  - Chambers are intentionally clean rectangles. This avoids visual clutter in a 32x32 grid where the
    main novelty should be the four-side control scheme, not messy wall geometry.

Implicit assumption audit for `main()`
--------------------------------------
  - Original assumption: a valid legend is a border-touching rectangle. This still holds, but corners
    are no longer special, so detection must accept side-middle border rectangles.
  - Original assumption: there may be only 1-2 legends. This no longer holds; the solver must allow
    3-4 independent color->direction mappings.
  - Original assumption: each enclosed region can infer its controller from the majority adjacent wall
    color. This still holds because every chamber has a single wall color and legends remain outside
    the chambers, so blue/red legend cells never contaminate border voting.
  - Original assumption: sliding logic is independent per enclosed region. This still holds because the
    generator keeps chambers disjoint; no yellow can leave its chamber or interfere with another one.

Placement legality and cleanliness
----------------------------------
  - Legends live in fixed edge slots with background around them, so they never overlap each other.
  - Chambers use disjoint side-specific bounding boxes, so walls and interiors cannot overlap.
  - Reverse-slid yellow inputs are chosen only from legal upstream cells inside the chamber, and an
    oracle check confirms the produced input solves to the intended output.

Visual observability
--------------------
  - A human looking at 2-4 examples can immediately see that the legends are not in corners anymore:
    they sit at the midpoints of the grid edges.
  - The four possible directions are tied to four spatial sides, producing a strong "commands from all
    sides" pattern that is visible before one even traces the yellow motion.
  - No hidden ordering uses numeric color ids or any non-visual property; the only bindings are visible
    color matches and visible arrow orientation.

New solver assumptions introduced here
--------------------------------------
  - Legends use the canonical 5x5 cardinal template with exact color counts {fill: 19, blue: 4, red: 2}.
    The generator enforces this directly.
  - Each legend fill color is unique and controls exactly one chamber.
  - Chambers remain ordinary enclosed passable components consisting only of {0, 4} inside colored walls.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H = 32
GRID_W = 32
BG = 0
MOVER = 4
CENTER_COLOR = 1
INDICATOR_COLOR = 2

WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
SIDE_ORDER = ["top", "bottom", "left", "right"]
SIDE_DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
LEGEND_TOP_LEFT = {
    "top": (0, 14),
    "bottom": (27, 14),
    "left": (14, 0),
    "right": (14, 27),
}
REGION_OPTIONS = {
    "top": [(7, 12, 6, 8), (7, 13, 6, 7)],
    "bottom": [(19, 12, 6, 8), (19, 13, 6, 7)],
    "left": [(12, 6, 9, 6), (13, 6, 8, 6)],
    "right": [(12, 20, 9, 6), (13, 20, 8, 6)],
}

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 3
GEN_RETRIES = 200


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = CENTER_COLOR
    g[1, 2] = CENTER_COLOR
    g[1, 3] = INDICATOR_COLOR
    g[3, 1] = CENTER_COLOR
    g[3, 2] = CENTER_COLOR
    g[3, 3] = INDICATOR_COLOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    dirs: dict[int, tuple[int, int]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        if len(pts) != 25:
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_candidates = [col for col, cnt in counts.items() if cnt == 19]
        center_ok = counts.get(CENTER_COLOR) == 4
        indicator_ok = counts.get(INDICATOR_COLOR) == 2
        if len(fill_candidates) != 1 or not center_ok or not indicator_ok:
            continue
        fill = int(fill_candidates[0])
        if fill in {bg, MOVER, CENTER_COLOR, INDICATOR_COLOR}:
            continue

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == CENTER_COLOR], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == INDICATOR_COLOR], dtype=float)
        cc = center.mean(axis=0)
        ci = indic.mean(axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if (dr, dc) not in SIDE_DIRS.values():
            continue
        dirs[fill] = (dr, dc)
    return dirs


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside colored chambers using 3-4 edge-middle legends as the direction source.

    Input structure:
    - The grid is larger than the source task and contains 3-4 small 5x5 legends mounted at the middle
      of the top/bottom/left/right border positions.
    - Each legend has fill color C plus blue center pixels and red indicator pixels; the same color C
      also appears as the wall color of exactly one enclosed chamber elsewhere in the grid.
    - Yellow pixels lie inside those enclosed chambers, which are otherwise passable background cells.

    Rule:
    - Detect every border legend, decode its direction from blue-center to red-indicator, and map that
      direction to the legend's fill color C.
    - For each enclosed chamber, identify its wall color C from the adjacent non-passable border cells.
    - Slide all yellow pixels in that chamber along the decoded direction, letting them stack against the
      chamber's far wall or against already-settled yellows from the same chamber.

    Key insight:
    - The important relation is not the legend's absolute side location by itself, but the visible
      color match between a side-mounted legend and its chamber wall; the arrow orientation inside that
      legend tells the chamber how its yellow pixels move.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert 3 <= len(dirs) <= 4, "expected 3-4 edge-middle legends"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v in {bg, MOVER, CENTER_COLOR, INDICATOR_COLOR}:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER

    return out


def _rect_region(color: int, direction: tuple[int, int], outer: tuple[int, int, int, int], side: str) -> dict:
    r0, c0, h, w = outer
    passable = {(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)}
    walls = set()
    for r in range(r0, r0 + h):
        walls.add((r, c0))
        walls.add((r, c0 + w - 1))
    for c in range(c0, c0 + w):
        walls.add((r0, c))
        walls.add((r0 + h - 1, c))
    return {
        "color": color,
        "dir": direction,
        "side": side,
        "passable": passable,
        "walls": walls,
        "outer": outer,
        "outputs": [],
    }


def _paint_region(grid: np.ndarray, region: dict) -> None:
    for r, c in region["walls"]:
        grid[r, c] = region["color"]


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _frontier_targets(region_set: set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    frontier = []
    for p in region_set:
        if not _slide_step_valid(region_set, p[0], p[1], dr, dc):
            frontier.append(p)
    frontier.sort(key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    return frontier


def _find_reverse_slide(
    region_set: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in region_set or (pr, pc + dc) not in region_set:
                    ok = False
                    break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(region_set, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_region_yellows(rng: random.Random, region: dict, grid: np.ndarray) -> bool:
    blob = region["passable"]
    dr, dc = region["dir"]
    n_yellows = rng.choice(YELLOWS_PER_REGION)

    frontier = _frontier_targets(blob, dr, dc)
    if not frontier:
        return False

    first_pool = frontier[: min(6, len(frontier))]
    rng.shuffle(first_pool)
    for first in first_pool:
        output_sets = [[first]]
        if n_yellows == 2:
            preferred: list[tuple[int, int]] = []
            pr, pc = first[0] - dr, first[1] - dc
            if (pr, pc) in blob and (pr, pc) != first:
                preferred.append((pr, pc))
            for cand in frontier:
                if cand != first and cand not in preferred:
                    preferred.append(cand)
            output_sets = [[first, second] for second in preferred[: min(6, len(preferred))]]

        for outputs in output_sets:
            out_sorted = sorted(outputs, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
            already_stopped = set()
            inputs: list[tuple[int, int]] = []
            ok = True
            for out_pos in out_sorted:
                candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, inputs, rng)
                if candidate is None:
                    ok = False
                    break
                inputs.append(candidate)
                already_stopped.add(out_pos)
            if not ok:
                continue
            for r, c in inputs:
                grid[r, c] = MOVER
            region["outputs"] = outputs
            return True
    return False


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    labeled, n_comp = label_mask(grid == BG, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, GRID_H - 1) or c in (0, GRID_W - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["passable"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = input_grid.copy()
    expected[expected == MOVER] = BG
    for reg in regions:
        if not reg["outputs"]:
            return False
        for r, c in reg["outputs"]:
            expected[r, c] = MOVER
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    n_legends = rng.choice([3, 4])
    sides = rng.sample(SIDE_ORDER, n_legends)
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    regions = []
    for side, color in zip(sides, colors):
        dr, dc = SIDE_DIRS[side]
        patch = _legend_5x5(color, dr, dc)
        lr0, lc0 = LEGEND_TOP_LEFT[side]
        paste(grid, patch, lr0, lc0)

        outer = rng.choice(REGION_OPTIONS[side])
        region = _rect_region(color, (dr, dc), outer, side)
        _paint_region(grid, region)
        regions.append(region)

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    for region in regions:
        if not _place_region_yellows(rng, region, input_grid):
            return None

    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
