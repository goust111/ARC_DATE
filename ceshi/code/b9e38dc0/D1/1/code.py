# Variant of b9e38dc0
# Dimension: D1 — 主路径机制——填充与辐射 (path)
# Change: flood_fill + ray_casting——先洪水填充半包围区域的内部，再沿包围边框的开口方向向外辐射延伸... → 填充后向所有开口方向辐射

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4


DESIGN_ANALYSIS = """
Variant design analysis
=======================

a) Concrete behavior and trace
- The wall stays the dominant non-background object, but it becomes a thin rectangular room
  border with two missing border segments ("openings") on adjacent sides.
- A fill-colored seed blob is placed inside the room. The solver first fills the room interior.
- After the room is filled, each opening emits its own straight beam outward to the grid edge.
  Example: if the room has a top opening and a left opening, the old single-opening logic would
  pick one main side and only emit one beam; the new logic must emit both the upward beam and the
  leftward beam. That difference is directly visible in the output as multiple border-reaching rays.

b) Complexity calibration
- Openings are restricted to exactly two adjacent sides, each with length 1 or 2.
- Across the dataset, the adjacent-side pair varies, so the multi-opening rule is still visually
  obvious while keeping the wall connected and the two rays orthogonal.
- Obstacles remain sparse singleton pixels and can still cast shadows, but the core novelty is
  the need to follow several rays rather than one principal ray.

c) Hidden assumptions from the original solver that break here
- Endpoint detection no longer works: a frame-with-gaps does not encode one opening via two
  degree-1 endpoints, so the original endpoint scan would either fail or detect the wrong thing.
- Summing endpoint outward vectors into one `rad` is invalid: different openings may point to
  different sides, so there is no single global radiate direction.
- The original barrier trick mixes "inside fill" and "outside wedge" into one BFS region. That
  relied on a single wedge bounded by two rays. In this variant, interior fill must be computed
  first inside the room bbox, then each opening ray is processed independently.
- Shadow processing over one global scan direction is also invalid. Shadows must be applied along
  each opening's own beam direction.

d) Placement legality
- The generator always builds a 1-pixel-thick rectangular frame wholly inside the grid.
- Openings are punched only on non-corner border cells, so the wall stays one connected component.
- The seed blob is placed strictly inside the frame; obstacles are placed only on background cells
  and kept away from the wall so the fill color remains the non-wall color closest to the wall.

e) Visual observability
- The key clue is structural, not color-id-based: one room clearly has several side gaps, and the
  output shows one straight beam per gap. Because openings span both axes across the dataset,
  humans can infer "all openings radiate" rather than memorizing a fixed side.

f) Visual cleanliness
- Only one room is generated, with exactly two openings and at most five singleton obstacles.
- Rays are axis-aligned strips of width 1-2, which keeps the output readable and avoids messy
  overlapping geometry.

g) Color-change inferability
- No hidden color remapping is used. The seed color simply expands, the wall color stays fixed,
  and obstacle colors stay fixed.

h) Generator-side anti-degeneration checks
- Every generated instance must have two openings.
- Every instance must differ from a single-opening baseline solver by a non-trivial number of
  filled cells, so samples that accidentally behave like the original task are rejected.

i) New solver assumptions and generator guarantees
- The solver assumes the chosen wall component is a thin frame whose bbox border contains the wall
  except for explicit openings. The generator enforces exactly that geometry.
- The solver assumes the seed is inside the room bbox interior. The generator places it there.
- The solver assumes obstacles are not wall-colored and may block beams independently. The
  generator preserves those color/placement conditions.
"""


# -- Configuration constants --------------------------------------------------
GRID_H = (10, 18)
GRID_W = (14, 19)

FRAME_H = (6, 9)
FRAME_W = (7, 10)
FRAME_MARGIN = 2

WALL_PIXELS_MIN = 18
WALL_PIXELS_MAX = 30

OPENING_LEN = (1, 2)

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]

MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.4
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 16
MIN_MULTI_BONUS = 4
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _bbox_from_points(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_fill_color(g: np.ndarray, bg: int, wall: int, wall_points: list[tuple[int, int]]) -> int:
    vals = [int(v) for v in np.unique(g) if int(v) not in (bg, wall)]
    assert vals, "expected at least one non-wall color"

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    return min(vals, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))


def _detect_openings(comp_mask: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    pts = np.argwhere(comp_mask)
    assert pts.size, "expected non-empty wall component"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)

    openings: list[dict] = []

    def add_runs(side: str, cells: list[tuple[int, int]]) -> None:
        run: list[tuple[int, int]] = []
        for r, c in cells:
            if not comp_mask[r, c]:
                run.append((r, c))
            elif run:
                openings.append(_make_opening(side, run))
                run = []
        if run:
            openings.append(_make_opening(side, run))

    def _make_opening(side: str, run: list[tuple[int, int]]) -> dict:
        if side == "top":
            direction = (-1, 0)
            inner = [(r0 + 1, c) for _, c in run]
        elif side == "bottom":
            direction = (1, 0)
            inner = [(r1 - 1, c) for _, c in run]
        elif side == "left":
            direction = (0, -1)
            inner = [(r, c0 + 1) for r, _ in run]
        else:
            direction = (0, 1)
            inner = [(r, c1 - 1) for r, _ in run]
        return {"side": side, "cells": run[:], "inner": inner, "dir": direction}

    add_runs("top", [(r0, c) for c in range(c0 + 1, c1)])
    add_runs("bottom", [(r1, c) for c in range(c0 + 1, c1)])
    add_runs("left", [(r, c0) for r in range(r0 + 1, r1)])
    add_runs("right", [(r, c1) for r in range(r0 + 1, r1)])

    return openings, (r0, c0, r1, c1)


def _solve_core(grid: np.ndarray, opening_limit: int | None = None) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall

    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    wall_points = [p for comp in comps for p in comp]
    fill = _find_fill_color(g, bg, wall, wall_points)
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return g

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    openings, (r0, c0, r1, c1) = _detect_openings(comp_mask)
    assert openings, "expected at least one opening"
    if opening_limit is not None:
        openings = openings[:opening_limit]

    room_mask = np.zeros_like(comp_mask, dtype=bool)
    if r1 - r0 > 1 and c1 - c0 > 1:
        room_mask[r0 + 1:r1, c0 + 1:c1] = True
    passable_room = room_mask & (~comp_mask) & ((g == bg) | (g == fill))

    q = [p for p in fill_pos if passable_room[p[0], p[1]]]
    seen = np.zeros_like(passable_room, dtype=bool)
    for r, c in q:
        seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable_room[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    out = g.copy()
    out[seen & (g == bg)] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    for opening in openings:
        dr, dc = opening["dir"]
        for r, c in opening["cells"]:
            rr, cc = r, c
            while 0 <= rr < h and 0 <= cc < w:
                if obstacles[rr, cc]:
                    break
                if out[rr, cc] == bg or out[rr, cc] == fill:
                    out[rr, cc] = fill
                rr += dr
                cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a room-like enclosure and send the fill outward through every visible opening.

    Input structure:
    - The background is the most common color. The wall is the most common non-background color
      and forms a thin rectangular enclosure with one or more missing border segments.
    - A non-wall seed color appears inside the enclosure. Any other non-background colors are
      obstacle pixels placed on background cells.
    Rule:
    - First flood-fill the room interior reachable from the seed inside the enclosure.
    - Then continue the fill outward through every border opening, extending one straight beam per
      opening until the grid edge. Obstacles remain unchanged and stop the beam behind them.

    Key insight: The missing wall segments, not a single dominant mouth, define the radiating
    paths; each opening contributes its own outward beam after the interior is filled.
    """
    return _solve_core(grid)


def _single_opening_baseline(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, opening_limit=1)


def _choose_openings(frame_h: int, frame_w: int) -> list[tuple[str, int, int]]:
    sides = random.choice(
        [
            ["top", "left"],
            ["top", "right"],
            ["bottom", "left"],
            ["bottom", "right"],
        ]
    )

    choices = []
    for side in sides:
        per_side = []
        for length in range(OPENING_LEN[0], OPENING_LEN[1] + 1):
            if side in ("top", "bottom"):
                lo = 2
                hi = frame_w - length - 2
            else:
                lo = 2
                hi = frame_h - length - 2
            if lo <= hi:
                for start in range(lo, hi + 1):
                    per_side.append((side, start, length))
        if not per_side:
            return []
        choices.append(per_side)

    valid_pairs = []
    for a in choices[0]:
        a0, a1 = a[1], a[1] + a[2] - 1
        for b in choices[1]:
            b0, b1 = b[1], b[1] + b[2] - 1
            if a1 < b0 or b1 < a0:
                valid_pairs.append([a, b])

    if not valid_pairs:
        return []
    return random.choice(valid_pairs)


def _build_frame(frame_h: int, frame_w: int, openings: list[tuple[str, int, int]]) -> set[tuple[int, int]]:
    pts = set()
    for c in range(frame_w):
        pts.add((0, c))
        pts.add((frame_h - 1, c))
    for r in range(frame_h):
        pts.add((r, 0))
        pts.add((r, frame_w - 1))

    for side, start, length in openings:
        for offset in range(length):
            if side == "top":
                pts.discard((0, start + offset))
            elif side == "bottom":
                pts.discard((frame_h - 1, start + offset))
            elif side == "left":
                pts.discard((start + offset, 0))
            else:
                pts.discard((start + offset, frame_w - 1))

    # A thin border with two disjoint edge gaps would split into two components.
    # Add a 1-cell-deep interior brace across one gap so the wall stays connected
    # while the bbox-edge openings remain visible to the solver.
    for side, start, length in openings:
        if side == "left":
            for offset in range(length):
                pts.add((start + offset, 1))
            break
        if side == "right":
            for offset in range(length):
                pts.add((start + offset, frame_w - 2))
            break
        if side == "top":
            for offset in range(length):
                pts.add((1, start + offset))
            break
        if side == "bottom":
            for offset in range(length):
                pts.add((frame_h - 2, start + offset))
            break
    return pts


def _place_frame(h: int, w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    r0 = random.randint(FRAME_MARGIN, h - frame_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, w - frame_w - FRAME_MARGIN)
    return r0, c0


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(
    grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _border_touch_count(mask: np.ndarray) -> int:
    hits = 0
    if mask[0, :].any():
        hits += 1
    if mask[-1, :].any():
        hits += 1
    if mask[:, 0].any():
        hits += 1
    if mask[:, -1].any():
        hits += 1
    return hits


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    frame_h = random.randint(*FRAME_H)
    frame_w = random.randint(*FRAME_W)
    if h - frame_h < 2 * FRAME_MARGIN or w - frame_w < 2 * FRAME_MARGIN:
        return None

    openings = _choose_openings(frame_h, frame_w)
    if len(openings) < 2:
        return None
    local_wall = _build_frame(frame_h, frame_w, openings)
    if not (WALL_PIXELS_MIN <= len(local_wall) <= WALL_PIXELS_MAX):
        return None
    local_mask = np.zeros((frame_h, frame_w), dtype=bool)
    for r, c in local_wall:
        local_mask[r, c] = True
    if len(components_mask(local_mask, connectivity=8)) != 1:
        return None

    anchor_r, anchor_c = _place_frame(h, w, frame_h, frame_w)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for r, c in local_wall:
        grid[anchor_r + r, anchor_c + c] = wall_color

    interior = [
        (anchor_r + r, anchor_c + c)
        for r in range(1, frame_h - 1)
        for c in range(1, frame_w - 1)
    ]
    interior_set = set(interior)
    adjacent = [
        (r, c) for r, c in interior
        if r in (anchor_r + 1, anchor_r + frame_h - 2) or c in (anchor_c + 1, anchor_c + frame_w - 2)
    ]
    if not adjacent:
        return None

    blob = _grow_seed_blob(random.choice(adjacent), interior_set)
    for r, c in blob:
        grid[r, c] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)
    if not nominal_fill_zone.any():
        return None

    try:
        out = main(grid)
        base = _single_opening_baseline(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    multi_bonus = (out == fill_color) & (base != fill_color)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if _border_touch_count(new_fill) < 1:
        return None
    if int(multi_bonus.sum()) < MIN_MULTI_BONUS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D1 variant input")
