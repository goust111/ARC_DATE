# Combo variant of 332f06d7 (index 32)
# combo_spec: D2/0+D9/0+D19/0
# Name: 无宽度约束+固定N=1+始终可达

from __future__ import annotations

"""
Design analysis
===============

This combo intentionally collapses the original square-sliding task into a pure
single-pixel path-tracing task:

a) Concretization and behavior trace
------------------------------------
- Original task:
  there is a blue winding river, a black n×n mover square, and a red n×n goal
  square. Motion is computed on the grid of feasible n×n placements. If the
  river narrows below n, the mover gets stuck before the chokepoint.
- Combo task:
  n is fixed to 1, so the mover and destination are always single pixels.
  Because a 1×1 object only needs a traversable pixel, width-based chokepoints
  lose their blocking meaning. Combined with the "always reachable" requirement,
  the task becomes: follow the connected blue path from the black pixel to the
  red pixel, then place black on the red pixel and repaint the old black pixel
  blue.
- Concrete miniature trace:
  suppose black is at (1,1), red is at (9,4), and the blue path snakes through
  several turns with some 1-pixel necks.
  Original n=3 logic would scan 3×3 windows and could stop before a width-1
  neck. In this combo, n=1, so every blue pixel is a feasible placement; BFS on
  pixels reaches the red endpoint directly.

b) Complexity calibration
-------------------------
- The task is simpler than the base task, but not reduced to a trivial swap:
  the generator still creates a long winding blue corridor with multiple turns,
  spanning most rows and columns.
- The visual cue remains structural and human-visible: the only legal route is
  the blue channel; the black pixel does not move through green background.
- I keep the object vocabulary minimal: one black pixel, one red pixel, one
  connected blue river, one green background.

c) Implicit-assumption audit for main()
--------------------------------------
- The original solver assumed there may be multiple same-sized solid squares and
  had to infer the common size n. That assumption is removed: this combo
  explicitly assumes exactly one black pixel and one red pixel.
- The original feasible-window scan over all n×n top-left placements no longer
  applies because n is always 1. It is replaced with ordinary 4-neighbor BFS on
  traversable pixels.
- The original fallback branch for unreachable destinations conflicts with
  D19/0. The new solver asserts that the destination is reachable; the generator
  guarantees this.
- Width-based chokepoint logic from D2/0 is not merely ignored accidentally:
  it is intentionally subsumed by D9/0. Once n=1, "width >= n" becomes the same
  as plain pixel connectivity.

d) Placement legality
---------------------
- The generator paints the blue river first on a square grid, then overwrites
  one river pixel with black and one river pixel with red.
- The endpoint is extended to a non-corner boundary cell, keeping the target on
  an outer edge as in the base family.
- The mover is kept one cell away from an edge, so start and destination remain
  visually distinct and do not overlap.

e) Visual observability
-----------------------
- Across a few samples, a human sees a single black pixel traveling along a
  winding blue path to a single red pixel on the border.
- No hidden color ordering is used for inference. Black and red are fixed
  visible roles; blue is the only traversable path color.
- Although D2/0 and D19/0 are behaviorally absorbed by the fixed n=1 regime,
  the resulting task family is still visibly distinct from the original family:
  all objects are single pixels and every example is a pure path-following case.

f) Visual cleanliness
---------------------
- Only four colors are present.
- There are no distractor overlays or partially overlapping markers.
- The river is one clean connected corridor; black and red just mark its ends.

g) Variant-metric interpretation
--------------------------------
- For this combo, a useful quantitative signature is:
  1. detected mover size is always 1,
  2. destination is always reachable by pixel BFS,
  3. the solver always performs an exact move-to-target with no fallback.
- Comparing against the original solver on the same generated inputs yields the
  same output, but that is expected here because D9/0 intentionally forces the
  problem into the original task's n=1 reachable subregime. This is not treated
  as generator failure; the visible combo effect is the regime collapse itself.

h) New main() assumptions that generate_input() must satisfy
-----------------------------------------------------------
- exactly one black pixel exists,
- exactly one red pixel exists,
- there is at least one non-background path color,
- black and red lie on the same connected traversable component,
- red is on a non-corner border cell,
- black is inside the grid and near an edge.
The generator enforces all of these and validates each sample with main().
"""

from collections import Counter, deque
import random

import numpy as np


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _single_pixel_of_color(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    if len(pts) != 1:
        raise AssertionError(f"expected exactly one pixel of color {color}, found {len(pts)}")
    return int(pts[0, 0]), int(pts[0, 1])


def _infer_bg_and_river(grid: np.ndarray) -> tuple[int, int]:
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    river = None
    for color, _count in Counter(map(int, g.flatten())).most_common():
        if color in (bg, _MOVER, _DEST):
            continue
        river = int(color)
        break
    if river is None:
        raise AssertionError("river color not found")
    return bg, river


def _bfs_pixels(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    detect the unique black pixel and unique red pixel, treat blue+black+red as
    one traversable path, and move black to the red endpoint.
    """
    g = np.array(grid, dtype=int)
    _bg, river = _infer_bg_and_river(g)
    sr, sc = _single_pixel_of_color(g, _MOVER)
    dr, dc = _single_pixel_of_color(g, _DEST)

    traversable = (g == river) | (g == _MOVER) | (g == _DEST)
    dist = _bfs_pixels(traversable, (sr, sc))
    if dist[dr, dc] < 0:
        raise AssertionError("destination is unreachable in this always-reachable combo")

    out = g.copy()
    out[sr, sc] = river
    out[dr, dc] = _MOVER
    return out


def _snake_walk(dim: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for r in range(dim):
        cols = range(dim) if r % 2 == 0 else range(dim - 1, -1, -1)
        for c in cols:
            path.append((r, c))
    return path


def _transform_coord(r: int, c: int, dim: int, transform_id: int) -> tuple[int, int]:
    last = dim - 1
    if transform_id == 0:
        return r, c
    if transform_id == 1:
        return c, last - r
    if transform_id == 2:
        return last - r, last - c
    if transform_id == 3:
        return last - c, r
    if transform_id == 4:
        return r, last - c
    if transform_id == 5:
        return last - r, c
    if transform_id == 6:
        return c, r
    return last - c, last - r


def _transformed_snake(dim: int) -> list[tuple[int, int]]:
    base = _snake_walk(dim)
    transform_id = random.randint(0, 7)
    return [_transform_coord(r, c, dim, transform_id) for r, c in base]


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _compute_offsets(size: int, num_coarse: int) -> list[int] | None:
    if num_coarse < 2:
        return None
    first = 1
    last = size - 2
    if last <= first:
        return None
    vals = [int(round(first + i * (last - first) / (num_coarse - 1))) for i in range(num_coarse)]
    if vals[0] != first or vals[-1] != last:
        return None
    if len(set(vals)) != len(vals):
        return None
    if any(vals[i + 1] - vals[i] < 2 for i in range(len(vals) - 1)):
        return None
    return vals


def _render_channel(
    grid: np.ndarray,
    coarse_path: list[tuple[int, int]],
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_path: list[tuple[int, int]] = []
    for cr, cc in coarse_path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_path.append((pr, pc))
        grid[pr, pc] = _RIVER

    for (pr1, pc1), (pr2, pc2) in zip(pixel_path, pixel_path[1:]):
        if pr1 == pr2:
            c0, c1 = sorted((pc1, pc2))
            grid[pr1, c0 : c1 + 1] = _RIVER
        else:
            r0, r1 = sorted((pr1, pr2))
            grid[r0 : r1 + 1, pc1] = _RIVER
    return pixel_path


def _extend_endpoint_to_edge(
    grid: np.ndarray,
    pixel: tuple[int, int],
    coarse: tuple[int, int],
    coarse_dim: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    pr, pc = pixel
    cr, cc = coarse
    candidates: list[tuple[int, int]] = []
    if cr == 0:
        candidates.append((0, pc))
    if cr == coarse_dim - 1:
        candidates.append((h - 1, pc))
    if cc == 0:
        candidates.append((pr, 0))
    if cc == coarse_dim - 1:
        candidates.append((pr, w - 1))

    random.shuffle(candidates)
    candidates.sort(key=lambda rc: abs(rc[0] - pr) + abs(rc[1] - pc))

    for nr, nc in candidates:
        if (nr in (0, h - 1)) and (nc in (0, w - 1)):
            continue
        if nr == pr:
            c0, c1 = sorted((pc, nc))
            if np.isin(grid[pr, c0 : c1 + 1], [_BG, _RIVER]).all():
                grid[pr, c0 : c1 + 1] = _RIVER
                return nr, nc
        else:
            r0, r1 = sorted((pr, nr))
            if np.isin(grid[r0 : r1 + 1, pc], [_BG, _RIVER]).all():
                grid[r0 : r1 + 1, pc] = _RIVER
                return nr, nc
    return None


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    sr, sc = _single_pixel_of_color(out, _MOVER)
    dr, dc = _single_pixel_of_color(out, _DEST)
    out[sr, sc] = _RIVER
    out[dr, dc] = _MOVER
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(10, 16)
    coarse_choices = [dim for dim in range(4, 7) if _compute_offsets(size, dim) is not None]
    if not coarse_choices:
        return None
    coarse_dim = random.choice(coarse_choices)
    offsets_r = _compute_offsets(size, coarse_dim)
    offsets_c = _compute_offsets(size, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None

    coarse_path = _transformed_snake(coarse_dim)
    if _count_turns(coarse_path) < 4:
        return None

    grid = np.full((size, size), _BG, dtype=int)
    pixel_path = _render_channel(grid, coarse_path, offsets_r, offsets_c)

    river_ratio = float(np.mean(grid == _RIVER))
    if not (0.20 <= river_ratio <= 0.57):
        return None

    mover_r, mover_c = pixel_path[0]
    dest_r, dest_c = pixel_path[-1]
    extended = _extend_endpoint_to_edge(grid, (dest_r, dest_c), coarse_path[-1], coarse_dim)
    if extended is None:
        return None
    dest_r, dest_c = extended

    if min(mover_r, size - 1 - mover_r, mover_c, size - 1 - mover_c) != 1:
        return None
    if not (dest_r in (0, size - 1) or dest_c in (0, size - 1)):
        return None
    if (dest_r in (0, size - 1)) and (dest_c in (0, size - 1)):
        return None

    grid[mover_r, mover_c] = _MOVER
    grid[dest_r, dest_c] = _DEST

    traversable = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    dist = _bfs_pixels(traversable, (mover_r, mover_c))
    if dist[dest_r, dest_c] < 0:
        return None

    expected = _expected_output(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 300 attempts")
