# Variant of 332f06d7
# Dimension: D4 — 源对象形态 (src)
# Change: 黑色实心n*n正方形，尺寸n=1~4，位于通道内部但不接触网格边缘... → 粒子像素移动体

"""
Design analysis for the D4 variant.

- Concrete shape change:
  The original mover was an n×n solid square whose footprint itself encoded whether
  it could squeeze through the river. Here the mover is always a single black pixel.
  The red destination is also reduced to a single pixel so the input/output relation
  is visually legible from a few examples. The blue river remains a winding 4-connected
  path. In stuck examples, a short yellow run is painted directly on the blue path as a
  closed gate. Those gate pixels are visible as part of the corridor drawing but are not
  traversable.

- Behavior trace on one local segment:
  Original logic on a narrow section:
  `trav[r:r+n, c:c+n].all()` fails once the river becomes thinner than the square, so
  BFS over square placements stops before the choke.
  Variant logic on the analogous section:
  the river stays single-pixel passable in shape, but one or more path cells are recolored
  to the gate color. BFS now runs on plain pixels; it can step on river, source, and
  destination, but not on gate cells. The particle therefore stops on the last blue cell
  before the yellow gate. The difference is visible in the input itself: geometry no
  longer narrows, the blocking cue is the marker color on the channel.

- Complexity calibration:
  Using only a single gate pixel would make gate markers visually easy to miss and would
  also collide with the "singleton object" detector. The generator therefore uses a gate
  run of length 2..4, while mover and destination remain the only singleton colors.
  This keeps the task readable and avoids ambiguity without adding extra object families.

- Hidden assumptions reviewed against the old solver:
  The original `main()` depended on three old invariants:
  1. non-background components include solid same-sized squares,
  2. mover and destination are identified by that shared size and then by color ordering,
  3. traversability is tested with n×n windows.
  None of those remain valid after the D4 change. The new solver instead assumes:
  1. exactly one interior singleton non-background pixel is the mover,
  2. exactly one boundary singleton non-background pixel is the destination,
  3. among remaining colors, the most frequent one is the river,
  4. any extra color on the river is a closed gate and is therefore non-traversable.
  The generator explicitly guarantees these conditions.

- Placement legality:
  The generator builds an ordered simple path first, paints it blue, then optionally
  overwrites an interior sub-run with yellow gate pixels. The mover is placed at the
  first path cell, which is always one step inside the border; the destination is placed
  at the extended boundary endpoint. Gate cells are chosen away from both endpoints, so
  no overlap occurs and the solver sees a valid single start / single goal layout.

- Visual observability:
  Humans can infer the rule from examples because the only moving object is the lone
  black particle, the red endpoint sits on the border, and blocked examples contain an
  obvious yellow interruption on the blue route. The solver never sorts by color id;
  mover vs. destination is determined by interior-vs-boundary placement.

- Visual cleanliness:
  The path is a one-cell-wide snake with no branches and no self-overlap. That keeps the
  "farthest reachable" position unique and prevents layered clutter. The gate is a short
  contiguous run, so the blocking site reads as one coherent marker rather than noise.

- Color mapping discipline:
  Colors are fixed and sparse: background, river, mover, destination, optional gate.
  No hidden color permutation or id-based dispatch is required for either the user or
  the solver.
"""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

_BG = 3
_RIVER = 1
_MOVER = 0
_DEST = 2
_GATE = 4
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _is_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _bfs(mask: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _farthest_cell(dist: np.ndarray, goal: tuple[int, int]) -> tuple[int, int]:
    best = int(dist.max())
    candidates = np.argwhere(dist == best)
    dr, dc = goal
    candidates = sorted(
        ((int(r), int(c)) for r, c in candidates),
        key=lambda rc: (abs(rc[0] - dr) + abs(rc[1] - dc), rc[0], rc[1]),
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a single particle along a winding river, unless a gate marker blocks the route.

    Input structure:
    - The corner-majority background surrounds one long blue river.
    - Exactly one interior singleton pixel is the mover, and exactly one boundary singleton
      pixel is the destination.
    - Some inputs also place a short gate-colored segment on the river itself.

    Rule:
    - The mover may travel 4-neighbor through river cells and into the destination pixel.
    - Gate-colored pixels are drawn on the river but are closed, so the mover cannot step on them.
    - If the destination is reachable, the mover lands on it; otherwise it stops at the
      farthest river cell reachable before the gate.
    - The mover's original location is repainted as river.

    Key insight:
    The blocking mechanism is no longer geometric width. A visible gate marker on the
    channel now plays the role that narrow chokepoints used to play for large squares.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    counts = Counter(map(int, g.flatten()))

    singleton_points: list[tuple[int, int, int]] = []
    for color, count in counts.items():
        if int(color) == bg or int(count) != 1:
            continue
        loc = np.argwhere(g == int(color))
        assert len(loc) == 1
        r, c = map(int, loc[0])
        singleton_points.append((int(color), r, c))

    mover_candidates = [item for item in singleton_points if not _is_boundary((h, w), item[1], item[2])]
    dest_candidates = [item for item in singleton_points if _is_boundary((h, w), item[1], item[2])]
    assert len(mover_candidates) == 1
    assert len(dest_candidates) == 1
    mover_col, sr, sc = mover_candidates[0]
    dest_col, dr, dc = dest_candidates[0]

    river = None
    for color, _count in counts.most_common():
        if int(color) in {bg, mover_col, dest_col}:
            continue
        river = int(color)
        break
    assert river is not None

    traversable = (g == river) | (g == mover_col) | (g == dest_col)
    assert traversable[sr, sc]
    dist = _bfs(traversable, sr, sc)

    if int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
    else:
        tr, tc = _farthest_cell(dist, (dr, dc))

    out = g.copy()
    out[sr, sc] = river
    out[tr, tc] = mover_col
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _compute_offsets(size: int, count: int) -> list[int] | None:
    if count == 1:
        return [size // 2]
    first = 1
    last = size - 2
    usable = last - first
    min_gap = 3
    if usable < min_gap * (count - 1):
        return None
    extra = usable - min_gap * (count - 1)
    gaps = [min_gap] * (count - 1)
    for i in range(extra):
        gaps[i % (count - 1)] += 1
    offsets = [first]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    return offsets


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


def _expand_walk(
    walk: list[tuple[int, int]],
    offsets_r: list[int],
    offsets_c: list[int],
    coarse_rows: int,
    coarse_cols: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    ordered = [(offsets_r[walk[0][0]], offsets_c[walk[0][1]])]
    for cr, cc in walk[1:]:
        tr, tc = offsets_r[cr], offsets_c[cc]
        r, c = ordered[-1]
        while (r, c) != (tr, tc):
            if r < tr:
                r += 1
            elif r > tr:
                r -= 1
            elif c < tc:
                c += 1
            else:
                c -= 1
            ordered.append((r, c))

    end_r, end_c = walk[-1]
    r, c = ordered[-1]
    h, w = shape
    if end_r == 0:
        while r > 0:
            r -= 1
            ordered.append((r, c))
    elif end_r == coarse_rows - 1:
        while r < h - 1:
            r += 1
            ordered.append((r, c))
    elif end_c == 0:
        while c > 0:
            c -= 1
            ordered.append((r, c))
    elif end_c == coarse_cols - 1:
        while c < w - 1:
            c += 1
            ordered.append((r, c))
    return ordered


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    return len(path) == len(set(path))


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    for r, c in path:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    size = random.randint(10, 20)
    h = w = size

    max_coarse = min(6, 1 + (size - 3) // 3)
    if max_coarse < 3:
        return None
    coarse_dim = random.randint(3, max_coarse)
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(size, coarse_rows)
    offsets_c = _compute_offsets(size, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None

    edge_cells = set()
    for c in range(1, coarse_cols - 1):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    if len(edge_cells) < 2:
        return None
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda cell: -(abs(cell[0] - start_coarse[0]) + abs(cell[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 5:
        return None
    if _count_turns(walk) < 3:
        return None

    path = _expand_walk(walk, offsets_r, offsets_c, coarse_rows, coarse_cols, (h, w))
    if len(path) < max(12, size):
        return None
    if not _is_simple_path(path):
        return None

    mover_r, mover_c = path[0]
    dest_r, dest_c = path[-1]
    if _is_boundary((h, w), mover_r, mover_c):
        return None
    if not _is_boundary((h, w), dest_r, dest_c):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    _paint_path(grid, path, _RIVER)

    gate_start = None
    gate_len = 0
    if not reached:
        max_gate_len = min(4, len(path) // 5)
        if max_gate_len < 2:
            return None
        gate_len = random.randint(2, max_gate_len)
        lo = max(3, len(path) // 4)
        hi = len(path) - gate_len - 3
        if hi < lo:
            return None
        gate_start = random.randint(lo, hi)
        for idx in range(gate_start, gate_start + gate_len):
            gr, gc = path[idx]
            grid[gr, gc] = _GATE

    grid[mover_r, mover_c] = _MOVER
    grid[dest_r, dest_c] = _DEST

    if reached:
        tr, tc = dest_r, dest_c
    else:
        assert gate_start is not None
        tr, tc = path[gate_start - 1]

    output = grid.copy()
    output[mover_r, mover_c] = _RIVER
    output[tr, tc] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
