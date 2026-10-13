"""Trace labeled grey pipelines, recolor them by endpoint labels, and resolve intersections by legend priority.

Input structure:
- The grid uses a dominant background color.
- One color forms one-cell-wide grey pipe paths composed of horizontal/vertical segments with 90° turns.
- Every pipe has two endpoint-adjacent label pixels of the same unique color, each on the side opposite to the pipe direction at a border endpoint.
- The same label colors appear once more in a vertical priority legend near the top-left; higher legend rows mean higher overlay priority.
- Pipe pixels can overlap at crossing points, while labels and the legend must remain unchanged.

Rule:
1. Identify the background color, the grey path color, and all label colors by their adjacency to grey pixels and to the top-left priority legend.
2. For each label color, recover both endpoint-adjacent grey pixels and trace the connected grey path from one endpoint to the other by continuing straight at crossings.
3. Repaint every traced grey pixel with that pipe's label color, keeping for each pixel the color with the highest priority at intersections.
4. Leave labels, legend, and non-path cells unchanged.

Key insight:
The two endpoint labels establish pipe ownership, while the legend gives the global cover order to resolve ambiguous colors at shared crossing pixels.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DIRECTIONS: Sequence[Tuple[int, int]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def _inside(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(pos: Tuple[int, int], h: int, w: int) -> List[Tuple[int, int]]:
    r, c = pos
    out: List[Tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if _inside(nr, nc, h, w):
            out.append((nr, nc))
    return out


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _shortest_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    h: int,
    w: int,
) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not _inside(nr, nc, h, w) or (nr, nc) in blocked or (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                q.clear()
                break
            q.append((nr, nc))

    if goal not in parent:
        return None

    path: List[Tuple[int, int]] = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _edge_inward_step(pos: Tuple[int, int], h: int, w: int) -> Tuple[int, int]:
    r, c = pos
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    if c == w - 1:
        return (0, -1)
    return (0, 0)


def _border_endpoint_candidates(h: int, w: int, blocked_rows: Iterable[int]) -> List[Tuple[int, int]]:
    blocked_rows = set(blocked_rows)
    out: List[Tuple[int, int]] = []

    for c in range(1, w - 1):
        out.append((0, c))
    for c in range(1, w - 1):
        out.append((h - 1, c))
    for r in range(1, h - 1):
        if r not in blocked_rows:
            out.append((r, 0))
            out.append((r, w - 1))

    return out


def _choose_label_for_endpoint(
    endpoint: Tuple[int, int],
    inward: Tuple[int, int],
    forbidden: set[Tuple[int, int]],
    h: int,
    w: int,
    path_cells: set[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    r, c = endpoint
    dr, dc = inward
    candidates: List[Tuple[int, int]] = []
    for cand_dr, cand_dc in DIRECTIONS:
        if (cand_dr, cand_dc) == (dr, dc):
            continue
        nr, nc = r + cand_dr, c + cand_dc
        if not _inside(nr, nc, h, w):
            continue
        if (nr, nc) in forbidden:
            continue
        if (nr, nc) in path_cells:
            continue
        candidates.append((nr, nc))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _touches_grey(mask: np.ndarray, pos: Tuple[int, int], h: int, w: int) -> bool:
    return any(mask[nr, nc] for nr, nc in _neighbors(pos, h, w))


def _trace_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grey_mask: np.ndarray,
) -> Optional[List[Tuple[int, int]]]:
    h, w = grey_mask.shape
    if not grey_mask[start] or not grey_mask[end]:
        return None

    path: List[Tuple[int, int]] = [start]
    cur = start
    prev: Optional[Tuple[int, int]] = None
    visited = {start}

    for _ in range(h * w * 2):
        if cur == end:
            return path

        neigh = [
            (nr, nc)
            for nr, nc in _neighbors(cur, h, w)
            if grey_mask[nr, nc] and (prev is None or (nr, nc) != prev)
        ]

        if not neigh:
            return None

        nxt: Optional[Tuple[int, int]] = None
        if prev is not None:
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            straight = (cur[0] + dr, cur[1] + dc)
            if 0 <= straight[0] < h and 0 <= straight[1] < w and grey_mask[straight] and straight != prev:
                if straight in neigh:
                    nxt = straight
        if nxt is None and len(neigh) == 1:
            nxt = neigh[0]
        if nxt is None:
            # fallback: choose the neighbor closest to the end, avoiding early loops.
            neigh = [n for n in neigh if n not in visited or n == end]
            if not neigh:
                return None
            neigh.sort(key=lambda p: abs(p[0] - end[0]) + abs(p[1] - end[1]))
            nxt = neigh[0]

        if nxt in visited and nxt != end:
            return None

        prev = cur
        cur = nxt
        visited.add(cur)
        path.append(cur)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = np.array(grid, copy=True)
    h, w = out.shape

    bg = _background_color(out)
    colors, counts = np.unique(out, return_counts=True)
    if colors.size <= 1:
        return out

    non_bg_colors = [int(c) for c in colors if int(c) != bg]
    if not non_bg_colors:
        return out

    # Grey is the dominant non-background color.
    grey_color = non_bg_colors[np.argmax([int(np.sum(out == c)) for c in non_bg_colors])]
    grey_mask = out == grey_color
    if not grey_mask.any():
        return out

    label_info: Dict[int, Dict[str, list[tuple[int, int]]]] = {}
    for c in non_bg_colors:
        if c == grey_color:
            continue

        pos = [tuple(int(x) for x in p) for p in np.argwhere(out == c)]
        if len(pos) < 2:
            continue

        endpoint_pos: List[Tuple[int, int]] = []
        legend_pos: List[Tuple[int, int]] = []
        for p in pos:
            if _touches_grey(grey_mask, p, h, w):
                endpoint_pos.append(p)
            else:
                legend_pos.append(p)

        if len(endpoint_pos) == 2:
            label_info[c] = {"endpoints": endpoint_pos, "legend": legend_pos}

    if not label_info:
        return out

    # Legend order by top-to-bottom position (higher priority on top).
    ranked = []
    unranked = []
    for c, info in label_info.items():
        legend = info["legend"]
        if len(legend) == 1:
            r, _ = legend[0]
            ranked.append((r, c))
        else:
            unranked.append(c)
    ranked.sort()
    priority: Dict[int, int] = {}
    for i, (_, c) in enumerate(ranked):
        priority[c] = i
    base = len(ranked)
    for j, c in enumerate(sorted(unranked)):
        priority[c] = base + j

    best_rank = np.full((h, w), fill_value=10**9, dtype=np.int32)
    best_color = np.full((h, w), -1, dtype=np.int32)

    for c, info in label_info.items():
        endpoints = info["endpoints"]
        end_greys: List[Tuple[int, int]] = []
        for ep in endpoints:
            nbrs = [n for n in _neighbors(ep, h, w) if grey_mask[n]]
            if not nbrs:
                break
            # choose the neighbor nearest to the opposite endpoint when ambiguous.
            if len(end_greys) == 0:
                chosen = nbrs[0]
            else:
                er, ec = endpoints[1]
                nbrs.sort(key=lambda p: abs(p[0] - er) + abs(p[1] - ec))
                chosen = nbrs[0]
            end_greys.append(chosen)
        if len(end_greys) != 2:
            continue

        path = _trace_path(end_greys[0], end_greys[1], grey_mask)
        if path is None:
            continue

        rank = priority.get(c, len(priority))
        for p in path:
            if rank < best_rank[p]:
                best_rank[p] = rank
                best_color[p] = c

    for r in range(h):
        for cc in range(w):
            if best_color[r, cc] >= 0:
                out[r, cc] = best_color[r, cc]

    return out


def generate_input() -> np.ndarray:
    random.seed()
    # Palette: keep ARC color IDs but avoid using numeric order as semantics.
    bg = 0
    grey = 8
    palette = [1, 2, 3, 4, 5, 6, 7, 9]

    while True:
        h = random.randint(12, 28)
        w = random.randint(12, 28)
        num_pipes = random.randint(2, 5)
        if h < num_pipes + 4:
            continue

        if w < 12:
            continue

        colors = random.sample(palette, num_pipes)

        # Shuffle color order to create independent priority pattern.
        random.shuffle(colors)

        grid = np.full((h, w), bg, dtype=np.int64)
        legend_rows = list(range(1, num_pipes + 1))
        if any(r >= h for r in legend_rows):
            continue

        blocked_cells: set[tuple[int, int]] = set()
        for i, c in enumerate(colors):
            r = legend_rows[i]
            grid[r, 0] = c
            blocked_cells.add((r, 0))

        candidate_endpoints = _border_endpoint_candidates(h, w, legend_rows)
        if len(candidate_endpoints) < 2 * num_pipes:
            continue

        occupied_endpoints: set[tuple[int, int]] = set()
        all_path_cells: set[tuple[int, int]] = set()
        success = True

        for pi, c in enumerate(colors):
            placed = False
            for _attempt in range(300):
                start, end = random.sample(candidate_endpoints, 2)
                if start in occupied_endpoints or end in occupied_endpoints:
                    continue

                start_inner = (start[0] + _edge_inward_step(start, h, w)[0], start[1] + _edge_inward_step(start, h, w)[1])
                end_inner = (end[0] + _edge_inward_step(end, h, w)[0], end[1] + _edge_inward_step(end, h, w)[1])
                if not _inside(*start_inner, h, w) or not _inside(*end_inner, h, w):
                    continue
                if start_inner in blocked_cells or end_inner in blocked_cells:
                    continue

                cross_prob = 0.5
                interior: Optional[List[Tuple[int, int]]] = None

                if all_path_cells and random.random() < cross_prob:
                    anchor = random.choice(tuple(all_path_cells))
                    p1 = _shortest_path(start_inner, anchor, blocked_cells, h, w)
                    if p1 is not None:
                        # Allow overlap on existing paths so the anchor truly creates a crossing.
                        p2 = _shortest_path(anchor, end_inner, blocked_cells, h, w)
                        if p2 is not None:
                            if len(p1) > 0 and len(p2) > 0:
                                interior = p1 + p2[1:]

                if interior is None:
                    interior = _shortest_path(start_inner, end_inner, blocked_cells, h, w)

                if interior is None or len(interior) < 1:
                    continue

                full_path: List[Tuple[int, int]] = [start] + interior + [end]
                path_cells = set(full_path)

                s_in = _edge_inward_step(start, h, w)
                e_in = _edge_inward_step(end, h, w)
                lab1 = _choose_label_for_endpoint(start, s_in, blocked_cells, h, w, path_cells)
                lab2 = _choose_label_for_endpoint(end, e_in, blocked_cells, h, w, path_cells)
                if lab1 is None or lab2 is None:
                    continue
                if lab1 == lab2 or lab1 == start or lab1 == end or lab2 == start or lab2 == end:
                    continue

                # Apply this pipe.
                for rr, cc in full_path:
                    grid[rr, cc] = grey
                grid[lab1] = c
                grid[lab2] = c

                blocked_cells.add(lab1)
                blocked_cells.add(lab2)
                occupied_endpoints.add(start)
                occupied_endpoints.add(end)
                all_path_cells.update(path_cells)
                placed = True
                break

            if not placed:
                success = False
                break

        if not success:
            continue

        # Post-check to avoid ambiguous generation.
        valid = True
        for c in colors:
            if np.sum(grid == c) != 3:
                valid = False
                break
        if not valid:
            continue

        # Ensure every label color can be recovered and every pipe has route shape.
        if np.sum(grid == grey) == 0:
            continue

        return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print("shape", sample.shape)
