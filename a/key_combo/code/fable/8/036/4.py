from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import neighbors4
except Exception:
    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, list[int]]] = []

    for axis, length in ((0, H), (1, W)):
        for color in np.unique(grid):
            color = int(color)
            indices: list[int] = []
            for idx in range(length):
                line = grid[idx, :] if axis == 0 else grid[:, idx]
                if np.all(line == color):
                    indices.append(idx)
            if len(indices) < 2:
                continue
            panel_size = indices[0]
            if panel_size <= 0:
                continue
            if any(indices[i + 1] - indices[i] != panel_size + 1 for i in range(len(indices) - 1)):
                continue
            if length - 1 - indices[-1] != panel_size:
                continue
            panel_count = len(indices) + 1
            if panel_count < 3:
                continue
            candidates.append((axis, color, panel_size, indices))

    assert len(candidates) == 1, "expected a unique panel-sequence split"
    axis, separator_color, panel_size, _indices = candidates[0]
    panel_count = len(_indices) + 1

    panels: list[np.ndarray] = []
    for i in range(panel_count):
        start = i * (panel_size + 1)
        if axis == 0:
            panels.append(grid[start : start + panel_size, :].copy())
        else:
            panels.append(grid[:, start : start + panel_size].copy())
    return panels, int(axis), int(separator_color)


def _infer_flood_color(panels: list[np.ndarray]) -> int:
    colors = sorted({int(v) for panel in panels for v in panel.flat})
    counts_by_color = {color: [int(np.count_nonzero(panel == color)) for panel in panels] for color in colors}
    increasing = [color for color, counts in counts_by_color.items() if all(b > a for a, b in zip(counts, counts[1:]))]
    assert len(increasing) == 1, "expected a unique strictly growing flood color"
    return int(increasing[0])


def _bfs_dist(passable: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    H, W = passable.shape
    dist = np.full((H, W), -1, dtype=int)
    sr, sc = seed
    assert passable[sr, sc], "seed must lie on a passable channel"
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    dist[sr, sc] = 0

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < H and 0 <= cc < W and passable[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _schedule_from_candidate(
    panels: list[np.ndarray], flood_color: int, wall_color: int
) -> dict[str, object] | None:
    wall_mask = panels[0] == wall_color
    if any(not np.array_equal(panel == wall_color, wall_mask) for panel in panels[1:]):
        return None

    flood_masks = [panel == flood_color for panel in panels]
    if any(np.any(mask & wall_mask) for mask in flood_masks):
        return None

    seed_coords = np.argwhere(flood_masks[0])
    if seed_coords.shape[0] != 1:
        return None
    seed = tuple(map(int, seed_coords[0]))

    passable = ~wall_mask
    dist = _bfs_dist(passable, seed)
    thresholds: list[int] = []

    for mask in flood_masks:
        values = dist[mask]
        if values.size == 0 or np.any(values < 0):
            return None
        threshold = int(values.max())
        expected = (dist >= 0) & (dist <= threshold)
        if not np.array_equal(mask, expected):
            return None
        thresholds.append(threshold)

    diffs = [b - a for a, b in zip(thresholds, thresholds[1:])]
    if thresholds[0] != 0 or not diffs or len(set(diffs)) != 1 or diffs[0] <= 0:
        return None

    step = int(diffs[0])
    next_threshold = thresholds[-1] + step
    next_mask = (dist >= 0) & (dist <= next_threshold)
    if np.array_equal(next_mask, flood_masks[-1]):
        return None

    return {
        "seed": seed,
        "dist": dist,
        "step": step,
        "wall_mask": wall_mask,
        "next_mask": next_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next flood frame by continuing the same channel-distance advance through the repeated maze panels.

    Input structure:
    - The grid is a sequence of three or more equal panels arranged in one row or one column, separated by full divider lines.
    - Every panel contains the same wall maze, with all non-wall cells forming background channels.
    - One flood color starts from a single seed cell and occupies a larger channel-distance ball in each later panel.

    Rule:
    - Split the input into equal panels and identify the unique color whose occupied area grows strictly from panel to panel.
    - Determine which remaining panel color is the wall color by checking which choice makes every frame equal to all channel cells whose four-neighbor maze distance from the seed is at most k*s for a fixed positive step s.
    - Copy the last panel and expand the flood by one more step s through passable channel cells only, leaving walls unchanged.

    Key insight:
    - The changing boundary is not a plain Manhattan dilation in the whole rectangle; it is a fixed-speed wave measured on the maze's channel graph, so turns and dead ends reveal the true next frame.
    """
    g = np.asarray(grid, dtype=int)
    panels, _axis, _separator = _split_panels(g)
    flood_color = _infer_flood_color(panels)
    candidate_walls = sorted({int(v) for panel in panels for v in panel.flat if int(v) != flood_color})

    fits: list[tuple[int, dict[str, object]]] = []
    for wall_color in candidate_walls:
        info = _schedule_from_candidate(panels, flood_color, wall_color)
        if info is not None:
            fits.append((wall_color, info))

    assert len(fits) == 1, "expected a unique wall color consistent with a fixed flood schedule"
    wall_color, info = fits[0]

    out = panels[-1].copy()
    out[np.asarray(info["next_mask"], dtype=bool)] = flood_color
    out[np.asarray(info["wall_mask"], dtype=bool)] = wall_color
    return out


MAX_ATTEMPTS = 400
PANEL_COUNT_CHOICES = (3, 4, 5)
STEP_WEIGHTS = {1: 2, 2: 3, 3: 3}


def _carve_maze(cell_h: int, cell_w: int) -> np.ndarray:
    h = 2 * cell_h + 1
    w = 2 * cell_w + 1
    maze = np.zeros((h, w), dtype=bool)
    visited = np.zeros((cell_h, cell_w), dtype=bool)
    stack = [(random.randrange(cell_h), random.randrange(cell_w))]
    visited[stack[0]] = True

    while stack:
        cr, cc = stack[-1]
        maze[2 * cr + 1, 2 * cc + 1] = True
        neighbors: list[tuple[int, int, int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))
        if not neighbors:
            stack.pop()
            continue

        nr, nc, dr, dc = random.choice(neighbors)
        visited[nr, nc] = True
        maze[2 * cr + 1, 2 * cc + 1] = True
        maze[2 * nr + 1, 2 * nc + 1] = True
        maze[2 * cr + 1 + dr, 2 * cc + 1 + dc] = True
        stack.append((nr, nc))

    extra_edges: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if maze[r, c]:
                continue
            if maze[r - 1, c] and maze[r + 1, c] and not (maze[r, c - 1] or maze[r, c + 1]):
                extra_edges.append((r, c))
            elif maze[r, c - 1] and maze[r, c + 1] and not (maze[r - 1, c] or maze[r + 1, c]):
                extra_edges.append((r, c))
    random.shuffle(extra_edges)
    for r, c in extra_edges[: random.randint(0, min(3, len(extra_edges)))]:
        maze[r, c] = True

    return maze


def _render_panel(passable: np.ndarray, bg: int, wall: int, flood: int, flooded: np.ndarray) -> np.ndarray:
    panel = np.full(passable.shape, wall, dtype=int)
    panel[passable] = bg
    panel[flooded] = flood
    return panel


def _assemble_input(panels: list[np.ndarray], separator: int, axis: int) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    count = len(panels)
    if axis == 1:
        out = np.full((panel_h, count * panel_w + count - 1), separator, dtype=int)
        for i, panel in enumerate(panels):
            c0 = i * (panel_w + 1)
            out[:, c0 : c0 + panel_w] = panel
        return out

    out = np.full((count * panel_h + count - 1, panel_w), separator, dtype=int)
    for i, panel in enumerate(panels):
        r0 = i * (panel_h + 1)
        out[r0 : r0 + panel_h, :] = panel
    return out


def _valid_layout(axis: int, panel_count: int, panel_h: int, panel_w: int) -> bool:
    if axis == 1:
        return panel_h <= 30 and panel_count * panel_w + panel_count - 1 <= 30
    return panel_w <= 30 and panel_count * panel_h + panel_count - 1 <= 30


def _choose_geometry() -> tuple[int, int, int, int]:
    axis = random.choice((0, 1))
    panel_count = random.choice(PANEL_COUNT_CHOICES)

    for _ in range(100):
        if axis == 1:
            cell_h = random.randint(2, 5)
            cell_w = random.randint(2, 4)
        else:
            cell_h = random.randint(2, 4)
            cell_w = random.randint(2, 5)
        panel_h = 2 * cell_h + 1
        panel_w = 2 * cell_w + 1
        if _valid_layout(axis, panel_count, panel_h, panel_w):
            return axis, panel_count, cell_h, cell_w
    raise RuntimeError("failed to choose a valid panel geometry")


def _semantic_checks(
    maze: np.ndarray,
    dist: np.ndarray,
    seed: tuple[int, int],
    step: int,
    panel_count: int,
) -> bool:
    dry_cells = int(np.count_nonzero((dist >= 0) & (dist > panel_count * step)))
    if dry_cells <= 0:
        return False

    last_threshold = (panel_count - 1) * step
    next_threshold = panel_count * step
    if np.count_nonzero(dist == next_threshold) == 0:
        return False
    if np.count_nonzero(dist == last_threshold) == 0:
        return False

    sr, sc = seed
    manhattan = np.fromfunction(lambda r, c: np.abs(r - sr) + np.abs(c - sc), maze.shape, dtype=int)
    true_next = (dist >= 0) & (dist <= next_threshold)
    naive_next = maze & (manhattan <= next_threshold)
    if np.array_equal(true_next, naive_next):
        return False

    degrees = np.zeros_like(dist)
    for r, c in np.argwhere(maze):
        degrees[r, c] = sum(
            1
            for rr, cc in neighbors4(int(r), int(c))
            if 0 <= rr < maze.shape[0] and 0 <= cc < maze.shape[1] and maze[rr, cc]
        )
    if not np.any((dist >= 0) & (dist <= next_threshold) & (degrees >= 3)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    axis, panel_count, cell_h, cell_w = _choose_geometry()
    maze = _carve_maze(cell_h, cell_w)
    passable_coords = np.argwhere(maze)
    random.shuffle(passable_coords)

    seed = None
    dist = None
    for coord in passable_coords:
        candidate_seed = tuple(map(int, coord))
        candidate_dist = _bfs_dist(maze, candidate_seed)
        dmax = int(candidate_dist.max())
        if dmax >= panel_count + 2:
            seed = candidate_seed
            dist = candidate_dist
            break
    if seed is None or dist is None:
        return None

    max_step = min(3, int(dist.max()) // panel_count)
    options = [s for s in range(1, max_step + 1) if panel_count * s < int(dist.max())]
    if not options:
        return None
    weighted = [s for s in options for _ in range(STEP_WEIGHTS.get(s, 1))]
    step = random.choice(weighted)
    if not _semantic_checks(maze, dist, seed, step, panel_count):
        return None

    colors = random.sample(range(10), 4)
    bg, wall, flood, separator = map(int, colors)

    panels = []
    for t in range(panel_count):
        flooded = (dist >= 0) & (dist <= t * step)
        panels.append(_render_panel(maze, bg, wall, flood, flooded))
    inp = _assemble_input(panels, separator, axis)
    expected = _render_panel(maze, bg, wall, flood, (dist >= 0) & (dist <= panel_count * step))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-flood panel sequence")
