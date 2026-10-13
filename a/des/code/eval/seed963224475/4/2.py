from __future__ import annotations

import random

import numpy as np


COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _find_legend_boxes(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    h, w = grid.shape
    groups: dict[tuple[int, int], list[int]] = {}
    for r in range(h - 2):
        for c in range(w - 2):
            sub = grid[r : r + 3, c : c + 3]
            border_vals = [
                int(sub[0, 0]), int(sub[0, 1]), int(sub[0, 2]),
                int(sub[1, 0]), int(sub[1, 2]),
                int(sub[2, 0]), int(sub[2, 1]), int(sub[2, 2]),
            ]
            border = border_vals[0]
            if border == bg or int(sub[1, 1]) != bg:
                continue
            if all(v == border for v in border_vals):
                groups.setdefault((c, border), []).append(r)

    assert groups, "expected at least one legend box"
    best_key = max(groups, key=lambda key: (len(groups[key]), key[0]))
    c0, path_color = best_key
    rows = sorted(groups[best_key])

    boxes: list[tuple[int, int]] = []
    for r in rows:
        if boxes and r < boxes[-1][0] + 3:
            continue
        boxes.append((int(r), int(c0)))

    assert boxes, "legend box detection failed"
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r, c in boxes:
        legend_mask[r : r + 3, c : c + 3] = True
    return boxes, legend_mask, int(path_color)


def _maze_details(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
    np.ndarray,
    list[tuple[int, int]],
    dict[int, tuple[int, int]],
    list[tuple[int, int]],
]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    boxes, legend_mask, path_color = _find_legend_boxes(g, bg)

    passable = (g != bg) & (~legend_mask)
    colors = [int(v) for v in np.unique(g) if int(v) not in (bg, path_color)]
    seed_pos: dict[int, tuple[int, int]] = {}
    for color in colors:
        pts = np.argwhere((g == color) & (~legend_mask))
        if pts.shape[0] == 1:
            seed_pos[color] = (int(pts[0, 0]), int(pts[0, 1]))

    assert seed_pos, "expected unique seed pixels"
    centers = [(r + 1, c + 1) for r, c in sorted(boxes)]
    return g, int(bg), int(path_color), passable, centers, seed_pos, boxes


def _simulate(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], dict[str, int]]:
    g, bg, path_color, passable, centers, seed_pos, _boxes = _maze_details(grid)
    h, w = g.shape
    state = g.copy()

    boundary = np.zeros_like(g, dtype=bool)
    frontier = [pos for _, pos in sorted(seed_pos.items(), key=lambda item: (item[1][0], item[1][1]))]

    tie_cells = 0
    dominant_wins = 0

    while frontier:
        claims: dict[tuple[int, int], set[int]] = {}
        for r, c in frontier:
            color = int(state[r, c])
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not boundary[rr, cc]:
                    if int(state[rr, cc]) == path_color:
                        claims.setdefault((rr, cc), set()).add(color)

        if not claims:
            break

        next_frontier: list[tuple[int, int]] = []
        for (r, c), colors in sorted(claims.items()):
            if len(colors) == 1:
                winner = next(iter(colors))
                state[r, c] = int(winner)
                next_frontier.append((r, c))
                continue

            best_colors: list[int] = []
            best_count = -1
            for color in sorted(colors):
                count = 0
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and int(state[rr, cc]) == int(color):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_colors = [int(color)]
                elif count == best_count:
                    best_colors.append(int(color))

            if len(best_colors) == 1:
                winner = best_colors[0]
                state[r, c] = int(winner)
                next_frontier.append((r, c))
                dominant_wins += 1
            else:
                boundary[r, c] = True
                tie_cells += 1

        frontier = next_frontier

    degrees = np.zeros_like(g, dtype=int)
    for r in range(h):
        for c in range(w):
            if not passable[r, c]:
                continue
            degrees[r, c] = sum(
                1
                for rr, cc in _neighbors4(r, c)
                if 0 <= rr < h and 0 <= cc < w and passable[rr, cc]
            )

    counts = {int(color): 0 for color in seed_pos}
    for r in range(h):
        for c in range(w):
            if passable[r, c] and degrees[r, c] == 1:
                color = int(state[r, c])
                if color in counts:
                    counts[color] += 1

    order = sorted(
        counts,
        key=lambda color: (-counts[color], seed_pos[color][0], seed_pos[color][1]),
    )
    out = state.copy()
    for (r, c), color in zip(centers, order, strict=False):
        out[r, c] = int(color)

    stats = {
        "tie_cells": int(tie_cells),
        "dominant_wins": int(dominant_wins),
        "passable_cells": int(passable.sum()),
        "colored_cells": int(sum(v != path_color for v in out[passable].tolist())),
    }
    return out, counts, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simultaneously spread each seed color through the maze, resolve same-step
    conflicts by already-colored neighbor counts, then rank colors by how many
    maze dead ends they own and place that order into the right-side legend.
    """
    out, _counts, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


def _carve_maze(cell_h: int, cell_w: int, rng: random.Random) -> np.ndarray:
    visited = np.zeros((cell_h, cell_w), dtype=bool)
    stack = [(rng.randrange(cell_h), rng.randrange(cell_w))]
    visited[stack[0]] = True
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < cell_h and 0 <= cc < cell_w and not visited[rr, cc]:
                neighbors.append((rr, cc))
        if not neighbors:
            stack.pop()
            continue
        rr, cc = rng.choice(neighbors)
        visited[rr, cc] = True
        a = (r, c)
        b = (rr, cc)
        edges.add(tuple(sorted((a, b))))
        stack.append((rr, cc))

    maze = np.zeros((2 * cell_h + 1, 2 * cell_w + 1), dtype=bool)
    for r in range(cell_h):
        for c in range(cell_w):
            maze[2 * r + 1, 2 * c + 1] = True
    for (r0, c0), (r1, c1) in edges:
        maze[2 * r0 + 1, 2 * c0 + 1] = True
        maze[2 * r1 + 1, 2 * c1 + 1] = True
        maze[r0 + r1 + 1, c0 + c1 + 1] = True
    return maze


def _passable_degrees(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros_like(mask, dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            deg[r, c] = sum(
                1
                for rr, cc in _neighbors4(r, c)
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]
            )
    return deg


def _legend_rows(height: int, count: int, gap: int, rng: random.Random) -> list[int] | None:
    total = count * 3 + (count - 1) * gap
    if total > height:
        return None
    top = rng.randint(0, height - total)
    return [top + i * (3 + gap) for i in range(count)]


def _build_candidate_grid(
    maze_mask: np.ndarray,
    seed_positions: list[tuple[int, int]],
    seed_colors: list[int],
    legend_gap_left: int,
    legend_gap_vertical: int,
    right_margin: int,
    rng: random.Random,
) -> np.ndarray | None:
    maze_h, maze_w = maze_mask.shape
    legend_rows = _legend_rows(maze_h, len(seed_positions), legend_gap_vertical, rng)
    if legend_rows is None:
        return None

    legend_c = maze_w + legend_gap_left
    total_w = legend_c + 3 + right_margin
    grid = np.zeros((maze_h, total_w), dtype=int)
    grid[:, :maze_w][maze_mask] = 5

    for (r, c), color in zip(seed_positions, seed_colors, strict=False):
        grid[r, c] = int(color)

    for r in legend_rows:
        grid[r : r + 3, legend_c : legend_c + 3] = 5
        grid[r + 1, legend_c + 1] = 0

    return grid


def _seedable_dead_ends(maze_mask: np.ndarray) -> list[tuple[int, int]]:
    deg = _passable_degrees(maze_mask)
    pts = [
        (int(r), int(c))
        for r, c in zip(*np.where(maze_mask), strict=False)
        if int(deg[r, c]) == 1
    ]
    pts.sort()
    return pts


def _interesting(stats: dict[str, int], target_mode: str) -> bool:
    if target_mode == "tie":
        return stats["tie_cells"] > 0
    if target_mode == "dominant":
        return stats["dominant_wins"] > 0
    return stats["tie_cells"] > 0 and stats["dominant_wins"] > 0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(6, 9)
    cell_w = rng.randint(7, 10)
    maze_mask = _carve_maze(cell_h, cell_w, rng)
    leaves = _seedable_dead_ends(maze_mask)
    if len(leaves) < 5:
        return None

    maze_h = maze_mask.shape[0]
    max_boxes = (maze_h + 1) // 4
    max_seed_count = min(5, max_boxes, len(leaves) - 1)
    if max_seed_count < 3:
        return None

    target_mode = rng.choice(("tie", "dominant", "both"))
    tries = 60
    for _ in range(tries):
        seed_count = rng.randint(3, max_seed_count)
        seeds = sorted(rng.sample(leaves, seed_count))
        if len(set(seeds)) != seed_count:
            continue
        colors = rng.sample(list(COLOR_POOL), seed_count)
        legend_gap_left = rng.choice((1, 2))
        legend_gap_vertical = rng.choice((1, 2))
        right_margin = rng.choice((1, 2))
        grid = _build_candidate_grid(
            maze_mask,
            seeds,
            colors,
            legend_gap_left,
            legend_gap_vertical,
            right_margin,
            rng,
        )
        if grid is None:
            continue

        out, counts, stats = _simulate(grid)
        if not _interesting(stats, target_mode):
            continue
        dead_end_counts = sorted(counts.values(), reverse=True)
        if len(set(dead_end_counts)) != len(dead_end_counts):
            continue
        if dead_end_counts[0] <= 1:
            continue
        if int(np.sum(out != grid)) < seed_count + 8:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
