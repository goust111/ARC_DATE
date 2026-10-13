import heapq

import numpy as np


BG = 0
PRISM = 5
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 5000


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _perpendicular_dirs(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    prism_positions = {tuple(map(int, pt)) for pt in np.argwhere(g == PRISM)}
    marker_positions: set[tuple[int, int]] = set()
    prism_filter: dict[tuple[int, int], int] = {}

    for pr, pc in prism_positions:
        colored_neighbors: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            nr, nc = pr + dr, pc + dc
            if _inside(h, w, nr, nc) and int(g[nr, nc]) not in (BG, PRISM):
                colored_neighbors.append((nr, nc))
        assert len(colored_neighbors) == 1, "each prism must have exactly one filter marker"
        marker = colored_neighbors[0]
        assert marker not in marker_positions, "a filter marker may not belong to two prisms"
        marker_positions.add(marker)
        prism_filter[(pr, pc)] = int(g[marker])

    towers: list[tuple[int, int]] = []
    tower_colors: set[int] = set()
    for pt in np.argwhere((g != BG) & (g != PRISM)):
        pos = (int(pt[0]), int(pt[1]))
        if pos in marker_positions:
            continue
        color = int(g[pos])
        assert color not in tower_colors, "tower colors must be distinct"
        tower_colors.add(color)
        towers.append(pos)

    towers.sort()
    tower_index = {pos: idx for idx, pos in enumerate(towers)}
    assert towers, "expected at least one signal tower"
    return {
        "grid": g,
        "towers": towers,
        "tower_index": tower_index,
        "marker_positions": marker_positions,
        "prisms": prism_positions,
        "prism_filter": prism_filter,
    }


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    scene = _parse_scene(grid)
    g = np.asarray(scene["grid"], dtype=int)
    towers = list(scene["towers"])
    tower_index = dict(scene["tower_index"])
    markers = set(scene["marker_positions"])
    prisms = set(scene["prisms"])
    prism_filter = dict(scene["prism_filter"])

    h, w = g.shape
    out = g.copy()
    queue: list[tuple[int, int, int, int, int, int, int, int]] = []
    push_id = 0
    for tower in towers:
        color = int(g[tower])
        idx = int(tower_index[tower])
        for dr, dc in DIRS:
            heapq.heappush(queue, (0, idx, push_id, tower[0], tower[1], dr, dc, color))
            push_id += 1

    prism_claim: dict[tuple[int, int], tuple[int, int]] = {}
    conflicts: set[tuple[int, int]] = set()
    stats = {
        "boundary_hits": 0,
        "tower_hits": 0,
        "prism_triggered": 0,
        "suppressed_prism_hits": 0,
        "marker_passes": 0,
        "painted_cells": 0,
        "conflict_count": 0,
    }

    while queue:
        dist, tower_idx, _order, r, c, dr, dc, color = heapq.heappop(queue)
        nr, nc = r + dr, c + dc
        if not _inside(h, w, nr, nc):
            stats["boundary_hits"] += 1
            continue

        nxt = (nr, nc)
        if nxt in tower_index:
            stats["tower_hits"] += 1
            continue

        if nxt in prisms:
            priority = (dist + 1, tower_idx)
            if nxt not in prism_claim:
                prism_claim[nxt] = priority
                stats["prism_triggered"] += 1
                new_color = int(prism_filter[nxt])
                for ndr, ndc in _perpendicular_dirs(dr, dc):
                    heapq.heappush(queue, (dist + 1, tower_idx, push_id, nr, nc, ndr, ndc, new_color))
                    push_id += 1
            else:
                stats["suppressed_prism_hits"] += 1
            continue

        cell = int(g[nr, nc])
        if cell == BG:
            prev = int(out[nr, nc])
            if prev == BG:
                out[nr, nc] = int(color)
                stats["painted_cells"] += 1
            elif prev != int(color):
                conflicts.add(nxt)
            heapq.heappush(queue, (dist + 1, tower_idx, push_id, nr, nc, dr, dc, color))
            push_id += 1
            continue

        assert nxt in markers, "unexpected non-black blocker"
        stats["marker_passes"] += 1
        heapq.heappush(queue, (dist + 1, tower_idx, push_id, nr, nc, dr, dc, color))
        push_id += 1

    stats["conflict_count"] = len(conflicts)
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Draw tower rays that are recolored and redirected by first-hit prisms."""
    out, _stats = _simulate(grid)
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_gap_pair() -> tuple[int, int, str]:
    mode = int(np.random.randint(3))
    if mode == 0:
        before = _randint(2, 4)
        after = _randint(before + 1, min(before + 3, 6))
        return before, after, "left"
    if mode == 1:
        after = _randint(2, 4)
        before = _randint(after + 1, min(after + 3, 6))
        return before, after, "right"
    tie = _randint(2, 5)
    return tie, tie, "left"


def _build_linear_case() -> tuple[np.ndarray, int]:
    prism_count = 1 + int(np.random.rand() < 0.5)
    use_left_stopper = bool(np.random.rand() < 0.65)
    use_right_stopper = bool(np.random.rand() < 0.65)
    if not use_left_stopper and not use_right_stopper:
        if np.random.rand() < 0.5:
            use_left_stopper = True
        else:
            use_right_stopper = True

    tower_rel = [0]
    prism_rel: list[int] = []
    winners: list[str] = []
    cursor = 0
    for _ in range(prism_count):
        left_gap, right_gap, winner = _sample_gap_pair()
        cursor += left_gap
        prism_rel.append(cursor)
        cursor += right_gap
        tower_rel.append(cursor)
        winners.append(winner)

    left_stopper_rel = -_randint(2, 4) if use_left_stopper else None
    right_stopper_rel = cursor + _randint(2, 4) if use_right_stopper else None

    marker_rel = []
    for rel, winner in zip(prism_rel, winners):
        marker_rel.append(rel + 1 if winner == "left" else rel - 1)

    all_rel = tower_rel + prism_rel + marker_rel
    if left_stopper_rel is not None:
        all_rel.append(left_stopper_rel)
    if right_stopper_rel is not None:
        all_rel.append(right_stopper_rel)

    left_margin = _randint(1, 4)
    right_margin = _randint(1, 4)
    width = max(all_rel) - min(all_rel) + 1 + left_margin + right_margin
    height = _randint(9, 16)
    row = _randint(2, height - 3)
    offset = left_margin - min(all_rel)

    grid = np.full((height, width), BG, dtype=int)

    contest_towers = [(row, offset + rel) for rel in tower_rel]
    stoppers: list[tuple[int, int]] = []
    if left_stopper_rel is not None:
        stoppers.append((row, offset + left_stopper_rel))
    if right_stopper_rel is not None:
        stoppers.append((row, offset + right_stopper_rel))
    prisms = [(row, offset + rel) for rel in prism_rel]
    markers = [(row, offset + rel) for rel in marker_rel]

    total_colored = len(contest_towers) + len(stoppers) + len(markers)
    colors = list(map(int, np.random.choice(PALETTE, size=total_colored, replace=False)))
    idx = 0

    for r, c in contest_towers:
        grid[r, c] = colors[idx]
        idx += 1
    for r, c in stoppers:
        grid[r, c] = colors[idx]
        idx += 1
    for r, c in prisms:
        grid[r, c] = PRISM
    for r, c in markers:
        grid[r, c] = colors[idx]
        idx += 1

    if np.random.rand() < 0.5:
        grid = grid.T.copy()

    return grid, prism_count


def _is_interesting(grid: np.ndarray, prism_count: int) -> bool:
    try:
        out, stats = _simulate(grid)
    except AssertionError:
        return False

    if stats["prism_triggered"] != prism_count:
        return False
    if stats["suppressed_prism_hits"] < prism_count:
        return False
    if stats["tower_hits"] < 2:
        return False
    if stats["boundary_hits"] == 0:
        return False
    if stats["painted_cells"] < 10:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, prism_count = _build_linear_case()
        if _is_interesting(grid, prism_count):
            return grid
    raise RuntimeError("failed to generate a valid ray-prism puzzle input")
