from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


SEED_COLORS = (1, 2, 3)
OBSTACLE = 5
EDGE_FILL = 7
ENCLOSED_FILL = 4
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _grow_territories(grid: np.ndarray, *, conflicts: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    frontiers = {
        color: {tuple(map(int, p)) for p in np.argwhere(out == color)}
        for color in SEED_COLORS
        if np.any(out == color)
    }

    while frontiers:
        claims: dict[tuple[int, int], set[int]] = defaultdict(set)
        for color, cells in frontiers.items():
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                        claims[(nr, nc)].add(color)

        if not claims:
            break

        next_frontiers = {color: set() for color in frontiers}
        for (r, c), colors in claims.items():
            if conflicts and len(colors) >= 2:
                out[r, c] = OBSTACLE
                continue
            color = min(colors)
            out[r, c] = color
            next_frontiers.setdefault(color, set()).add((r, c))

        frontiers = {color: cells for color, cells in next_frontiers.items() if cells}

    return out


def _recolor_voids(grown: np.ndarray) -> np.ndarray:
    out = grown.copy()
    for coords in _components_mask(out == 0):
        fill = EDGE_FILL if _touches_edge(coords, out.shape) else ENCLOSED_FILL
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all seed colors simultaneously, turn growth collisions into grey walls, then classify the remaining black gaps.

    Input structure:
    - The grid is black background with 3-5 short grey wall segments and 2-3 isolated seed pixels.
    - Seed pixels use distinct colors chosen from blue, red, and green.
    - Grey pixels are fixed obstacles; black pixels are empty space that may be claimed during growth.

    Rule:
    - Expand every seed color synchronously through 4-neighbor black cells.
    - If one step gives a black cell claims from two or more different seed colors, that cell becomes grey instead of belonging to any territory.
    - Continue until no seed can claim any further black cell.
    - Split the remaining black cells into 4-connected regions and recolor each edge-touching region orange, while recoloring each fully enclosed region yellow.
    - Keep every grey obstacle, seed cell, and grown territory cell unchanged.

    Key insight:
    - The decisive structure is not just the original walls: simultaneous growth creates new grey blockers, and those emergent blockers can seal off black cavities that were originally open.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grown = _grow_territories(g, conflicts=True)
    return _recolor_voids(grown)


def _paint_segment(
    grid: np.ndarray,
    segments: list[tuple[str, int, int, int]],
    *,
    orientation: str,
    r: int,
    c: int,
    length: int,
) -> None:
    if orientation == "h":
        grid[r, c : c + length] = OBSTACLE
    else:
        grid[r : r + length, c] = OBSTACLE
    segments.append((orientation, r, c, length))


def _seed_is_valid(grid: np.ndarray, seeds: list[tuple[int, int]], r: int, c: int) -> bool:
    if grid[r, c] != 0:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == OBSTACLE:
            return False
    return all(abs(r - sr) + abs(c - sc) > 1 for sr, sc in seeds)


def _obstacle_components_are_short_lines(grid: np.ndarray) -> bool:
    comps = _components_mask(grid == OBSTACLE)
    if not (3 <= len(comps) <= 5):
        return False
    for coords in comps:
        if not (2 <= len(coords) <= 4):
            return False
        rows = {int(r) for r in coords[:, 0]}
        cols = {int(c) for c in coords[:, 1]}
        if len(rows) != 1 and len(cols) != 1:
            return False
        if len(rows) == 1:
            ordered = sorted(int(c) for c in cols)
        else:
            ordered = sorted(int(r) for r in rows)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            return False
    return True


def _void_stats(grown: np.ndarray) -> tuple[int, int, int]:
    edge_count = 0
    enclosed_count = 0
    enclosed_area = 0
    for coords in _components_mask(grown == 0):
        if _touches_edge(coords, grown.shape):
            edge_count += 1
        else:
            enclosed_count += 1
            enclosed_area += len(coords)
    return edge_count, enclosed_count, enclosed_area


def _semantic_checks(grid: np.ndarray) -> bool:
    colors = set(map(int, np.unique(grid)))
    if not colors <= {0, 1, 2, 3, 5}:
        return False

    seed_points = [tuple(map(int, p)) for p in np.argwhere(np.isin(grid, SEED_COLORS))]
    if not (2 <= len(seed_points) <= 3):
        return False
    seed_colors = [int(grid[p]) for p in seed_points]
    if len(seed_colors) != len(set(seed_colors)):
        return False
    if any(abs(r1 - r2) + abs(c1 - c2) <= 1 for i, (r1, c1) in enumerate(seed_points) for r2, c2 in seed_points[i + 1 :]):
        return False
    for r, c in seed_points:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == OBSTACLE:
                return False

    if not _obstacle_components_are_short_lines(grid):
        return False

    grown_conflict = _grow_territories(grid, conflicts=True)
    grown_plain = _grow_territories(grid, conflicts=False)
    conflict_pixels = int(np.count_nonzero(grown_conflict == OBSTACLE) - np.count_nonzero(grid == OBSTACLE))
    if conflict_pixels < 2:
        return False

    edge_count, enclosed_count, enclosed_area = _void_stats(grown_conflict)
    if edge_count < 1 or enclosed_count < 1:
        return False

    _, plain_enclosed_count, plain_enclosed_area = _void_stats(grown_plain)
    if plain_enclosed_area >= enclosed_area or plain_enclosed_count >= enclosed_count:
        return False

    solved = main(grid)
    if int(np.count_nonzero(solved == EDGE_FILL)) == 0 or int(np.count_nonzero(solved == ENCLOSED_FILL)) == 0:
        return False
    expected = _recolor_voids(grown_conflict)
    return np.array_equal(solved, expected)


def _build_base_case(rng: random.Random) -> tuple[np.ndarray, list[tuple[str, int, int, int]], list[tuple[int, int]], list[int]]:
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)
    segments: list[tuple[str, int, int, int]] = []

    pocket = rng.randint(3, 4)
    _paint_segment(grid, segments, orientation="v", r=0, c=pocket, length=pocket)
    _paint_segment(grid, segments, orientation="h", r=pocket, c=0, length=pocket)

    seed_row = rng.randint(0, 1)
    pillar_len = rng.randint(3, 4)
    pillar_top_max = min(seed_row + 4, h - pillar_len)
    pillar_top = rng.randint(seed_row + 2, pillar_top_max)
    pillar_col = rng.randint(pocket + 2, w - 5)
    _paint_segment(grid, segments, orientation="v", r=pillar_top, c=pillar_col, length=pillar_len)
    _paint_segment(grid, segments, orientation="v", r=pillar_top, c=pillar_col + 2, length=pillar_len)

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    seeds = [(seed_row, pillar_col - 1), (seed_row, pillar_col + 3)]
    for color, (r, c) in zip(colors[:2], seeds, strict=False):
        grid[r, c] = color

    return grid, segments, seeds, colors


def _extra_segment_candidates(grid: np.ndarray, seeds: list[tuple[int, int]]) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    options: list[tuple[str, int, int, int]] = []
    for orientation in ("h", "v"):
        for length in (2, 3, 4):
            r_limit = h if orientation == "h" else h - length + 1
            c_limit = w - length + 1 if orientation == "h" else w
            for r in range(r_limit):
                for c in range(c_limit):
                    coords = [(r, c + i) for i in range(length)] if orientation == "h" else [(r + i, c) for i in range(length)]
                    if any(grid[rr, cc] != 0 for rr, cc in coords):
                        continue
                    if any(abs(rr - sr) + abs(cc - sc) <= 1 for rr, cc in coords for sr, sc in seeds):
                        continue
                    ok = True
                    for rr, cc in coords:
                        for nr, nc in _neighbors4(rr, cc):
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == OBSTACLE and (nr, nc) not in coords:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        options.append((orientation, r, c, length))
    return options


def _try_add_extra_segment(
    rng: random.Random,
    grid: np.ndarray,
    segments: list[tuple[str, int, int, int]],
    seeds: list[tuple[int, int]],
) -> None:
    options = _extra_segment_candidates(grid, seeds)
    rng.shuffle(options)
    for orientation, r, c, length in options:
        trial = grid.copy()
        trial_segments = list(segments)
        _paint_segment(trial, trial_segments, orientation=orientation, r=r, c=c, length=length)
        if _semantic_checks(trial):
            grid[:, :] = trial
            segments[:] = trial_segments
            return


def _third_seed_candidates(grid: np.ndarray, seeds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not _seed_is_valid(grid, seeds, r, c):
                continue
            out.append((r, c))
    out.sort(
        key=lambda cell: (
            min(
                cell[0] + cell[1],
                cell[0] + (w - 1 - cell[1]),
                (h - 1 - cell[0]) + cell[1],
                (h - 1 - cell[0]) + (w - 1 - cell[1]),
            ),
            cell[0],
            cell[1],
        ),
        reverse=True,
    )
    return out


def _try_add_third_seed(
    rng: random.Random,
    grid: np.ndarray,
    seeds: list[tuple[int, int]],
    colors: list[int],
) -> None:
    remaining = [color for color in colors if color not in [int(grid[r, c]) for r, c in seeds]]
    if not remaining:
        return
    candidates = _third_seed_candidates(grid, seeds)
    if not candidates:
        return
    sample = candidates[: max(8, len(candidates) // 3)]
    rng.shuffle(sample)
    color = remaining[0]
    for r, c in sample:
        trial = grid.copy()
        trial[r, c] = color
        if _semantic_checks(trial):
            grid[:, :] = trial
            seeds.append((r, c))
            return


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, segments, seeds, colors = _build_base_case(rng)
        if rng.random() < 0.55:
            _try_add_extra_segment(rng, grid, segments, seeds)
        if rng.random() < 0.45:
            _try_add_third_seed(rng, grid, seeds, colors)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
