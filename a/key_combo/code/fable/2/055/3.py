from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BG = 0
WALL = 1
GREY = 5
MARKER = 6
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in coords:
        if int(r) in (0, h - 1) or int(c) in (0, w - 1):
            return True
    return False


def _find_unique_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape[0] == 1, "expected one magenta marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _structural_wall_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    blue = np.asarray(grid == WALL, dtype=bool)
    for comp in components_mask(blue):
        if _touches_border(comp, grid.shape):
            mask[comp[:, 0], comp[:, 1]] = True
    return mask


def _find_energy_bar(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid == ORANGE)
    border_touching = []
    for comp in comps:
        if _touches_border(comp, grid.shape):
            border_touching.append(comp)
    assert len(border_touching) == 1, "expected one border-touching orange energy bar"
    return border_touching[0]


def _bfs_dist(grid: np.ndarray, ignore_decor: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker = _find_unique_marker(g)
    structural = _structural_wall_mask(g)
    if ignore_decor:
        blocked = structural.copy()
    else:
        blocked = g == WALL
    blocked |= g == ORANGE

    dist = np.full((h, w), -1, dtype=int)
    mr, mc = marker
    assert not blocked[mr, mc]
    q = deque([(mr, mc)])
    dist[mr, mc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and not blocked[nr, nc]:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _ranked_grey_tiles(grid: np.ndarray, dist: np.ndarray) -> list[tuple[int, int, int]]:
    grey = np.argwhere(grid == GREY)
    ranked = []
    seen = set()
    for r, c in grey:
        r = int(r)
        c = int(c)
        d = int(dist[r, c])
        assert d >= 0, "all grey tiles must be reachable"
        assert d not in seen, "grey-tile geodesic distances must be unique"
        seen.add(d)
        ranked.append((d, r, c))
    ranked.sort()
    return ranked


def _expected_selection(grid: np.ndarray) -> tuple[set[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    structural = _structural_wall_mask(g)
    decor = (g == WALL) & ~structural
    bar = _find_energy_bar(g)
    dist = _bfs_dist(g, ignore_decor=True)
    ranked = _ranked_grey_tiles(g, dist)
    n = int(bar.shape[0])
    assert len(ranked) > n, "need more grey tiles than energy cells"
    chosen = {(r, c) for _d, r, c in ranked[:n]}
    return chosen, structural, decor, bar


def main(grid: np.ndarray) -> np.ndarray:
    """
    Charge the geodesically nearest floor tiles from the corner marker and drain the edge energy bar.

    Input structure:
    - The dominant background is open floor; blue pixels are either border-touching structural walls or interior decorative fragments.
    - One magenta marker occupies a corner floor cell, several single-cell grey floor tiles lie in the main region, and one orange 1-cell-thick bar sits on a grid edge inside a wall-separated side chamber.

    Rule:
    - Treat only border-connected blue components as blocking walls; erase every other blue fragment to background before measuring paths.
    - From the marker, run a 4-neighbor flood fill around the structural walls, compute each grey tile's shortest path length, and recolor the N nearest grey tiles orange, where N is the energy-bar length.
    - Recolor the whole orange energy bar grey after the transfer; keep the remaining grey tiles grey and leave structural walls unchanged.

    Key insight:
    - The ordering is by geodesic distance through the corridor network, not straight-line closeness, and the interior blue fragments are fake obstacles that must be removed before distance ranking.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    chosen, structural, decor, bar = _expected_selection(g)

    out = g.copy()
    out[decor] = bg
    out[bar[:, 0], bar[:, 1]] = GREY
    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = ORANGE if (rr, cc) in chosen else GREY
    return out


def _euclid_sq_from_marker(grid: np.ndarray, cell: tuple[int, int]) -> int:
    mr, mc = _find_unique_marker(grid)
    r, c = cell
    return int((r - mr) ** 2 + (c - mc) ** 2)


def _can_place_decor(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) == WALL:
                return False
    return True


def _make_base_layout(rng: random.Random) -> tuple[np.ndarray, dict[str, int]]:
    left_w = rng.randint(4, 6)
    mid_w = rng.randint(5, 7)
    right_w = rng.randint(5, 7)
    chamber_w = rng.randint(2, 4)
    h = rng.randint(18, 24)
    w = left_w + 3 + mid_w + 3 + right_w + 2 + chamber_w
    grid = np.full((h, w), BG, dtype=int)

    wall_a = left_w
    wall_b = wall_a + 3 + mid_w
    wall_c = wall_b + 3 + right_w
    grid[:, wall_a : wall_a + 3] = WALL
    grid[:, wall_b : wall_b + 3] = WALL
    grid[:, wall_c : wall_c + 2] = WALL
    grid[:, wall_c + 2 :] = BG
    grid[0, 0] = MARKER

    gap_a_top = h - 3
    grid[gap_a_top:, wall_a : wall_a + 3] = BG
    gap_b_bottom = 2
    grid[:gap_b_bottom, wall_b : wall_b + 3] = BG

    door_r = rng.randint(h // 3, h // 2)
    if door_r >= gap_a_top - 2:
        door_r = gap_a_top - 3
    grid[door_r, wall_a] = BG
    grid[door_r, wall_a + 2] = BG
    grid[door_r - 1, wall_a + 1] = BG
    grid[door_r + 1, wall_a + 1] = BG
    grid[door_r, wall_a + 1] = WALL

    n = rng.randint(2, 3)
    bar_r0 = rng.randint(1, h - n - 1)
    grid[bar_r0 : bar_r0 + n, w - 1] = ORANGE

    meta = {
        "wall_a": wall_a,
        "wall_b": wall_b,
        "wall_c": wall_c,
        "door_r": door_r,
        "gap_a_top": gap_a_top,
        "n": n,
    }
    return grid, meta


def _sample_tiles(base: np.ndarray, meta: dict[str, int], rng: random.Random) -> np.ndarray | None:
    n = meta["n"]

    true_dist = _bfs_dist(base, ignore_decor=True)
    wrong_dist = _bfs_dist(base, ignore_decor=False)
    marker = _find_unique_marker(base)

    pool = []
    for r, c in np.argwhere((base == BG) & (true_dist >= 0)):
        r = int(r)
        c = int(c)
        if (r, c) == marker:
            continue
        if r in (0, base.shape[0] - 1) or c in (0, base.shape[1] - 1):
            continue
        pool.append((r, c))

    if len(pool) < n + 3:
        return None

    sensitive_cells = [
        cell
        for cell in pool
        if wrong_dist[cell[0], cell[1]] >= 0
        and wrong_dist[cell[0], cell[1]] - true_dist[cell[0], cell[1]] >= 6
    ]
    misleading_cells = sorted(
        pool,
        key=lambda cell: (_euclid_sq_from_marker(base, cell), -int(true_dist[cell[0], cell[1]]), cell),
    )[:10]
    easy_cells = sorted(pool, key=lambda cell: (int(true_dist[cell[0], cell[1]]), cell))
    if not sensitive_cells:
        return None

    def can_add(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
        r, c = cell
        return all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen)

    shortlist: list[tuple[int, int]] = []
    for group in (
        sorted(
            sensitive_cells,
            key=lambda cell: (
                -(int(wrong_dist[cell[0], cell[1]]) - int(true_dist[cell[0], cell[1]])),
                int(true_dist[cell[0], cell[1]]),
                cell,
            ),
        )[:10],
        sorted(
            misleading_cells,
            key=lambda cell: (_euclid_sq_from_marker(base, cell), -int(true_dist[cell[0], cell[1]]), cell),
        )[:10],
        easy_cells[:10],
    ):
        for cell in group:
            if cell not in shortlist:
                shortlist.append(cell)

    total_tiles = n + 3
    if len(shortlist) < total_tiles:
        return None

    best_choice: list[tuple[int, int]] | None = None
    best_score: tuple[int, int, int] | None = None
    for combo in itertools.combinations(shortlist, total_tiles):
        cells = list(combo)
        if any(not can_add(cell, [c for c in cells if c != cell]) for cell in cells):
            continue
        if len({int(true_dist[r, c]) for r, c in cells}) != len(cells):
            continue
        if len({_euclid_sq_from_marker(base, cell) for cell in cells}) != len(cells):
            continue

        ranked_true = sorted((int(true_dist[r, c]), r, c) for r, c in cells)
        ranked_wrong = sorted(
            ((int(wrong_dist[r, c]), r, c) for r, c in cells),
            key=lambda item: (10_000 if item[0] < 0 else item[0], item[1], item[2]),
        )
        ranked_euclid = sorted((_euclid_sq_from_marker(base, (r, c)), r, c) for r, c in cells)

        chosen_true = {(r, c) for _d, r, c in ranked_true[:n]}
        chosen_wrong = {(r, c) for _d, r, c in ranked_wrong[:n]}
        chosen_euclid = {(r, c) for _d, r, c in ranked_euclid[:n]}
        if chosen_true == chosen_wrong or chosen_true == chosen_euclid:
            continue

        sensitive_score = sum(
            max(0, int(wrong_dist[r, c]) - int(true_dist[r, c]))
            for r, c in chosen_true
        )
        euclid_score = sum(
            int(true_dist[r, c]) - _euclid_sq_from_marker(base, (r, c))
            for r, c in (chosen_true ^ chosen_euclid)
        )
        score = (sensitive_score, euclid_score, -sum(int(true_dist[r, c]) for r, c in cells))
        if best_score is None or score > best_score:
            best_score = score
            best_choice = cells

    if best_choice is not None:
        grid = base.copy()
        for r, c in best_choice:
            grid[r, c] = GREY
        if _validate_generated_case(grid):
            return grid
    return None


def _transform_case(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(grid, k=rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _validate_generated_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        marker = _find_unique_marker(g)
        if marker not in {
            (0, 0),
            (0, g.shape[1] - 1),
            (g.shape[0] - 1, 0),
            (g.shape[0] - 1, g.shape[1] - 1),
        }:
            return False
        structural = _structural_wall_mask(g)
        decor = (g == WALL) & ~structural
        if not structural.any() or not decor.any():
            return False
        bar = _find_energy_bar(g)
        n = int(bar.shape[0])
        true_dist = _bfs_dist(g, ignore_decor=True)
        wrong_dist = _bfs_dist(g, ignore_decor=False)
        ranked = _ranked_grey_tiles(g, true_dist)
        if len(ranked) <= n:
            return False
        grey_cells = [(r, c) for _d, r, c in ranked]
        euclid = [(_euclid_sq_from_marker(g, (r, c)), r, c) for _d, r, c in ranked]
        euclid.sort()
        if len({d for d, _r, _c in euclid}) != len(euclid):
            return False
        if not any(
            wrong_dist[r, c] >= 0 and wrong_dist[r, c] - true_dist[r, c] >= 2
            for _d, r, c in ranked
        ):
            return False

        chosen_true = {(r, c) for _d, r, c in ranked[:n]}
        chosen_euclid = {(r, c) for _e, r, c in euclid[:n]}

        wrong_ranked = []
        for _d, r, c in ranked:
            wd = int(wrong_dist[r, c])
            if wd < 0:
                wd = 10_000
            wrong_ranked.append((wd, r, c))
        wrong_ranked.sort()
        chosen_wrong = {(r, c) for _wd, r, c in wrong_ranked[:n]}

        if chosen_true == chosen_euclid or chosen_true == chosen_wrong:
            return False

        out = main(g)
        if out.shape != g.shape:
            return False
        bg = get_bg(g)
        if not np.all(out[decor] == bg):
            return False
        if not np.all(out[bar[:, 0], bar[:, 1]] == GREY):
            return False
        for r, c in grey_cells:
            expect = ORANGE if (r, c) in chosen_true else GREY
            if int(out[r, c]) != expect:
                return False
        if int(out[marker[0], marker[1]]) != MARKER:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        try:
            base, meta = _make_base_layout(rng)
            tiled = _sample_tiles(base, meta, rng)
            if tiled is None:
                continue
            grid = _transform_case(tiled, rng)
            if _validate_generated_case(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid geodesic-charge ARC input")
