from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GRAY = 5

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
GEN_PAIR_ATTEMPTS = 240
GEN_SIDE_RANGE = (14, 19)
GEN_EMITTER_RANGE = (4, 6)
GEN_MIN_SEED_GAP = 4
GEN_MIN_COMPONENT_SIZE = 28
GEN_MIN_WALL_FRACTION = 0.16
GEN_MAX_WALL_FRACTION = 0.42
GEN_MIN_REGION_WIN = 6
GEN_MIN_TIE_CELLS = 2
_GEN_CALL_COUNT = 0


def _find_unique_cell(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.shape[0] == 1, f"expected exactly one cell of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _freeze_emitters(g: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)]


def _build_walls(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Advance four-way gray rays on the evolving grid with synchronous collision handling."""
    g = np.asarray(grid, dtype=int).copy()
    h, w = map(int, g.shape)
    emitters = _freeze_emitters(g)

    beams = []
    for r, c in emitters:
        for dr, dc in DIRS4:
            beams.append([r, c, dr, dc, True])

    meta = {
        "emitters": emitters,
        "collision_cells": set(),
        "stop_edge": 0,
        "stop_wall": 0,
        "stop_seed": 0,
        "stop_simultaneous": 0,
        "advanced_cells": 0,
    }

    while True:
        proposals: dict[tuple[int, int], list[int]] = {}
        any_active = False
        for idx, beam in enumerate(beams):
            if not beam[4]:
                continue
            any_active = True
            r, c, dr, dc, _alive = beam
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                beam[4] = False
                meta["stop_edge"] += 1
                continue
            nxt = int(g[nr, nc])
            if nxt != BLACK:
                beam[4] = False
                if nxt == GRAY:
                    meta["stop_wall"] += 1
                else:
                    meta["stop_seed"] += 1
                continue
            proposals.setdefault((nr, nc), []).append(idx)

        if not any_active or not proposals:
            break

        for (nr, nc), ids in proposals.items():
            g[nr, nc] = GRAY
            if len(ids) > 1:
                meta["collision_cells"].add((int(nr), int(nc)))
                meta["stop_simultaneous"] += len(ids)
            else:
                meta["advanced_cells"] += 1
            for idx in ids:
                beams[idx][0] = int(nr)
                beams[idx][1] = int(nc)
                if len(ids) > 1:
                    beams[idx][4] = False

    return g, meta


def _bfs_distances(walls: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = map(int, walls.shape)
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque([seed])
    dist[seed] = 0

    while q:
        r, c = q.popleft()
        base = int(dist[r, c])
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1 and int(walls[nr, nc]) == BLACK:
                dist[nr, nc] = base + 1
                q.append((int(nr), int(nc)))
    return dist


def _paint_corridors(walls: np.ndarray, red_seed: tuple[int, int], blue_seed: tuple[int, int]) -> tuple[np.ndarray, dict]:
    out = np.asarray(walls, dtype=int).copy()
    red_dist = _bfs_distances(walls, red_seed)
    blue_dist = _bfs_distances(walls, blue_seed)

    black_mask = walls == BLACK
    reachable_red = red_dist >= 0
    reachable_blue = blue_dist >= 0

    red_wins = black_mask & reachable_red & (~reachable_blue | (red_dist < blue_dist))
    blue_wins = black_mask & reachable_blue & (~reachable_red | (blue_dist < red_dist))
    ties = black_mask & reachable_red & reachable_blue & (red_dist == blue_dist)
    unreachable = black_mask & ~reachable_red & ~reachable_blue

    out[ties] = YELLOW

    red_even = red_wins & ((red_dist % 2) == 0)
    red_odd = red_wins & ((red_dist % 2) == 1)
    blue_even = blue_wins & ((blue_dist % 2) == 0)
    blue_odd = blue_wins & ((blue_dist % 2) == 1)

    out[red_even] = RED
    out[red_odd] = BLUE
    out[blue_even] = BLUE
    out[blue_odd] = RED
    out[red_seed] = RED
    out[blue_seed] = BLUE

    info = {
        "red_dist": red_dist,
        "blue_dist": blue_dist,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "ties": ties,
        "unreachable": unreachable,
        "red_even": red_even,
        "red_odd": red_odd,
        "blue_even": blue_even,
        "blue_odd": blue_odd,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build gray wall rays from isolated emitters, then color the surviving corridor by competitive alternating BFS.

    Input structure:
    - Black background contains several isolated gray single-pixel emitters, plus one red seed and one blue seed.
    - The gray emitters are pairwise separated in both row and column, so each emitter can launch four straight rays.
    - The two seeds lie in corridor cells rather than on emitter rows or columns.

    Rule:
    - Freeze the original gray pixels as emitters and advance one ray from each emitter in each cardinal direction on the evolving grid.
    - A ray paints black cells gray one step at a time, stopping before an out-of-bounds step or before stepping into an already non-black cell; if two rays reach the same black cell on the same step, that meeting cell turns gray and both rays stop there.
    - After the wall grid is complete, run separate 4-connected BFS waves from the red and blue seeds, expanding only through black cells.
    - Each black corridor cell goes to the nearer seed; equal distances become yellow. In a red-owned region, even BFS distance stays red and odd BFS distance flips to blue; in a blue-owned region, even stays blue and odd flips to red.
    - Gray wall cells and the two seed cells keep their wall/seed colors. Any black pocket unreachable from both seeds stays black.

    Key insight:
    - The gray pixels do not mark the final fill directly: they first generate a collision-limited wall network, and that dynamically built topology is what determines the two-seed distance competition and parity coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_seed = _find_unique_cell(g, RED)
    blue_seed = _find_unique_cell(g, BLUE)
    walls, _meta = _build_walls(g)
    out, _info = _paint_corridors(walls, red_seed, blue_seed)
    return out


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = np.zeros((h, w), dtype=bool)
        comp[r, c] = True
        while q:
            cr, cc = q.popleft()
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    comp[nr, nc] = True
                    q.append((int(nr), int(nc)))
        components.append(comp)
    return components


def _naive_full_wall_mask(shape: tuple[int, int], emitters: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    rows = {r for r, _c in emitters}
    cols = {c for _r, c in emitters}
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    return np.isin(rr, list(rows)) | np.isin(cc, list(cols))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _legal_seed_cells(walls: np.ndarray, emitters: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _ in emitters}
    cols = {c for _, c in emitters}
    return [
        (int(r), int(c))
        for r, c in np.argwhere(walls == BLACK)
        if int(r) not in rows and int(c) not in cols
    ]


def _pair_metrics(
    walls: np.ndarray,
    red_seed: tuple[int, int],
    blue_seed: tuple[int, int],
    component: np.ndarray,
) -> dict:
    out, info = _paint_corridors(walls, red_seed, blue_seed)
    black_count = int(np.count_nonzero(walls == BLACK))
    component_size = int(np.count_nonzero(component))

    return {
        "output": out,
        "info": info,
        "black_count": black_count,
        "component_size": component_size,
        "reachable_count": int(np.count_nonzero((walls == BLACK) & ~info["unreachable"])),
        "unreachable_count": int(np.count_nonzero(info["unreachable"])),
        "red_win_count": int(np.count_nonzero(info["red_wins"])),
        "blue_win_count": int(np.count_nonzero(info["blue_wins"])),
        "tie_count": int(np.count_nonzero(info["ties"])),
        "red_even_count": int(np.count_nonzero(info["red_even"])),
        "red_odd_count": int(np.count_nonzero(info["red_odd"])),
        "blue_even_count": int(np.count_nonzero(info["blue_even"])),
        "blue_odd_count": int(np.count_nonzero(info["blue_odd"])),
    }


def _good_seed_pair(
    walls: np.ndarray,
    emitters: list[tuple[int, int]],
    red_seed: tuple[int, int],
    blue_seed: tuple[int, int],
    component: np.ndarray,
    require_tie: bool,
) -> bool:
    if _manhattan(red_seed, blue_seed) < GEN_MIN_SEED_GAP:
        return False

    metrics = _pair_metrics(walls, red_seed, blue_seed, component)
    if metrics["component_size"] < GEN_MIN_COMPONENT_SIZE:
        return False
    if metrics["red_win_count"] < GEN_MIN_REGION_WIN or metrics["blue_win_count"] < GEN_MIN_REGION_WIN:
        return False
    if metrics["red_even_count"] < 1 or metrics["red_odd_count"] < 1:
        return False
    if metrics["blue_even_count"] < 1 or metrics["blue_odd_count"] < 1:
        return False
    if require_tie:
        return metrics["tie_count"] >= GEN_MIN_TIE_CELLS
    return metrics["tie_count"] == 0


def _place_emitters(h: int, w: int, n_emitters: int) -> list[tuple[int, int]]:
    rows = sorted(random.sample(range(1, h - 1), n_emitters))
    cols = sorted(random.sample(range(1, w - 1), n_emitters))
    random.shuffle(cols)
    return list(zip(rows, cols))


def _build_input_grid(
    shape: tuple[int, int],
    emitters: list[tuple[int, int]],
    red_seed: tuple[int, int],
    blue_seed: tuple[int, int],
) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for r, c in emitters:
        g[r, c] = GRAY
    g[red_seed] = RED
    g[blue_seed] = BLUE
    return g


def _sample_seed_pair(
    walls: np.ndarray,
    emitters: list[tuple[int, int]],
    require_tie: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    legal = _legal_seed_cells(walls, emitters)
    if len(legal) < 2:
        return None

    component_map: dict[tuple[int, int], np.ndarray] = {}
    for comp in _component_masks(walls == BLACK):
        for cell in np.argwhere(comp):
            component_map[(int(cell[0]), int(cell[1]))] = comp

    legal = [cell for cell in legal if component_map[cell].sum() >= GEN_MIN_COMPONENT_SIZE]
    if len(legal) < 2:
        return None

    pair_source = list(itertools.combinations(legal, 2))
    random.shuffle(pair_source)
    for red_seed, blue_seed in pair_source[:GEN_PAIR_ATTEMPTS]:
        comp = component_map[red_seed]
        if comp is not component_map[blue_seed]:
            continue
        if _good_seed_pair(walls, emitters, red_seed, blue_seed, comp, require_tie):
            return red_seed, blue_seed
    return None


def _semantic_input_ok(grid: np.ndarray, require_tie: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    red = np.argwhere(g == RED)
    blue = np.argwhere(g == BLUE)
    emitters = np.argwhere(g == GRAY)
    if red.shape[0] != 1 or blue.shape[0] != 1:
        return False
    if not (GEN_EMITTER_RANGE[0] <= emitters.shape[0] <= GEN_EMITTER_RANGE[1]):
        return False

    emitter_rows = [int(r) for r, _c in emitters]
    emitter_cols = [int(c) for _r, c in emitters]
    if len(set(emitter_rows)) != len(emitter_rows) or len(set(emitter_cols)) != len(emitter_cols):
        return False

    red_seed = (int(red[0, 0]), int(red[0, 1]))
    blue_seed = (int(blue[0, 0]), int(blue[0, 1]))
    if red_seed[0] in emitter_rows or red_seed[1] in emitter_cols:
        return False
    if blue_seed[0] in emitter_rows or blue_seed[1] in emitter_cols:
        return False

    walls, meta = _build_walls(g)
    wall_fraction = float(np.count_nonzero(walls == GRAY)) / float(walls.size)
    if not (GEN_MIN_WALL_FRACTION <= wall_fraction <= GEN_MAX_WALL_FRACTION):
        return False
    if meta["stop_wall"] + meta["stop_simultaneous"] == 0:
        return False

    naive = _naive_full_wall_mask(g.shape, [tuple(map(int, rc)) for rc in emitters])
    if np.array_equal(walls == GRAY, naive):
        return False

    corridor = walls.copy()
    corridor[red_seed] = BLACK
    corridor[blue_seed] = BLACK
    components = _component_masks(corridor == BLACK)
    cell_to_comp: dict[tuple[int, int], np.ndarray] = {}
    for comp in components:
        for r, c in np.argwhere(comp):
            cell_to_comp[(int(r), int(c))] = comp
    if red_seed not in cell_to_comp or blue_seed not in cell_to_comp:
        return False
    if cell_to_comp[red_seed] is not cell_to_comp[blue_seed]:
        return False

    if not _good_seed_pair(walls, [tuple(map(int, rc)) for rc in emitters], red_seed, blue_seed, cell_to_comp[red_seed], require_tie):
        return False

    return True


def _gen_attempt(require_tie: bool) -> np.ndarray | None:
    h = random.randint(*GEN_SIDE_RANGE)
    w = random.randint(*GEN_SIDE_RANGE)
    n_emitters = random.randint(*GEN_EMITTER_RANGE)
    if n_emitters > min(h - 2, w - 2):
        return None

    emitters = _place_emitters(h, w, n_emitters)
    base = np.zeros((h, w), dtype=int)
    for r, c in emitters:
        base[r, c] = GRAY

    walls, meta = _build_walls(base)
    if meta["stop_wall"] + meta["stop_simultaneous"] == 0:
        return None

    pair = _sample_seed_pair(walls, emitters, require_tie)
    if pair is None:
        return None
    red_seed, blue_seed = pair
    grid = _build_input_grid((h, w), emitters, red_seed, blue_seed)
    if not _semantic_input_ok(grid, require_tie):
        return None

    solver_out = main(grid)
    if solver_out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    require_tie = (_GEN_CALL_COUNT % 2) == 0
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(require_tie)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-wall corridor input")
