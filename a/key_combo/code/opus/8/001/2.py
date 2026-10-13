from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
VISIBLE_PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _infer_periodic_tile(grid: np.ndarray) -> np.ndarray:
    visible = np.argwhere((grid != BLACK) & (grid != RED) & (grid != GREY))
    assert len(visible) > 0, "expected visible tile samples"

    h, w = grid.shape
    for p in range(1, min(h, w) + 1):
        tile = np.full((p, p), -1, dtype=int)
        ok = True
        for r, c in visible:
            rr = int(r) % p
            cc = int(c) % p
            color = int(grid[r, c])
            if tile[rr, cc] == -1:
                tile[rr, cc] = color
            elif tile[rr, cc] != color:
                ok = False
                break
        if ok and np.all(tile != -1):
            return tile

    raise AssertionError("no complete repeating tile fits the visible samples")


def _spiral_trace(seed: tuple[int, int], shape: tuple[int, int]):
    r, c = seed
    leg = 1
    dir_idx = 0
    limit = 2 * max(shape) + 6
    while leg <= limit:
        for _ in range(2):
            dr, dc = DIRS[dir_idx]
            for _step in range(leg):
                r += dr
                c += dc
                yield r, c
            dir_idx = (dir_idx + 1) % 4
        leg += 1


def _render_full_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p = tile.shape[0]
    rows = np.arange(shape[0])[:, None] % p
    cols = np.arange(shape[1])[None, :] % p
    return tile[rows, cols]


def _has_exact_period(tile: np.ndarray) -> bool:
    p = tile.shape[0]
    for q in range(1, p):
        if all(int(tile[r, c]) == int(tile[r % q, c % q]) for r in range(p) for c in range(p)):
            return False
    return True


def _sample_tile(p: int) -> np.ndarray:
    while True:
        color_count = random.randint(2, min(3, len(VISIBLE_PALETTE)))
        colors = random.sample(VISIBLE_PALETTE, color_count)
        tile = np.random.choice(colors, size=(p, p), replace=True).astype(int)
        if len(set(map(int, tile.flat))) != color_count:
            continue
        if _has_exact_period(tile):
            return tile


def _collect_in_grid_spiral(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in _spiral_trace(seed, shape):
        if _in_bounds(shape, r, c):
            cells.append((r, c))
    return cells


def _residue_positions(
    shape: tuple[int, int], p: int, rr: int, cc: int
) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(rr, h, p) for c in range(cc, w, p)]


def _choose_seed(shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    margin = min(4, max(2, min(h, w) // 4))
    candidates = [(r, c) for r in range(margin, h - margin) for c in range(margin, w - margin)]
    random.shuffle(candidates)
    for seed in candidates:
        if len(_collect_in_grid_spiral(seed, shape)) >= 24:
            return seed
    return None


def _build_support_cells(
    full: np.ndarray,
    p: int,
    forbidden: set[tuple[int, int]],
    avoid_path: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    chosen: set[tuple[int, int]] = set()
    shape = full.shape
    for rr in range(p):
        for cc in range(p):
            candidates = [pos for pos in _residue_positions(shape, p, rr, cc) if pos not in forbidden and pos not in chosen]
            if not candidates:
                return None
            preferred = [pos for pos in candidates if pos not in avoid_path]
            pick_from = preferred if preferred else candidates
            chosen.add(random.choice(pick_from))
    return chosen


def _oracle_output(
    inp: np.ndarray,
    full: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray | None:
    out = inp.copy()
    sr, sc = seed
    out[sr, sc] = int(full[sr, sc])
    hit = False
    for r, c in _spiral_trace(seed, inp.shape):
        if not _in_bounds(inp.shape, r, c):
            continue
        if int(inp[r, c]) == RED:
            hit = True
            break
        if int(inp[r, c]) == BLACK:
            out[r, c] = int(full[r, c])
    if not hit:
        return None
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reveal a periodic color table only along a blocker-stopped spiral path.

    Input structure:
    - The grid has a black background with sparse visible tile samples in colors other than black, red, and gray.
    - One gray pixel marks the spiral seed, and several red pixels mark blockers on otherwise missing cells.
    - The visible samples agree with one smallest square periodic tile, while many other tile positions remain black.

    Rule:
    - Infer the smallest period p such that every visible sample color is determined by its (row mod p, col mod p) residue class, and reconstruct the full p x p tile.
    - Replace the gray seed by its tile color, then trace a square spiral from that seed using directions right, down, left, up with step lengths 1, 1, 2, 2, 3, 3, ...
    - For each in-bounds spiral cell, fill it with the tile color only if the input cell was black; leave visible sample colors unchanged; stop immediately when the spiral reaches a red blocker, leaving that blocker red.

    Key insight:
    - The periodic tile decides what color each hidden cell should have, but the spiral decides which hidden cells are allowed to become visible at all.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    seed_cells = np.argwhere(grid == GREY)
    assert len(seed_cells) == 1, "expected one gray seed"
    seed = tuple(map(int, seed_cells[0]))

    tile = _infer_periodic_tile(grid)
    p = tile.shape[0]

    out = grid.copy()
    sr, sc = seed
    out[sr, sc] = int(tile[sr % p, sc % p])

    hit_blocker = False
    for r, c in _spiral_trace(seed, grid.shape):
        if not _in_bounds(grid.shape, r, c):
            continue
        cell = int(grid[r, c])
        if cell == RED:
            hit_blocker = True
            break
        if cell == BLACK:
            out[r, c] = int(tile[r % p, c % p])

    assert hit_blocker, "spiral must encounter a red blocker"
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(11, 16)
        w = random.randint(11, 16)
        p = random.choice((2, 3, 4))
        tile = _sample_tile(p)
        full = _render_full_grid(tile, (h, w))

        seed = _choose_seed((h, w))
        if seed is None:
            continue
        spiral_cells = _collect_in_grid_spiral(seed, (h, w))
        if len(spiral_cells) < 12:
            continue

        stop_idx = random.randint(5, min(28, len(spiral_cells) - 1))
        stop_cell = spiral_cells[stop_idx]
        early_path = set(spiral_cells[:stop_idx])

        required_supports = _build_support_cells(
            full,
            p,
            forbidden={seed, stop_cell},
            avoid_path=early_path | {stop_cell},
        )
        if required_supports is None:
            continue

        inp = np.zeros((h, w), dtype=int)
        for r, c in required_supports:
            inp[r, c] = int(full[r, c])

        extra_support_budget = random.randint(2, 8)
        support_candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in required_supports and (r, c) not in {seed, stop_cell}
        ]
        random.shuffle(support_candidates)
        extras_added = 0
        for r, c in support_candidates:
            if extras_added >= extra_support_budget:
                break
            if (r, c) in early_path and random.random() < 0.8:
                continue
            inp[r, c] = int(full[r, c])
            extras_added += 1

        blocker_count = random.randint(2, 4)
        blocker_cells = [stop_cell]
        blocker_candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if inp[r, c] == BLACK and (r, c) != seed and (r, c) not in early_path and (r, c) != stop_cell
        ]
        if len(blocker_candidates) < blocker_count - 1:
            continue
        blocker_cells.extend(random.sample(blocker_candidates, blocker_count - 1))

        sr, sc = seed
        inp[sr, sc] = GREY
        for r, c in blocker_cells:
            inp[r, c] = RED

        out = _oracle_output(inp, full, seed)
        if out is None:
            continue
        if np.count_nonzero((inp == BLACK) & (out != BLACK)) < 4:
            continue
        if np.count_nonzero((inp == BLACK) & (out == BLACK)) < 8:
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp

    raise RuntimeError("failed to generate a valid spiral-mask periodic-tile input")
