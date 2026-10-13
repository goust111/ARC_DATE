from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
PERIOD_CHOICES = ((3, 3), (3, 4), (4, 3), (4, 4))
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)

GRID_MIN = 11
GRID_MAX = 16
SEED_COUNT_WEIGHTS = (1, 2, 2, 3, 3)
MAX_GENERATION_ATTEMPTS = 500


def _candidate_periods(h: int, w: int) -> list[tuple[int, int]]:
    return sorted(
        ((pr, pc) for pr in range(1, h + 1) for pc in range(1, w + 1)),
        key=lambda item: (item[0] * item[1], item[0], item[1]),
    )


def _infer_periodic_tile(
    grid: np.ndarray, ignored_colors: set[int] | None = None
) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    ignored = set() if ignored_colors is None else set(int(v) for v in ignored_colors)
    visible = np.argwhere(~np.isin(g, list(ignored)))
    assert visible.size > 0, "need at least one visible periodic sample"

    h, w = g.shape
    for pr, pc in _candidate_periods(h, w):
        tile = np.full((pr, pc), -1, dtype=int)
        seen = np.zeros((pr, pc), dtype=bool)
        ok = True
        for r, c in visible:
            rr, cc = int(r) % pr, int(c) % pc
            value = int(g[r, c])
            if seen[rr, cc]:
                if int(tile[rr, cc]) != value:
                    ok = False
                    break
            else:
                seen[rr, cc] = True
                tile[rr, cc] = value
        if ok and np.all(seen):
            return pr, pc, tile

    raise AssertionError("no complete periodic tile fits the visible non-red samples")


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pr, pc = tile.shape
    rows = np.arange(shape[0])[:, None] % pr
    cols = np.arange(shape[1])[None, :] % pc
    return tile[rows, cols]


def _ray_cells(
    tile: np.ndarray, shape: tuple[int, int], seed: tuple[int, int]
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    pr, pc = tile.shape
    r, c = seed
    color = int(tile[r % pr, c % pc])
    cells: list[tuple[int, int]] = []

    while True:
        nr, nc = r + 1, c + 1
        if nr >= h or nc >= w:
            return cells, "boundary"
        if int(tile[nr % pr, nc % pc]) != color:
            return cells, "mismatch"
        cells.append((nr, nc))
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the periodic pattern and use it to determine each red seed's diagonal ray color and stopping condition.

    Input structure:
    - A mostly periodic colored grid contains black holes and 1-3 red singleton seed cells.
    - The periodic pattern uses non-black, non-red colors and repeats with a smallest row/column period.
    - Every true tile residue appears at least once among the visible nonzero non-red cells.

    Rule:
    - Ignore red seeds and infer the smallest periodic tile consistent with all visible nonzero non-red cells.
    - Fill every black hole from that tile while keeping the red seed cells unchanged.
    - For each red seed, look up its would-be periodic color from the inferred tile, then extend a ray down-right.
    - Paint each down-right step with that color while the next tile color matches it; stop before the first mismatch or the grid boundary.

    Key insight:
    - The periodic tile is the shared latent structure: it repairs the holes, encodes each seed's ray color, and also defines where the ray must stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pr, pc, tile = _infer_periodic_tile(g, ignored_colors={BLACK, RED})
    out = g.copy()
    holes = out == BLACK
    if np.any(holes):
        rr = np.arange(g.shape[0])[:, None] % pr
        cc = np.arange(g.shape[1])[None, :] % pc
        out[holes] = tile[rr, cc][holes]

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    for seed in seeds:
        for r, c in _ray_cells(tile, g.shape, seed)[0]:
            assert int(g[r, c]) != RED, "ray paths are expected not to hit another seed"
            out[r, c] = int(tile[seed[0] % pr, seed[1] % pc])
        out[seed] = RED

    return out


def _tile_has_exact_period(tile: np.ndarray) -> bool:
    pr, pc, inferred = _infer_periodic_tile(tile, ignored_colors=set())
    return (pr, pc) == tile.shape and np.array_equal(inferred, tile)


def _sample_tile(pr: int, pc: int) -> np.ndarray | None:
    color_count = random.choice((2, 3))
    colors = random.sample(PALETTE, color_count)
    tile = np.array(
        [[random.choice(colors) for _ in range(pc)] for _ in range(pr)],
        dtype=int,
    )
    if len(set(map(int, tile.flatten()))) < 2:
        return None
    if not _tile_has_exact_period(tile):
        return None

    same_step = False
    diff_step = False
    for r in range(pr):
        for c in range(pc):
            if int(tile[r, c]) == int(tile[(r + 1) % pr, (c + 1) % pc]):
                same_step = True
            else:
                diff_step = True
    if not (same_step and diff_step):
        return None
    return tile


def _counts_by_residue(shape: tuple[int, int], period: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pr, pc = period
    counts = np.zeros((pr, pc), dtype=int)
    for r in range(h):
        for c in range(w):
            counts[r % pr, c % pc] += 1
    return counts


def _seed_candidates(tile: np.ndarray, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    pr, pc = tile.shape
    candidates: list[dict] = []
    for r in range(h - 2):
        for c in range(w - 2):
            cells, stop_reason = _ray_cells(tile, shape, (r, c))
            if len(cells) < 2:
                continue
            candidates.append(
                {
                    "seed": (r, c),
                    "diag": r - c,
                    "color": int(tile[r % pr, c % pc]),
                    "cells": cells,
                    "stop": stop_reason,
                    "length": len(cells),
                }
            )
    return candidates


def _choose_seeds(tile: np.ndarray, shape: tuple[int, int]) -> list[dict] | None:
    candidates = _seed_candidates(tile, shape)
    if not candidates:
        return None

    by_stop: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_stop[item["stop"]].append(item)

    target_count = random.choice(SEED_COUNT_WEIGHTS)
    unique_diagonals = len({item["diag"] for item in candidates})
    target_count = min(target_count, unique_diagonals)
    if target_count == 0:
        return None

    def sample_from_pool(pool: list[dict], chosen: list[dict], used_diags: set[int]) -> dict | None:
        shuffled = pool[:]
        random.shuffle(shuffled)
        for item in shuffled:
            if item["diag"] not in used_diags:
                return item
        return None

    require_boundary = target_count >= 2 and by_stop["boundary"] and by_stop["mismatch"] and random.random() < 0.45

    for _ in range(200):
        chosen: list[dict] = []
        used_diags: set[int] = set()

        if require_boundary:
            first = sample_from_pool(by_stop["boundary"], chosen, used_diags)
            second = sample_from_pool(by_stop["mismatch"], chosen, used_diags)
            if first is None or second is None:
                continue
            chosen.extend([first, second])
            used_diags.update((first["diag"], second["diag"]))
        else:
            first = sample_from_pool(candidates, chosen, used_diags)
            if first is None:
                continue
            chosen.append(first)
            used_diags.add(first["diag"])

        pool = candidates[:]
        random.shuffle(pool)
        for item in pool:
            if len(chosen) >= target_count:
                break
            if item["diag"] in used_diags:
                continue
            chosen.append(item)
            used_diags.add(item["diag"])

        if len(chosen) != target_count:
            continue

        if len(chosen) > 1:
            signatures = {(item["color"], item["length"], item["stop"]) for item in chosen}
            if len(signatures) < 2:
                continue

        return sorted(chosen, key=lambda item: item["seed"])

    return None


def _enumerate_rectangles(shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape
    rects = []
    for rh in range(1, 4):
        for rw in range(1, 5):
            if rh * rw > 8:
                continue
            for r0 in range(h - rh + 1):
                for c0 in range(w - rw + 1):
                    rects.append((r0, c0, rh, rw))
    random.shuffle(rects)
    return rects


def _place_holes(
    base: np.ndarray,
    tile: np.ndarray,
    seeds: list[dict],
) -> np.ndarray | None:
    h, w = base.shape
    pr, pc = tile.shape
    seed_cells = {item["seed"] for item in seeds}
    ray_cells = {cell for item in seeds for cell in item["cells"]}
    available = _counts_by_residue((h, w), (pr, pc))
    for r, c in seed_cells:
        available[r % pr, c % pc] -= 1
    if np.any(available <= 0):
        return None

    holes = np.zeros((h, w), dtype=bool)
    rect_target = random.randint(4, 7)
    rects = _enumerate_rectangles((h, w))
    placed = 0

    for r0, c0, rh, rw in rects:
        if placed >= rect_target:
            break
        region = holes[r0 : r0 + rh, c0 : c0 + rw]
        if np.any(region):
            continue

        cells = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
        if any(cell in seed_cells for cell in cells):
            continue

        delta = np.zeros_like(available)
        for r, c in cells:
            delta[r % pr, c % pc] += 1
        if np.any(available - delta <= 0):
            continue

        holes[r0 : r0 + rh, c0 : c0 + rw] = True
        available -= delta
        placed += 1

    if placed < 3:
        return None
    if not np.any(holes & ~np.isin(base, [RED])):
        return None

    ray_mask = np.zeros((h, w), dtype=bool)
    for r, c in ray_cells:
        ray_mask[r, c] = True
    if np.count_nonzero(holes & ~ray_mask) < 3:
        return None

    return holes


def _render_input(base: np.ndarray, holes: np.ndarray, seeds: list[dict]) -> np.ndarray:
    inp = base.copy()
    inp[holes] = BLACK
    for item in seeds:
        r, c = item["seed"]
        inp[r, c] = RED
    return inp


def _expected_output(base: np.ndarray, seeds: list[dict]) -> np.ndarray:
    out = base.copy()
    for item in seeds:
        r, c = item["seed"]
        out[r, c] = RED
        for rr, cc in item["cells"]:
            out[rr, cc] = int(item["color"])
    return out


def _validate_instance(
    inp: np.ndarray,
    base: np.ndarray,
    tile: np.ndarray,
    seeds: list[dict],
    holes: np.ndarray,
) -> bool:
    try:
        pr, pc, inferred = _infer_periodic_tile(inp, ignored_colors={BLACK, RED})
    except AssertionError:
        return False
    if (pr, pc) != tile.shape or not np.array_equal(inferred, tile):
        return False

    expected = _expected_output(base, seeds)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    repaired = base.copy()
    for item in seeds:
        repaired[item["seed"]] = RED
    if not np.any(inp == BLACK):
        return False
    if not np.any(repaired != inp):
        return False
    if np.count_nonzero(holes & (expected != inp)) < 3:
        return False
    if len(seeds) > 1:
        signatures = {(item["color"], item["length"], item["stop"]) for item in seeds}
        if len(signatures) < 2:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    pr, pc = random.choice(PERIOD_CHOICES)
    if pr > h or pc > w:
        return None

    tile = None
    for _ in range(100):
        tile = _sample_tile(pr, pc)
        if tile is not None:
            break
    if tile is None:
        return None

    base = _tile_grid(tile, (h, w))
    seeds = _choose_seeds(tile, (h, w))
    if seeds is None:
        return None

    holes = _place_holes(base, tile, seeds)
    if holes is None:
        return None

    inp = _render_input(base, holes, seeds)
    if not _validate_instance(inp, base, tile, seeds, holes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid periodic-repair ray puzzle")
