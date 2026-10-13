from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_GEN_RETRY_LIMIT = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _tile_2x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(grid, dtype=int), (2, 2))


def _mark_seed_diagonals(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == RED):
        rr = int(r)
        cc = int(c)
        for dr, dc in _DIAGONALS:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(out.shape, nr, nc) and out[nr, nc] == BLACK:
                out[nr, nc] = TEAL
    return out


def _closed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked = g == GREEN
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _N4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(g.shape, nr, nc) and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            seen[r, c] = True
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in _N4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(g.shape, nr, nc) and not blocked[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(comp)
    return regions


def _fill_active_regions(marked_tiled: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[bool]]:
    out = np.asarray(marked_tiled, dtype=int).copy()
    regions = _closed_regions(out)
    active_flags: list[bool] = []
    for region in regions:
        active = any(out[r, c] == TEAL for r, c in region)
        active_flags.append(active)
        if active:
            for r, c in region:
                if out[r, c] == BLACK:
                    out[r, c] = TEAL
    return out, regions, active_flags


def _simulate(inp: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]], list[bool]]:
    tiled = _tile_2x2(inp)
    marked = _mark_seed_diagonals(tiled)
    filled, regions, active_flags = _fill_active_regions(marked)
    return tiled, marked, regions, active_flags, filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the small grid, mark red seeds' diagonal neighbors teal, then fill only the emergent closed regions that receive teal activation.

    Input structure:
    - The input is a small black grid with green border-anchored line fragments and one or two red seed pixels.
    - The green fragments are open in the single tile: they touch grid edges but do not enclose any region on their own.
    - Red seeds lie on black cells and are adjacent to the green structure.

    Rule:
    - Replicate the input into a 2x2 tiling to form the output canvas.
    - On the tiled canvas, each red seed paints its four diagonal black neighbors teal.
    - Treat green cells as walls and find every non-green region that is 4-disconnected from the outer border; these are the closed interiors created by stitching tile edges together.
    - If a closed region contains at least one teal cell, fill every black cell in that region teal; otherwise leave that region unchanged.
    - Preserve green walls, red seeds, and already teal cells.

    Key insight:
    - The closures do not exist in the original tile; they emerge only after identical edge fragments meet across tile seams, and the diagonal teal marks then select which of those new interiors are flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    _, _, _, _, filled = _simulate(g)
    return filled


def _enumerate_row_choices(height: int, count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    choices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for tops in combinations(range(1, height - 1), count):
        top_max = max(tops)
        for bottoms in combinations(range(top_max + 1, height - 1), count):
            if max(tops) < min(bottoms):
                choices.append((tops, bottoms))
    return choices


def _enumerate_col_pairs(width: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left_end in range(1, width - 2):
        for right_start in range(left_end + 3, width - 1):
            inside_gap = right_start - left_end - 1
            seam_width = width + left_end - right_start - 1
            if inside_gap >= 2 and seam_width >= 2:
                pairs.append((left_end, right_start))
    return pairs


def _build_row_stack_base(
    height: int,
    width: int,
    row_count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    row_choices = _enumerate_row_choices(height, row_count)
    col_pairs = _enumerate_col_pairs(width)
    if not row_choices or not col_pairs:
        return None

    tops, bottoms = row_choices[int(rng.integers(0, len(row_choices)))]
    left_end, right_start = col_pairs[int(rng.integers(0, len(col_pairs)))]

    grid = np.zeros((height, width), dtype=int)

    top_limit = max(tops)
    bottom_limit = min(bottoms)
    for c in (left_end, right_start):
        grid[: top_limit + 1, c] = GREEN
        grid[bottom_limit:, c] = GREEN

    for r in tops + bottoms:
        grid[r, : left_end + 1] = GREEN
        grid[r, right_start:] = GREEN

    if _closed_regions(grid):
        return None
    if len(_closed_regions(_tile_2x2(grid))) < 2:
        return None
    return grid


def _build_base(rng: np.random.Generator) -> np.ndarray | None:
    mode = str(rng.choice(["rows", "cols"]))
    if mode == "rows":
        height = int(rng.integers(7, 12))
        width = int(rng.integers(6, 11))
        row_count_options = [count for count in (2, 3) if _enumerate_row_choices(height, count)]
        if not row_count_options:
            return None
        row_count = int(rng.choice(row_count_options))
        return _build_row_stack_base(height, width, row_count, rng)

    height = int(rng.integers(6, 11))
    width = int(rng.integers(7, 12))
    col_count_options = [count for count in (2, 3) if _enumerate_row_choices(width, count)]
    if not col_count_options:
        return None
    col_count = int(rng.choice(col_count_options))
    base_t = _build_row_stack_base(width, height, col_count, rng)
    if base_t is None:
        return None
    return base_t.T.copy()


def _seed_candidates(base: np.ndarray) -> list[tuple[int, int]]:
    h, w = base.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if base[r, c] != BLACK:
                continue
            if any(_in_bounds(base.shape, r + dr, c + dc) and base[r + dr, c + dc] == GREEN for dr, dc in _N4):
                candidates.append((r, c))
    return candidates


def _activation_signature(base: np.ndarray, seeds: tuple[tuple[int, int], ...]) -> tuple[bool, np.ndarray | None]:
    inp = base.copy()
    for r, c in seeds:
        inp[r, c] = RED

    if _closed_regions(inp):
        return False, None

    tiled, marked, regions, active_flags, filled = _simulate(inp)
    active_count = sum(active_flags)
    if len(regions) < 2 or active_count == 0 or active_count == len(regions):
        return False, None
    if np.array_equal(marked, filled):
        return False, None

    active_region_has_fill = False
    for region, active in zip(regions, active_flags):
        if active and any(marked[r, c] == BLACK for r, c in region):
            active_region_has_fill = True
            break
    if not active_region_has_fill:
        return False, None

    return True, inp


def _place_seeds(base: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    candidates = _seed_candidates(base)
    if not candidates:
        return None

    valid_inputs: list[np.ndarray] = []
    candidate_lists = [list(combinations(candidates, 1)), list(combinations(candidates, 2))]
    rng.shuffle(candidate_lists)
    for combos in candidate_lists:
        if not combos:
            continue
        rng.shuffle(combos)
        for seeds in combos[: min(len(combos), 120)]:
            ok, inp = _activation_signature(base, seeds)
            if ok and inp is not None:
                valid_inputs.append(inp)
        if valid_inputs:
            break

    if not valid_inputs:
        return None
    return valid_inputs[int(rng.integers(0, len(valid_inputs)))]


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    base = _build_base(rng)
    if base is None:
        return None
    inp = _place_seeds(base, rng)
    if inp is None:
        return None

    out = main(inp)
    tiled = _tile_2x2(inp)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.any(inp == RED)
    assert set(map(int, np.unique(inp))) <= {BLACK, RED, GREEN}
    assert set(map(int, np.unique(out))) <= {BLACK, RED, GREEN, TEAL}
    assert not _closed_regions(base)
    assert len(_closed_regions(_tile_2x2(base))) >= 2
    assert out.shape == (2 * inp.shape[0], 2 * inp.shape[1])
    assert not np.array_equal(out, tiled)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emergent-closure ARC input")
