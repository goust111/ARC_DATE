from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (10, 18)
SEED_COUNT_RANGE = (3, 4)
UNHIT_BLOCK_COUNT_RANGE = (1, 2)
MIN_SEED_GAP = 2
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _block_cells(anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = anchor
    return ((r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1))


def _block_halo(anchor: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    r0, c0 = anchor
    halo = set()
    for r in range(max(0, r0 - 1), min(height, r0 + 3)):
        for c in range(max(0, c0 - 1), min(width, c0 + 3)):
            halo.add((r, c))
    return halo


def _find_gray_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = grid == GREY
    blocks: list[tuple[int, int]] = []
    for coords in _components_mask(mask):
        assert coords.shape[0] == 4, "every gray component must be a 2x2 block"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "gray components must be exactly 2x2"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "gray blocks must be solid"
        blocks.append((r0, c0))
    return sorted(blocks)


def _trace_path(height: int, width: int, start_col: int) -> tuple[list[tuple[int, int]], list[int]]:
    r = height - 1
    c = int(start_col)
    dc = -1 if c == width - 1 else 1
    bounce_count = 0
    path: list[tuple[int, int]] = []
    bounces_before_hit: list[int] = []

    while True:
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= width:
            dc *= -1
            bounce_count += 1
            nc = c + dc
        r, c = nr, nc
        path.append((r, c))
        bounces_before_hit.append(bounce_count)

    return path, bounces_before_hit


def _simulate_seed_hit(
    height: int,
    width: int,
    start_col: int,
    cell_to_block: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    r = height - 1
    c = int(start_col)
    dc = -1 if c == width - 1 else 1

    while True:
        nr = r - 1
        if nr < 0:
            return None
        nc = c + dc
        if nc < 0 or nc >= width:
            dc *= -1
            nc = c + dc
        r, c = nr, nc
        anchor = cell_to_block.get((r, c))
        if anchor is not None:
            return anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the gray 2x2 blocks that are struck by invisible bouncing rays from the bottom seeds.

    Input structure:
    - A black background grid has several single colored seed pixels on the bottom row.
    - The interior contains several floating 2x2 gray blocks, with gray distinct from every seed color.
    - No visible path is drawn between seeds and blocks.

    Rule:
    - Each seed emits one diagonal ray upward: it starts up-right, except a seed on the right edge starts up-left.
    - Every step moves one row up and one column sideways; hitting the left or right border flips only the horizontal direction.
    - When the ray first enters any cell of a gray block, that entire 2x2 block is recolored to the seed's color; a ray that exits past the top without a hit does nothing.
    - Any gray block that is not hit by any ray is erased back to black, while the bottom seeds remain unchanged.

    Key insight:
    - The only visible evidence is which gray blocks survive and what colors they become, so the solver must mentally simulate the invisible reflected diagonal paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)

    non_seed_colors = np.argwhere((g != BACKGROUND) & (g != GREY))
    assert non_seed_colors.size > 0, "expected at least one bottom seed"
    assert np.all(non_seed_colors[:, 0] == height - 1), "non-gray colors must only appear on the bottom row"

    seed_cols = [int(c) for c in np.flatnonzero((g[-1] != BACKGROUND) & (g[-1] != GREY))]
    seed_colors = [int(g[-1, c]) for c in seed_cols]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be distinct"
    for c, color in zip(seed_cols, seed_colors):
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "each seed color must appear exactly once"
        assert tuple(map(int, coords[0])) == (height - 1, c), "seed must sit on the bottom row"

    blocks = _find_gray_blocks(g)
    assert blocks, "expected at least one gray block"
    cell_to_block: dict[tuple[int, int], tuple[int, int]] = {}
    for anchor in blocks:
        r0, c0 = anchor
        assert 0 < r0 < height - 2 and 0 < c0 < width - 2, "gray blocks must float in the interior"
        for cell in _block_cells(anchor):
            cell_to_block[cell] = anchor

    hits: dict[tuple[int, int], int] = {}
    for c, color in zip(seed_cols, seed_colors):
        anchor = _simulate_seed_hit(height, width, c, cell_to_block)
        if anchor is None:
            continue
        assert anchor not in hits, "a gray block must not be hit by multiple seeds"
        hits[anchor] = color

    out = g.copy()
    for r0, c0 in blocks:
        out[r0 : r0 + 2, c0 : c0 + 2] = BACKGROUND
    for (r0, c0), color in hits.items():
        out[r0 : r0 + 2, c0 : c0 + 2] = int(color)
    return out


def _max_seed_count(width: int) -> int:
    return 1 + len(range(0, width - 1, MIN_SEED_GAP))


def _sample_seed_columns(width: int, seed_count: int) -> list[int] | None:
    cols = [width - 1]
    pool = list(range(width - 1))
    random.shuffle(pool)
    for c in pool:
        if all(abs(c - prev) >= MIN_SEED_GAP for prev in cols):
            cols.append(c)
            if len(cols) == seed_count:
                cols.sort()
                return cols
    return None


def _collect_hit_candidates(
    paths: list[list[tuple[int, int]]],
    bounce_counts: list[list[int]],
    seed_idx: int,
    shape: tuple[int, int],
) -> list[dict]:
    height, width = shape
    other_path_cells = set()
    for other_idx, other_path in enumerate(paths):
        if other_idx != seed_idx:
            other_path_cells.update(other_path)

    path_index = {cell: idx for idx, cell in enumerate(paths[seed_idx])}
    candidates: list[dict] = []
    for r0 in range(1, height - 2):
        for c0 in range(1, width - 2):
            anchor = (r0, c0)
            cells = set(_block_cells(anchor))
            if cells & other_path_cells:
                continue
            hit_steps = [path_index[cell] for cell in cells if cell in path_index]
            if not hit_steps:
                continue
            hit_step = min(hit_steps)
            candidates.append(
                {
                    "anchor": anchor,
                    "cells": cells,
                    "halo": _block_halo(anchor, shape),
                    "hit_step": hit_step,
                    "uses_bounce": bounce_counts[seed_idx][hit_step] > 0,
                }
            )
    random.shuffle(candidates)
    return candidates


def _choose_hit_seed_set(
    candidate_map: dict[int, list[dict]],
    right_seed_idx: int,
    seed_count: int,
) -> tuple[list[int], int] | None:
    eligible = [idx for idx, candidates in candidate_map.items() if candidates]
    bounce_ready = [idx for idx in eligible if any(candidate["uses_bounce"] for candidate in candidate_map[idx])]
    max_hits = min(3, len(eligible), seed_count - 1)
    if max_hits < 1 or not bounce_ready:
        return None

    if max_hits >= 2 and random.random() < 0.7:
        hit_count = 2
    else:
        hit_count = random.randint(1, max_hits)

    if right_seed_idx in bounce_ready and random.random() < 0.7:
        bounce_seed_idx = right_seed_idx
    else:
        bounce_seed_idx = random.choice(bounce_ready)

    chosen = [bounce_seed_idx]
    remaining = [idx for idx in eligible if idx != bounce_seed_idx]
    if right_seed_idx in remaining and len(chosen) < hit_count and random.random() < 0.6:
        chosen.append(right_seed_idx)
        remaining.remove(right_seed_idx)
    random.shuffle(remaining)
    chosen.extend(remaining[: hit_count - len(chosen)])
    return chosen, bounce_seed_idx


def _assign_hit_blocks(
    candidate_map: dict[int, list[dict]],
    hit_seed_indices: list[int],
    bounce_seed_idx: int,
) -> dict[int, dict] | None:
    filtered: dict[int, list[dict]] = {}
    for seed_idx in hit_seed_indices:
        candidates = candidate_map[seed_idx]
        if seed_idx == bounce_seed_idx:
            candidates = [candidate for candidate in candidates if candidate["uses_bounce"]]
        if not candidates:
            return None
        shuffled = candidates[:]
        random.shuffle(shuffled)
        filtered[seed_idx] = shuffled

    order = sorted(hit_seed_indices, key=lambda idx: (idx != bounce_seed_idx, len(filtered[idx])))

    def rec(pos: int, forbidden: set[tuple[int, int]], picked: dict[int, dict]) -> dict[int, dict] | None:
        if pos == len(order):
            return dict(picked)
        seed_idx = order[pos]
        for candidate in filtered[seed_idx]:
            if candidate["cells"] & forbidden:
                continue
            picked[seed_idx] = candidate
            result = rec(pos + 1, forbidden | candidate["halo"], picked)
            if result is not None:
                return result
            picked.pop(seed_idx, None)
        return None

    return rec(0, set(), {})


def _select_unhit_blocks(
    path_union: set[tuple[int, int]],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[dict] | None:
    height, width = shape
    candidates: list[dict] = []
    for r0 in range(1, height - 2):
        for c0 in range(1, width - 2):
            anchor = (r0, c0)
            cells = set(_block_cells(anchor))
            if cells & path_union:
                continue
            candidates.append(
                {
                    "anchor": anchor,
                    "cells": cells,
                    "halo": _block_halo(anchor, shape),
                }
            )
    random.shuffle(candidates)

    target_count = random.randint(*UNHIT_BLOCK_COUNT_RANGE)
    selected: list[dict] = []
    blocked = set(forbidden)
    for candidate in candidates:
        if candidate["cells"] & blocked:
            continue
        selected.append(candidate)
        blocked |= candidate["halo"]
        if len(selected) == target_count:
            break
    if not selected:
        return None
    return selected


def _render_input_and_expected(
    shape: tuple[int, int],
    seed_cols: list[int],
    seed_colors: list[int],
    hit_blocks: dict[int, dict],
    unhit_blocks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    for c, color in zip(seed_cols, seed_colors):
        inp[height - 1, c] = int(color)
        expected[height - 1, c] = int(color)

    for block in unhit_blocks:
        r0, c0 = block["anchor"]
        inp[r0 : r0 + 2, c0 : c0 + 2] = GREY

    for seed_idx, block in hit_blocks.items():
        r0, c0 = block["anchor"]
        inp[r0 : r0 + 2, c0 : c0 + 2] = GREY
        expected[r0 : r0 + 2, c0 : c0 + 2] = int(seed_colors[seed_idx])

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    max_seed_count = min(SEED_COUNT_RANGE[1], _max_seed_count(width))
    if max_seed_count < SEED_COUNT_RANGE[0]:
        return None
    seed_count = random.randint(SEED_COUNT_RANGE[0], max_seed_count)
    seed_cols = _sample_seed_columns(width, seed_count)
    if seed_cols is None:
        return None

    seed_colors = random.sample(SEED_COLORS, seed_count)
    right_seed_idx = seed_cols.index(width - 1)

    paths: list[list[tuple[int, int]]] = []
    bounce_counts: list[list[int]] = []
    for col in seed_cols:
        path, bounces = _trace_path(height, width, col)
        paths.append(path)
        bounce_counts.append(bounces)

    candidate_map = {
        seed_idx: _collect_hit_candidates(paths, bounce_counts, seed_idx, (height, width))
        for seed_idx in range(seed_count)
    }
    chosen = _choose_hit_seed_set(candidate_map, right_seed_idx, seed_count)
    if chosen is None:
        return None
    hit_seed_indices, bounce_seed_idx = chosen
    hit_blocks = _assign_hit_blocks(candidate_map, hit_seed_indices, bounce_seed_idx)
    if hit_blocks is None:
        return None

    forbidden = set()
    for block in hit_blocks.values():
        forbidden |= block["halo"]
    path_union = {cell for path in paths for cell in path}
    unhit_blocks = _select_unhit_blocks(path_union, (height, width), forbidden)
    if unhit_blocks is None:
        return None

    inp, expected = _render_input_and_expected(
        (height, width), seed_cols, seed_colors, hit_blocks, unhit_blocks
    )

    assert len(hit_blocks) >= 1
    assert len(hit_blocks) < seed_count
    assert len(unhit_blocks) >= 1
    assert any(block["uses_bounce"] for block in hit_blocks.values())

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-ray recolor input")
