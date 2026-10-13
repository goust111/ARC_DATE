from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SPRITE_COLORS = (1, 2)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FIXED_SPRITES = (
    np.array(
        [
            [1, 0, 2, 0],
            [1, 2, 2, 0],
            [0, 2, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 1, 0],
            [0, 2, 1, 2],
            [0, 0, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 2, 2],
            [2, 2, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprite_and_anchors(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[int, int, int]]]:
    comps = _components(grid != BG)
    assert comps, "expected foreground"
    comps.sort(key=len, reverse=True)
    sprite_coords = comps[0]
    assert len(sprite_coords) > 1, "expected a multi-pixel sprite"
    for comp in comps[1:]:
        assert len(comp) == 1, "non-sprite foreground must be singleton anchors"

    r0, c0, r1, c1 = _bbox(sprite_coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    source_cells: set[tuple[int, int]] = set()
    for r, c in sprite_coords:
        patch[r - r0, c - c0] = int(grid[r, c])
        source_cells.add((r, c))

    anchors = [(r, c, int(grid[r, c])) for comp in comps[1:] for r, c in comp]
    return patch, source_cells, anchors


def _variants(patch: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(patch, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _pairs_from_patch(patch: np.ndarray, r0: int, c0: int) -> set[tuple[tuple[int, int], int]]:
    pairs: set[tuple[tuple[int, int], int]] = set()
    for rr, cc in np.argwhere(patch != BG):
        pairs.add(((r0 + int(rr), c0 + int(cc)), int(patch[rr, cc])))
    return pairs


def _candidate_record(
    variant: np.ndarray, grid_shape: tuple[int, int], r0: int, c0: int
) -> dict[str, object]:
    h, _w = variant.shape
    H, _W = grid_shape
    mirror_r0 = H - r0 - h
    top_pairs = _pairs_from_patch(variant, r0, c0)
    bottom_pairs = _pairs_from_patch(np.flipud(variant), mirror_r0, c0)
    all_pairs = top_pairs | bottom_pairs
    return {
        "variant": variant,
        "r0": int(r0),
        "c0": int(c0),
        "top_pairs": top_pairs,
        "bottom_pairs": bottom_pairs,
        "all_pairs": all_pairs,
        "cells": {pos for pos, _ in all_pairs},
    }


def _enumerate_candidates(
    sprite_patch: np.ndarray, source_cells: set[tuple[int, int]], grid_shape: tuple[int, int]
) -> list[dict[str, object]]:
    H, W = grid_shape
    half = H // 2
    candidates: list[dict[str, object]] = []
    for variant_index, variant in enumerate(_variants(sprite_patch)):
        h, w = variant.shape
        if h > half or w > W:
            continue
        for r0 in range(half - h + 1):
            for c0 in range(W - w + 1):
                cand = _candidate_record(variant, grid_shape, r0, c0)
                if cand["cells"] & source_cells:
                    continue
                cand["variant_index"] = variant_index
                candidates.append(cand)
    return candidates


def _matching_candidates(
    candidates: list[dict[str, object]], anchors: list[tuple[int, int, int]]
) -> list[dict[str, object]]:
    required = {((r, c), color) for r, c, color in anchors}
    return [cand for cand in candidates if required <= cand["all_pairs"]]


def _find_solution(grid: np.ndarray) -> tuple[dict[str, object], np.ndarray, list[tuple[int, int, int]]]:
    sprite_patch, source_cells, anchors = _extract_sprite_and_anchors(grid)
    assert anchors, "expected sparse anchors"
    candidates = _enumerate_candidates(sprite_patch, source_cells, grid.shape)
    matches = _matching_candidates(candidates, anchors)
    assert len(matches) == 1, "expected a unique transform and placement"
    return matches[0], sprite_patch, anchors


def _paint_pairs(grid: np.ndarray, pairs: set[tuple[tuple[int, int], int]]) -> None:
    for (r, c), color in pairs:
        grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place one rotated/flipped sprite in the upper half so its vertical mirror in the lower half explains all sparse anchors.

    Input structure:
    - A black background contains one multicolor 4-connected sprite made of blue and red pixels, plus several isolated blue/red singleton anchors.
    - The sprite is the unique large foreground component; every other foreground component is a one-pixel anchor.
    - Grid height is even, so the horizontal mirror axis lies between rows H/2-1 and H/2.

    Rule:
    - Separate the large sprite from the singleton anchors.
    - Generate the sprite's 8 rotation/reflection variants.
    - For every variant and every upper-half placement that stays above the axis, place the variant there and place its top-bottom mirror symmetrically below the axis.
    - Keep the unique candidate whose two placed copies cover every anchor with matching color.
    - Erase the original corner sprite and the anchors, then draw the full upper copy and its mirrored lower copy.

    Key insight:
    - The anchors are only sparse evidence, so rotation search and mirror-constrained placement must be solved together; neither half alone is guaranteed to determine the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    solution, _sprite_patch, _anchors = _find_solution(g)
    out = np.zeros_like(g)
    _paint_pairs(out, solution["all_pairs"])
    return out


def _canonical_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _grow_connected_mask(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (rng.randrange(h), rng.randrange(w))
    selected = {seed}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in selected:
                frontier.add((nr, nc))

    add_frontier(seed)
    while len(selected) < area:
        if not frontier:
            return _grow_connected_mask(rng, h, w, area)
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        add_frontier(cell)

    for r, c in selected:
        mask[r, c] = True

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _max_independent_size(mask: np.ndarray) -> int:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    best = 0
    for bits in range(1 << len(coords)):
        chosen: list[tuple[int, int]] = []
        ok = True
        for i, cell in enumerate(coords):
            if not (bits & (1 << i)):
                continue
            for other in chosen:
                if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1:
                    ok = False
                    break
            if not ok:
                break
            chosen.append(cell)
        if ok:
            best = max(best, len(chosen))
    return best


def _make_sprite(rng: random.Random) -> np.ndarray:
    for _ in range(500):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        area = rng.randint(6, min(8, h * w - 1))
        mask = _grow_connected_mask(rng, h, w, area)
        if _max_independent_size(mask) < 3:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        rng.shuffle(coords)
        split = rng.randint(2, area - 2)
        patch = np.zeros(mask.shape, dtype=int)
        for i, (r, c) in enumerate(coords):
            patch[r, c] = SPRITE_COLORS[0] if i < split else SPRITE_COLORS[1]

        keys = {_canonical_key(v) for v in _variants(patch)}
        if len(keys) != 8:
            continue
        return patch
    raise RuntimeError("failed to generate an asymmetric sprite")


def _corner_positions(rng: random.Random, H: int, W: int, h: int, w: int) -> list[tuple[int, int]]:
    top = [0, 1] if H - h >= 1 else [0]
    left = [0, 1] if W - w >= 1 else [0]
    bottom = [H - h, H - h - 1] if H - h - 1 >= 0 else [H - h]
    right = [W - w, W - w - 1] if W - w - 1 >= 0 else [W - w]
    positions = [
        (top[0], left[0]),
        (top[-1], right[0]),
        (bottom[0], left[-1]),
        (bottom[-1], right[-1]),
    ]
    rng.shuffle(positions)
    return list(dict.fromkeys((int(r), int(c)) for r, c in positions))


def _build_grid(
    shape: tuple[int, int], sprite_patch: np.ndarray, sprite_pos: tuple[int, int], anchors: list[tuple[int, int, int]]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    sr, sc = sprite_pos
    for rr, cc in np.argwhere(sprite_patch != BG):
        grid[sr + int(rr), sc + int(cc)] = int(sprite_patch[rr, cc])
    for r, c, color in anchors:
        assert grid[r, c] == BG
        grid[r, c] = color
    return grid


def _isolated_anchor_subset(
    rng: random.Random, target_pairs: list[tuple[tuple[int, int], int]], need: int, half: int
) -> list[tuple[int, int, int]] | None:
    for _ in range(300):
        shuffled = target_pairs[:]
        rng.shuffle(shuffled)
        chosen: list[tuple[tuple[int, int], int]] = []
        for pair in shuffled:
            (r, c), _color = pair
            if any(abs(r - rr) + abs(c - cc) == 1 for (rr, cc), _ in chosen):
                continue
            chosen.append(pair)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            continue
        upper = [p for p in chosen if p[0][0] < half]
        lower = [p for p in chosen if p[0][0] >= half]
        colors = {color for _, color in chosen}
        if len(upper) < 2 or len(lower) < 2 or colors != set(SPRITE_COLORS):
            continue
        return [(r, c, color) for (r, c), color in chosen]
    return None


def _matches_pairs(pairs: set[tuple[tuple[int, int], int]], need: set[tuple[tuple[int, int], int]]) -> bool:
    return need <= pairs


def _choose_anchor_subset(
    rng: random.Random,
    target: dict[str, object],
    candidates: list[dict[str, object]],
    half: int,
    require_counterexample: bool,
) -> list[tuple[int, int, int]] | None:
    target_pairs = list(target["all_pairs"])
    sizes = list(range(6, min(10, len(target_pairs)) + 1))
    rng.shuffle(sizes)
    sizes = sizes[:3]
    for need in sizes:
        for _ in range(60):
            anchors = _isolated_anchor_subset(rng, target_pairs, need, half)
            if anchors is None:
                continue
            full_need = {((r, c), color) for r, c, color in anchors}
            full_matches = [cand for cand in candidates if _matches_pairs(cand["all_pairs"], full_need)]
            if len(full_matches) != 1 or full_matches[0] is not target:
                continue

            top_need = {pair for pair in full_need if pair[0][0] < half}
            bottom_need = full_need - top_need
            top_matches = sum(_matches_pairs(cand["top_pairs"], top_need) for cand in candidates)
            bottom_matches = sum(_matches_pairs(cand["bottom_pairs"], bottom_need) for cand in candidates)
            if require_counterexample and top_matches <= 1 and bottom_matches <= 1:
                continue
            return anchors
    return None


def _expected_output(shape: tuple[int, int], pairs: set[tuple[tuple[int, int], int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _paint_pairs(out, pairs)
    return out


def _clear_singleton_anchor(
    pair: tuple[tuple[int, int], int],
    chosen: list[tuple[tuple[int, int], int]],
    source_cells: set[tuple[int, int]],
) -> bool:
    (r, c), _color = pair
    for rr, cc in source_cells:
        if abs(r - rr) + abs(c - cc) == 1:
            return False
    for (rr, cc), _ in chosen:
        if abs(r - rr) + abs(c - cc) == 1:
            return False
    return True


def _greedy_unique_anchors(
    rng: random.Random,
    target: dict[str, object],
    candidates: list[dict[str, object]],
    half: int,
    source_cells: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    pairs = list(target["all_pairs"])
    rng.shuffle(pairs)
    pairs.sort(key=lambda pair: (pair[0][0] >= half, pair[1], pair[0][0], pair[0][1]))

    chosen: list[tuple[tuple[int, int], int]] = []
    for pair in pairs:
        if not _clear_singleton_anchor(pair, chosen, source_cells):
            continue
        chosen.append(pair)
        if len(chosen) < 6:
            continue
        need = set(chosen)
        upper = [p for p in chosen if p[0][0] < half]
        lower = [p for p in chosen if p[0][0] >= half]
        colors = {color for _, color in chosen}
        if len(upper) < 2 or len(lower) < 2 or colors != set(SPRITE_COLORS):
            continue
        matches = [cand for cand in candidates if _matches_pairs(cand["all_pairs"], need)]
        if len(matches) == 1 and matches[0] is target:
            return [(r, c, color) for (r, c), color in chosen]
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(180):
        H = rng.choice([12, 14, 16])
        W = rng.choice([14, 15, 16, 17, 18])
        sprite_patch = rng.choice(FIXED_SPRITES).copy()
        if rng.random() < 0.5:
            sprite_patch = np.where(sprite_patch == 1, 2, np.where(sprite_patch == 2, 1, 0))
        h, w = sprite_patch.shape

        sprite_pos = None
        for pos in _corner_positions(rng, H, W, h, w):
            sprite_pos = pos
            break
        assert sprite_pos is not None

        source_cells = {
            (sprite_pos[0] + int(rr), sprite_pos[1] + int(cc))
            for rr, cc in np.argwhere(sprite_patch != BG)
        }
        candidates = _enumerate_candidates(sprite_patch, source_cells, (H, W))
        if len(candidates) < 2:
            continue

        rng.shuffle(candidates)
        anchors = None
        target = None
        for cand in candidates[:80]:
            anchors = _greedy_unique_anchors(rng, cand, candidates, H // 2, source_cells)
            if anchors is None:
                continue
            target = cand
            break
        if anchors is None or target is None:
            continue

        grid = _build_grid((H, W), sprite_patch, sprite_pos, anchors)

        comps = _components(grid != BG)
        areas = sorted((len(comp) for comp in comps), reverse=True)
        if not areas or areas[0] != int(np.count_nonzero(sprite_patch)) or any(area != 1 for area in areas[1:]):
            continue

        expected = _expected_output((H, W), target["all_pairs"])
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue

        solution, _sprite_check, anchors_check = _find_solution(grid)
        if solution is not target and solution["all_pairs"] != target["all_pairs"]:
            continue
        if len(anchors_check) != len(anchors):
            continue

        full_need = {((r, c), color) for r, c, color in anchors}
        top_need = {pair for pair in full_need if pair[0][0] < H // 2}
        bottom_need = full_need - top_need
        top_matches = sum(_matches_pairs(cand["top_pairs"], top_need) for cand in candidates)
        bottom_matches = sum(_matches_pairs(cand["bottom_pairs"], bottom_need) for cand in candidates)
        if top_matches <= 1 and bottom_matches <= 1:
            continue

        assert max(grid.shape) <= 30 and max(solved.shape) <= 30
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
