from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
MAX_SIDE = 30

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
RECTANGLE_DIMS = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 6)
    if 6 <= h * w <= 18
)

GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
GENERATION_ATTEMPTS = 400
PARTITION_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 50

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _bbox_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _is_connected(coords) -> bool:
    coord_set = {tuple(cell) for cell in coords}
    if not coord_set:
        return False

    start = next(iter(coord_set))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coord_set)


def _is_full_bbox(coords: tuple[tuple[int, int], ...]) -> bool:
    h, w = _bbox_dims(coords)
    return len(coords) == h * w


def _make_piece(coords, color: int, pair_id: int) -> dict:
    rel = _normalize_coords(coords)
    h, w = _bbox_dims(rel)
    return {
        "cells": rel,
        "h": h,
        "w": w,
        "area": len(rel),
        "color": int(color),
        "pair_id": int(pair_id),
    }


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            objects.append(_make_piece(coords, color=color, pair_id=-1))

    return objects


def _factor_dims(area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        dims.append((h, w))
        if h != w:
            dims.append((w, h))
    return sorted(dims)


def _all_pair_layouts(a: dict, b: dict) -> list[dict]:
    layouts = []
    seen = set()
    for anchor_is_a in (True, False):
        anchor = a if anchor_is_a else b
        shifted = b if anchor_is_a else a
        anchor_cells = set(anchor["cells"])
        total_area = anchor["area"] + shifted["area"]

        for h, w in _factor_dims(total_area):
            if h < anchor["h"] or w < anchor["w"] or h < shifted["h"] or w < shifted["w"]:
                continue
            for dr in range(h - shifted["h"] + 1):
                for dc in range(w - shifted["w"] + 1):
                    moved_cells = tuple((r + dr, c + dc) for r, c in shifted["cells"])
                    if any(cell in anchor_cells for cell in moved_cells):
                        continue
                    key = (h, w, anchor_is_a, dr, dc)
                    if key in seen:
                        continue
                    seen.add(key)
                    layouts.append(
                        {
                            "h": h,
                            "w": w,
                            "anchor_is_a": anchor_is_a,
                            "offset": (dr, dc),
                        }
                    )

    layouts.sort(key=lambda item: (item["h"], item["w"], item["anchor_is_a"], item["offset"]))
    return layouts


def _solve_pairing(objects: list[dict]) -> list[tuple[int, int, dict]]:
    n = len(objects)
    compatibility: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            layouts = _all_pair_layouts(objects[i], objects[j])
            if layouts:
                compatibility[(i, j)] = layouts[0]

    def backtrack(remaining: tuple[int, ...]):
        if not remaining:
            return []
        i = remaining[0]
        for j in remaining[1:]:
            key = (i, j)
            if key not in compatibility:
                continue
            next_remaining = tuple(x for x in remaining if x not in key)
            sub = backtrack(next_remaining)
            if sub is not None:
                return [(i, j, compatibility[key])] + sub
        return None

    solution = backtrack(tuple(range(n)))
    assert solution is not None
    return solution


def _render_pair(a: dict, b: dict, layout: dict) -> np.ndarray:
    rect = np.full((layout["h"], layout["w"]), BACKGROUND, dtype=int)
    if layout["anchor_is_a"]:
        a_cells = a["cells"]
        b_cells = tuple((r + layout["offset"][0], c + layout["offset"][1]) for r, c in b["cells"])
    else:
        b_cells = b["cells"]
        a_cells = tuple((r + layout["offset"][0], c + layout["offset"][1]) for r, c in a["cells"])

    for r, c in a_cells:
        rect[r, c] = a["color"]
    for r, c in b_cells:
        rect[r, c] = b["color"]

    assert not np.any(rect == BACKGROUND)
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, rebuild each pair as a rectangle, and pack the rectangles from smallest to largest.

    Input structure:
    - The background is black.
    - Several scattered 4-connected monochrome fragments appear on the grid.
    - The fragment count is even, and each fragment belongs to exactly one complementary pair.

    Rule:
    - Find pairs of fragments that can fill a solid rectangle exactly after translation only, with no rotation or reflection.
    - Reconstruct each matched rectangle while preserving the two original fragment colors inside it.
    - Sort the reconstructed rectangles by area, break ties by rectangle width, then place them left-to-right with one black separator column and their top edges aligned to the first row.

    Key insight:
    - The hidden relation is purely geometric: two scattered shapes belong together when one occupies exactly the cells the other leaves empty inside some rectangle.
    """
    g = np.asarray(grid, dtype=int)
    objects = _find_objects(g)
    assert objects and len(objects) % 2 == 0

    pairings = _solve_pairing(objects)
    rectangles = []
    for order_idx, (i, j, layout) in enumerate(pairings):
        rect = _render_pair(objects[i], objects[j], layout)
        rectangles.append(
            {
                "grid": rect,
                "h": layout["h"],
                "w": layout["w"],
                "order": order_idx,
            }
        )

    rectangles.sort(key=lambda item: (item["h"] * item["w"], item["w"], item["h"], item["order"]))
    out_h = max(item["h"] for item in rectangles)
    out_w = sum(item["w"] for item in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)

    col = 0
    for item in rectangles:
        rect = item["grid"]
        out[: item["h"], col : col + item["w"]] = rect
        col += item["w"] + 1
    return out


def _random_connected_subset(h: int, w: int, target_area: int, rng: np.random.Generator):
    start = (int(rng.integers(h)), int(rng.integers(w)))
    subset = {start}
    while len(subset) < target_area:
        frontier = []
        for r, c in subset:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in subset:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = tuple(dict.fromkeys(frontier))
        subset.add(frontier[int(rng.integers(len(frontier)))])
    return subset


def _thin_l_partition(h: int, w: int, rng: np.random.Generator):
    arm_h = int(rng.integers(2, h + 1))
    arm_w = int(rng.integers(2, w + 1))
    if arm_h == h and arm_w == w:
        if h > 2 and (w == 2 or rng.random() < 0.5):
            arm_h -= 1
        elif w > 2:
            arm_w -= 1
        else:
            return None

    base = {(0, c) for c in range(arm_w)}
    base |= {(r, 0) for r in range(arm_h)}

    corner = int(rng.integers(4))
    if corner == 0:
        first = base
    elif corner == 1:
        first = {(r, w - 1 - c) for r, c in base}
    elif corner == 2:
        first = {(h - 1 - r, c) for r, c in base}
    else:
        first = {(h - 1 - r, w - 1 - c) for r, c in base}

    all_cells = {(r, c) for r in range(h) for c in range(w)}
    second = all_cells - first
    return first, second


def _random_partition(h: int, w: int, rng: np.random.Generator):
    total = h * w
    target_area = int(rng.integers(3, total - 2))
    first = _random_connected_subset(h, w, target_area, rng)
    if first is None:
        return None
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    second = all_cells - first
    return first, second


def _valid_partition(first, second) -> bool:
    if first is None or second is None:
        return False
    if len(first) < 3 or len(second) < 3:
        return False
    if not _is_connected(first) or not _is_connected(second):
        return False
    norm_first = _normalize_coords(first)
    norm_second = _normalize_coords(second)
    if _is_full_bbox(norm_first) or _is_full_bbox(norm_second):
        return False
    return True


def _generate_partition(h: int, w: int, rng: np.random.Generator):
    for _ in range(PARTITION_ATTEMPTS):
        if rng.random() < 0.5:
            candidate = _random_partition(h, w, rng)
            fallback = _thin_l_partition(h, w, rng)
        else:
            candidate = _thin_l_partition(h, w, rng)
            fallback = _random_partition(h, w, rng)

        if candidate is not None and _valid_partition(*candidate):
            return candidate
        if fallback is not None and _valid_partition(*fallback):
            return fallback
    return None


def _pairing_is_unambiguous(pieces: list[dict]) -> bool:
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            layouts = _all_pair_layouts(pieces[i], pieces[j])
            same_pair = pieces[i]["pair_id"] == pieces[j]["pair_id"]
            if same_pair and len(layouts) != 1:
                return False
            if not same_pair and layouts:
                return False
    return True


def _scatter_pieces(pieces: list[dict], h: int, w: int, rng: np.random.Generator):
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pieces)
    for piece_idx in rng.permutation(len(pieces)):
        piece = pieces[int(piece_idx)]
        candidates = []
        for r0 in range(h - piece["h"] + 1):
            for c0 in range(w - piece["w"] + 1):
                cells = [(r0 + r, c0 + c) for r, c in piece["cells"]]
                if any(blocked[r, c] for r, c in cells):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        placements[int(piece_idx)] = (r0, c0)
        for r, c in piece["cells"]:
            ar = r0 + r
            ac = c0 + c
            for nr in range(ar - 1, ar + 2):
                for nc in range(ac - 1, ac + 2):
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True

    return placements


def _build_grid(pieces: list[dict], rng: np.random.Generator):
    max_piece_h = max(piece["h"] for piece in pieces)
    max_piece_w = max(piece["w"] for piece in pieces)
    min_side = max(GRID_MIN_SIDE, max_piece_h + 4, max_piece_w + 4)

    for _ in range(PLACEMENT_ATTEMPTS):
        h = int(rng.integers(min_side, GRID_MAX_SIDE + 1))
        w = int(rng.integers(min_side, GRID_MAX_SIDE + 1))
        placements = _scatter_pieces(pieces, h, w, rng)
        if placements is None:
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for piece, placement in zip(pieces, placements):
            assert placement is not None
            r0, c0 = placement
            for r, c in piece["cells"]:
                grid[r0 + r, c0 + c] = piece["color"]
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    dim_indices = _RNG.permutation(len(RECTANGLE_DIMS))[:pair_count]
    dims = [RECTANGLE_DIMS[int(i)] for i in dim_indices]
    colors = _RNG.choice(np.array(COLOR_POOL, dtype=int), size=2 * pair_count, replace=False)

    pieces: list[dict] = []
    for pair_id, (h, w) in enumerate(dims):
        partition = _generate_partition(h, w, _RNG)
        if partition is None:
            return None
        first, second = partition
        if _RNG.random() < 0.5:
            first, second = second, first
        pieces.append(_make_piece(first, int(colors[2 * pair_id]), pair_id))
        pieces.append(_make_piece(second, int(colors[2 * pair_id + 1]), pair_id))

    if not _pairing_is_unambiguous(pieces):
        return None

    grid = _build_grid(pieces, _RNG)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > MAX_SIDE or max(out.shape) > MAX_SIDE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle sample")
