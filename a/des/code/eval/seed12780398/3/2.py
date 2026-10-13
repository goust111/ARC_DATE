from __future__ import annotations

from collections import deque
from functools import lru_cache
import random

import numpy as np


BG = 0
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG_DIRS = DIRS + ((1, 1), (1, -1), (-1, 1), (-1, -1))
COLORS = tuple(range(1, 10))

# Generator bounds.
PAIR_COUNT_RANGE = (2, 4)
BBOX_HEIGHT_RANGE = (5, 8)
BBOX_WIDTH_RANGE = (5, 8)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
PLACEMENT_MARGIN = 1
GENERATION_ATTEMPTS = 400


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr, dc in DIAG_DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _mask_from_coords(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors8(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox, mask = _mask_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "shape": tuple(map(int, mask.shape)),
                    "area": int(mask.sum()),
                }
            )

    return comps


def _are_complements(a: dict, b: dict) -> bool:
    if a["shape"] != b["shape"]:
        return False
    return bool(np.array_equal(a["mask"], ~b["mask"]))


def _choose_large_small(a: dict, b: dict) -> tuple[dict, dict]:
    if int(a["area"]) > int(b["area"]):
        return a, b
    if int(b["area"]) > int(a["area"]):
        return b, a
    if tuple(map(int, a["bbox"])) <= tuple(map(int, b["bbox"])):
        return a, b
    return b, a


def _best_pairs(comps: list[dict]) -> list[tuple[int, int]]:
    n = len(comps)
    candidates = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            candidates[i][j] = _are_complements(comps[i], comps[j])

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if mask == 0:
            return 0, 0, ()

        bit = mask & -mask
        i = int(bit.bit_length() - 1)

        best_count, best_area, best_pairs = solve(mask ^ (1 << i))
        for j in range(i + 1, n):
            if not (mask & (1 << j)) or not candidates[i][j]:
                continue
            count, area, pairs = solve(mask ^ (1 << i) ^ (1 << j))
            count += 1
            area += int(comps[i]["area"]) + int(comps[j]["area"])
            if (count, area) > (best_count, best_area):
                best_count = count
                best_area = area
                best_pairs = ((i, j),) + pairs

        return best_count, best_area, best_pairs

    return list(solve((1 << n) - 1)[2])


def _paint_rectangle(out: np.ndarray, bbox: tuple[int, int, int, int], fill: int, border: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(fill)
    out[r0, c0 : c1 + 1] = int(border)
    out[r1, c0 : c1 + 1] = int(border)
    out[r0 : r1 + 1, c0] = int(border)
    out[r0 : r1 + 1, c1] = int(border)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, merge each pair at the larger fragment's bbox,
    and recolor the merged rectangle according to the area-based rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g)
    out = g.copy()

    for i, j in _best_pairs(comps):
        a = comps[i]
        b = comps[j]
        large, small = _choose_large_small(a, b)

        out[a["coords"][:, 0], a["coords"][:, 1]] = BG
        out[b["coords"][:, 0], b["coords"][:, 1]] = BG
        _paint_rectangle(out, large["bbox"], int(large["color"]), int(small["color"]))

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, coords[0])
    q = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    total = 0

    while q:
        r, c = q.popleft()
        total += 1
        for nr, nc in _neighbors8(r, c, h, w):
            if not mask[nr, nc] or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))

    return total == int(mask.sum())


def _edge_mixed(mask: np.ndarray) -> bool:
    edges = (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    return all(bool(edge.any()) and not bool(edge.all()) for edge in edges)


def _irregular(mask: np.ndarray) -> bool:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    return bool(
        any(0 < int(v) < mask.shape[1] for v in row_counts)
        and any(0 < int(v) < mask.shape[0] for v in col_counts)
    )


def _random_shortest_path(
    start: tuple[int, int], goal: tuple[int, int], blocked: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = map(int, blocked.shape)
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cell = q.popleft()
        if cell == goal:
            break
        nbrs = list(_neighbors(cell[0], cell[1], h, w))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            nxt = (int(nr), int(nc))
            if nxt in parent:
                continue
            if blocked[nr, nc] and nxt not in {start, goal}:
                continue
            parent[nxt] = cell
            q.append(nxt)

    if goal not in parent:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _grow_connected_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    top = (0, random.randint(1, w - 2))
    bottom = (h - 1, random.randint(1, w - 2))
    left = (random.randint(1, h - 2), 0)
    right = (random.randint(1, h - 2), w - 1)
    cycle = [top, right, bottom, left]
    if random.random() < 0.5:
        cycle = [top, left, bottom, right]
    shift = random.randrange(4)
    anchors = cycle[shift:] + cycle[:shift]

    mask = np.zeros((h, w), dtype=bool)
    current = anchors[0]
    mask[current] = True
    for anchor in anchors[1:]:
        path = _random_shortest_path(current, anchor, mask)
        if path is None:
            return None
        for cell in path:
            mask[cell] = True
        current = anchor

    if int(mask.sum()) > int(target_area):
        return None

    while int(mask.sum()) < int(target_area):
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors(int(r), int(c), h, w):
                if mask[nr, nc]:
                    continue
                frontier.add((int(nr), int(nc)))
        if not frontier:
            return None

        frontier_list = list(frontier)
        random.shuffle(frontier_list)
        pick = None
        for nr, nc in frontier_list:
            trial = mask.copy()
            trial[nr, nc] = True
            if _edge_mixed(trial) and _connected(~trial):
                pick = (nr, nc)
                break
        if pick is None:
            return None
        mask[pick] = True

    return mask


def _random_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        split = random.randint(1, h - 2)
        top_cuts = [random.randint(1, w - 2)]
        for _row in range(1, split + 1):
            prev = int(top_cuts[-1])
            lo = max(1, prev - 1)
            hi = min(w - 2, prev + 1)
            top_cuts.append(random.randint(lo, hi))

        bottom_rows = h - split - 1
        bottom_cuts = [random.choice([max(1, top_cuts[-1] - 1), top_cuts[-1], min(w - 2, top_cuts[-1] + 1)])]
        for _row in range(1, bottom_rows):
            prev = int(bottom_cuts[-1])
            lo = max(1, prev - 1)
            hi = min(w - 2, prev + 1)
            bottom_cuts.append(random.randint(lo, hi))

        mask = np.zeros((h, w), dtype=bool)
        for r in range(split + 1):
            mask[r, : int(top_cuts[r])] = True
        for idx, r in enumerate(range(split + 1, h)):
            mask[r, int(bottom_cuts[idx]) :] = True

        other = ~mask
        if not (_connected(mask) and _connected(other)):
            continue
        if not (_edge_mixed(mask) and _edge_mixed(other)):
            continue
        if not (_irregular(mask) and _irregular(other)):
            continue
        if int(mask.sum()) == int(other.sum()):
            continue

        turns = random.randrange(4) if h == w else random.choice([0, 2])
        mask = np.rot90(mask, turns)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        other = ~mask
        return mask.copy(), other.copy()

    return None


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, int(r0) - PLACEMENT_MARGIN)
    cc0 = max(0, int(c0) - PLACEMENT_MARGIN)
    rr1 = min(int(occupied.shape[0]), int(r0 + h + PLACEMENT_MARGIN))
    cc1 = min(int(occupied.shape[1]), int(c0 + w + PLACEMENT_MARGIN))
    occupied[rr0:rr1, cc0:cc1] = True


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, int(r0) - PLACEMENT_MARGIN)
    cc0 = max(0, int(c0) - PLACEMENT_MARGIN)
    rr1 = min(int(occupied.shape[0]), int(r0 + h + PLACEMENT_MARGIN))
    cc1 = min(int(occupied.shape[1]), int(c0 + w + PLACEMENT_MARGIN))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_bbox(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = map(int, occupied.shape)
    positions = [
        (r, c)
        for r in range(H - h + 1)
        for c in range(W - w + 1)
        if _can_place(occupied, r, c, h, w)
    ]
    if not positions:
        return None
    r0, c0 = random.choice(positions)
    _reserve(occupied, r0, c0, h, w)
    return int(r0), int(c0)


def _estimate_grid_size(dims: list[tuple[int, int]], singleton_count: int) -> tuple[int, int]:
    footprint = sum((h + 2 * PLACEMENT_MARGIN) * (w + 2 * PLACEMENT_MARGIN) for h, w in dims)
    footprint += singleton_count * (1 + 2 * PLACEMENT_MARGIN) ** 2
    side = int(np.ceil(np.sqrt(footprint * 1.35)))
    h_lo = max(GRID_HEIGHT_RANGE[0], side - 2)
    h_hi = min(GRID_HEIGHT_RANGE[1], side + 4)
    w_lo = max(GRID_WIDTH_RANGE[0], side - 2)
    w_hi = min(GRID_WIDTH_RANGE[1], side + 4)
    if h_lo > h_hi:
        h_lo = h_hi = GRID_HEIGHT_RANGE[1]
    if w_lo > w_hi:
        w_lo = w_hi = GRID_WIDTH_RANGE[1]
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _expected_from_spec(
    shape: tuple[int, int], pairs: list[dict], singletons: list[tuple[int, int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for row, col, color in singletons:
        out[int(row), int(col)] = int(color)

    for pair in pairs:
        a = pair["a"]
        b = pair["b"]
        large, small = _choose_large_small(a, b)
        _paint_rectangle(out, large["bbox"], int(large["color"]), int(small["color"]))

    return out


def _build_component_record(mask: np.ndarray, color: int, top: int, left: int) -> dict:
    coords = np.argwhere(mask)
    coords = coords + np.array([int(top), int(left)], dtype=int)
    bbox, local_mask = _mask_from_coords(coords)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "mask": local_mask,
        "shape": tuple(map(int, local_mask.shape)),
        "area": int(local_mask.sum()),
    }


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    max_singletons = max(0, 9 - 2 * pair_count)
    if max_singletons > 0 and random.random() < 0.5:
        singleton_count = random.randint(1, min(2, max_singletons))
    else:
        singleton_count = 0

    total_colors = 2 * pair_count + singleton_count
    colors = random.sample(COLORS, total_colors)

    dims_used: set[tuple[int, int]] = set()
    pair_specs: list[tuple[np.ndarray, np.ndarray]] = []
    dims: list[tuple[int, int]] = []

    for _ in range(pair_count):
        made = False
        choices = [
            (h, w)
            for h in range(BBOX_HEIGHT_RANGE[0], BBOX_HEIGHT_RANGE[1] + 1)
            for w in range(BBOX_WIDTH_RANGE[0], BBOX_WIDTH_RANGE[1] + 1)
            if (h, w) not in dims_used
        ]
        random.shuffle(choices)
        for h, w in choices:
            parts = _random_partition(h, w)
            if parts is None:
                continue
            dims_used.add((int(h), int(w)))
            pair_specs.append(parts)
            dims.append((int(h), int(w)))
            made = True
            break
        if not made:
            return None

    H, W = _estimate_grid_size(dims * 2, singleton_count)
    occupied = np.zeros((H, W), dtype=bool)
    inp = np.zeros((H, W), dtype=int)
    pairs: list[dict] = []
    color_idx = 0

    for mask_a, mask_b in pair_specs:
        h, w = map(int, mask_a.shape)
        pos_a = _place_bbox(occupied, h, w)
        pos_b = _place_bbox(occupied, h, w)
        if pos_a is None or pos_b is None:
            return None

        color_a = int(colors[color_idx])
        color_b = int(colors[color_idx + 1])
        color_idx += 2
        _stamp_mask(inp, mask_a, pos_a[0], pos_a[1], color_a)
        _stamp_mask(inp, mask_b, pos_b[0], pos_b[1], color_b)

        comp_a = _build_component_record(mask_a, color_a, pos_a[0], pos_a[1])
        comp_b = _build_component_record(mask_b, color_b, pos_b[0], pos_b[1])
        pairs.append({"a": comp_a, "b": comp_b})

    singletons: list[tuple[int, int, int]] = []
    for _ in range(singleton_count):
        pos = _place_bbox(occupied, 1, 1)
        if pos is None:
            return None
        color = int(colors[color_idx])
        color_idx += 1
        inp[pos[0], pos[1]] = color
        singletons.append((int(pos[0]), int(pos[1]), color))

    expected = _expected_from_spec(inp.shape, pairs, singletons)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
