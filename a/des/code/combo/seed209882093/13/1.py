from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
TIE_MODE_WEIGHT = 2
GENERIC_MODE_WEIGHT = 6
ALL_SEEDED_WEIGHT = 1


def _analyze_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    region_id = -np.ones((h, w), dtype=int)
    regions: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if int(grid[r0, c0]) == GRAY or int(region_id[r0, c0]) != -1:
                continue

            q = deque([(r0, c0)])
            region_id[r0, c0] = len(regions)
            cells: list[tuple[int, int]] = []
            seeds: list[tuple[int, int, int]] = []
            min_r = max_r = r0
            min_c = max_c = c0
            touches_border = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_border = True

                value = int(grid[r, c])
                if value not in (BG, GRAY):
                    seeds.append((r, c, value))

                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != GRAY and int(region_id[nr, nc]) == -1:
                        region_id[nr, nc] = len(regions)
                        q.append((nr, nc))

            if len(seeds) > 1:
                raise AssertionError("each region may contain at most one seed pixel")

            regions.append(
                {
                    "cells": cells,
                    "cell_set": set(cells),
                    "area": len(cells),
                    "bbox": (min_r, min_c, max_r, max_c),
                    "touches_border": touches_border,
                    "seed": seeds[0] if seeds else None,
                    "neighbors": set(),
                }
            )

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GRAY:
                continue

            up = int(region_id[r - 1, c]) if r > 0 else -1
            down = int(region_id[r + 1, c]) if r + 1 < h else -1
            left = int(region_id[r, c - 1]) if c > 0 else -1
            right = int(region_id[r, c + 1]) if c + 1 < w else -1

            if up >= 0 and down >= 0 and up != down:
                regions[up]["neighbors"].add(down)
                regions[down]["neighbors"].add(up)
            if left >= 0 and right >= 0 and left != right:
                regions[left]["neighbors"].add(right)
                regions[right]["neighbors"].add(left)

    return regions, region_id


def _winner_key(region: dict, seed: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(region["area"]), -int(seed[0]), -int(seed[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-delimited region from its seed; empty regions inherit the color of the
    largest adjacent seeded region, breaking area ties by smaller seed row.
    """
    g = np.asarray(grid, dtype=int)
    regions, _ = _analyze_regions(g)
    out = g.copy()

    seeded: dict[int, tuple[int, int, int]] = {
        rid: region["seed"] for rid, region in enumerate(regions) if region["seed"] is not None
    }

    for rid, seed in seeded.items():
        color = int(seed[2])
        for r, c in regions[rid]["cells"]:
            out[r, c] = color

    for rid, region in enumerate(regions):
        if rid in seeded:
            continue
        if region["touches_border"]:
            continue

        candidates = [nbr for nbr in region["neighbors"] if nbr in seeded]
        if not candidates:
            continue

        chosen = max(candidates, key=lambda nbr: _winner_key(regions[nbr], seeded[nbr]))
        color = int(seeded[chosen][2])
        for r, c in region["cells"]:
            out[r, c] = color

    return out


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _cell_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def _partition_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _cell_boxes(heights: list[int], widths: list[int]) -> tuple[list[int], list[int], int, int]:
    row_starts: list[int] = []
    cur = 1
    for height in heights:
        row_starts.append(cur)
        cur += height + 1
    h = cur

    col_starts: list[int] = []
    cur = 1
    for width in widths:
        col_starts.append(cur)
        cur += width + 1
    w = cur
    return row_starts, col_starts, h, w


def _render_partition(
    heights: list[int],
    widths: list[int],
    opened_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    row_starts, col_starts, h, w = _cell_boxes(heights, widths)
    grid = np.full((h, w), GRAY, dtype=int)

    for r, height in enumerate(heights):
        for c, width in enumerate(widths):
            r0 = row_starts[r]
            c0 = col_starts[c]
            grid[r0 : r0 + height, c0 : c0 + width] = BG

    for (r1, c1), (r2, c2) in opened_edges:
        if r1 == r2:
            r = r1
            c = min(c1, c2)
            wall_col = col_starts[c] + widths[c]
            r0 = row_starts[r]
            grid[r0 : r0 + heights[r], wall_col] = BG
        else:
            r = min(r1, r2)
            c = c1
            wall_row = row_starts[r] + heights[r]
            c0 = col_starts[c]
            grid[wall_row, c0 : c0 + widths[c]] = BG

    return grid


def _random_partition() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    if rows * cols < 4:
        return None

    if random.random() < 0.3:
        shared_height = random.randint(2, 5)
        heights = [shared_height] * rows
    else:
        heights = [random.randint(2, 5) for _ in range(rows)]

    if random.random() < 0.3:
        shared_width = random.randint(2, 5)
        widths = [shared_width] * cols
    else:
        widths = [random.randint(2, 5) for _ in range(cols)]

    row_starts, col_starts, h, w = _cell_boxes(heights, widths)
    _ = (row_starts, col_starts)
    if h > 25 or w > 25:
        return None

    dsu = _DSU(rows * cols)
    edges = _partition_edges(rows, cols)
    random.shuffle(edges)

    min_regions = 4 if rows * cols >= 4 else 3
    target_regions = random.randint(min_regions, min(rows * cols, 8))
    max_cells_per_region = random.randint(2, min(5, rows * cols))

    opened: list[tuple[tuple[int, int], tuple[int, int]]] = []
    region_count = rows * cols
    for a, b in edges:
        if region_count <= target_regions:
            break
        ia = _cell_index(a[0], a[1], cols)
        ib = _cell_index(b[0], b[1], cols)
        ra = dsu.find(ia)
        rb = dsu.find(ib)
        if ra == rb:
            continue
        if dsu.size[ra] + dsu.size[rb] > max_cells_per_region and random.random() < 0.8:
            continue
        if dsu.union(ia, ib):
            opened.append((a, b))
            region_count -= 1

    if region_count > target_regions:
        for a, b in edges:
            if region_count <= target_regions:
                break
            ia = _cell_index(a[0], a[1], cols)
            ib = _cell_index(b[0], b[1], cols)
            if dsu.union(ia, ib):
                opened.append((a, b))
                region_count -= 1

    return _render_partition(heights, widths, opened)


def _tie_partition() -> np.ndarray:
    rows = random.randint(3, 5)
    height = random.randint(2, 4)
    width = random.randint(3, 6)
    heights = [height] * rows
    widths = [width]
    return _render_partition(heights, widths, [])


def _seed_position(region: dict) -> tuple[int, int]:
    preferred = []
    for r, c in region["cells"]:
        exposed = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in region["cell_set"]:
                exposed += 1
        if exposed <= 2:
            preferred.append((r, c))
    pool = preferred if preferred else region["cells"]
    return random.choice(pool)


def _vertical_ordered_pair(a: dict, b: dict) -> tuple[int, int] | None:
    a_min_r, _, a_max_r, _ = a["bbox"]
    b_min_r, _, b_max_r, _ = b["bbox"]
    if a_max_r < b_min_r:
        return (0, 1)
    if b_max_r < a_min_r:
        return (1, 0)
    return None


def _find_tie_focus(regions: list[dict]) -> tuple[int, int, int] | None:
    for rid, region in enumerate(regions):
        neighbors = sorted(region["neighbors"])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                b = neighbors[j]
                if regions[a]["area"] != regions[b]["area"]:
                    continue
                order = _vertical_ordered_pair(regions[a], regions[b])
                if order is None:
                    continue
                top = a if order[0] == 0 else b
                bottom = b if order[0] == 0 else a
                return rid, top, bottom
    return None


def _find_area_focus(regions: list[dict]) -> tuple[int, int, int] | None:
    candidates = [rid for rid, region in enumerate(regions) if len(region["neighbors"]) >= 2]
    random.shuffle(candidates)
    for rid in candidates:
        neighbors = list(regions[rid]["neighbors"])
        random.shuffle(neighbors)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                b = neighbors[j]
                if regions[a]["area"] != regions[b]["area"]:
                    return rid, a, b
    return None


def _cover_empty_regions(
    seeded: set[int],
    forced_empty: set[int],
    avoid_first: set[int],
    adjacency: list[set[int]],
    max_seed_count: int,
) -> set[int] | None:
    all_regions = set(range(len(adjacency)))

    def uncovered() -> list[int]:
        empties = all_regions - seeded
        return [rid for rid in empties if not (adjacency[rid] & seeded)]

    while True:
        missing = uncovered()
        if not missing:
            return seeded
        candidates = list(all_regions - seeded - forced_empty)
        if not candidates:
            return None
        preferred = [rid for rid in candidates if rid not in avoid_first]
        pool = preferred if preferred else candidates
        best = None
        best_key = None
        for rid in pool:
            score = sum(1 for miss in missing if rid in adjacency[miss])
            if score == 0:
                continue
            key = (score, random.random())
            if best is None or key > best_key:
                best = rid
                best_key = key
        if best is None:
            return None
        seeded.add(best)
        if len(seeded) > max_seed_count:
            return None


def _choose_seeded_regions(case_kind: str, regions: list[dict]) -> set[int] | None:
    n_regions = len(regions)
    adjacency = [set(region["neighbors"]) for region in regions]
    all_regions = set(range(n_regions))
    max_seed_count = min(len(SEED_COLORS), n_regions)

    if case_kind == "all_seeded":
        return all_regions if n_regions <= len(SEED_COLORS) else None

    seeded: set[int] = set()
    forced_empty: set[int] = set()
    avoid_first: set[int] = set()

    if case_kind == "tie":
        focus = _find_tie_focus(regions)
        if focus is None:
            return None
        empty_rid, top_rid, bottom_rid = focus
        seeded.update([top_rid, bottom_rid])
        forced_empty.add(empty_rid)
    else:
        focus = _find_area_focus(regions)
        if focus is not None:
            empty_rid, first_rid, second_rid = focus
            seeded.update([first_rid, second_rid])
            forced_empty.add(empty_rid)
            avoid_first.update(adjacency[empty_rid] - seeded)

    if len(seeded) > max_seed_count:
        return None

    for rid in random.sample(list(all_regions - seeded - forced_empty), len(all_regions - seeded - forced_empty)):
        if len(seeded) >= max_seed_count:
            break
        if rid in avoid_first and random.random() < 0.85:
            continue
        if random.random() < 0.45:
            seeded.add(rid)

    if not seeded:
        seeded.add(random.randrange(n_regions))

    if len(seeded) == n_regions:
        removable = [
            rid
            for rid in seeded
            if rid not in forced_empty and len(adjacency[rid] & (seeded - {rid})) > 0
        ]
        if removable and random.random() < 0.9:
            seeded.remove(random.choice(removable))

    seeded = _cover_empty_regions(seeded, forced_empty, avoid_first, adjacency, max_seed_count)
    if seeded is None or len(seeded) < 2:
        return None

    if case_kind != "all_seeded" and len(seeded) == n_regions:
        removable = [
            rid
            for rid in seeded
            if rid not in forced_empty and len(adjacency[rid] & (seeded - {rid})) > 0
        ]
        if removable:
            seeded.remove(random.choice(removable))
            seeded = _cover_empty_regions(seeded, forced_empty, avoid_first, adjacency, max_seed_count)
            if seeded is None:
                return None

    if case_kind == "tie":
        focus = _find_tie_focus(regions)
        assert focus is not None
        empty_rid, top_rid, _bottom_rid = focus
        if empty_rid in seeded:
            return None
        focus_neighbors = regions[empty_rid]["neighbors"] & seeded
        if focus_neighbors != {top_rid, _bottom_rid}:
            return None

    return seeded


def _expected_output(
    base_grid: np.ndarray,
    regions: list[dict],
    seeded_info: dict[int, tuple[int, int, int]],
) -> np.ndarray | None:
    out = base_grid.copy()
    fill_color: dict[int, int] = {}

    for rid, seed in seeded_info.items():
        fill_color[rid] = int(seed[2])

    for rid, region in enumerate(regions):
        if rid in fill_color:
            continue
        candidates = [nbr for nbr in region["neighbors"] if nbr in seeded_info]
        if not candidates:
            return None
        chosen = max(candidates, key=lambda nbr: _winner_key(regions[nbr], seeded_info[nbr]))
        fill_color[rid] = int(seeded_info[chosen][2])

    for rid, color in fill_color.items():
        for r, c in regions[rid]["cells"]:
            out[r, c] = color
    return out


def _make_case(case_kind: str) -> np.ndarray | None:
    if case_kind == "tie":
        base_grid = _tie_partition()
    else:
        base_grid = _random_partition()
        if base_grid is None:
            return None

    regions, _ = _analyze_regions(base_grid)
    if not (3 <= len(regions) <= 8):
        return None

    if case_kind == "generic" and _find_area_focus(regions) is None:
        return None

    seeded_regions = _choose_seeded_regions(case_kind, regions)
    if seeded_regions is None:
        return None

    colors = random.sample(SEED_COLORS, len(seeded_regions))
    seeded_info: dict[int, tuple[int, int, int]] = {}
    inp = base_grid.copy()
    for rid, color in zip(sorted(seeded_regions), colors):
        sr, sc = _seed_position(regions[rid])
        inp[sr, sc] = color
        seeded_info[rid] = (sr, sc, color)

    expected = _expected_output(base_grid, regions, seeded_info)
    if expected is None:
        return None
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    kinds = ["tie", "generic", "all_seeded"]
    weights = [TIE_MODE_WEIGHT, GENERIC_MODE_WEIGHT, ALL_SEEDED_WEIGHT]
    for _ in range(GEN_RETRIES):
        case_kind = random.choices(kinds, weights=weights, k=1)[0]
        grid = _make_case(case_kind)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
