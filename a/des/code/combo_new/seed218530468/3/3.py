import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
SEED_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _region_distances(cells: list[tuple[int, int]], shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    h, w = shape
    region = set(cells)
    dist: dict[tuple[int, int], int] = {}
    q = deque()

    for r, c in cells:
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            dist[(r, c)] = 1
            q.append((r, c))
            continue
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in region:
                dist[(r, c)] = 1
                q.append((r, c))
                break

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in region and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))

    return dist


def _region_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for cells in _components4(grid != GREY):
        dist = _region_distances(cells, grid.shape)
        seeds = [(r, c) for r, c in cells if int(grid[r, c]) not in (BLACK, GREY)]
        infos.append(
            {
                "cells": cells,
                "dist": dist,
                "depth": max(dist.values()),
                "seeds": seeds,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow inward contour rings inside each grey-bounded region according to its seed count.

    Input structure:
    - The background is black and thin grey walls partition the canvas into closed regions.
    - Each region contains black interior cells and may also contain zero or more single-pixel seeds.
    - All seeds inside the same region share one color chosen from a small foreground palette.

    Rule:
    - Treat every 4-connected non-grey area as one region separated from the others by grey walls.
    - For each seeded region, count its seed pixels N and take the seed color for that region.
    - Color every region cell whose inward layer distance from the region boundary is at most N, where layer 1 is the set of cells directly inside the grey boundary; stop early if those N layers already cover the whole region.
    - Leave unseeded regions black and keep all grey walls unchanged.

    Key insight:
    The seeds are not expanded from their own positions; they only encode how many boundary-shaped inward layers of their enclosing region must be painted.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _region_infos(g):
        seeds = info["seeds"]
        if not seeds:
            continue

        seed_colors = [int(g[r, c]) for r, c in seeds]
        fill_color = int(Counter(seed_colors).most_common(1)[0][0])
        layers = len(seeds)

        for (r, c), dist in info["dist"].items():
            if dist <= layers:
                out[r, c] = fill_color

    return out


def _find(x: int, parent: list[int]) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(a: int, b: int, parent: list[int], size: list[int]) -> bool:
    ra = _find(a, parent)
    rb = _find(b, parent)
    if ra == rb:
        return False
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    return True


def _cell_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def _build_partition(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    target_regions = rng.randint(3, min(7, n - 1))
    parent = list(range(n))
    size = [1] * n

    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            idx = _cell_index(r, c, cols)
            if r + 1 < rows:
                edges.append((idx, _cell_index(r + 1, c, cols)))
            if c + 1 < cols:
                edges.append((idx, _cell_index(r, c + 1, cols)))
    rng.shuffle(edges)

    regions = n
    for a, b in edges:
        if regions <= target_regions:
            break
        if _union(a, b, parent, size):
            regions -= 1

    labels = np.zeros((rows, cols), dtype=int)
    root_to_label: dict[int, int] = {}
    next_label = 0
    for r in range(rows):
        for c in range(cols):
            root = _find(_cell_index(r, c, cols), parent)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[r, c] = root_to_label[root]
    return labels


def _render_partition(labels: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = labels.shape
    height = rows * (cell_h + 1) + 1
    width = cols * (cell_w + 1) + 1
    grid = np.full((height, width), GREY, dtype=int)

    for r in range(rows):
        for c in range(cols):
            top = 1 + r * (cell_h + 1)
            left = 1 + c * (cell_w + 1)
            grid[top : top + cell_h, left : left + cell_w] = BLACK

    # Remove internal walls between coarse cells that belong to the same region.
    for r in range(rows):
        for c in range(cols):
            top = 1 + r * (cell_h + 1)
            left = 1 + c * (cell_w + 1)
            if r + 1 < rows and labels[r, c] == labels[r + 1, c]:
                wall_row = top + cell_h
                grid[wall_row, left : left + cell_w] = BLACK
            if c + 1 < cols and labels[r, c] == labels[r, c + 1]:
                wall_col = left + cell_w
                grid[top : top + cell_h, wall_col] = BLACK

    return grid


def _pick_seed_positions(
    rng: random.Random,
    cells: list[tuple[int, int]],
    dist: dict[tuple[int, int], int],
    count: int,
    min_unfilled_layer: int | None = None,
) -> list[tuple[int, int]]:
    pool = list(cells)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    if min_unfilled_layer is not None:
        deep = [cell for cell in pool if dist[cell] > min_unfilled_layer]
        if deep:
            first = rng.choice(deep)
            chosen.append(first)
            used.add(first)

    for r, c in pool:
        if (r, c) in used:
            continue
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            used.add((r, c))
            if len(chosen) == count:
                return chosen

    for r, c in pool:
        if (r, c) not in used:
            chosen.append((r, c))
            used.add((r, c))
            if len(chosen) == count:
                return chosen

    return chosen


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_rows = rng.randint(2, 6)
    coarse_cols = rng.randint(2, 6)

    max_cell_h = min(4, (29 // coarse_rows) - 1)
    max_cell_w = min(4, (29 // coarse_cols) - 1)
    if max_cell_h < 3 or max_cell_w < 3:
        return None

    cell_h = rng.randint(3, max_cell_h)
    cell_w = rng.randint(3, max_cell_w)
    labels = _build_partition(rng, coarse_rows, coarse_cols)
    grid = _render_partition(labels, cell_h, cell_w)

    infos = _region_infos(grid)
    if len(infos) < 3:
        return None

    partial_candidates = [i for i, info in enumerate(infos) if info["depth"] >= 2 and len(info["cells"]) >= 3]
    if not partial_candidates:
        return None

    partial_idx = rng.choice(partial_candidates)
    other_regions = [i for i in range(len(infos)) if i != partial_idx]
    full_idx = rng.choice(other_regions)

    seeded_modes: dict[int, str] = {partial_idx: "partial", full_idx: "full"}
    for idx, info in enumerate(infos):
        if idx in seeded_modes:
            continue
        if rng.random() < 0.45:
            continue
        if info["depth"] >= 2 and rng.random() < 0.5:
            seeded_modes[idx] = "partial"
        else:
            seeded_modes[idx] = "full"

    if len(seeded_modes) == len(infos):
        removable = [idx for idx in seeded_modes if idx not in (partial_idx, full_idx)]
        if removable:
            del seeded_modes[rng.choice(removable)]

    for idx, mode in seeded_modes.items():
        info = infos[idx]
        area = len(info["cells"])
        depth = info["depth"]

        if mode == "partial":
            max_count = min(area, depth - 1, 4)
            if max_count < 1:
                return None
            count = rng.randint(1, max_count)
            keep_dot = idx == partial_idx
            positions = _pick_seed_positions(
                rng,
                info["cells"],
                info["dist"],
                count,
                min_unfilled_layer=count if keep_dot else None,
            )
        else:
            min_count = min(area, max(1, depth))
            max_count = min(area, depth + 3)
            count = rng.randint(min_count, max_count)
            positions = _pick_seed_positions(rng, info["cells"], info["dist"], count)

        if len(positions) != count:
            return None

        color = rng.choice(SEED_COLORS)
        for r, c in positions:
            grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out == GREY, grid == GREY):
        return None

    partial_info = infos[partial_idx]
    full_info = infos[full_idx]
    partial_seeds = [(r, c) for r, c in partial_info["cells"] if int(grid[r, c]) not in (BLACK, GREY)]
    full_seeds = [(r, c) for r, c in full_info["cells"] if int(grid[r, c]) not in (BLACK, GREY)]
    if not partial_seeds or not full_seeds:
        return None

    partial_color = int(grid[partial_seeds[0]])
    full_color = int(grid[full_seeds[0]])

    partial_filled = sum(1 for cell in partial_info["cells"] if int(out[cell]) == partial_color)
    full_filled = sum(1 for cell in full_info["cells"] if int(out[cell]) == full_color)

    if partial_filled >= len(partial_info["cells"]):
        return None
    if full_filled != len(full_info["cells"]):
        return None
    if not any(
        all(int(out[cell]) == BLACK for cell in infos[idx]["cells"])
        for idx in range(len(infos))
        if idx not in seeded_modes
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-ring ARC input")
