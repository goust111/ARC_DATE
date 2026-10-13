from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
import random


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver helpers ---------------


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield int(r + dr), int(c + dc)


def _infer_colors(g: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, g.flat))
    border = list(map(int, g[0, :])) + list(map(int, g[-1, :]))
    if g.shape[0] > 2:
        border += list(map(int, g[1:-1, 0]))
        border += list(map(int, g[1:-1, -1]))

    border_counts = Counter(border)
    border_color, border_freq = max(border_counts.items(), key=lambda item: item[1])
    if border_freq == len(border):
        wall = int(border_color)
        remaining = [color for color in counts if color != wall]
        assert remaining, "expected a non-wall background color"
        bg = max(remaining, key=lambda color: counts[color])
        return int(bg), int(wall)

    bg = max(counts, key=lambda color: counts[color])
    remaining = [color for color in counts if color != bg]
    assert remaining, "expected a wall color distinct from the background"
    wall = max(remaining, key=lambda color: counts[color])
    return int(bg), int(wall)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                if rr in {0, h - 1} or cc in {0, w - 1}:
                    touches_border = True

                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append({"coords": coords, "touches_border": touches_border})

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each marked room's exclusive gray wall segments with the room's marker color.

    Input structure:
    - A background canvas is partitioned by one wall color into multiple 4-connected enclosed regions.
    - Some regions contain exactly one non-background, non-wall marker pixel, and marked regions use distinct colors.
    - Gray wall pixels can border one marked region, two marked regions, or only unmarked/outside space.

    Rule:
    - Identify the enclosed non-wall regions and determine which ones contain a marker.
    - For each marked region, collect the wall pixels that are 4-neighbors of that region's background cells.
    - Recolor a wall pixel only when it borders exactly one marked region; if it borders two or more marked regions, keep it as the wall color.
    - Leave all walls for unmarked regions unchanged, and erase every marker pixel back to the background color.

    Key insight:
    - The marker does not fill its whole room; it only names that room's private wall segments, while shared walls stay gray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, wall = _infer_colors(g)
    out = g.copy()
    out[(g != bg) & (g != wall)] = bg

    wall_to_regions: dict[tuple[int, int], set[int]] = defaultdict(set)
    region_color: dict[int, int] = {}
    region_id = 0

    for comp in _components(g != wall):
        coords = comp["coords"]
        colors = {int(g[r, c]) for r, c in coords}
        marker_colors = colors - {bg}
        assert len(marker_colors) <= 1, "each enclosed region must contain at most one marker color"

        if comp["touches_border"] or not marker_colors:
            continue

        marker_color = int(next(iter(marker_colors)))
        region_color[region_id] = marker_color

        for r, c in coords:
            if int(g[r, c]) != bg:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == wall:
                    wall_to_regions[(nr, nc)].add(region_id)

        region_id += 1

    for (r, c), touching_regions in wall_to_regions.items():
        if len(touching_regions) == 1:
            out[r, c] = region_color[next(iter(touching_regions))]

    return out


# --------------- generator helpers ---------------

COLOR_POOL = tuple(range(10))
COARSE_ROWS = (3, 4)
COARSE_COLS = (3, 4)
CELL_SIZE_RANGE = (4, 6)
REGION_COUNT_MIN = 3
REGION_COUNT_MAX = 7
EXTRA_MARKED_MAX = 2
GENERATION_ATTEMPTS = 400


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _cell_offsets(sizes: list[int]) -> list[int]:
    offsets: list[int] = []
    cursor = 1
    for size in sizes:
        offsets.append(cursor)
        cursor += size + 1
    return offsets


def _make_partition() -> tuple[np.ndarray, np.ndarray, dict[int, set[int]], set[int]] | None:
    coarse_h = random.randint(*COARSE_ROWS)
    coarse_w = random.randint(*COARSE_COLS)
    row_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(coarse_h)]
    col_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(coarse_w)]
    grid_h = sum(row_sizes) + coarse_h + 1
    grid_w = sum(col_sizes) + coarse_w + 1
    if max(grid_h, grid_w) > 30:
        return None

    cell_count = coarse_h * coarse_w
    target_regions = random.randint(
        REGION_COUNT_MIN,
        min(REGION_COUNT_MAX, cell_count - 1),
    )

    dsu = _DSU(cell_count)
    edges: list[tuple[int, int]] = []
    for r in range(coarse_h):
        for c in range(coarse_w):
            idx = r * coarse_w + c
            if r + 1 < coarse_h:
                edges.append((idx, (r + 1) * coarse_w + c))
            if c + 1 < coarse_w:
                edges.append((idx, r * coarse_w + c + 1))

    random.shuffle(edges)
    components = cell_count
    for a, b in edges:
        if components <= target_regions:
            break
        if dsu.union(a, b):
            components -= 1

    roots = sorted({dsu.find(i) for i in range(cell_count)})
    root_to_region = {root: rid for rid, root in enumerate(roots)}
    coarse_labels = np.array(
        [[root_to_region[dsu.find(r * coarse_w + c)] for c in range(coarse_w)] for r in range(coarse_h)],
        dtype=int,
    )

    region_map = np.full((grid_h, grid_w), -1, dtype=int)
    row_offsets = _cell_offsets(row_sizes)
    col_offsets = _cell_offsets(col_sizes)

    for r in range(coarse_h):
        for c in range(coarse_w):
            region_map[
                row_offsets[r] : row_offsets[r] + row_sizes[r],
                col_offsets[c] : col_offsets[c] + col_sizes[c],
            ] = int(coarse_labels[r, c])

    for r in range(coarse_h):
        for c in range(coarse_w - 1):
            if int(coarse_labels[r, c]) == int(coarse_labels[r, c + 1]):
                wall_c = col_offsets[c] + col_sizes[c]
                region_map[row_offsets[r] : row_offsets[r] + row_sizes[r], wall_c] = int(coarse_labels[r, c])

    for r in range(coarse_h - 1):
        for c in range(coarse_w):
            if int(coarse_labels[r, c]) == int(coarse_labels[r + 1, c]):
                wall_r = row_offsets[r] + row_sizes[r]
                region_map[wall_r, col_offsets[c] : col_offsets[c] + col_sizes[c]] = int(coarse_labels[r, c])

    for r in range(coarse_h - 1):
        for c in range(coarse_w - 1):
            block = {
                int(coarse_labels[r, c]),
                int(coarse_labels[r, c + 1]),
                int(coarse_labels[r + 1, c]),
                int(coarse_labels[r + 1, c + 1]),
            }
            if len(block) == 1:
                region_map[row_offsets[r] + row_sizes[r], col_offsets[c] + col_sizes[c]] = next(iter(block))

    adjacency: dict[int, set[int]] = defaultdict(set)
    frame_regions: set[int] = set()
    for r in range(coarse_h):
        for c in range(coarse_w):
            rid = int(coarse_labels[r, c])
            if r in {0, coarse_h - 1} or c in {0, coarse_w - 1}:
                frame_regions.add(rid)
            if r + 1 < coarse_h and int(coarse_labels[r, c]) != int(coarse_labels[r + 1, c]):
                a = int(coarse_labels[r, c])
                b = int(coarse_labels[r + 1, c])
                adjacency[a].add(b)
                adjacency[b].add(a)
            if c + 1 < coarse_w and int(coarse_labels[r, c]) != int(coarse_labels[r, c + 1]):
                a = int(coarse_labels[r, c])
                b = int(coarse_labels[r, c + 1])
                adjacency[a].add(b)
                adjacency[b].add(a)

    return region_map, coarse_labels, adjacency, frame_regions


def _marker_candidates(region_map: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    h, w = region_map.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            rid = int(region_map[r, c])
            if rid < 0:
                continue
            if all(int(region_map[r + dr, c + dc]) == rid for dr, dc in DIRS):
                candidates[rid].append((int(r), int(c)))
    return candidates


def _choose_marked_regions(
    all_regions: list[int],
    adjacency: dict[int, set[int]],
    frame_regions: set[int],
    candidates: dict[int, list[tuple[int, int]]],
) -> list[int] | None:
    pair_options = [
        (a, b)
        for a in frame_regions
        for b in adjacency[a]
        if candidates[a] and candidates[b]
    ]
    if not pair_options or len(all_regions) < 3:
        return None

    a, b = random.choice(pair_options)
    marked = [int(a), int(b)]

    extras = [rid for rid in all_regions if rid not in {a, b} and candidates[rid]]
    random.shuffle(extras)
    extra_max = min(EXTRA_MARKED_MAX, len(extras), len(all_regions) - 3)
    if extra_max > 0:
        extra_count = random.randint(0, extra_max)
        marked.extend(extras[:extra_count])

    if len(set(marked)) < 2 or len(set(marked)) >= len(all_regions):
        return None
    return marked


def _expected_output(
    grid: np.ndarray,
    region_map: np.ndarray,
    region_markers: dict[int, int],
    bg: int,
    wall: int,
) -> tuple[np.ndarray, dict[tuple[int, int], set[int]]]:
    out = grid.copy()
    out[(grid != bg) & (grid != wall)] = bg

    wall_to_regions: dict[tuple[int, int], set[int]] = defaultdict(set)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            rid = int(region_map[r, c])
            if rid < 0 or rid not in region_markers or int(grid[r, c]) != bg:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(region_map[nr, nc]) < 0:
                    wall_to_regions[(nr, nc)].add(rid)

    for (r, c), touching_regions in wall_to_regions.items():
        if len(touching_regions) == 1:
            out[r, c] = region_markers[next(iter(touching_regions))]

    return out, wall_to_regions


def _generate_case() -> np.ndarray | None:
    partition = _make_partition()
    if partition is None:
        return None
    region_map, coarse_labels, adjacency, frame_regions = partition

    all_regions = sorted(set(map(int, coarse_labels.flat)))
    candidates = _marker_candidates(region_map)
    marked_regions = _choose_marked_regions(all_regions, adjacency, frame_regions, candidates)
    if marked_regions is None:
        return None

    bg, wall = random.sample(COLOR_POOL, 2)
    marker_palette = [color for color in COLOR_POOL if color not in {bg, wall}]
    random.shuffle(marker_palette)
    region_markers = {rid: int(marker_palette[i]) for i, rid in enumerate(marked_regions)}

    grid = np.full(region_map.shape, wall, dtype=int)
    grid[region_map >= 0] = bg

    used_positions: set[tuple[int, int]] = set()
    for rid in marked_regions:
        choices = [pos for pos in candidates[rid] if pos not in used_positions]
        if not choices:
            return None
        pos = random.choice(choices)
        used_positions.add(pos)
        grid[pos] = region_markers[rid]

    counts = Counter(map(int, grid.flat))
    if counts[bg] <= counts[wall]:
        return None

    expected, wall_to_regions = _expected_output(grid, region_map, region_markers, bg, wall)
    exclusive_walls = sum(len(regions) == 1 for regions in wall_to_regions.values())
    shared_walls = sum(len(regions) >= 2 for regions in wall_to_regions.values())
    if exclusive_walls == 0 or shared_walls == 0:
        return None

    assert np.array_equal(main(grid), expected), "generator produced a sample inconsistent with main()"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-partition sample")
