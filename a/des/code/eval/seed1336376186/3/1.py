from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

PRIORITY_COLORS = [RED, BLUE, GREEN, YELLOW, MAGENTA, ORANGE]
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_COLORS)}

MIN_REGION_SIDE = 3
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 31)
TARGET_REGION_RANGE = (6, 10)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    if BLACK in counts:
        return BLACK
    return counts.most_common(1)[0][0]


def _wall_color(grid: np.ndarray, bg: int) -> int:
    values = [int(v) for v in np.unique(grid) if int(v) != bg]
    if GREY in values:
        return GREY
    assert values, "expected a wall color"
    return max(values, key=lambda color: int(np.count_nonzero(grid == color)))


def _label_enclosed_regions(grid: np.ndarray, wall: int):
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    passable = grid != wall
    regions: dict[int, list[tuple[int, int]]] = {}
    next_label = 0

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] != 0:
                continue
            queue = deque([(r, c)])
            labels[r, c] = -1
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if passable[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = -1
                        queue.append((nr, nc))
            if touches_edge:
                for rr, cc in cells:
                    labels[rr, cc] = 0
                continue
            next_label += 1
            for rr, cc in cells:
                labels[rr, cc] = next_label
            regions[next_label] = cells

    return labels, regions


def _build_region_graph(labels: np.ndarray, wall_mask: np.ndarray):
    h, w = labels.shape
    adjacency = {label: set() for label in range(1, int(labels.max()) + 1)}
    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)
        if 0 < c < w - 1:
            left = int(labels[r, c - 1])
            right = int(labels[r, c + 1])
            if left > 0 and right > 0 and left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
        if 0 < r < h - 1:
            up = int(labels[r - 1, c])
            down = int(labels[r + 1, c])
            if up > 0 and down > 0 and up != down:
                adjacency[up].add(down)
                adjacency[down].add(up)
    return adjacency


def _shortest_path(adjacency: dict[int, set[int]], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
    queue = deque([start])
    parent = {start: -1}
    while queue:
        node = queue.popleft()
        for nxt in sorted(adjacency[node]):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == end:
                queue.clear()
                break
            queue.append(nxt)
    assert end in parent, "anchor regions must be connected through the map"
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _anchor_groups(grid: np.ndarray, bg: int, wall: int):
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere((grid != bg) & (grid != wall)):
        groups[int(grid[r, c])].append((int(r), int(c)))
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the region graph induced by grey walls, then fill intermediate regions
    on the shortest region-chain between each same-colored anchor pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    wall = _wall_color(g, bg)

    labels, regions = _label_enclosed_regions(g, wall)
    assert regions, "expected enclosed regions"
    adjacency = _build_region_graph(labels, g == wall)

    anchors = _anchor_groups(g, bg, wall)
    fill_requests: dict[int, list[int]] = defaultdict(list)

    for color, pts in anchors.items():
        assert len(pts) == 2, "each anchor color must appear exactly twice"
        start = int(labels[pts[0]])
        end = int(labels[pts[1]])
        assert start > 0 and end > 0 and start != end, "paired anchors must occupy different enclosed regions"
        path = _shortest_path(adjacency, start, end)
        for region_id in path[1:-1]:
            fill_requests[region_id].append(color)

    out = g.copy()
    for region_id, colors in fill_requests.items():
        chosen = min(colors, key=lambda color: (PRIORITY_RANK.get(color, len(PRIORITY_RANK)), color))
        for r, c in regions[region_id]:
            if int(g[r, c]) == bg:
                out[r, c] = chosen

    return out


def _generate_partition(rng: random.Random) -> np.ndarray | None:
    def random_parts(total: int, count: int, minimum: int) -> list[int]:
        extra = total - count * minimum
        if extra < 0:
            raise ValueError("total too small")
        cuts = sorted(rng.sample(range(extra + count - 1), count - 1)) if count > 1 else []
        parts = []
        prev = -1
        for cut in cuts + [extra + count - 1]:
            parts.append(cut - prev)
            prev = cut
        return [part + minimum - 1 for part in parts]

    count = rng.randrange(*TARGET_REGION_RANGE)
    orientation = rng.choice(["v", "h"])

    if orientation == "v":
        h = rng.randrange(12, 23)
        min_region_sum = count * MIN_REGION_SIDE
        max_region_sum = 28 - (count - 1)
        if min_region_sum > max_region_sum:
            return None
        region_sum = rng.randint(min_region_sum, max_region_sum)
        widths = random_parts(region_sum, count, MIN_REGION_SIDE)
        w = sum(widths) + (count - 1) + 2
        grid = np.full((h, w), BLACK, dtype=int)
        grid[0, :] = GREY
        grid[-1, :] = GREY
        grid[:, 0] = GREY
        grid[:, -1] = GREY
        cursor = 1
        for idx, width in enumerate(widths):
            cursor += width
            if idx != count - 1:
                grid[:, cursor] = GREY
                cursor += 1
        return grid

    w = rng.randrange(12, 23)
    min_region_sum = count * MIN_REGION_SIDE
    max_region_sum = 26 - (count - 1)
    if min_region_sum > max_region_sum:
        return None
    region_sum = rng.randint(min_region_sum, max_region_sum)
    heights = random_parts(region_sum, count, MIN_REGION_SIDE)
    h = sum(heights) + (count - 1) + 2
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    cursor = 1
    for idx, height in enumerate(heights):
        cursor += height
        if idx != count - 1:
            grid[cursor, :] = GREY
            cursor += 1
    return grid


def _graph_is_tree(adjacency: dict[int, set[int]]) -> bool:
    if not adjacency:
        return False
    root = next(iter(adjacency))
    seen = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return len(seen) == len(adjacency) and edge_count == len(adjacency) - 1


def _pair_record(adjacency: dict[int, set[int]], a: int, b: int):
    path = _shortest_path(adjacency, a, b)
    inter = set(path[1:-1])
    return {
        "endpoints": (a, b),
        "path": path,
        "intermediate": inter,
        "distance": len(path) - 1,
    }


def _combo_valid(combo: tuple[dict, ...], require_overlap: bool) -> bool:
    endpoints: set[int] = set()
    for record in combo:
        a, b = record["endpoints"]
        if a in endpoints or b in endpoints:
            return False
        endpoints.add(a)
        endpoints.add(b)
        if not record["intermediate"]:
            return False

    shared = False
    for left, right in combinations(combo, 2):
        overlap = left["intermediate"] & right["intermediate"]
        if overlap:
            shared = True
        elif not require_overlap:
            continue
        if not require_overlap and overlap:
            return False
    return shared if require_overlap else True


def _select_pairs(adjacency: dict[int, set[int]], require_overlap: bool, rng: random.Random):
    labels = sorted(adjacency)
    candidates = []
    for a, b in combinations(labels, 2):
        record = _pair_record(adjacency, a, b)
        if record["distance"] >= 2:
            candidates.append(record)
    if len(candidates) < 2:
        return None

    desired_sizes = [rng.choice([2, 3]), 2, 3]
    seen_sizes = []
    for size in desired_sizes:
        if size not in seen_sizes:
            seen_sizes.append(size)
    for size in seen_sizes:
        combos = list(combinations(candidates, size))
        rng.shuffle(combos)
        for combo in combos:
            if _combo_valid(combo, require_overlap):
                return list(combo)
    return None


def _anchor_candidates(region_id: int, labels: np.ndarray, region_cells: dict[int, list[tuple[int, int]]]):
    h, w = labels.shape
    inner = []
    for r, c in region_cells[region_id]:
        if all(labels[nr, nc] == region_id for nr, nc in _neighbors4(r, c, h, w)):
            inner.append((r, c))
    return inner or region_cells[region_id]


def _place_anchors(
    grid: np.ndarray,
    labels: np.ndarray,
    regions: dict[int, list[tuple[int, int]]],
    pair_records: list[dict],
    rng: random.Random,
) -> dict[int, list[int]]:
    by_region: dict[int, list[int]] = defaultdict(list)
    colors = rng.sample(PRIORITY_COLORS, len(pair_records))
    rng.shuffle(pair_records)
    for color, record in zip(colors, pair_records):
        for region_id in record["endpoints"]:
            candidates = _anchor_candidates(region_id, labels, regions)
            r, c = rng.choice(candidates)
            grid[r, c] = color
        for region_id in record["intermediate"]:
            by_region[region_id].append(color)
    return by_region


def _valid_generated_example(
    inp: np.ndarray,
    expected_overlap: bool,
    expected_region_colors: dict[int, list[int]],
) -> bool:
    out = main(inp)
    if np.array_equal(inp, out):
        return False

    bg = _background_color(inp)
    wall = _wall_color(inp, bg)
    labels, regions = _label_enclosed_regions(inp, wall)
    overlap_regions = [rid for rid, colors in expected_region_colors.items() if len(colors) >= 2]
    if expected_overlap and not overlap_regions:
        return False
    if not expected_overlap and overlap_regions:
        return False

    for rid, colors in expected_region_colors.items():
        cells = [(r, c) for r, c in regions[rid] if int(inp[r, c]) == bg]
        if not cells:
            return False
        chosen = min(colors, key=lambda color: (PRIORITY_RANK[color], color))
        if any(int(out[r, c]) != chosen for r, c in cells):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    require_overlap = rng.random() < 0.5
    base = _generate_partition(rng)
    if base is None:
        return None

    labels, regions = _label_enclosed_regions(base, GREY)
    adjacency = _build_region_graph(labels, base == GREY)
    if not _graph_is_tree(adjacency):
        return None

    pair_records = _select_pairs(adjacency, require_overlap, rng)
    if pair_records is None:
        return None

    grid = base.copy()
    expected_region_colors = _place_anchors(grid, labels, regions, pair_records, rng)
    if not _valid_generated_example(grid, require_overlap, expected_region_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
