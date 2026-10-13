from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MARKER_COLOR = 5
BOUNDARY_COLOR = 0

GEN_ATTEMPTS = 300
CORRUPT_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _region_sort_key(region: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = region["bbox"]
    return (int(r0), int(c0), int(r1 - r0), int(c1 - c0))


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != BOUNDARY_COLOR
    seen = np.zeros((h, w), dtype=bool)
    region_index = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(g.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            colors = [int(g[rr, cc]) for rr, cc in coords if int(g[rr, cc]) not in (BOUNDARY_COLOR, MARKER_COLOR)]
            assert colors, "each region must contain a visible non-marker color"
            color_counts = Counter(colors)
            dominant_count = max(color_counts.values())
            dominant_colors = [color for color, count in color_counts.items() if count == dominant_count]
            assert len(dominant_colors) == 1, "each region must have a unique visible fill color"
            fill_color = int(dominant_colors[0])
            markers = [(int(rr), int(cc)) for rr, cc in coords if int(g[rr, cc]) == MARKER_COLOR]

            rid = len(regions)
            for rr, cc in coords:
                region_index[rr, cc] = rid
            regions.append(
                {
                    "coords": coord_arr,
                    "bbox": _bbox(coord_arr),
                    "color": fill_color,
                    "markers": markers,
                    "marked": bool(markers),
                }
            )

    assert regions, "expected at least one region"
    return regions, region_index


def _build_adjacency(grid: np.ndarray, region_index: np.ndarray, n_regions: int) -> list[set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    adjacency = [set() for _ in range(n_regions)]

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BOUNDARY_COLOR:
                continue
            if 0 < r < h - 1:
                a = int(region_index[r - 1, c])
                b = int(region_index[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            if 0 < c < w - 1:
                a = int(region_index[r, c - 1])
                b = int(region_index[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return adjacency


def _color_order(grid: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in np.asarray(grid, dtype=int).ravel():
        color = int(value)
        if color in (BOUNDARY_COLOR, MARKER_COLOR) or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _analyze(grid: np.ndarray) -> tuple[list[dict], list[set[int]], list[int]]:
    regions, region_index = _extract_regions(grid)
    adjacency = _build_adjacency(grid, region_index, len(regions))
    palette = _color_order(grid)
    assert 2 <= len(palette) <= 4, "expected a 2-4 color country palette"
    return regions, adjacency, palette


def _solve_marked_regions(regions: list[dict], adjacency: list[set[int]], palette: list[int]) -> dict[int, int]:
    marked = [rid for rid, region in enumerate(regions) if region["marked"]]
    if not marked:
        return {}

    palette_pos = {color: idx for idx, color in enumerate(palette)}
    counts = Counter(int(region["color"]) for region in regions if not region["marked"])
    canonical_marked = sorted(marked, key=lambda rid: _region_sort_key(regions[rid]))

    def search_order_key(rid: int) -> tuple[int, int, tuple[int, int, int, int]]:
        marked_neighbors = sum(1 for other in adjacency[rid] if regions[other]["marked"])
        return (-marked_neighbors, -len(adjacency[rid]), _region_sort_key(regions[rid]))

    search_order = sorted(marked, key=search_order_key)
    assigned: dict[int, int] = {}
    best_assignment: dict[int, int] | None = None
    best_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def legal_colors(rid: int) -> list[int]:
        blocked = {int(regions[rid]["color"])}
        for nb in adjacency[rid]:
            if regions[nb]["marked"]:
                if nb in assigned:
                    blocked.add(int(assigned[nb]))
            else:
                blocked.add(int(regions[nb]["color"]))
        options = [color for color in palette if color not in blocked]
        return sorted(options, key=lambda color: (counts[color] + 1, palette_pos[color]))

    def score_assignment() -> tuple[tuple[int, ...], tuple[int, ...]]:
        count_profile = tuple(sorted((counts[color] for color in palette), reverse=True))
        assignment_profile = tuple(palette_pos[assigned[rid]] for rid in canonical_marked)
        return count_profile, assignment_profile

    def dfs(index: int) -> None:
        nonlocal best_assignment, best_key
        if index == len(search_order):
            key = score_assignment()
            if best_key is None or key < best_key:
                best_key = key
                best_assignment = assigned.copy()
            return

        rid = search_order[index]
        options = legal_colors(rid)
        assert options, "every marked region must have at least one legal recolor"
        for color in options:
            assigned[rid] = int(color)
            counts[color] += 1
            dfs(index + 1)
            counts[color] -= 1
            del assigned[rid]

    dfs(0)
    assert best_assignment is not None, "failed to find a valid recoloring"
    return best_assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray-marked conflicting countries so adjacent regions end up with a balanced valid map coloring.

    Input structure:
    - Black 0-cells form one-cell-thick boundary lines that split the grid into
      irregular non-zero connected regions.
    - Each region is mostly filled with one visible color from a shared 2-4 color
      palette, and some regions contain a gray 5 marker pixel.
    - A marked region is one of the countries whose current color conflicts with
      at least one adjacent country across a black boundary segment.

    Rule:
    - Treat every non-zero 4-connected component as one country, even if it
      contains a gray marker pixel.
    - Build the country adjacency graph by checking which countries sit on the
      two sides of each black boundary segment.
    - Recolor every marked country with a palette color that differs from all of
      its adjacent countries and from its original visible color.
    - If several joint recolorings are legal, choose the one whose final region
      color counts are the most globally balanced, then remove the gray marker.

    Key insight:
    The puzzle is a small graph-coloring repair task: the black lines define the
    planar region graph, and the gray marker limits the recoloring search to only
    the countries that must be changed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions, adjacency, palette = _analyze(g)
    assignment = _solve_marked_regions(regions, adjacency, palette)

    out = g.copy()
    for rid, new_color in assignment.items():
        coords = regions[rid]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)

    assert not np.any(out == MARKER_COLOR), "markers must be removed in the output"
    solved_regions, solved_adjacency, _ = _analyze(out)
    for rid, neighbors in enumerate(solved_adjacency):
        for nb in neighbors:
            assert solved_regions[rid]["color"] != solved_regions[nb]["color"], "adjacent countries must differ"
    return out


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

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
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _tile_id(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def _sample_partition(rng: random.Random) -> tuple[np.ndarray, list[set[int]], dict[int, list[tuple[int, int]]]]:
    tile_rows = rng.randint(3, 5)
    tile_cols = rng.randint(3, 5)
    heights = [rng.randint(2, 4) for _ in range(tile_rows)]
    widths = [rng.randint(2, 4) for _ in range(tile_cols)]

    height = sum(heights) + tile_rows + 1
    width = sum(widths) + tile_cols + 1
    assert height <= 30 and width <= 30

    total_tiles = tile_rows * tile_cols
    edges: list[tuple[int, int]] = []
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            tid = _tile_id(tr, tc, tile_cols)
            if tr + 1 < tile_rows:
                edges.append((tid, _tile_id(tr + 1, tc, tile_cols)))
            if tc + 1 < tile_cols:
                edges.append((tid, _tile_id(tr, tc + 1, tile_cols)))

    dsu = _DSU(total_tiles)
    lower_regions = max(5, total_tiles // 3)
    upper_regions = min(total_tiles - 1, 11)
    if lower_regions > upper_regions:
        lower_regions = upper_regions
    target_regions = rng.randint(lower_regions, upper_regions)
    rng.shuffle(edges)
    merges_needed = total_tiles - target_regions
    for a, b in edges:
        if merges_needed == 0:
            break
        if dsu.union(a, b):
            merges_needed -= 1
    if merges_needed != 0:
        raise AssertionError("failed to create enough merged regions")

    root_to_region: dict[int, int] = {}
    tile_region = np.full((tile_rows, tile_cols), -1, dtype=int)
    region_tiles: dict[int, list[tuple[int, int]]] = {}
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            root = dsu.find(_tile_id(tr, tc, tile_cols))
            if root not in root_to_region:
                root_to_region[root] = len(root_to_region)
                region_tiles[root_to_region[root]] = []
            rid = root_to_region[root]
            tile_region[tr, tc] = rid
            region_tiles[rid].append((tr, tc))

    n_regions = len(region_tiles)
    adjacency = [set() for _ in range(n_regions)]
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            rid = int(tile_region[tr, tc])
            if tr + 1 < tile_rows:
                nb = int(tile_region[tr + 1, tc])
                if rid != nb:
                    adjacency[rid].add(nb)
                    adjacency[nb].add(rid)
            if tc + 1 < tile_cols:
                nb = int(tile_region[tr, tc + 1])
                if rid != nb:
                    adjacency[rid].add(nb)
                    adjacency[nb].add(rid)

    row_starts: list[int] = []
    row_ends: list[int] = []
    cursor = 1
    for h in heights:
        row_starts.append(cursor)
        row_ends.append(cursor + h - 1)
        cursor += h + 1

    col_starts: list[int] = []
    col_ends: list[int] = []
    cursor = 1
    for w in widths:
        col_starts.append(cursor)
        col_ends.append(cursor + w - 1)
        cursor += w + 1

    grid = np.zeros((height, width), dtype=int)
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            rid = int(tile_region[tr, tc])
            rs, re = row_starts[tr], row_ends[tr]
            cs, ce = col_starts[tc], col_ends[tc]
            grid[rs : re + 1, cs : ce + 1] = rid + 1
            if tr + 1 < tile_rows and int(tile_region[tr + 1, tc]) == rid:
                wall_r = re + 1
                grid[wall_r, cs : ce + 1] = rid + 1
            if tc + 1 < tile_cols and int(tile_region[tr, tc + 1]) == rid:
                wall_c = ce + 1
                grid[rs : re + 1, wall_c] = rid + 1

    return grid, adjacency, region_tiles


def _color_graph(adjacency: list[set[int]], palette: list[int], rng: random.Random) -> dict[int, int] | None:
    n_regions = len(adjacency)
    order = sorted(range(n_regions), key=lambda rid: -len(adjacency[rid]))
    assignment: dict[int, int] = {}

    def dfs(index: int) -> bool:
        if index == len(order):
            return True
        rid = order[index]
        colors = palette[:]
        rng.shuffle(colors)
        colors.sort(key=lambda color: sum(1 for nb in adjacency[rid] if assignment.get(nb) == color))
        for color in colors:
            if any(assignment.get(nb) == color for nb in adjacency[rid]):
                continue
            assignment[rid] = color
            if dfs(index + 1):
                return True
            del assignment[rid]
        return False

    if not dfs(0):
        return None
    if len(set(assignment.values())) != len(palette):
        return None
    return assignment


def _render_colored_map(region_template: np.ndarray, region_colors: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(region_template)
    for value in np.unique(region_template):
        key = int(value)
        if key == 0:
            continue
        out[region_template == key] = int(region_colors[key - 1])
    return out


def _pick_marker_cell(region_coords: np.ndarray, region_set: set[Coord], rng: random.Random) -> Coord:
    scored: list[tuple[int, int, int]] = []
    for r_raw, c_raw in region_coords:
        r = int(r_raw)
        c = int(c_raw)
        same_neighbors = sum((nr, nc) in region_set for nr, nc in _neighbors4(r, c))
        scored.append((same_neighbors, r, c))
    best_score = max(score for score, _r, _c in scored)
    candidates = [(r, c) for score, r, c in scored if score == best_score]
    return rng.choice(candidates)


def _make_corrupted_input(solution: np.ndarray, rng: random.Random) -> np.ndarray | None:
    regions, adjacency, palette = _analyze(solution)
    if len(regions) < 5:
        return None

    counts = Counter(int(region["color"]) for region in regions)
    eligible = [rid for rid, region in enumerate(regions) if len(region["coords"]) >= 4 and counts[region["color"]] >= 2]
    if len(eligible) < 1:
        return None

    for _ in range(CORRUPT_ATTEMPTS):
        marked_count = rng.randint(1, min(3, len(eligible)))
        shuffled = eligible[:]
        rng.shuffle(shuffled)
        selected: list[int] = []
        for rid in shuffled:
            if any(rid in adjacency[other] for other in selected):
                continue
            selected.append(rid)
            if len(selected) == marked_count:
                break
        if len(selected) != marked_count:
            continue

        baseline = counts.copy()
        for rid in selected:
            baseline[regions[rid]["color"]] -= 1
        if any(baseline[color] <= 0 for color in palette):
            continue

        wrong_colors: dict[int, int] = {}
        for rid in selected:
            neighbor_colors = list({regions[nb]["color"] for nb in adjacency[rid] if regions[nb]["color"] != regions[rid]["color"]})
            if not neighbor_colors:
                break
            neighbor_colors.sort(key=lambda color: (-baseline[color], palette.index(color)))
            wrong_colors[rid] = int(neighbor_colors[0])
        if len(wrong_colors) != len(selected):
            continue

        inp = solution.copy()
        for rid in selected:
            coords = regions[rid]["coords"]
            inp[coords[:, 0], coords[:, 1]] = int(wrong_colors[rid])
            region_set = {tuple(map(int, coord)) for coord in coords}
            mr, mc = _pick_marker_cell(coords, region_set, rng)
            inp[mr, mc] = MARKER_COLOR

        try:
            out = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(out, solution):
            continue
        return inp

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    try:
        region_template, adjacency, _region_tiles = _sample_partition(rng)
    except AssertionError:
        return None

    palette_size = rng.choice((2, 3, 4))
    palette = list(rng.sample(PALETTE_POOL, palette_size))
    region_colors = _color_graph(adjacency, palette, rng)
    if region_colors is None:
        return None

    solution = _render_colored_map(region_template, region_colors)
    return _make_corrupted_input(solution, rng)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid country-recoloring input")
