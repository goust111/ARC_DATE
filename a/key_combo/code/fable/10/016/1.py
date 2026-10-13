from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 2
SEPARATOR = 5
PALETTE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            labels[r, c] = current
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        q.append((nr, nc))
    return labels, current


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    separator_rows = [
        r for r in range(1, g.shape[0]) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert separator_rows, "expected a full-width separator row"
    separator_r = int(separator_rows[0])
    separator_color = int(g[separator_r, 0])

    code_row = g[separator_r - 1]
    non_bg = np.flatnonzero(code_row != bg)
    assert len(non_bg) > 0 and int(non_bg[0]) == 0, "palette must start at the left edge"
    last = int(non_bg[-1])
    assert np.all(code_row[: last + 1] != bg), "palette must be contiguous"
    code = code_row[: last + 1].astype(int)

    lower = g[separator_r + 1 :].copy()
    assert lower.size > 0
    seed_color = int(code[0])
    seed_cells = np.argwhere(lower == seed_color)
    assert len(seed_cells) == 1, "expected exactly one seed pixel below the separator"

    present = {int(x) for x in np.unique(lower)}
    assert seed_color != bg and seed_color != separator_color
    wall_candidates = sorted(present - {bg, seed_color})
    assert len(wall_candidates) == 1, "expected exactly one wall color below the separator"
    wall_color = int(wall_candidates[0])

    traversable = lower != wall_color
    labels, region_count = _label_components(traversable)
    assert region_count >= 2, "walls must split the canvas into multiple regions"

    seed_r, seed_c = map(int, seed_cells[0])
    seed_region = int(labels[seed_r, seed_c])
    assert seed_region > 0

    graph = {region_id: set() for region_id in range(1, region_count + 1)}
    wall_pixels = np.argwhere(lower == wall_color)
    assert len(wall_pixels) > 0
    for r0, c0 in wall_pixels:
        r = int(r0)
        c = int(c0)
        if 0 < c < lower.shape[1] - 1:
            left = int(labels[r, c - 1])
            right = int(labels[r, c + 1])
            if left > 0 and right > 0 and left != right:
                graph[left].add(right)
                graph[right].add(left)
        if 0 < r < lower.shape[0] - 1:
            up = int(labels[r - 1, c])
            down = int(labels[r + 1, c])
            if up > 0 and down > 0 and up != down:
                graph[up].add(down)
                graph[down].add(up)

    depths = {seed_region: 0}
    q = deque([seed_region])
    while q:
        cur = q.popleft()
        for nxt in sorted(graph[cur]):
            if nxt not in depths:
                depths[nxt] = depths[cur] + 1
                q.append(nxt)

    assert len(depths) == region_count, "all regions must be wall-reachable from the seed region"
    max_depth = max(depths.values())
    assert max_depth < len(code), "palette must cover every wall-crossing depth"

    return {
        "grid": g,
        "bg": bg,
        "separator_r": separator_r,
        "separator_color": separator_color,
        "code": code,
        "lower": lower,
        "wall_color": wall_color,
        "labels": labels,
        "graph": graph,
        "depths": depths,
        "seed": (seed_r, seed_c),
        "seed_region": seed_region,
        "max_depth": max_depth,
    }


def _shortest_path_counts(graph: dict[int, set[int]], start: int) -> tuple[dict[int, int], dict[int, int]]:
    dist = {start: 0}
    count = {start: 1}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in sorted(graph[cur]):
            cand = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = cand
                count[nxt] = count[cur]
                q.append(nxt)
            elif dist[nxt] == cand:
                count[nxt] += count[cur]
    return dist, count


def _min_pixel_distances(labels: np.ndarray, seed: tuple[int, int]) -> dict[int, int]:
    sr, sc = seed
    out: dict[int, int] = {}
    for region_id in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        coords = np.argwhere(labels == region_id)
        out[region_id] = int(np.min(np.abs(coords[:, 0] - sr) + np.abs(coords[:, 1] - sc)))
    return out


def _has_geometry_counterexample(depths: dict[int, int], min_dists: dict[int, int]) -> bool:
    region_ids = sorted(depths)
    for a in region_ids:
        for b in region_ids:
            if depths[a] < depths[b] and min_dists[a] > min_dists[b]:
                return True
    return False


def _choose_seed_coord(lo: int, hi: int) -> int:
    values = list(range(lo, hi + 1))
    edge_weights = [3 if v in (lo, hi) else 1 for v in values]
    return int(random.choices(values, weights=edge_weights, k=1)[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every wall-separated background region by indexing the top palette with its minimum wall-crossing count from the seed region.

    Input structure:
    - The top row is a left-aligned palette of non-background colors, followed by background.
    - A full-width separator row sits directly below that palette.
    - The lower canvas is a background field cut by one-cell-thick red wall segments, and it contains exactly one seed pixel whose color matches the first palette entry.

    Rule:
    - Keep the palette row, separator row, and every wall pixel unchanged.
    - Treat each 4-connected non-wall canvas region as a node, and connect two nodes when a wall pixel has those two regions on opposite sides.
    - Let the seed's region have depth 0; any region at minimum wall-crossing depth d is recolored uniformly with palette[d], and the seed pixel becomes part of that region color.

    Key insight:
    - The palette is indexed by BFS depth in the region-adjacency graph, not by geometric distance and not by a two-color alternation across walls.
    """
    info = _parse_input(grid)
    out = info["grid"].copy()
    lower_out = out[info["separator_r"] + 1 :]
    labels = info["labels"]
    for region_id, depth in info["depths"].items():
        lower_out[labels == region_id] = int(info["code"][depth])
    return out


def _build_lower_canvas(
    row_sizes: list[int], col_sizes: list[int]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    lower = np.full((h, w), BG, dtype=int)

    row_boxes: list[tuple[int, int]] = []
    cur = 0
    for idx, size in enumerate(row_sizes):
        row_boxes.append((cur, cur + size - 1))
        cur += size
        if idx != len(row_sizes) - 1:
            lower[cur, :] = WALL
            cur += 1

    col_boxes: list[tuple[int, int]] = []
    cur = 0
    for idx, size in enumerate(col_sizes):
        col_boxes.append((cur, cur + size - 1))
        cur += size
        if idx != len(col_sizes) - 1:
            lower[:, cur] = WALL
            cur += 1

    cells = []
    for r0, r1 in row_boxes:
        for c0, c1 in col_boxes:
            cells.append((r0, c0, r1, c1))
    return lower, cells


def _make_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)
    if row_count == 2 and col_count == 2:
        return None

    row_sizes = [random.randint(2, 6) for _ in range(row_count)]
    col_sizes = [random.randint(2, 6) for _ in range(col_count)]
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        return None

    lower, cells = _build_lower_canvas(row_sizes, col_sizes)
    h, w = lower.shape

    seed_cell_r = random.randrange(row_count)
    seed_cell_c = random.randrange(col_count)
    max_depth = max(
        abs(seed_cell_r - rr) + abs(seed_cell_c - cc)
        for rr in range(row_count)
        for cc in range(col_count)
    )
    if not (3 <= max_depth <= len(PALETTE_COLORS) - 1):
        return None

    cell_index = seed_cell_r * col_count + seed_cell_c
    r0, c0, r1, c1 = cells[cell_index]
    seed_r = _choose_seed_coord(r0, r1)
    seed_c = _choose_seed_coord(c0, c1)

    code = np.array(random.sample(PALETTE_COLORS, max_depth + 1), dtype=int)
    grid = np.full((h + 2, w), BG, dtype=int)
    grid[0, : len(code)] = code
    grid[1, :] = SEPARATOR
    grid[2:, :] = lower
    grid[2 + seed_r, seed_c] = int(code[0])

    try:
        info = _parse_input(grid)
    except AssertionError:
        return None

    if info["max_depth"] != max_depth:
        return None

    _, path_counts = _shortest_path_counts(info["graph"], info["seed_region"])
    if max(path_counts.values()) < 2:
        return None

    min_dists = _min_pixel_distances(info["labels"], info["seed"])
    if not _has_geometry_counterexample(info["depths"], min_dists):
        return None

    out = main(grid)
    if not np.array_equal(out[: info["separator_r"] + 1], grid[: info["separator_r"] + 1]):
        return None
    if not np.array_equal(out[2:, :][lower == WALL], np.full(int(np.sum(lower == WALL)), WALL, dtype=int)):
        return None
    for region_id, depth in info["depths"].items():
        vals = np.unique(out[2:, :][info["labels"] == region_id])
        if len(vals) != 1 or int(vals[0]) != int(code[depth]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-crossing palette task")
