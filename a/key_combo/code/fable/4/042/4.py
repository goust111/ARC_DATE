from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _gate_orientation(grid: np.ndarray, r: int, c: int) -> str | None:
    color = int(grid[r, c])
    if color in (BG, WALL):
        return None
    if r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1:
        return None

    def wallish(rr: int, cc: int) -> bool:
        return int(grid[rr, cc]) in (WALL, color)

    vertical = (
        wallish(r - 1, c)
        and wallish(r + 1, c)
        and not wallish(r, c - 1)
        and not wallish(r, c + 1)
    )
    horizontal = (
        wallish(r, c - 1)
        and wallish(r, c + 1)
        and not wallish(r - 1, c)
        and not wallish(r + 1, c)
    )
    if vertical == horizontal:
        return None
    return "v" if vertical else "h"


def _classify_layout(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    gate_mask = np.zeros(grid.shape, dtype=bool)
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, WALL))
    for r, c in np.argwhere((grid != BG) & (grid != WALL)):
        rr = int(r)
        cc = int(c)
        gate_mask[rr, cc] = _gate_orientation(grid, rr, cc) is not None

    sources: dict[int, tuple[int, int]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        source_cells = [rc for rc in coords if not gate_mask[rc]]
        assert len(source_cells) == 1, f"color {color} must have exactly one source pixel"
        sources[color] = source_cells[0]
    assert len(sources) >= 2, "expected at least two source colors"
    return sources, gate_mask


def _distances_for_color(
    grid: np.ndarray,
    color: int,
    start: tuple[int, int],
    gate_mask: np.ndarray,
) -> np.ndarray:
    inf = grid.shape[0] * grid.shape[1] + 1
    dist = np.full(grid.shape, inf, dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or dist[nr, nc] != inf:
                continue
            cell = int(grid[nr, nc])
            if cell == BG or (cell == color and gate_mask[nr, nc]):
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood every reachable black cell from multiple colored seeds while colored gate segments only admit their matching color.

    Input structure:
    - Black background is partitioned by grey wall lines into rooms and corridors.
    - Each non-grey color has one single-pixel source somewhere in open space.
    - Straight colored gate segments are embedded inside grey walls; a gate color matches one source color.

    Rule:
    - All sources expand simultaneously in 4-neighbor steps across black cells.
    - Grey walls block everyone; a colored gate cell behaves like a passable tunnel cell only for the source of the same color, and traversing it still costs one step per cell.
    - Recolor each black cell with the uniquely earliest-arriving source color; ties stay black; walls, gates, and source pixels remain unchanged.

    Key insight:
    - Every source sees a different maze because each color unlocks only its own gates, so ownership is determined by per-color geodesic distance rather than plain Manhattan proximity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources, gate_mask = _classify_layout(g)
    dist_by_color = {
        color: _distances_for_color(g, color, start, gate_mask)
        for color, start in sources.items()
    }
    inf = g.shape[0] * g.shape[1] + 1

    out = g.copy()
    for r, c in np.argwhere(g == BG):
        rr = int(r)
        cc = int(c)
        best_color = None
        best_dist = inf
        tied = False
        for color, dist in dist_by_color.items():
            d = int(dist[rr, cc])
            if d < best_dist:
                best_dist = d
                best_color = color
                tied = False
            elif d == best_dist and d < inf:
                tied = True
        if best_color is not None and best_dist < inf and not tied:
            out[rr, cc] = int(best_color)
    return out


def _sample_lines(length: int, count: int, *, margin: int = 4, gap: int = 5) -> list[int] | None:
    candidates = list(range(margin, length - margin))
    if len(candidates) < count:
        return None
    for _ in range(120):
        picks = sorted(random.sample(candidates, count))
        if picks[0] < margin or length - 1 - picks[-1] < margin:
            continue
        if any(b - a < gap for a, b in zip(picks, picks[1:])):
            continue
        return picks
    return None


def _build_regions(
    height: int,
    width: int,
    h_walls: list[int],
    v_walls: list[int],
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    row_bounds = [-1, *h_walls, height]
    col_bounds = [-1, *v_walls, width]
    regions: list[dict] = []
    region_id_by_strip: dict[tuple[int, int], int] = {}
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0 = row_bounds[i] + 1
            r1 = row_bounds[i + 1] - 1
            c0 = col_bounds[j] + 1
            c1 = col_bounds[j + 1] - 1
            region_id = len(regions)
            region = {
                "id": region_id,
                "strip": (i, j),
                "bbox": (r0, c0, r1, c1),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
            regions.append(region)
            region_id_by_strip[(i, j)] = region_id
    return regions, region_id_by_strip


def _build_edges(
    height: int,
    width: int,
    h_walls: list[int],
    v_walls: list[int],
    region_id_by_strip: dict[tuple[int, int], int],
) -> list[dict]:
    row_bounds = [-1, *h_walls, height]
    col_bounds = [-1, *v_walls, width]
    edges: list[dict] = []

    for i, wall_r in enumerate(h_walls):
        for j in range(len(col_bounds) - 1):
            c0 = col_bounds[j] + 1
            c1 = col_bounds[j + 1] - 1
            if c1 - c0 + 1 < 3:
                continue
            edges.append(
                {
                    "id": len(edges),
                    "axis": "h",
                    "wall": wall_r,
                    "lo": c0 + 1,
                    "hi": c1 - 1,
                    "a": region_id_by_strip[(i, j)],
                    "b": region_id_by_strip[(i + 1, j)],
                }
            )

    for j, wall_c in enumerate(v_walls):
        for i in range(len(row_bounds) - 1):
            r0 = row_bounds[i] + 1
            r1 = row_bounds[i + 1] - 1
            if r1 - r0 + 1 < 3:
                continue
            edges.append(
                {
                    "id": len(edges),
                    "axis": "v",
                    "wall": wall_c,
                    "lo": r0 + 1,
                    "hi": r1 - 1,
                    "a": region_id_by_strip[(i, j)],
                    "b": region_id_by_strip[(i, j + 1)],
                }
            )
    return edges


def _other_region(edge: dict, region_id: int) -> int:
    return int(edge["b"]) if int(edge["a"]) == region_id else int(edge["a"])


def _pick_seed_cell(region: dict, occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = region["bbox"]
    rows = list(range(r0 + 1, r1)) or list(range(r0, r1 + 1))
    cols = list(range(c0 + 1, c1)) or list(range(c0, c1 + 1))
    cells = [(r, c) for r in rows for c in cols if (r, c) not in occupied]
    if not cells:
        return None
    return random.choice(cells)


def _pick_tie_cells(region: dict) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = region["bbox"]
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    if not rows or not cols:
        return None
    if len(cols) >= 3:
        row = random.choice(rows)
        return [(row, cols[0]), (row, cols[-1])]
    if len(rows) >= 3:
        col = random.choice(cols)
        return [(rows[0], col), (rows[-1], col)]
    return None


def _edge_gate_cells(edge: dict) -> list[tuple[int, int]]:
    span = int(edge["hi"]) - int(edge["lo"]) + 1
    length = 2 if span >= 5 and random.random() < 0.45 else 1
    length = min(length, span)
    start = random.randint(int(edge["lo"]), int(edge["hi"]) - length + 1)
    if edge["axis"] == "h":
        return [(int(edge["wall"]), c) for c in range(start, start + length)]
    return [(r, int(edge["wall"])) for r in range(start, start + length)]


def _paint_mask(mask: np.ndarray, region: dict) -> None:
    r0, c0, r1, c1 = region["bbox"]
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 26)
    width = random.randint(18, 26)
    h_count = random.randint(1, 2)
    v_count = random.randint(1, 2)
    h_walls = _sample_lines(height, h_count)
    v_walls = _sample_lines(width, v_count)
    if h_walls is None or v_walls is None:
        return None

    regions, region_id_by_strip = _build_regions(height, width, h_walls, v_walls)
    edges = _build_edges(height, width, h_walls, v_walls, region_id_by_strip)
    if len(regions) < 4 or not edges:
        return None

    tie_candidates = [
        region for region in regions
        if max(int(region["h"]), int(region["w"])) >= 5 and min(int(region["h"]), int(region["w"])) >= 3
    ]
    tie_required = len(regions) == 4 or random.random() < 0.6
    if tie_required and not tie_candidates:
        return None

    max_sources = min(5, len(regions) - 2 + (1 if tie_required else 0))
    if max_sources < 3:
        return None
    source_count = random.randint(3, max_sources)
    colors = random.sample(SOURCE_COLORS, source_count)

    tie_region = random.choice(tie_candidates) if tie_required else None
    distinct_seed_regions = source_count - 1 if tie_required else source_count
    if distinct_seed_regions > len(regions) - 2:
        return None

    chosen_regions: list[dict] = []
    if tie_region is not None:
        chosen_regions.append(tie_region)
    remaining_regions = [region for region in regions if region is not tie_region]
    chosen_regions.extend(random.sample(remaining_regions, distinct_seed_regions - len(chosen_regions)))

    seeded_region_ids = {int(region["id"]) for region in chosen_regions}
    unseeded_region_ids = {int(region["id"]) for region in regions} - seeded_region_ids
    if len(unseeded_region_ids) < 2:
        return None
    dead_region_id = random.choice(sorted(unseeded_region_ids))
    active_unseeded = unseeded_region_ids - {dead_region_id}
    if not active_unseeded:
        return None

    color_region: dict[int, int] = {}
    region_colors: dict[int, list[int]] = {}
    if tie_region is not None:
        tie_id = int(tie_region["id"])
        color_region[colors[0]] = tie_id
        color_region[colors[1]] = tie_id
        region_colors.setdefault(tie_id, []).extend([colors[0], colors[1]])
        rest_regions = [region for region in chosen_regions if int(region["id"]) != tie_id]
        for color, region in zip(colors[2:], rest_regions):
            rid = int(region["id"])
            color_region[color] = rid
            region_colors.setdefault(rid, []).append(color)
    else:
        for color, region in zip(colors, chosen_regions):
            rid = int(region["id"])
            color_region[color] = rid
            region_colors.setdefault(rid, []).append(color)

    grid = np.full((height, width), BG, dtype=int)
    for r in h_walls:
        grid[r, :] = WALL
    for c in v_walls:
        grid[:, c] = WALL

    occupied: set[tuple[int, int]] = set()
    for region in regions:
        rid = int(region["id"])
        assigned = region_colors.get(rid, [])
        if len(assigned) == 2:
            tie_cells = _pick_tie_cells(region)
            if tie_cells is None:
                return None
            for color, cell in zip(assigned, tie_cells):
                if cell in occupied:
                    return None
                occupied.add(cell)
                grid[cell] = color
        elif len(assigned) == 1:
            cell = _pick_seed_cell(region, occupied)
            if cell is None:
                return None
            occupied.add(cell)
            grid[cell] = assigned[0]

    edges_by_region: dict[int, list[dict]] = {int(region["id"]): [] for region in regions}
    allowed_edges: list[dict] = []
    for edge in edges:
        edges_by_region[int(edge["a"])].append(edge)
        edges_by_region[int(edge["b"])].append(edge)
        if dead_region_id not in (int(edge["a"]), int(edge["b"])):
            allowed_edges.append(edge)
    if not allowed_edges:
        return None

    gates_by_edge: dict[int, int] = {}
    reachable = {color: {region_id} for color, region_id in color_region.items()}
    active_frontier = [
        (color, edge)
        for color, region_id in color_region.items()
        for edge in edges_by_region[region_id]
        if int(edge["id"]) not in gates_by_edge and _other_region(edge, region_id) in active_unseeded
        and dead_region_id not in (int(edge["a"]), int(edge["b"]))
    ]
    if not active_frontier:
        return None
    first_color, first_edge = random.choice(active_frontier)
    gates_by_edge[int(first_edge["id"])] = int(first_color)
    reachable[first_color].add(_other_region(first_edge, color_region[first_color]))

    target_gate_count = random.randint(max(2, source_count - 1), min(len(allowed_edges), source_count + 2))
    for _ in range(40):
        if len(gates_by_edge) >= target_gate_count:
            break
        color = random.choice(colors)
        frontier: list[tuple[dict, int]] = []
        preferred: list[tuple[dict, int]] = []
        for region_id in reachable[color]:
            for edge in edges_by_region[region_id]:
                eid = int(edge["id"])
                if eid in gates_by_edge or dead_region_id in (int(edge["a"]), int(edge["b"])):
                    continue
                other = _other_region(edge, region_id)
                if other in reachable[color]:
                    continue
                frontier.append((edge, other))
                if other in active_unseeded:
                    preferred.append((edge, other))
        if not frontier:
            continue
        edge, other = random.choice(preferred if preferred else frontier)
        gates_by_edge[int(edge["id"])] = int(color)
        reachable[color].add(int(other))

    if len(gates_by_edge) < 2:
        return None

    for edge in edges:
        color = gates_by_edge.get(int(edge["id"]))
        if color is None:
            continue
        for r, c in _edge_gate_cells(edge):
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    bg_mask = grid == BG
    if not np.array_equal(out[~bg_mask], grid[~bg_mask]):
        return None
    if int(np.sum((out != BG) & bg_mask)) < 18:
        return None

    changed_colors = {int(out[r, c]) for r, c in np.argwhere((out != BG) & bg_mask)}
    if len(changed_colors) < 2:
        return None

    dead_mask = np.zeros(grid.shape, dtype=bool)
    _paint_mask(dead_mask, regions[dead_region_id])
    if not np.any((out == BG) & bg_mask & dead_mask):
        return None

    active_mask = np.zeros(grid.shape, dtype=bool)
    for region_id in active_unseeded:
        _paint_mask(active_mask, regions[region_id])
    if not np.any((out != BG) & bg_mask & active_mask):
        return None

    if tie_required and not np.any((out == BG) & bg_mask & ~dead_mask):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
