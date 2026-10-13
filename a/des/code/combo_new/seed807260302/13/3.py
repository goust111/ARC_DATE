from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


WALL = 0
MAROON = 9
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FILL_COLORS = tuple(range(1, 9))

GEN_RETRY_LIMIT = 400
CELL_SIZE_OPTIONS = (3, 4)
CHAMBER_ROWS_RANGE = (3, 6)
CHAMBER_COLS_RANGE = (3, 6)
ARROW_COUNT_OPTIONS = (1, 2, 3)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _border_mode(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _same_color_component(
    g: np.ndarray,
    start: tuple[int, int],
    color: int,
) -> set[tuple[int, int]]:
    h, w = g.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (
                _in_bounds(h, w, nr, nc)
                and (nr, nc) not in seen
                and int(g[nr, nc]) == color
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _parse_arrows(g: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _mask_components(g == MAROON):
        assert len(comp) == 3, "each maroon marker must be a 3-pixel arrow"
        comp_set = set(comp)
        degrees = {
            cell: sum((nr, nc) in comp_set for nr, nc in _neighbors4(*cell))
            for cell in comp
        }
        corners = [cell for cell, degree in degrees.items() if degree == 2]
        ends = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(corners) == 1 and len(ends) == 2, "arrow must be an L shape"
        corner = corners[0]
        v0 = (ends[0][0] - corner[0], ends[0][1] - corner[1])
        v1 = (ends[1][0] - corner[0], ends[1][1] - corner[1])
        assert abs(v0[0]) + abs(v0[1]) == 1
        assert abs(v1[0]) + abs(v1[1]) == 1
        assert v0[0] * v1[0] + v0[1] * v1[1] == 0
        arrows.append(
            {
                "cells": tuple(sorted(comp)),
                "corner": corner,
                "ends": tuple(ends),
            }
        )
    arrows.sort(key=lambda item: tuple(item["corner"]))  # type: ignore[arg-type]
    return arrows


def _infer_host_color(
    g: np.ndarray,
    arrow_cells: tuple[tuple[int, int], ...],
    wall: int,
) -> int:
    arrow_set = set(arrow_cells)
    counts: Counter[int] = Counter()
    h, w = g.shape
    for r, c in arrow_cells:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in arrow_set:
                continue
            color = int(g[nr, nc])
            if color in (wall, MAROON):
                continue
            counts[color] += 1
    assert counts, "arrow must sit inside a colored source region"
    best = max(counts.values())
    colors = [color for color, cnt in counts.items() if cnt == best]
    assert len(colors) == 1, "source color must be structurally unique"
    return int(colors[0])


def _fill_unprocessed_arrows(
    g: np.ndarray,
    arrows: list[dict[str, object]],
    start_idx: int,
    wall: int,
) -> tuple[np.ndarray, dict[int, int]]:
    filled = g.copy()
    hosts: dict[int, int] = {}
    for idx in range(start_idx, len(arrows)):
        cells = arrows[idx]["cells"]  # type: ignore[assignment]
        host = _infer_host_color(g, cells, wall)
        hosts[idx] = host
        for r, c in cells:
            filled[r, c] = host
    return filled, hosts


def _head_target_start(
    filled: np.ndarray,
    arrow: dict[str, object],
    wall: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = filled.shape
    corner = arrow["corner"]  # type: ignore[assignment]
    ends = arrow["ends"]  # type: ignore[assignment]
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for er, ec in ends:
        vec = (er - corner[0], ec - corner[1])
        wr, wc = er + vec[0], ec + vec[1]
        tr, tc = er + 2 * vec[0], ec + 2 * vec[1]
        if not _in_bounds(h, w, wr, wc) or not _in_bounds(h, w, tr, tc):
            continue
        target_color = int(filled[tr, tc])
        if int(filled[wr, wc]) == wall and target_color not in (wall, MAROON):
            hits.append((vec, (tr, tc)))
    assert len(hits) == 1, "each arrow must point through exactly one divider wall"
    return hits[0]


def _adjacent_same_color_regions(
    filled: np.ndarray,
    region: set[tuple[int, int]],
    color: int,
    wall: int,
) -> set[tuple[int, int]]:
    h, w = filled.shape
    seen = set(region)
    out: set[tuple[int, int]] = set()
    for r, c in region:
        for dr, dc in FOUR_DIRS:
            wr, wc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc
            if not _in_bounds(h, w, wr, wc) or not _in_bounds(h, w, nr, nc):
                continue
            if int(filled[wr, wc]) != wall or int(filled[nr, nc]) != color:
                continue
            if (nr, nc) in seen:
                continue
            comp = _same_color_component(filled, (nr, nc), color)
            seen |= comp
            out |= comp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor map regions by following maroon L-arrows and a one-step color cascade.

    Input structure:
    - A thin black divider network partitions the grid into irregular colored regions with no empty cells.
    - Some regions contain a maroon 3-pixel L marker: one corner pixel plus two orthogonal arm pixels.
    - One arm points toward a neighboring region across a single black divider cell; the other arm is just the tail bend.

    Rule:
    - Process the arrows in corner order from top to bottom, then left to right.
    - For each arrow, infer its source region color from the surrounding non-maroon cells, then find the target region across the divider in the arrowhead direction.
    - Recolor the whole target region to the source color, then also recolor every region of the target's old color that shares a divider edge with that target region; do not recurse beyond that one layer.
    - Replace the current arrow's maroon pixels with the final source-region color, while later arrows remain maroon until their own turn.

    Key insight:
    - The maroon L does not mark a simple flood-fill cell: it selects one region in the region-adjacency graph, then transfers the displaced color to same-colored neighboring regions exactly one divider hop away.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h > 0 and w > 0

    wall = _border_mode(g)
    arrows = _parse_arrows(g)
    assert arrows, "expected at least one maroon arrow"

    out = g.copy()
    future_arrow_cells = {
        idx: set(arrows[idx]["cells"])  # type: ignore[arg-type]
        for idx in range(len(arrows))
    }

    for idx, arrow in enumerate(arrows):
        filled, hosts = _fill_unprocessed_arrows(out, arrows, idx, wall)
        src_color = hosts[idx]
        _head_vec, target_start = _head_target_start(filled, arrow, wall)
        old_color = int(filled[target_start])
        target_region = _same_color_component(filled, target_start, old_color)
        cascade_region = _adjacent_same_color_regions(
            filled,
            target_region,
            old_color,
            wall,
        )

        later_cells: set[tuple[int, int]] = set()
        for later_idx in range(idx + 1, len(arrows)):
            later_cells |= future_arrow_cells[later_idx]

        for r, c in target_region | cascade_region:
            if (r, c) not in later_cells:
                out[r, c] = src_color

        for r, c in arrow["cells"]:  # type: ignore[assignment]
            out[r, c] = src_color

    return out


def _choose_canvas() -> tuple[int, int, int]:
    for _ in range(100):
        cell_size = random.choice(CELL_SIZE_OPTIONS)
        max_rows = CHAMBER_ROWS_RANGE[1]
        max_cols = CHAMBER_COLS_RANGE[1]
        rows = random.randint(CHAMBER_ROWS_RANGE[0], max_rows)
        cols = random.randint(CHAMBER_COLS_RANGE[0], max_cols)
        height = rows * cell_size + rows + 1
        width = cols * cell_size + cols + 1
        if max(height, width) <= 30:
            return rows, cols, cell_size
    raise AssertionError("unable to choose a valid ARC-sized canvas")


def _grow_partition(rows: int, cols: int, n_regions: int) -> np.ndarray:
    grid = np.full((rows, cols), -1, dtype=int)
    seeds = random.sample([(r, c) for r in range(rows) for c in range(cols)], n_regions)
    for idx, (r, c) in enumerate(seeds):
        grid[r, c] = idx

    while np.any(grid < 0):
        frontier: list[tuple[int, int, list[int]]] = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] >= 0:
                    continue
                neighbors = sorted(
                    {
                        int(grid[nr, nc])
                        for nr, nc in _neighbors4(r, c)
                        if _in_bounds(rows, cols, nr, nc) and grid[nr, nc] >= 0
                    }
                )
                if neighbors:
                    frontier.append((r, c, neighbors))
        assert frontier
        r, c, neighbors = random.choice(frontier)
        grid[r, c] = random.choice(neighbors)
    return grid


def _region_graph(partition: np.ndarray) -> tuple[dict[int, list[tuple[int, int]]], dict[int, set[int]]]:
    rows, cols = partition.shape
    cells: dict[int, list[tuple[int, int]]] = {}
    adjacency: dict[int, set[int]] = {}
    for r in range(rows):
        for c in range(cols):
            region = int(partition[r, c])
            cells.setdefault(region, []).append((r, c))
            adjacency.setdefault(region, set())
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if not _in_bounds(rows, cols, nr, nc):
                    continue
                other = int(partition[nr, nc])
                if other != region:
                    adjacency[region].add(other)
                    adjacency.setdefault(other, set()).add(region)
    return cells, adjacency


def _arrow_candidates(partition: np.ndarray) -> list[dict[str, object]]:
    rows, cols = partition.shape
    candidates: list[dict[str, object]] = []
    for r in range(rows):
        for c in range(cols):
            source = int(partition[r, c])
            for dr, dc in FOUR_DIRS:
                nr, nc = r + dr, c + dc
                if not _in_bounds(rows, cols, nr, nc):
                    continue
                target = int(partition[nr, nc])
                if target == source:
                    continue
                candidates.append(
                    {
                        "source_region": source,
                        "target_region": target,
                        "source_cell": (r, c),
                        "dir": (dr, dc),
                        "order": r * cols + c,
                    }
                )
    return candidates


def _interior_origin(cell_r: int, cell_c: int, cell_size: int) -> tuple[int, int]:
    return 1 + cell_r * (cell_size + 1), 1 + cell_c * (cell_size + 1)


def _place_arrow_pixels(
    cell_r: int,
    cell_c: int,
    cell_size: int,
    direction: tuple[int, int],
) -> dict[str, object]:
    r0, c0 = _interior_origin(cell_r, cell_c, cell_size)
    dr, dc = direction
    bend = random.choice([-1, 1])

    if dr == 0 and dc == 1:
        base = random.randint(0, cell_size - 2)
        if bend > 0:
            corner = (r0 + base, c0 + cell_size - 2)
            tail = (r0 + base + 1, c0 + cell_size - 2)
            tip = (r0 + base, c0 + cell_size - 1)
        else:
            corner = (r0 + base + 1, c0 + cell_size - 2)
            tail = (r0 + base, c0 + cell_size - 2)
            tip = (r0 + base + 1, c0 + cell_size - 1)
    elif dr == 0 and dc == -1:
        base = random.randint(0, cell_size - 2)
        if bend > 0:
            corner = (r0 + base, c0 + 1)
            tail = (r0 + base + 1, c0 + 1)
            tip = (r0 + base, c0)
        else:
            corner = (r0 + base + 1, c0 + 1)
            tail = (r0 + base, c0 + 1)
            tip = (r0 + base + 1, c0)
    elif dr == 1 and dc == 0:
        base = random.randint(0, cell_size - 2)
        if bend > 0:
            corner = (r0 + cell_size - 2, c0 + base)
            tail = (r0 + cell_size - 2, c0 + base + 1)
            tip = (r0 + cell_size - 1, c0 + base)
        else:
            corner = (r0 + cell_size - 2, c0 + base + 1)
            tail = (r0 + cell_size - 2, c0 + base)
            tip = (r0 + cell_size - 1, c0 + base + 1)
    else:
        base = random.randint(0, cell_size - 2)
        if bend > 0:
            corner = (r0 + 1, c0 + base)
            tail = (r0 + 1, c0 + base + 1)
            tip = (r0, c0 + base)
        else:
            corner = (r0 + 1, c0 + base + 1)
            tail = (r0 + 1, c0 + base)
            tip = (r0, c0 + base + 1)

    cells = tuple(sorted([corner, tail, tip]))
    return {"cells": cells, "corner": corner}


def _choose_arrows(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    if not candidates:
        return None

    desired = min(random.choice(ARROW_COUNT_OPTIONS), len(candidates))
    selected: list[dict[str, object]] = []
    used_cells: set[tuple[int, int]] = set()
    used_sources: set[int] = set()
    used_targets: set[int] = set()

    if desired >= 2 and random.random() < 0.5:
        starts = []
        for cand in candidates:
            successors = [
                other
                for other in candidates
                if other["source_region"] == cand["target_region"]
                and other["source_cell"] != cand["source_cell"]
                and other["source_region"] not in used_sources
                and other["target_region"] not in used_targets
            ]
            if successors:
                starts.append((cand, successors))
        if starts:
            first, successors = random.choice(starts)
            second = random.choice(successors)
            selected.extend([first, second])
            used_cells.add(first["source_cell"])  # type: ignore[arg-type]
            used_cells.add(second["source_cell"])  # type: ignore[arg-type]
            used_sources.add(int(first["source_region"]))
            used_sources.add(int(second["source_region"]))
            used_targets.add(int(first["target_region"]))
            used_targets.add(int(second["target_region"]))

    while len(selected) < desired:
        pool = [
            cand
            for cand in candidates
            if cand["source_cell"] not in used_cells
            and int(cand["source_region"]) not in used_sources
            and int(cand["target_region"]) not in used_targets
        ]
        if not pool:
            break
        cand = random.choice(pool)
        selected.append(cand)
        used_cells.add(cand["source_cell"])  # type: ignore[arg-type]
        used_sources.add(int(cand["source_region"]))
        used_targets.add(int(cand["target_region"]))

    return selected or None


def _plan_colors(
    n_regions: int,
    adjacency: dict[int, set[int]],
    arrows: list[dict[str, object]],
) -> tuple[dict[int, int], list[dict[str, int]]] | None:
    palette = random.sample(FILL_COLORS, random.randint(3, min(6, len(FILL_COLORS))))
    initial: dict[int, int] = {}
    current: dict[int, int] = {}
    features: list[dict[str, int]] = []

    force_cascade = [
        len(adjacency[int(arrow["target_region"])] - {int(arrow["source_region"])}) > 0
        for arrow in arrows
    ]
    if any(force_cascade) and random.random() < 0.6:
        forced_idx = random.choice([i for i, ok in enumerate(force_cascade) if ok])
    else:
        forced_idx = -1

    for idx, arrow in enumerate(arrows):
        source = int(arrow["source_region"])
        target = int(arrow["target_region"])

        if source not in current:
            color = random.choice(palette)
            initial[source] = color
            current[source] = color
        src_color = current[source]

        if target in current:
            old_color = current[target]
            if old_color == src_color:
                return None
        else:
            options = [color for color in palette if color != src_color]
            old_color = random.choice(options)
            initial[target] = old_color
            current[target] = old_color

        neighbors = list(adjacency[target] - {source})
        random.shuffle(neighbors)
        max_cascade = min(2, len(neighbors))
        wanted = 0
        if max_cascade > 0:
            if idx == forced_idx:
                wanted = random.randint(1, max_cascade)
            elif random.random() < 0.5:
                wanted = random.randint(0, max_cascade)

        already_same = [nbr for nbr in neighbors if nbr in current and current[nbr] == old_color]
        free_neighbors = [nbr for nbr in neighbors if nbr not in current]
        wanted = min(wanted, len(already_same) + len(free_neighbors))

        need_same = max(0, wanted - len(already_same))
        same_choices = set(random.sample(free_neighbors, need_same))
        for nbr in free_neighbors:
            if nbr in same_choices:
                initial[nbr] = old_color
                current[nbr] = old_color
            else:
                options = [color for color in palette if color != old_color]
                color = random.choice(options)
                initial[nbr] = color
                current[nbr] = color

        cascade_neighbors = [nbr for nbr in neighbors if current[nbr] == old_color]
        current[target] = src_color
        for nbr in cascade_neighbors:
            current[nbr] = src_color

        features.append(
            {
                "source_color": src_color,
                "old_color": old_color,
                "cascade_count": len(cascade_neighbors),
            }
        )

    for region in range(n_regions):
        if region not in initial:
            initial[region] = random.choice(palette)

    return initial, features


def _render_grid(
    partition: np.ndarray,
    cell_size: int,
    region_colors: dict[int, int],
    arrows: list[dict[str, object]],
) -> np.ndarray:
    chamber_rows, chamber_cols = partition.shape
    height = chamber_rows * cell_size + chamber_rows + 1
    width = chamber_cols * cell_size + chamber_cols + 1
    grid = np.full((height, width), WALL, dtype=int)

    for r in range(chamber_rows):
        for c in range(chamber_cols):
            color = region_colors[int(partition[r, c])]
            rr, cc = _interior_origin(r, c, cell_size)
            grid[rr : rr + cell_size, cc : cc + cell_size] = color

    for r in range(chamber_rows):
        for c in range(chamber_cols):
            color = region_colors[int(partition[r, c])]
            rr, cc = _interior_origin(r, c, cell_size)
            if r + 1 < chamber_rows and int(partition[r + 1, c]) == int(partition[r, c]):
                grid[rr + cell_size, cc : cc + cell_size] = color
            if c + 1 < chamber_cols and int(partition[r, c + 1]) == int(partition[r, c]):
                grid[rr : rr + cell_size, cc + cell_size] = color

    for r in range(chamber_rows - 1):
        for c in range(chamber_cols - 1):
            around = {
                int(partition[r, c]),
                int(partition[r + 1, c]),
                int(partition[r, c + 1]),
                int(partition[r + 1, c + 1]),
            }
            if len(around) == 1:
                rr, cc = _interior_origin(r, c, cell_size)
                color = region_colors[next(iter(around))]
                grid[rr + cell_size, cc + cell_size] = color

    for arrow in arrows:
        for r, c in arrow["cells"]:  # type: ignore[assignment]
            grid[r, c] = MAROON

    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(_border_mode(grid)) != WALL:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _build_example() -> np.ndarray | None:
    chamber_rows, chamber_cols, cell_size = _choose_canvas()
    chamber_count = chamber_rows * chamber_cols
    max_regions = min(chamber_count - 1, 9)
    min_regions = min(5, max_regions)
    if min_regions < 2:
        return None

    n_regions = random.randint(min_regions, max_regions)
    partition = _grow_partition(chamber_rows, chamber_cols, n_regions)
    _region_cells, adjacency = _region_graph(partition)
    candidates = _arrow_candidates(partition)
    chosen = _choose_arrows(candidates)
    if not chosen:
        return None

    arrows: list[dict[str, object]] = []
    for cand in chosen:
        cell_r, cell_c = cand["source_cell"]  # type: ignore[assignment]
        placed = _place_arrow_pixels(cell_r, cell_c, cell_size, cand["dir"])  # type: ignore[arg-type]
        arrow = {
            **cand,
            **placed,
        }
        arrows.append(arrow)

    arrows.sort(key=lambda item: tuple(item["corner"]))  # type: ignore[arg-type]
    plan = _plan_colors(n_regions, adjacency, arrows)
    if plan is None:
        return None
    region_colors, features = plan

    if not any(feature["cascade_count"] > 0 for feature in features) and len(arrows) == 1 and random.random() < 0.5:
        return None

    grid = _render_grid(partition, cell_size, region_colors, arrows)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_example()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input grid")
