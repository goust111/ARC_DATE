from __future__ import annotations

import random

import numpy as np


PRIMARY_COLORS = (1, 2, 4)  # blue, red, yellow
MIX_COLORS = {
    frozenset((1, 2)): 6,  # blue + red -> magenta
    frozenset((2, 4)): 7,  # red + yellow -> orange
    frozenset((1, 4)): 3,  # blue + yellow -> green
}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_COLOR = 0
MAX_ATTEMPTS = 400


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _is_perpendicular(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def _trace_path(mask: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = start
    dr, dc = step
    cells = [start]
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w and bool(mask[r, c])):
            break
        cells.append((r, c))
    return cells


def _find_paths(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    mask = g != bg
    marker_cells: set[tuple[int, int]] = set()
    paths: list[dict[str, object]] = []
    seen_sources: set[tuple[int, int]] = set()

    for r, c in map(tuple, np.argwhere(mask)):
        nbrs = _neighbors(mask, int(r), int(c))
        if len(nbrs) != 1:
            continue

        (nr, nc) = nbrs[0]
        if int(g[nr, nc]) != int(g[r, c]) or int(g[r, c]) not in PRIMARY_COLORS:
            continue

        nexts = [pos for pos in _neighbors(mask, nr, nc) if pos != (int(r), int(c))]
        if len(nexts) != 1:
            continue

        step = (nexts[0][0] - nr, nexts[0][1] - nc)
        marker_vec = (int(r) - nr, int(c) - nc)
        if not _is_perpendicular(marker_vec, step):
            continue

        source = (nr, nc)
        if source in seen_sources:
            raise AssertionError("expected each source endpoint to have a single marker leaf")

        seen_sources.add(source)
        marker_cells.add((int(r), int(c)))
        cells = _trace_path(mask, source, step)
        orient = "h" if step[0] == 0 else "v"
        paths.append(
            {
                "source": source,
                "marker": (int(r), int(c)),
                "step": step,
                "cells": cells,
                "color": int(g[source]),
                "orient": orient,
            }
        )

    if not paths:
        raise AssertionError("expected at least one marked path source")

    covered = set(marker_cells)
    for path in paths:
        covered.update(path["cells"])

    occupied = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if covered != occupied:
        raise AssertionError("every non-background cell must belong to a path or source marker")

    return paths, marker_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each marked primary-color path to its first crossing and recolor the downstream branch with the mixed color.

    Input structure:
    - The background is the dominant color.
    - Foreground cells form thin straight horizontal or vertical paths in the primary colors blue, red, and yellow.
    - Each path has one source end marked by a same-color 2x1 or 1x2 domino, implemented as a one-cell marker leaf attached to the endpoint.
    - Horizontal and vertical paths may overlap at single-pixel crossings whose visible color is one of the two path colors.

    Rule:
    - Read each source marker to determine the path endpoint and the signal direction from that endpoint toward the far end of the straight path.
    - The signal keeps the original path unchanged until it reaches the first crossing on that path.
    - At an activated crossing, mix the two path colors: red+blue->magenta, red+yellow->orange, blue+yellow->green.
    - From that crossing, paint the cells farther from each path's source with the mixed color, stopping immediately before the next crossing on that path; crossing pixels themselves keep their existing color.

    Key insight:
    - The thickened endpoint identifies both the source location and the path orientation, so the solver can direct each straight path, find its first crossing, and then recolor only the downstream segment between crossings.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    bg = _background(g)
    colors = set(map(int, np.unique(g)))
    if not colors <= {bg, *PRIMARY_COLORS}:
        raise AssertionError("input should only contain background plus blue/red/yellow")

    paths, _marker_cells = _find_paths(g, bg)

    cell_hits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pid, path in enumerate(paths):
        for idx, cell in enumerate(path["cells"]):
            cell_hits.setdefault(cell, []).append((pid, idx))

    crossings: dict[tuple[int, int], dict[str, object]] = {}
    for cell, hits in cell_hits.items():
        if len(hits) == 1:
            continue
        if len(hits) != 2:
            raise AssertionError("a crossing must involve exactly two paths")
        (a, ia), (b, ib) = hits
        if paths[a]["orient"] == paths[b]["orient"]:
            raise AssertionError("overlap must be between one horizontal and one vertical path")
        pair = frozenset((int(paths[a]["color"]), int(paths[b]["color"])))
        if pair not in MIX_COLORS:
            raise AssertionError("crossing paths must use two distinct primary colors")
        crossings[cell] = {"paths": (a, b), "index": {a: ia, b: ib}}

    path_crossings: list[list[tuple[int, int]]] = [[] for _ in paths]
    for cell, info in crossings.items():
        a, b = info["paths"]
        path_crossings[a].append(cell)
        path_crossings[b].append(cell)

    cross_pos: list[dict[tuple[int, int], int]] = []
    for pid, cells in enumerate(path_crossings):
        ordered = sorted(cells, key=lambda cell: int(crossings[cell]["index"][pid]))
        path_crossings[pid] = ordered
        cross_pos.append({cell: i for i, cell in enumerate(ordered)})

    activated = {
        cells[0]
        for cells in path_crossings
        if cells
    }

    out = g.copy()
    for cell in activated:
        a, b = crossings[cell]["paths"]
        mix = MIX_COLORS[frozenset((int(paths[a]["color"]), int(paths[b]["color"])))]
        for pid in (a, b):
            cells = path_crossings[pid]
            here = cross_pos[pid][cell]
            start_idx = int(crossings[cell]["index"][pid])
            stop_idx = len(paths[pid]["cells"])
            if here + 1 < len(cells):
                stop_idx = int(crossings[cells[here + 1]]["index"][pid])
            for rr, cc in paths[pid]["cells"][start_idx + 1 : stop_idx]:
                if int(out[rr, cc]) not in (int(g[rr, cc]), mix):
                    raise AssertionError("conflicting mixed colors on the same branch")
                out[rr, cc] = mix

    return out


def _apply_random_transform(tile: np.ndarray) -> np.ndarray:
    out = np.rot90(tile, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _build_pair_group() -> np.ndarray:
    h = random.randint(8, 11)
    w = random.randint(8, 11)
    row = random.randint(3, h - 4)
    col = random.randint(3, w - 4)
    h_color, v_color = random.sample(PRIMARY_COLORS, 2)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    grid[row, 1 : w - 1] = h_color
    grid[row - 1, 1] = h_color
    grid[1 : h - 1, col] = v_color
    grid[1, col - 1] = v_color
    grid[row, col] = random.choice((h_color, v_color))
    return _apply_random_transform(grid)


def _build_chain_group() -> np.ndarray:
    count = random.choice((2, 3))
    gap = random.choice((3, 4))
    cols = [3 + i * gap for i in range(count)]
    w = cols[-1] + 4
    h = random.randint(8, 11)
    row = random.randint(3, h - 4)
    main_color, branch_color = random.sample(PRIMARY_COLORS, 2)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    grid[row, 1 : w - 1] = main_color
    grid[row - 1, 1] = main_color
    for col in cols:
        grid[1 : h - 1, col] = branch_color
        grid[1, col - 1] = branch_color
        grid[row, col] = random.choice((main_color, branch_color))
    return _apply_random_transform(grid)


def _build_isolated_group() -> np.ndarray:
    h = random.randint(5, 8)
    w = random.randint(6, 11)
    row = random.randint(2, h - 3)
    color = random.choice(PRIMARY_COLORS)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[row, 1 : w - 1] = color
    grid[row - 1, 1] = color
    return _apply_random_transform(grid)


def _layout_groups(groups: list[np.ndarray]) -> np.ndarray | None:
    groups = groups[:]
    random.shuffle(groups)
    total_width = sum(tile.shape[1] for tile in groups) + max(0, len(groups) - 1)
    row_count = 1 if len(groups) == 1 or (total_width <= 24 and random.random() < 0.5) else 2

    rows: list[list[np.ndarray]] = [[] for _ in range(row_count)]
    row_widths = [0] * row_count
    row_heights = [0] * row_count
    for tile in sorted(groups, key=lambda x: (x.shape[1], x.shape[0]), reverse=True):
        idx = min(range(row_count), key=lambda i: row_widths[i])
        gap = 1 if rows[idx] else 0
        rows[idx].append(tile)
        row_widths[idx] += gap + tile.shape[1]
        row_heights[idx] = max(row_heights[idx], tile.shape[0])

    inner_w = max(row_widths) if row_widths else 0
    row_gaps = [random.randint(1, 3) for _ in range(max(0, row_count - 1))]
    inner_h = sum(row_heights) + sum(row_gaps)
    if inner_h > 28 or inner_w > 28:
        return None

    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    max_bottom = 30 - (inner_h + pad_top)
    max_right = 30 - (inner_w + pad_left)
    if max_bottom < 1 or max_right < 1:
        return None
    pad_bottom = random.randint(1, min(3, max_bottom))
    pad_right = random.randint(1, min(3, max_right))

    H = inner_h + pad_top + pad_bottom
    W = inner_w + pad_left + pad_right
    grid = np.full((H, W), BG_COLOR, dtype=int)

    r0 = pad_top
    for row_idx, tiles in enumerate(rows):
        if not tiles:
            continue
        used_w = row_widths[row_idx]
        c0 = pad_left + random.randint(0, inner_w - used_w)

        for i, tile in enumerate(tiles):
            th, tw = tile.shape
            y = r0 + random.randint(0, row_heights[row_idx] - th)
            mask = tile != BG_COLOR
            sub = grid[y : y + th, c0 : c0 + tw]
            if np.any(sub[mask] != BG_COLOR):
                return None
            sub[mask] = tile[mask]
            c0 += tw
            if i + 1 < len(tiles):
                c0 += 1

        r0 += row_heights[row_idx]
        if row_idx < len(row_gaps):
            r0 += row_gaps[row_idx]

    return _apply_random_transform(grid)


def _gen_attempt() -> np.ndarray | None:
    active_count = random.randint(1, 3)
    include_chain = random.random() < 0.5
    groups: list[np.ndarray] = []

    if include_chain:
        groups.append(_build_chain_group())
    else:
        groups.append(_build_pair_group())

    while len([g for g in groups if g is not None]) < active_count:
        builder = random.choice((_build_pair_group, _build_chain_group))
        groups.append(builder())

    isolated_count = random.randint(0, max(0, 3 - active_count))
    for _ in range(isolated_count):
        groups.append(_build_isolated_group())

    grid = _layout_groups(groups)
    if grid is None:
        return None

    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-path puzzle")
