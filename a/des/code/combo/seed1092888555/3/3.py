import random
from collections import Counter

import numpy as np


BG = 8
T_COLOR = 2
PALETTE = (1, 3, 4, 5, 6, 7, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = DIRS
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _t_cells(junction: tuple[int, int], stem_dir: tuple[int, int], stem_len: int) -> set[tuple[int, int]]:
    jr, jc = junction
    dr, dc = stem_dir
    cells = {(jr, jc)}
    if dr != 0:
        cells.add((jr, jc - 1))
        cells.add((jr, jc + 1))
    else:
        cells.add((jr - 1, jc))
        cells.add((jr + 1, jc))
    for step in range(1, stem_len + 1):
        cells.add((jr + dr * step, jc + dc * step))
    return cells


def _decode_t_component(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    cells = {tuple(map(int, rc)) for rc in coords}
    degrees = {}
    neighbors = {}
    for r, c in cells:
        nbrs = []
        for dr, dc in DIRS:
            if (r + dr, c + dc) in cells:
                nbrs.append((dr, dc))
        neighbors[(r, c)] = nbrs
        degrees[(r, c)] = len(nbrs)

    junctions = [rc for rc, deg in degrees.items() if deg == 3]
    assert len(junctions) == 1, "expected exactly one T junction"
    junction = junctions[0]
    nbr_dirs = neighbors[junction]
    assert len(nbr_dirs) == 3

    branch_lengths = {}
    for dr, dc in nbr_dirs:
        length = 0
        cr, cc = junction
        while True:
            cr += dr
            cc += dc
            if (cr, cc) not in cells:
                break
            length += 1
            sideways = 0
            for sdr, sdc in DIRS:
                if (sdr, sdc) == (dr, dc) or (sdr, sdc) == (-dr, -dc):
                    continue
                if (cr + sdr, cc + sdc) in cells:
                    sideways += 1
            assert sideways == 0, "branch must be straight"
        assert length >= 1
        branch_lengths[(dr, dc)] = length

    stem_dir = None
    for dr, dc in nbr_dirs:
        opp = (-dr, -dc)
        if opp in branch_lengths and branch_lengths[(dr, dc)] == 1 and branch_lengths[opp] == 1:
            stem_dir = next(d for d in nbr_dirs if d not in {(dr, dc), opp})
            break
    assert stem_dir is not None, "failed to identify crossbar"

    stem_len = branch_lengths[stem_dir]
    assert 1 <= stem_len <= 4
    assert len(cells) == stem_len + 3
    return junction, OPPOSITE[stem_dir], stem_len


def _find_legend(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    markers = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] in (bg, T_COLOR):
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            markers.append((r, c, int(grid[r, c])))
    assert len(markers) == 4, "expected exactly four edge markers"
    markers.sort(key=lambda item: (item[1], item[0]))
    return [color for _r, _c, color in markers]


def _ray_trace(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int], bg: int) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    path = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return path, "edge"
        if int(grid[r, c]) != int(bg):
            return path, "object"
        path.append((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _find_legend(g, bg)

    out = g.copy()
    red_components = components_mask(g == T_COLOR, connectivity=4)
    assert red_components, "expected at least one red T object"

    for comp in red_components:
        junction, ray_dir, stem_len = _decode_t_component(comp)
        color = legend[stem_len - 1]
        ray_cells, _stop = _ray_trace(g, junction, ray_dir, bg)
        for r, c in ray_cells:
            out[r, c] = color
    return out


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r == 0:
            return False
        if (r, c) in occupied:
            return False
        for dr, dc in DIRS:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _sample_marker_cols(width: int) -> list[int]:
    cols = sorted(random.sample(range(1, width - 1), 4))
    if min(cols[i + 1] - cols[i] for i in range(3)) < 2:
        return []
    return cols


def _sample_junction(stem_dir: tuple[int, int], stem_len: int, h: int, w: int) -> tuple[int, int] | None:
    row_lo, row_hi = 2, h - 2
    col_lo, col_hi = 1, w - 2
    dr, dc = stem_dir
    if dr == -1:
        row_lo = max(row_lo, stem_len)
    if dr == 1:
        row_hi = min(row_hi, h - 1 - stem_len)
    if dc == -1:
        col_lo = max(col_lo, stem_len)
    if dc == 1:
        col_hi = min(col_hi, w - 1 - stem_len)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _termination_type(grid: np.ndarray, junction: tuple[int, int], ray_dir: tuple[int, int]) -> tuple[str, int]:
    path, stop = _ray_trace(grid, junction, ray_dir, BG)
    return stop, len(path)


def _build_grid(objects: list[dict], marker_cols: list[int], legend_colors: list[int], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for col, color in zip(marker_cols, legend_colors):
        grid[0, col] = color
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = T_COLOR
    return grid


def _balanced_cycle(items: list) -> list:
    order = items[:]
    random.shuffle(order)
    return order


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 24)
    marker_cols = _sample_marker_cols(w)
    if not marker_cols:
        return None

    legend_colors = random.sample(PALETTE, 4)
    n_objects = random.randint(3, 6)

    stem_order = _balanced_cycle(DIRS)
    length_order = _balanced_cycle([1, 2, 3, 4])
    stem_dirs = [stem_order[i % 4] for i in range(n_objects)]
    stem_lens = [length_order[i % 4] for i in range(n_objects)]

    objects = []
    occupied = {(0, c) for c in marker_cols}

    source_dir = stem_dirs[0]
    source_len = stem_lens[0]
    for _ in range(80):
        source_junction = _sample_junction(source_dir, source_len, h, w)
        if source_junction is None:
            return None
        source_cells = _t_cells(source_junction, source_dir, source_len)
        if not _can_place(source_cells, occupied, h, w):
            continue
        ray_dir = OPPOSITE[source_dir]
        clear_path, stop = _ray_trace(np.full((h, w), BG, dtype=int), source_junction, ray_dir, BG)
        if stop != "edge" or len(clear_path) < 4:
            continue
        break
    else:
        return None

    objects.append({"junction": source_junction, "stem_dir": source_dir, "stem_len": source_len, "cells": source_cells})
    occupied |= source_cells

    blocker_dir = stem_dirs[1]
    blocker_len = stem_lens[1]
    ray_dr, ray_dc = OPPOSITE[source_dir]
    blocker_placed = False
    for gap in random.sample(list(range(3, 8)), 5):
        bj = (source_junction[0] + ray_dr * gap, source_junction[1] + ray_dc * gap)
        blocker_cells = _t_cells(bj, blocker_dir, blocker_len)
        if not _can_place(blocker_cells, occupied, h, w):
            continue
        blocker_placed = True
        objects.append({"junction": bj, "stem_dir": blocker_dir, "stem_len": blocker_len, "cells": blocker_cells})
        occupied |= blocker_cells
        break
    if not blocker_placed:
        return None

    boundary_dir = stem_dirs[2]
    boundary_len = stem_lens[2]
    for _ in range(120):
        boundary_junction = _sample_junction(boundary_dir, boundary_len, h, w)
        if boundary_junction is None:
            return None
        boundary_cells = _t_cells(boundary_junction, boundary_dir, boundary_len)
        if not _can_place(boundary_cells, occupied, h, w):
            continue
        trial = _build_grid(
            objects + [{"junction": boundary_junction, "stem_dir": boundary_dir, "stem_len": boundary_len, "cells": boundary_cells}],
            marker_cols,
            legend_colors,
            h,
            w,
        )
        stop, length = _termination_type(trial, boundary_junction, OPPOSITE[boundary_dir])
        if stop == "edge" and length >= 2:
            objects.append({"junction": boundary_junction, "stem_dir": boundary_dir, "stem_len": boundary_len, "cells": boundary_cells})
            occupied |= boundary_cells
            break
    else:
        return None

    for i in range(3, n_objects):
        stem_dir = stem_dirs[i]
        stem_len = stem_lens[i]
        placed = False
        for _ in range(80):
            junction = _sample_junction(stem_dir, stem_len, h, w)
            if junction is None:
                break
            cells = _t_cells(junction, stem_dir, stem_len)
            if not _can_place(cells, occupied, h, w):
                continue
            placed = True
            objects.append({"junction": junction, "stem_dir": stem_dir, "stem_len": stem_len, "cells": cells})
            occupied |= cells
            break
        if not placed:
            return None

    grid = _build_grid(objects, marker_cols, legend_colors, h, w)

    stop_kinds = Counter()
    for obj in objects:
        stop, length = _termination_type(grid, obj["junction"], OPPOSITE[obj["stem_dir"]])
        if length == 0:
            return None
        stop_kinds[stop] += 1
    if stop_kinds["edge"] == 0 or stop_kinds["object"] == 0:
        return None

    decoded = [_decode_t_component(comp) for comp in components_mask(grid == T_COLOR, connectivity=4)]
    if len(decoded) != len(objects):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid T-ray ARC input")
