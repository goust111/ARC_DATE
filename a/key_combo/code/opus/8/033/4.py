from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
PLUS = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 9]
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _divider_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    divider_rows = [
        r
        for r in range(h)
        if np.all(np.isin(grid[r], [DIVIDER, PLUS])) and np.count_nonzero(grid[r] == DIVIDER) >= w - 3
    ]
    divider_cols = [
        c
        for c in range(w)
        if np.all(np.isin(grid[:, c], [DIVIDER, PLUS])) and np.count_nonzero(grid[:, c] == DIVIDER) >= h - 3
    ]
    return divider_rows, divider_cols


def _segment_bounds(dividers: list[int], size: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    prev = -1
    for d in list(dividers) + [size]:
        if d - prev > 1:
            bounds.append((prev + 1, d - 1))
        prev = d
    return bounds


def _build_region_map(
    shape: tuple[int, int],
    row_bounds: list[tuple[int, int]],
    col_bounds: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    region_map = np.full(shape, -1, dtype=int)
    rid_to_rc: dict[int, tuple[int, int]] = {}
    rid = 0
    for rr, (r0, r1) in enumerate(row_bounds):
        for cc, (c0, c1) in enumerate(col_bounds):
            region_map[r0 : r1 + 1, c0 : c1 + 1] = rid
            rid_to_rc[rid] = (rr, cc)
            rid += 1
    return region_map, rid_to_rc


def _region_main_colors(
    grid: np.ndarray,
    row_bounds: list[tuple[int, int]],
    col_bounds: list[tuple[int, int]],
) -> dict[int, int]:
    colors: dict[int, int] = {}
    rid = 0
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = [int(v) for v in region.flat if int(v) not in (BG, DIVIDER, PLUS)]
            counts = Counter(vals)
            assert counts, "each region must contain at least one non-black non-grey non-plus color"
            top = counts.most_common()
            assert len(top) == 1 or top[0][1] > top[1][1], "region main color must be unique"
            colors[rid] = int(top[0][0])
            rid += 1
    return colors


def _find_pluses(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != PLUS:
                continue
            if not all(int(grid[r + dr, c + dc]) == PLUS for dr, dc in DIRS4):
                continue
            if any(int(grid[r + dr, c + dc]) == PLUS for dr in (-1, 1) for dc in (-1, 1)):
                continue
            centers.append((r, c))
    return centers


def _marker_dirs(grid: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        mr = r + 2 * dr
        mc = c + 2 * dc
        if _in_bounds(grid.shape, mr, mc) and int(grid[mr, mc]) not in (BG, DIVIDER, PLUS):
            out.append((dr, dc))
    return out


def _ray_trace(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    divider_rows: set[int],
    divider_cols: set[int],
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if r not in divider_rows and c not in divider_cols:
            out.append((r, c))
        r += dr
        c += dc
    return out


def _analyze(grid: np.ndarray, *, require_region_colors: bool = True) -> dict:
    divider_rows, divider_cols = _divider_indices(grid)
    row_bounds = _segment_bounds(divider_rows, grid.shape[0])
    col_bounds = _segment_bounds(divider_cols, grid.shape[1])
    region_map, rid_to_rc = _build_region_map(grid.shape, row_bounds, col_bounds)
    main_colors = (
        _region_main_colors(grid, row_bounds, col_bounds) if require_region_colors else {}
    )
    pluses = _find_pluses(grid)
    rays = []
    div_row_set = set(divider_rows)
    div_col_set = set(divider_cols)
    for center in pluses:
        for direction in _marker_dirs(grid, center):
            mr = center[0] + 2 * direction[0]
            mc = center[1] + 2 * direction[1]
            cells = _ray_trace(grid.shape, (mr, mc), direction, div_row_set, div_col_set)
            rays.append({"center": center, "dir": direction, "marker": (mr, mc), "cells": cells})
    return {
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds,
        "region_map": region_map,
        "rid_to_rc": rid_to_rc,
        "main_colors": main_colors,
        "pluses": pluses,
        "rays": rays,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend one-pixel markers away from cyan '+' shapes while recoloring the extension by partition region.

    Input structure:
    - Black background contains grey full-row and full-column divider lines that partition the canvas into rectangles.
    - Each rectangle has scattered pixels of one dominant non-black/non-grey/non-cyan color, which defines that region's main color.
    - One or two cyan 5-pixel '+' objects appear, and some arm tips have a single colored marker two cells from the center.

    Rule:
    - Detect the grey divider rows and columns to recover the rectangular regions.
    - For each region, identify its main color as the unique most frequent non-black/non-grey/non-cyan color inside that rectangle.
    - For each marker adjacent to a '+' arm, extend a one-pixel periodic tiling away from the '+' along that side until the grid edge.
    - Skip divider-line cells and never overwrite existing non-black cells; every new extension pixel takes the main color of the region it lands in.

    Key insight:
    - The marker only supplies direction; the divider partition supplies the color, so the same straight ray changes color exactly when it crosses into a new rectangular region.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    out = g.copy()
    divider_rows = set(info["divider_rows"])
    divider_cols = set(info["divider_cols"])
    region_map = info["region_map"]
    main_colors = info["main_colors"]

    for ray in info["rays"]:
        for r, c in ray["cells"]:
            if r in divider_rows or c in divider_cols:
                continue
            rid = int(region_map[r, c])
            assert rid >= 0
            if int(g[r, c]) == PLUS or int(g[r, c]) == DIVIDER:
                continue
            if int(g[r, c]) == int(main_colors[rid]):
                continue
            out[r, c] = int(main_colors[rid])
    return out


def _choose_spaced_positions(
    size: int,
    count: int,
    rng: np.random.Generator,
    *,
    margin: int = 3,
    min_gap: int = 4,
) -> list[int] | None:
    candidates = list(range(margin, size - margin))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= min_gap for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _assign_region_colors(
    n_rows: int,
    n_cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    colors = np.zeros((n_rows, n_cols), dtype=int)
    for rr in range(n_rows):
        for cc in range(n_cols):
            forbidden = set()
            if rr > 0:
                forbidden.add(int(colors[rr - 1, cc]))
            if cc > 0:
                forbidden.add(int(colors[rr, cc - 1]))
            choices = [c for c in PALETTE if c not in forbidden]
            colors[rr, cc] = int(rng.choice(np.array(choices, dtype=int)))
    return colors


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _candidate_centers(
    shape: tuple[int, int],
    divider_rows: set[int],
    divider_cols: set[int],
    occupied: set[tuple[int, int]],
    centers: list[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    out: list[dict] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if r in divider_rows and c in divider_cols:
                continue
            if any(max(abs(r - er), abs(c - ec)) < 4 for er, ec in centers):
                continue
            plus_cells = _plus_cells((r, c))
            if plus_cells & occupied:
                continue
            marker_dirs = []
            for dr, dc in DIRS4:
                mr, mc = r + 2 * dr, c + 2 * dc
                if not _in_bounds(shape, mr, mc):
                    continue
                if mr in divider_rows or mc in divider_cols:
                    continue
                if (mr, mc) in occupied or (mr, mc) in plus_cells:
                    continue
                marker_dirs.append((dr, dc))
            if len(marker_dirs) < 2:
                continue
            touches_divider = any(pr in divider_rows or pc in divider_cols for pr, pc in plus_cells)
            out.append(
                {
                    "center": (r, c),
                    "plus_cells": plus_cells,
                    "dirs": marker_dirs,
                    "touches_divider": touches_divider,
                }
            )
    return out


def _pick_pluses(
    shape: tuple[int, int],
    divider_rows: list[int],
    divider_cols: list[int],
    rng: np.random.Generator,
) -> list[dict] | None:
    divider_row_set = set(divider_rows)
    divider_col_set = set(divider_cols)
    plus_count = int(rng.choice(np.array([1, 2, 2], dtype=int)))
    occupied: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    placed: list[dict] = []
    need_touch = bool(divider_rows or divider_cols) and bool(rng.random() < 0.7)

    for idx in range(plus_count):
        candidates = _candidate_centers(shape, divider_row_set, divider_col_set, occupied, centers)
        if not candidates:
            return None
        if need_touch and idx == 0:
            touching = [cand for cand in candidates if cand["touches_divider"]]
            if touching:
                candidates = touching
        chosen = candidates[int(rng.integers(len(candidates)))]
        k = 2 if len(chosen["dirs"]) == 2 else int(rng.integers(2, len(chosen["dirs"]) + 1))
        dirs = list(chosen["dirs"])
        rng.shuffle(dirs)
        marker_dirs = dirs[:k]
        marker_cells = {(chosen["center"][0] + 2 * dr, chosen["center"][1] + 2 * dc) for dr, dc in marker_dirs}
        occupied |= chosen["plus_cells"] | marker_cells
        centers.append(chosen["center"])
        placed.append({"center": chosen["center"], "dirs": marker_dirs, "plus_cells": chosen["plus_cells"]})
    return placed


def _select_blockers(
    rays: list[dict],
    grid: np.ndarray,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    if not rays or rng.random() >= 0.6:
        return set()
    candidates = []
    for ray in rays:
        cells = ray["cells"]
        if len(cells) >= 3:
            for pos in cells[1:-1]:
                if int(grid[pos]) == BG:
                    candidates.append(pos)
    if not candidates:
        return set()
    rng.shuffle(candidates)
    out: set[tuple[int, int]] = set()
    target = 1 if len(candidates) == 1 else int(rng.choice(np.array([1, 2], dtype=int)))
    for pos in candidates:
        out.add(pos)
        if len(out) >= target:
            break
    return out


def _semantic_check(grid: np.ndarray) -> tuple[bool, str]:
    try:
        info = _analyze(grid)
        out = main(grid)
    except Exception as exc:
        return False, f"analyze/main failed: {exc}"

    if not info["pluses"] or len(info["pluses"]) > 2:
        return False, "wrong plus count"

    divider_rows = set(info["divider_rows"])
    divider_cols = set(info["divider_cols"])
    region_map = info["region_map"]
    main_colors = info["main_colors"]

    occupied = set()
    for ray in info["rays"]:
        center = ray["center"]
        dirs = _marker_dirs(grid, center)
        if len(dirs) < 2:
            return False, "plus with fewer than 2 markers"
        occupied.add(ray["marker"])

    all_ray_cells: set[tuple[int, int]] = set()
    changed = 0
    crosses_region = False
    blocker_seen = False
    for ray in info["rays"]:
        cells = ray["cells"]
        region_seq = [int(region_map[r, c]) for r, c in cells]
        if len(set(region_seq)) >= 2:
            crosses_region = True
        for cell in cells:
            if cell in all_ray_cells:
                return False, "intersecting rays"
            all_ray_cells.add(cell)
            r, c = cell
            if r in divider_rows or c in divider_cols:
                continue
            expect = int(main_colors[int(region_map[r, c])])
            if int(grid[r, c]) in (BG,):
                if int(out[r, c]) != expect:
                    return False, "wrong painted color"
                changed += 1
            elif int(grid[r, c]) == expect:
                if int(out[r, c]) != int(grid[r, c]):
                    return False, "protected pixel overwritten"
                blocker_seen = True
            else:
                if int(out[r, c]) != expect:
                    return False, "marker-colored path cell not repainted"
                changed += 1

    if changed == 0:
        return False, "no output change"
    if not crosses_region:
        return False, "no ray crosses a region boundary"
    if len(set(map(int, out.flat)) - {BG, DIVIDER, PLUS}) < 2:
        return False, "output too color-poor"
    if np.array_equal(grid, out):
        return False, "identity transform"
    return True, "ok"


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(17, 23))
    w = int(rng.integers(17, 23))
    n_h = int(rng.integers(1, 4))
    n_v = int(rng.integers(1, 4))
    divider_rows = _choose_spaced_positions(h, n_h, rng, margin=3, min_gap=4)
    divider_cols = _choose_spaced_positions(w, n_v, rng, margin=3, min_gap=4)
    if divider_rows is None or divider_cols is None:
        return None

    row_bounds = _segment_bounds(divider_rows, h)
    col_bounds = _segment_bounds(divider_cols, w)
    if min(r1 - r0 + 1 for r0, r1 in row_bounds) < 4:
        return None
    if min(c1 - c0 + 1 for c0, c1 in col_bounds) < 4:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[divider_rows, :] = DIVIDER
    grid[:, divider_cols] = DIVIDER

    region_colors = _assign_region_colors(len(row_bounds), len(col_bounds), rng)
    region_map, rid_to_rc = _build_region_map(grid.shape, row_bounds, col_bounds)
    region_main_by_id = {
        rid: int(region_colors[rr, cc]) for rid, (rr, cc) in rid_to_rc.items()
    }

    pluses = _pick_pluses(grid.shape, divider_rows, divider_cols, rng)
    if pluses is None:
        return None

    plus_cells: set[tuple[int, int]] = set()
    marker_cells: set[tuple[int, int]] = set()
    for plus in pluses:
        for r, c in plus["plus_cells"]:
            grid[r, c] = PLUS
            plus_cells.add((r, c))
        for dr, dc in plus["dirs"]:
            mr = plus["center"][0] + 2 * dr
            mc = plus["center"][1] + 2 * dc
            grid[mr, mc] = int(rng.choice(np.array(PALETTE, dtype=int)))
            marker_cells.add((mr, mc))

    info = _analyze(grid, require_region_colors=False)
    all_paths: set[tuple[int, int]] = set()
    touches_crossing = False
    for ray in info["rays"]:
        if len(ray["cells"]) < 2:
            return None
        region_ids = [int(info["region_map"][r, c]) for r, c in ray["cells"]]
        touches_crossing = touches_crossing or len(set(region_ids)) >= 2
        for cell in ray["cells"]:
            if cell in plus_cells or cell in marker_cells or cell in all_paths:
                return None
            all_paths.add(cell)
    if not touches_crossing:
        return None

    blockers = _select_blockers(info["rays"], grid, rng)
    region_forced: dict[int, set[tuple[int, int]]] = {}
    for pos in blockers:
        rid = int(region_map[pos])
        region_forced.setdefault(rid, set()).add(pos)

    free_by_region: dict[int, list[tuple[int, int]]] = {}
    marker_counts_by_region: dict[int, Counter[int]] = {}
    for rid in region_main_by_id:
        free_by_region[rid] = []
        marker_counts_by_region[rid] = Counter()

    for r0, c0 in marker_cells:
        rid = int(region_map[r0, c0])
        marker_counts_by_region[rid][int(grid[r0, c0])] += 1

    for r in range(h):
        for c in range(w):
            if r in divider_rows or c in divider_cols:
                continue
            if (r, c) in plus_cells or (r, c) in marker_cells:
                continue
            rid = int(region_map[r, c])
            free_by_region[rid].append((r, c))

    for rid, main_color in region_main_by_id.items():
        main_color = int(main_color)
        forced = set(region_forced.get(rid, set()))
        free = free_by_region[rid]
        if len(free) < max(5, len(forced)):
            return None
        other_counts = [count for color, count in marker_counts_by_region[rid].items() if int(color) != main_color]
        need = max(2, len(forced), (max(other_counts) + 1) if other_counts else 2)
        target = max(need, int(rng.integers(2, 6)))
        if target > 5:
            return None
        available = [pos for pos in free if pos not in forced]
        rng.shuffle(available)
        if len(available) + len(forced) < target:
            return None
        chosen = list(forced) + available[: target - len(forced)]
        for r, c in chosen:
            grid[r, c] = main_color

    ok, _msg = _semantic_check(grid)
    if not ok:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
