from __future__ import annotations

import random

import numpy as np


MOVE_LEN = 2
BG_COLOR = 5
FG_COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_BOUNDS = (12, 19)
GRID_W_BOUNDS = (12, 19)
PAIR_COUNT_BOUNDS = (1, 4)
TARGET_AREA_BOUNDS = (1, 6)
RAY_GAP_BOUNDS = (2, 6)

GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250
POLYOMINO_ATTEMPTS = 80

DIR_CONFIGS = (
    {"dir": (-1, -1), "missing": (0, 0)},
    {"dir": (-1, 1), "missing": (0, 1)},
    {"dir": (1, -1), "missing": (1, 0)},
    {"dir": (1, 1), "missing": (1, 1)},
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _arrow_info(comp: dict[str, object]) -> dict[str, object] | None:
    coords = np.asarray(comp["coords"], dtype=int)
    if coords.shape[0] != 3:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None

    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    full = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = full - local
    if len(missing) != 1:
        return None

    mr, mc = next(iter(missing))
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    return {
        "dir": (dr, dc),
        "missing_local": (mr, mc),
        "missing_global": (int(r0 + mr), int(c0 + mc)),
    }


def _in_bounds(shape: tuple[int, int], coord: tuple[int, int]) -> bool:
    r, c = coord
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_target(
    grid: np.ndarray,
    bg: int,
    cell_to_comp: np.ndarray,
    arrow_ids: set[int],
    comps: list[dict[str, object]],
    arrow: dict[str, object],
) -> int:
    r, c = arrow["missing_global"]
    dr, dc = arrow["dir"]
    while True:
        r += dr
        c += dc
        if not _in_bounds(grid.shape, (r, c)):
            break
        if int(grid[r, c]) == int(bg):
            continue
        comp_idx = int(cell_to_comp[r, c])
        if comp_idx in arrow_ids:
            continue
        if int(comps[comp_idx]["color"]) == int(arrow["color"]):
            continue
        return comp_idx
    raise AssertionError("every arrow must see a non-arrow target on its diagonal ray")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect 2x2-minus-one-corner arrow objects, find the nearest target on each arrow's diagonal ray,
    move that target two cells in the arrow's pointing direction, and erase the arrows.

    The arrow direction is the missing-corner direction of the 3-cell L shape, so the four
    possible directions are NW, NE, SW, and SE.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one foreground object"

    cell_to_comp = np.full(g.shape, -1, dtype=int)
    arrows: list[dict[str, object]] = []
    arrow_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        coords = np.asarray(comp["coords"], dtype=int)
        cell_to_comp[coords[:, 0], coords[:, 1]] = idx
        info = _arrow_info(comp)
        if info is None:
            continue
        info["index"] = idx
        info["coords"] = coords
        info["color"] = int(comp["color"])
        arrows.append(info)
        arrow_ids.add(idx)

    assert arrows, "expected at least one arrow"
    arrow_colors = [int(arrow["color"]) for arrow in arrows]
    assert len(set(arrow_colors)) == len(arrow_colors), "arrow colors must be unique"

    assignments: list[tuple[dict[str, object], int]] = []
    claimed_targets: set[int] = set()
    for arrow in arrows:
        target_idx = _trace_target(g, bg, cell_to_comp, arrow_ids, comps, arrow)
        assert target_idx not in claimed_targets, "each target must belong to exactly one arrow"
        claimed_targets.add(target_idx)
        assignments.append((arrow, target_idx))

    out = g.copy()
    for arrow in arrows:
        coords = np.asarray(arrow["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    for target_idx in claimed_targets:
        coords = np.asarray(comps[target_idx]["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = out != bg
    for arrow, target_idx in assignments:
        dr, dc = arrow["dir"]
        delta = np.array([MOVE_LEN * dr, MOVE_LEN * dc], dtype=int)
        moved = np.asarray(comps[target_idx]["coords"], dtype=int) + delta
        assert moved[:, 0].min() >= 0 and moved[:, 0].max() < g.shape[0]
        assert moved[:, 1].min() >= 0 and moved[:, 1].max() < g.shape[1]
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "moved targets must not overlap"
        out[moved[:, 0], moved[:, 1]] = int(comps[target_idx]["color"])
        occupied[moved[:, 0], moved[:, 1]] = True

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _transform_cells(cells: tuple[tuple[int, int], ...], rot_k: int, flip_lr: bool) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    if flip_lr:
        arr[:, 1] *= -1
    for _ in range(rot_k % 4):
        arr = np.column_stack((-arr[:, 1], arr[:, 0]))
    return _normalize_cells({(int(r), int(c)) for r, c in arr})


def _random_target_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(POLYOMINO_ATTEMPTS):
        area = random.randint(*TARGET_AREA_BOUNDS)
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            cells.add((r + dr, c + dc))
        shape = _transform_cells(_normalize_cells(cells), random.randint(0, 3), bool(random.getrandbits(1)))
        arr = np.array(shape, dtype=int)
        h = int(arr[:, 0].max()) + 1
        w = int(arr[:, 1].max()) + 1
        if h > 4 or w > 4:
            continue
        if len(shape) == 3 and h == 2 and w == 2:
            continue
        return shape
    raise RuntimeError("failed to build a non-arrow target shape")


def _touches(existing: set[tuple[int, int]], new_cells: set[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in existing:
                    return True
    return False


def _coords_from_shape(shape: tuple[tuple[int, int], ...], shift_r: int, shift_c: int) -> set[tuple[int, int]]:
    return {(int(shift_r + r), int(shift_c + c)) for r, c in shape}


def _arrow_cells(top_r: int, top_c: int, missing: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set()
    for rr in (0, 1):
        for cc in (0, 1):
            if (rr, cc) != missing:
                cells.add((int(top_r + rr), int(top_c + cc)))
    return cells


def _build_pair(
    h: int,
    w: int,
    cfg: dict[str, object],
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    reserved_in: set[tuple[int, int]],
) -> dict[str, object] | None:
    dr, dc = cfg["dir"]
    mr, mc = cfg["missing"]
    move_r = MOVE_LEN * dr
    move_c = MOVE_LEN * dc

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        shape = _random_target_shape()
        arr = np.array(shape, dtype=int)
        max_r = int(arr[:, 0].max())
        max_c = int(arr[:, 1].max())
        anchor_candidates = [cell for cell in shape if (cell[0] - dr, cell[1] - dc) not in shape]
        anchor_r, anchor_c = random.choice(anchor_candidates)
        gap = random.randint(*RAY_GAP_BOUNDS)

        shift_r_lo = max(0, -move_r)
        shift_r_hi = min(h - 1 - max_r, h - 1 - move_r - max_r)
        shift_c_lo = max(0, -move_c)
        shift_c_hi = min(w - 1 - max_c, w - 1 - move_c - max_c)
        if shift_r_lo > shift_r_hi or shift_c_lo > shift_c_hi:
            continue

        anchor_global_r_lo = anchor_r + shift_r_lo
        anchor_global_r_hi = anchor_r + shift_r_hi
        anchor_global_c_lo = anchor_c + shift_c_lo
        anchor_global_c_hi = anchor_c + shift_c_hi

        anchor_global_r_lo = max(anchor_global_r_lo, gap * dr + mr)
        anchor_global_r_hi = min(anchor_global_r_hi, h - 2 + gap * dr + mr)
        anchor_global_c_lo = max(anchor_global_c_lo, gap * dc + mc)
        anchor_global_c_hi = min(anchor_global_c_hi, w - 2 + gap * dc + mc)
        if anchor_global_r_lo > anchor_global_r_hi or anchor_global_c_lo > anchor_global_c_hi:
            continue

        anchor_global_r = random.randint(anchor_global_r_lo, anchor_global_r_hi)
        anchor_global_c = random.randint(anchor_global_c_lo, anchor_global_c_hi)
        shift_r = anchor_global_r - anchor_r
        shift_c = anchor_global_c - anchor_c

        target_cells = _coords_from_shape(shape, shift_r, shift_c)
        moved_cells = {(r + move_r, c + move_c) for r, c in target_cells}
        miss_r = anchor_global_r - gap * dr
        miss_c = anchor_global_c - gap * dc
        arrow_top_r = miss_r - mr
        arrow_top_c = miss_c - mc
        arrow_cells = _arrow_cells(arrow_top_r, arrow_top_c, cfg["missing"])
        corridor = {(miss_r + step * dr, miss_c + step * dc) for step in range(1, gap)}

        if not all(_in_bounds((h, w), cell) for cell in target_cells | moved_cells | arrow_cells | corridor):
            continue
        if target_cells & reserved_in or arrow_cells & reserved_in:
            continue
        if corridor & occupied_in:
            continue
        if target_cells & occupied_in or arrow_cells & occupied_in:
            continue
        if _touches(occupied_in, target_cells | arrow_cells):
            continue
        if _touches(target_cells, arrow_cells):
            continue
        if moved_cells & occupied_out:
            continue
        if _touches(occupied_out, moved_cells):
            continue

        return {
            "target_cells": target_cells,
            "moved_cells": moved_cells,
            "arrow_cells": arrow_cells,
            "corridor": corridor,
            "dir": (dr, dc),
        }
    return None


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_BOUNDS)
    w = random.randint(*GRID_W_BOUNDS)
    pair_count = random.randint(*PAIR_COUNT_BOUNDS)
    if 2 * pair_count > len(FG_COLORS):
        return None

    colors = random.sample(FG_COLORS, 2 * pair_count)
    arrow_colors = colors[:pair_count]
    target_colors = colors[pair_count:]
    dir_cfgs = random.sample(DIR_CONFIGS, pair_count)

    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    reserved_in: set[tuple[int, int]] = set()
    pairs: list[dict[str, object]] = []

    for idx in range(pair_count):
        pair = _build_pair(h, w, dir_cfgs[idx], occupied_in, occupied_out, reserved_in)
        if pair is None:
            return None
        pair["arrow_color"] = arrow_colors[idx]
        pair["target_color"] = target_colors[idx]
        pairs.append(pair)
        occupied_in.update(pair["target_cells"])
        occupied_in.update(pair["arrow_cells"])
        occupied_out.update(pair["moved_cells"])
        reserved_in.update(pair["corridor"])

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for pair in pairs:
        for r, c in pair["target_cells"]:
            grid[r, c] = int(pair["target_color"])
        for r, c in pair["arrow_cells"]:
            grid[r, c] = int(pair["arrow_color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-to-target grid")
