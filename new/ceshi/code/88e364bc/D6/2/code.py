# Variant of 88e364bc
# Dimension: D6 — 区域边界形状 (obstacle)
# Change: 不规则闭合边界——1像素厚的彩色墙壁形成有台阶/凹槽的不规则封闭区域（填充率0.47-0.75），区域内部12-97格... → 嵌套区域（区域内有子区域）
#
# Design analysis
# - Source-task note: the requested reference task `13e47133` is available here only as
#   `13e47133.json`; no solver file exists in the allowed local resources, so the nested
#   mechanism below is derived from the "concentric frames" idea rather than copied code.
# - a) Concrete behavior:
#   Input now always contains one irregular outer region and one irregular child region inside it.
#   The two walls use different colors, and each wall color has its own border legend. Yellow
#   cells in the annulus between the two walls follow the outer legend; yellow cells inside the
#   child region follow the inner legend. Example trace:
#   original logic on the annulus = count adjacent wall colors and take the majority;
#   variant logic on the annulus = choose the smallest enclosing wall loop that still contains the
#   whole passable component. For the annulus that is the outer wall, while for the child region it
#   is the inner wall. This is the only semantic change; sliding physics stay unchanged.
# - b) Complexity calibration:
#   The generator fixes exactly one nested pair so the hierarchical cue is always visible. The two
#   directions are forced to differ, and the annulus is forced to touch both wall colors, so the
#   new behavior is repeatedly supported instead of appearing as a rare corner case.
# - c) Hidden assumptions in the original main():
#   `passable` and `_slide_points` remain valid because movement inside a single connected passable
#   component is unchanged. The fragile part is `border_col = max(border_counts, key=...)`, which
#   assumed each enclosed component touched only one meaningful wall color. In the nested variant
#   the annulus touches both the outer and inner walls, so majority vote can pick the wrong one.
#   The solver therefore replaces majority-vote ownership with explicit wall-containment analysis.
# - d) Placement legality:
#   Legends are buffered away from the play area. The inner blob is grown inside an erosion of the
#   outer blob so its wall fits completely inside the outer region. The generator rejects any sample
#   where the annulus disconnects, where expected enclosed components are missing, or where the new
#   solver cannot reproduce the intended result.
# - e) Visual observability:
#   Every sample has two legends, two wall colors, and yellow pixels in both levels. Because the
#   directions differ, the solver must treat the two enclosed components differently in the same
#   grid, making the hierarchical rule visible from a few examples.
# - f) Visual neatness:
#   Only one nested pair is used. This avoids clutter while still making the two levels obvious.
#   The child wall is fully inside the outer region with at least one ring of free space around it.
# - g) Color inference:
#   Colors are still used only as visible keys: legend fill color matches a wall color. No ordering
#   depends on numeric color ids.
# - 9) New solver assumptions and generator guarantees:
#   The solver assumes every movable enclosed component is inside at least one closed wall loop whose
#   color has a legend. The generator guarantees this by constructing the outer/inner loops first and
#   by validating that the enclosed background components are exactly the annulus and the child area.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains at least
    two additional colors: a "center" color (more frequent) and an "indicator" color (less frequent).

    The movement direction for regions with border color `C` is:
      sign(centroid(indicator) - centroid(center))
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        others.sort()
        if len(others) < 2:
            continue
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cents = []
            for _, col in others[:2]:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _inside_single_wall(wall_cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    """Cells enclosed by one wall loop when only that loop is treated as solid."""
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall_cells:
        wall_mask[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not wall_mask[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall_mask[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or wall_mask[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not wall_mask[nr, nc]:
                stack.append((nr, nc))

    inside = set()
    for r in range(h):
        for c in range(w):
            if not wall_mask[r, c] and not reachable[r, c]:
                inside.add((r, c))
    return inside


def _wall_infos(g: np.ndarray, dirs: dict[int, tuple[int, int]], bg: int) -> list[dict]:
    """Collect non-border wall loops and the cells each loop encloses."""
    h, w = g.shape
    infos: list[dict] = []
    for color in dirs:
        for pts in components_mask(g == color, connectivity=8):
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
                continue
            cells = {(int(r), int(c)) for r, c in pts}
            inside = _inside_single_wall(cells, h, w)
            if not inside:
                continue
            infos.append(
                {
                    "color": color,
                    "cells": cells,
                    "inside": inside,
                    "inside_area": len(inside),
                }
            )
    return infos


def _fallback_border_color(
    g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int, mover: int
) -> int | None:
    border_counts: dict[int, int] = {}
    h, w = g.shape
    for r, c in pts:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in pset:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: enclosed passable components use the direction of their nearest enclosing wall.

    With nesting, a passable component can be enclosed by multiple wall loops. The annulus between
    the two loops belongs to the outer wall, while the child region belongs to the inner wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"
    walls = _wall_infos(g, dirs, bg)

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        candidates = [info for info in walls if pset <= info["inside"]]
        if candidates:
            owner = min(candidates, key=lambda info: (info["inside_area"], info["color"]))
            border_col = int(owner["color"])
        else:
            border_col = _fallback_border_color(g, pts, pset, bg, mover)
            if border_col is None:
                continue
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
CORNER_PAIRS = [("TL", "TR"), ("BL", "BR"), ("TL", "BL"), ("TR", "BR")]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

OUTER_SIZE = (60, 92)
INNER_SIZE = (14, 34)
MIN_ANNULUS_SIZE = 12
INNER_MARGIN = 2
BLOB_GROW_RETRIES = 30
PAIR_RETRIES = 80

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 400


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier: list[tuple[int, int]] = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _try_grow_valid_blob(
    rng: random.Random, canvas: np.ndarray, occupied: np.ndarray, target_size: int, h: int, w: int
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _erode_blob(blob: set[tuple[int, int]], steps: int) -> set[tuple[int, int]]:
    cur = set(blob)
    for _ in range(steps):
        nxt = set()
        for r, c in cur:
            if all((nr, nc) in cur for nr, nc in neighbors4(r, c)):
                nxt.add((r, c))
        cur = nxt
        if not cur:
            break
    return cur


def _region_adj_counts(
    region: set[tuple[int, int]], outer_wall: set[tuple[int, int]], inner_wall: set[tuple[int, int]]
) -> tuple[int, int]:
    outer_adj = 0
    inner_adj = 0
    for r, c in region:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in outer_wall:
                outer_adj += 1
            elif (nr, nc) in inner_wall:
                inner_adj += 1
    return outer_adj, inner_adj


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _add_outer_extension(
    blob: set[tuple[int, int]], r0: int, c0: int, h: int, w: int, side: str, span0: int, span: int, depth: int
) -> None:
    if side == "top":
        blob |= _rect_cells(r0 - depth, c0 + span0, depth, span)
    elif side == "bottom":
        blob |= _rect_cells(r0 + h, c0 + span0, depth, span)
    elif side == "left":
        blob |= _rect_cells(r0 + span0, c0 - depth, span, depth)
    elif side == "right":
        blob |= _rect_cells(r0 + span0, c0 + w, span, depth)


def _make_inner_jagged_blob(r0: int, c0: int, h: int, w: int, rng: random.Random) -> set[tuple[int, int]]:
    blob = _rect_cells(r0, c0, h, w)

    top_cols = [c for c in range(c0 + 1, c0 + w - 1, 2)]
    bot_cols = [c for c in range(c0 + (0 if len(top_cols) < 2 else 2), c0 + w - 1, 2)]
    left_rows = [r for r in range(r0 + 1, r0 + h - 1, 2)]
    right_rows = [r for r in range(r0 + (0 if len(left_rows) < 2 else 2), r0 + h - 1, 2)]

    rng.shuffle(top_cols)
    rng.shuffle(bot_cols)
    rng.shuffle(left_rows)
    rng.shuffle(right_rows)

    for c in top_cols[: min(1, len(top_cols))]:
        blob.add((r0 - 1, c))
    for c in bot_cols[: min(1, len(bot_cols))]:
        blob.add((r0 + h, c))
    for r in left_rows[: min(1, len(left_rows))]:
        blob.add((r, c0 - 1))
    for r in right_rows[: min(1, len(right_rows))]:
        blob.add((r, c0 + w))

    return blob


def _validate_expected_components(grid: np.ndarray, expected_regions: list[set[tuple[int, int]]]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    if len(bounded) != len(expected_regions):
        return False
    return all(comp in expected_regions for comp in bounded)


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_region_yellows(rng: random.Random, region: dict, grid: np.ndarray) -> bool:
    blob = region["interior"]
    dr, dc = region["dir"]
    n_yellows = rng.choice(YELLOWS_PER_REGION)

    cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
    far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
    tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

    first = tier1[rng.randrange(len(tier1))]
    output_positions = [first]
    if n_yellows == 2:
        pr, pc = first[0] - dr, first[1] - dc
        cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
        if cands:
            second = cands[rng.randrange(len(cands))]
            if second != first:
                output_positions.append(second)

    out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
    already_stopped: set[tuple[int, int]] = set()
    yellows_input: list[tuple[int, int]] = []
    for out_pos in out_sorted:
        candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
        if candidate is None:
            return False
        yellows_input.append(candidate)
        already_stopped.add(out_pos)

    for r, c in yellows_input:
        grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    regions_mask = np.zeros(input_grid.shape, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


def _random_direction(rng: random.Random) -> tuple[int, int]:
    if rng.random() < CARDINAL_PROB:
        return rng.choice(CARDINAL_DIRS)
    return rng.choice(DIAGONAL_DIRS)


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = grid.shape
    corners = list(rng.choice(CORNER_PAIRS))
    outer_color, inner_color = rng.sample(WALL_COLOR_CHOICES, 2)
    outer_dir = _random_direction(rng)
    inner_dir = rng.choice([d for d in ALL_DIRS if d != outer_dir])

    legend_occupied = np.zeros((h, w), dtype=bool)
    for corner, color, direction in (
        (corners[0], outer_color, outer_dir),
        (corners[1], inner_color, inner_dir),
    ):
        dr, dc = direction
        patch = _legend_5x5(color, dr, dc) if direction in CARDINAL_DIRS else _legend_6x7(color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True

    return legend_occupied, {
        "outer_color": outer_color,
        "inner_color": inner_color,
        "outer_dir": outer_dir,
        "inner_dir": inner_dir,
    }


def _build_nested_regions(rng: random.Random, legend_occupied: np.ndarray, legend_info: dict) -> list[dict] | None:
    h, w = legend_occupied.shape
    for _ in range(PAIR_RETRIES):
        outer_h = rng.randint(8, 10)
        outer_w = rng.randint(8, 10)
        if rng.random() < 0.9:
            side = rng.choice(["top", "bottom", "left", "right"])
            depth = 1
            if side in ("top", "bottom"):
                span = rng.randint(2, max(2, outer_w - 3))
                span0 = rng.randint(1, outer_w - span - 1)
            else:
                span = rng.randint(2, max(2, outer_h - 3))
                span0 = rng.randint(1, outer_h - span - 1)
        else:
            side = "top"
            span0 = 1
            span = 2
            depth = 0

        outer_candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]], int, int]] = []
        for r0 in range(2, h - outer_h - 1):
            for c0 in range(2, w - outer_w - 1):
                outer_blob = _rect_cells(r0, c0, outer_h, outer_w)
                if depth:
                    _add_outer_extension(outer_blob, r0, c0, outer_h, outer_w, side, span0, span, depth)
                outer_wall = _blob_wall_cells(outer_blob, h, w)
                if any(rr in (0, h - 1) or cc in (0, w - 1) or legend_occupied[rr, cc] for rr, cc in outer_wall):
                    continue
                if any(legend_occupied[rr, cc] for rr, cc in outer_blob):
                    continue
                outer_candidates.append((outer_blob, outer_wall, r0, c0))
        if not outer_candidates:
            continue
        outer_blob, outer_wall, r0, c0 = rng.choice(outer_candidates)

        inner_h = rng.randint(3, 4)
        inner_w = rng.randint(4, 5)
        if outer_h < inner_h + 4 or outer_w < inner_w + 4:
            continue
        inner_candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]] = []
        for ir0 in range(r0 + 2, r0 + outer_h - inner_h - 1):
            for ic0 in range(c0 + 2, c0 + outer_w - inner_w - 1):
                for _variant in range(3):
                    inner_blob = _make_inner_jagged_blob(ir0, ic0, inner_h, inner_w, rng)
                    inner_wall = _blob_wall_cells(inner_blob, h, w)
                    if any(cell not in outer_blob for cell in inner_wall):
                        continue
                    if not all(cell in outer_blob for cell in inner_blob):
                        continue

                    annulus = outer_blob - inner_blob - inner_wall
                    if len(annulus) < MIN_ANNULUS_SIZE or len(inner_blob) < INNER_SIZE[0]:
                        continue

                    annulus_mask = np.zeros((h, w), dtype=bool)
                    for rr, cc in annulus:
                        annulus_mask[rr, cc] = True
                    if len(components_mask(annulus_mask, connectivity=4)) != 1:
                        continue

                    outer_adj, inner_adj = _region_adj_counts(annulus, outer_wall, inner_wall)
                    if outer_adj == 0 or inner_adj == 0:
                        continue

                    grid = np.zeros((h, w), dtype=int)
                    for rr, cc in outer_wall:
                        grid[rr, cc] = int(legend_info["outer_color"])
                    for rr, cc in inner_wall:
                        grid[rr, cc] = int(legend_info["inner_color"])
                    if not _validate_expected_components(grid, [annulus, inner_blob]):
                        continue
                    inner_candidates.append((inner_blob, inner_wall, annulus))
        if not inner_candidates:
            continue

        inner_blob, inner_wall, annulus = rng.choice(inner_candidates)
        return [
            {
                "name": "outer",
                "color": legend_info["outer_color"],
                "dir": legend_info["outer_dir"],
                "interior": annulus,
                "walls": outer_wall,
            },
            {
                "name": "inner",
                "color": legend_info["inner_color"],
                "dir": legend_info["inner_dir"],
                "interior": inner_blob,
                "walls": inner_wall,
            },
        ]
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legend_info = _place_legends(rng, grid)
    regions = _build_nested_regions(rng, legend_occupied, legend_info)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["color"])

    if not _validate_expected_components(grid, [reg["interior"] for reg in regions]):
        return None

    input_grid = grid.copy()
    for reg in regions:
        if not _place_region_yellows(rng, reg, input_grid):
            return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
