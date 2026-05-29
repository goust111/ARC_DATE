# Combo variant of 88e364bc (index 12)
# combo_spec: D6/2+D20/2
# Name: 嵌套区域+包含关系
#
# Design analysis
# - a) Concrete behavior:
#   The grid contains one irregular outer wall and one irregular child wall inside it. Each wall
#   color has its own border legend. Yellow cells in the annulus between the two walls slide using
#   the outer legend, while yellow cells inside the child wall slide using the child legend. The
#   original solver used a local majority of adjacent wall colors. That breaks here: the annulus can
#   touch the child wall more often than the outer wall even though it is not inside the child
#   region. The fused rule therefore uses containment depth, not adjacency frequency, to choose the
#   controlling legend.
# - b) Complexity calibration:
#   The generator always emits exactly one nested parent-child pair. That is the smallest readable
#   layout that still forces the hierarchy rule. Both levels contain yellow movers, and the two
#   directions are always different, so the examples visibly require two separate decisions.
# - c) Hidden assumptions in the original main():
#   `passable = (g == bg) | (g == mover)` still works because sliding happens inside connected
#   passable components. `_slide_points()` also remains correct because the movement physics are
#   unchanged. The broken assumption is `border_col = max(border_counts, key=...)`: once a component
#   can border both an outer wall and a nested inner wall, local adjacency no longer identifies the
#   owner. The new solver replaces that line with explicit wall-containment analysis and a recursive
#   descent over the wall tree.
# - d) Placement legality:
#   Legends are buffered away from the play area. The child blob is placed strictly inside the outer
#   blob so its wall never overlaps the outer wall or the legends. The generator validates that the
#   only enclosed background components are the outer annulus and the inner region, that both are
#   connected, and that the reverse-slid input reproduces the intended output.
# - e) Visual observability:
#   Every sample contains two legends, two wall colors, and yellow cells in both levels. The annulus
#   is forced to touch both walls, and its adjacent-child-wall count is forced to exceed its
#   adjacent-outer-wall count. A human can therefore see that “nearest contained region” is the only
#   rule that fits the examples, because a naive local-wall heuristic would disagree.
# - f) Visual neatness:
#   Only one nested pair is used, leaving clear empty space between legends and the play area. The
#   inner wall is jagged but compact, so the hierarchy is visible without turning the 20x20 board
#   into a maze.
# - g) Inferability of the change:
#   Color ids are only used as visible keys between a legend fill and a wall loop. No logic depends
#   on numeric color ordering. The “priority” introduced by D20/2 is encoded structurally by
#   containment depth: the deepest enclosing wall that still contains a component wins.
# - 9) New solver assumptions and generator guarantees:
#   The solver assumes each movable enclosed component is fully contained in at least one closed wall
#   loop whose color has a legend, and that nested loops form a proper inclusion tree rather than
#   crossing. The generator guarantees this by constructing the parent wall first, fitting the child
#   wall entirely inside it, and rejecting any sample where containment ownership or the fused
#   behavior is ambiguous.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


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
PAIR_RETRIES = 80
GEN_RETRIES = 400
MAX_SLIDE_STEPS = 5
YELLOWS_PER_REGION = [1, 2]

OUTER_TEMPLATE = {(r, c) for r in range(9) for c in range(9)} - {(0, 0)}
INNER_TEMPLATE = {
    (2, 3),
    (2, 5),
    (2, 6),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (6, 3),
    (6, 4),
    (6, 5),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 5),
    (7, 6),
}
SYMMETRIES = ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v"]


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and map fill color -> movement direction.

    The fill color is the wall color controlled by that legend. Direction is the sign of the
    center-to-indicator centroid offset.
    """
    h, w = g.shape
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            rank = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                rank.append((np.sum((pts_col.mean(axis=0) - rect_center) ** 2), col))
            rank.sort()
            center_col = rank[0][1]
            indicator_col = rank[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue
        vr, vc = (indicator.mean(axis=0) - center.mean(axis=0)).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), (dr, dc))
    return {color: direction for color, (_size, direction) in best.items()}


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
    for r0, c0 in ordered:
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
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall_cells:
        wall_mask[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
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


def _wall_infos(g: np.ndarray, dirs: dict[int, tuple[int, int]]) -> list[dict]:
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
                    "color": int(color),
                    "cells": cells,
                    "inside": inside,
                    "inside_area": len(inside),
                    "children": [],
                    "parent": None,
                }
            )

    for idx, info in enumerate(infos):
        parent_idx = None
        for j, other in enumerate(infos):
            if idx == j:
                continue
            if info["inside"] < other["inside"]:
                if parent_idx is None or other["inside_area"] < infos[parent_idx]["inside_area"]:
                    parent_idx = j
        info["parent"] = parent_idx
    for idx, info in enumerate(infos):
        parent_idx = info["parent"]
        if parent_idx is not None:
            infos[parent_idx]["children"].append(idx)
    for info in infos:
        info["children"].sort(key=lambda j: infos[j]["inside_area"])
    return infos


def _adjacent_wall_counts(
    g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int, mover: int
) -> dict[int, int]:
    counts: dict[int, int] = {}
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
            counts[v] = counts.get(v, 0) + 1
    return counts


def _majority_owner_color(
    g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int, mover: int
) -> int | None:
    counts = _adjacent_wall_counts(g, pts, pset, bg, mover)
    if not counts:
        return None
    return max(counts, key=counts.get)


def _descend_owner(
    pset: set[tuple[int, int]], infos: list[dict], idx: int
) -> dict | None:
    info = infos[idx]
    for child_idx in info["children"]:
        if pset <= infos[child_idx]["inside"]:
            deeper = _descend_owner(pset, infos, child_idx)
            if deeper is not None:
                return deeper
    return info


def _containment_owner_color(
    g: np.ndarray,
    pts: np.ndarray,
    pset: set[tuple[int, int]],
    bg: int,
    mover: int,
    infos: list[dict],
) -> int | None:
    root_candidates = [idx for idx, info in enumerate(infos) if info["parent"] is None and pset <= info["inside"]]
    root_candidates.sort(key=lambda idx: infos[idx]["inside_area"])
    for idx in root_candidates:
        owner = _descend_owner(pset, infos, idx)
        if owner is not None:
            return int(owner["color"])
    return _majority_owner_color(g, pts, pset, bg, mover)


def _solve_with_owner(grid: np.ndarray, owner_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"
    infos = _wall_infos(g, dirs)

    passable = (g == bg) | (g == mover)
    out = g.copy()
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        if owner_mode == "containment":
            border_col = _containment_owner_color(g, pts, pset, bg, mover, infos)
        else:
            border_col = _majority_owner_color(g, pts, pset, bg, mover)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover
    return out


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_owner(grid, owner_mode="majority")


def _fusion_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    infos = _wall_infos(g, dirs)
    passable = (g == bg) | (g == mover)

    records = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue
        pset = {(int(r), int(c)) for r, c in pts}
        owner = _containment_owner_color(g, pts, pset, bg, mover, infos)
        counts = _adjacent_wall_counts(g, pts, pset, bg, mover)
        naive = max(counts, key=counts.get) if counts else None
        records.append(
            {
                "owner": owner,
                "naive": naive,
                "colors_touched": len(counts),
                "counts": counts,
                "yellow_count": len(movers),
            }
        )

    fused_out = _solve_with_owner(g, owner_mode="containment")
    baseline_out = _solve_with_owner(g, owner_mode="majority")
    return {
        "levels": len(infos),
        "regions": len(records),
        "conflict_regions": sum(1 for rec in records if rec["owner"] != rec["naive"]),
        "multi_touch_conflicts": sum(
            1 for rec in records if rec["colors_touched"] >= 2 and rec["owner"] != rec["naive"]
        ),
        "baseline_diff_cells": int(np.sum(fused_out != baseline_out)),
        "records": records,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside nested colored enclosures by following the legend of the deepest wall
    that still contains their region.

    Input structure:
    - Background cells and yellow movers live inside colored 1-cell-thick wall loops.
    - Corner legends on the border use their fill color to identify a matching wall color, and their
      center-versus-indicator offset encodes a movement direction.
    - One outer enclosed area contains a smaller enclosed child area with a different wall color, so
      the outer annulus and the inner pocket are separate passable regions in one containment tree.

    Rule:
    - Detect every non-border wall loop that has a legend, compute which cells it encloses, and
      organize those loops by containment.
    - For each enclosed passable component, descend to the most specific enclosing wall loop.
    - Slide all yellow cells in that component in the direction given by that wall's legend, while
      respecting walls, nested child walls, and previously stopped yellow cells.

    Key insight:
    - A nested outer annulus can touch the inner wall more often than its own outer wall, so local
      adjacent-wall majority is unreliable. The controlling direction is determined by geometric
      containment depth: the deepest enclosing wall wins.
    """
    return _solve_with_owner(grid, owner_mode="containment")


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


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _transform_local_cells(cells: set[tuple[int, int]], size: int, mode: str) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        if mode == "id":
            nr, nc = r, c
        elif mode == "rot90":
            nr, nc = c, size - 1 - r
        elif mode == "rot180":
            nr, nc = size - 1 - r, size - 1 - c
        elif mode == "rot270":
            nr, nc = size - 1 - c, r
        elif mode == "flip_h":
            nr, nc = r, size - 1 - c
        elif mode == "flip_v":
            nr, nc = size - 1 - r, c
        else:
            raise ValueError(mode)
        out.add((nr, nc))
    return out


def _offset_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


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
    stack: list[tuple[int, int]] = []
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
    for _ in range(30):
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

    for c in top_cols[: min(2, len(top_cols))]:
        blob.add((r0 - 1, c))
    for c in bot_cols[: min(2, len(bot_cols))]:
        blob.add((r0 + h, c))
    for r in left_rows[: min(2, len(left_rows))]:
        blob.add((r, c0 - 1))
    for r in right_rows[: min(2, len(right_rows))]:
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


def _random_direction(rng: random.Random) -> tuple[int, int]:
    if rng.random() < CARDINAL_PROB:
        return rng.choice(CARDINAL_DIRS)
    return rng.choice(DIAGONAL_DIRS)


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = grid.shape
    corners = list(rng.choice(CORNER_PAIRS))
    outer_color, inner_color = rng.sample(WALL_COLOR_CHOICES, 2)
    outer_dir = _random_direction(rng)
    inner_dir = rng.choice([direction for direction in ALL_DIRS if direction != outer_dir])

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
        mode = rng.choice(SYMMETRIES)
        outer_local = _transform_local_cells(OUTER_TEMPLATE, 9, mode)
        inner_local = _transform_local_cells(INNER_TEMPLATE, 9, mode)

        candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]] = []
        for r0 in range(2, h - 10):
            for c0 in range(2, w - 10):
                outer_blob = _offset_cells(outer_local, r0, c0)
                inner_blob = _offset_cells(inner_local, r0, c0)
                outer_wall = _blob_wall_cells(outer_blob, h, w)
                inner_wall = _blob_wall_cells(inner_blob, h, w)
                if any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in outer_wall):
                    continue
                if any(legend_occupied[rr, cc] for rr, cc in outer_blob | outer_wall):
                    continue
                if any(cell not in outer_blob for cell in inner_wall):
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
                if outer_adj == 0 or inner_adj <= outer_adj:
                    continue

                grid = np.zeros((h, w), dtype=int)
                for rr, cc in outer_wall:
                    grid[rr, cc] = int(legend_info["outer_color"])
                for rr, cc in inner_wall:
                    grid[rr, cc] = int(legend_info["inner_color"])
                if not _validate_expected_components(grid, [annulus, inner_blob]):
                    continue
                candidates.append((outer_blob, outer_wall, inner_blob, inner_wall))
        if not candidates:
            continue

        outer_blob, outer_wall, inner_blob, inner_wall = rng.choice(candidates)
        annulus = outer_blob - inner_blob - inner_wall
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
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False

    metrics = _fusion_metrics(input_grid)
    if metrics["levels"] != 2 or metrics["regions"] != 2:
        return False
    if metrics["multi_touch_conflicts"] < 1:
        return False
    if metrics["baseline_diff_cells"] < 1:
        return False
    return True


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
