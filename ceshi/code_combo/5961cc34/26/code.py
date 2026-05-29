# Combo variant of 5961cc34 (index 26)
# combo_spec: D1/3+D17/2
# Name: 最短路径传播+干扰色块阻挡

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior
# - The visible vocabulary stays the same as the base task: azure background,
#   blue diamond/octagon blobs, one green edge marking each blob's port, and a
#   yellow-on-red start marker at the bottom.
# - The fused change is that rays no longer keep traveling in a straight line.
#   A ray still exits through the start cap or through an activated blob's green
#   edge, but after that forced first step it follows a shortest 4-neighbor path
#   through background cells until it reaches the next blob.
# - Green-side blocking is preserved as the observable obstacle rule: a blob is
#   only activatable through one of its blue cells. Its green edge itself is not
#   enterable, so a distractor whose green side lies on the natural corridor can
#   force a detour. Example sketch:
#   * Base task: start goes straight up, then a blob shoots straight left/right.
#   * Fused task: start still begins upward, but can bend around obstacles to
#     the next blue body. Later rays do the same after leaving their green port.
#   * Difference: "direction" now fixes the launch side and tie-break order, not
#     a forever-straight beam; distractors change the chosen shortest route.
#
# b) Complexity calibration
# - The variant keeps exactly one start marker and one port side per blob, so
#   the problem is still about reading the same objects, not about inferring a
#   new symbol system.
# - To keep the shortest-path rule human-visible, the generator requires at
#   least one bent red path in the solved output; otherwise the sample would
#   collapse back toward the straight-ray original.
# - To keep the obstacle rule visible, every generated instance contains exactly
#   one non-activated blob whose removal shortens or changes the solved network.
#
# c) Solver assumptions that change
# - Original code assumed "step along one fixed direction until blue/boundary".
#   That breaks immediately once paths may turn. The fused solver therefore
#   replaces line tracing with BFS over background cells.
# - Original code could treat all later green cells as harmless paint targets.
#   That is no longer valid: green cells of non-source blobs are blocking faces.
# - Original code only needed to detect the first blue cell on a line. The new
#   version must detect the nearest reachable blue cell under obstacle-aware BFS,
#   with a deterministic direction-biased tie break.
#
# d) Placement legality
# - Blob footprints still cannot overlap and still keep a one-cell 8-neighbor
#   buffer from each other and from the start marker.
# - Chain blobs are placed only when the current source can reach them under the
#   same BFS rule used by the solver.
# - The blocker is placed by anchoring its single green pixel onto an actually
#   used path cell from the chain-only solve; this guarantees it interferes with
#   a real route instead of being a cosmetic extra.
#
# e) Visual observability
# - Across examples, the solver output shows red routes that bend around blue/
#   green bodies rather than continuing as perfectly straight lanes.
# - The blocker remains blue/green in the input but disappears in the output,
#   while the surrounding red route takes a longer route than it would without
#   that blocker. This is visible from a few examples without hidden metadata.
#
# f) Clean visualization
# - Only one blocker is used, and all chain blobs use single-pixel ports. That
#   keeps the routed paths readable and avoids wide multi-lane detours that
#   would clutter small ARC grids.
# - The generator rejects placements where the blocker collapses the cascade or
#   creates extra inactive blobs, so the final scenes stay sparse and legible.
#
# g) Generator guarantees for new assumptions
# - Every activatable blob remains reachable through a blue side.
# - The blocker's green side sits on a previously used shortest path cell, so it
#   truly acts as an obstacle under the fused rule.
# - The acceptance check quantifies both dimensions: at least one path bends,
#   and removing the blocker changes the solved output and reduces total red
#   path length.
# -----------------------------------------------------------------------------


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
LEFT_TURN = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
RIGHT_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / blocker ---
CHAIN_LEN_CHOICES = [1, 2, 3]
CHAIN_LEN_WEIGHTS = [2, 3, 3]
GEN_RETRIES = 500
BLOB_PLACE_TRIES = 120
MIN_PATH_TO_BLOB = 4
MIN_FORWARD_PROGRESS = 2


def _in_bounds_cell(r: int, c: int, H: int, W: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _neighbor_order(d: tuple[int, int]) -> list[tuple[int, int]]:
    return [tuple(d), LEFT_TURN[tuple(d)], RIGHT_TURN[tuple(d)], OPP[tuple(d)]]


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _sort_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if tuple(d) in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if tuple(d) in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_coords([(int(r), int(c)) for r, c in pts[keep]])


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

    port_pixels: list[tuple[int, int]] = []
    blocked_entries: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for gr, gc in green:
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[int(br), int(bc)] or (int(br), int(bc)) in green_set:
            return None
        cell = (int(br), int(bc))
        port_pixels.append(cell)
        blocked_entries.setdefault(cell, set()).add(tuple(OPP[tuple(port_dir)]))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (int(h), int(w)),
        "mask": mask,
        "patch": patch,
        "blue": _sort_coords(blue),
        "full": _sort_coords(full),
        "green": _sort_coords(green),
        "port_pixels": _sort_coords(port_pixels),
        "port_dir": (int(port_dir[0]), int(port_dir[1])),
        "blocked_entries": blocked_entries,
    }


def _all_specs(*, single_port_only: bool = False) -> list[dict]:
    out: list[dict] = []
    for size in SHAPES:
        for d in DIRS:
            spec = _blob_spec(size, d)
            if spec is None:
                continue
            if single_port_only and len(spec["green"]) != 1:
                continue
            out.append(spec)
    return out


SINGLE_PORT_SPECS = _all_specs(single_port_only=True)
ALL_SPECS = _all_specs(single_port_only=False)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _make_blob_instance(spec: dict, r0: int, c0: int) -> dict | None:
    h, w = spec["size"]
    full = _offset(spec["full"], r0, c0)
    if not full:
        return None
    blocked_entries: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for local_cell, moves in spec.get("blocked_entries", {}).items():
        global_cell = (int(r0 + local_cell[0]), int(c0 + local_cell[1]))
        blocked_entries[global_cell] = set(tuple(m) for m in moves)
    return {
        "spec": spec,
        "r0": int(r0),
        "c0": int(c0),
        "full": full,
        "blue": _offset(spec["blue"], r0, c0),
        "green": _offset(spec["green"], r0, c0),
        "port_pixels": _offset(spec["port_pixels"], r0, c0),
        "port_dir": tuple(spec["port_dir"]),
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "blocked_entries": blocked_entries,
    }


def _blob_center(blob: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = blob["bbox"]
    return ((int(r0) + int(r1)) / 2.0, (int(c0) + int(c1)) / 2.0)


def _progress_along(src: tuple[int, int], dst: tuple[float, float], d: tuple[int, int]) -> float:
    return (float(dst[0]) - float(src[0])) * int(d[0]) + (float(dst[1]) - float(src[1])) * int(d[1])


def _paint_blob(inp: np.ndarray, occupied: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], int(blob["r0"]), int(blob["c0"]), transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True


def _find_start(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[tuple[int, int]]]:
    H, W = grid.shape
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != int(H - 1):
                continue
            if any(int(grid[rr, c0]) != int(col) for rr in range(int(r0), int(r1) + 1)):
                continue
            if int(r0) - 1 < 0:
                continue
            cap = int(grid[int(r0) - 1, int(c0)])
            if cap == int(bg) or cap == int(col):
                continue
            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )
    assert len(start_candidates) == 1
    return start_candidates[0]


def _find_blob_colors(grid: np.ndarray, bg: int, red_col: int, yellow_col: int) -> tuple[int, int]:
    counts = Counter(map(int, grid.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _ in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    H, W = grid.shape
    for col in colors_present(grid, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(grid == int(col))
        if pts.size == 0:
            continue
        found = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(blue_col):
                    found = True
                    break
            if found:
                break
        if found:
            green_candidates.append(int(col))
    assert len(green_candidates) == 1
    return blue_col, int(green_candidates[0])


def _extract_blobs(grid: np.ndarray, blue_col: int, green_col: int) -> tuple[np.ndarray, list[dict]]:
    H, W = grid.shape
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(grid == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = int(bi)
        blobs.append({"pts": _sort_coords(pts), "ports": [], "full": set(pts), "blocked_entries": {}})

    for bi, blob in enumerate(blobs):
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in sorted(seen_dirs):
            green_pixels = sorted(
                {
                    (int(r + ddr), int(c + ddc))
                    for r, c in pts
                    if 0 <= int(r + ddr) < H
                    and 0 <= int(c + ddc) < W
                    and int(grid[int(r + ddr), int(c + ddc)]) == int(green_col)
                }
            )
            blue_port_pixels = [
                (int(r), int(c))
                for r, c in pts
                if (int(r + ddr), int(c + ddc)) in set(green_pixels)
            ]
            if not green_pixels:
                continue
            port = {
                "dir": (int(ddr), int(ddc)),
                "green": _sort_coords(green_pixels),
                "pixels": _sort_coords(blue_port_pixels),
            }
            blob["ports"].append(port)
            blob["full"].update(port["green"])
            blocked_move = OPP[(int(ddr), int(ddc))]
            for cell in port["pixels"]:
                blob["blocked_entries"].setdefault((int(cell[0]), int(cell[1])), set()).add(tuple(blocked_move))
        blob["full"] = set((int(r), int(c)) for r, c in blob["full"])
        blob["id"] = int(bi)
        blob["green"] = _sort_coords([p for port in blob["ports"] for p in port["green"]])
    return blob_id, blobs


def _reconstruct_path(prev: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = (int(goal[0]), int(goal[1]))
    while cur is not None:
        path.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    path.reverse()
    return path


def _path_to_goal_cells(
    H: int,
    W: int,
    traversable: set[tuple[int, int]],
    marker_cells: list[tuple[int, int]],
    d: tuple[int, int],
    goal_cells: set[tuple[int, int]],
    blocked_goal_entries: dict[tuple[int, int], set[tuple[int, int]]] | None = None,
) -> list[tuple[int, int]] | None:
    if not goal_cells:
        return None

    q: deque[tuple[int, int]] = deque()
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    order = _neighbor_order(d)
    blocked_goal_entries = {} if blocked_goal_entries is None else blocked_goal_entries

    for mr, mc in _sort_coords(marker_cells):
        sr = int(mr + d[0])
        sc = int(mc + d[1])
        if not _in_bounds_cell(sr, sc, H, W):
            continue
        start = (int(sr), int(sc))
        if start in prev:
            continue
        if start in goal_cells and tuple(d) not in blocked_goal_entries.get(start, set()):
            prev[start] = None
            return _reconstruct_path(prev, start)
        if start in traversable:
            prev[start] = None
            q.append(start)

    while q:
        r, c = q.popleft()
        for dr, dc in order:
            nr = int(r + dr)
            nc = int(c + dc)
            nxt = (int(nr), int(nc))
            if not _in_bounds_cell(nr, nc, H, W) or nxt in prev:
                continue
            move = (int(dr), int(dc))
            if nxt in goal_cells and move not in blocked_goal_entries.get(nxt, set()):
                prev[nxt] = (int(r), int(c))
                return _reconstruct_path(prev, nxt)
            if nxt in traversable:
                prev[nxt] = (int(r), int(c))
                q.append(nxt)
    return None


def _path_to_boundary_side(
    H: int,
    W: int,
    traversable: set[tuple[int, int]],
    marker_cells: list[tuple[int, int]],
    d: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if tuple(d) == UP:
        goals = {(0, int(c)) for c in range(W)}
    elif tuple(d) == DOWN:
        goals = {(int(H - 1), int(c)) for c in range(W)}
    elif tuple(d) == LEFT:
        goals = {(int(r), 0) for r in range(H)}
    else:
        goals = {(int(r), int(W - 1)) for r in range(H)}
    goals &= traversable
    return _path_to_goal_cells(H, W, traversable, marker_cells, d, goals)


def _path_to_specific_blob(
    H: int,
    W: int,
    marker_cells: list[tuple[int, int]],
    d: tuple[int, int],
    target_blue: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    blocked_goal_entries: dict[tuple[int, int], set[tuple[int, int]]] | None = None,
) -> list[tuple[int, int]] | None:
    traversable = {(int(r), int(c)) for r in range(H) for c in range(W) if (int(r), int(c)) not in blocked}
    return _path_to_goal_cells(H, W, traversable, marker_cells, d, target_blue, blocked_goal_entries)


def _bent_path(path_cells: list[tuple[int, int]]) -> bool:
    if len(path_cells) < 2:
        return False
    rows = {int(r) for r, _ in path_cells}
    cols = {int(c) for _, c in path_cells}
    return len(rows) > 1 and len(cols) > 1


def _analyze(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    red_col, yellow_col, start_col, start_seg = _find_start(g, bg)
    blue_col, green_col = _find_blob_colors(g, bg, red_col, yellow_col)
    blob_id, blobs = _extract_blobs(g, blue_col, green_col)

    out = np.full_like(g, int(bg))
    activated: set[int] = set()
    blocked_blobs: set[int] = set()
    queue: deque[int] = deque()
    trace: list[dict] = []
    green_owner = {}
    for bi, blob in enumerate(blobs):
        for cell in blob["green"]:
            green_owner[(int(cell[0]), int(cell[1]))] = int(bi)

    def activate_blob_at(r: int, c: int) -> int:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return int(bi)
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)
        return int(bi)

    def emit(marker_cells: list[tuple[int, int]], d: tuple[int, int], source_blob: int | None) -> None:
        for r, c in marker_cells:
            out[int(r), int(c)] = int(red_col)

        goal_owner: dict[tuple[int, int], int] = {}
        for bi, blob in enumerate(blobs):
            if int(bi) in activated or int(bi) in blocked_blobs:
                continue
            for r, c in blob["pts"]:
                goal_owner[(int(r), int(c))] = int(bi)

        traversable = {(int(r), int(c)) for r in range(H) for c in range(W) if int(g[r, c]) == int(bg)}
        q: deque[tuple[int, int]] = deque()
        prev: dict[tuple[int, int], tuple[int, int] | None] = {}
        path: list[tuple[int, int]] | None = None
        order = _neighbor_order(d)

        for mr, mc in _sort_coords(marker_cells):
            sr = int(mr + d[0])
            sc = int(mc + d[1])
            if not _in_bounds_cell(sr, sc, H, W):
                continue
            start = (int(sr), int(sc))
            if start in prev:
                continue
            if start in goal_owner:
                bi = int(goal_owner[start])
                if tuple(d) in blobs[bi]["blocked_entries"].get(start, set()):
                    blocked_blobs.add(int(bi))
                else:
                    prev[start] = None
                    path = _reconstruct_path(prev, start)
                    break
            elif start in traversable:
                prev[start] = None
                q.append(start)

        while path is None and q:
            r, c = q.popleft()
            for dr, dc in order:
                nr = int(r + dr)
                nc = int(c + dc)
                nxt = (int(nr), int(nc))
                if not _in_bounds_cell(nr, nc, H, W) or nxt in prev:
                    continue
                move = (int(dr), int(dc))
                if nxt in goal_owner:
                    bi = int(goal_owner[nxt])
                    if int(bi) in blocked_blobs:
                        continue
                    if move in blobs[bi]["blocked_entries"].get(nxt, set()):
                        blocked_blobs.add(int(bi))
                        continue
                    prev[nxt] = (int(r), int(c))
                    path = _reconstruct_path(prev, nxt)
                    break
                owner = green_owner.get(nxt)
                if owner is not None and int(owner) not in activated:
                    blocked_blobs.add(int(owner))
                    continue
                if nxt in traversable:
                    prev[nxt] = (int(r), int(c))
                    q.append(nxt)
            if path is not None:
                break

        if path is not None:
            paint_coords(out, path, red_col)
            bi = int(goal_owner[path[-1]])
            activate_blob_at(int(path[-1][0]), int(path[-1][1]))
            trace.append(
                {
                    "marker_cells": [(int(r), int(c)) for r, c in marker_cells],
                    "direction": tuple(d),
                    "path_cells": [(int(r), int(c)) for r, c in path],
                    "target_type": "blob",
                    "target_blob": int(bi),
                    "source_blob": None if source_blob is None else int(source_blob),
                    "bent": _bent_path(path),
                }
            )
            return

        boundary_path = _path_to_boundary_side(H, W, traversable, marker_cells, d)
        if boundary_path is not None:
            paint_coords(out, boundary_path, red_col)
        trace.append(
            {
                "marker_cells": [(int(r), int(c)) for r, c in marker_cells],
                "direction": tuple(d),
                "path_cells": [] if boundary_path is None else [(int(r), int(c)) for r, c in boundary_path],
                "target_type": "boundary",
                "target_blob": None,
                "source_blob": None if source_blob is None else int(source_blob),
                "bent": False if boundary_path is None else _bent_path(boundary_path),
            }
        )

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _ in start_seg)
    assert int(top_r) > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    emit([(int(top_r - 1), int(start_col))], UP, None)

    while queue:
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            emit(port["green"], tuple(port["dir"]), bi)

    return {
        "out": out,
        "trace": trace,
        "activated": set(map(int, activated)),
        "blobs": blobs,
        "unactivated_ids": sorted(int(i) for i in range(len(blobs)) if int(i) not in activated),
        "blob_bend_count": sum(1 for step in trace if step["target_type"] == "blob" and bool(step["bent"])),
        "boundary_bend_count": sum(1 for step in trace if step["target_type"] == "boundary" and bool(step["bent"])),
        "bend_count": sum(1 for step in trace if bool(step["bent"])),
        "red_count": int(np.count_nonzero(out == int(red_col))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red cascade whose beams leave through ports, then follow shortest obstacle-avoiding routes.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs each have one green edge showing their port side.
    - A yellow cap above a short bottom red segment marks the unique starting launcher.

    Rule:
    - Paint the start marker red, force one step upward, then route to the nearest reachable blue blob along a shortest 4-neighbor path through background.
    - Each activated blob turns red and launches again from its own green edge; later rays are routed the same way.
    - Green edges of other blobs are blocking faces, so distractor blobs can force detours. If no new blob is reachable, the ray continues by shortest path to the boundary on its launch side.

    Key insight: The green edge still tells you where a ray exits, but after that the red trail is determined by shortest-path search around blue/green obstacles rather than by straight-line tracing.
    """
    return _analyze(grid)["out"]


def _random_legal_blob(
    H: int,
    W: int,
    spec: dict,
    occupied: np.ndarray,
    current_markers: list[tuple[int, int]],
    direction: tuple[int, int],
    existing_blobs: list[dict],
) -> dict | None:
    occ_pad = _dilate8(occupied)
    origins = [(int(r0), int(c0)) for r0 in range(H - spec["size"][0] + 1) for c0 in range(W - spec["size"][1] + 1)]
    random.shuffle(origins)
    src = tuple(current_markers[0])

    for r0, c0 in origins[:BLOB_PLACE_TRIES]:
        blob = _make_blob_instance(spec, r0, c0)
        assert blob is not None
        if any(bool(occ_pad[int(r), int(c)]) for r, c in blob["full"]):
            continue
        if _progress_along(src, _blob_center(blob), direction) < float(MIN_FORWARD_PROGRESS):
            continue
        blocked = {(int(r), int(c)) for r, c in np.argwhere(occupied)}
        blocked_entries = {
            (int(cell[0]), int(cell[1])): set(tuple(m) for m in moves)
            for cell, moves in blob["blocked_entries"].items()
        }
        path = _path_to_specific_blob(
            H,
            W,
            current_markers,
            direction,
            set(blob["blue"]),
            blocked | set(blob["green"]),
            blocked_entries,
        )
        if path is None or len(path) < MIN_PATH_TO_BLOB:
            continue
        return blob
    return None


def _build_chain_input() -> tuple[np.ndarray, np.ndarray, dict] | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = int(H - start_h - 1)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[int(cap_r), int(start_col)] = YELLOW
    occupied[int(cap_r), int(start_col)] = True
    for r in range(int(cap_r + 1), int(H)):
        inp[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    chain_len = random.choices(CHAIN_LEN_CHOICES, weights=CHAIN_LEN_WEIGHTS)[0]
    blobs: list[dict] = []
    current_markers = [(int(cap_r), int(start_col))]
    direction = UP

    for _ in range(int(chain_len)):
        spec = random.choice(SINGLE_PORT_SPECS)
        blob = _random_legal_blob(H, W, spec, occupied, current_markers, direction, blobs)
        if blob is None:
            return None
        _paint_blob(inp, occupied, blob)
        blobs.append(blob)
        current_markers = list(blob["green"])
        direction = tuple(blob["port_dir"])

    analysis = _analyze(inp)
    if len(analysis["activated"]) != int(chain_len):
        return None
    if analysis["unactivated_ids"]:
        return None
    return inp, occupied, analysis


def _blocker_metrics(grid: np.ndarray, analysis: dict | None = None) -> dict:
    if analysis is None:
        analysis = _analyze(grid)
    if len(analysis["unactivated_ids"]) != 1:
        return {"activated_same": False, "detour_gain": 0, "output_diff": 0}

    blocker = analysis["blobs"][int(analysis["unactivated_ids"][0])]
    stripped = np.array(grid, copy=True)
    for r, c in blocker["full"]:
        stripped[int(r), int(c)] = BG
    stripped_analysis = _analyze(stripped)
    return {
        "activated_same": len(stripped_analysis["activated"]) == len(analysis["activated"]),
        "detour_gain": int(analysis["red_count"] - stripped_analysis["red_count"]),
        "output_diff": int(np.count_nonzero(analysis["out"] != stripped_analysis["out"])),
    }


def _try_add_blocker(inp: np.ndarray, occupied: np.ndarray, chain_analysis: dict) -> tuple[np.ndarray, dict] | None:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    focus_cells = [
        (int(r), int(c))
        for step in chain_analysis["trace"]
        if step["path_cells"]
        for r, c in step["path_cells"]
    ]
    if not focus_cells:
        focus_cells = [(int(H // 2), int(W // 2))]

    for _ in range(320):
        spec = random.choice(ALL_SPECS)
        h, w = spec["size"]
        rr, cc = random.choice(focus_cells)
        if random.random() < 0.8:
            r0 = random.randint(max(0, int(rr - h + 1)), min(int(H - h), int(rr)))
            c0 = random.randint(max(0, int(cc - w + 1)), min(int(W - w), int(cc)))
        else:
            r0 = random.randint(0, int(H - h))
            c0 = random.randint(0, int(W - w))

        blob = _make_blob_instance(spec, int(r0), int(c0))
        if blob is None:
            continue
        if any(bool(occ_pad[int(r), int(c)]) for r, c in blob["full"]):
            continue

        candidate = np.array(inp, copy=True)
        candidate_occ = np.array(occupied, copy=True)
        _paint_blob(candidate, candidate_occ, blob)
        analysis = _analyze(candidate)
        if len(analysis["activated"]) != len(chain_analysis["activated"]):
            continue
        if len(analysis["unactivated_ids"]) != 1:
            continue
        # Keep the shortest-path propagation visibly distinct from the straight-line
        # original by requiring at least one bent path that actually reaches a blob.
        if analysis["blob_bend_count"] < 1:
            continue
        metrics = _blocker_metrics(candidate, analysis)
        if not metrics["activated_same"]:
            continue
        if metrics["detour_gain"] <= 0 or metrics["output_diff"] <= 0:
            continue
        return candidate, analysis
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_chain_input()
        if built is None:
            continue
        inp, occupied, chain_analysis = built
        blocked = _try_add_blocker(inp, occupied, chain_analysis)
        if blocked is None:
            continue
        final_inp, final_analysis = blocked
        if not np.array_equal(main(final_inp), final_analysis["out"]):
            continue
        return final_inp
    raise RuntimeError("failed to generate combo input for 5961cc34 index 26")
