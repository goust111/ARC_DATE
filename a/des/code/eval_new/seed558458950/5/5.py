"""Transfer room contents between adjacent closed rooms through colored wall valves.

Input structure:
- The background is the most frequent color.
- Gray (one-cell-thick) walls partition the grid into axis-aligned rectangular rooms.
- Each room interior is either empty (background) or filled with a single non-background color.
- Shared walls may contain valve pixels whose color is one of the room fill colors.

Rule:
- Detect the background and wall colors, identify each room as a component of non-wall, non-valve cells.
- For every valve, inspect its two adjacent rooms. If exactly one room has fill color equal to the valve color, transfer from that room to the other.
- Apply all transfers simultaneously: every source room becomes background, every destination room is filled with the most frequent incoming transfer color (counted by how many active valves bring that color).
- Wall pixels and valve pixels are preserved.

Key insight:
The valve does not point to a target directly; it indicates which room color should flow out, so direction comes from color matching and multiple competing inputs are resolved only by transfer counts.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np


BG_COLOR = 0
WALL_COLOR = 7
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _infer_colors(grid: np.ndarray) -> Tuple[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    count_by = {int(v): int(c) for v, c in zip(vals, counts) if int(v) != bg}

    if not count_by:
        return bg, bg

    # In generated tasks, gray is usually 7; keep a strong preference when it is dominant.
    if WALL_COLOR != bg and WALL_COLOR in count_by:
        wall_count = count_by[WALL_COLOR]
        largest = max(count_by.values())
        if wall_count == largest or wall_count >= 0.45 * largest:
            return bg, WALL_COLOR

    total = grid.size
    min_area = max(4, int(total * 0.03))
    candidates = [c for c in count_by if count_by[c] >= min_area]
    if not candidates:
        candidates = list(count_by.keys())

    # Walls are long thin lines; they tend to have fewer same-color orthogonal pairs per-cell
    # than compact fill areas.
    def ratio(c: int) -> float:
        mask = grid == c
        area = int(mask.sum())
        if area <= 1:
            return float("inf")
        pairs = int(np.sum(mask[:-1, :] & mask[1:, :]) + np.sum(mask[:, :-1] & mask[:, 1:]))
        return pairs / area

    wall = min(candidates, key=lambda c: (ratio(c), -count_by[c]))
    return bg, int(wall)


def _find_valve_candidates(grid: np.ndarray, bg: int, wall: int) -> List[Tuple[int, int]]:
    h, w = grid.shape
    out: List[Tuple[int, int]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color == bg or color == wall:
                continue

            up = int(grid[r - 1, c]) != wall
            down = int(grid[r + 1, c]) != wall
            left = int(grid[r, c - 1]) != wall
            right = int(grid[r, c + 1]) != wall

            if (up and down) != (left and right):
                out.append((r, c))

    return out


def _label_rooms(grid: np.ndarray, blocked: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    h, w = grid.shape
    comp = -np.ones((h, w), dtype=np.int32)
    rooms: Dict[int, np.ndarray] = {}
    next_id = 0

    for r in range(h):
        for c in range(w):
            if blocked[r, c] or comp[r, c] != -1:
                continue

            q = deque([(r, c)])
            comp[r, c] = next_id
            cells: List[Tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not blocked[nr, nc]
                        and comp[nr, nc] == -1
                    ):
                        comp[nr, nc] = next_id
                        q.append((nr, nc))

            rooms[next_id] = np.array(cells, dtype=np.int32)
            next_id += 1

    return comp, rooms


def main(grid: np.ndarray) -> np.ndarray:
    inp = np.asarray(grid)
    if inp.ndim != 2 or inp.size == 0:
        return inp.copy()

    bg, wall = _infer_colors(inp)

    wall_mask = inp == wall
    valve_candidates = _find_valve_candidates(inp, bg, wall)
    valve_set: Set[Tuple[int, int]] = set((int(r), int(c)) for r, c in valve_candidates)

    blocked = wall_mask.copy()
    for vr, vc in valve_set:
        blocked[vr, vc] = True

    comp, rooms = _label_rooms(inp, blocked)
    if not rooms:
        return inp.copy()

    room_color: Dict[int, int] = {}
    for rid, cells in rooms.items():
        vals, cnts = np.unique(inp[tuple(zip(*cells))], return_counts=True)
        non_bg_idx = [i for i, v in enumerate(vals) if int(v) != bg]
        if non_bg_idx:
            room_color[rid] = int(vals[non_bg_idx[np.argmax(cnts[non_bg_idx])]])
        else:
            room_color[rid] = bg

    incoming: Dict[int, Dict[int, int]] = {rid: {} for rid in rooms}
    sources: Set[int] = set()

    h, w = inp.shape
    for vr, vc in valve_candidates:
        vid = int(inp[vr, vc])
        adj: Set[int] = set()

        for dr, dc in DIRS:
            nr, nc = vr + dr, vc + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                adj.add(int(comp[nr, nc]))

        if len(adj) != 2:
            continue

        a, b = tuple(adj)
        ca = room_color[a]
        cb = room_color[b]

        if ca == vid and cb != vid:
            sources.add(a)
            incoming[b][vid] = incoming[b].get(vid, 0) + 1
        elif cb == vid and ca != vid:
            sources.add(b)
            incoming[a][vid] = incoming[a].get(vid, 0) + 1

    out = inp.copy()

    for rid, cells in rooms.items():
        final_color = room_color[rid]

        counts = incoming[rid]
        if counts:
            max_count = max(counts.values())
            winner = None
            for color, cnt in counts.items():
                if cnt == max_count:
                    winner = color
                    break
            final_color = int(winner)
        elif rid in sources:
            final_color = bg

        if final_color == room_color[rid]:
            continue

        rs = cells[:, 0]
        cs = cells[:, 1]
        out[rs, cs] = final_color

    return out


def _paint_room(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    color: int,
) -> None:
    grid[top:bottom, left:right] = color


def _collect_boundaries(
    row_cuts: List[int],
    col_cuts: List[int],
    room_rows: int,
    room_cols: int,
) -> List[Dict[str, object]]:
    boundaries: List[Dict[str, object]] = []

    for i in range(room_rows):
        for j in range(room_cols - 1):
            x = col_cuts[j + 1]
            y0 = row_cuts[i] + 1
            y1 = row_cuts[i + 1] - 1
            cells = [(y, x) for y in range(y0, y1 + 1)]
            if cells:
                boundaries.append(
                    {
                        "rooms": ((i, j), (i, j + 1)),
                        "cells": cells,
                    }
                )

    for i in range(room_rows - 1):
        for j in range(room_cols):
            y = row_cuts[i + 1]
            x0 = col_cuts[j] + 1
            x1 = col_cuts[j + 1] - 1
            cells = [(y, x) for x in range(x0, x1 + 1)]
            if cells:
                boundaries.append(
                    {
                        "rooms": ((i, j), (i + 1, j)),
                        "cells": cells,
                    }
                )

    return boundaries


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]

    for _ in range(300):
        room_rows = int(rng.integers(2, 5))
        room_cols = int(rng.integers(2, 5))

        row_heights = [int(rng.integers(2, 5)) for _ in range(room_rows)]
        col_widths = [int(rng.integers(2, 5)) for _ in range(room_cols)]

        h = sum(row_heights) + room_rows + 1
        w = sum(col_widths) + room_cols + 1
        if h > 30 or w > 30:
            continue

        row_cuts = [0]
        for rh in row_heights:
            row_cuts.append(row_cuts[-1] + 1 + rh)
        col_cuts = [0]
        for cw in col_widths:
            col_cuts.append(col_cuts[-1] + 1 + cw)

        if row_cuts[-1] != h - 1 or col_cuts[-1] != w - 1:
            continue

        grid = np.full((h, w), BG_COLOR, dtype=np.int32)

        for r in row_cuts:
            grid[r, :] = WALL_COLOR
        for c in col_cuts:
            grid[:, c] = WALL_COLOR

        room_ids = [(i, j) for i in range(room_rows) for j in range(room_cols)]

        num_filled = int(rng.integers(2, len(room_ids) + 1))
        fill_indices = rng.choice(len(room_ids), size=num_filled, replace=False)
        filled_rooms = {room_ids[int(idx)] for idx in fill_indices}

        rng.shuffle(palette)
        fill_colors = palette * ((len(room_ids) // len(palette)) + 2)
        ptr = 0
        room_color: Dict[Tuple[int, int], int] = {}
        for idx, rid in enumerate(room_ids):
            if rid in filled_rooms:
                room_color[rid] = fill_colors[ptr]
                ptr += 1
            else:
                room_color[rid] = BG_COLOR

        used_fill = [room_color[rid] for rid in room_ids if room_color[rid] != BG_COLOR]
        if len(set(used_fill)) < 2 and len(room_ids) >= 2:
            # ensure at least two different colors for richer direction/mismatch behavior
            a = room_ids[0]
            if room_color[a] == BG_COLOR:
                a = room_ids[1]
            if room_color[a] != BG_COLOR:
                alt = 1
                while fill_colors[alt] == room_color[a] or fill_colors[alt] == BG_COLOR:
                    alt += 1
                room_color[a] = fill_colors[alt]
                used_fill = [room_color[rid] for rid in room_ids if room_color[rid] != BG_COLOR]

        if not used_fill:
            continue

        for i in range(room_rows):
            top = row_cuts[i] + 1
            bottom = row_cuts[i + 1]
            for j in range(room_cols):
                left = col_cuts[j] + 1
                right = col_cuts[j + 1]
                _paint_room(grid, top, bottom, left, right, room_color[(i, j)])

        boundaries = _collect_boundaries(row_cuts, col_cuts, room_rows, room_cols)
        room_to_boundaries: Dict[Tuple[int, int], List[int]] = {rid: [] for rid in room_ids}
        for bi, b in enumerate(boundaries):
            (a, b_room) = b["rooms"]
            room_to_boundaries[a].append(bi)
            room_to_boundaries[b_room].append(bi)

        used_coords: Set[Tuple[int, int]] = set()
        valve_count = 0

        for b in boundaries:
            if rng.random() >= 0.72:
                continue

            (ra, rb) = b["rooms"]
            color_a = room_color[ra]
            color_b = room_color[rb]

            active_a = color_a != BG_COLOR and color_a != color_b
            active_b = color_b != BG_COLOR and color_b != color_a

            choices: List[Tuple[str, int]] = []
            if active_a:
                choices.append(("active_a", color_a))
            if active_b:
                choices.append(("active_b", color_b))

            inactive_candidates = [c for c in set(used_fill) if c not in (color_a, color_b)]
            if inactive_candidates:
                choices.append(("inactive", int(rng.choice(inactive_candidates))))

            if not choices:
                continue

            mode, valve_color = choices[int(rng.integers(0, len(choices)))]

            cells = [coord for coord in b["cells"] if coord not in used_coords]
            if not cells:
                continue

            vr, vc = cells[int(rng.integers(0, len(cells)))]
            grid[vr, vc] = valve_color
            used_coords.add((vr, vc))
            valve_count += 1

        # Add a conflict target room where multiple incoming colors can compete.
        if rng.random() < 0.75:
            for _ in range(2):
                empty_rooms = [r for r in room_ids if room_color[r] == BG_COLOR and len(room_to_boundaries[r]) >= 3]
                if not empty_rooms:
                    break
                target = tuple(empty_rooms[int(rng.integers(0, len(empty_rooms)))] )

                candidates: List[Tuple[int, int]] = []
                neigh_colors: Dict[int, List[int]] = {}
                for bi in room_to_boundaries[target]:
                    a, b_room = boundaries[bi]["rooms"]
                    src = b_room if a == target else a
                    sc = room_color[src]
                    if sc != BG_COLOR:
                        neigh_colors.setdefault(sc, []).append(bi)

                colors = list(neigh_colors.keys())
                if len(colors) < 2:
                    continue

                rng.shuffle(colors)
                c1, c2 = colors[0], colors[1]
                boundaries_for_c1 = neigh_colors[c1]
                boundaries_for_c2 = neigh_colors[c2]

                ok = 0
                for bc, color in ((boundaries_for_c1[0], c1), (boundaries_for_c2[0], c2)):
                    cells = [coord for coord in boundaries[bc]["cells"] if coord not in used_coords]
                    if not cells:
                        continue
                    vr, vc = cells[int(rng.integers(0, len(cells)))]
                    grid[vr, vc] = color
                    used_coords.add((vr, vc))
                    valve_count += 1
                    ok += 1

                # Give c1 a stronger count when possible so result is deterministic.
                if ok == 2 and len(boundaries_for_c1) >= 2:
                    bc = boundaries_for_c1[1]
                    cells = [coord for coord in boundaries[bc]["cells"] if coord not in used_coords]
                    if cells:
                        vr, vc = cells[int(rng.integers(0, len(cells)))]
                        grid[vr, vc] = c1
                        used_coords.add((vr, vc))
                        valve_count += 1
                if valve_count:
                    break

        if valve_count == 0:
            continue

        return np.asarray(grid, dtype=np.int32)

    # Fallback: emit a small deterministic layout if all random attempts fail.
    return np.array(
        [
            [7, 7, 7, 7, 7, 7, 7],
            [7, 0, 0, 1, 7, 2, 7],
            [7, 0, 0, 0, 7, 0, 7],
            [7, 3, 0, 0, 0, 0, 7],
            [7, 0, 0, 0, 4, 0, 7],
            [7, 7, 7, 7, 7, 7, 7],
        ],
        dtype=np.int32,
    )
