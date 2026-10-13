import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_wall(cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    cell_set = set(cells)
    if r0 == r1:
        expected = {(r0, c) for c in range(c0, c1 + 1)}
        if cell_set == expected:
            return {
                "orientation": "horizontal",
                "row": int(r0),
                "span": set(range(c0, c1 + 1)),
                "cells": cell_set,
            }
    if c0 == c1:
        expected = {(r, c0) for r in range(r0, r1 + 1)}
        if cell_set == expected:
            return {
                "orientation": "vertical",
                "col": int(c0),
                "span": set(range(r0, r1 + 1)),
                "cells": cell_set,
            }
    return None


def _parse_arrow(cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    patch = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        patch[int(r - r0), int(c - c0)] = True

    if h == 3 and w >= 3:
        row_counts = [int(x) for x in patch.sum(axis=1)]
        if row_counts == [1, w, 1]:
            top_cols = np.flatnonzero(patch[0])
            bottom_cols = np.flatnonzero(patch[2])
            if top_cols.size == 1 and bottom_cols.size == 1 and int(top_cols[0]) == int(bottom_cols[0]):
                tip_col = int(top_cols[0])
                if patch[1, tip_col] and tip_col in (0, w - 1):
                    ok = True
                    for cc in range(w):
                        col_count = int(patch[:, cc].sum())
                        if cc == tip_col:
                            ok = ok and col_count == 3
                        else:
                            ok = ok and col_count == 1 and bool(patch[1, cc])
                    if ok:
                        direction = "left" if tip_col == 0 else "right"
                        if direction == "right":
                            shaft = [(r0 + 1, c0 + cc) for cc in range(w - 1)]
                        else:
                            shaft = [(r0 + 1, c0 + cc) for cc in range(1, w)]
                        return {
                            "direction": direction,
                            "move": DIR_TO_VEC[direction],
                            "full_cells": set(cells),
                            "shaft_cells": set(shaft),
                            "tip_coord": int(c0 + tip_col),
                            "front_span": {int(r0), int(r0 + 1), int(r0 + 2)},
                        }

    if w == 3 and h >= 3:
        col_counts = [int(x) for x in patch.sum(axis=0)]
        if col_counts == [1, h, 1]:
            left_rows = np.flatnonzero(patch[:, 0])
            right_rows = np.flatnonzero(patch[:, 2])
            if left_rows.size == 1 and right_rows.size == 1 and int(left_rows[0]) == int(right_rows[0]):
                tip_row = int(left_rows[0])
                if patch[tip_row, 1] and tip_row in (0, h - 1):
                    ok = True
                    for rr in range(h):
                        row_count = int(patch[rr].sum())
                        if rr == tip_row:
                            ok = ok and row_count == 3
                        else:
                            ok = ok and row_count == 1 and bool(patch[rr, 1])
                    if ok:
                        direction = "up" if tip_row == 0 else "down"
                        if direction == "down":
                            shaft = [(r0 + rr, c0 + 1) for rr in range(h - 1)]
                        else:
                            shaft = [(r0 + rr, c0 + 1) for rr in range(1, h)]
                        return {
                            "direction": direction,
                            "move": DIR_TO_VEC[direction],
                            "full_cells": set(cells),
                            "shaft_cells": set(shaft),
                            "tip_coord": int(r0 + tip_row),
                            "front_span": {int(c0), int(c0 + 1), int(c0 + 2)},
                        }

    return None


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(int(r + dr * steps), int(c + dc * steps)) for r, c in cells}


def _match_wall(arrow: dict, walls: list[dict]) -> int:
    direction = arrow["direction"]
    tip_coord = int(arrow["tip_coord"])
    front_span = set(arrow["front_span"])
    candidates: list[int] = []

    if direction == "right":
        for wall in walls:
            if wall["orientation"] != "vertical":
                continue
            if int(wall["col"]) <= tip_coord:
                continue
            if front_span & set(wall["span"]):
                dist = int(wall["col"]) - tip_coord - 1
                if dist > 0:
                    candidates.append(dist)
    elif direction == "left":
        for wall in walls:
            if wall["orientation"] != "vertical":
                continue
            if int(wall["col"]) >= tip_coord:
                continue
            if front_span & set(wall["span"]):
                dist = tip_coord - int(wall["col"]) - 1
                if dist > 0:
                    candidates.append(dist)
    elif direction == "down":
        for wall in walls:
            if wall["orientation"] != "horizontal":
                continue
            if int(wall["row"]) <= tip_coord:
                continue
            if front_span & set(wall["span"]):
                dist = int(wall["row"]) - tip_coord - 1
                if dist > 0:
                    candidates.append(dist)
    else:
        assert direction == "up"
        for wall in walls:
            if wall["orientation"] != "horizontal":
                continue
            if int(wall["row"]) >= tip_coord:
                continue
            if front_span & set(wall["span"]):
                dist = tip_coord - int(wall["row"]) - 1
                if dist > 0:
                    candidates.append(dist)

    assert candidates, "each arrow must have a same-color wall in front of it"
    return min(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each colored arrow toward its matching wall, leaving a grey shaft trail behind.

    Input structure:
    - The grid is a black background with several disconnected monochromatic components.
    - Arrow components are 4-connected shapes whose bbox has one dimension 3: a 1-pixel shaft plus a 3-pixel tip at one end.
    - Wall components are same-color 1-pixel-thick straight line segments, horizontal or vertical.
    - Each arrow has exactly one same-color wall ahead of its pointing direction; different colors do not interact.

    Rule:
    - Detect each arrow's direction from which side of its bbox contains the 3-pixel tip.
    - Move the whole arrow straight forward until its front edge becomes 4-neighbor adjacent to a same-color wall.
    - Clear the original arrow, paint every vacated shaft cell along the motion path grey, and draw the arrow at its final location in its original color.
    - Keep all walls unchanged.

    Key insight:
    - The salient cue is structural rather than color-based: the widened 3-pixel tip reveals direction, and only walls of the same color matter for stopping distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows: list[dict] = []
    walls_by_color: dict[int, list[dict]] = defaultdict(list)

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components_of_color(g, color):
            arrow = _parse_arrow(comp)
            if arrow is not None:
                arrow["color"] = int(color)
                arrows.append(arrow)
                continue
            wall = _parse_wall(comp)
            assert wall is not None, f"unrecognized component for color {color}"
            walls_by_color[int(color)].append(wall)

    out = g.copy()
    motions: list[dict] = []

    for arrow in arrows:
        for r, c in arrow["full_cells"]:
            out[int(r), int(c)] = BG

    for arrow in arrows:
        color = int(arrow["color"])
        dist = _match_wall(arrow, walls_by_color[color])
        dr, dc = arrow["move"]
        final_cells = _shift_cells(arrow["full_cells"], dr, dc, dist)
        trail_cells: set[tuple[int, int]] = set()
        for step in range(dist):
            trail_cells |= _shift_cells(arrow["shaft_cells"], dr, dc, step)
        trail_cells -= final_cells
        motions.append({"color": color, "final": final_cells, "trail": trail_cells})

    for motion in motions:
        for r, c in motion["trail"]:
            if out[int(r), int(c)] == BG:
                out[int(r), int(c)] = GREY

    for motion in motions:
        for r, c in motion["final"]:
            out[int(r), int(c)] = int(motion["color"])

    return out


SLOT = 11
GAP = 1
SHAFT_LEN_RANGE = (2, 4)
MOVE_DIST_RANGE = (2, 5)
WALL_LEN_RANGE = (3, 6)
OUTER_PAD_MAX = 3
MAX_ATTEMPTS = 400


def _arrow_cells(direction: str, shaft_len: int, top: int, left: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    full: set[tuple[int, int]] = set()
    shaft: set[tuple[int, int]] = set()

    if direction == "right":
        for cc in range(shaft_len):
            shaft.add((int(top + 1), int(left + cc)))
        full |= shaft
        full |= {
            (int(top + 0), int(left + shaft_len)),
            (int(top + 1), int(left + shaft_len)),
            (int(top + 2), int(left + shaft_len)),
        }
    elif direction == "left":
        for cc in range(1, shaft_len + 1):
            shaft.add((int(top + 1), int(left + cc)))
        full |= shaft
        full |= {
            (int(top + 0), int(left + 0)),
            (int(top + 1), int(left + 0)),
            (int(top + 2), int(left + 0)),
        }
    elif direction == "down":
        for rr in range(shaft_len):
            shaft.add((int(top + rr), int(left + 1)))
        full |= shaft
        full |= {
            (int(top + shaft_len), int(left + 0)),
            (int(top + shaft_len), int(left + 1)),
            (int(top + shaft_len), int(left + 2)),
        }
    else:
        assert direction == "up"
        for rr in range(1, shaft_len + 1):
            shaft.add((int(top + rr), int(left + 1)))
        full |= shaft
        full |= {
            (int(top + 0), int(left + 0)),
            (int(top + 0), int(left + 1)),
            (int(top + 0), int(left + 2)),
        }

    return full, shaft


def _build_piece(direction: str, color: int) -> np.ndarray:
    shaft_len = random.randint(*SHAFT_LEN_RANGE)
    move_dist = random.randint(*MOVE_DIST_RANGE)
    wall_len = random.randint(*WALL_LEN_RANGE)

    if direction in ("right", "left"):
        height = wall_len
        width = move_dist + shaft_len + 2
        arrow_top = random.randint(0, wall_len - 3)
        arrow_left = 0 if direction == "right" else 1 + move_dist
        wall_cells = {(int(r), int(width - 1 if direction == "right" else 0)) for r in range(height)}
    else:
        height = move_dist + shaft_len + 2
        width = wall_len
        arrow_left = random.randint(0, wall_len - 3)
        arrow_top = 0 if direction == "down" else 1 + move_dist
        wall_cells = {(int(height - 1 if direction == "down" else 0), int(c)) for c in range(width)}

    full_cells, _shaft_cells = _arrow_cells(direction, shaft_len, arrow_top, arrow_left)
    piece = np.zeros((height, width), dtype=int)
    for r, c in wall_cells | full_cells:
        piece[int(r), int(c)] = int(color)
    return piece


def _layout_for_count(count: int) -> tuple[int, int]:
    if count == 1:
        return 1, 1
    if count == 2:
        return random.choice([(1, 2), (2, 1)])
    return 2, 2


def _gen_attempt() -> np.ndarray:
    count = random.randint(1, 4)
    rows_slots, cols_slots = _layout_for_count(count)
    chosen_colors = random.sample(COLOR_POOL, count)
    directions = [random.choice(list(DIR_TO_VEC)) for _ in range(count)]
    pieces = [_build_piece(direction, color) for direction, color in zip(directions, chosen_colors)]

    top_pad = random.randint(0, OUTER_PAD_MAX)
    left_pad = random.randint(0, OUTER_PAD_MAX)
    max_bottom = min(OUTER_PAD_MAX, 30 - (top_pad + rows_slots * SLOT + (rows_slots - 1) * GAP))
    max_right = min(OUTER_PAD_MAX, 30 - (left_pad + cols_slots * SLOT + (cols_slots - 1) * GAP))
    if max_bottom < 0 or max_right < 0:
        raise RuntimeError("slot layout exceeds ARC size limit")
    bottom_pad = random.randint(0, max_bottom)
    right_pad = random.randint(0, max_right)

    height = top_pad + rows_slots * SLOT + (rows_slots - 1) * GAP + bottom_pad
    width = left_pad + cols_slots * SLOT + (cols_slots - 1) * GAP + right_pad
    grid = np.zeros((height, width), dtype=int)

    slot_origins = []
    for sr in range(rows_slots):
        for sc in range(cols_slots):
            r0 = top_pad + sr * (SLOT + GAP)
            c0 = left_pad + sc * (SLOT + GAP)
            slot_origins.append((int(r0), int(c0)))

    chosen_slots = random.sample(slot_origins, count)
    for piece, (slot_r0, slot_c0) in zip(pieces, chosen_slots):
        ph, pw = piece.shape
        assert ph <= SLOT and pw <= SLOT
        r_off = random.randint(0, SLOT - ph)
        c_off = random.randint(0, SLOT - pw)
        rr = int(slot_r0 + r_off)
        cc = int(slot_c0 + c_off)
        patch = grid[rr : rr + ph, cc : cc + pw]
        assert not np.any((patch != BG) & (piece != BG))
        patch[piece != BG] = piece[piece != BG]
        grid[rr : rr + ph, cc : cc + pw] = patch

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        out = main(inp)
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        return inp
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
