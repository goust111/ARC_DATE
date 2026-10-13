from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in {BLACK, GREY})
SIDES = ("top", "bottom", "left", "right")


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 4 or ww < 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((hh, ww), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == GREY):
            continue
        if np.any(sub[~border] == GREY):
            continue
        candidates.append((hh * ww, (r0, c0, r1, c1)))
    if not candidates:
        raise ValueError("grey collection frame not found")
    candidates.sort()
    return candidates[-1][1]


def _find_block(color_mask: np.ndarray) -> tuple[int, int]:
    h, w = color_mask.shape
    found: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if color_mask[r : r + 2, c : c + 2].all():
                found.append((r, c))
    if not found:
        raise ValueError("2x2 block not found")
    return found[0]


def _block_side(block_r: int, block_c: int, frame: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame
    if block_r + 1 < fr0:
        return "top"
    if block_r > fr1:
        return "bottom"
    if block_c + 1 < fc0:
        return "left"
    if block_c > fc1:
        return "right"
    raise ValueError("block is not outside the frame")


def _destination(block_r: int, block_c: int, side: str, frame: tuple[int, int, int, int]) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame
    if side == "top":
        return fr0 + 1, block_c
    if side == "bottom":
        return fr1 - 2, block_c
    if side == "left":
        return block_r, fc0 + 1
    if side == "right":
        return block_r, fc1 - 2
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    out = g.copy()

    legend_entries: list[tuple[int, int]] = []
    moved: list[tuple[int, tuple[int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue

        comps = _components(g == color, connectivity=8)
        singletons = [comp[0] for comp in comps if len(comp) == 1]
        if len(singletons) != 1:
            continue
        legend_r, legend_c = singletons[0]
        legend_entries.append((legend_r, color))

        coords = [(r, c) for r, c in np.argwhere(g == color) if (int(r), int(c)) != (legend_r, legend_c)]
        if not coords:
            continue
        color_mask = np.zeros_like(g, dtype=bool)
        for r, c in coords:
            color_mask[r, c] = True
            out[r, c] = BLACK

        block_r, block_c = _find_block(color_mask)
        side = _block_side(block_r, block_c, frame)
        moved.append((color, _destination(block_r, block_c, side, frame)))

    priority = {color: idx for idx, (_, color) in enumerate(sorted(legend_entries))}
    for color, (dest_r, dest_c) in sorted(moved, key=lambda item: priority.get(item[0], 10**6), reverse=True):
        out[dest_r : dest_r + 2, dest_c : dest_c + 2] = color
    return out


GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
INTERIOR_H_RANGE = (5, 8)
INTERIOR_W_RANGE = (5, 8)
FRAME_MARGIN = 5
TRAVEL_MIN = 5
TRAVEL_MAX = 7
EXTRA_BLOCKS_MAX = 2
GEN_ATTEMPTS = 400


def _pick_axis_values(
    options: list[int], count: int, rng: random.Random, forced: int | None = None
) -> list[int] | None:
    base = [] if forced is None else [forced]
    for _ in range(80):
        pool = options[:]
        rng.shuffle(pool)
        picked = base[:]
        for value in pool:
            if value in picked:
                continue
            if all(abs(value - other) >= 2 for other in picked):
                picked.append(value)
            if len(picked) == count:
                return picked
    return None


def _legend_positions(h: int, w: int, count: int, corner: str) -> list[tuple[int, int]]:
    col = 1 if "L" in corner else w - 2
    if "T" in corner:
        start_r = 1
    else:
        start_r = h - count - 1
    return [(start_r + idx, col) for idx in range(count)]


def _build_object_cells(
    side: str, block_r: int, block_c: int, frame: tuple[int, int, int, int], rng: random.Random
) -> set[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame
    cells = {(block_r + dr, block_c + dc) for dr in range(2) for dc in range(2)}

    if side == "top":
        shaft_c = block_c + rng.randrange(2)
        tip_c = block_c + 1 if shaft_c == block_c else block_c
        for r in range(block_r + 2, fr0 - 1):
            cells.add((r, shaft_c))
        cells.add((fr0 - 1, tip_c))
    elif side == "bottom":
        shaft_c = block_c + rng.randrange(2)
        tip_c = block_c + 1 if shaft_c == block_c else block_c
        for r in range(fr1 + 2, block_r):
            cells.add((r, shaft_c))
        cells.add((fr1 + 1, tip_c))
    elif side == "left":
        shaft_r = block_r + rng.randrange(2)
        tip_r = block_r + 1 if shaft_r == block_r else block_r
        for c in range(block_c + 2, fc0 - 1):
            cells.add((shaft_r, c))
        cells.add((tip_r, fc0 - 1))
    elif side == "right":
        shaft_r = block_r + rng.randrange(2)
        tip_r = block_r + 1 if shaft_r == block_r else block_r
        for c in range(fc1 + 2, block_c):
            cells.add((shaft_r, c))
        cells.add((tip_r, fc1 + 1))
    else:
        raise ValueError(side)
    return cells


def _expected_output(
    grid: np.ndarray,
    priority_order: list[int],
    final_positions: dict[int, tuple[int, int]],
    object_cells: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for cells in object_cells.values():
        for r, c in cells:
            out[r, c] = BLACK
    for color in reversed(priority_order):
        r0, c0 = final_positions[color]
        out[r0 : r0 + 2, c0 : c0 + 2] = color
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    interior_h = rng.randint(*INTERIOR_H_RANGE)
    interior_w = rng.randint(*INTERIOR_W_RANGE)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    min_h = max(GRID_H_RANGE[0], frame_h + 2 * FRAME_MARGIN)
    min_w = max(GRID_W_RANGE[0], frame_w + 2 * FRAME_MARGIN)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    frame_r0 = rng.randint(FRAME_MARGIN, h - FRAME_MARGIN - frame_h)
    frame_c0 = rng.randint(FRAME_MARGIN, w - FRAME_MARGIN - frame_w)
    frame = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    inner_r0 = frame_r0 + 1
    inner_c0 = frame_c0 + 1

    extra_blocks = rng.randint(0, EXTRA_BLOCKS_MAX)
    extra_sides = rng.sample(list(SIDES), k=extra_blocks)
    side_counts = {side: 1 + int(side in extra_sides) for side in SIDES}
    total_blocks = sum(side_counts.values())
    colors = list(rng.sample(PALETTE, k=total_blocks))
    priority_order = colors[:]
    rng.shuffle(priority_order)

    overlap_mode = rng.choice(("none", "single", "double"))
    col_options = list(range(inner_c0, inner_c0 + interior_w - 1))
    row_options = list(range(inner_r0, inner_r0 + interior_h - 1))
    leftmost_col = col_options[0]
    rightmost_col = col_options[-1]
    topmost_row = row_options[0]
    bottommost_row = row_options[-1]
    mid_cols = col_options[1:-1]
    mid_rows = row_options[1:-1]
    if not mid_cols or not mid_rows:
        return None

    forced_cols = {
        "top": rng.choice(mid_cols),
        "bottom": rng.choice(mid_cols),
    }
    forced_rows = {
        "left": rng.choice(mid_rows),
        "right": rng.choice(mid_rows),
    }

    if overlap_mode == "single":
        corner = rng.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            forced_cols["top"] = leftmost_col
            forced_rows["left"] = topmost_row
        elif corner == "tr":
            forced_cols["top"] = rightmost_col
            forced_rows["right"] = topmost_row
        elif corner == "bl":
            forced_cols["bottom"] = leftmost_col
            forced_rows["left"] = bottommost_row
        else:
            forced_cols["bottom"] = rightmost_col
            forced_rows["right"] = bottommost_row
    elif overlap_mode == "double":
        pair = rng.choice(("tl+br", "tr+bl"))
        if pair == "tl+br":
            forced_cols["top"] = leftmost_col
            forced_rows["left"] = topmost_row
            forced_cols["bottom"] = rightmost_col
            forced_rows["right"] = bottommost_row
        else:
            forced_cols["top"] = rightmost_col
            forced_rows["right"] = topmost_row
            forced_cols["bottom"] = leftmost_col
            forced_rows["left"] = bottommost_row

    top_cols = _pick_axis_values(col_options, side_counts["top"], rng, forced_cols["top"])
    bottom_cols = _pick_axis_values(col_options, side_counts["bottom"], rng, forced_cols["bottom"])
    left_rows = _pick_axis_values(row_options, side_counts["left"], rng, forced_rows["left"])
    right_rows = _pick_axis_values(row_options, side_counts["right"], rng, forced_rows["right"])
    if not top_cols or not bottom_cols or not left_rows or not right_rows:
        return None

    final_slots = {
        "top": [(inner_r0, col) for col in top_cols],
        "bottom": [(frame[2] - 2, col) for col in bottom_cols],
        "left": [(row, inner_c0) for row in left_rows],
        "right": [(row, frame[3] - 2) for row in right_rows],
    }

    grid = np.full((h, w), BLACK, dtype=int)
    fr0, fc0, fr1, fc1 = frame
    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY

    corner = rng.choice(("TL", "TR", "BL", "BR"))
    legend_cells = _legend_positions(h, w, total_blocks, corner)
    for color, (r, c) in zip(priority_order, legend_cells):
        grid[r, c] = color

    object_cells: dict[int, set[tuple[int, int]]] = {}
    final_positions: dict[int, tuple[int, int]] = {}

    color_iter = iter(colors)
    for side in SIDES:
        for final_r, final_c in final_slots[side]:
            color = next(color_iter)
            if side in ("top", "bottom"):
                if side == "top":
                    max_travel = min(TRAVEL_MAX, final_r)
                    if max_travel < TRAVEL_MIN:
                        return None
                    travel = rng.randint(TRAVEL_MIN, max_travel)
                    block_r = final_r - travel
                    block_c = final_c
                else:
                    max_travel = min(TRAVEL_MAX, h - final_r - 2)
                    if max_travel < TRAVEL_MIN:
                        return None
                    travel = rng.randint(TRAVEL_MIN, max_travel)
                    block_r = final_r + travel
                    block_c = final_c
            else:
                if side == "left":
                    max_travel = min(TRAVEL_MAX, final_c)
                    if max_travel < TRAVEL_MIN:
                        return None
                    travel = rng.randint(TRAVEL_MIN, max_travel)
                    block_r = final_r
                    block_c = final_c - travel
                else:
                    max_travel = min(TRAVEL_MAX, w - final_c - 2)
                    if max_travel < TRAVEL_MIN:
                        return None
                    travel = rng.randint(TRAVEL_MIN, max_travel)
                    block_r = final_r
                    block_c = final_c + travel

            cells = _build_object_cells(side, block_r, block_c, frame, rng)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                return None
            if any(grid[r, c] != BLACK for r, c in cells):
                return None
            for r, c in cells:
                grid[r, c] = color
            object_cells[color] = cells
            final_positions[color] = (final_r, final_c)

    expected = _expected_output(grid, priority_order, final_positions, object_cells)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input")
