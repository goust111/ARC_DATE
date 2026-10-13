import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MAROON = 9
SPRITE_COLORS = (1, 3, 4, 6, 7, 8)
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY, connectivity=4):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        perimeter = 2 * (height + width) - 4
        if len(comp) != perimeter:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if height > 2 and width > 2 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _parse_arrow(comp: list[tuple[int, int]]) -> dict[str, int] | None:
    if len(comp) != 3:
        return None
    pts = [(int(r), int(c)) for r, c in comp]
    rows = {}
    cols = {}
    for r, c in pts:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    for row, row_cols in rows.items():
        if len(row_cols) != 2:
            continue
        lo, hi = sorted(row_cols)
        if hi - lo != 2:
            continue
        mid = lo + 1
        others = [(r, c) for r, c in pts if r != row]
        if len(others) == 1 and others[0][1] == mid and abs(others[0][0] - row) == 1:
            return {
                "direction": "up" if others[0][0] < row else "down",
                "base_row": int(row),
                "base_col": int(mid),
            }

    for col, col_rows in cols.items():
        if len(col_rows) != 2:
            continue
        lo, hi = sorted(col_rows)
        if hi - lo != 2:
            continue
        mid = lo + 1
        others = [(r, c) for r, c in pts if c != col]
        if len(others) == 1 and others[0][0] == mid and abs(others[0][1] - col) == 1:
            return {
                "direction": "left" if others[0][1] < col else "right",
                "base_row": int(mid),
                "base_col": int(col),
            }

    return None


def _find_arrows(grid: np.ndarray) -> list[dict[str, int]]:
    arrows: list[dict[str, int]] = []
    for comp in _components(grid == RED, connectivity=8):
        arrow = _parse_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
    return arrows


def _match_arrow(
    frame: tuple[int, int, int, int], arrows: list[dict[str, int]]
) -> dict[str, int]:
    r0, c0, r1, c1 = frame
    matches = []
    for arrow in arrows:
        direction = arrow["direction"]
        base_row = arrow["base_row"]
        base_col = arrow["base_col"]
        if direction == "up" and base_row == r0 - 1 and c0 < base_col < c1:
            matches.append(arrow)
        elif direction == "down" and base_row == r1 + 1 and c0 < base_col < c1:
            matches.append(arrow)
        elif direction == "left" and base_col == c0 - 1 and r0 < base_row < r1:
            matches.append(arrow)
        elif direction == "right" and base_col == c1 + 1 and r0 < base_row < r1:
            matches.append(arrow)
    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each red V-shaped arrow, count maroon pixels inside its grey frame, then copy the
    interior sprite outside the frame by that distance while removing maroon pixels.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[out == MAROON] = BLACK

    frames = _find_frames(g)
    arrows = _find_arrows(g)
    assert frames

    for frame in frames:
        r0, c0, r1, c1 = frame
        arrow = _match_arrow(frame, arrows)
        dr, dc = DIR_VECTORS[arrow["direction"]]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        distance = int(np.sum(interior == MAROON))
        sprite_mask = (interior != BLACK) & (interior != MAROON) & (interior != GREY)
        for rr, cc in np.argwhere(sprite_mask):
            nr = r0 + 1 + int(rr) + dr * distance
            nc = c0 + 1 + int(cc) + dc * distance
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(interior[rr, cc])

    return out


def _random_connected_shape(height: int, width: int) -> list[tuple[int, int]]:
    target = random.randint(4, max(4, min(height * width - 1, 7)))
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {start}
    while len(cells) < target:
        base = random.choice(tuple(frontier))
        neigh = []
        for dr, dc in DIR_VECTORS.values():
            nr, nc = base[0] + dr, base[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                neigh.append((nr, nc))
        if not neigh:
            frontier.remove(base)
            if not frontier:
                frontier = set(cells)
            continue
        chosen = random.choice(neigh)
        cells.add(chosen)
        frontier.add(chosen)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if min(rows) == max(rows) or min(cols) == max(cols):
        return _random_connected_shape(height, width)
    return sorted(cells)


def _shift_sprite(
    sprite_cells: list[tuple[int, int, int]], direction: str, distance: int
) -> list[tuple[int, int, int]]:
    dr, dc = DIR_VECTORS[direction]
    return [(r + dr * distance, c + dc * distance, color) for r, c, color in sprite_cells]


def _frame_border_cells(frame_h: int, frame_w: int) -> list[tuple[int, int, int]]:
    cells = []
    for c in range(frame_w):
        cells.append((0, c, GREY))
        cells.append((frame_h - 1, c, GREY))
    for r in range(1, frame_h - 1):
        cells.append((r, 0, GREY))
        cells.append((r, frame_w - 1, GREY))
    return cells


def _arrow_cells(direction: str, frame_h: int, frame_w: int, anchor: int) -> list[tuple[int, int, int]]:
    if direction == "up":
        return [(-2, anchor, RED), (-1, anchor - 1, RED), (-1, anchor + 1, RED)]
    if direction == "down":
        return [(frame_h + 1, anchor, RED), (frame_h, anchor - 1, RED), (frame_h, anchor + 1, RED)]
    if direction == "left":
        return [(anchor, -2, RED), (anchor - 1, -1, RED), (anchor + 1, -1, RED)]
    return [(anchor, frame_w + 1, RED), (anchor - 1, frame_w, RED), (anchor + 1, frame_w, RED)]


def _object_spec(direction: str, clipped: bool) -> dict[str, object]:
    frame_h = random.randint(8, 10)
    frame_w = random.randint(8, 10)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    shape_h = random.randint(2, min(4, inner_h))
    shape_w = random.randint(2, min(4, inner_w))
    shape = _random_connected_shape(shape_h, shape_w)
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    shape_h = max(rows) + 1
    shape_w = max(cols) + 1
    max_top = inner_h - shape_h
    max_left = inner_w - shape_w

    if direction == "up":
        top = random.randint(0, min(1, max_top))
        left = random.randint(0, max_left)
    elif direction == "down":
        bottom_margin = random.randint(0, min(1, max_top))
        top = max_top - bottom_margin
        left = random.randint(0, max_left)
    elif direction == "left":
        top = random.randint(0, max_top)
        left = random.randint(0, min(1, max_left))
    else:
        top = random.randint(0, max_top)
        right_margin = random.randint(0, min(1, max_left))
        left = max_left - right_margin

    palette = random.sample(SPRITE_COLORS, random.randint(2, 4))
    sprite_cells: list[tuple[int, int, int]] = []
    color_list = [palette[i % len(palette)] for i in range(len(shape))]
    random.shuffle(color_list)
    if len(set(color_list)) == 1:
        color_list[0] = palette[0]
        color_list[1] = palette[1]
    for idx, (r, c) in enumerate(shape):
        sprite_cells.append((top + r, left + c, color_list[idx]))

    sprite_set = {(r, c) for r, c, _ in sprite_cells}
    if direction == "up":
        min_distance = max(r for r, _, _ in sprite_cells) + 4
    elif direction == "down":
        min_distance = inner_h + 3 - min(r for r, _, _ in sprite_cells)
    elif direction == "left":
        min_distance = max(c for _, c, _ in sprite_cells) + 4
    else:
        min_distance = inner_w + 3 - min(c for _, c, _ in sprite_cells)

    free_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in sprite_set]
    if len(free_cells) < min_distance:
        raise ValueError("not enough room for maroon count")
    max_distance = min(len(free_cells), min_distance + 3)
    distance = random.randint(min_distance, max_distance)

    preferred = []
    for r, c in free_cells:
        if min(abs(r - sr) + abs(c - sc) for sr, sc in sprite_set) <= 2:
            preferred.append((r, c))
    if len(preferred) >= distance:
        maroons = random.sample(preferred, distance)
    else:
        maroons = preferred[:]
        remaining = [cell for cell in free_cells if cell not in preferred]
        maroons.extend(random.sample(remaining, distance - len(maroons)))

    anchor = random.randint(2, frame_w - 3) if direction in ("up", "down") else random.randint(2, frame_h - 3)
    arrow = _arrow_cells(direction, frame_h, frame_w, anchor)
    border = _frame_border_cells(frame_h, frame_w)
    input_cells = border + arrow
    input_cells.extend((r + 1, c + 1, color) for r, c, color in sprite_cells)
    input_cells.extend((r + 1, c + 1, MAROON) for r, c in maroons)
    projected = _shift_sprite([(r + 1, c + 1, color) for r, c, color in sprite_cells], direction, distance)

    if clipped:
        edge_bias = 3
    else:
        edge_bias = 0
    return {
        "direction": direction,
        "distance": distance,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "input_cells": input_cells,
        "projected": projected,
        "edge_bias": edge_bias,
    }


def _reserve_cells(frame_h: int, frame_w: int, input_cells: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    reserve = {(r, c) for r in range(frame_h) for c in range(frame_w)}
    reserve.update((r, c) for r, c, _ in input_cells if not (0 <= r < frame_h and 0 <= c < frame_w))
    return reserve


def _pad_points(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _place_specs(specs: list[dict[str, object]], h: int, w: int) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    specs = sorted(
        specs,
        key=lambda spec: (
            -len(spec["projected"]),
            -spec["frame_h"] * spec["frame_w"],
            0 if spec["edge_bias"] else 1,
        ),
    )
    for spec in specs:
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        input_cells = list(spec["input_cells"])
        projected = list(spec["projected"])
        clipped = bool(spec["edge_bias"])

        input_rows = [r for r, _, _ in input_cells]
        input_cols = [c for _, c, _ in input_cells]
        r_lo = -min(input_rows)
        r_hi = h - 1 - max(input_rows)
        c_lo = -min(input_cols)
        c_hi = w - 1 - max(input_cols)

        if not clipped:
            proj_rows = [r for r, _, _ in projected]
            proj_cols = [c for _, c, _ in projected]
            r_lo = max(r_lo, -min(proj_rows))
            r_hi = min(r_hi, h - 1 - max(proj_rows))
            c_lo = max(c_lo, -min(proj_cols))
            c_hi = min(c_hi, w - 1 - max(proj_cols))

        if r_lo > r_hi or c_lo > c_hi:
            return None

        for _ in range(200):
            if clipped:
                if spec["direction"] == "up":
                    row_choices = list(range(r_lo, min(r_hi, r_lo + 2) + 1))
                    col_choices = list(range(c_lo, c_hi + 1))
                elif spec["direction"] == "down":
                    row_choices = list(range(max(r_lo, r_hi - 2), r_hi + 1))
                    col_choices = list(range(c_lo, c_hi + 1))
                elif spec["direction"] == "left":
                    row_choices = list(range(r_lo, r_hi + 1))
                    col_choices = list(range(c_lo, min(c_hi, c_lo + 2) + 1))
                else:
                    row_choices = list(range(r_lo, r_hi + 1))
                    col_choices = list(range(max(c_lo, c_hi - 2), c_hi + 1))
            else:
                row_choices = list(range(r_lo, r_hi + 1))
                col_choices = list(range(c_lo, c_hi + 1))

            r0 = random.choice(row_choices)
            c0 = random.choice(col_choices)

            in_bounds_proj = []
            clipped_out = False
            for pr, pc, color in projected:
                gr, gc = r0 + pr, c0 + pc
                if 0 <= gr < h and 0 <= gc < w:
                    in_bounds_proj.append((gr, gc, color))
                else:
                    clipped_out = True

            if clipped:
                if not clipped_out or not in_bounds_proj:
                    continue
            elif clipped_out:
                continue

            reserve = {(r0 + r, c0 + c) for r, c in _reserve_cells(frame_h, frame_w, input_cells)}
            reserve |= {(gr, gc) for gr, gc, _ in in_bounds_proj}
            reserve = _pad_points(reserve, h, w)
            if reserve & blocked:
                continue

            paint_input = [(r0 + r, c0 + c, color) for r, c, color in input_cells]
            placed.append(
                {
                    **spec,
                    "r0": r0,
                    "c0": c0,
                    "paint_input": paint_input,
                    "paint_output": in_bounds_proj,
                }
            )
            blocked |= reserve
            break
        else:
            return None

    return placed


def _render_input(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        for r, c, color in spec["paint_input"]:
            grid[r, c] = color
    return grid


def _render_output(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray:
    out = _render_input(specs, h, w)
    out[out == MAROON] = BLACK
    for spec in specs:
        for r, c, color in spec["paint_output"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(26, 34)
    w = random.randint(26, 34)
    obj_count = random.randint(2, 4)
    directions = random.sample(list(DIR_VECTORS.keys()), obj_count)
    clip_flags = [False] * obj_count
    clip_count = random.randint(1, obj_count - 1)
    for idx in random.sample(range(obj_count), clip_count):
        clip_flags[idx] = True

    specs = []
    try:
        for direction, clipped in zip(directions, clip_flags):
            specs.append(_object_spec(direction, clipped))
    except ValueError:
        return None

    placed = _place_specs(specs, h, w)
    if placed is None:
        return None

    inp = _render_input(placed, h, w)
    out = _render_output(placed, h, w)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
