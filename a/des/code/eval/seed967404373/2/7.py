import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

SOURCE_COLORS = (BLUE, RED, GREEN, ORANGE, AZURE, BROWN)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


def _collect_sources(grid):
    h, w = grid.shape
    sources = []
    seen = set()
    for r in range(h):
        for c in (0, w - 1):
            if (r, c) in seen:
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            seen.add((r, c))
            if r in (0, h - 1):
                raise AssertionError("corner sources are ambiguous")
            direction = RIGHT if c == 0 else LEFT
            sources.append({"pos": (r, c), "color": color, "dir": direction})
    for c in range(w):
        for r in (0, h - 1):
            if (r, c) in seen:
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            seen.add((r, c))
            if c in (0, w - 1):
                raise AssertionError("corner sources are ambiguous")
            direction = DOWN if r == 0 else UP
            sources.append({"pos": (r, c), "color": color, "dir": direction})
    return sources, seen


def _mirror_orientations(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    orientations = {}
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != GREY:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            if not 2 <= len(coords) <= 4:
                raise AssertionError("mirror length must be between 2 and 4")
            rows = {rr for rr, _ in coords}
            cols = {cc for _, cc in coords}
            if len(rows) == 1:
                orientation = "h"
            elif len(cols) == 1:
                orientation = "v"
            else:
                raise AssertionError("mirror must be horizontal or vertical")
            for cell in coords:
                orientations[cell] = orientation
    return orientations


def _reflect(direction, orientation):
    if orientation == "v":
        if direction == RIGHT:
            return UP
        if direction == LEFT:
            return DOWN
        return None
    if orientation == "h":
        if direction == DOWN:
            return LEFT
        if direction == UP:
            return RIGHT
        return None
    return None


def _trace_contributions(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources, source_cells = _collect_sources(g)
    mirror_orientation = _mirror_orientations(g)
    contributions = defaultdict(set)

    for source in sources:
        r, c = source["pos"]
        dr, dc = source["dir"]
        color = int(source["color"])
        seen_states = set()

        while True:
            state = (r, c, dr, dc)
            if state in seen_states:
                break
            seen_states.add(state)

            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in source_cells:
                break

            cell = int(g[nr, nc])
            if cell == GREY:
                new_dir = _reflect((dr, dc), mirror_orientation.get((nr, nc)))
                if new_dir is None:
                    break
                dr, dc = new_dir
                continue
            if cell != BLACK:
                break

            r, c = nr, nc
            contributions[(r, c)].add(color)

    return contributions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect colored edge rays on grey mirror segments and mix beam colors at intersections.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    contributions = _trace_contributions(g)

    for (r, c), colors in contributions.items():
        n_colors = len(colors)
        if n_colors == 1:
            out[r, c] = next(iter(colors))
        elif n_colors == 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = MAGENTA

    return out


def _place_mirror_segment(height, width, orientation, anchor, blocked, rng):
    r, c = anchor
    lengths = [2, 3, 4]
    rng.shuffle(lengths)

    for length in lengths:
        if orientation == "h":
            start_min = max(1, c - length + 1)
            start_max = min(c, width - length - 1)
            if start_min > start_max:
                continue
            starts = list(range(start_min, start_max + 1))
            rng.shuffle(starts)
            for c0 in starts:
                cells = [(r, cc) for cc in range(c0, c0 + length)]
                if any(cell in blocked for cell in cells):
                    continue
                if anchor not in cells:
                    continue
                return cells
        else:
            start_min = max(1, r - length + 1)
            start_max = min(r, height - length - 1)
            if start_min > start_max:
                continue
            starts = list(range(start_min, start_max + 1))
            rng.shuffle(starts)
            for r0 in starts:
                cells = [(rr, c) for rr in range(r0, r0 + length)]
                if any(cell in blocked for cell in cells):
                    continue
                if anchor not in cells:
                    continue
                return cells
    return None


def _build_horizontal_case(height, width, use_magenta, rng):
    y = rng.randint(4, height - 5)
    colors = rng.sample(SOURCE_COLORS, 4 if use_magenta else 3)
    sources = [(y, 0, colors[0]), (y, width - 1, colors[1])]
    primary_path = {(y, c) for c in range(1, width - 1)}

    if use_magenta:
        third_side = rng.choice(("top", "bottom"))
        if third_side == "top":
            x = rng.randint(3, width - 5)
            sources.append((0, x, colors[2]))
            primary_path |= {(r, x) for r in range(1, height)}
            secondary_side = "bottom"
        else:
            x = rng.randint(4, width - 4)
            sources.append((height - 1, x, colors[2]))
            primary_path |= {(r, x) for r in range(0, height - 1)}
            secondary_side = "top"
    else:
        x = None
        secondary_side = rng.choice(("top", "bottom"))

    secondary_color = colors[-1]
    if secondary_side == "bottom":
        cs_min = 2 if x is None else x + 2
        cs_max = width - 3
        if cs_min > cs_max:
            return None
        cs = rng.randint(cs_min, cs_max)
        turn_row = rng.randint(y + 2, height - 3)
        sources.append((height - 1, cs, secondary_color))
        secondary_path = {(r, cs) for r in range(turn_row, height - 1)}
        secondary_path |= {(turn_row, c) for c in range(cs + 1, width)}
        mirror_request = ("h", (turn_row - 1, cs))
    else:
        cs_min = 2
        cs_max = width - 3 if x is None else x - 2
        if cs_min > cs_max:
            return None
        cs = rng.randint(cs_min, cs_max)
        turn_row = rng.randint(2, y - 2)
        sources.append((0, cs, secondary_color))
        secondary_path = {(r, cs) for r in range(1, turn_row + 1)}
        secondary_path |= {(turn_row, c) for c in range(0, cs)}
        mirror_request = ("h", (turn_row + 1, cs))

    if primary_path & secondary_path:
        return None

    return {
        "sources": sources,
        "paths": primary_path | secondary_path,
        "mirror_request": mirror_request,
        "mix_count": 3 if use_magenta else 2,
        "secondary_color": secondary_color,
    }


def _build_vertical_case(height, width, use_magenta, rng):
    x = rng.randint(4, width - 5)
    colors = rng.sample(SOURCE_COLORS, 4 if use_magenta else 3)
    sources = [(0, x, colors[0]), (height - 1, x, colors[1])]
    primary_path = {(r, x) for r in range(1, height - 1)}

    if use_magenta:
        third_side = rng.choice(("left", "right"))
        if third_side == "left":
            y = rng.randint(3, height - 5)
            sources.append((y, 0, colors[2]))
            primary_path |= {(y, c) for c in range(1, width)}
            secondary_side = "right"
        else:
            y = rng.randint(4, height - 4)
            sources.append((y, width - 1, colors[2]))
            primary_path |= {(y, c) for c in range(0, width - 1)}
            secondary_side = "left"
    else:
        y = None
        secondary_side = rng.choice(("left", "right"))

    secondary_color = colors[-1]
    if secondary_side == "left":
        rs_min = 2
        rs_max = height - 3 if y is None else y - 2
        if rs_min > rs_max:
            return None
        rs = rng.randint(rs_min, rs_max)
        turn_col = rng.randint(2, x - 2)
        sources.append((rs, 0, secondary_color))
        secondary_path = {(rs, c) for c in range(1, turn_col + 1)}
        secondary_path |= {(r, turn_col) for r in range(0, rs)}
        mirror_request = ("v", (rs, turn_col + 1))
    else:
        rs_min = 2 if y is None else y + 2
        rs_max = height - 3
        if rs_min > rs_max:
            return None
        rs = rng.randint(rs_min, rs_max)
        turn_col = rng.randint(x + 2, width - 3)
        sources.append((rs, width - 1, secondary_color))
        secondary_path = {(rs, c) for c in range(turn_col, width - 1)}
        secondary_path |= {(r, turn_col) for r in range(rs + 1, height)}
        mirror_request = ("v", (rs, turn_col - 1))

    if primary_path & secondary_path:
        return None

    return {
        "sources": sources,
        "paths": primary_path | secondary_path,
        "mirror_request": mirror_request,
        "mix_count": 3 if use_magenta else 2,
        "secondary_color": secondary_color,
    }


def _gen_attempt(rng):
    height = rng.randint(15, 22)
    width = rng.randint(15, 22)
    use_magenta = rng.choice((False, True))
    axis = rng.choice(("horizontal", "vertical"))

    if axis == "horizontal":
        spec = _build_horizontal_case(height, width, use_magenta, rng)
    else:
        spec = _build_vertical_case(height, width, use_magenta, rng)
    if spec is None:
        return None

    blocked = {(r, c) for r, c, _ in spec["sources"]} | set(spec["paths"])
    orientation, anchor = spec["mirror_request"]
    mirror_cells = _place_mirror_segment(height, width, orientation, anchor, blocked, rng)
    if mirror_cells is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c, color in spec["sources"]:
        grid[r, c] = int(color)
    for r, c in mirror_cells:
        grid[r, c] = GREY

    if not _validate_generated(grid, spec["mix_count"], spec["secondary_color"]):
        return None
    return grid


def _validate_generated(grid, mix_count, secondary_color):
    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.any(out == YELLOW):
        return False
    if mix_count == 2 and np.any(out == MAGENTA):
        return False
    if mix_count == 3 and not np.any(out == MAGENTA):
        return False
    if np.count_nonzero(grid == GREY) < 2:
        return False
    if np.count_nonzero(out == secondary_color) <= 1:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray input")
