import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _in_bounds(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _touches_edge(cells, h, w):
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _legend_mapping(grid, bg):
    h, w = grid.shape
    candidates = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg:
                continue
            for name, (dr, dc) in DIRS.items():
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc
                if not (_in_bounds(r1, c1, h, w) and _in_bounds(r2, c2, h, w)):
                    continue
                marker = int(grid[r1, c1])
                if marker == bg or marker == color:
                    continue
                if int(grid[r2, c2]) != marker:
                    continue
                cells = [(r, c), (r1, c1), (r2, c2)]
                if _touches_edge(cells, h, w):
                    candidates.setdefault(color, []).append(name)

    mapping = {}
    for color, names in candidates.items():
        mapping[color] = Counter(names).most_common(1)[0][0]
    return mapping


def _same_color_components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(nr, nc, h, w)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _perimeter_cells(r1, r2, c1, c2):
    cells = set()
    for c in range(c1, c2 + 1):
        cells.add((r1, c))
        cells.add((r2, c))
    for r in range(r1, r2 + 1):
        cells.add((r, c1))
        cells.add((r, c2))
    return cells


def _find_rectangles(grid, bg):
    rects = []
    for color, cells in _same_color_components(grid, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)
        if r2 - r1 + 1 < 4 or c2 - c1 + 1 < 4:
            continue
        perim = _perimeter_cells(r1, r2, c1, c2)
        cell_set = set(cells)
        if not cell_set.issubset(perim):
            continue
        if len(cell_set) != len(perim) - 1:
            continue
        missing = [cell for cell in perim if cell not in cell_set]
        if len(missing) != 1:
            continue
        gr, gc = missing[0]
        if (gr in (r1, r2)) and (gc in (c1, c2)):
            continue
        if gr == r1:
            gap_dir = "up"
        elif gr == r2:
            gap_dir = "down"
        elif gc == c1:
            gap_dir = "left"
        elif gc == c2:
            gap_dir = "right"
        else:
            continue
        rects.append(
            {
                "color": color,
                "r1": r1,
                "r2": r2,
                "c1": c1,
                "c2": c2,
                "gap": missing[0],
                "dir": gap_dir,
            }
        )
    return rects


def _interior_components(grid, rect, bg):
    r1, r2, c1, c2 = rect["r1"], rect["r2"], rect["c1"], rect["c2"]
    border = rect["color"]
    seen = set()
    comps = []
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            color = int(grid[r, c])
            if color in (bg, border) or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if not (r1 < nr < r2 and c1 < nc < c2):
                        continue
                    if (nr, nc) in seen or int(grid[nr, nc]) != color:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _escaped_cells(cells, rect, direction):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if direction == "up":
        dr, dc = rect["r1"] - 1 - max(rs), 0
    elif direction == "down":
        dr, dc = rect["r2"] + 1 - min(rs), 0
    elif direction == "left":
        dr, dc = 0, rect["c1"] - 1 - max(cs)
    else:
        dr, dc = 0, rect["c2"] + 1 - min(cs)
    return [(r + dr, c + dc) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    """Move each trapped colored fragment out through the matching directional gap.

    Input structure:
    The background is the most common color. One or more same-colored rectangular
    outlines sit in the grid, each with exactly one missing border cell on one
    side. Their interiors contain small single-color connected fragments. Along
    the outer grid edge are legend marks: one fragment color pixel followed by a
    two-pixel marker line, so the line direction gives that color's direction.

    Rule:
    Read the edge legends to build a color-to-direction table. For every
    rectangular outline, identify the missing border cell and treat its side as
    the outward gap direction. Interior fragments whose legend direction matches
    that gap are translated straight outward until the whole fragment is just
    outside the outline. Interior fragments with any other direction are replaced
    by background. Existing borders and legend marks remain unchanged.

    Key insight:
    The edge marks are a lookup table; a fragment survives only when its looked-up
    direction equals the direction of the enclosing rectangle's gap.
    """
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    bg = _background(grid)
    mapping = _legend_mapping(grid, bg)
    rects = _find_rectangles(grid, bg)
    out = grid.copy()

    for rect in rects:
        for color, cells in _interior_components(grid, rect, bg):
            for r, c in cells:
                out[r, c] = bg
            if mapping.get(color) != rect["dir"]:
                continue
            for nr, nc in _escaped_cells(cells, rect, rect["dir"]):
                if _in_bounds(nr, nc, h, w) and int(out[nr, nc]) == bg:
                    out[nr, nc] = color
    return out


def _draw_legend(grid, direction, color, marker):
    if direction == "up":
        cells = [(2, 2), (1, 2), (0, 2)]
    elif direction == "down":
        cells = [(0, 6), (1, 6), (2, 6)]
    elif direction == "left":
        cells = [(0, 12), (0, 11), (0, 10)]
    else:
        cells = [(0, 16), (0, 17), (0, 18)]
    grid[cells[0]] = color
    grid[cells[1]] = marker
    grid[cells[2]] = marker


def _shape_templates():
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]


def _fits_fragment(cells, used):
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in used:
                    return False
    return True


def _place_fragment(rng, r1, r2, c1, c2, used):
    templates = _shape_templates()
    for _ in range(80):
        shape = rng.choice(templates)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if r2 - r1 - 3 < sh or c2 - c1 - 3 < sw:
            continue
        top = rng.randint(r1 + 2, r2 - 1 - sh)
        left = rng.randint(c1 + 2, c2 - 1 - sw)
        cells = [(top + r, left + c) for r, c in shape]
        if _fits_fragment(cells, used):
            used.update(cells)
            return cells
    return None


def _gap_for_cells(rng, cells, direction):
    if direction in ("up", "down"):
        return rng.choice([c for _, c in cells])
    return rng.choice([r for r, _ in cells])


def _try_place_rectangle(grid, blocked, rng, border, dir_to_color):
    h, w = grid.shape
    bg = int(grid[0, 0])
    for _ in range(160):
        rh = rng.randint(7, 10)
        rw = rng.randint(7, 11)
        if h - rh - 3 < 7 or w - rw - 3 < 3:
            continue
        r1 = rng.randint(7, h - rh - 3)
        c1 = rng.randint(3, w - rw - 3)
        r2 = r1 + rh - 1
        c2 = c1 + rw - 1
        if blocked[r1 - 2 : r2 + 3, c1 - 2 : c2 + 3].any():
            continue

        direction = rng.choice(list(DIRS))
        match_color = dir_to_color[direction]
        other_colors = [color for name, color in dir_to_color.items() if name != direction]
        frag_count = rng.choice([2, 2, 3])
        colors = [match_color] + rng.sample(other_colors, frag_count - 1)

        used = set()
        fragments = []
        match_cells = _place_fragment(rng, r1, r2, c1, c2, used)
        if match_cells is None:
            continue
        fragments.append((match_color, match_cells))
        ok = True
        for color in colors[1:]:
            cells = _place_fragment(rng, r1, r2, c1, c2, used)
            if cells is None:
                ok = False
                break
            fragments.append((color, cells))
        if not ok:
            continue

        if direction == "up":
            gap = (r1, _gap_for_cells(rng, match_cells, direction))
        elif direction == "down":
            gap = (r2, _gap_for_cells(rng, match_cells, direction))
        elif direction == "left":
            gap = (_gap_for_cells(rng, match_cells, direction), c1)
        else:
            gap = (_gap_for_cells(rng, match_cells, direction), c2)

        rect = {"r1": r1, "r2": r2, "c1": c1, "c2": c2, "dir": direction}
        escaped = _escaped_cells(match_cells, rect, direction)
        if not all(_in_bounds(r, c, h, w) and not blocked[r, c] for r, c in escaped):
            continue

        for c in range(c1, c2 + 1):
            grid[r1, c] = border
            grid[r2, c] = border
        for r in range(r1, r2 + 1):
            grid[r, c1] = border
            grid[r, c2] = border
        grid[gap] = bg
        for color, cells in fragments:
            for r, c in cells:
                grid[r, c] = color
        blocked[r1 - 2 : r2 + 3, c1 - 2 : c2 + 3] = True
        return True
    return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    bg = 0
    border = 5
    usable_colors = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(80):
        h = rng.randint(22, 30)
        w = rng.randint(22, 30)
        grid = np.full((h, w), bg, dtype=int)
        colors = rng.sample(usable_colors, 4)
        directions = list(DIRS)
        rng.shuffle(colors)
        dir_to_color = dict(zip(directions, colors))

        for direction, color in dir_to_color.items():
            _draw_legend(grid, direction, color, border)

        blocked = np.zeros((h, w), dtype=bool)
        blocked[:5, :] = True
        blocked[grid != bg] = True

        wanted = rng.randint(1, 3)
        placed = 0
        for _rect in range(wanted):
            if _try_place_rectangle(grid, blocked, rng, border, dir_to_color):
                placed += 1
        if placed == wanted:
            return grid

    grid = np.full((24, 24), bg, dtype=int)
    dir_to_color = {"up": 1, "down": 2, "left": 3, "right": 4}
    for direction, color in dir_to_color.items():
        _draw_legend(grid, direction, color, border)
    blocked = np.zeros((24, 24), dtype=bool)
    blocked[:5, :] = True
    blocked[grid != bg] = True
    _try_place_rectangle(grid, blocked, rng, border, dir_to_color)
    return grid
