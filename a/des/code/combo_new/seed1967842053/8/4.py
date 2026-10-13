import numpy as np


BLACK = 0
GREY = 5


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        cur = stack.pop()
        for nb in _neighbors(cur):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _tail_direction(cells):
    cells = set(cells)
    candidates = []
    for cell in cells:
        same = [nb for nb in _neighbors(cell) if nb in cells]
        if len(same) != 1:
            continue
        rest = cells - {cell}
        if _is_connected(rest):
            nr, nc = same[0]
            r, c = cell
            candidates.append((r - nr, c - nc))
    if not candidates:
        return (0, 0)
    return candidates[0]


def _border_lengths(grid):
    h, w = grid.shape
    coords = set()
    for c in range(w):
        coords.add((0, c))
        coords.add((h - 1, c))
    for r in range(h):
        coords.add((r, 0))
        coords.add((r, w - 1))
    lengths = {}
    for r, c in coords:
        color = int(grid[r, c])
        if color not in (BLACK, GREY):
            lengths[color] = lengths.get(color, 0) + 1
    return lengths, coords


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored object in the direction shown by its tail.

    Input structure:
    The grid has a black background, colored ruler segments on the four
    outer edges, and one small interior 4-connected object for each ruler
    color.  Each object has a single protruding tail pixel; removing that
    pixel leaves the remaining object connected.  Ruler colors are unique,
    and their visible edge lengths are unique movement distances.

    Rule:
    For every interior object, identify the tail pixel and move the whole
    object in the tail direction.  The distance is the length of the same
    colored edge ruler.  If that distance would leave the grid, shorten the
    move so the object just touches the boundary.  Original object cells
    become grey trail cells, and the colored edge rulers remain unchanged.

    Key insight:
    The transformation combines three independent clues: local object
    geometry gives direction, color matching links object to ruler, and the
    global ruler length gives distance.
    """
    grid = np.asarray(grid)
    h, w = grid.shape
    out = grid.copy()
    lengths, border = _border_lengths(grid)

    moves = []
    for color, dist in lengths.items():
        cells = {
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if int(grid[r, c]) == color
        }
        if not cells:
            continue
        dr, dc = _tail_direction(cells)
        if (dr, dc) == (0, 0):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if dr < 0:
            limit = min(rows)
        elif dr > 0:
            limit = h - 1 - max(rows)
        elif dc < 0:
            limit = min(cols)
        else:
            limit = w - 1 - max(cols)
        step = min(int(dist), int(limit))
        moved = {(r + dr * step, c + dc * step) for r, c in cells}
        moves.append((color, cells, moved))

    for _, cells, _ in moves:
        for r, c in cells:
            out[r, c] = GREY
    for color, _, moved in moves:
        for r, c in moved:
            out[r, c] = color

    for r, c in border:
        if int(grid[r, c]) not in (BLACK, GREY):
            out[r, c] = grid[r, c]
    return out


def _shape(direction, side):
    body = {(1, 1), (1, 2), (2, 1), (2, 2)}
    if direction == (-1, 0):
        tail = (0, 1 + side)
    elif direction == (1, 0):
        tail = (3, 1 + side)
    elif direction == (0, -1):
        tail = (1 + side, 0)
    else:
        tail = (1 + side, 3)
    cells = body | {tail}
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _translated(cells, anchor):
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in cells}


def _in_grid(cells, h, w, interior=False):
    lo_r, hi_r = (1, h - 2) if interior else (0, h - 1)
    lo_c, hi_c = (1, w - 2) if interior else (0, w - 1)
    return all(lo_r <= r <= hi_r and lo_c <= c <= hi_c for r, c in cells)


def _marker_cells(grid):
    h, w = grid.shape
    cells = set()
    for c in range(w):
        if grid[0, c] != BLACK:
            cells.add((0, c))
        if grid[h - 1, c] != BLACK:
            cells.add((h - 1, c))
    for r in range(h):
        if grid[r, 0] != BLACK:
            cells.add((r, 0))
        if grid[r, w - 1] != BLACK:
            cells.add((r, w - 1))
    return cells


def _max_step(cells, direction, h, w):
    dr, dc = direction
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if dr < 0:
        return min(rows)
    if dr > 0:
        return h - 1 - max(rows)
    if dc < 0:
        return min(cols)
    return w - 1 - max(cols)


def _place_object(rng, h, w, shape, direction, dist, clipped, blocked, markers):
    max_r = h - max(r for r, _ in shape)
    max_c = w - max(c for _, c in shape)
    anchors = [(r, c) for r in range(max_r) for c in range(max_c)]
    rng.shuffle(anchors)
    for anchor in anchors:
        cells = _translated(shape, anchor)
        if not _in_grid(cells, h, w, interior=True):
            continue
        limit = _max_step(cells, direction, h, w)
        if clipped:
            if not (3 <= limit < dist):
                continue
        elif limit < dist:
            continue
        step = min(dist, limit)
        dr, dc = direction
        moved = {(r + dr * step, c + dc * step) for r, c in cells}
        if cells & blocked or moved & blocked or moved & markers:
            continue
        return cells, moved
    return None, None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = np.array([1, 2, 3, 4, 6, 7, 8, 9])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(200):
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        grid = np.zeros((h, w), dtype=int)
        lengths = list(rng.choice(np.arange(4, 10), size=4, replace=False))
        colors = list(rng.choice(palette, size=4, replace=False))

        specs = [
            ("top", colors[0], int(lengths[0])),
            ("right", colors[1], int(lengths[1])),
            ("bottom", colors[2], int(lengths[2])),
            ("left", colors[3], int(lengths[3])),
        ]
        for edge, color, length in specs:
            if edge in ("top", "bottom"):
                start = int(rng.integers(1, w - length))
                row = 0 if edge == "top" else h - 1
                grid[row, start:start + length] = color
            else:
                start = int(rng.integers(1, h - length))
                col = w - 1 if edge == "right" else 0
                grid[start:start + length, col] = color

        markers = _marker_cells(grid)
        blocked = set(markers)
        placed = []
        order = list(range(4))
        rng.shuffle(order)
        ok = True
        clipped_targets = [False, True, bool(rng.integers(0, 2)), bool(rng.integers(0, 2))]
        rng.shuffle(clipped_targets)

        for idx, clipped in zip(order, clipped_targets):
            _, color, dist = specs[idx]
            direction = directions[int(rng.integers(0, 4))]
            shape = _shape(direction, int(rng.integers(0, 2)))
            cells, moved = _place_object(
                rng, h, w, shape, direction, dist, clipped, blocked, markers
            )
            if cells is None:
                ok = False
                break
            for r, c in cells:
                grid[r, c] = color
            blocked |= cells | moved
            placed.append(cells)

        if ok and len(placed) == 4:
            return grid

    return generate_input()
