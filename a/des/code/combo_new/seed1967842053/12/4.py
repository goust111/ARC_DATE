import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8)


def _inside(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _ray_cells(shape, start, direction):
    r, c = start
    dr, dc = direction
    cells = []
    r += dr
    c += dc
    while _inside(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _first_grey(grid, start, direction):
    for r, c in _ray_cells(grid.shape, start, direction):
        if grid[r, c] == GREY:
            return r, c
    return None


def _line_specs(grid):
    specs = []
    for color in LINE_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        if np.all(rows == rows[0]):
            r = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            specs.append((color, "h", r, c0, c1))
        elif np.all(cols == cols[0]):
            c = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            specs.append((color, "v", c, r0, r1))
        else:
            continue
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """Move marked line ends to the first grey barrier or crop them when no barrier exists.

    Input structure:
    The grid has black background, several non-overlapping grey solid rectangular
    obstacles, and several single-color horizontal or vertical one-pixel-wide
    line segments. Each line has a maroon marker one cell past each endpoint on
    the line axis, so the marker shows the outward direction for that endpoint.

    Rule:
    For each endpoint, scan outward through the adjacent maroon marker. If the
    ray reaches a grey obstacle, extend that line end through any black/marker
    cells up to the cell just before the obstacle. If the ray reaches the grid
    edge without grey, crop that end inward by floor(original_length / 2) cells.
    Remove all maroon markers. Keep grey obstacles and black background fixed.

    Key insight:
    The maroon pixels are not objects to preserve; they encode the direction of
    each endpoint, and each endpoint independently chooses between extension and
    midpoint cropping depending on the first grey obstacle on that ray.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    specs = _line_specs(grid)
    new_segments = []

    out[out == MAROON] = BLACK
    for color, orient, fixed, lo, hi in specs:
        out[grid == color] = BLACK
        length = hi - lo + 1
        crop = length // 2

        if orient == "h":
            r = fixed
            left_hit = _first_grey(grid, (r, lo), (0, -1))
            right_hit = _first_grey(grid, (r, hi), (0, 1))
            new_lo = left_hit[1] + 1 if left_hit is not None else lo + crop
            new_hi = right_hit[1] - 1 if right_hit is not None else hi - crop
            new_segments.append((color, orient, r, new_lo, new_hi))
        else:
            c = fixed
            up_hit = _first_grey(grid, (lo, c), (-1, 0))
            down_hit = _first_grey(grid, (hi, c), (1, 0))
            new_lo = up_hit[0] + 1 if up_hit is not None else lo + crop
            new_hi = down_hit[0] - 1 if down_hit is not None else hi - crop
            new_segments.append((color, orient, c, new_lo, new_hi))

    for color, orient, fixed, lo, hi in new_segments:
        if lo > hi:
            continue
        if orient == "h":
            r = fixed
            for c in range(max(0, lo), min(out.shape[1] - 1, hi) + 1):
                if out[r, c] != GREY:
                    out[r, c] = color
        else:
            c = fixed
            for r in range(max(0, lo), min(out.shape[0] - 1, hi) + 1):
                if out[r, c] != GREY:
                    out[r, c] = color
    return out


def _max_step(shape, endpoint, direction):
    h, w = shape
    r, c = endpoint
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return h - 1 - r
    if dc < 0:
        return c
    return w - 1 - c


def _rand_choice(rng, values):
    return values[int(rng.integers(0, len(values)))]


def _obstacle_for_endpoint(shape, endpoint, direction, rng):
    h, w = shape
    r, c = endpoint
    dr, dc = direction
    max_step = _max_step(shape, endpoint, direction)
    if max_step < 2:
        return None

    dist = int(rng.integers(2, min(max_step, 8) + 1))
    er = r + dr * dist
    ec = c + dc * dist
    cells = set()

    if dr == 0:
        max_depth = ec + 1 if dc < 0 else w - ec
        depth = int(rng.integers(1, min(3, max_depth) + 1))
        height = int(rng.integers(1, min(3, h) + 1))
        top_min = max(0, r - height + 1)
        top_max = min(r, h - height)
        top = int(rng.integers(top_min, top_max + 1))
        c0, c1 = (ec - depth + 1, ec) if dc < 0 else (ec, ec + depth - 1)
        if c0 < 0 or c1 >= w:
            return None
        for rr in range(top, top + height):
            for cc in range(c0, c1 + 1):
                cells.add((rr, cc))
    else:
        max_depth = er + 1 if dr < 0 else h - er
        depth = int(rng.integers(1, min(3, max_depth) + 1))
        width = int(rng.integers(1, min(3, w) + 1))
        left_min = max(0, c - width + 1)
        left_max = min(c, w - width)
        left = int(rng.integers(left_min, left_max + 1))
        r0, r1 = (er - depth + 1, er) if dr < 0 else (er, er + depth - 1)
        if r0 < 0 or r1 >= h:
            return None
        for rr in range(r0, r1 + 1):
            for cc in range(left, left + width):
                cells.add((rr, cc))

    path = [(r + dr * step, c + dc * step) for step in range(1, dist)]
    return cells, path


def _has_conflict(cells, occupied, forbidden_any, forbidden_grey=None):
    forbidden_grey = forbidden_grey or set()
    return any(
        cell in occupied or cell in forbidden_any or cell in forbidden_grey
        for cell in cells
    )


def _place_line(grid, occupied, forbidden_any, forbidden_grey, color, branches, rng):
    shape = grid.shape
    h, w = shape
    for _ in range(300):
        horizontal = bool(rng.integers(0, 2))
        axis_limit = w if horizontal else h
        lengths = [length for length in (5, 7, 9, 11, 13) if length <= axis_limit - 2]
        if not lengths:
            return False
        length = _rand_choice(rng, lengths)

        if horizontal:
            r = int(rng.integers(1, h - 1))
            c0 = int(rng.integers(1, w - length))
            c1 = c0 + length - 1
            line_cells = {(r, c) for c in range(c0, c1 + 1)}
            marker_cells = {(r, c0 - 1), (r, c1 + 1)}
            endpoints = [((r, c0), (0, -1), branches[0]), ((r, c1), (0, 1), branches[1])]
        else:
            c = int(rng.integers(1, w - 1))
            r0 = int(rng.integers(1, h - length))
            r1 = r0 + length - 1
            line_cells = {(r, c) for r in range(r0, r1 + 1)}
            marker_cells = {(r0 - 1, c), (r1 + 1, c)}
            endpoints = [((r0, c), (-1, 0), branches[0]), ((r1, c), (1, 0), branches[1])]

        base_cells = line_cells | marker_cells
        if _has_conflict(base_cells, occupied, forbidden_any):
            continue

        local_busy = set(base_cells)
        obstacles = []
        new_forbidden_any = []
        new_forbidden_grey = []
        failed = False

        for endpoint, direction, should_extend in endpoints:
            if should_extend:
                made = _obstacle_for_endpoint(shape, endpoint, direction, rng)
                if made is None:
                    failed = True
                    break
                obstacle_cells, path = made
                path_after_marker = [cell for cell in path if cell not in base_cells]
                if obstacle_cells & local_busy:
                    failed = True
                    break
                if _has_conflict(obstacle_cells, occupied, forbidden_any, forbidden_grey):
                    failed = True
                    break
                if _has_conflict(path_after_marker, occupied, forbidden_any):
                    failed = True
                    break
                if any(cell in local_busy for cell in path_after_marker):
                    failed = True
                    break
                obstacles.append(obstacle_cells)
                local_busy.update(obstacle_cells)
                local_busy.update(path_after_marker)
                new_forbidden_any.extend(path)
                new_forbidden_grey.extend(path)
            else:
                ray = _ray_cells(shape, endpoint, direction)
                ray_after_marker = [cell for cell in ray if cell not in base_cells]
                if _has_conflict(ray_after_marker, occupied, forbidden_any, forbidden_grey):
                    failed = True
                    break
                if any(cell in local_busy for cell in ray_after_marker):
                    failed = True
                    break
                local_busy.update(ray_after_marker)
                new_forbidden_any.extend(ray)
                new_forbidden_grey.extend(ray)

        if failed:
            continue

        for r, c in line_cells:
            grid[r, c] = color
        for r, c in marker_cells:
            grid[r, c] = MAROON
        for cells in obstacles:
            for r, c in cells:
                grid[r, c] = GREY

        occupied.update(base_cells)
        for cells in obstacles:
            occupied.update(cells)
        forbidden_any.update(new_forbidden_any)
        forbidden_grey.update(new_forbidden_grey)
        return True
    return False


def _place_extra_obstacles(grid, occupied, forbidden_any, forbidden_grey, rng, count):
    h, w = grid.shape
    for _ in range(count):
        for _ in range(150):
            height = int(rng.integers(1, 4))
            width = int(rng.integers(1, 4))
            if height >= h or width >= w:
                continue
            top = int(rng.integers(0, h - height + 1))
            left = int(rng.integers(0, w - width + 1))
            cells = {
                (r, c)
                for r in range(top, top + height)
                for c in range(left, left + width)
            }
            if _has_conflict(cells, occupied, forbidden_any, forbidden_grey):
                continue
            for r, c in cells:
                grid[r, c] = GREY
            occupied.update(cells)
            break


def _fallback_input(rng):
    h = int(rng.integers(17, 24))
    w = int(rng.integers(17, 24))
    grid = np.zeros((h, w), dtype=int)
    r = h // 2
    c0 = 5
    c1 = min(w - 5, c0 + 6)
    grid[r, c0:c1 + 1] = 2
    grid[r, c0 - 1] = MAROON
    grid[r, c1 + 1] = MAROON
    grid[r - 1:r + 2, 1:3] = GREY
    grid[2:4, w - 4:w - 2] = GREY
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(80):
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        forbidden_any = set()
        forbidden_grey = set()

        max_lines = min(3, len(LINE_COLORS))
        n_lines = int(rng.integers(1, max_lines + 1))
        colors = list(LINE_COLORS)
        rng.shuffle(colors)
        branch_flags = np.array([True] * n_lines + [False] * n_lines)
        rng.shuffle(branch_flags)

        ok = True
        for i in range(n_lines):
            branches = (bool(branch_flags[2 * i]), bool(branch_flags[2 * i + 1]))
            if not _place_line(
                grid,
                occupied,
                forbidden_any,
                forbidden_grey,
                colors[i],
                branches,
                rng,
            ):
                ok = False
                break
        if not ok:
            continue

        _place_extra_obstacles(
            grid,
            occupied,
            forbidden_any,
            forbidden_grey,
            rng,
            int(rng.integers(0, 3)),
        )
        return grid
    return _fallback_input(rng)
