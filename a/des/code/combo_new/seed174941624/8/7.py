import numpy as np


BG = 0
START_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 500


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _square_ring_info(points: list[tuple[int, int]]) -> tuple[bool, int, tuple[int, int, int, int] | None]:
    if len(points) < 8:
        return False, 0, None
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3:
        return False, 0, None
    pts = set(points)
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if pts != expected:
        return False, 0, None
    return True, h, (r0, c0, r1, c1)


def _direction_from_post(post: tuple[int, int], start: tuple[int, int]) -> tuple[int, int]:
    pr, pc = post
    sr, sc = start
    if pr == sr and pc > sc:
        return (0, 1)
    if pr == sr and pc < sc:
        return (0, -1)
    if pc == sc and pr > sr:
        return (1, 0)
    if pc == sc and pr < sr:
        return (-1, 0)
    raise AssertionError("direction post must lie strictly on the start row or column")


def _extract_program(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    singleton_colors = [color for color in colors if int(np.count_nonzero(g == color)) == 1]
    assert len(singleton_colors) == 1, "expected a unique singleton start color"
    start_color = singleton_colors[0]
    start_pos = tuple(map(int, np.argwhere(g == start_color)[0]))

    frames: list[tuple[int, int]] = []
    dir_by_color: dict[int, tuple[int, int]] = {}

    for color in colors:
        if color == start_color:
            continue
        comps = _components(g == color)
        ring_candidates: list[tuple[int, list[tuple[int, int]]]] = []
        singletons: list[tuple[int, int]] = []
        for comp in comps:
            if len(comp) == 1:
                singletons.append(comp[0])
                continue
            ok, size, _bbox_info = _square_ring_info(comp)
            if ok:
                ring_candidates.append((size, comp))
        if not ring_candidates:
            continue
        size, _ring = max(ring_candidates, key=lambda item: item[0])
        aligned_posts = [
            cell
            for cell in singletons
            if cell[0] == start_pos[0] or cell[1] == start_pos[1]
        ]
        assert len(aligned_posts) == 1, "each frame color must have exactly one aligned direction post"
        frames.append((size, color))
        dir_by_color[color] = _direction_from_post(aligned_posts[0], start_pos)

    assert frames, "expected at least one square frame"
    frames.sort(reverse=True)
    sequence = [(color, size - 1) for size, color in frames]
    return bg, start_color, start_pos, sequence, dir_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a movement program from nested square frames and draw the resulting colored polyline from the marked start.

    Input structure:
    - A black-background grid contains a left cluster of strictly nested 1-pixel-wide colored square frames.
    - The same frame colors also appear as isolated single-pixel direction posts in the right blank area.
    - The right area contains one unique single-pixel start marker whose color does not appear elsewhere; every direction post shares its row or column with that start.

    Rule:
    - Read the frame colors from the outermost square to the innermost square.
    - For a frame with side length S, its movement length is S - 1.
    - For each color, use the aligned direction post to decode one cardinal direction relative to the start marker, then execute the moves in the outer-to-inner color order.
    - Draw each step in its frame color on the traversed cells, excluding the step's starting cell, and remove the original frames and direction posts from the output while keeping the start marker.

    Key insight:
    - The nested frames encode an ordered instruction sequence twice over: color order gives step order, and frame size gives step length, while the detached same-color posts reveal the direction mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg, start_color, start_pos, sequence, dir_by_color = _extract_program(g)

    out = np.full_like(g, bg)
    out[start_pos] = start_color

    r, c = start_pos
    for color, length in sequence:
        dr, dc = dir_by_color[color]
        for _ in range(length):
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "decoded path must stay within the grid"
            out[r, c] = color

    out[start_pos] = start_color
    return out


def _choose_frame_sizes(rng: np.random.Generator, layer_count: int) -> list[int]:
    inner = int(rng.integers(3, 6))
    outer = inner + 2 * (layer_count - 1)
    return [outer - 2 * idx for idx in range(layer_count)]


def _try_relative_path(
    rng: np.random.Generator,
    lengths: list[int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(300):
        dirs = [DIRS[int(rng.integers(0, len(DIRS)))] for _ in lengths]
        if len(lengths) >= 2 and len(set(dirs)) == 1:
            continue

        pos = (0, 0)
        occupied: set[tuple[int, int]] = set()
        ok = True
        for direction, length in zip(dirs, lengths):
            dr, dc = direction
            segment: list[tuple[int, int]] = []
            for step in range(1, length + 1):
                cell = (pos[0] + dr * step, pos[1] + dc * step)
                if cell == (0, 0) or cell in occupied:
                    ok = False
                    break
                segment.append(cell)
            if not ok:
                break
            occupied.update(segment)
            pos = segment[-1]
        if ok:
            return dirs, occupied
    return None


def _ray_extent(cells: set[tuple[int, int]], direction: tuple[int, int]) -> int:
    dr, dc = direction
    best = 0
    for r, c in cells:
        if dr != 0:
            if c == 0 and r * dr > 0:
                best = max(best, abs(r))
        else:
            if r == 0 and c * dc > 0:
                best = max(best, abs(c))
    return best


def _build_posts(
    rng: np.random.Generator,
    colors: list[int],
    dirs: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    posts: dict[int, tuple[int, int]] = {}
    for direction in DIRS:
        group = [color for color, cur_dir in zip(colors, dirs) if cur_dir == direction]
        if not group:
            continue
        extent = _ray_extent(path_cells, direction)
        base = extent + int(rng.integers(2, 4))
        distances = [base + 2 * idx for idx in range(len(group))]
        rng.shuffle(distances)
        dr, dc = direction
        for color, dist in zip(group, distances):
            posts[color] = (dr * dist, dc * dist)
    return posts


def _draw_ring(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _expected_output(
    shape: tuple[int, int],
    start_pos: tuple[int, int],
    colors: list[int],
    sizes: list[int],
    dirs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[start_pos] = START_COLOR
    r, c = start_pos
    for color, size, direction in zip(colors, sizes, dirs):
        dr, dc = direction
        for _ in range(size - 1):
            r += dr
            c += dc
            out[r, c] = color
    out[start_pos] = START_COLOR
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layer_count = int(rng.integers(2, 5))
    sizes = _choose_frame_sizes(rng, layer_count)
    lengths = [size - 1 for size in sizes]
    colors = [int(v) for v in rng.permutation(np.array(COLOR_POOL, dtype=int))[:layer_count]]

    rel = _try_relative_path(rng, lengths)
    if rel is None:
        return None
    dirs, path_cells = rel
    posts = _build_posts(rng, colors, dirs, path_cells)

    rel_cells = set(path_cells)
    rel_cells.add((0, 0))
    rel_cells.update(posts.values())
    min_r = min(r for r, _ in rel_cells)
    max_r = max(r for r, _ in rel_cells)
    min_c = min(c for _, c in rel_cells)
    max_c = max(c for _, c in rel_cells)
    right_h = max_r - min_r + 1
    right_w = max_c - min_c + 1

    outer = sizes[0]
    gap = int(rng.integers(2, 5))
    min_h = max(outer, right_h)
    min_w = outer + gap + right_w
    if min_h > 30 or min_w > 30:
        return None

    total_h = int(rng.integers(min_h, min(30, min_h + 5) + 1))
    total_w = int(rng.integers(min_w, min(30, min_w + 5) + 1))

    left_top = int(rng.integers(0, total_h - outer + 1))
    left_left = int(rng.integers(0, min(2, total_w - min_w) + 1))

    right_min_col_lo = left_left + outer + gap
    right_min_col_hi = total_w - right_w
    if right_min_col_lo > right_min_col_hi:
        return None
    right_min_col = int(rng.integers(right_min_col_lo, right_min_col_hi + 1))
    right_min_row = int(rng.integers(0, total_h - right_h + 1))

    row_shift = right_min_row - min_r
    col_shift = right_min_col - min_c
    start_pos = (row_shift, col_shift)

    expected = _expected_output((total_h, total_w), start_pos, colors, sizes, dirs)
    if np.any(expected[:, : right_min_col_lo] != BG):
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    for idx, (color, size) in enumerate(zip(colors, sizes)):
        _draw_ring(grid, left_top + idx, left_left + idx, size, color)

    grid[start_pos] = START_COLOR
    for color, (rr, cc) in posts.items():
        tr = rr + row_shift
        tc = cc + col_shift
        if not (0 <= tr < total_h and 0 <= tc < total_w):
            return None
        if grid[tr, tc] != BG:
            return None
        grid[tr, tc] = color

    out = main(grid)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to create a valid puzzle instance")
