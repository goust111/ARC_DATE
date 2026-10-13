import numpy as np
from collections import Counter, deque


BACKGROUND = 0
WALL_COLOR = 5
ARROW_COLOR = 2

DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_8 = DIRS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

ARROW_TEMPLATES = {
    (0, 1): frozenset([(0, 0), (1, 1), (2, 0)]),  # >
    (0, -1): frozenset([(0, 1), (1, 0), (2, 1)]),  # <
    (1, 0): frozenset([(0, 0), (0, 2), (1, 1)]),  # v
    (-1, 0): frozenset([(1, 0), (1, 2), (0, 1)]),  # ^
}


def _components(grid: np.ndarray, color: int, dirs=DIRS_4):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    comps = []
    for rr in range(h):
        for cc in range(w):
            if visited[rr, cc] or grid[rr, cc] != color:
                continue
            q = deque([(rr, cc)])
            visited[rr, cc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in dirs:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    return min_r, max_r, min_c, max_c


def _norm_shape(cells):
    min_r, max_r, min_c, max_c = _bbox(cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _is_solid_square(cells):
    min_r, max_r, min_c, max_c = _bbox(cells)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if h not in (2, 3) or w not in (2, 3) or h != w:
        return False, None
    area = h * w
    if len(cells) != area:
        return False, None
    return True, h


def _is_straight_line(cells):
    min_r, max_r, min_c, max_c = _bbox(cells)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if len(cells) == 1:
        return False, None
    if h == 1 and w == len(cells):
        return True, 'h'
    if w == 1 and h == len(cells):
        return True, 'v'
    return False, None


def _get_background(grid: np.ndarray):
    flat = grid.reshape(-1)
    counts = Counter(flat.tolist())
    return counts.most_common(1)[0][0]


def _find_arrow_components(grid, bg_color):
    h, w = grid.shape
    arrows = []
    for color in np.unique(grid):
        if color == bg_color:
            continue
        comps = _components(grid, int(color), dirs=DIRS_8)
        for comp in comps:
            if len(comp) != 3:
                continue
            shape = _norm_shape(comp)
            for d, template in ARROW_TEMPLATES.items():
                if shape == template:
                    arrows.append(
                        {
                            'color': int(color),
                            'dir': d,
                            'cells': comp,
                            'bbox': _bbox(comp),
                        }
                    )
                    break
    return arrows


def _find_wall_color(grid, bg_color):
    best_color = None
    best_components = []
    best_len = 0

    for color in np.unique(grid):
        color = int(color)
        if color == bg_color:
            continue
        total_len = 0
        line_cells = []
        for comp in _components(grid, color, dirs=DIRS_4):
            ok, orient = _is_straight_line(comp)
            if ok and len(comp) >= 2:
                total_len += len(comp)
                line_cells.extend(comp)
        if total_len > best_len:
            best_len = total_len
            best_color = color
            best_components = line_cells
    if best_color is None:
        return None, set()
    return best_color, set(best_components)


def _measure_to_wall(grid, source, dirv, wall_cells, bg):
    min_r, max_r, min_c, max_c = source['bbox']
    h, w = grid.shape
    dr, dc = dirv
    hits = []

    if (dr, dc) == (0, 1):
        for rr in range(min_r, max_r + 1):
            c = max_c + 1
            d = 0
            while 0 <= c < w and grid[rr, c] == bg:
                d += 1
                c += 1
            if 0 <= c < w and (rr, c) in wall_cells:
                hits.append((d, rr, c))
    elif (dr, dc) == (0, -1):
        for rr in range(min_r, max_r + 1):
            c = min_c - 1
            d = 0
            while 0 <= c < w and grid[rr, c] == bg:
                d += 1
                c -= 1
            if 0 <= c < w and (rr, c) in wall_cells:
                hits.append((d, rr, c))
    elif (dr, dc) == (1, 0):
        for cc in range(min_c, max_c + 1):
            r = max_r + 1
            d = 0
            while 0 <= r < h and grid[r, cc] == bg:
                d += 1
                r += 1
            if 0 <= r < h and (r, cc) in wall_cells:
                hits.append((d, r, cc))
    else:
        for cc in range(min_c, max_c + 1):
            r = min_r - 1
            d = 0
            while 0 <= r < h and grid[r, cc] == bg:
                d += 1
                r -= 1
            if 0 <= r < h and (r, cc) in wall_cells:
                hits.append((d, r, cc))

    if not hits:
        return None
    best = min(x[0] for x in hits)
    near_hits = [(r, c) for d, r, c in hits if d == best]
    return {
        'dist': best,
        'hits': near_hits,
    }


def _arrow_matches_source(source, arrow, dirv):
    src_min_r, src_max_r, src_min_c, src_max_c = source['bbox']
    a_min_r, a_max_r, a_min_c, a_max_c = arrow['bbox']
    gap_r1 = src_min_r - a_max_r
    gap_r2 = a_min_r - src_max_r
    gap_c1 = src_min_c - a_max_c
    gap_c2 = a_min_c - src_max_c

    if dirv == (0, 1):
        if arrow['dir'] != dirv:
            return False
        if gap_c1 not in (1, 2):
            return False
        if a_max_r < src_min_r or a_min_r > src_max_r:
            return False
        return True

    if dirv == (0, -1):
        if arrow['dir'] != dirv:
            return False
        if gap_c2 not in (1, 2):
            return False
        if a_max_r < src_min_r or a_min_r > src_max_r:
            return False
        return True

    if dirv == (1, 0):
        if arrow['dir'] != dirv:
            return False
        if gap_r2 not in (1, 2):
            return False
        if a_max_c < src_min_c or a_min_c > src_max_c:
            return False
        return True

    if dirv == (-1, 0):
        if arrow['dir'] != dirv:
            return False
        if gap_r1 not in (1, 2):
            return False
        if a_max_c < src_min_c or a_min_c > src_max_c:
            return False
        return True

    return False


def _paint_scaled_projection(out, source, dirv, dist_info, scale):
    h, w = out.shape
    min_r, max_r, min_c, max_c = source['bbox']
    src_h = max_r - min_r + 1
    src_w = max_c - min_c + 1
    color = source['color']

    tgt_h = src_h * scale
    tgt_w = src_w * scale

    if not dist_info['hits']:
        return
    # choose one anchored wall pixel
    wall_r, wall_c = dist_info['hits'][0]

    if dirv == (0, 1):
        dst_c = wall_c + 1
        dst_r = int(round((min_r + max_r) / 2 - (tgt_h - 1) / 2))
    elif dirv == (0, -1):
        dst_c = wall_c - tgt_w
        dst_r = int(round((min_r + max_r) / 2 - (tgt_h - 1) / 2))
    elif dirv == (1, 0):
        dst_r = wall_r + 1
        dst_c = int(round((min_c + max_c) / 2 - (tgt_w - 1) / 2))
    else:
        dst_r = wall_r - tgt_h
        dst_c = int(round((min_c + max_c) / 2 - (tgt_w - 1) / 2))

    if dst_r < 0 or dst_c < 0 or dst_r + tgt_h > h or dst_c + tgt_w > w:
        return

    for rr in range(min_r, max_r + 1):
        for cc in range(min_c, max_c + 1):
            sr = (rr - min_r) * scale
            sc = (cc - min_c) * scale
            for dr in range(scale):
                for dc in range(scale):
                    out[dst_r + sr + dr, dst_c + sc + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Project source patches across walls according to arrow direction and gap distance.

Input structure:
- Background is black and dominant.
- One or more gray straight horizontal/vertical wall line segments are present.
- Small 2x2 or 3x3 monochrome source patches sit on one side of a wall.
- A red 3-pixel triangular arrow is adjacent to each source patch, on the side opposite the wall, and points toward the wall.

Rule:
Find each valid source patch, identify its arrow direction, then measure D as the number of background cells from its nearest edge to the first gray wall cell in that direction. If D<=2, draw a 1x copy; if D is 3 or 4 draw a 2x copy; if D>=5 draw a 3x copy. Paint the scaled copy on the wall’s opposite side (through the wall), touching the wall and centered in the perpendicular axis. Keep source patches, arrows and wall pixels unchanged.

Key insight:
Direction comes from the red marker, while scale is determined by the measured free-space distance, and the copy is positioned by wall-touching placement with perpendicular centering."""

    grid = np.array(grid, dtype=np.int64)
    h, w = grid.shape
    out = grid.copy()

    bg = _get_background(grid)
    arrows = _find_arrow_components(grid, bg)
    if not arrows:
        return out

    wall_color, wall_cells = _find_wall_color(grid, bg)
    if wall_color is None:
        return out

    # source candidates are solid 2x2 or 3x3 monochrome blocks.
    sources = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, wall_color, ARROW_COLOR):
            continue
        for comp in _components(grid, color, dirs=DIRS_4):
            ok, sz = _is_solid_square(comp)
            if ok:
                sources.append({'color': color, 'cells': comp, 'bbox': _bbox(comp), 'size': sz})

    for source in sources:
        best = []
        for dirv in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            arrows_dir = [a for a in arrows if a['dir'] == dirv]
            if not arrows_dir:
                continue
            if not any(_arrow_matches_source(source, a, dirv) for a in arrows_dir):
                continue
            dist = _measure_to_wall(grid, source, dirv, wall_cells, bg)
            if dist is None:
                continue
            best.append((dist['dist'], dirv, dist))

        if not best:
            continue

        # choose the nearest match for this source
        best.sort(key=lambda x: x[0])
        dist, dirv, dist_info = best[0]
        if dist <= 2:
            scale = 1
        elif dist <= 4:
            scale = 2
        else:
            scale = 3
        _paint_scaled_projection(out, source, dirv, dist_info, scale)

    return out


def _all_clear(grid, used, cells):
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if used[r, c]:
            return False
    return True


def _draw_cells(grid, used, cells, color):
    for r, c in cells:
        grid[r, c] = color
        used[r, c] = True


def _paint_source_square(grid, used, top, left, size, color):
    cells = [(r, c) for r in range(top, top + size) for c in range(left, left + size)]
    if not _all_clear(grid, used, cells):
        return None
    _draw_cells(grid, used, cells, color)
    return cells


def _paint_wall_segment(grid, used, fixed_row, start, length, orientation, color):
    if orientation == 'h':
        if start < 0 or start + length > grid.shape[1] or fixed_row < 0 or fixed_row >= grid.shape[0]:
            return None
        cells = [(fixed_row, start + i) for i in range(length)]
    else:
        if start < 0 or start + length > grid.shape[0] or fixed_row < 0 or fixed_row >= grid.shape[1]:
            return None
        cells = [(start + i, fixed_row) for i in range(length)]

    if not _all_clear(grid, used, cells):
        return None
    _draw_cells(grid, used, cells, color)
    return cells


def _paint_arrow(grid, used, anchor_r, anchor_c, shape, color=ARROW_COLOR):
    cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
    if not _all_clear(grid, used, cells):
        return None
    _draw_cells(grid, used, cells, color)
    return cells


def _scale_from_distance(rng: np.random.Generator, distance):
    if distance <= 2:
        return 1
    if distance <= 4:
        return 2
    return 3


def generate_input() -> np.ndarray:
    """Randomly sample valid wall-arrow-pattern instances with varied scale-distance cases."""
    rng = np.random.default_rng()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    palette = [1, 3, 4, 6, 7, 8, 9]

    for _ in range(100):
        h = int(rng.integers(14, 27))
        w = int(rng.integers(14, 27))
        grid = np.full((h, w), BACKGROUND, dtype=np.int64)
        used = np.zeros((h, w), dtype=bool)

        pair_count = int(rng.integers(1, 4))
        placed = 0

        for p in range(pair_count):
            color = palette[p % len(palette)]
            for _ in range(500):
                dirv = directions[int(rng.integers(0, 4))]

                branch = int(rng.integers(0, 3))
                if branch == 0:
                    dist = int(rng.integers(0, 3))
                elif branch == 1:
                    dist = int(rng.integers(3, 5))
                else:
                    dist = int(rng.integers(5, 8))

                scale = _scale_from_distance(rng, dist)
                size = int(rng.choice([2, 3]))
                src_cells = []
                wall_cells = []
                arrow_cells = []

                if dirv == (0, 1):  # source left, wall right, arrow left
                    tgt_w = size * scale
                    # horizontal row with arrow height 3
                    if h < 3 or w < (size + 1 + dist + 1 + tgt_w + 2):
                        continue

                    src_top = int(rng.integers(0, h - 2))
                    # ensure arrow and source fit
                    if src_top > h - 3:
                        continue
                    hi = w - (size + dist + tgt_w)
                    if hi <= 2:
                        continue
                    src_left = int(rng.integers(2, hi))

                    wall_col = src_left + size + dist
                    wall_row = src_top
                    wall_len = size

                    wall_cells = _paint_wall_segment(
                        grid,
                        used,
                        wall_col,
                        wall_row,
                        wall_len,
                        'v',
                        WALL_COLOR,
                    )
                    if wall_cells is None:
                        continue

                    src_cells = _paint_source_square(
                        grid,
                        used,
                        src_top,
                        src_left,
                        size,
                        color,
                    )
                    if src_cells is None:
                        continue

                    arrow_cells = _paint_arrow(
                        grid,
                        used,
                        src_top,
                        src_left - 2,
                        ARROW_TEMPLATES[(0, 1)],
                        ARROW_COLOR,
                    )
                    if arrow_cells is None:
                        continue

                    # guard projection would stay in grid
                    center_r = (src_top + src_top + size - 1) / 2
                    dest_r = int(round(center_r - (size * scale - 1) / 2))
                    dest_c = wall_col + 1
                    if dest_r < 0 or dest_r + size * scale > h or dest_c + tgt_w > w:
                        # rollback
                        for rr, cc in wall_cells + src_cells + arrow_cells:
                            grid[rr, cc] = BACKGROUND
                            used[rr, cc] = False
                        continue

                elif dirv == (0, -1):  # source right, wall left, arrow right
                    tgt_w = size * scale
                    if h < 3 or w < (size + 1 + dist + 1 + tgt_w + 2):
                        continue

                    src_top = int(rng.integers(0, h - 2))
                    if src_top > h - 3:
                        continue
                    src_left = int(rng.integers(0, w - size))
                    if src_left + size + 2 + dist + 1 + tgt_w >= w:
                        continue

                    wall_col = src_left - dist - 1
                    if wall_col < 0:
                        continue
                    wall_row = src_top
                    wall_len = size

                    wall_cells = _paint_wall_segment(
                        grid,
                        used,
                        wall_col,
                        wall_row,
                        wall_len,
                        'v',
                        WALL_COLOR,
                    )
                    if wall_cells is None:
                        continue

                    src_cells = _paint_source_square(
                        grid,
                        used,
                        src_top,
                        src_left,
                        size,
                        color,
                    )
                    if src_cells is None:
                        continue

                    arrow_cells = _paint_arrow(
                        grid,
                        used,
                        src_top,
                        src_left + size,
                        ARROW_TEMPLATES[(0, -1)],
                        ARROW_COLOR,
                    )
                    if arrow_cells is None:
                        continue

                    center_r = (src_top + src_top + size - 1) / 2
                    dest_r = int(round(center_r - (size * scale - 1) / 2))
                    dest_c = wall_col - tgt_w
                    if dest_r < 0 or dest_r + size * scale > h or dest_c < 0:
                        for rr, cc in wall_cells + src_cells + arrow_cells:
                            grid[rr, cc] = BACKGROUND
                            used[rr, cc] = False
                        continue

                elif dirv == (1, 0):  # source up, wall down, arrow up
                    tgt_h = size * scale
                    if w < 3 or h < (size + 1 + dist + 1 + tgt_h + 1):
                        continue

                    src_left = int(rng.integers(0, w - 2))
                    if src_left > w - 3:
                        continue
                    src_top = int(rng.integers(1, h - size))

                    wall_row = src_top + size + dist
                    if wall_row >= h:
                        continue
                    wall_col = src_left
                    wall_len = size

                    wall_cells = _paint_wall_segment(
                        grid,
                        used,
                        wall_row,
                        wall_col,
                        wall_len,
                        'h',
                        WALL_COLOR,
                    )
                    if wall_cells is None:
                        continue

                    src_cells = _paint_source_square(
                        grid,
                        used,
                        src_top,
                        src_left,
                        size,
                        color,
                    )
                    if src_cells is None:
                        continue

                    arrow_cells = _paint_arrow(
                        grid,
                        used,
                        src_top - 1,
                        src_left,
                        ARROW_TEMPLATES[(1, 0)],
                        ARROW_COLOR,
                    )
                    if arrow_cells is None:
                        continue

                    center_c = (src_left + src_left + size - 1) / 2
                    dest_r = wall_row + 1
                    dest_c = int(round(center_c - (size * scale - 1) / 2))
                    if dest_c < 0 or dest_c + size * scale > w or dest_r + tgt_h > h:
                        for rr, cc in wall_cells + src_cells + arrow_cells:
                            grid[rr, cc] = BACKGROUND
                            used[rr, cc] = False
                        continue

                else:  # dirv == (-1, 0), source down, wall up, arrow down
                    tgt_h = size * scale
                    if w < 3 or h < (size + 1 + dist + 1 + tgt_h + 1):
                        continue

                    src_left = int(rng.integers(0, w - 2))
                    if src_left > w - 3:
                        continue
                    src_top = int(rng.integers(0, h - size - 1))
                    if src_top <= dist:
                        continue

                    wall_row = src_top - dist - 1
                    if wall_row < 0:
                        continue
                    wall_col = src_left
                    wall_len = size

                    wall_cells = _paint_wall_segment(
                        grid,
                        used,
                        wall_row,
                        wall_col,
                        wall_len,
                        'h',
                        WALL_COLOR,
                    )
                    if wall_cells is None:
                        continue

                    src_cells = _paint_source_square(
                        grid,
                        used,
                        src_top,
                        src_left,
                        size,
                        color,
                    )
                    if src_cells is None:
                        continue

                    arrow_cells = _paint_arrow(
                        grid,
                        used,
                        src_top + size - 1,
                        src_left,
                        ARROW_TEMPLATES[(-1, 0)],
                        ARROW_COLOR,
                    )
                    if arrow_cells is None:
                        continue

                    center_c = (src_left + src_left + size - 1) / 2
                    dest_r = wall_row - tgt_h
                    dest_c = int(round(center_c - (size * scale - 1) / 2))
                    if dest_r < 0 or dest_c < 0 or dest_c + size * scale > w:
                        for rr, cc in wall_cells + src_cells + arrow_cells:
                            grid[rr, cc] = BACKGROUND
                            used[rr, cc] = False
                        continue

                # pair successfully placed
                placed += 1
                break

        if placed >= 1:
            return grid

    # fallback
    return np.full((20, 20), BACKGROUND, dtype=np.int64)
