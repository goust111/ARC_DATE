import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_COUNTS = (4, 6, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _closed_black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                comps.append(cells)
    return comps


def _count_runs(values: list[int]) -> int:
    if not values:
        return 0
    values.sort()
    runs = 1
    prev = values[0]
    for value in values[1:]:
        if value != prev + 1:
            runs += 1
        prev = value
    return runs


def _region_side_count(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    h, w = grid.shape
    region = set(cells)
    horiz: dict[tuple[int, int], list[int]] = defaultdict(list)
    vert: dict[tuple[int, int], list[int]] = defaultdict(list)

    for r, c in cells:
        if r == 0 or (r - 1, c) not in region:
            color = -1 if r == 0 else int(grid[r - 1, c])
            if color != BLACK:
                horiz[(r, color)].append(c)
        if r == h - 1 or (r + 1, c) not in region:
            color = -1 if r == h - 1 else int(grid[r + 1, c])
            if color != BLACK:
                horiz[(r + 1, color)].append(c)
        if c == 0 or (r, c - 1) not in region:
            color = -1 if c == 0 else int(grid[r, c - 1])
            if color != BLACK:
                vert[(c, color)].append(r)
        if c == w - 1 or (r, c + 1) not in region:
            color = -1 if c == w - 1 else int(grid[r, c + 1])
            if color != BLACK:
                vert[(c + 1, color)].append(r)

    sides = 0
    for values in horiz.values():
        sides += _count_runs(values)
    for values in vert.values():
        sides += _count_runs(values)
    return sides


def _legend_mapping(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != GREY or (c > 0 and int(grid[r, c - 1]) == GREY):
                c += 1
                continue
            end = c
            while end + 1 < w and int(grid[r, end + 1]) == GREY:
                end += 1
            if (r > 0 and np.any(grid[r - 1, c : end + 1] == GREY)) or (r + 1 < h and np.any(grid[r + 1, c : end + 1] == GREY)):
                c = end + 1
                continue
            fill = None
            for cc in range(end + 1, w):
                value = int(grid[r, cc])
                if value == BLACK:
                    continue
                if value == GREY:
                    break
                fill = value
                break
            if fill is not None:
                mapping[end - c + 1] = fill
            c = end + 1
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed black regions according to the legend's side-count -> fill-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _legend_mapping(g)
    out = g.copy()
    for cells in _closed_black_components(g):
        side_count = _region_side_count(g, cells)
        fill = mapping.get(side_count)
        if fill is None:
            continue
        for r, c in cells:
            out[r, c] = fill
    return out


def _rotate_vertices(vertices: list[tuple[int, int]], h: int, w: int, turns: int):
    pts = list(vertices)
    hh, ww = h, w
    for _ in range(turns % 4):
        pts = [(c, hh - 1 - r) for r, c in pts]
        hh, ww = ww, hh
    return pts, hh, ww


def _draw_path(vertices: list[tuple[int, int]], colors: list[int], h: int, w: int, *, closed: bool) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    pairs = list(zip(vertices, vertices[1:] + ([vertices[0]] if closed else [])))
    for (r0, c0), (r1, c1), color in zip((a for a, _ in pairs), (b for _, b in pairs), colors):
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            grid[r0, lo : hi + 1] = color
        else:
            lo, hi = sorted((r0, r1))
            grid[lo : hi + 1, c0] = color
    return grid


def _random_side_colors(count: int, palette: list[int]) -> list[int]:
    color = random.choice(palette)
    return [color] * count


def _make_shape(side_count: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(120):
        if side_count == 4:
            h = random.randint(5, 10)
            w = random.randint(5, 11)
            vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
        elif side_count == 6:
            h = random.randint(8, 12)
            w = random.randint(8, 12)
            cut_h = random.randint(3, h - 4)
            cut_w = random.randint(3, w - 4)
            vertices = [
                (0, 0),
                (0, w - cut_w - 1),
                (cut_h, w - cut_w - 1),
                (cut_h, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
            ]
        else:
            h = random.randint(10, 14)
            w = random.randint(10, 14)
            cut_h1 = random.randint(3, h - 5)
            cut_w1 = random.randint(3, w - 5)
            cut_h2 = random.randint(3, h - 5)
            cut_w2 = random.randint(3, w - 5)
            vertices = [
                (0, 0),
                (0, w - cut_w1 - 1),
                (cut_h1, w - cut_w1 - 1),
                (cut_h1, w - 1),
                (h - 1, w - 1),
                (h - 1, cut_w2),
                (h - cut_h2 - 1, cut_w2),
                (h - cut_h2 - 1, 0),
            ]

        vertices, h, w = _rotate_vertices(vertices, h, w, random.randrange(4))
        colors = _random_side_colors(side_count, palette)
        shape = _draw_path(vertices, colors, h, w, closed=True)
        comps = _closed_black_components(shape)
        if len(comps) != 1:
            continue
        if _region_side_count(shape, comps[0]) != side_count:
            continue
        return shape
    return None


def _make_open_distractor(palette: list[int]) -> np.ndarray | None:
    for _ in range(80):
        if random.random() < 0.5:
            length = random.randint(4, 8)
            if random.random() < 0.5:
                h, w = 1, length
                vertices = [(0, 0), (0, length - 1)]
            else:
                h, w = length, 1
                vertices = [(0, 0), (length - 1, 0)]
        else:
            w = random.randint(4, 8)
            h = random.randint(4, 8)
            vertices = [(0, 0), (0, w - 1), (h - 1, w - 1)]
            vertices, h, w = _rotate_vertices(vertices, h, w, random.randrange(4))
        colors = _random_side_colors(len(vertices) - 1, palette)
        shape = _draw_path(vertices, colors, h, w, closed=False)
        if _closed_black_components(shape):
            continue
        return shape
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _place_object(
    canvas: np.ndarray,
    obj: np.ndarray,
    used_boxes: list[tuple[int, int, int, int]],
    blocked_box: tuple[int, int, int, int],
) -> bool:
    h, w = canvas.shape
    oh, ow = obj.shape
    min_top, max_top = 1, h - oh - 2
    min_left, max_left = 1, w - ow - 2
    if min_top > max_top or min_left > max_left:
        return False
    for _ in range(300):
        top = random.randint(min_top, max_top)
        left = random.randint(min_left, max_left)
        box = (top, left, top + oh - 1, left + ow - 1)
        if _boxes_overlap(box, blocked_box, pad=1):
            continue
        if any(_boxes_overlap(box, other, pad=1) for other in used_boxes):
            continue
        patch = canvas[top : top + oh, left : left + ow]
        mask = obj != BLACK
        if np.any(patch[mask] != BLACK):
            continue
        patch[mask] = obj[mask]
        used_boxes.append(box)
        return True
    return False


def _draw_legend(canvas: np.ndarray, entries: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    h, w = canvas.shape
    height = 2 * len(entries) - 1
    max_count = max(count for count, _ in entries)
    top = h - height
    left = w - (max_count + 2)
    for idx, (count, color) in enumerate(entries):
        r = top + 2 * idx
        canvas[r, left : left + count] = GREY
        canvas[r, left + count + 1] = color
    return top, left, h - 1, w - 1


def _gen_attempt() -> np.ndarray | None:
    legend_size = random.choice([2, 3])
    legend_counts = random.sample(list(SIDE_COUNTS), legend_size)
    fill_colors = random.sample(list(COLOR_POOL), legend_size)
    legend_entries = list(zip(legend_counts, fill_colors))
    random.shuffle(legend_entries)

    line_palette = [color for color in COLOR_POOL if color not in fill_colors]
    if len(line_palette) < 2:
        return None

    counts = list(legend_counts)
    extra = random.randint(0, 2)
    for _ in range(extra):
        counts.append(random.choice(legend_counts))
    random.shuffle(counts)

    shapes: list[np.ndarray] = []
    for count in counts:
        shape = _make_shape(count, line_palette)
        if shape is None:
            return None
        shapes.append(shape)

    open_lines: list[np.ndarray] = []
    if random.random() < 0.5:
        distractor = _make_open_distractor(line_palette)
        if distractor is None:
            return None
        open_lines.append(distractor)

    max_shape_h = max(obj.shape[0] for obj in shapes + open_lines)
    max_shape_w = max(obj.shape[1] for obj in shapes + open_lines)
    h = random.randint(max(22, max_shape_h + 8), max(30, max_shape_h + 10))
    w = random.randint(max(26, max_shape_w + 12), max(36, max_shape_w + 14))
    canvas = np.zeros((h, w), dtype=int)

    blocked_box = _draw_legend(canvas, legend_entries)
    used_boxes: list[tuple[int, int, int, int]] = [blocked_box]

    random.shuffle(shapes)
    for shape in shapes:
        if not _place_object(canvas, shape, used_boxes, blocked_box):
            return None

    for line in open_lines:
        if not _place_object(canvas, line, used_boxes, blocked_box):
            return None

    closed = _closed_black_components(canvas)
    if len(closed) != len(shapes):
        return None
    observed = sorted(_region_side_count(canvas, cells) for cells in closed)
    if observed != sorted(counts):
        return None
    if not _legend_mapping(canvas):
        return None

    out = main(canvas)
    if np.array_equal(out, canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
