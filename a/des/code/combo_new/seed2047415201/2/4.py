import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEGEND_COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE)
MAX_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _are_neighbors(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    same_vertical_span = _intervals_overlap(ar0, ar1, br0, br1)
    same_horizontal_span = _intervals_overlap(ac0, ac1, bc0, bc1)

    # The generator uses a one-cell black seam. The zero-gap case keeps the
    # solver valid for already-separated rectangles that directly touch.
    if same_vertical_span and (0 <= bc0 - ac1 - 1 <= 1 or 0 <= ac0 - bc1 - 1 <= 1):
        return True
    if same_horizontal_span and (0 <= br0 - ar1 - 1 <= 1 or 0 <= ar0 - br1 - 1 <= 1):
        return True
    return False


def _legend_from_grid(g: np.ndarray) -> tuple[int, ...]:
    marker = np.argwhere((g != BLACK) & (g != GREY))
    assert marker.shape[0] >= 1, "expected a color marker column"

    cols = {int(c) for _, c in marker}
    assert len(cols) == 1, "all non-grey markers must be in one column"
    entries = sorted((int(r), int(g[r, c])) for r, c in marker)
    assert len(entries) >= 5, "expected at least five ordered marker colors"

    colors = tuple(color for _, color in entries[:5])
    assert len(set(colors)) == 5, "marker colors must be distinct"
    return colors


def _grey_rectangles(g: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    rects = []
    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert coords.shape[0] == height * width, "each grey object must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY)
        rects.append((coords, (r0, c0, r1, c1)))
    assert rects, "expected grey rectangles"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle by the number of neighboring rectangles it has, using the ordered marker column as the palette.

    Input structure:
    - The background is black.
    - Grey solid rectangles form a compact tiled panel; one-cell black seams make
      the rectangle boundaries visible while preserving the logical shared-edge layout.
    - A single marker column on the right lists the colors for neighbor counts
      one through five, from top to bottom.

    Rule:
    - Find every solid grey rectangle.
    - Count how many different grey rectangles share a horizontal or vertical
      boundary with it, allowing the visible one-cell black seam between them.
    - Recolor the rectangle with the marker color at position count - 1.
    - Keep the black background, seams, and marker column unchanged.

    Key insight:
    - The original grey color carries no identity; the color is determined by
      object topology, and the right-side marker column is an ordinal legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend = _legend_from_grid(g)
    rects = _grey_rectangles(g)

    out = g.copy()
    for i, (_, bbox_i) in enumerate(rects):
        count = 0
        for j, (_, bbox_j) in enumerate(rects):
            if i != j and _are_neighbors(bbox_i, bbox_j):
                count += 1
        assert 1 <= count <= len(legend), "neighbor count must be represented by the marker column"
        r0, c0, r1, c1 = bbox_i
        out[r0 : r1 + 1, c0 : c1 + 1] = legend[count - 1]

    return out


def _logical_degrees(rects: list[tuple[int, int, int, int]]) -> list[int]:
    degrees = [0 for _ in rects]
    for i, a in enumerate(rects):
        ax0, ay0, ax1, ay1 = a
        for j, b in enumerate(rects[i + 1 :], start=i + 1):
            bx0, by0, bx1, by1 = b
            adjacent = False
            if (ax1 == bx0 or bx1 == ax0) and max(ay0, by0) < min(ay1, by1):
                adjacent = True
            if (ay1 == by0 or by1 == ay0) and max(ax0, bx0) < min(ax1, bx1):
                adjacent = True
            if adjacent:
                degrees[i] += 1
                degrees[j] += 1
    return degrees


def _split_partition(width: int, height: int, n_rects: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, width, height)]
    for _ in range(n_rects - 1):
        candidates = []
        for idx, (x0, y0, x1, y1) in enumerate(rects):
            if x1 - x0 >= 2:
                candidates.append((idx, "v"))
            if y1 - y0 >= 2:
                candidates.append((idx, "h"))
        if not candidates:
            return None
        idx, axis = random.choice(candidates)
        x0, y0, x1, y1 = rects.pop(idx)
        if axis == "v":
            cut = random.randint(x0 + 1, x1 - 1)
            rects.append((x0, y0, cut, y1))
            rects.append((cut, y0, x1, y1))
        else:
            cut = random.randint(y0 + 1, y1 - 1)
            rects.append((x0, y0, x1, cut))
            rects.append((x0, cut, x1, y1))
    random.shuffle(rects)
    return rects


def _base_partition() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    rects = [
        (3, 0, 4, 1),
        (3, 1, 4, 2),
        (2, 0, 3, 2),
        (0, 2, 4, 3),
        (0, 3, 4, 4),
        (0, 0, 2, 1),
        (0, 1, 2, 2),
    ]
    return 4, 4, rects


def _transform_partition(
    width: int, height: int, rects: list[tuple[int, int, int, int]]
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    op = random.choice(("id", "flip_x", "flip_y", "rot90", "rot180", "rot270"))
    out = []

    for x0, y0, x1, y1 in rects:
        if op == "id":
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            nw, nh = width, height
        elif op == "flip_x":
            nx0, nx1 = width - x1, width - x0
            ny0, ny1 = y0, y1
            nw, nh = width, height
        elif op == "flip_y":
            nx0, nx1 = x0, x1
            ny0, ny1 = height - y1, height - y0
            nw, nh = width, height
        elif op == "rot90":
            nx0, nx1 = height - y1, height - y0
            ny0, ny1 = x0, x1
            nw, nh = height, width
        elif op == "rot180":
            nx0, nx1 = width - x1, width - x0
            ny0, ny1 = height - y1, height - y0
            nw, nh = width, height
        else:
            nx0, nx1 = y0, y1
            ny0, ny1 = width - x1, width - x0
            nw, nh = height, width
        out.append((nx0, ny0, nx1, ny1))
    return nw, nh, out


def _sample_partition() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    for _ in range(200):
        width = random.randint(4, 6)
        height = random.randint(4, 6)
        n_rects = random.randint(7, min(11, width * height))
        rects = _split_partition(width, height, n_rects)
        if rects is None:
            continue
        degrees = _logical_degrees(rects)
        if set(range(1, 6)).issubset(degrees) and max(degrees) <= 5:
            return _transform_partition(width, height, rects)
    width, height, rects = _base_partition()
    return _transform_partition(width, height, rects)


def _starts(sizes: list[int], offset: int) -> list[int]:
    starts = []
    cur = offset
    for size in sizes:
        starts.append(cur)
        cur += size + 1
    return starts


def _physical_bbox(
    rect: tuple[int, int, int, int],
    x_starts: list[int],
    y_starts: list[int],
    unit_w: list[int],
    unit_h: list[int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    c0 = x_starts[x0]
    c1 = x_starts[x1 - 1] + unit_w[x1 - 1] - 1
    r0 = y_starts[y0]
    r1 = y_starts[y1 - 1] + unit_h[y1 - 1] - 1
    return r0, c0, r1, c1


def _expected_grid(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    degrees: list[int],
    bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for degree, (r0, c0, r1, c1) in zip(degrees, bboxes):
        out[r0 : r1 + 1, c0 : c1 + 1] = LEGEND_COLORS[degree - 1]
    return out


def _gen_attempt() -> np.ndarray | None:
    logical_w, logical_h, rects = _sample_partition()
    degrees = _logical_degrees(rects)
    if not set(range(1, 6)).issubset(degrees) or max(degrees) > 5:
        return None

    unit_w = [random.randint(2, 4) for _ in range(logical_w)]
    unit_h = [random.randint(2, 4) for _ in range(logical_h)]
    content_w = sum(unit_w) + logical_w - 1
    content_h = sum(unit_h) + logical_h - 1

    top = random.randint(0, 2)
    left = random.randint(0, 2)
    bottom = random.randint(1, 3)
    right = random.randint(1, 2)
    legend_gap = random.randint(2, 3)

    height = top + content_h + bottom
    width = left + content_w + legend_gap + 1 + right
    if height < 5 or height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    x_starts = _starts(unit_w, left)
    y_starts = _starts(unit_h, top)

    bboxes = []
    for rect in rects:
        r0, c0, r1, c1 = _physical_bbox(rect, x_starts, y_starts, unit_w, unit_h)
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        bboxes.append((r0, c0, r1, c1))

    legend_col = left + content_w + legend_gap
    legend_start = random.randint(0, height - len(LEGEND_COLORS))
    for k, color in enumerate(LEGEND_COLORS):
        grid[legend_start + k, legend_col] = color

    expected = _expected_grid(grid, rects, degrees, bboxes)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
