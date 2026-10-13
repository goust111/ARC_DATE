import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400
RECT_RETRIES = 300
CLUSTER_RETRIES = 300
SMALL_SHAPES = {
    1: [((0, 0),)],
    2: [((0, 0), (0, 1)), ((0, 0), (1, 0))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
    ],
}


def _interior_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 - 1) * int(c1 - c0 - 1)


def _neighbors4(grid: np.ndarray, r: int, c: int, color: int) -> set[str]:
    dirs = set()
    if r > 0 and int(grid[r - 1, c]) == color:
        dirs.add("U")
    if r + 1 < grid.shape[0] and int(grid[r + 1, c]) == color:
        dirs.add("D")
    if c > 0 and int(grid[r, c - 1]) == color:
        dirs.add("L")
    if c + 1 < grid.shape[1] and int(grid[r, c + 1]) == color:
        dirs.add("R")
    return dirs


def _border_is_color(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
) -> bool:
    r0, c0, r1, c1 = rect
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _find_rectangles(grid: np.ndarray, frame_color: int = GREY) -> list[tuple[int, int, int, int]]:
    tl: set[tuple[int, int]] = set()
    tr: set[tuple[int, int]] = set()
    bl: set[tuple[int, int]] = set()
    br: set[tuple[int, int]] = set()

    for r, c in np.argwhere(grid == frame_color):
        dirs = _neighbors4(grid, int(r), int(c), frame_color)
        if dirs == {"R", "D"}:
            tl.add((int(r), int(c)))
        elif dirs == {"L", "D"}:
            tr.add((int(r), int(c)))
        elif dirs == {"R", "U"}:
            bl.add((int(r), int(c)))
        elif dirs == {"L", "U"}:
            br.add((int(r), int(c)))

    rects = set()
    for r0, c0 in tl:
        for r1, c1 in br:
            if r1 <= r0 + 1 or c1 <= c0 + 1:
                continue
            if (r0, c1) not in tr or (r1, c0) not in bl:
                continue
            rect = (r0, c0, r1, c1)
            if not _border_is_color(grid, rect, frame_color):
                continue
            if not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != frame_color):
                continue
            rects.add(rect)

    return sorted(rects)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each scattered color globally, rank-match colors to hollow rectangles, then fill them in rank order so later ranks overwrite earlier overlaps.

    Input structure:
    - The background is black and the rectangle frames are grey hollow boxes.
    - Several non-grey colors appear as multiple disconnected scattered clusters.
    - The number of scattered colors equals the number of grey hollow rectangles, and rectangles may partially overlap.

    Rule:
    - Count the total number of pixels of each scattered color across all of its clusters.
    - Sort colors by total count and sort rectangles by interior area; the k-th smallest color claims the k-th smallest rectangle.
    - Starting from the smallest rank, fill each claimed rectangle interior with its color, preserving all grey frame pixels; later fills overwrite earlier colors only on non-grey overlapping interior cells.
    - Remove all original scattered clusters.

    Key insight:
    - The same rank controls both pairing and drawing order, so overlap ownership is temporal rather than purely geometric: the larger rank wins only where interiors overlap and no grey frame blocks the overwrite.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)]
    rects = _find_rectangles(g, GREY)

    assert colors, "expected at least one scattered color"
    assert rects, "expected at least one hollow rectangle"
    assert len(colors) == len(rects), "number of colors must match number of rectangles"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    areas = [_interior_area(rect) for rect in rects]
    assert len(set(counts.values())) == len(colors), "color totals must rank uniquely"
    assert len(set(areas)) == len(rects), "rectangle areas must rank uniquely"

    ranked_colors = sorted(colors, key=lambda color: counts[color])
    ranked_rects = sorted(rects, key=_interior_area)

    out = np.where(g == GREY, GREY, BLACK)
    for color, rect in zip(ranked_colors, ranked_rects):
        r0, c0, r1, c1 = rect
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior != GREY] = color
    return out


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _interior_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0] + 1, b[0] + 1)
    c0 = max(a[1] + 1, b[1] + 1)
    r1 = min(a[2] - 1, b[2] - 1)
    c1 = min(a[3] - 1, b[3] - 1)
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _corners(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _point_on_border(rect: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    if not (r0 <= r <= r1 and c0 <= c <= c1):
        return False
    return r in (r0, r1) or c in (c0, c1)


def _rect_has_conflicting_corner(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return any(_point_on_border(b, p) for p in _corners(a)) or any(
        _point_on_border(a, p) for p in _corners(b)
    )


def _expanded_mask(
    shape: tuple[int, int],
    rects: Iterable[tuple[int, int, int, int]],
    pad: int,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects:
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h - 1, r1 + pad)
        cc1 = min(w - 1, c1 + pad)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _sample_rect_dims() -> tuple[int, int]:
    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 6)
    return inner_h, inner_w


def _sample_rect(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    inner_h, inner_w = _sample_rect_dims()
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(1, w - outer_w - 1)
    return (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)


def _sample_overlap_pair(shape: tuple[int, int]) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    for _ in range(RECT_RETRIES):
        a = _sample_rect(shape)
        ah = a[2] - a[0] + 1
        aw = a[3] - a[1] + 1

        inner_h2, inner_w2 = _sample_rect_dims()
        outer_h2 = inner_h2 + 2
        outer_w2 = inner_w2 + 2

        max_dr = max(2, min(ah - 3, 4))
        max_dc = max(2, min(aw - 3, 4))
        dr = random.randint(2, max_dr)
        dc = random.randint(2, max_dc)
        b = (a[0] + dr, a[1] + dc, a[0] + dr + outer_h2 - 1, a[1] + dc + outer_w2 - 1)
        if b[2] >= h - 1 or b[3] >= w - 1:
            continue
        if _interior_area(a) == _interior_area(b):
            continue
        if _bbox_overlap(a, b) == 0:
            continue
        overlap = _interior_overlap(a, b)
        if overlap < 2:
            continue
        if overlap > max(4, min(_interior_area(a), _interior_area(b)) // 2):
            continue
        if _rect_has_conflicting_corner(a, b):
            continue
        if len({a[0], a[2], b[0], b[2]}) < 4:
            continue
        if len({a[1], a[3], b[1], b[3]}) < 4:
            continue
        if b[2] < a[2] and b[3] < a[3]:
            continue
        return [a, b]
    return None


def _rectangles_valid(
    rects: list[tuple[int, int, int, int]],
    overlap_pairs_target: int,
) -> bool:
    if len({_interior_area(rect) for rect in rects}) != len(rects):
        return False

    edge_rows = [rect[0] for rect in rects] + [rect[2] for rect in rects]
    edge_cols = [rect[1] for rect in rects] + [rect[3] for rect in rects]
    if len(set(edge_rows)) != len(edge_rows) or len(set(edge_cols)) != len(edge_cols):
        return False

    overlap_pairs = 0
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if _rect_has_conflicting_corner(rect_a, rect_b):
                return False
            outer_overlap = _bbox_overlap(rect_a, rect_b)
            inner_overlap = _interior_overlap(rect_a, rect_b)
            if outer_overlap:
                if inner_overlap < 2:
                    return False
                if inner_overlap > max(4, min(_interior_area(rect_a), _interior_area(rect_b)) // 2):
                    return False
                overlap_pairs += 1
            else:
                rr0 = max(rect_a[0] - 1, rect_b[0] - 1)
                rr1 = min(rect_a[2] + 1, rect_b[2] + 1)
                cc0 = max(rect_a[1] - 1, rect_b[1] - 1)
                cc1 = min(rect_a[3] + 1, rect_b[3] + 1)
                if rr0 <= rr1 and cc0 <= cc1:
                    return False
    return overlap_pairs == overlap_pairs_target


def _generate_rectangles(shape: tuple[int, int], n_rects: int, want_overlap: bool) -> list[tuple[int, int, int, int]] | None:
    overlap_pairs_target = 1 if want_overlap else 0
    for _ in range(RECT_RETRIES):
        if want_overlap:
            pair = _sample_overlap_pair(shape)
            if pair is None:
                continue
            rects = pair[:]
        else:
            rects = []

        for _ in range(RECT_RETRIES):
            if len(rects) >= n_rects:
                break
            rect = _sample_rect(shape)
            if any(_interior_area(rect) == _interior_area(other) for other in rects):
                continue
            if any(_bbox_overlap(rect, other) for other in rects):
                continue
            if any(_rect_has_conflicting_corner(rect, other) for other in rects):
                continue
            if any(
                len({rect[0], rect[2], other[0], other[2]}) < 4
                or len({rect[1], rect[3], other[1], other[3]}) < 4
                for other in rects
            ):
                continue
            expanded_a = (rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1)
            if any(_bbox_overlap(expanded_a, (o[0], o[1], o[2], o[3])) for o in rects):
                continue
            rects.append(rect)
        if len(rects) < n_rects:
            continue

        if not _rectangles_valid(rects, overlap_pairs_target):
            continue

        probe = np.zeros(shape, dtype=int)
        for rect in rects:
            _draw_frame(probe, rect)
        if sorted(rects) != _find_rectangles(probe, GREY):
            continue
        return rects
    return None


def _normalize_shape(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    norm = sorted((r - r0, c - c0) for r, c in pts)
    return tuple(norm)


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    k = random.randint(0, 3)
    for _ in range(k):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if random.random() < 0.5:
        pts[:, 1] *= -1
    return _normalize_shape((tuple(map(int, p)) for p in pts))


def _partition_count(total: int) -> list[int]:
    min_parts = max(2, (total + 2) // 3)
    max_parts = min(4, total)
    n_parts = random.randint(min_parts, max_parts)
    parts = [1] * n_parts
    remaining = total - n_parts
    while remaining > 0:
        idx = random.randrange(n_parts)
        if parts[idx] < 3:
            parts[idx] += 1
            remaining -= 1
    random.shuffle(parts)
    return parts


def _can_place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    cells = []
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c] or int(grid[r, c]) != BLACK:
            return False
        cells.append((r, c))

    for r, c in cells:
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLACK and (rr, cc) not in cells:
                return False
    return True


def _place_shape(grid: np.ndarray, color: int, shape: tuple[tuple[int, int], ...], top: int, left: int, blocked: np.ndarray) -> None:
    h, w = grid.shape
    cells = []
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        grid[r, c] = color
        cells.append((r, c))
    for r, c in cells:
        for dr, dc in DIRS8 + [(0, 0)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _place_scattered_clusters(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    counts: list[int],
) -> bool:
    blocked = _expanded_mask(grid.shape, rects, pad=1)
    blocked[grid == GREY] = True

    for color, total in zip(colors, counts):
        for size in _partition_count(total):
            base_shape = random.choice(SMALL_SHAPES[size])
            shape = _transform_shape(base_shape)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            placed = False
            for _ in range(CLUSTER_RETRIES):
                top = random.randint(0, grid.shape[0] - max_r - 1)
                left = random.randint(0, grid.shape[1] - max_c - 1)
                if not _can_place_shape(grid, shape, top, left, blocked):
                    continue
                _place_shape(grid, color, shape, top, left, blocked)
                placed = True
                break
            if not placed:
                return False
    return True


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    transformed = np.array(grid, copy=True)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    if random.random() < 0.5:
        transformed = np.rot90(transformed, k=random.randint(1, 3))
    return np.asarray(transformed, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    n_rects = random.choice([2, 3])
    want_overlap = random.random() < 0.75

    rects = _generate_rectangles((h, w), n_rects, want_overlap)
    if rects is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        _draw_frame(grid, rect)

    colors = random.sample(NON_SPECIAL_COLORS, n_rects)
    counts = random.sample([4, 5, 6, 7, 8], n_rects)
    if not _place_scattered_clusters(grid, rects, colors, counts):
        return None

    grid = _apply_random_symmetry(grid)

    rects_detected = _find_rectangles(grid, GREY)
    if len(rects_detected) != n_rects:
        return None
    if len(set(_interior_area(rect) for rect in rects_detected)) != n_rects:
        return None

    present_colors = [int(c) for c in np.unique(grid) if int(c) not in (BLACK, GREY)]
    if sorted(present_colors) != sorted(colors):
        return None
    if len({int(np.count_nonzero(grid == color)) for color in present_colors}) != n_rects:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
