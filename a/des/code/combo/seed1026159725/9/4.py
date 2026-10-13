import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    color: int
    orient: str
    fixed: int
    start: int
    end: int

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        cells = segment_cells(self)
        return cells[0], cells[-1]


def dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


def infer_segment(color: int, coords: np.ndarray) -> Segment:
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert len(coords) >= 2, "every visible line needs at least two pixels"
    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.all(rows == rows[0]):
        return Segment(color=color, orient="h", fixed=int(rows[0]), start=int(cols.min()), end=int(cols.max()))
    if np.all(cols == cols[0]):
        return Segment(color=color, orient="v", fixed=int(cols[0]), start=int(rows.min()), end=int(rows.max()))

    diffs = rows - cols
    if np.all(diffs == diffs[0]):
        return Segment(color=color, orient="d1", fixed=int(diffs[0]), start=int(rows.min()), end=int(rows.max()))

    sums = rows + cols
    if np.all(sums == sums[0]):
        return Segment(color=color, orient="d2", fixed=int(sums[0]), start=int(rows.min()), end=int(rows.max()))

    raise AssertionError(f"color {color} is not a straight horizontal/vertical/diagonal line")


def segment_cells(seg: Segment) -> list[tuple[int, int]]:
    if seg.orient == "h":
        return [(seg.fixed, c) for c in range(seg.start, seg.end + 1)]
    if seg.orient == "v":
        return [(r, seg.fixed) for r in range(seg.start, seg.end + 1)]
    if seg.orient == "d1":
        return [(r, r - seg.fixed) for r in range(seg.start, seg.end + 1)]
    if seg.orient == "d2":
        return [(r, seg.fixed - r) for r in range(seg.start, seg.end + 1)]
    raise AssertionError(f"unknown orientation {seg.orient}")


def extract_segments(grid: np.ndarray) -> tuple[int, list[Segment]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = dominant_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected at least one foreground line"

    segments: list[Segment] = []
    for color in colors:
        coords = np.argwhere(g == color)
        seg = infer_segment(color, coords)
        segments.append(seg)
    return bg, segments


def ordering_from_grid(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    _, segments = extract_segments(g)
    n = len(segments)

    color_to_idx = {seg.color: idx for idx, seg in enumerate(segments)}
    cell_to_segments: dict[tuple[int, int], list[int]] = {}
    for idx, seg in enumerate(segments):
        for cell in segment_cells(seg):
            cell_to_segments.setdefault(cell, []).append(idx)

    edges = {idx: set() for idx in range(n)}
    crossing_count = 0
    for (r, c), ids in cell_to_segments.items():
        if len(ids) == 1:
            continue
        assert len(ids) == 2, "only pairwise crossings are supported"
        crossing_count += 1
        visible_color = int(g[r, c])
        assert visible_color in color_to_idx, "crossing must show one of the line colors"
        top = color_to_idx[visible_color]
        assert top in ids, "visible color at a crossing must belong to one of the crossing lines"
        for other in ids:
            if other != top:
                edges[other].add(top)

    if n > 1:
        assert crossing_count > 0, "multiple lines need at least one crossing"

    indeg = {idx: 0 for idx in range(n)}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst != src
            indeg[dst] += 1

    available = [idx for idx in range(n) if indeg[idx] == 0]
    order: list[int] = []
    while available:
        assert len(available) == 1, "crossing constraints must imply a unique layer order"
        node = available.pop()
        order.append(node)
        for nxt in edges[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)

    assert len(order) == n, "layer constraints must form an acyclic total order"
    return [segments[idx].color for idx in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover full straight line segments, infer pairwise over/under relations from
    visible crossing colors, and output the unique layer order as a single column.

    The returned column is arranged spatially from top layer at the top row to
    bottom layer at the bottom row, so the bottom-to-top order can be read by
    looking upward.
    """
    bottom_to_top = ordering_from_grid(np.asarray(grid, dtype=int))
    return np.asarray(bottom_to_top[::-1], dtype=int).reshape(-1, 1)


MIN_GRID = 16
MAX_GRID = 24
MIN_LINES = 3
MAX_LINES = 5
MIN_LEN = 6
MAX_LEN = 11
MAX_ATTEMPTS = 400
INNER_MARGIN = 1


def sample_segment(color: int, h: int, w: int, rng: random.Random) -> Segment | None:
    orient = rng.choice(("h", "v", "d1", "d2"))

    if orient == "h":
        max_len = min(MAX_LEN, w - 2 * INNER_MARGIN)
        if max_len < MIN_LEN:
            return None
        length = rng.randint(MIN_LEN, max_len)
        r = rng.randint(INNER_MARGIN, h - 1 - INNER_MARGIN)
        c0 = rng.randint(INNER_MARGIN, w - INNER_MARGIN - length)
        return Segment(color=color, orient=orient, fixed=r, start=c0, end=c0 + length - 1)

    if orient == "v":
        max_len = min(MAX_LEN, h - 2 * INNER_MARGIN)
        if max_len < MIN_LEN:
            return None
        length = rng.randint(MIN_LEN, max_len)
        c = rng.randint(INNER_MARGIN, w - 1 - INNER_MARGIN)
        r0 = rng.randint(INNER_MARGIN, h - INNER_MARGIN - length)
        return Segment(color=color, orient=orient, fixed=c, start=r0, end=r0 + length - 1)

    if orient == "d1":
        k_low = 1 - (w - 2)
        k_high = (h - 2) - 1
        ks = list(range(k_low, k_high + 1))
        rng.shuffle(ks)
        for k in ks:
            rmin = max(1, 1 + k)
            rmax = min(h - 2, (w - 2) + k)
            span = rmax - rmin + 1
            if span < MIN_LEN:
                continue
            length = rng.randint(MIN_LEN, min(MAX_LEN, span))
            r0 = rng.randint(rmin, rmax - length + 1)
            return Segment(color=color, orient=orient, fixed=k, start=r0, end=r0 + length - 1)
        return None

    s_low = 2
    s_high = (h - 2) + (w - 2)
    sums = list(range(s_low, s_high + 1))
    rng.shuffle(sums)
    for s in sums:
        rmin = max(1, s - (w - 2))
        rmax = min(h - 2, s - 1)
        span = rmax - rmin + 1
        if span < MIN_LEN:
            continue
        length = rng.randint(MIN_LEN, min(MAX_LEN, span))
        r0 = rng.randint(rmin, rmax - length + 1)
        return Segment(color=color, orient=orient, fixed=s, start=r0, end=r0 + length - 1)
    return None


def render_segments(segments: list[Segment], bg: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for seg in segments:
        for r, c in segment_cells(seg):
            grid[r, c] = seg.color
    return grid


def sample_layered_segments(h: int, w: int, colors: list[int], rng: random.Random) -> list[Segment] | None:
    segments: list[Segment] = []
    used_points: set[tuple[int, int]] = set()

    for idx, color in enumerate(colors):
        for _ in range(200):
            seg = sample_segment(color, h, w, rng)
            if seg is None:
                continue
            seg_cells = set(segment_cells(seg))
            intersections: list[tuple[int, int]] = []
            crosses_prev = idx == 0
            valid = True

            for old_idx, old in enumerate(segments):
                old_cells = set(segment_cells(old))
                overlap = seg_cells & old_cells
                if len(overlap) > 1:
                    valid = False
                    break
                if len(overlap) == 1:
                    point = next(iter(overlap))
                    if point in seg.endpoints or point in old.endpoints:
                        valid = False
                        break
                    intersections.append(point)
                    if old_idx == idx - 1:
                        crosses_prev = True

            if not valid or not crosses_prev:
                continue
            if len(set(intersections)) != len(intersections):
                continue
            if any(point in used_points for point in intersections):
                continue

            segments.append(seg)
            used_points.update(intersections)
            break
        else:
            return None

    return segments


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    line_count = rng.randint(MIN_LINES, MAX_LINES)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    rng.shuffle(palette)
    colors = palette[:line_count]

    segments = sample_layered_segments(h, w, colors, rng)
    if segments is None:
        return None

    grid = render_segments(segments, bg, (h, w))
    if dominant_color(grid) != bg:
        return None

    for seg in segments:
        visible = int(np.count_nonzero(grid == seg.color))
        if visible < 2:
            return None

    expected = np.asarray(colors[::-1], dtype=int).reshape(-1, 1)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-line input")
