import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PATH_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SEGMENT_COUNT_CHOICES = (2, 3, 4)
LEGEND_CORNERS = ("tl", "tr", "bl", "br")
PATTERN_TYPES = ("checker_a", "checker_b", "vstripe", "hstripe")
GRID_SIZE_BY_LOOP_COUNT = {
    1: ((16, 24), (16, 24)),
    2: ((20, 28), (20, 28)),
    3: ((24, 30), (24, 30)),
}
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
SEGMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_cells(side: str, start: int, end: int, rect_h: int, rect_w: int) -> set[tuple[int, int]]:
    if side == "top":
        return {(0, c) for c in range(start, end + 1)}
    if side == "bottom":
        return {(rect_h - 1, c) for c in range(start, end + 1)}
    if side == "left":
        return {(r, 0) for r in range(start, end + 1)}
    if side == "right":
        return {(r, rect_w - 1) for r in range(start, end + 1)}
    raise ValueError(f"unknown side: {side}")


def _segments_are_disconnected(segments: list[set[tuple[int, int]]]) -> bool:
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            if seg_a & seg_b:
                return False
            for r, c in seg_a:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in seg_b:
                        return False
    return True


def _sample_segments(
    rng: random.Random, rect_h: int, rect_w: int, segment_count: int
) -> list[set[tuple[int, int]]] | None:
    side_specs = {
        "top": rect_w,
        "bottom": rect_w,
        "left": rect_h,
        "right": rect_h,
    }
    candidates: dict[str, list[set[tuple[int, int]]]] = {}
    for side, length in side_specs.items():
        segs: list[set[tuple[int, int]]] = []
        starts = sorted(set(v for v in (0, 1, 2) if v < length))
        ends = sorted(set(v for v in (length - 1, length - 2, length - 3) if v >= 0), reverse=True)
        for start in starts:
            for end in ends:
                if start > end or end - start + 1 < 2:
                    continue
                segs.append(_segment_cells(side, start, end, rect_h, rect_w))
        candidates[side] = segs

    sides = list(candidates)
    for _ in range(SEGMENT_ATTEMPTS):
        chosen_sides = rng.sample(sides, segment_count)
        segments = [rng.choice(candidates[side]) for side in chosen_sides]
        if not _segments_are_disconnected(segments):
            continue
        union = set().union(*segments)
        rows = [r for r, _ in union]
        cols = [c for _, c in union]
        if min(rows) != 0 or max(rows) != rect_h - 1:
            continue
        if min(cols) != 0 or max(cols) != rect_w - 1:
            continue
        return segments
    return None


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _make_pattern(pattern_type: str, color_a: int, color_b: int) -> np.ndarray:
    if pattern_type == "checker_a":
        return np.array([[color_a, color_b], [color_b, color_a]], dtype=int)
    if pattern_type == "checker_b":
        return np.array([[color_b, color_a], [color_a, color_b]], dtype=int)
    if pattern_type == "vstripe":
        return np.array([[color_a, color_b], [color_a, color_b]], dtype=int)
    if pattern_type == "hstripe":
        return np.array([[color_a, color_a], [color_b, color_b]], dtype=int)
    raise ValueError(f"unknown pattern type: {pattern_type}")


def _legend_bbox(h: int, w: int, corner: str, entry_count: int) -> tuple[int, int, int, int]:
    legend_h = 3 * entry_count + 1
    legend_w = 5
    if corner == "tl":
        return (0, 0, legend_h - 1, legend_w - 1)
    if corner == "tr":
        return (0, w - legend_w, legend_h - 1, w - 1)
    if corner == "bl":
        return (h - legend_h, 0, h - 1, legend_w - 1)
    if corner == "br":
        return (h - legend_h, w - legend_w, h - 1, w - 1)
    raise ValueError(f"unknown corner: {corner}")


def _draw_legend(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    entries: list[tuple[int, np.ndarray]],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    marker_c = c0 + 1
    tile_c = c0 + 2
    inner_r = r0 + 1
    for index, (path_color, pattern) in enumerate(entries):
        base_r = inner_r + 3 * index
        grid[base_r, marker_c] = path_color
        grid[base_r : base_r + 2, tile_c : tile_c + 2] = pattern


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 4:
            continue
        if not (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        ):
            continue
        ok = True
        expected_size = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(comp) != expected_size:
            continue
        comp_set = set(comp)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != ((rr, cc) in comp_set):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((area, (r0, c0, r1, c1)))
    assert candidates, "legend frame not found"
    _, bbox = max(candidates, key=lambda item: item[0])
    r0, c0, r1, c1 = bbox
    marker_c = c0 + 1
    tile_c = c0 + 2
    mapping: dict[int, np.ndarray] = {}
    r = r0 + 1
    while r + 1 <= r1 - 1:
        marker = int(grid[r, marker_c])
        tile = grid[r : r + 2, tile_c : tile_c + 2]
        if (
            marker not in (BG, GREY)
            and tile.shape == (2, 2)
            and not np.any(tile == BG)
            and not np.any(tile == GREY)
        ):
            mapping[marker] = tile.copy()
            r += 3
        else:
            r += 1
    assert mapping, "legend mapping must be non-empty"
    return bbox, mapping


def _rectangles_from_paths(grid: np.ndarray, ignore_bbox: tuple[int, int, int, int]) -> dict[int, tuple[int, int, int, int]]:
    work = grid.copy()
    lr0, lc0, lr1, lc1 = ignore_bbox
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    rectangles: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(work) if int(v) != BG):
        mask = work == color
        comps = _components(mask)
        assert 2 <= len(comps) <= 4, "each path color must have 2-4 disconnected segments"
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            assert len(set(rows)) == 1 or len(set(cols)) == 1, "each segment must be straight"
        pts = np.argwhere(mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        on_border = (
            (pts[:, 0] == r0)
            | (pts[:, 0] == r1)
            | (pts[:, 1] == c0)
            | (pts[:, 1] == c1)
        )
        assert bool(on_border.all()), "all path pixels must lie on one rectangle border"
        rectangles[color] = (int(r0), int(c0), int(r1), int(c1))
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox, mapping = _find_legend(g)
    rectangles = _rectangles_from_paths(g, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    for color, (r0, c0, r1, c1) in rectangles.items():
        assert color in mapping, "every path color must appear in the legend"
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        tile = mapping[color]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BG:
                    out[r, c] = int(tile[(r - (r0 + 1)) % 2, (c - (c0 + 1)) % 2])

    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    loop_count = rng.choice((1, 2, 3))
    (h_lo, h_hi), (w_lo, w_hi) = GRID_SIZE_BY_LOOP_COUNT[loop_count]
    h = rng.randint(h_lo, h_hi)
    w = rng.randint(w_lo, w_hi)
    corner = rng.choice(LEGEND_CORNERS)
    legend_h = 3 * loop_count + 1
    if h < legend_h or w < 5:
        return None

    path_colors = rng.sample(PATH_COLOR_POOL, loop_count)
    fill_pool = [color for color in PATH_COLOR_POOL if color not in path_colors]
    legend_order = path_colors[:]
    rng.shuffle(legend_order)
    mapping: dict[int, np.ndarray] = {}
    for color in path_colors:
        color_a, color_b = rng.sample(fill_pool, 2)
        pattern_type = rng.choice(PATTERN_TYPES)
        mapping[color] = _make_pattern(pattern_type, color_a, color_b)

    grid = np.full((h, w), BG, dtype=int)
    legend_box = _legend_bbox(h, w, corner, loop_count)
    _draw_legend(grid, legend_box, [(color, mapping[color]) for color in legend_order])

    occupied = np.zeros((h, w), dtype=bool)
    if not _reserve(occupied, legend_box, pad=1):
        return None

    for color in path_colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            segment_count = rng.choice(SEGMENT_COUNT_CHOICES)
            rect_h_max = min(10 if loop_count == 1 else 8, h - 2)
            rect_w_max = min(12 if loop_count == 1 else 10, w - 2)
            rect_h = rng.randint(5, rect_h_max)
            rect_w = rng.randint(5, rect_w_max)
            segments = _sample_segments(rng, rect_h, rect_w, segment_count)
            if segments is None:
                continue
            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _reserve(occupied, bbox, pad=1):
                continue
            for seg in segments:
                for sr, sc in seg:
                    grid[r0 + sr, c0 + sc] = color
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
