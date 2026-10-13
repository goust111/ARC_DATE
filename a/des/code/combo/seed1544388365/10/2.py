import random
from typing import NamedTuple

import numpy as np


GREY = 5
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
PAIR_COUNT_BY_SIZE = {
    "small": (1, 2),
    "large": (2, 3),
}
LINE_LENGTH_RANGE = (3, 8)
LINE_GAP_RANGE = (2, 6)
MAX_OBSTACLES_PER_PAIR = 3
PALETTE = tuple(range(10))


class Segment(NamedTuple):
    orientation: str
    fixed: int
    start: int
    end: int


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    idx = int(np.argmax(counts))
    assert int(np.sum(counts == counts[idx])) == 1, "background must be uniquely dominant"
    return int(values[idx])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_from_cells(cells: list[tuple[int, int]]) -> Segment:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(cells) == c1 - c0 + 1, "horizontal segment must be contiguous"
        return Segment("h", row, c0, c1)
    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(cells) == r1 - r0 + 1, "vertical segment must be contiguous"
        return Segment("v", col, r0, r1)
    raise AssertionError("component is not a straight line segment")


def _find_pairs(grid: np.ndarray) -> tuple[int, list[tuple[int, Segment, Segment]]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    assert bg != GREY, "background must differ from obstacle color"

    pair_specs: list[tuple[int, Segment, Segment]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY):
            continue
        comps = _components4(g == color)
        assert len(comps) == 2, "each non-bg, non-grey color must form exactly two segments"
        seg_a = _segment_from_cells(comps[0])
        seg_b = _segment_from_cells(comps[1])
        assert seg_a.orientation == seg_b.orientation, "paired segments must be parallel"
        assert seg_a.start == seg_b.start and seg_a.end == seg_b.end, "paired segments must align"
        assert seg_a.fixed != seg_b.fixed, "segments must be distinct"
        pair_specs.append((color, seg_a, seg_b))

    assert pair_specs, "expected at least one colored pair"
    return bg, pair_specs


def _apply_horizontal_pair(
    source: np.ndarray,
    out: np.ndarray,
    bg: int,
    color: int,
    seg_a: Segment,
    seg_b: Segment,
) -> None:
    top, bottom = sorted((seg_a.fixed, seg_b.fixed))
    left, right = seg_a.start, seg_a.end
    shadow = np.zeros_like(source, dtype=bool)

    for rr, cc in np.argwhere(source[top + 1 : bottom, left : right + 1] == GREY):
        r = top + 1 + int(rr)
        c = left + int(cc)
        dist_top = r - top
        dist_bottom = bottom - r
        if dist_top < dist_bottom:
            shadow[r:bottom, c] = True
        elif dist_bottom < dist_top:
            shadow[top + 1 : r + 1, c] = True
        else:
            shadow[r, c] = True

    for r in range(top + 1, bottom):
        for c in range(left, right + 1):
            if int(source[r, c]) == bg and not shadow[r, c]:
                out[r, c] = color


def _apply_vertical_pair(
    source: np.ndarray,
    out: np.ndarray,
    bg: int,
    color: int,
    seg_a: Segment,
    seg_b: Segment,
) -> None:
    left, right = sorted((seg_a.fixed, seg_b.fixed))
    top, bottom = seg_a.start, seg_a.end
    shadow = np.zeros_like(source, dtype=bool)

    for rr, cc in np.argwhere(source[top : bottom + 1, left + 1 : right] == GREY):
        r = top + int(rr)
        c = left + 1 + int(cc)
        dist_left = c - left
        dist_right = right - c
        if dist_left < dist_right:
            shadow[r, c:right] = True
        elif dist_right < dist_left:
            shadow[r, left + 1 : c + 1] = True
        else:
            shadow[r, c] = True

    for r in range(top, bottom + 1):
        for c in range(left + 1, right):
            if int(source[r, c]) == bg and not shadow[r, c]:
                out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair matching-color parallel line segments, fill the strip between them,
    and leave one-sided background shadows behind grey obstacles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, pair_specs = _find_pairs(g)
    out = g.copy()

    for color, seg_a, seg_b in pair_specs:
        if seg_a.orientation == "h":
            _apply_horizontal_pair(g, out, bg, color, seg_a, seg_b)
        else:
            _apply_vertical_pair(g, out, bg, color, seg_a, seg_b)

    return out


def _bbox_clear(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_bbox(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_obstacle_count(rng: random.Random, max_count: int, force_nonzero: bool) -> int:
    max_count = min(max_count, MAX_OBSTACLES_PER_PAIR)
    low = 1 if force_nonzero else 0
    options = list(range(low, max_count + 1))
    weights = []
    for value in options:
        if value == 0:
            weights.append(2)
        elif value == 1:
            weights.append(5)
        elif value == 2:
            weights.append(3)
        else:
            weights.append(1)
    return int(rng.choices(options, weights=weights, k=1)[0])


def _choose_horizontal_obstacles(
    rng: random.Random,
    top: int,
    bottom: int,
    left: int,
    right: int,
    count: int,
) -> list[tuple[int, int]]:
    top_rows = [r for r in range(top + 1, bottom) if r - top < bottom - r]
    bottom_rows = [r for r in range(top + 1, bottom) if bottom - r < r - top]
    cols = list(range(left, right + 1))
    rng.shuffle(cols)

    chosen: list[tuple[int, int]] = []
    side_offset = rng.randrange(2)
    for idx, c in enumerate(cols[:count]):
        if top_rows and bottom_rows:
            rows = top_rows if (idx + side_offset) % 2 == 0 else bottom_rows
        else:
            rows = top_rows or bottom_rows
        chosen.append((rng.choice(rows), c))
    return chosen


def _choose_vertical_obstacles(
    rng: random.Random,
    top: int,
    bottom: int,
    left: int,
    right: int,
    count: int,
) -> list[tuple[int, int]]:
    left_cols = [c for c in range(left + 1, right) if c - left < right - c]
    right_cols = [c for c in range(left + 1, right) if right - c < c - left]
    rows = list(range(top, bottom + 1))
    rng.shuffle(rows)

    chosen: list[tuple[int, int]] = []
    side_offset = rng.randrange(2)
    for idx, r in enumerate(rows[:count]):
        if left_cols and right_cols:
            cols = left_cols if (idx + side_offset) % 2 == 0 else right_cols
        else:
            cols = left_cols or right_cols
        chosen.append((r, rng.choice(cols)))
    return chosen


def _place_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    color: int,
    orientation: str,
    force_nonzero_obstacle: bool,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        if orientation == "h":
            max_len = min(LINE_LENGTH_RANGE[1], w - 2)
            max_gap = min(LINE_GAP_RANGE[1], h - 3)
            if max_len < LINE_LENGTH_RANGE[0] or max_gap < LINE_GAP_RANGE[0]:
                return False
            length = rng.randint(LINE_LENGTH_RANGE[0], max_len)
            gap = rng.randint(LINE_GAP_RANGE[0], max_gap)
            bbox_h = gap + 2
            bbox_w = length
            r0 = rng.randint(0, h - bbox_h)
            c0 = rng.randint(0, w - bbox_w)
            r1 = r0 + bbox_h - 1
            c1 = c0 + bbox_w - 1
            if not _bbox_clear(reserved, r0, c0, r1, c1):
                continue

            top = r0
            bottom = r1
            left = c0
            right = c1
            obstacle_count = _sample_obstacle_count(rng, length, force_nonzero_obstacle)
            obstacles = _choose_horizontal_obstacles(rng, top, bottom, left, right, obstacle_count)

            grid[top, left : right + 1] = color
            grid[bottom, left : right + 1] = color
            for r, c in obstacles:
                grid[r, c] = GREY
            _reserve_bbox(reserved, r0, c0, r1, c1)
            return True

        max_len = min(LINE_LENGTH_RANGE[1], h - 2)
        max_gap = min(LINE_GAP_RANGE[1], w - 3)
        if max_len < LINE_LENGTH_RANGE[0] or max_gap < LINE_GAP_RANGE[0]:
            return False
        length = rng.randint(LINE_LENGTH_RANGE[0], max_len)
        gap = rng.randint(LINE_GAP_RANGE[0], max_gap)
        bbox_h = length
        bbox_w = gap + 2
        r0 = rng.randint(0, h - bbox_h)
        c0 = rng.randint(0, w - bbox_w)
        r1 = r0 + bbox_h - 1
        c1 = c0 + bbox_w - 1
        if not _bbox_clear(reserved, r0, c0, r1, c1):
            continue

        top = r0
        bottom = r1
        left = c0
        right = c1
        obstacle_count = _sample_obstacle_count(rng, length, force_nonzero_obstacle)
        obstacles = _choose_vertical_obstacles(rng, top, bottom, left, right, obstacle_count)

        grid[top : bottom + 1, left] = color
        grid[top : bottom + 1, right] = color
        for r, c in obstacles:
            grid[r, c] = GREY
        _reserve_bbox(reserved, r0, c0, r1, c1)
        return True

    return False


def _sample_pair_count(rng: random.Random, h: int, w: int) -> int:
    size_key = "large" if min(h, w) >= 16 else "small"
    return int(rng.choice(PAIR_COUNT_BY_SIZE[size_key]))


def _sample_orientations(rng: random.Random, pair_count: int) -> list[str]:
    if pair_count == 1:
        return [rng.choice(("h", "v"))]
    if rng.random() < 0.5:
        orientations = ["h", "v"] + [rng.choice(("h", "v")) for _ in range(pair_count - 2)]
        rng.shuffle(orientations)
        return orientations
    return [rng.choice(("h", "v")) for _ in range(pair_count)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    pair_count = _sample_pair_count(rng, h, w)

    palette = [color for color in PALETTE if color != GREY]
    rng.shuffle(palette)
    bg = palette[0]
    pair_colors = palette[1 : 1 + pair_count]

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    orientations = _sample_orientations(rng, pair_count)
    forced_index = rng.randrange(pair_count)

    for idx, (color, orientation) in enumerate(zip(pair_colors, orientations)):
        if not _place_pair(grid, reserved, rng, color, orientation, idx == forced_index):
            return None

    if not np.any(grid == GREY):
        return None

    try:
        _find_pairs(grid)
        if not np.any(main(grid) != grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
