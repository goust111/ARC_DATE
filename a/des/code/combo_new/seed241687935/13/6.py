import random
from collections import Counter, deque

import numpy as np


BG = 0
DICT_COLOR = 8
BOUNDARY_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 300
DICT_PATTERNS = {
    2: np.ones((3, 3), dtype=bool),
    3: np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    4: np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _ring_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[np.ndarray, bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_boundary = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((np.array(cells, dtype=int), touches_boundary))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_patterns(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    non_bg = np.argwhere(grid != bg)
    assert non_bg.size, "dictionary is missing"
    order = np.lexsort((non_bg[:, 1], non_bg[:, 0]))
    r0, c0 = map(int, non_bg[order[0]])

    patterns: dict[int, np.ndarray] = {}
    for idx, key in enumerate(range(2, 6)):
        rr = r0 + 4 * idx
        block = grid[rr : rr + 3, c0 : c0 + 3]
        assert block.shape == (3, 3), "dictionary blocks must be 3x3"
        patterns[key] = block != bg
    return patterns


def _fill_region(
    out: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    pattern: np.ndarray,
    color: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = inner_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if pattern[(r - r0) % 3, (c - c0) % 3]:
                out[r, c] = color
            else:
                out[r, c] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangular region with a dictionary-selected pattern colored by its dominant border color.

    Input structure:
    - The background is black.
    - The upper-left corner contains a vertical dictionary of four 3x3 pattern tiles,
      separated by single black rows and drawn in one non-background color.
    - Elsewhere, one-pixel-thick colored horizontal and vertical line segments form
      closed rectangular borders around black interior regions.

    Rule:
    - Read the four dictionary tiles from top to bottom as the fill masks for border-color
      counts K = 2, 3, 4, 5.
    - Find each enclosed black region whose cells form a solid rectangle and whose
      immediately surrounding ring is fully non-background.
    - Count the number of distinct colors on that surrounding ring, choose the matching
      dictionary pattern, and find the border color occupying the most ring pixels.
    - Tile the chosen pattern from the region's upper-left interior corner using that
      dominant border color; leave black cells in the pattern as background.
      Keep all dictionary pixels and border lines unchanged.

    Key insight:
    - The upper-left stack is a pattern legend, not an ordinary object, and each rectangle
      must be solved by combining two local measurements: distinct border-color count
      chooses the pattern, while border-color frequency chooses the drawing color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    patterns = _extract_patterns(g, bg)
    out = g.copy()
    h, w = g.shape

    for coords, touches_boundary in _components(g == bg):
        if touches_boundary:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(coords) != area:
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue

        border = _ring_coords(r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        border_colors = [int(g[r, c]) for r, c in border]
        if any(color == bg for color in border_colors):
            continue

        unique_count = len(set(border_colors))
        if unique_count not in patterns:
            continue

        counts = Counter(border_colors)
        fill_color = max(counts.items(), key=lambda item: item[1])[0]
        _fill_region(out, (r0, c0, r1, c1), patterns[unique_count], fill_color, bg)

    return out


def _draw_dictionary(grid: np.ndarray) -> None:
    for idx, key in enumerate(range(2, 6)):
        r0 = 4 * idx
        pattern = DICT_PATTERNS[key]
        grid[r0 : r0 + 3, 0:3][pattern] = DICT_COLOR


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 2) -> None:
    h, w = occupied.shape
    occupied[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _segment_lengths(total: int, count: int, min_len: int = 2) -> list[int] | None:
    if total < count * min_len:
        return None
    extra = total - count * min_len
    for _ in range(80):
        lengths = (np.random.multinomial(extra, [1.0 / count] * count) + min_len).tolist()
        longest = max(lengths)
        if lengths.count(longest) == 1:
            return [int(x) for x in lengths]
    return None


def _paint_boundary(grid: np.ndarray, ring: list[tuple[int, int]], k: int) -> dict[str, object] | None:
    colors = random.sample(BOUNDARY_COLORS, k)
    lengths = _segment_lengths(len(ring), k)
    if lengths is None:
        return None

    start = random.randrange(len(ring))
    ordered_ring = ring[start:] + ring[:start]
    cursor = 0
    ring_counts: Counter[int] = Counter()
    for color, length in zip(colors, lengths, strict=True):
        for r, c in ordered_ring[cursor : cursor + length]:
            grid[r, c] = color
            ring_counts[color] += 1
        cursor += length

    dominant_color, dominant_count = max(ring_counts.items(), key=lambda item: item[1])
    if list(ring_counts.values()).count(dominant_count) != 1:
        return None
    return {"k": k, "dominant": dominant_color, "counts": dict(ring_counts)}


def _place_rectangle(
    grid: np.ndarray,
    occupied: np.ndarray,
    outer_h: int,
    outer_w: int,
    k: int,
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        ring = _ring_coords(r0, c0, r1, c1)
        info = _paint_boundary(grid, ring, k)
        if info is None:
            grid[r0 : r1 + 1, c0 : c1 + 1] = BG
            continue

        _reserve(occupied, r0, c0, r1, c1, pad=2)
        return True
    return False


def _sample_grid_shape() -> tuple[int, int]:
    return random.randint(18, 30), random.randint(18, 30)


def _sample_rectangle_size(h: int, w: int, k: int) -> tuple[int, int]:
    min_outer_h = 4
    min_outer_w = max(4, (2 * k + 4) // 2)
    max_outer_h = min(10, h - 2)
    max_outer_w = min(12, w - 2)
    outer_h = random.randint(min_outer_h, max_outer_h)
    outer_w = random.randint(min_outer_w, max_outer_w)
    if 2 * (outer_h + outer_w) - 4 < 2 * k:
        return _sample_rectangle_size(h, w, k)
    return outer_h, outer_w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_grid_shape()
    grid = np.full((h, w), BG, dtype=int)
    _draw_dictionary(grid)

    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, 0, 0, 14, 2, pad=2)

    num_rects = random.randint(1, 4)
    ks = [random.randint(2, 5) for _ in range(num_rects)]
    random.shuffle(ks)

    placed = 0
    for k in ks:
        outer_h, outer_w = _sample_rectangle_size(h, w, k)
        if _place_rectangle(grid, occupied, outer_h, outer_w, k):
            placed += 1

    if placed != num_rects:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
