from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
PAIR_COUNT_CHOICES = (2, 3)
FRAME_COUNT_MAX = 3
FRAME_HEIGHT_RANGE = (4, 8)
FRAME_WIDTH_RANGE = (4, 10)
PALETTE_GAP_CHOICES = (0, 1, 2)
COUNT_PATTERNS = {
    1: [(1,), (2,), (3,), (4,), (5,)],
    2: [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2)],
    3: [(2, 1, 1), (3, 1, 1), (3, 2, 1), (4, 1, 1), (4, 2, 1)],
}


def _color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            components.append((color, coords))

    return components


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(coords) != expected:
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color, coords in _color_components(grid):
        bbox = _frame_bbox(coords)
        if bbox is not None:
            frames.append((color, bbox))
    return frames


def _palette_mapping(bottom_row: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = int(bottom_row.shape[0])
    while c < width:
        if int(bottom_row[c]) == BLACK:
            c += 1
            continue
        assert c + 1 < width and int(bottom_row[c + 1]) != BLACK, "palette must be made of adjacent pairs"
        a = int(bottom_row[c])
        b = int(bottom_row[c + 1])
        assert a != b, "palette pairs must use two distinct colors"
        if a in mapping:
            assert mapping[a] == b
        if b in mapping:
            assert mapping[b] == a
        mapping[a] = b
        mapping[b] = a
        c += 2
    assert mapping, "expected at least one palette pair on the bottom row"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with its voted interior color, then recolor the frame via the bottom palette.

    Input structure:
    - The background is black.
    - Several one-pixel-thick closed rectangular frames appear in distinct colors.
    - Inside each frame, a few non-black single-pixel fragments are scattered over an otherwise black interior.
    - On the bottom boundary, a palette row contains horizontal color pairs; each pair encodes a mutual complement mapping.

    Rule:
    - Read the bottom palette row as adjacent two-pixel pairs, interpreting each pair as a bidirectional complement lookup.
    - For every closed rectangular frame, count the non-black fragment colors inside it and select the unique most frequent color.
    - Fill the entire interior of that frame with the selected color, then repaint the frame border with the selected color's complement from the palette.

    Key insight:
    - The interior fragments identify which palette key to use; the palette then determines the border recolor, so both the vote and the lookup are required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a 2D ARC grid"

    mapping = _palette_mapping(g[-1, :])
    frames = _find_frames(g)
    assert frames, "expected at least one closed rectangular frame"

    out = g.copy()
    for _frame_color, (r0, c0, r1, c1) in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        votes = Counter(map(int, interior[interior != BLACK]))
        assert votes, "each frame interior must contain at least one color fragment"
        ranked = votes.most_common(2)
        main_color, top_count = ranked[0]
        if len(ranked) > 1:
            assert top_count > ranked[1][1], "main color must have a unique majority"
        assert main_color in mapping, "voted fill color must appear in the palette row"

        out[r0 + 1 : r1, c0 + 1 : c1] = main_color
        border_color = mapping[main_color]
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color

    out[-1, :] = g[-1, :]
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _choose_fragment_plan(area: int, palette_colors: list[int]) -> tuple[list[int], tuple[int, ...]]:
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for n_colors, patterns in COUNT_PATTERNS.items():
        if n_colors > len(palette_colors):
            continue
        for pattern in patterns:
            if sum(pattern) <= area:
                candidates.append((n_colors, pattern))
    assert candidates, "interior must have enough room for a valid voting pattern"

    n_colors, pattern = random.choice(candidates)
    chosen = random.sample(palette_colors, n_colors)
    majority = random.choice(chosen)
    others = [color for color in chosen if color != majority]
    random.shuffle(others)
    return [majority] + others, pattern


def _scatter_fragments(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    colors: list[int],
    counts: tuple[int, ...],
) -> int:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen_cells = random.sample(cells, sum(counts))

    idx = 0
    for color, count in zip(colors, counts):
        for _ in range(count):
            r, c = chosen_cells[idx]
            idx += 1
            grid[r, c] = color

    return colors[0]


def _place_frame_boxes(height: int, width: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    occupied = np.zeros((height, width), dtype=bool)

    max_frame_h = min(FRAME_HEIGHT_RANGE[1], height - 3)
    max_frame_w = min(FRAME_WIDTH_RANGE[1], width - 2)
    if max_frame_h < FRAME_HEIGHT_RANGE[0] or max_frame_w < FRAME_WIDTH_RANGE[0]:
        return None

    for _ in range(frame_count):
        placed = False
        for _ in range(150):
            frame_h = random.randint(FRAME_HEIGHT_RANGE[0], max_frame_h)
            frame_w = random.randint(FRAME_WIDTH_RANGE[0], max_frame_w)
            max_r0 = height - frame_h - 2
            max_c0 = width - frame_w - 2
            if max_r0 < 1 or max_c0 < 1:
                continue

            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            boxes.append((r0, c0, r1, c1))
            placed = True
            break

        if not placed:
            return None

    return boxes


def _build_palette_row(width: int, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, dict[int, int]] | None:
    gaps = [random.choice(PALETTE_GAP_CHOICES) for _ in range(max(0, len(pairs) - 1))]
    palette_len = 2 * len(pairs) + sum(gaps)
    if palette_len > width:
        return None

    row = np.zeros(width, dtype=int)
    start = random.randint(0, width - palette_len)
    order = pairs[:]
    random.shuffle(order)

    mapping: dict[int, int] = {}
    pos = start
    for idx, (a, b) in enumerate(order):
        if random.random() < 0.5:
            a, b = b, a
        row[pos] = a
        row[pos + 1] = b
        mapping[a] = b
        mapping[b] = a
        pos += 2
        if idx < len(gaps):
            pos += gaps[idx]

    return row, mapping


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    max_frames_by_color = min(FRAME_COUNT_MAX, len(NON_BLACK_COLORS) - 2 * pair_count)
    frame_count = random.randint(1, max_frames_by_color)

    min_width = max(12, 2 * pair_count + 6)
    min_height = 10 + 2 * frame_count
    height = random.randint(min_height, 28)
    width = random.randint(min_width, 30)

    all_colors = random.sample(NON_BLACK_COLORS, 2 * pair_count + frame_count)
    palette_colors = all_colors[: 2 * pair_count]
    frame_colors = all_colors[2 * pair_count :]
    pairs = [tuple(palette_colors[2 * i : 2 * i + 2]) for i in range(pair_count)]

    palette = _build_palette_row(width, pairs)
    if palette is None:
        return None
    palette_row, mapping = palette

    boxes = _place_frame_boxes(height, width, frame_count)
    if boxes is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    inp[-1, :] = palette_row
    out[-1, :] = palette_row

    palette_values = list(mapping)
    for bbox, frame_color in zip(boxes, frame_colors):
        _draw_frame(inp, bbox, frame_color)

        r0, c0, r1, c1 = bbox
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        colors, counts = _choose_fragment_plan(area, palette_values)
        majority = _scatter_fragments(inp, bbox, colors, counts)

        _draw_frame(out, bbox, mapping[majority])
        out[r0 + 1 : r1, c0 + 1 : c1] = majority

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
