import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
SEGMENT_COUNT_RANGE = (2, 4)
SEGMENT_LEN_RANGE = (5, 6)
SEGMENT_GAP_RANGE = (0, 1)
OUTER_GAP_RANGE = (0, 1)
MAX_OBJECTS = 7
MAX_GENERATION_RETRIES = 300


def _parse_segments(line: np.ndarray, offset: int) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    idx = 0
    while idx < len(line):
        color = int(line[idx])
        if color == BG:
            idx += 1
            continue
        start = idx
        while idx + 1 < len(line) and int(line[idx + 1]) == color:
            idx += 1
        segments.append(
            {
                "color": color,
                "start": start + offset,
                "end": idx + offset,
                "rank": len(segments),
            }
        )
        idx += 1
    return segments


def _build_lookup(segments: list[dict[str, int]]) -> dict[int, tuple[int, int]]:
    lookup: dict[int, tuple[int, int]] = {}
    for seg in segments:
        for pos in range(seg["start"], seg["end"] + 1):
            lookup[pos] = (seg["color"], seg["rank"])
    return lookup


def _monochrome_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(1, h):
        for c in range(1, w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 1 <= nr < h and 1 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each interior object using the higher-priority edge index color from its row or column zone.

    Input structure:
    - The background is black, with the top row and left column acting as separate color index bars.
    - Non-black top-row segments mark column zones, and non-black left-column segments mark row zones.
    - The interior contains monochromatic 4-connected objects whose colors are distinct from the index-bar colors.

    Rule:
    - Read contiguous non-black segments on the top row to get column-zone colors and their left-to-right priority ranks.
    - Read contiguous non-black segments on the left column to get row-zone colors and their top-to-bottom priority ranks.
    - For each interior object, find the unique row zone and column zone containing all of its cells; if the row-zone rank is smaller than the column-zone rank, recolor the object with the row-zone color, otherwise recolor it with the column-zone color.
    - Leave the index bars and the black background unchanged.

    Key insight:
    - The task is solved by combining two independent edge legends: zone membership comes from segment coverage, while the winning recolor comes from comparing the two legend orderings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "expected room for bars and interior"

    col_segments = _parse_segments(g[0, 1:], offset=1)
    row_segments = _parse_segments(g[1:, 0], offset=1)
    assert col_segments and row_segments, "expected both index bars to contain colored segments"

    col_lookup = _build_lookup(col_segments)
    row_lookup = _build_lookup(row_segments)

    out = g.copy()
    for coords in _monochrome_components(g):
        row_infos = {row_lookup.get(int(r)) for r, _c in coords}
        col_infos = {col_lookup.get(int(c)) for _r, c in coords}
        assert len(row_infos) == 1 and None not in row_infos, "object must lie within one row zone"
        assert len(col_infos) == 1 and None not in col_infos, "object must lie within one column zone"

        row_color, row_rank = next(iter(row_infos))
        col_color, col_rank = next(iter(col_infos))
        target = row_color if row_rank < col_rank else col_color
        out[coords[:, 0], coords[:, 1]] = int(target)

    return out


def _sample_axis_segments(n_segments: int) -> tuple[list[tuple[int, int]], int]:
    lengths = [random.randint(*SEGMENT_LEN_RANGE) for _ in range(n_segments)]
    inner_gaps = [random.randint(*SEGMENT_GAP_RANGE) for _ in range(max(0, n_segments - 1))]
    outer_before = random.randint(*OUTER_GAP_RANGE)
    outer_after = random.randint(*OUTER_GAP_RANGE)

    pos = 1 + outer_before
    segments: list[tuple[int, int]] = []
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        segments.append((start, end))
        pos = end + 1
        if idx < len(inner_gaps):
            pos += inner_gaps[idx]
    axis_size = pos + outer_after
    return segments, axis_size


def _frontier(cells: set[tuple[int, int]], box_h: int, box_w: int) -> list[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                border.add((nr, nc))
    return list(border)


def _random_connected_mask(box_h: int, box_w: int) -> np.ndarray:
    max_cells = min(box_h * box_w, 7)
    if max_cells == 1:
        target = 1
    elif random.random() < 0.18:
        target = 1
    else:
        target = random.randint(2, max_cells)

    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target:
        options = _frontier(cells, box_h, box_w)
        if not options:
            break
        cells.add(random.choice(options))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _paint_object(
    grid: np.ndarray,
    row_seg: tuple[int, int],
    col_seg: tuple[int, int],
    color: int,
) -> None:
    row_start, row_end = row_seg
    col_start, col_end = col_seg
    inner_h = row_end - row_start - 1
    inner_w = col_end - col_start - 1
    assert inner_h >= 1 and inner_w >= 1

    box_h = random.randint(1, min(inner_h, 4))
    box_w = random.randint(1, min(inner_w, 4))
    top = random.randint(row_start + 1, row_end - box_h)
    left = random.randint(col_start + 1, col_end - box_w)
    mask = _random_connected_mask(box_h, box_w)

    patch = grid[top : top + box_h, left : left + box_w]
    assert not bool(patch[mask].any()), "generator tried to overlap objects"
    patch[mask] = int(color)


def _choose_regions(row_count: int, col_count: int) -> list[tuple[int, int]]:
    row_win = [(r, c) for r in range(row_count) for c in range(col_count) if r < c]
    col_win_strict = [(r, c) for r in range(row_count) for c in range(col_count) if r > c]
    tie = [(i, i) for i in range(min(row_count, col_count))]

    assert row_win and col_win_strict, "expected both priority outcomes to exist"

    chosen: list[tuple[int, int]] = [random.choice(row_win), random.choice(col_win_strict)]
    if tie and random.random() < 0.5:
        chosen.append(random.choice(tie))

    remaining = [
        (r, c)
        for r in range(row_count)
        for c in range(col_count)
        if (r, c) not in chosen
    ]
    random.shuffle(remaining)
    extra = random.randint(0, min(len(remaining), MAX_OBJECTS - len(chosen)))
    chosen.extend(remaining[:extra])
    return chosen


def _choose_palette(row_count: int, col_count: int) -> tuple[list[int], list[int], list[int]]:
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    row_colors = colors[:row_count]
    col_colors = colors[row_count : row_count + col_count]
    neutral = colors[row_count + col_count :]
    assert neutral, "need at least one neutral object color"
    random.shuffle(neutral)
    neutral_pool = neutral[: random.randint(1, min(3, len(neutral)))]
    return row_colors, col_colors, neutral_pool


def _render_bars(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    row_colors: list[int],
    col_colors: list[int],
) -> None:
    for (start, end), color in zip(col_segments, col_colors):
        grid[0, start : end + 1] = int(color)
    for (start, end), color in zip(row_segments, row_colors):
        grid[start : end + 1, 0] = int(color)


def _branch_counts(
    regions: list[tuple[int, int]]
) -> tuple[int, int, int]:
    row_win = sum(1 for r, c in regions if r < c)
    tie = sum(1 for r, c in regions if r == c)
    col_strict = sum(1 for r, c in regions if r > c)
    return row_win, tie, col_strict


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*SEGMENT_COUNT_RANGE)
    col_count = random.randint(*SEGMENT_COUNT_RANGE)

    row_segments, height = _sample_axis_segments(row_count)
    col_segments, width = _sample_axis_segments(col_count)
    if height > 30 or width > 30:
        return None

    row_colors, col_colors, neutral_pool = _choose_palette(row_count, col_count)
    grid = np.full((height, width), BG, dtype=int)
    _render_bars(grid, row_segments, col_segments, row_colors, col_colors)

    regions = _choose_regions(row_count, col_count)
    row_win, tie, col_strict = _branch_counts(regions)
    if row_win == 0 or (tie + col_strict) == 0:
        return None

    for row_idx, col_idx in regions:
        color = random.choice(neutral_pool)
        _paint_object(grid, row_segments[row_idx], col_segments[col_idx], color)

    try:
        out = main(grid)
    except Exception:
        return None

    changed = (out != grid)
    if not bool(changed.any()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {MAX_GENERATION_RETRIES} attempts")
